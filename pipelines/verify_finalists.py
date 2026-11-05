"""
Verification and Fair Comparison of CinePulse Tournament Finalists.
Evaluates:
1. Actual Published Production Full Pipeline (TT=100, CF=100, Pop=30, 17f LGBM, MMR lambda=0.85, beta=0.03)
2. Old High-Diversity Baseline Pipeline (TT=100, CF=100, Pop=30, 17f LGBM, MMR lambda=0.65, beta=0.15)
3. Original Pure LightGBM Ranker (TT=100, CF=100, Pop=30, 17f LGBM, No MMR)
4. New 24-Feature Pure Ranker with Baseline Candidate Pool (TT=100, CF=100, Pop=30)
5. New 24-Feature Pure Ranker with Expanded Candidate Pool (TT=200, CF=100, Pop=30)
6. New Ranker with Exp 5 Winner Fusion (0.70 LGBM + 0.10 CF + 0.20 TT, No MMR)
7. New Ranker with Exp Winner Blend (0.75 LGBM + 0.18 CF + 0.07 TT, No MMR)
8. New Ranker with Relevance-First MMR (lambda=0.95, beta=0.01, No Fusion)
9. Grand Champion Full Pipeline (0.75 LGBM + 0.18 CF + 0.07 TT + MMR lambda=0.95, beta=0.01)
10. Grand Champion Alt Full Pipeline (0.70 LGBM + 0.10 CF + 0.20 TT + MMR lambda=0.95, beta=0.01)
11. New 24-Feature Ranker with Published Production MMR (lambda=0.85, beta=0.03)

All evaluated simultaneously on identical 4,980 validation users with paired bootstrap significance tests.
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import torch

from pipelines.exp1_item_cf import EnhancedItemCF
from pipelines.exp3_ranker import FastRankingFeatureExtractor
from pipelines.tournament_eval import TournamentEvaluator
from src.core.config import settings
from src.core.logging import logger
from src.evaluation.metrics import compute_paired_bootstrap_ci
from src.models.baselines import PopularityRecommender
from src.models.two_tower import TwoTowerRecommender
from src.reranking.mmr import MMRReRanker
from src.retrieval.faiss_index import FaissCandidateIndex


def run_verification():
    logger.info("=== Running Finalist Verification & Fair Comparison ===")
    evaluator = TournamentEvaluator()
    device = evaluator.device

    output_dir = Path("artifacts/experiments/relevance_v2")
    models_dir = output_dir / "models"

    # 1. Load component models
    logger.info("Loading baseline and experimental models...")
    pop_model = PopularityRecommender()
    pop_model.fit(evaluator.train_ratings, num_movies=evaluator.num_movies)

    cf_model = EnhancedItemCF(top_k_neighbors=200, shrinkage_lambda=25.0)
    cf_model.fit(evaluator.train_ratings, num_users=evaluator.num_users, num_movies=evaluator.num_movies)

    # Two-Tower & FAISS
    two_tower_path = Path("artifacts/models/two_tower_best.pt")
    tt_model = TwoTowerRecommender(
        num_users=evaluator.num_users,
        num_movies=evaluator.num_movies,
        num_genres=len(evaluator.genre_cols),
        embedding_dim=settings.embedding_dim,
    ).to(device)
    ckpt = torch.load(two_tower_path, map_location=device, weights_only=True)
    tt_model.load_state_dict(ckpt["model_state_dict"])
    tt_model.eval()

    faiss_index = FaissCandidateIndex(dimension=settings.embedding_dim)
    faiss_index.load(Path("artifacts/indices"))

    item_embeddings = np.load("artifacts/models/item_embeddings.npy")

    # Baseline 17-feature ranker
    booster_base = lgb.Booster(model_file="artifacts/models/ranker_lgbm.txt")

    # Champion 24-feature ranker
    champion_ranker_path = models_dir / "ranker_lgbm_exp3.txt"
    booster_exp = lgb.Booster(model_file=str(champion_ranker_path))

    fast_extractor = FastRankingFeatureExtractor(
        movies_df=evaluator.movies_df,
        genre_cols=evaluator.genre_cols,
        global_mean_rating=evaluator.mappings.get("global_rating_mean", 3.5),
    )

    # Rerankers
    mmr_prod = MMRReRanker(evaluator.movies_df, evaluator.genre_cols, default_lambda=0.85, default_novelty_weight=0.03)
    mmr_old_high_div = MMRReRanker(evaluator.movies_df, evaluator.genre_cols, default_lambda=0.65, default_novelty_weight=0.15)
    mmr_rel_first = MMRReRanker(evaluator.movies_df, evaluator.genre_cols, default_lambda=0.95, default_novelty_weight=0.01)

    eval_users = evaluator.std_users

    # Batch encode user embeddings
    logger.info("Precomputing user representations for all 4,980 validation users...")
    user_embs_dict = {}
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(eval_users), batch_size):
            b_users = eval_users[i : i + batch_size]
            b_size = len(b_users)
            u_arr = np.array(b_users, dtype=np.int64)
            padded_hist = np.full((b_size, settings.max_sequence_length), evaluator.num_movies, dtype=np.int64)
            h_lens = np.zeros(b_size, dtype=np.int64)
            g_prefs = np.zeros((b_size, len(evaluator.genre_cols)), dtype=np.float32)

            for idx, u in enumerate(b_users):
                hist = evaluator.user_histories.get(u, [])
                h_len = min(len(hist), settings.max_sequence_length)
                h_lens[idx] = h_len
                if h_len > 0:
                    padded_hist[idx, -h_len:] = hist[-h_len:]
                    valid_h = [m for m in hist if m < evaluator.num_movies]
                    if valid_h:
                        g_prefs[idx] = evaluator.item_genres[valid_h].mean(axis=0)

            u_tensor = torch.tensor(u_arr, dtype=torch.long, device=device)
            h_tensor = torch.tensor(padded_hist, dtype=torch.long, device=device)
            hl_tensor = torch.tensor(h_lens, dtype=torch.long, device=device)
            gp_tensor = torch.tensor(g_prefs, dtype=torch.float32, device=device)

            embs = tt_model.user_tower(u_tensor, h_tensor, hl_tensor, gp_tensor).cpu().numpy()
            for idx, u in enumerate(b_users):
                user_embs_dict[u] = embs[idx]

    user_val_ratings_map: dict[int, dict[int, float]] = {}
    for row in evaluator.train_ratings[["user_idx", "movie_idx", "rating"]].itertuples(index=False):
        u, m, r = int(row.user_idx), int(row.movie_idx), float(row.rating)
        user_val_ratings_map.setdefault(u, {})[m] = r

    # We define the configurations to evaluate
    configs = [
        "1. Published Production Full Pipeline (17f + MMR 0.85/0.03)",
        "2. Old High-Diversity Baseline Pipeline (17f + MMR 0.65/0.15)",
        "3. Original Pure LightGBM Ranker (17f, Baseline Pool, No MMR)",
        "4. New 24f Pure Ranker (Ranker-V5, Baseline Pool TT=100/CF=100)",
        "5. New 24f Pure Ranker (Ranker-V5, Expanded Pool TT=200/CF=100)",
        "6. New 24f Ranker + Exp 5 Blend (0.70/0.10/0.20, No MMR)",
        "7. New 24f Ranker + Grand Champ Blend (0.75/0.18/0.07, No MMR)",
        "8. New 24f Ranker + Relevance-First MMR (0.95/0.01, No Fusion)",
        "9. Grand Champion Full Pipeline (0.75/0.18/0.07 + MMR 0.95/0.01)",
        "10. Grand Champion Alt Full Pipeline (0.70/0.10/0.20 + MMR 0.95/0.01)",
        "11. New 24f Ranker + Production MMR (0.85/0.03, No Fusion)",
    ]

    recs_by_config = {c: {} for c in configs}
    lats_by_config = {c: [] for c in configs}

    logger.info("Generating and scoring recommendations across all configurations...")

    for u in eval_users:
        h = evaluator.user_histories.get(u, [])
        consumed = set(h)
        u_emb = user_embs_dict[u]
        u_ratings = user_val_ratings_map.get(u, {})

        # 1. Retrieval
        # Expanded candidates: TT=200, CF=100, Pop=30
        # Baseline candidates: TT=100, CF=100, Pop=30
        t0_start = time.perf_counter()
        tt_search_results = faiss_index.search(u_emb, top_k=200, exclude_item_indices=consumed)
        cf_search_results = [
            (idx, float(score)) for idx, score in cf_model.recommend(h, top_k=100) if idx not in consumed
        ]
        pop_search_results = [
            (idx, float(score)) for idx, score in pop_model.recommend(h, top_k=30) if idx not in consumed
        ]

        # Sources for baseline pool (first 100 TT, 100 CF, 30 Pop)
        base_sources = {}
        base_cf_map = {}
        for item_idx, _ in tt_search_results[:100]:
            base_sources.setdefault(item_idx, set()).add("two_tower")
        for item_idx, score in cf_search_results:
            base_sources.setdefault(item_idx, set()).add("item_cf")
            base_cf_map[item_idx] = score
        for item_idx, _ in pop_search_results:
            base_sources.setdefault(item_idx, set()).add("popularity")
        base_cand_list = list(base_sources.keys())

        # Sources for expanded pool (200 TT, 100 CF, 30 Pop)
        exp_sources = {}
        exp_cf_map = {}
        for item_idx, _ in tt_search_results:
            exp_sources.setdefault(item_idx, set()).add("two_tower")
        for item_idx, score in cf_search_results:
            exp_sources.setdefault(item_idx, set()).add("item_cf")
            exp_cf_map[item_idx] = score
        for item_idx, _ in pop_search_results:
            exp_sources.setdefault(item_idx, set()).add("popularity")
        exp_cand_list = list(exp_sources.keys())

        # Dot products
        def get_tt_scores(c_list):
            cand_arr = np.array(c_list, dtype=np.int64)
            valid_mask = cand_arr < len(item_embeddings)
            tt_scores = {}
            if np.any(valid_mask):
                cand_embs = item_embeddings[cand_arr[valid_mask]]
                sims = np.dot(cand_embs, u_emb)
                for idx_c, sim in zip(cand_arr[valid_mask], sims):
                    tt_scores[int(idx_c)] = float(sim)
            for c in c_list:
                tt_scores.setdefault(c, 0.0)
            return tt_scores

        base_tt_scores = get_tt_scores(base_cand_list)
        exp_tt_scores = get_tt_scores(exp_cand_list)

        # Baseline 17 features on Base pool
        base_mat_17 = fast_extractor.extract_matrix(
            u, h, u_ratings, base_cand_list, base_tt_scores, base_cf_map, base_sources, use_expanded=False
        )
        lgbm_base_scores = booster_base.predict(base_mat_17)
        base_ranked_pairs = [(base_cand_list[i], float(lgbm_base_scores[i])) for i in np.argsort(-lgbm_base_scores)]

        # 24 features on Base pool
        base_mat_24 = fast_extractor.extract_matrix(
            u, h, u_ratings, base_cand_list, base_tt_scores, base_cf_map, base_sources, use_expanded=True
        )
        exp_ranker_base_pool_scores = booster_exp.predict(base_mat_24)

        # 24 features on Expanded pool
        exp_mat_24 = fast_extractor.extract_matrix(
            u, h, u_ratings, exp_cand_list, exp_tt_scores, exp_cf_map, exp_sources, use_expanded=True
        )
        exp_lgbm_scores = booster_exp.predict(exp_mat_24)
        exp_ranked_pairs = [(exp_cand_list[i], float(exp_lgbm_scores[i])) for i in np.argsort(-exp_lgbm_scores)]

        # Score Normalization on Expanded Pool
        def min_max(arr):
            mn, mx = arr.min(), arr.max()
            diff = mx - mn
            return (arr - mn) / (diff + 1e-6) if diff > 0 else np.zeros_like(arr)

        lgbm_norm = min_max(exp_lgbm_scores)
        cf_raw = np.array([exp_cf_map.get(c, 0.0) for c in exp_cand_list], dtype=np.float32)
        tt_raw = np.array([exp_tt_scores.get(c, 0.0) for c in exp_cand_list], dtype=np.float32)
        cf_norm = min_max(cf_raw)
        tt_norm = min_max(tt_raw)

        fused_70_10_20 = (0.70 * lgbm_norm) + (0.10 * cf_norm) + (0.20 * tt_norm)
        fused_75_18_07 = (0.75 * lgbm_norm) + (0.18 * cf_norm) + (0.07 * tt_norm)

        fused_pairs_70 = [(exp_cand_list[i], float(fused_70_10_20[i])) for i in np.argsort(-fused_70_10_20)]
        fused_pairs_75 = [(exp_cand_list[i], float(fused_75_18_07[i])) for i in np.argsort(-fused_75_18_07)]

        # --- EXECUTE CONFIGS ---
        # 1. Published Production Full Pipeline (17f + MMR 0.85/0.03)
        recs_1 = [x["movie_idx"] for x in mmr_prod.rerank(base_ranked_pairs, top_k=100, lambda_param=0.85, novelty_weight=0.03, excluded_item_indices=consumed)]
        recs_by_config["1. Published Production Full Pipeline (17f + MMR 0.85/0.03)"][u] = recs_1
        lats_by_config["1. Published Production Full Pipeline (17f + MMR 0.85/0.03)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 2. Old High-Diversity Baseline Pipeline (17f + MMR 0.65/0.15)
        recs_2 = [x["movie_idx"] for x in mmr_old_high_div.rerank(base_ranked_pairs, top_k=100, lambda_param=0.65, novelty_weight=0.15, excluded_item_indices=consumed)]
        recs_by_config["2. Old High-Diversity Baseline Pipeline (17f + MMR 0.65/0.15)"][u] = recs_2
        lats_by_config["2. Old High-Diversity Baseline Pipeline (17f + MMR 0.65/0.15)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 3. Original Pure LightGBM Ranker (17f, Baseline Pool, No MMR)
        recs_3 = [c for c, _ in base_ranked_pairs[:100]]
        recs_by_config["3. Original Pure LightGBM Ranker (17f, Baseline Pool, No MMR)"][u] = recs_3
        lats_by_config["3. Original Pure LightGBM Ranker (17f, Baseline Pool, No MMR)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 4. New 24f Pure Ranker (Ranker-V5, Baseline Pool TT=100/CF=100)
        recs_4 = [base_cand_list[i] for i in np.argsort(-exp_ranker_base_pool_scores)[:100]]
        recs_by_config["4. New 24f Pure Ranker (Ranker-V5, Baseline Pool TT=100/CF=100)"][u] = recs_4
        lats_by_config["4. New 24f Pure Ranker (Ranker-V5, Baseline Pool TT=100/CF=100)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 5. New 24f Pure Ranker (Ranker-V5, Expanded Pool TT=200/CF=100)
        recs_5 = [c for c, _ in exp_ranked_pairs[:100]]
        recs_by_config["5. New 24f Pure Ranker (Ranker-V5, Expanded Pool TT=200/CF=100)"][u] = recs_5
        lats_by_config["5. New 24f Pure Ranker (Ranker-V5, Expanded Pool TT=200/CF=100)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 6. New 24f Ranker + Exp 5 Blend (0.70/0.10/0.20, No MMR)
        recs_6 = [c for c, _ in fused_pairs_70[:100]]
        recs_by_config["6. New 24f Ranker + Exp 5 Blend (0.70/0.10/0.20, No MMR)"][u] = recs_6
        lats_by_config["6. New 24f Ranker + Exp 5 Blend (0.70/0.10/0.20, No MMR)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 7. New 24f Ranker + Grand Champ Blend (0.75/0.18/0.07, No MMR)
        recs_7 = [c for c, _ in fused_pairs_75[:100]]
        recs_by_config["7. New 24f Ranker + Grand Champ Blend (0.75/0.18/0.07, No MMR)"][u] = recs_7
        lats_by_config["7. New 24f Ranker + Grand Champ Blend (0.75/0.18/0.07, No MMR)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 8. New 24f Ranker + Relevance-First MMR (0.95/0.01, No Fusion)
        recs_8 = [x["movie_idx"] for x in mmr_rel_first.rerank(exp_ranked_pairs, top_k=100, lambda_param=0.95, novelty_weight=0.01, excluded_item_indices=consumed)]
        recs_by_config["8. New 24f Ranker + Relevance-First MMR (0.95/0.01, No Fusion)"][u] = recs_8
        lats_by_config["8. New 24f Ranker + Relevance-First MMR (0.95/0.01, No Fusion)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 9. Grand Champion Full Pipeline (0.75/0.18/0.07 + MMR 0.95/0.01)
        recs_9 = [x["movie_idx"] for x in mmr_rel_first.rerank(fused_pairs_75, top_k=100, lambda_param=0.95, novelty_weight=0.01, excluded_item_indices=consumed)]
        recs_by_config["9. Grand Champion Full Pipeline (0.75/0.18/0.07 + MMR 0.95/0.01)"][u] = recs_9
        lats_by_config["9. Grand Champion Full Pipeline (0.75/0.18/0.07 + MMR 0.95/0.01)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 10. Grand Champion Alt Full Pipeline (0.70/0.10/0.20 + MMR 0.95/0.01)
        recs_10 = [x["movie_idx"] for x in mmr_rel_first.rerank(fused_pairs_70, top_k=100, lambda_param=0.95, novelty_weight=0.01, excluded_item_indices=consumed)]
        recs_by_config["10. Grand Champion Alt Full Pipeline (0.70/0.10/0.20 + MMR 0.95/0.01)"][u] = recs_10
        lats_by_config["10. Grand Champion Alt Full Pipeline (0.70/0.10/0.20 + MMR 0.95/0.01)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 11. New 24f Ranker + Production MMR (0.85/0.03, No Fusion)
        recs_11 = [x["movie_idx"] for x in mmr_prod.rerank(exp_ranked_pairs, top_k=100, lambda_param=0.85, novelty_weight=0.03, excluded_item_indices=consumed)]
        recs_by_config["11. New 24f Ranker + Production MMR (0.85/0.03, No Fusion)"][u] = recs_11
        lats_by_config["11. New 24f Ranker + Production MMR (0.85/0.03, No Fusion)"].append((time.perf_counter() - t0_start) * 1000.0)

    # Evaluate all
    results = {}
    per_user_results = {}
    for c in configs:
        m, per_user = evaluator.evaluate_recommendations(
            recs_by_config[c],
            c,
            latencies_ms=lats_by_config[c],
            train_time_sec=0.0,
        )
        results[c] = m
        per_user_results[c] = per_user

    with open(output_dir / "finalist_verification_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Statistical Significance Comparisons (1000 bootstrap resamples)
    correct_base_name = "1. Published Production Full Pipeline (17f + MMR 0.85/0.03)"
    pure_base_name = "3. Original Pure LightGBM Ranker (17f, Baseline Pool, No MMR)"

    base_pub_ndcgs = [per_user_results[correct_base_name]["ndcg@10"][u] for u in eval_users]
    base_pure_ndcgs = [per_user_results[pure_base_name]["ndcg@10"][u] for u in eval_users]

    significance_reports = {}
    for c in configs:
        if c in (correct_base_name, pure_base_name):
            continue
        c_ndcgs = [per_user_results[c]["ndcg@10"][u] for u in eval_users]
        # Diff vs published production baseline
        diff_pub, ci_l_pub, ci_u_pub, p_pub = compute_paired_bootstrap_ci(c_ndcgs, base_pub_ndcgs)
        # Diff vs pure ranker baseline
        diff_pure, ci_l_pure, ci_u_pure, p_pure = compute_paired_bootstrap_ci(c_ndcgs, base_pure_ndcgs)

        significance_reports[c] = {
            "vs_published_prod_baseline": {
                "delta_ndcg@10": round(diff_pub, 4),
                "ci95_ndcg@10": [round(ci_l_pub, 4), round(ci_u_pub, 4)],
                "p_value": p_pub,
                "relative_gain_ndcg@10": round((diff_pub / max(1e-6, results[correct_base_name]["ndcg@10"])) * 100, 2),
            },
            "vs_pure_ranker_baseline": {
                "delta_ndcg@10": round(diff_pure, 4),
                "ci95_ndcg@10": [round(ci_l_pure, 4), round(ci_u_pure, 4)],
                "p_value": p_pure,
                "relative_gain_ndcg@10": round((diff_pure / max(1e-6, results[pure_base_name]["ndcg@10"])) * 100, 2),
            },
        }

    with open(output_dir / "finalist_significance.json", "w") as f:
        json.dump(significance_reports, f, indent=2)

    print("\n" + "=" * 135)
    print("FAIR HEAD-TO-HEAD COMPARISON OF ALL FINALISTS (All 4,980 Val Users, Identical Protocols)")
    print("=" * 135)
    fmt = "{:<55} {:<11} {:<11} {:<13} {:<8} {:<12} {:<9}"
    print(fmt.format("Configuration", "Recall@20", "NDCG@10", "Pos NDCG@10", "ILD", "Coverage@100", "P95 ms"))
    print("-" * 135)
    for c in configs:
        m = results[c]
        print(
            fmt.format(
                c,
                f"{m['recall@20']:.4f}",
                f"{m['ndcg@10']:.4f}",
                f"{m['positive_only_ndcg@10']:.4f}",
                f"{m['intra_list_diversity']:.4f}",
                f"{m['catalog_coverage@100']*100:.1f}%",
                f"{m['latency_p95_ms']:.2f}",
            )
        )
    print("-" * 135)


if __name__ == "__main__":
    run_verification()
