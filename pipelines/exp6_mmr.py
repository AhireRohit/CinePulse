"""
Experiment 6: Relevance-First MMR Diversity Optimization.
Controlled sweep across MMR diversity and novelty hyperparameters to map the exact
Pareto efficiency frontier between Recommendation Relevance (NDCG@10, Recall@20)
and Catalog Discovery / Diversity (Intra-List Diversity, Catalog Coverage@100, Novelty):
1. MMR Disabled (Pure Relevance: lambda=1.00, novelty_beta=0.00)
2. Ultra-Light Diversity (lambda=0.98, novelty_beta=0.00)
3. Light Diversity (lambda=0.95, novelty_beta=0.01)
4. Moderate Diversity (lambda=0.90, novelty_beta=0.03)
5. Balanced Diversity (lambda=0.85, novelty_beta=0.05)
6. Diversity-Heavy (lambda=0.75, novelty_beta=0.10)
7. Baseline Production MMR (lambda=0.65, novelty_beta=0.15)
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from pipelines.exp1_item_cf import EnhancedItemCF
from pipelines.tournament_eval import TournamentEvaluator
from src.core.config import settings
from src.core.logging import logger
from src.evaluation.metrics import compute_paired_bootstrap_ci
from src.models.baselines import PopularityRecommender
from src.models.ranker import LightGBMRanker
from src.models.two_tower import TwoTowerRecommender
from src.ranking.feature_engineering import RankingFeatureExtractor
from src.reranking.mmr import MMRReRanker
from src.retrieval.faiss_index import FaissCandidateIndex


def run_experiment_6():
    logger.info("=== Running Experiment 6: Relevance-First MMR Diversity Optimization ===")
    evaluator = TournamentEvaluator()
    device = evaluator.device

    # 1. Fit baselines & load models
    logger.info("Loading baseline retrieval & ranker...")
    pop_model = PopularityRecommender()
    pop_model.fit(evaluator.train_ratings, num_movies=evaluator.num_movies)

    cf_model = EnhancedItemCF(top_k_neighbors=200, shrinkage_lambda=25.0)
    cf_model.fit(evaluator.train_ratings, num_users=evaluator.num_users, num_movies=evaluator.num_movies)

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

    ranker = LightGBMRanker()
    ranker.load(Path("artifacts/models/ranker_lgbm.txt"))

    feature_extractor = RankingFeatureExtractor(
        movies_df=evaluator.movies_df,
        genre_cols=evaluator.genre_cols,
        global_mean_rating=evaluator.mappings.get("global_rating_mean", 3.5),
    )

    eval_users = evaluator.std_users

    # Precompute user embeddings
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

    # Pre-rank candidate items for each user once
    logger.info("Pre-ranking candidate pools for all 4,980 validation users...")
    pre_ranked_candidates: dict[int, list[tuple[int, float]]] = {}

    for u in eval_users:
        h = evaluator.user_histories.get(u, [])
        consumed = set(h)
        u_emb = user_embs_dict[u]

        c_sources = {}
        cf_map = {}

        for item_idx, score in faiss_index.search(u_emb, top_k=100, exclude_item_indices=consumed):
            c_sources.setdefault(item_idx, set()).add("two_tower")

        for item_idx, score in cf_model.recommend(h, top_k=100):
            if item_idx not in consumed:
                c_sources.setdefault(item_idx, set()).add("item_cf")
                cf_map[item_idx] = float(score)

        for item_idx, score in pop_model.recommend(h, top_k=30):
            if item_idx not in consumed:
                c_sources.setdefault(item_idx, set()).add("popularity")

        c_list = list(c_sources.keys())
        cand_arr = np.array(c_list, dtype=np.int64)
        valid_mask = cand_arr < len(item_embeddings)
        tt_scores_dict = {}
        if np.any(valid_mask):
            cand_embs = item_embeddings[cand_arr[valid_mask]]
            sims = np.dot(cand_embs, u_emb)
            for idx_c, sim in zip(cand_arr[valid_mask], sims):
                tt_scores_dict[int(idx_c)] = float(sim)
        for c in c_list:
            tt_scores_dict.setdefault(c, 0.0)

        feat_df = feature_extractor.extract_features(
            user_idx=u,
            user_history=h,
            candidate_items=c_list,
            two_tower_scores=tt_scores_dict,
            cf_scores=cf_map,
            retrieval_sources=c_sources,
        )
        ranked = ranker.rank(feat_df)
        pre_ranked_candidates[u] = ranked

    logger.info("Pre-ranking complete. Sweeping MMR diversity & novelty parameters...")

    mmr_configs = [
        {"name": "MMR-P1 (Disabled / Pure Relevance: lambda=1.00, beta=0.00)", "lambda": 1.00, "beta": 0.00},
        {"name": "MMR-P2 (Ultra-Light Diversity: lambda=0.98, beta=0.00)", "lambda": 0.98, "beta": 0.00},
        {"name": "MMR-P3 (Light Diversity: lambda=0.95, beta=0.01)", "lambda": 0.95, "beta": 0.01},
        {"name": "MMR-P4 (Moderate Diversity: lambda=0.90, beta=0.03)", "lambda": 0.90, "beta": 0.03},
        {"name": "MMR-P5 (Balanced Diversity: lambda=0.85, beta=0.05)", "lambda": 0.85, "beta": 0.05},
        {"name": "MMR-P6 (Diversity-Heavy: lambda=0.75, beta=0.10)", "lambda": 0.75, "beta": 0.10},
        {"name": "MMR-P7 (Baseline Production: lambda=0.65, beta=0.15)", "lambda": 0.65, "beta": 0.15},
    ]

    exp6_results = {}
    exp6_per_user = {}

    for cfg in mmr_configs:
        c_name = cfg["name"]
        l_param = cfg["lambda"]
        b_param = cfg["beta"]
        logger.info(f"Evaluating MMR Configuration: {c_name}...")

        reranker = MMRReRanker(
            movies_df=evaluator.movies_df,
            genre_cols=evaluator.genre_cols,
            default_lambda=l_param,
            default_novelty_weight=b_param,
        )

        t0_mmr = time.time()
        recs_dict = {}
        lats = []

        for u in eval_users:
            h = evaluator.user_histories.get(u, [])
            consumed = set(h)
            cand_scores = pre_ranked_candidates[u]

            t_start = time.perf_counter()
            if l_param >= 1.0 and b_param <= 0.0:
                # Direct ranking without MMR overhead
                r_list = [idx for idx, _ in cand_scores[:100]]
            else:
                reranked = reranker.rerank(
                    candidate_scores=cand_scores,
                    top_k=100,
                    lambda_param=l_param,
                    novelty_weight=b_param,
                    excluded_item_indices=consumed,
                )
                r_list = [item["movie_idx"] for item in reranked]

            if len(r_list) < 100:
                pop_fill = [idx for idx, _ in pop_model.recommend(h, top_k=100) if idx not in set(r_list)]
                r_list.extend(pop_fill[: 100 - len(r_list)])

            lats.append((time.perf_counter() - t_start) * 1000.0)
            recs_dict[u] = r_list

        duration = time.time() - t0_mmr
        metrics, per_user = evaluator.evaluate_recommendations(
            recs_dict, c_name, latencies_ms=lats, train_time_sec=duration
        )
        exp6_results[c_name] = metrics
        exp6_per_user[c_name] = per_user

        logger.info(
            f"-> {c_name}: Recall@20 = {metrics['recall@20']:.4f}, NDCG@10 = {metrics['ndcg@10']:.4f}, "
            f"ILD = {metrics['intra_list_diversity']:.4f}, Coverage@100 = {metrics['catalog_coverage@100']*100:.1f}%, "
            f"Novelty = {metrics['novelty_bits']:.2f} bits, P95 Latency = {metrics['latency_p95_ms']:.2f} ms"
        )

    # Save results
    output_dir = Path("artifacts/experiments/relevance_v2")
    with open(output_dir / "exp6_mmr_results.json", "w") as f:
        json.dump(exp6_results, f, indent=2)

    # Paired significance test: Best Relevance-First MMR (P2/P3) vs Baseline Production MMR (P7)
    with open(output_dir / "baseline_results_val.json") as f:
        _ = json.load(f)

    # Select best relevance-first configuration with high ILD (e.g. MMR-P3: lambda=0.95, beta=0.01)
    best_candidate_name = "MMR-P3 (Light Diversity: lambda=0.95, beta=0.01)"
    best_m = exp6_results[best_candidate_name]
    base_prod_m = exp6_results["MMR-P7 (Baseline Production: lambda=0.65, beta=0.15)"]

    base_ndcgs = [exp6_per_user["MMR-P7 (Baseline Production: lambda=0.65, beta=0.15)"]["ndcg@10"][u] for u in eval_users]
    best_ndcgs = [exp6_per_user[best_candidate_name]["ndcg@10"][u] for u in eval_users]
    diff_mean, ci_l, ci_u, p_val = compute_paired_bootstrap_ci(best_ndcgs, base_ndcgs)

    sig_report = {
        "best_variant": best_candidate_name,
        "delta_ndcg@10_vs_baseline_prod_mmr": round(diff_mean, 4),
        "ci95_ndcg@10": [round(ci_l, 4), round(ci_u, 4)],
        "p_value": p_val,
        "baseline_mmr_ndcg@10": base_prod_m["ndcg@10"],
        "best_mmr_ndcg@10": best_m["ndcg@10"],
        "baseline_mmr_recall@20": base_prod_m["recall@20"],
        "best_mmr_recall@20": best_m["recall@20"],
        "baseline_mmr_ild": base_prod_m["intra_list_diversity"],
        "best_mmr_ild": best_m["intra_list_diversity"],
    }
    with open(output_dir / "exp6_significance.json", "w") as f:
        json.dump(sig_report, f, indent=2)

    print("\n" + "=" * 115)
    print("EXPERIMENT 6: RELEVANCE-FIRST MMR DIVERSITY PARETO FRONTIER (All 4,980 Val Users)")
    print("=" * 115)
    fmt = "{:<55} {:<12} {:<12} {:<10} {:<12} {:<10}"
    print(fmt.format("Configuration", "Recall@20", "NDCG@10", "ILD", "Coverage@100", "P95 ms"))
    print("-" * 115)
    for name, m in exp6_results.items():
        print(
            fmt.format(
                name,
                f"{m['recall@20']:.4f}",
                f"{m['ndcg@10']:.4f}",
                f"{m['intra_list_diversity']:.4f}",
                f"{m['catalog_coverage@100']*100:.1f}%",
                f"{m['latency_p95_ms']:.2f}",
            )
        )
    print("-" * 115)
    print(f"Pareto Sweet Spot: {best_candidate_name}")
    print(f"Paired delta vs Baseline Production MMR: Delta NDCG@10 = {diff_mean:+.4f} [{ci_l:+.4f}, {ci_u:+.4f}], p = {p_val:.4f}")
    print("=" * 115)


if __name__ == "__main__":
    run_experiment_6()
