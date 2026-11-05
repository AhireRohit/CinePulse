"""
Tournament Grand Champion Pipeline:
Combines the empirically proven winners from Experiments 1 through 6:
1. Candidate Retrieval:
   - Two-Tower top-200 (FAISS IndexFlatIP)
   - Best Shrunk Item-CF top-100 (K=200, lambda=25.0)
   - Popularity top-30
   -> Yields 60.5% candidate union recall / oracle ceiling with only 12.7 ms latency.
2. Ranking:
   - High-Fidelity 24-Feature Vectorized Ranking with LightGBM LambdaRank
   - Tuned tree capacity (num_leaves=63, max_depth=8, reg_alpha=0.1, reg_lambda=1.0)
   - Leverages multi-source agreement signals and calibrated rating offsets.
3. Hybrid Score Fusion:
   - Blends normalized LightGBM score with CF and Two-Tower similarity
   - Preserves high ranking fidelity while anchoring on strong collaborative signals.
4. Relevance-First MMR:
   - Uses Pareto sweet spot (lambda=0.95, beta=0.01)
   - Resolves the catastrophic -34% NDCG drop of baseline MMR while preserving ILD > 0.70.
5. Ablation study isolating each component's incremental value.
6. Rigorous paired bootstrap statistical significance testing vs both baseline models.
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from pipelines.exp1_item_cf import EnhancedItemCF
from pipelines.exp3_ranker import FastRankingFeatureExtractor, construct_fast_ranking_dataset
from pipelines.tournament_eval import TournamentEvaluator
from src.core.config import settings
from src.core.logging import logger
from src.evaluation.metrics import compute_paired_bootstrap_ci
from src.models.baselines import PopularityRecommender
from src.models.two_tower import TwoTowerRecommender
from src.reranking.mmr import MMRReRanker
from src.retrieval.faiss_index import FaissCandidateIndex


def run_tournament_winner_evaluation():
    logger.info("=== Executing Tournament Grand Champion Pipeline Evaluation & Ablation ===")
    evaluator = TournamentEvaluator()
    device = evaluator.device

    output_dir = Path("artifacts/experiments/relevance_v2")
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fit best Item-CF and Popularity
    logger.info("Initializing component models...")
    pop_model = PopularityRecommender()
    pop_model.fit(evaluator.train_ratings, num_movies=evaluator.num_movies)

    cf_model = EnhancedItemCF(top_k_neighbors=200, shrinkage_lambda=25.0)
    cf_model.fit(evaluator.train_ratings, num_users=evaluator.num_users, num_movies=evaluator.num_movies)

    # 2. Load Two-Tower & FAISS
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

    fast_extractor = FastRankingFeatureExtractor(
        movies_df=evaluator.movies_df,
        genre_cols=evaluator.genre_cols,
        global_mean_rating=evaluator.mappings.get("global_rating_mean", 3.5),
    )

    # 3. Train or Load Champion LightGBM Ranker (from Exp 3)
    champion_ranker_path = models_dir / "ranker_lgbm_exp3.txt"
    if champion_ranker_path.exists():
        logger.info(f"Loading champion ranker from {champion_ranker_path}...")
        booster = lgb.Booster(model_file=str(champion_ranker_path))
    else:
        logger.info("Champion ranker not found on disk; training champion model...")
        X_train_base, X_train_exp, y_train, group_train = construct_fast_ranking_dataset(
            ratings_df=evaluator.train_ratings,
            movies_df=evaluator.movies_df,
            genre_cols=evaluator.genre_cols,
            user_histories=evaluator.user_histories,
            item_embeddings=item_embeddings,
            tt_model=tt_model,
            pop_model=pop_model,
            cf_model=cf_model,
            faiss_index=faiss_index,
            device=device,
            feature_extractor=fast_extractor,
            num_users_to_sample=3500,
            seed=42,
        )
        X_val_base, X_val_exp, y_val, group_val = construct_fast_ranking_dataset(
            ratings_df=pd.read_parquet("data/processed/ratings_val.parquet"),
            movies_df=evaluator.movies_df,
            genre_cols=evaluator.genre_cols,
            user_histories=evaluator.user_histories,
            item_embeddings=item_embeddings,
            tt_model=tt_model,
            pop_model=pop_model,
            cf_model=cf_model,
            faiss_index=faiss_index,
            device=device,
            feature_extractor=fast_extractor,
            num_users_to_sample=800,
            seed=101,
        )
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[5, 10, 20],
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
            n_estimators=350,
            learning_rate=0.025,
            num_leaves=63,
            max_depth=8,
            min_child_samples=30,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        model.fit(
            X_train_exp,
            y_train,
            group=group_train,
            eval_set=[(X_val_exp, y_val)],
            eval_group=[group_val],
            callbacks=callbacks,
        )
        model.booster_.save_model(str(champion_ranker_path))
        booster = model.booster_

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

    user_val_ratings_map: dict[int, dict[int, float]] = {}
    for row in evaluator.train_ratings[["user_idx", "movie_idx", "rating"]].itertuples(index=False):
        u, m, r = int(row.user_idx), int(row.movie_idx), float(row.rating)
        user_val_ratings_map.setdefault(u, {})[m] = r

    # Rerankers for ablation
    champion_mmr = MMRReRanker(
        movies_df=evaluator.movies_df,
        genre_cols=evaluator.genre_cols,
        default_lambda=0.95,
        default_novelty_weight=0.01,
    )
    prod_mmr = MMRReRanker(
        movies_df=evaluator.movies_df,
        genre_cols=evaluator.genre_cols,
        default_lambda=0.65,
        default_novelty_weight=0.15,
    )

    # We evaluate 5 controlled pipeline variants for complete ablation:
    # 1. Grand Champion (Full Tournament Winner: Ret-C3 + 24f LGBM + Score Fusion + MMR-P3)
    # 2. Champion Pure Ranker (No MMR: Ret-C3 + 24f LGBM + Score Fusion)
    # 3. Champion without Fusion (Ret-C3 + 24f LGBM direct + MMR-P3)
    # 4. Champion with Baseline MMR (Ret-C3 + 24f LGBM + Fusion + Old Prod MMR lambda=0.65)
    # 5. Baseline Hybrid Reproduction (Old TT + Old LGBM + Old MMR)
    pipelines_to_eval = [
        "1. Grand Champion (Full Pipeline: Winner)",
        "2. Champion Pure Ranker (Without MMR)",
        "3. Champion Ranker (Without Score Fusion)",
        "4. Champion Pipeline (With Old Baseline MMR lambda=0.65)",
    ]

    recs_by_pipeline = {p: {} for p in pipelines_to_eval}
    lats_by_pipeline = {p: [] for p in pipelines_to_eval}

    logger.info("Executing end-to-end evaluation for all 4,980 validation users...")
    for u in eval_users:
        h = evaluator.user_histories.get(u, [])
        consumed = set(h)
        u_emb = user_embs_dict[u]
        u_ratings = user_val_ratings_map.get(u, {})

        # Stage A: Candidate Retrieval (TT=200, CF=100, Pop=30)
        c_sources = {}
        cf_map = {}

        t0_start = time.perf_counter()
        for item_idx, score in faiss_index.search(u_emb, top_k=200, exclude_item_indices=consumed):
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

        # Stage B: Feature Extraction (24 Causal Features)
        feat_mat = fast_extractor.extract_matrix(
            u, h, u_ratings, c_list, tt_scores_dict, cf_map, c_sources, use_expanded=True
        )

        # Stage C: LightGBM Scoring
        lgbm_scores = booster.predict(feat_mat)

        # Score Normalization
        def min_max(arr):
            mn, mx = arr.min(), arr.max()
            diff = mx - mn
            return (arr - mn) / (diff + 1e-6) if diff > 0 else np.zeros_like(arr)

        lgbm_norm = min_max(lgbm_scores)
        cf_raw = np.array([cf_map.get(c, 0.0) for c in c_list], dtype=np.float32)
        tt_raw = np.array([tt_scores_dict.get(c, 0.0) for c in c_list], dtype=np.float32)
        cf_norm = min_max(cf_raw)
        tt_norm = min_max(tt_raw)

        # Score Fusion: 0.75 LGBM + 0.18 CF + 0.07 TT
        fused_scores = (0.75 * lgbm_norm) + (0.18 * cf_norm) + (0.07 * tt_norm)

        # 1. Champion Pure Ranker (Fused)
        pure_fused_order = np.argsort(-fused_scores)
        pure_recs = [c_list[idx] for idx in pure_fused_order]
        recs_by_pipeline["2. Champion Pure Ranker (Without MMR)"][u] = pure_recs[:100]
        lats_by_pipeline["2. Champion Pure Ranker (Without MMR)"].append((time.perf_counter() - t0_start) * 1000.0)

        # Candidate score pairs for MMR
        fused_pairs = [(c_list[idx], float(fused_scores[idx])) for idx in pure_fused_order]
        lgbm_direct_pairs = [(c_list[idx], float(lgbm_scores[idx])) for idx in np.argsort(-lgbm_scores)]

        # 2. Grand Champion (Fused + Relevance-First MMR lambda=0.95, beta=0.01)
        champ_reranked = champion_mmr.rerank(
            candidate_scores=fused_pairs,
            top_k=100,
            lambda_param=0.95,
            novelty_weight=0.01,
            excluded_item_indices=consumed,
        )
        champ_recs = [item["movie_idx"] for item in champ_reranked]
        recs_by_pipeline["1. Grand Champion (Full Pipeline: Winner)"][u] = champ_recs
        lats_by_pipeline["1. Grand Champion (Full Pipeline: Winner)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 3. Champion without Fusion (Direct LightGBM + MMR)
        no_fusion_reranked = champion_mmr.rerank(
            candidate_scores=lgbm_direct_pairs,
            top_k=100,
            lambda_param=0.95,
            novelty_weight=0.01,
            excluded_item_indices=consumed,
        )
        recs_by_pipeline["3. Champion Ranker (Without Score Fusion)"][u] = [item["movie_idx"] for item in no_fusion_reranked]
        lats_by_pipeline["3. Champion Ranker (Without Score Fusion)"].append((time.perf_counter() - t0_start) * 1000.0)

        # 4. Champion with Baseline MMR (Fused + Old MMR lambda=0.65, beta=0.15)
        old_mmr_reranked = prod_mmr.rerank(
            candidate_scores=fused_pairs,
            top_k=100,
            lambda_param=0.65,
            novelty_weight=0.15,
            excluded_item_indices=consumed,
        )
        recs_by_pipeline["4. Champion Pipeline (With Old Baseline MMR lambda=0.65)"][u] = [item["movie_idx"] for item in old_mmr_reranked]
        lats_by_pipeline["4. Champion Pipeline (With Old Baseline MMR lambda=0.65)"].append((time.perf_counter() - t0_start) * 1000.0)

    # Evaluate all variants
    tournament_results = {}
    tournament_per_user = {}

    for p_name in pipelines_to_eval:
        m, per_user = evaluator.evaluate_recommendations(
            recs_by_pipeline[p_name],
            p_name,
            latencies_ms=lats_by_pipeline[p_name],
            train_time_sec=0.0,
        )
        tournament_results[p_name] = m
        tournament_per_user[p_name] = per_user

    # Save results
    with open(output_dir / "tournament_champion_results.json", "w") as f:
        json.dump(tournament_results, f, indent=2)

    # Load baseline results
    with open(output_dir / "baseline_results_val.json") as f:
        baseline_results = json.load(f)
    base_ranker = baseline_results["Existing Pure Ranker"]
    base_full = baseline_results["Existing Full Pipeline"]

    with open(output_dir / "baseline_per_user_val.json") as f:
        base_user_data = json.load(f)

    # Significance Tests
    eval_u_list = evaluator.std_users
    champ_ndcgs = [tournament_per_user["1. Grand Champion (Full Pipeline: Winner)"]["ndcg@10"][u] for u in eval_u_list]
    pure_champ_ndcgs = [tournament_per_user["2. Champion Pure Ranker (Without MMR)"]["ndcg@10"][u] for u in eval_u_list]
    base_full_ndcgs = [base_user_data["Existing Full Pipeline"]["ndcg@10"][str(u)] for u in eval_u_list]
    base_pure_ndcgs = [base_user_data["Existing Pure Ranker"]["ndcg@10"][str(u)] for u in eval_u_list]

    diff1, c1_l, c1_u, p1 = compute_paired_bootstrap_ci(champ_ndcgs, base_full_ndcgs)
    diff2, c2_l, c2_u, p2 = compute_paired_bootstrap_ci(pure_champ_ndcgs, base_pure_ndcgs)

    final_significance = {
        "grand_champion_vs_baseline_full_pipeline": {
            "delta_ndcg@10": round(diff1, 4),
            "ci95_ndcg@10": [round(c1_l, 4), round(c1_u, 4)],
            "p_value": p1,
            "baseline_full_ndcg@10": base_full["ndcg@10"],
            "champion_full_ndcg@10": tournament_results["1. Grand Champion (Full Pipeline: Winner)"]["ndcg@10"],
            "baseline_full_recall@20": base_full["recall@20"],
            "champion_full_recall@20": tournament_results["1. Grand Champion (Full Pipeline: Winner)"]["recall@20"],
            "baseline_full_ild": base_full["intra_list_diversity"],
            "champion_full_ild": tournament_results["1. Grand Champion (Full Pipeline: Winner)"]["intra_list_diversity"],
        },
        "pure_champion_vs_baseline_pure_ranker": {
            "delta_ndcg@10": round(diff2, 4),
            "ci95_ndcg@10": [round(c2_l, 4), round(c2_u, 4)],
            "p_value": p2,
            "baseline_pure_ndcg@10": base_ranker["ndcg@10"],
            "champion_pure_ndcg@10": tournament_results["2. Champion Pure Ranker (Without MMR)"]["ndcg@10"],
            "baseline_pure_recall@20": base_ranker["recall@20"],
            "champion_pure_recall@20": tournament_results["2. Champion Pure Ranker (Without MMR)"]["recall@20"],
        },
    }
    with open(output_dir / "tournament_champion_significance.json", "w") as f:
        json.dump(final_significance, f, indent=2)

    print("\n" + "=" * 125)
    print("TOURNAMENT GRAND CHAMPION & ABLATION RESULTS (All 4,980 Val Users)")
    print("=" * 125)
    fmt = "{:<55} {:<12} {:<12} {:<10} {:<12} {:<10}"
    print(fmt.format("Configuration", "Recall@20", "NDCG@10", "ILD", "Coverage@100", "P95 ms"))
    print("-" * 125)
    print(
        fmt.format(
            "Existing Full Pipeline Baseline (TT+LGBM+MMR)",
            f"{base_full['recall@20']:.4f}",
            f"{base_full['ndcg@10']:.4f}",
            f"{base_full['intra_list_diversity']:.4f}",
            f"{base_full['catalog_coverage@100']*100:.1f}%",
            f"{base_full['latency_p95_ms']:.2f}",
        )
    )
    print(
        fmt.format(
            "Existing Pure LightGBM Ranker Baseline",
            f"{base_ranker['recall@20']:.4f}",
            f"{base_ranker['ndcg@10']:.4f}",
            f"{base_ranker['intra_list_diversity']:.4f}",
            f"{base_ranker['catalog_coverage@100']*100:.1f}%",
            f"{base_ranker['latency_p95_ms']:.2f}",
        )
    )
    print("-" * 125)
    for name, m in tournament_results.items():
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
    print("-" * 125)
    print(
        f"Grand Champion vs Baseline Full Pipeline: Delta NDCG@10 = {diff1:+.4f} [{c1_l:+.4f}, {c1_u:+.4f}], p = {p1:.4f}"
    )
    print(
        f"Champion Pure Ranker vs Baseline Pure Ranker: Delta NDCG@10 = {diff2:+.4f} [{c2_l:+.4f}, {c2_u:+.4f}], p = {p2:.4f}"
    )
    print("=" * 125)


if __name__ == "__main__":
    run_tournament_winner_evaluation()
