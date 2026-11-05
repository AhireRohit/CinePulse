"""
Experiment 5: Adaptive Hybrid Score Fusion.
Controlled investigation of multi-model score combination:
1. Pure LightGBM ranker score (alpha=1.0, beta=0.0, gamma=0.0)
2. Equal Weight Linear Blend (alpha=0.6, beta=0.25, gamma=0.15)
3. CF-Enhanced Ranker Blend (alpha=0.75, beta=0.20, gamma=0.05)
4. Two-Tower Enhanced Blend (alpha=0.70, beta=0.10, gamma=0.20)
5. Adaptive History-Gated Fusion:
   - Short history (<= 15 ratings): higher CF weight (beta=0.35, alpha=0.55, gamma=0.10)
   - Medium history (16-60 ratings): balanced (beta=0.20, alpha=0.70, gamma=0.10)
   - Long history (> 60 ratings): model heavy (beta=0.10, alpha=0.75, gamma=0.15)
6. Preference Concentration-Gated Fusion:
   - Concentrated genre focus: boost CF
   - Diverse broad taste: boost LightGBM
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
from src.retrieval.faiss_index import FaissCandidateIndex


def run_experiment_5():
    logger.info("=== Running Experiment 5: Adaptive Hybrid Score Fusion ===")
    evaluator = TournamentEvaluator()
    device = evaluator.device

    # 1. Fit baselines & load models
    logger.info("Loading models...")
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

    # Pre-extract scores and candidates per user to run all fusion variants in seconds
    logger.info("Precomputing individual model candidate score tuples for all 4,980 validation users...")
    user_candidate_data = {}  # u -> list of (item_idx, lgbm_score_norm, cf_score_norm, tt_score_norm)
    user_hist_lens = {}
    user_genre_concentrations = {}

    for u in eval_users:
        h = evaluator.user_histories.get(u, [])
        consumed = set(h)
        u_emb = user_embs_dict[u]
        user_hist_lens[u] = len(h)

        # Genre concentration: max genre proportion in history
        if h:
            valid_h = [m for m in h if m < evaluator.num_movies]
            if valid_h:
                g_counts = evaluator.item_genres[valid_h].sum(axis=0)
                tot = g_counts.sum()
                user_genre_concentrations[u] = float(g_counts.max() / tot) if tot > 0 else 0.0
            else:
                user_genre_concentrations[u] = 0.0
        else:
            user_genre_concentrations[u] = 0.0

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
        lgbm_scores_dict = dict(ranker.rank(feat_df))

        # Normalize scores within candidate set
        lgbm_raw = np.array([lgbm_scores_dict.get(c, -10.0) for c in c_list], dtype=np.float32)
        cf_raw = np.array([cf_map.get(c, 0.0) for c in c_list], dtype=np.float32)
        tt_raw = np.array([tt_scores_dict.get(c, 0.0) for c in c_list], dtype=np.float32)

        def min_max(arr):
            mn, mx = arr.min(), arr.max()
            diff = mx - mn
            return (arr - mn) / (diff + 1e-6) if diff > 0 else np.zeros_like(arr)

        lgbm_norm = min_max(lgbm_raw)
        cf_norm = min_max(cf_raw)
        tt_norm = min_max(tt_raw)

        user_candidate_data[u] = list(zip(c_list, lgbm_norm.tolist(), cf_norm.tolist(), tt_norm.tolist()))

    logger.info("Candidate scoring precomputed. Evaluating score fusion variants...")

    fusion_variants = [
        {"name": "Fusion-V1 (Pure LightGBM: alpha=1.0)", "type": "fixed", "alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        {"name": "Fusion-V2 (Equal Blend: 0.60 LGBM + 0.25 CF + 0.15 TT)", "type": "fixed", "alpha": 0.60, "beta": 0.25, "gamma": 0.15},
        {"name": "Fusion-V3 (CF-Heavy Blend: 0.70 LGBM + 0.25 CF + 0.05 TT)", "type": "fixed", "alpha": 0.70, "beta": 0.25, "gamma": 0.05},
        {"name": "Fusion-V4 (TT-Heavy Blend: 0.70 LGBM + 0.10 CF + 0.20 TT)", "type": "fixed", "alpha": 0.70, "beta": 0.10, "gamma": 0.20},
        {"name": "Fusion-V5 (High-Precision LGBM: 0.85 LGBM + 0.10 CF + 0.05 TT)", "type": "fixed", "alpha": 0.85, "beta": 0.10, "gamma": 0.05},
        {"name": "Fusion-V6 (Adaptive History-Length Gating)", "type": "history_gated"},
        {"name": "Fusion-V7 (Adaptive Concentration Gating)", "type": "concentration_gated"},
    ]

    exp5_results = {}
    exp5_per_user = {}

    for var in fusion_variants:
        v_name = var["name"]
        v_type = var["type"]
        t0_fuse = time.time()
        recs_dict = {}
        lats = []

        for u in eval_users:
            t_start = time.perf_counter()
            items_scores = user_candidate_data[u]
            h_len = user_hist_lens[u]
            conc = user_genre_concentrations[u]

            if v_type == "fixed":
                a, b, g = var["alpha"], var["beta"], var["gamma"]
            elif v_type == "history_gated":
                if h_len <= 20:
                    a, b, g = 0.55, 0.35, 0.10
                elif h_len <= 75:
                    a, b, g = 0.75, 0.18, 0.07
                else:
                    a, b, g = 0.82, 0.10, 0.08
            elif v_type == "concentration_gated":
                if conc >= 0.35:
                    # User is strongly specialized in a specific genre -> trust CF more
                    a, b, g = 0.65, 0.30, 0.05
                else:
                    # User has broad taste -> trust LightGBM cross-features
                    a, b, g = 0.85, 0.08, 0.07
            else:
                a, b, g = 1.0, 0.0, 0.0

            scored_items = []
            for item_idx, s_lgbm, s_cf, s_tt in items_scores:
                final_s = (a * s_lgbm) + (b * s_cf) + (g * s_tt)
                scored_items.append((item_idx, final_s))

            scored_items.sort(key=lambda x: x[1], reverse=True)
            r_list = [idx for idx, _ in scored_items]

            if len(r_list) < 100:
                h = evaluator.user_histories.get(u, [])
                pop_fill = [idx for idx, _ in pop_model.recommend(h, top_k=100) if idx not in set(r_list)]
                r_list.extend(pop_fill[: 100 - len(r_list)])

            lats.append((time.perf_counter() - t_start) * 1000.0)
            recs_dict[u] = r_list

        duration = time.time() - t0_fuse
        metrics, per_user = evaluator.evaluate_recommendations(
            recs_dict, v_name, latencies_ms=lats, train_time_sec=duration
        )
        exp5_results[v_name] = metrics
        exp5_per_user[v_name] = per_user

        logger.info(
            f"-> {v_name}: Recall@20 = {metrics['recall@20']:.4f}, NDCG@10 = {metrics['ndcg@10']:.4f}, "
            f"Pos NDCG@10 = {metrics['positive_only_ndcg@10']:.4f}, Latency P95 = {metrics['latency_p95_ms']:.2f} ms"
        )

    # Save results
    output_dir = Path("artifacts/experiments/relevance_v2")
    with open(output_dir / "exp5_fusion_results.json", "w") as f:
        json.dump(exp5_results, f, indent=2)

    best_variant_name = max(exp5_results.keys(), key=lambda k: exp5_results[k]["ndcg@10"])
    best_m = exp5_results[best_variant_name]
    pure_m = exp5_results["Fusion-V1 (Pure LightGBM: alpha=1.0)"]

    # Paired significance test
    pure_ndcgs = [exp5_per_user["Fusion-V1 (Pure LightGBM: alpha=1.0)"]["ndcg@10"][u] for u in eval_users]
    best_ndcgs = [exp5_per_user[best_variant_name]["ndcg@10"][u] for u in eval_users]
    diff_mean, ci_l, ci_u, p_val = compute_paired_bootstrap_ci(best_ndcgs, pure_ndcgs)

    sig_report = {
        "best_variant": best_variant_name,
        "delta_ndcg@10_vs_pure_lgbm": round(diff_mean, 4),
        "ci95_ndcg@10": [round(ci_l, 4), round(ci_u, 4)],
        "p_value": p_val,
        "pure_lgbm_ndcg@10": pure_m["ndcg@10"],
        "best_ndcg@10": best_m["ndcg@10"],
        "pure_lgbm_recall@20": pure_m["recall@20"],
        "best_recall@20": best_m["recall@20"],
    }
    with open(output_dir / "exp5_significance.json", "w") as f:
        json.dump(sig_report, f, indent=2)

    print("\n" + "=" * 115)
    print("EXPERIMENT 5: ADAPTIVE HYBRID SCORE FUSION ABLATION RESULTS (All 4,980 Val Users)")
    print("=" * 115)
    fmt = "{:<55} {:<12} {:<12} {:<12} {:<12} {:<10}"
    print(fmt.format("Configuration", "Recall@20", "NDCG@10", "Recall@100", "Pos NDCG@10", "P95 ms"))
    print("-" * 115)
    for name, m in exp5_results.items():
        print(
            fmt.format(
                name,
                f"{m['recall@20']:.4f}",
                f"{m['ndcg@10']:.4f}",
                f"{m['recall@100']:.4f}",
                f"{m['positive_only_ndcg@10']:.4f}",
                f"{m['latency_p95_ms']:.2f}",
            )
        )
    print("-" * 115)
    print(f"Tournament Best: {best_variant_name}")
    print(f"Paired delta vs Pure LightGBM: Delta NDCG@10 = {diff_mean:+.4f} [{ci_l:+.4f}, {ci_u:+.4f}], p = {p_val:.4f}")
    print("=" * 115)


if __name__ == "__main__":
    run_experiment_5()
