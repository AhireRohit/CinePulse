"""
Experiment 2: Candidate Retrieval Budget and Multi-Source Blending.
Investigates candidate retrieval stage depth and multi-source trade-offs:
1. Two-Tower retrieval depth K_TT in {100, 200, 300}
2. Item-CF retrieval depth K_CF in {50, 100, 150}
3. Popularity retrieval depth K_pop in {20, 30}
4. Single-source vs Multi-source retrieval
5. Downstream impact on Pure LightGBM Ranker and Full Pipeline
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
from src.models.baselines import PopularityRecommender
from src.models.ranker import LightGBMRanker
from src.models.two_tower import TwoTowerRecommender
from src.ranking.feature_engineering import RankingFeatureExtractor
from src.retrieval.faiss_index import FaissCandidateIndex


def run_experiment_2():
    logger.info("=== Running Experiment 2: Candidate Retrieval Budget & Multi-Source Blending ===")
    evaluator = TournamentEvaluator()
    device = evaluator.device

    # 1. Fit Popularity and Best Item-CF
    logger.info("Fitting candidate generators...")
    pop_model = PopularityRecommender()
    pop_model.fit(evaluator.train_ratings, num_movies=evaluator.num_movies)

    # Use the best CF model found in Exp 1 (CF-V6: K=200, shrinkage=25.0)
    best_cf_model = EnhancedItemCF(top_k_neighbors=200, shrinkage_lambda=25.0)
    best_cf_model.fit(evaluator.train_ratings, num_users=evaluator.num_users, num_movies=evaluator.num_movies)

    # 2. Load Two-Tower & FAISS Index
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

    # 3. Load LightGBM Ranker and Feature Extractor
    ranker = LightGBMRanker()
    ranker.load(Path("artifacts/models/ranker_lgbm.txt"))

    feature_extractor = RankingFeatureExtractor(
        movies_df=evaluator.movies_df,
        genre_cols=evaluator.genre_cols,
        global_mean_rating=evaluator.mappings.get("global_rating_mean", 3.5),
    )

    eval_users = evaluator.std_users  # All 4,980 validation users

    # Precompute user embeddings in batches
    logger.info("Precomputing Two-Tower user embeddings for all validation users...")
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
                    g_prefs[idx] = evaluator.item_genres[hist].mean(axis=0)

            u_tensor = torch.tensor(u_arr, dtype=torch.long, device=device)
            h_tensor = torch.tensor(padded_hist, dtype=torch.long, device=device)
            hl_tensor = torch.tensor(h_lens, dtype=torch.long, device=device)
            gp_tensor = torch.tensor(g_prefs, dtype=torch.float32, device=device)

            embs = tt_model.user_tower(u_tensor, h_tensor, hl_tensor, gp_tensor).cpu().numpy()
            for idx, u in enumerate(b_users):
                user_embs_dict[u] = embs[idx]

    # Retrieval configurations to evaluate
    retrieval_configs = [
        {"name": "Ret-C1 (Baseline: TT=100, CF=100, Pop=30)", "k_tt": 100, "k_cf": 100, "k_pop": 30},
        {"name": "Ret-C2 (Narrow: TT=100, CF=50, Pop=20)", "k_tt": 100, "k_cf": 50, "k_pop": 20},
        {"name": "Ret-C3 (Deep TT: TT=200, CF=100, Pop=30)", "k_tt": 200, "k_cf": 100, "k_pop": 30},
        {"name": "Ret-C4 (Wide TT: TT=300, CF=100, Pop=30)", "k_tt": 300, "k_cf": 100, "k_pop": 30},
        {"name": "Ret-C5 (Deep Hybrid: TT=200, CF=150, Pop=30)", "k_tt": 200, "k_cf": 150, "k_pop": 30},
        {"name": "Ret-C6 (Max Hybrid: TT=300, CF=150, Pop=30)", "k_tt": 300, "k_cf": 150, "k_pop": 30},
        {"name": "Ret-C7 (TT Only: TT=200, CF=0, Pop=0)", "k_tt": 200, "k_cf": 0, "k_pop": 0},
        {"name": "Ret-C8 (CF Only: TT=0, CF=200, Pop=0)", "k_tt": 0, "k_cf": 200, "k_pop": 0},
    ]

    exp2_results = {}
    exp2_per_user = {}

    for cfg in retrieval_configs:
        c_name = cfg["name"]
        k_tt = cfg["k_tt"]
        k_cf = cfg["k_cf"]
        k_pop = cfg["k_pop"]
        logger.info(f"Evaluating Candidate Retrieval Config: {c_name}...")

        t_start_cfg = time.time()
        pool_sizes = []
        pool_hits = []  # Fraction of users whose target item is in candidate pool
        ret_lats = []
        rank_lats = []
        ranked_recs = {}

        for u in eval_users:
            h = evaluator.user_histories.get(u, [])
            consumed = set(h)
            u_emb = user_embs_dict[u]
            target_item = next(iter(evaluator.ground_truth_std[u]))

            t0 = time.perf_counter()
            candidate_sources = {}
            cf_score_map = {}

            # Two-Tower
            if k_tt > 0:
                for item_idx, score in faiss_index.search(u_emb, top_k=k_tt, exclude_item_indices=consumed):
                    candidate_sources.setdefault(item_idx, set()).add("two_tower")

            # Best Item-CF
            if k_cf > 0:
                for item_idx, score in best_cf_model.recommend(h, top_k=k_cf):
                    if item_idx not in consumed:
                        candidate_sources.setdefault(item_idx, set()).add("item_cf")
                        cf_score_map[item_idx] = float(score)

            # Popularity
            if k_pop > 0:
                for item_idx, score in pop_model.recommend(h, top_k=k_pop):
                    if item_idx not in consumed:
                        candidate_sources.setdefault(item_idx, set()).add("popularity")

            t_ret = time.perf_counter()
            ret_lats.append((t_ret - t0) * 1000.0)

            c_indices = list(candidate_sources.keys())
            pool_sizes.append(len(c_indices))
            pool_hits.append(1 if target_item in candidate_sources else 0)

            # Downstream LightGBM Ranking
            cand_arr = np.array(c_indices, dtype=np.int64)
            if len(cand_arr) > 0:
                valid_mask = cand_arr < len(item_embeddings)
                cand_embs = item_embeddings[cand_arr[valid_mask]]
                dots = np.dot(cand_embs, u_emb)
                tt_scores = {int(idx_c): float(sim) for idx_c, sim in zip(cand_arr[valid_mask], dots)}
            else:
                tt_scores = {}
            for c in c_indices:
                tt_scores.setdefault(c, 0.0)

            cf_scores_dict = {c: cf_score_map.get(c, 0.0) for c in c_indices}
            sources = {c: candidate_sources[c] for c in c_indices}

            t_rank_start = time.perf_counter()
            feat_df = feature_extractor.extract_features(
                user_idx=u,
                user_history=h,
                candidate_items=c_indices,
                two_tower_scores=tt_scores,
                cf_scores=cf_scores_dict,
                retrieval_sources=sources,
            )
            ranked = ranker.rank(feat_df)
            t_rank_end = time.perf_counter()
            rank_lats.append((t_rank_end - t_rank_start) * 1000.0)

            recs = [idx for idx, _ in ranked]
            if len(recs) < 100:
                pop_fill = [idx for idx, _ in pop_model.recommend(h, top_k=100) if idx not in set(recs)]
                recs.extend(pop_fill[: 100 - len(recs)])
            ranked_recs[u] = recs

        total_duration = time.time() - t_start_cfg
        metrics, per_user = evaluator.evaluate_recommendations(
            ranked_recs, c_name, latencies_ms=rank_lats, train_time_sec=total_duration
        )

        # Enrich metrics with candidate pool statistics
        metrics["candidate_pool_mean"] = round(float(np.mean(pool_sizes)), 1)
        metrics["candidate_pool_p95"] = round(float(np.percentile(pool_sizes, 95)), 1)
        metrics["candidate_union_recall"] = round(float(np.mean(pool_hits)), 4)
        metrics["oracle_hit_rate_ceiling"] = metrics["candidate_union_recall"]
        metrics["retrieval_p50_ms"] = round(float(np.median(ret_lats)), 2)
        metrics["retrieval_p95_ms"] = round(float(np.percentile(ret_lats, 95)), 2)
        metrics["ranking_p95_ms"] = round(float(np.percentile(rank_lats, 95)), 2)

        exp2_results[c_name] = metrics
        exp2_per_user[c_name] = per_user

        logger.info(
            f"-> {c_name}: Union Recall = {metrics['candidate_union_recall']:.4f}, "
            f"Ranker Recall@20 = {metrics['recall@20']:.4f}, NDCG@10 = {metrics['ndcg@10']:.4f}, "
            f"Pool Size = {metrics['candidate_pool_mean']} (P95: {metrics['candidate_pool_p95']}), "
            f"Ret P95 Latency = {metrics['retrieval_p95_ms']} ms"
        )

    # Save results
    output_dir = Path("artifacts/experiments/relevance_v2")
    with open(output_dir / "exp2_retrieval_results.json", "w") as f:
        json.dump(exp2_results, f, indent=2)

    print("\n" + "=" * 115)
    print("EXPERIMENT 2: CANDIDATE RETRIEVAL BUDGET & MULTI-SOURCE BLENDING (All 4,980 Val Users)")
    print("=" * 115)
    fmt = "{:<40} {:<12} {:<12} {:<12} {:<12} {:<12} {:<10}"
    print(fmt.format("Configuration", "Pool Mean", "Union Recall", "Recall@20", "NDCG@10", "Recall@100", "Ret P95 ms"))
    print("-" * 115)
    for name, m in exp2_results.items():
        print(
            fmt.format(
                name,
                f"{m['candidate_pool_mean']:.1f}",
                f"{m['candidate_union_recall']:.4f}",
                f"{m['recall@20']:.4f}",
                f"{m['ndcg@10']:.4f}",
                f"{m['recall@100']:.4f}",
                f"{m['retrieval_p95_ms']:.2f}",
            )
        )
    print("=" * 115)


if __name__ == "__main__":
    run_experiment_2()
