"""
Phase 0: Baseline Evaluation on Frozen Validation Cohort.
Computes and records the exact reproducible baseline metrics for:
- Existing Full Pipeline (Retrieval + LightGBM Ranker + MMR)
- Existing Pure LightGBM Ranker (without MMR)
- Standalone Two-Tower
- Standalone Item-CF
- Popularity Baseline
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from pipelines.tournament_eval import TournamentEvaluator
from src.core.config import settings
from src.core.logging import logger
from src.models.baselines import ItemCollaborativeFiltering, PopularityRecommender
from src.models.ranker import LightGBMRanker
from src.models.two_tower import TwoTowerRecommender
from src.ranking.feature_engineering import RankingFeatureExtractor
from src.reranking.mmr import MMRReRanker
from src.retrieval.faiss_index import FaissCandidateIndex


def main():
    logger.info("=== Phase 0: Reproducing Baseline on Validation Cohort ===")
    evaluator = TournamentEvaluator()
    device = evaluator.device

    # 1. Fit baselines
    logger.info("Fitting baselines...")
    pop_model = PopularityRecommender()
    pop_model.fit(evaluator.train_ratings, num_movies=evaluator.num_movies)

    item_cf = ItemCollaborativeFiltering()
    item_cf.fit(evaluator.train_ratings, num_users=evaluator.num_users, num_movies=evaluator.num_movies)

    # 2. Load Two-Tower & FAISS
    logger.info("Loading Two-Tower & FAISS...")
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

    # 3. Load LightGBM Ranker & Reranker
    ranker = LightGBMRanker()
    ranker.load(Path("artifacts/models/ranker_lgbm.txt"))

    feature_extractor = RankingFeatureExtractor(
        movies_df=evaluator.movies_df,
        genre_cols=evaluator.genre_cols,
        global_mean_rating=evaluator.mappings.get("global_rating_mean", 3.5),
    )
    reranker = MMRReRanker(
        movies_df=evaluator.movies_df,
        genre_cols=evaluator.genre_cols,
        default_lambda=0.65,
        default_novelty_weight=0.15,
    )

    eval_users = evaluator.std_users  # All 4,980 eligible users

    # A. Standalone Popularity
    logger.info("Evaluating Popularity Baseline...")
    pop_recs = {}
    pop_lats = []
    for u in eval_users:
        h = evaluator.user_histories.get(u, [])
        t_start = time.perf_counter()
        recs = [idx for idx, _ in pop_model.recommend(h, top_k=100)]
        pop_lats.append((time.perf_counter() - t_start) * 1000.0)
        pop_recs[u] = recs
    pop_metrics, pop_user_data = evaluator.evaluate_recommendations(
        pop_recs, "Popularity Baseline", latencies_ms=pop_lats, train_time_sec=0.0
    )

    # B. Standalone Item-CF
    logger.info("Evaluating Item-Based CF...")
    cf_recs = {}
    cf_lats = []
    for u in eval_users:
        h = evaluator.user_histories.get(u, [])
        t_start = time.perf_counter()
        recs = [idx for idx, _ in item_cf.recommend(h, top_k=100)]
        if len(recs) < 100:
            pop_fill = [idx for idx, _ in pop_model.recommend(h, top_k=100) if idx not in set(recs)]
            recs.extend(pop_fill[: 100 - len(recs)])
        cf_lats.append((time.perf_counter() - t_start) * 1000.0)
        cf_recs[u] = recs
    cf_metrics, cf_user_data = evaluator.evaluate_recommendations(
        cf_recs, "Item-based CF", latencies_ms=cf_lats, train_time_sec=0.0
    )

    # C. Standalone Two-Tower
    logger.info("Evaluating Two-Tower (ANN Retrieval)...")
    tt_recs = {}
    tt_lats = []
    # Pre-generate embeddings in batches for high speed
    batch_size = 1024
    user_embs_dict = {}
    with torch.no_grad():
        for i in range(0, len(eval_users), batch_size):
            b_users = eval_users[i : i + batch_size]
            b_len = len(b_users)
            u_arr = np.array(b_users, dtype=np.int64)
            padded_hist = np.full((b_len, settings.max_sequence_length), evaluator.num_movies, dtype=np.int64)
            h_lens = np.zeros(b_len, dtype=np.int64)
            g_prefs = np.zeros((b_len, len(evaluator.genre_cols)), dtype=np.float32)

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

    for u in eval_users:
        h = evaluator.user_histories.get(u, [])
        consumed = set(h)
        u_emb = user_embs_dict[u]
        t_start = time.perf_counter()
        recs = [idx for idx, _ in faiss_index.search(u_emb, top_k=100, exclude_item_indices=consumed)]
        tt_lats.append((time.perf_counter() - t_start) * 1000.0)
        tt_recs[u] = recs
    tt_metrics, tt_user_data = evaluator.evaluate_recommendations(
        tt_recs, "Two-Tower (ANN Retrieval)", latencies_ms=tt_lats, train_time_sec=0.0
    )

    # D. Existing Pure LightGBM Ranker (without MMR)
    # E. Existing Full Pipeline (with MMR)
    logger.info("Evaluating Pure LightGBM Ranker and Full Pipeline + MMR...")
    pure_recs = {}
    pure_lats = []
    mmr_recs = {}
    mmr_lats = []

    for u in eval_users:
        h = evaluator.user_histories.get(u, [])
        consumed = set(h)
        u_emb = user_embs_dict[u]

        t_start = time.perf_counter()
        candidate_sources: dict[int, set[str]] = {}
        cf_score_map: dict[int, float] = {}

        # Two-Tower top-100
        for item_idx, score in faiss_index.search(u_emb, top_k=100, exclude_item_indices=consumed):
            candidate_sources.setdefault(item_idx, set()).add("two_tower")

        # Item-CF top-100
        for item_idx, score in item_cf.recommend(h, top_k=100):
            if item_idx not in consumed:
                candidate_sources.setdefault(item_idx, set()).add("item_cf")
                cf_score_map[item_idx] = float(score)

        # Popularity top-30
        for item_idx, score in pop_model.recommend(h, top_k=30):
            if item_idx not in consumed:
                candidate_sources.setdefault(item_idx, set()).add("popularity")

        c_indices = list(candidate_sources.keys())
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

        feat_df = feature_extractor.extract_features(
            user_idx=u,
            user_history=h,
            candidate_items=c_indices,
            two_tower_scores=tt_scores,
            cf_scores=cf_scores_dict,
            retrieval_sources=sources,
        )
        ranked = ranker.rank(feat_df)
        recs_pure = [idx for idx, _ in ranked]
        t_pure_end = time.perf_counter()
        pure_lats.append((t_pure_end - t_start) * 1000.0)
        pure_recs[u] = recs_pure

        # MMR Reranking
        reranked = reranker.rerank(ranked, top_k=100, lambda_param=0.65, novelty_weight=0.15)
        recs_mmr = [item["movie_idx"] for item in reranked]
        t_mmr_end = time.perf_counter()
        mmr_lats.append((t_mmr_end - t_start) * 1000.0)
        mmr_recs[u] = recs_mmr

    pure_metrics, pure_user_data = evaluator.evaluate_recommendations(
        pure_recs, "Existing Pure Ranker", latencies_ms=pure_lats, train_time_sec=0.0
    )
    mmr_metrics, mmr_user_data = evaluator.evaluate_recommendations(
        mmr_recs, "Existing Full Pipeline", latencies_ms=mmr_lats, train_time_sec=0.0
    )

    # Save all baseline metrics
    registry_dir = Path("artifacts/experiments/relevance_v2")
    registry_dir.mkdir(parents=True, exist_ok=True)

    baselines = {
        "Popularity Baseline": pop_metrics,
        "Item-based CF": cf_metrics,
        "Two-Tower (ANN Retrieval)": tt_metrics,
        "Existing Pure Ranker": pure_metrics,
        "Existing Full Pipeline": mmr_metrics,
    }

    with open(registry_dir / "baseline_results_val.json", "w") as f:
        json.dump(baselines, f, indent=2)

    # Save per-user results for paired tests
    per_user_all = {
        "Popularity Baseline": pop_user_data,
        "Item-based CF": cf_user_data,
        "Two-Tower (ANN Retrieval)": tt_user_data,
        "Existing Pure Ranker": pure_user_data,
        "Existing Full Pipeline": mmr_user_data,
    }
    with open(registry_dir / "baseline_per_user_val.json", "w") as f:
        json.dump(per_user_all, f)

    logger.info("=== Phase 0 Baseline Validation Complete ===")
    print("\n" + "=" * 90)
    print("PHASE 0 — BASELINE VALIDATION RESULTS (All 4,980 Eligible Validation Users)")
    print("=" * 90)
    fmt = "{:<32} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10}"
    print(fmt.format("Model", "Recall@20", "NDCG@10", "Recall@100", "Pos NDCG@10", "ILD", "P95 ms"))
    print("-" * 90)
    for name, m in baselines.items():
        print(
            fmt.format(
                name,
                f"{m['recall@20']:.4f}",
                f"{m['ndcg@10']:.4f}",
                f"{m['recall@100']:.4f}",
                f"{m['positive_only_ndcg@10']:.4f}",
                f"{m['intra_list_diversity']:.4f}",
                f"{m['latency_p95_ms']:.1f}",
            )
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
