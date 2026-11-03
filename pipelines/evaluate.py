import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.core.config import settings
from src.core.logging import logger
from src.evaluation.metrics import (
    compute_catalog_coverage,
    compute_intra_list_diversity,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_novelty,
    compute_recall_at_k,
)
from src.models.baselines import (
    ContentRecommender,
    ItemCollaborativeFiltering,
    PopularityRecommender,
)
from src.models.ranker import LightGBMRanker
from src.models.two_tower import TwoTowerRecommender
from src.ranking.feature_engineering import RankingFeatureExtractor
from src.reranking.mmr import MMRReRanker
from src.retrieval.faiss_index import FaissCandidateIndex


def run_evaluation(
    data_dir: Path = Path("data/processed"),
    artifacts_dir: Path = Path("artifacts"),
    num_eval_users: int = 500,
):
    metrics_dir = artifacts_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not settings.force_cpu else "cpu")
    logger.info(f"=== Starting Comprehensive Multi-Stage Offline Evaluation on {device} ===")

    # 1. Load data & ground truth
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    train_ratings = pd.read_parquet(data_dir / "ratings_train.parquet")
    test_ratings = pd.read_parquet(data_dir / "ratings_test.parquet")

    with open(data_dir / "mappings.json") as f:
        mappings = json.load(f)
    with open(data_dir / "user_train_histories.json") as f:
        user_histories = {int(k): v for k, v in json.load(f).items()}

    num_users = mappings["num_users"]
    num_movies = mappings["num_movies"]
    genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]

    # Group test ratings by user: ground truth relevant set (rating >= 3.0)
    test_ground_truth: dict[int, set[int]] = {}
    for _, row in test_ratings.iterrows():
        u = int(row["user_idx"])
        m = int(row["movie_idx"])
        r = float(row["rating"])
        if r >= 3.0:
            if u not in test_ground_truth:
                test_ground_truth[u] = set()
            test_ground_truth[u].add(m)

    eval_users = [u for u in test_ground_truth if u in user_histories][:num_eval_users]
    logger.info(f"Evaluating across {len(eval_users)} test users with verified positive interactions...")

    # 2. Fit Baselines
    logger.info("Fitting baseline recommenders...")
    pop_model = PopularityRecommender()
    pop_model.fit(train_ratings, num_movies=num_movies)

    item_cf = ItemCollaborativeFiltering()
    item_cf.fit(train_ratings, num_users=num_users, num_movies=num_movies)

    content_model = ContentRecommender()
    content_model.fit(movies_df, genre_cols=genre_cols)

    # 3. Load Two-Tower Model & FAISS Index
    logger.info("Loading Two-Tower model and FAISS ANN index...")
    two_tower_path = artifacts_dir / "models" / "two_tower_best.pt"
    tt_model = TwoTowerRecommender(
        num_users=num_users,
        num_movies=num_movies,
        num_genres=len(genre_cols),
        embedding_dim=settings.embedding_dim,
    ).to(device)

    if two_tower_path.exists():
        ckpt = torch.load(two_tower_path, map_location=device, weights_only=True)
        tt_model.load_state_dict(ckpt["model_state_dict"])
        tt_model.eval()

    faiss_index = FaissCandidateIndex(dimension=settings.embedding_dim)
    faiss_index.load(artifacts_dir / "indices")

    # 4. Load LightGBM Ranker
    ranker = LightGBMRanker()
    ranker_path = artifacts_dir / "models" / "ranker_lgbm.txt"
    if ranker_path.exists():
        ranker.load(ranker_path)

    feature_extractor = RankingFeatureExtractor(
        movies_df=movies_df,
        genre_cols=genre_cols,
        global_mean_rating=mappings.get("global_rating_mean", 3.5),
    )
    reranker = MMRReRanker(
        movies_df=movies_df,
        genre_cols=genre_cols,
        default_lambda=0.65,
        default_novelty_weight=0.15,
    )

    # Precompute genre matrix for ILD
    item_genres = movies_df[genre_cols].values.astype(np.float32)
    item_counts = movies_df["rating_count"].values.astype(np.float32)
    total_ratings_count = len(train_ratings)

    # Models to benchmark
    model_names = [
        "Popularity Baseline",
        "Item-based CF",
        "Content-based",
        "Two-Tower (ANN Retrieval)",
        "Two-Tower + LightGBM Ranker",
        "Full Pipeline (Retrieval + Ranker + MMR)",
    ]

    results_by_model: dict[str, dict] = {name: {} for name in model_names}

    # Evaluate each approach
    for model_name in model_names:
        logger.info(f"Evaluating {model_name}...")
        recalls_20, recalls_100 = [], []
        ndcgs_10 = []
        mrrs_10 = []
        all_recs_20 = []
        all_recs_100 = []
        ilds = []
        novelties = []
        latencies_ms = []

        for u in eval_users:
            hist = user_histories.get(u, [])
            consumed = set(hist)
            gt = test_ground_truth[u]

            t0 = time.perf_counter()

            if model_name == "Popularity Baseline":
                recs = [idx for idx, _ in pop_model.recommend(hist, top_k=100)]
            elif model_name == "Item-based CF":
                recs = [idx for idx, _ in item_cf.recommend(hist[-10:], top_k=100)]
                if len(recs) < 100:
                    pop_fill = [idx for idx, _ in pop_model.recommend(hist, top_k=100) if idx not in set(recs)]
                    recs.extend(pop_fill[:100 - len(recs)])
            elif model_name == "Content-based":
                recs = [idx for idx, _ in content_model.recommend(hist[-10:], top_k=100)]
            elif model_name == "Two-Tower (ANN Retrieval)":
                # User embedding query
                with torch.no_grad():
                    padded_hist = np.zeros(settings.max_sequence_length, dtype=np.int64)
                    h_len = min(len(hist), settings.max_sequence_length)
                    if h_len > 0:
                        padded_hist[-h_len:] = hist[-h_len:]
                    g_pref = item_genres[hist].mean(axis=0) if hist else np.zeros(len(genre_cols), dtype=np.float32)

                    u_tensor = torch.tensor([u], dtype=torch.long, device=device)
                    h_tensor = torch.tensor(padded_hist, dtype=torch.long, device=device).unsqueeze(0)
                    hl_tensor = torch.tensor([h_len], dtype=torch.long, device=device)
                    gp_tensor = torch.tensor(g_pref, dtype=torch.float32, device=device).unsqueeze(0)

                    u_emb = tt_model.user_tower(u_tensor, h_tensor, hl_tensor, gp_tensor).cpu().numpy()[0]
                recs = [idx for idx, _ in faiss_index.search(u_emb, top_k=100, exclude_item_indices=consumed)]
            elif model_name == "Two-Tower + LightGBM Ranker":
                with torch.no_grad():
                    padded_hist = np.zeros(settings.max_sequence_length, dtype=np.int64)
                    h_len = min(len(hist), settings.max_sequence_length)
                    if h_len > 0:
                        padded_hist[-h_len:] = hist[-h_len:]
                    g_pref = item_genres[hist].mean(axis=0) if hist else np.zeros(len(genre_cols), dtype=np.float32)

                    u_tensor = torch.tensor([u], dtype=torch.long, device=device)
                    h_tensor = torch.tensor(padded_hist, dtype=torch.long, device=device).unsqueeze(0)
                    hl_tensor = torch.tensor([h_len], dtype=torch.long, device=device)
                    gp_tensor = torch.tensor(g_pref, dtype=torch.float32, device=device).unsqueeze(0)

                    u_emb = tt_model.user_tower(u_tensor, h_tensor, hl_tensor, gp_tensor).cpu().numpy()[0]
                candidates = faiss_index.search(u_emb, top_k=100, exclude_item_indices=consumed)
                c_indices = [idx for idx, _ in candidates]
                tt_scores = {idx: s for idx, s in candidates}
                sources = {idx: "two_tower" for idx in c_indices}

                feat_df = feature_extractor.extract_features(
                    user_idx=u,
                    user_history=hist,
                    candidate_items=c_indices,
                    two_tower_scores=tt_scores,
                    cf_scores={},
                    retrieval_sources=sources,
                )
                if ranker.is_fitted and not feat_df.empty:
                    ranked = ranker.rank(feat_df)
                    recs = [idx for idx, _ in ranked]
                else:
                    recs = c_indices
            elif model_name == "Full Pipeline (Retrieval + Ranker + MMR)":
                with torch.no_grad():
                    padded_hist = np.zeros(settings.max_sequence_length, dtype=np.int64)
                    h_len = min(len(hist), settings.max_sequence_length)
                    if h_len > 0:
                        padded_hist[-h_len:] = hist[-h_len:]
                    g_pref = item_genres[hist].mean(axis=0) if hist else np.zeros(len(genre_cols), dtype=np.float32)

                    u_tensor = torch.tensor([u], dtype=torch.long, device=device)
                    h_tensor = torch.tensor(padded_hist, dtype=torch.long, device=device).unsqueeze(0)
                    hl_tensor = torch.tensor([h_len], dtype=torch.long, device=device)
                    gp_tensor = torch.tensor(g_pref, dtype=torch.float32, device=device).unsqueeze(0)

                    u_emb = tt_model.user_tower(u_tensor, h_tensor, hl_tensor, gp_tensor).cpu().numpy()[0]
                candidates = faiss_index.search(u_emb, top_k=100, exclude_item_indices=consumed)
                c_indices = [idx for idx, _ in candidates]
                tt_scores = {idx: s for idx, s in candidates}
                sources = {idx: "two_tower" for idx in c_indices}

                feat_df = feature_extractor.extract_features(
                    user_idx=u,
                    user_history=hist,
                    candidate_items=c_indices,
                    two_tower_scores=tt_scores,
                    cf_scores={},
                    retrieval_sources=sources,
                )
                ranked = ranker.rank(feat_df) if ranker.is_fitted and not feat_df.empty else candidates
                reranked = reranker.rerank(ranked, top_k=100, lambda_param=0.65, novelty_weight=0.15)
                recs = [item["movie_idx"] for item in reranked]

            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            recalls_20.append(compute_recall_at_k(recs, gt, k=20))
            recalls_100.append(compute_recall_at_k(recs, gt, k=100))
            ndcgs_10.append(compute_ndcg_at_k(recs, gt, k=10))
            mrrs_10.append(compute_mrr_at_k(recs, gt, k=10))
            all_recs_20.append(recs[:20])
            all_recs_100.append(recs[:100])
            ilds.append(compute_intra_list_diversity(recs[:20], item_genres))
            novelties.append(compute_novelty(recs[:20], item_counts, total_ratings_count))

        coverage_20 = compute_catalog_coverage(all_recs_20, num_movies)
        coverage_100 = compute_catalog_coverage(all_recs_100, num_movies)

        results_by_model[model_name] = {
            "recall@20": round(float(np.mean(recalls_20)), 4),
            "recall@100": round(float(np.mean(recalls_100)), 4),
            "ndcg@10": round(float(np.mean(ndcgs_10)), 4),
            "mrr@10": round(float(np.mean(mrrs_10)), 4),
            "catalog_coverage@20": round(float(coverage_20), 4),
            "catalog_coverage@100": round(float(coverage_100), 4),
            "intra_list_diversity": round(float(np.mean(ilds)), 4),
            "novelty_bits": round(float(np.mean(novelties)), 2),
            "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        }

    # Save detailed JSON metrics
    metrics_path = metrics_dir / "evaluation_results.json"
    with open(metrics_path, "w") as f:
        json.dump(results_by_model, f, indent=2)

    logger.info(f"=== Multi-Stage Offline Evaluation Results Saved to {metrics_path} ===")
    df_results = pd.DataFrame(results_by_model).T
    print("\n" + df_results.to_string() + "\n")
    return results_by_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Recommendation Models Offline")
    parser.add_argument("--num-users", type=int, default=500)
    args = parser.parse_args()
    run_evaluation(num_eval_users=args.num_users)
