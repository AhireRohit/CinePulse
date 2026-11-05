"""
CinePulse Relevance-v2 Tournament Evaluation Module.
Standardized, reproducible, high-throughput offline evaluation harness
for all 6 experiments on the validation cohort.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.core.config import settings
from src.core.logging import logger
from src.evaluation.metrics import (
    compute_bootstrap_ci,
    compute_catalog_coverage,
    compute_graded_ndcg_at_k,
    compute_intra_list_diversity,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_novelty,
    compute_recall_at_k,
)


class TournamentEvaluator:
    """
    Standardized evaluator for the relevance tournament.
    Evaluates candidates and ranked lists against the frozen validation cohort.
    """

    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        artifacts_dir: Path = Path("artifacts"),
        device: torch.device | None = None,
    ):
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() and not settings.force_cpu else "cpu"
        )

        logger.info(f"Initializing TournamentEvaluator on device: {self.device}")

        # 1. Load data
        self.movies_df = pd.read_parquet(data_dir / "movies.parquet")
        self.train_ratings = pd.read_parquet(data_dir / "ratings_train.parquet")
        self.val_ratings = pd.read_parquet(data_dir / "ratings_val.parquet")

        with open(data_dir / "mappings.json") as f:
            self.mappings = json.load(f)
        with open(data_dir / "user_train_histories.json") as f:
            self.user_histories = {int(k): v for k, v in json.load(f).items()}

        self.num_users = self.mappings["num_users"]
        self.num_movies = self.mappings["num_movies"]
        self.genre_cols = [c for c in self.movies_df.columns if c.startswith("genre_")]
        self.item_genres = self.movies_df[self.genre_cols].values.astype(np.float32)
        self.item_counts = self.movies_df["rating_count"].values.astype(np.float32)
        self.total_ratings_count = len(self.train_ratings)

        # 2. Build ground truth sets for validation
        self.ground_truth_std: dict[int, set[int]] = {}
        self.ground_truth_pos: dict[int, set[int]] = {}
        self.ground_truth_ratings: dict[int, dict[int, float]] = {}

        for _, row in self.val_ratings.iterrows():
            u = int(row["user_idx"])
            m = int(row["movie_idx"])
            r = float(row["rating"])

            if u not in self.ground_truth_ratings:
                self.ground_truth_ratings[u] = {}
            self.ground_truth_ratings[u][m] = r

            if r >= 3.0:
                self.ground_truth_std.setdefault(u, set()).add(m)
            if r >= 4.0:
                self.ground_truth_pos.setdefault(u, set()).add(m)

        # Eligible users: have training history and validation interaction
        self.all_users = sorted(list(self.ground_truth_ratings.keys()))
        self.std_users = [u for u in self.all_users if u in self.ground_truth_std and u in self.user_histories]
        self.pos_users = [u for u in self.all_users if u in self.ground_truth_pos and u in self.user_histories]

        logger.info(
            f"Evaluator ready: {len(self.all_users)} total val users, "
            f"{len(self.std_users)} standard (rating >= 3.0), {len(self.pos_users)} positive-only (rating >= 4.0)."
        )

    def evaluate_recommendations(
        self,
        recommendations: dict[int, list[int]],
        experiment_name: str,
        latencies_ms: list[float] | None = None,
        train_time_sec: float = 0.0,
    ) -> dict:
        """
        Computes all standard 15 tournament metrics from a precomputed recommendations dictionary
        {user_idx: list_of_movie_indices}.
        """
        rec_10, rec_20, rec_50, rec_100 = [], [], [], []
        rec_20_pos = []
        ndcg_5, ndcg_10, ndcg_20 = [], [], []
        ndcg_10_pos = []
        mrr_10 = []
        graded_ndcg_10 = []
        all_recs_20 = []
        all_recs_100 = []
        ilds = []
        novelties = []

        per_user_ndcg10: dict[int, float] = {}
        per_user_rec20: dict[int, float] = {}
        per_user_pos_ndcg10: dict[int, float] = {}
        per_user_pos_rec20: dict[int, float] = {}

        # 1. Standard benchmark evaluation (std_users)
        for u in self.std_users:
            recs = recommendations.get(u, [])
            gt_std = self.ground_truth_std[u]

            r10 = compute_recall_at_k(recs, gt_std, k=10)
            r20 = compute_recall_at_k(recs, gt_std, k=20)
            r50 = compute_recall_at_k(recs, gt_std, k=50)
            r100 = compute_recall_at_k(recs, gt_std, k=100)

            n5 = compute_ndcg_at_k(recs, gt_std, k=5)
            n10 = compute_ndcg_at_k(recs, gt_std, k=10)
            n20 = compute_ndcg_at_k(recs, gt_std, k=20)
            m10 = compute_mrr_at_k(recs, gt_std, k=10)

            rec_10.append(r10)
            rec_20.append(r20)
            rec_50.append(r50)
            rec_100.append(r100)
            ndcg_5.append(n5)
            ndcg_10.append(n10)
            ndcg_20.append(n20)
            mrr_10.append(m10)

            per_user_ndcg10[u] = n10
            per_user_rec20[u] = r20

            # List characteristics
            top20 = recs[:20]
            top100 = recs[:100]
            all_recs_20.append(top20)
            all_recs_100.append(top100)

            if len(top20) > 1:
                ilds.append(compute_intra_list_diversity(top20, self.item_genres))
            novelties.append(compute_novelty(top20, self.item_counts, self.total_ratings_count))

        # 2. Strict Positive-Only evaluation (pos_users: rating >= 4.0)
        for u in self.pos_users:
            recs = recommendations.get(u, [])
            gt_pos = self.ground_truth_pos[u]

            r20_pos = compute_recall_at_k(recs, gt_pos, k=20)
            n10_pos = compute_ndcg_at_k(recs, gt_pos, k=10)

            rec_20_pos.append(r20_pos)
            ndcg_10_pos.append(n10_pos)

            per_user_pos_rec20[u] = r20_pos
            per_user_pos_ndcg10[u] = n10_pos

        # 3. Graded NDCG across all users
        for u in self.all_users:
            recs = recommendations.get(u, [])
            gt_ratings = self.ground_truth_ratings.get(u, {})
            graded_ndcg_10.append(compute_graded_ndcg_at_k(recs, gt_ratings, k=10))

        # Catalog coverage
        cov_20 = compute_catalog_coverage(all_recs_20, self.num_movies)
        cov_100 = compute_catalog_coverage(all_recs_100, self.num_movies)

        # Bootstrap CIs for key metrics
        ndcg10_mean, ndcg10_ci_l, ndcg10_ci_u = compute_bootstrap_ci(ndcg_10)
        rec20_mean, rec20_ci_l, rec20_ci_u = compute_bootstrap_ci(rec_20)
        pos_ndcg10_mean, pos_ndcg10_ci_l, pos_ndcg10_ci_u = compute_bootstrap_ci(ndcg_10_pos)
        pos_rec20_mean, pos_rec20_ci_l, pos_rec20_ci_u = compute_bootstrap_ci(rec_20_pos)

        p50_lat = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
        p95_lat = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

        metrics_result = {
            "experiment_name": experiment_name,
            "cohort": "validation_all_eligible",
            "eval_users_std": len(self.std_users),
            "eval_users_pos": len(self.pos_users),
            "eval_users_all": len(self.all_users),
            "recall@10": round(float(np.mean(rec_10)), 4),
            "recall@20": round(float(np.mean(rec_20)), 4),
            "recall@20_ci95": [rec20_ci_l, rec20_ci_u],
            "recall@50": round(float(np.mean(rec_50)), 4),
            "recall@100": round(float(np.mean(rec_100)), 4),
            "ndcg@5": round(float(np.mean(ndcg_5)), 4),
            "ndcg@10": round(float(np.mean(ndcg_10)), 4),
            "ndcg@10_ci95": [ndcg10_ci_l, ndcg10_ci_u],
            "ndcg@20": round(float(np.mean(ndcg_20)), 4),
            "mrr@10": round(float(np.mean(mrr_10)), 4),
            "positive_only_recall@20": round(float(np.mean(rec_20_pos)), 4),
            "positive_only_recall@20_ci95": [pos_rec20_ci_l, pos_rec20_ci_u],
            "positive_only_ndcg@10": round(float(np.mean(ndcg_10_pos)), 4),
            "positive_only_ndcg@10_ci95": [pos_ndcg10_ci_l, pos_ndcg10_ci_u],
            "graded_ndcg@10": round(float(np.mean(graded_ndcg_10)), 4),
            "catalog_coverage@20": round(float(cov_20), 4),
            "catalog_coverage@100": round(float(cov_100), 4),
            "intra_list_diversity": round(float(np.mean(ilds)), 4),
            "novelty_bits": round(float(np.mean(novelties)), 2),
            "latency_p50_ms": round(p50_lat, 2),
            "latency_p95_ms": round(p95_lat, 2),
            "train_or_opt_runtime_sec": round(float(train_time_sec), 2),
            "protocol_note": (
                "Single-held-out validation item per user. In this protocol, Recall@K is mathematically "
                "equivalent to Hit Rate@K. It measures whether the specific held-out interaction is in top-K, "
                "not recall over all movies the user would like."
            ),
        }

        # Save per-user arrays for paired significance tests
        per_user_data = {
            "ndcg@10": per_user_ndcg10,
            "recall@20": per_user_rec20,
            "positive_only_ndcg@10": per_user_pos_ndcg10,
            "positive_only_recall@20": per_user_pos_rec20,
        }

        return metrics_result, per_user_data
