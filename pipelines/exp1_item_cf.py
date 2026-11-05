"""
Experiment 1: Stronger Item-Based Collaborative Filtering.
Controlled investigation of:
1. Similarity shrinkage parameter lambda (0, 10, 25, 50)
2. Neighbor count K (50, 100, 200)
3. Rating-aware user history weighting (unweighted vs rating-centered vs positive-only vs dislike suppression)
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from pipelines.tournament_eval import TournamentEvaluator
from src.core.logging import logger
from src.evaluation.metrics import compute_paired_bootstrap_ci


class EnhancedItemCF:
    """
    Enhanced Item-Based Collaborative Filtering with:
    - Centered cosine similarity
    - Support-based similarity shrinkage: S_ij * N_ij / (N_ij + lambda_shrink)
    - Configurable top-K neighbor graph (50, 100, 200)
    - Rating-aware user profile weighting & dislike suppression
    """

    def __init__(self, top_k_neighbors: int = 100, shrinkage_lambda: float = 25.0):
        self.top_k_neighbors = top_k_neighbors
        self.shrinkage_lambda = shrinkage_lambda
        self.item_similarity: dict[int, list[tuple[int, float]]] = {}
        self.num_movies = 0

    def fit(self, ratings_df: pd.DataFrame, num_users: int, num_movies: int):
        self.num_movies = num_movies
        ratings = ratings_df.copy()

        # Compute user means for centering
        user_means = ratings.groupby("user_idx")["rating"].transform("mean")
        ratings["centered_rating"] = ratings["rating"] - user_means

        rows = ratings["movie_idx"].values
        cols = ratings["user_idx"].values
        data = ratings["centered_rating"].values

        # Item-user centered matrix [num_movies, num_users]
        item_user = csr_matrix((data, (rows, cols)), shape=(num_movies, num_users))
        # Binary support matrix to compute co-occurrence counts N_ij
        binary_item_user = csr_matrix((np.ones_like(rows, dtype=np.float32), (rows, cols)), shape=(num_movies, num_users))

        # Vectorized raw cosine similarity and co-occurrence counts
        sim_matrix = cosine_similarity(item_user)  # [num_movies, num_movies] dense array
        co_counts = (binary_item_user @ binary_item_user.T).toarray()  # [num_movies, num_movies] dense array

        # Zero out self similarity
        np.fill_diagonal(sim_matrix, 0.0)

        # Apply shrinkage if requested: s' = s * (n / (n + lambda))
        if self.shrinkage_lambda > 0.0:
            shrink_factors = co_counts / (co_counts + self.shrinkage_lambda)
            sim_matrix = sim_matrix * shrink_factors

        # Keep only positive similarities to avoid recommending opposites
        sim_matrix[sim_matrix < 0.0] = 0.0

        # Extract top-K neighbors per item
        self.item_similarity = {}
        for i in range(num_movies):
            row = sim_matrix[i]
            k_eff = min(self.top_k_neighbors, num_movies - 1)
            part = np.argpartition(-row, k_eff)[:k_eff]
            part = part[row[part] > 0.0]
            if len(part) == 0:
                self.item_similarity[i] = []
                continue
            sort_order = np.argsort(-row[part])
            sorted_indices = part[sort_order]
            self.item_similarity[i] = list(
                zip(sorted_indices.tolist(), row[sorted_indices].tolist())
            )

    def recommend(
        self,
        user_history: list[int],
        user_ratings: dict[int, float] | None = None,
        weighting_mode: str = "rating_centered",  # 'uniform', 'rating_centered', 'positive_only', 'dislike_suppression'
        top_k: int = 100,
    ) -> list[tuple[int, float]]:
        if not user_history:
            return []

        consumed = set(user_history)
        candidate_scores: dict[int, float] = {}
        history_len = len(user_history)

        for pos, item_idx in enumerate(user_history):
            recency = 0.5 + 0.5 * ((pos + 1) / history_len)
            r = user_ratings.get(item_idx, 3.5) if user_ratings else 3.5

            if weighting_mode == "uniform":
                item_w = 1.0 * recency
            elif weighting_mode == "positive_only":
                # Only items rated >= 3.5 contribute positively; <= 3.0 contribute 0
                item_w = max(0.0, r - 3.0) * recency
            elif weighting_mode == "dislike_suppression":
                # Items rated 4-5 have positive weight, 3 is small, 1-2 are zeroed out
                if r >= 4.0:
                    item_w = (r - 2.5) * recency
                elif r >= 3.0:
                    item_w = 0.5 * recency
                else:
                    item_w = 0.0  # Completely suppress disliked seeds
            elif weighting_mode == "rating_centered":
                # Centered around 3.0
                item_w = (r - 3.0) * recency
            else:
                item_w = 1.0 * recency

            if item_w == 0.0:
                continue

            for neighbor_idx, sim in self.item_similarity.get(item_idx, []):
                if neighbor_idx not in consumed:
                    candidate_scores[neighbor_idx] = candidate_scores.get(neighbor_idx, 0.0) + (sim * item_w)

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def run_experiment_1():
    logger.info("=== Running Experiment 1: Item-Based Collaborative Filtering Optimization ===")
    evaluator = TournamentEvaluator()
    eval_users = evaluator.std_users

    # Build user training rating dictionary {user_idx: {movie_idx: rating}}
    logger.info("Building user training rating maps for rating-aware weighting...")
    user_train_ratings_map: dict[int, dict[int, float]] = {}
    for _, row in evaluator.train_ratings.iterrows():
        u = int(row["user_idx"])
        m = int(row["movie_idx"])
        r = float(row["rating"])
        if u not in user_train_ratings_map:
            user_train_ratings_map[u] = {}
        user_train_ratings_map[u][m] = r

    # Popularity model for fallback fill
    from src.models.baselines import PopularityRecommender

    pop_model = PopularityRecommender()
    pop_model.fit(evaluator.train_ratings, num_movies=evaluator.num_movies)

    # Load baseline CF metrics for comparison
    with open("artifacts/experiments/relevance_v2/baseline_results_val.json") as f:
        baseline_results = json.load(f)
    baseline_cf = baseline_results["Item-based CF"]

    # Controlled Grid of CF Variants
    variants = [
        {"name": "CF-V1 (K=100, Shrink=0, Uniform)", "k": 100, "shrink": 0.0, "mode": "uniform"},
        {"name": "CF-V2 (K=100, Shrink=10, Uniform)", "k": 100, "shrink": 10.0, "mode": "uniform"},
        {"name": "CF-V3 (K=100, Shrink=25, Uniform)", "k": 100, "shrink": 25.0, "mode": "uniform"},
        {"name": "CF-V4 (K=100, Shrink=50, Uniform)", "k": 100, "shrink": 50.0, "mode": "uniform"},
        {"name": "CF-V5 (K=50, Shrink=25, Uniform)", "k": 50, "shrink": 25.0, "mode": "uniform"},
        {"name": "CF-V6 (K=200, Shrink=25, Uniform)", "k": 200, "shrink": 25.0, "mode": "uniform"},
        {"name": "CF-V7 (K=100, Shrink=25, Positive-Only)", "k": 100, "shrink": 25.0, "mode": "positive_only"},
        {"name": "CF-V8 (K=100, Shrink=25, Dislike-Suppression)", "k": 100, "shrink": 25.0, "mode": "dislike_suppression"},
        {"name": "CF-V9 (K=100, Shrink=25, Rating-Centered)", "k": 100, "shrink": 25.0, "mode": "rating_centered"},
    ]

    exp1_results = {}
    exp1_per_user = {}

    for var in variants:
        v_name = var["name"]
        k = var["k"]
        shrink = var["shrink"]
        mode = var["mode"]
        logger.info(f"Fitting & evaluating variant: {v_name}...")

        t_start_train = time.time()
        model = EnhancedItemCF(top_k_neighbors=k, shrinkage_lambda=shrink)
        model.fit(evaluator.train_ratings, num_users=evaluator.num_users, num_movies=evaluator.num_movies)
        train_duration = time.time() - t_start_train

        recs_dict = {}
        lats = []
        for u in eval_users:
            h = evaluator.user_histories.get(u, [])
            u_ratings = user_train_ratings_map.get(u, {})
            t0 = time.perf_counter()
            r_list = [idx for idx, _ in model.recommend(h, user_ratings=u_ratings, weighting_mode=mode, top_k=100)]
            if len(r_list) < 100:
                pop_fill = [idx for idx, _ in pop_model.recommend(h, top_k=100) if idx not in set(r_list)]
                r_list.extend(pop_fill[: 100 - len(r_list)])
            lats.append((time.perf_counter() - t0) * 1000.0)
            recs_dict[u] = r_list

        metrics, per_user = evaluator.evaluate_recommendations(
            recs_dict, v_name, latencies_ms=lats, train_time_sec=train_duration
        )
        exp1_results[v_name] = metrics
        exp1_per_user[v_name] = per_user

        logger.info(
            f"-> {v_name}: Recall@20 = {metrics['recall@20']:.4f}, NDCG@10 = {metrics['ndcg@10']:.4f}, "
            f"Pos NDCG@10 = {metrics['positive_only_ndcg@10']:.4f}, ILD = {metrics['intra_list_diversity']:.4f}"
        )

    # Save results
    output_dir = Path("artifacts/experiments/relevance_v2")
    with open(output_dir / "exp1_item_cf_results.json", "w") as f:
        json.dump(exp1_results, f, indent=2)

    # Find the best variant by NDCG@10
    best_variant_name = max(exp1_results.keys(), key=lambda k: exp1_results[k]["ndcg@10"])
    best_m = exp1_results[best_variant_name]

    # Paired significance test against baseline CF
    with open(output_dir / "baseline_per_user_val.json") as f:
        baseline_user_data = json.load(f)["Item-based CF"]

    eval_u_list = evaluator.std_users
    base_ndcgs = [baseline_user_data["ndcg@10"][str(u)] for u in eval_u_list]
    best_ndcgs = [exp1_per_user[best_variant_name]["ndcg@10"][u] for u in eval_u_list]

    diff_mean, ci_l, ci_u, p_val = compute_paired_bootstrap_ci(best_ndcgs, base_ndcgs)
    significance_report = {
        "best_variant": best_variant_name,
        "delta_ndcg@10_vs_baseline_cf": round(diff_mean, 4),
        "ci95_ndcg@10": [round(ci_l, 4), round(ci_u, 4)],
        "p_value": p_val,
        "baseline_ndcg@10": baseline_cf["ndcg@10"],
        "best_ndcg@10": best_m["ndcg@10"],
        "baseline_recall@20": baseline_cf["recall@20"],
        "best_recall@20": best_m["recall@20"],
    }
    with open(output_dir / "exp1_significance.json", "w") as f:
        json.dump(significance_report, f, indent=2)

    print("\n" + "=" * 105)
    print("EXPERIMENT 1: ITEM-BASED COLLABORATIVE FILTERING ABLATION RESULTS (All 4,980 Val Users)")
    print("=" * 105)
    fmt = "{:<48} {:<12} {:<12} {:<12} {:<12} {:<10}"
    print(fmt.format("Configuration", "Recall@20", "NDCG@10", "Recall@100", "Pos NDCG@10", "ILD"))
    print("-" * 105)
    print(
        fmt.format(
            "Baseline Item-CF",
            f"{baseline_cf['recall@20']:.4f}",
            f"{baseline_cf['ndcg@10']:.4f}",
            f"{baseline_cf['recall@100']:.4f}",
            f"{baseline_cf['positive_only_ndcg@10']:.4f}",
            f"{baseline_cf['intra_list_diversity']:.4f}",
        )
    )
    for name, m in exp1_results.items():
        print(
            fmt.format(
                name,
                f"{m['recall@20']:.4f}",
                f"{m['ndcg@10']:.4f}",
                f"{m['recall@100']:.4f}",
                f"{m['positive_only_ndcg@10']:.4f}",
                f"{m['intra_list_diversity']:.4f}",
            )
        )
    print("-" * 105)
    print(f"Tournament Best: {best_variant_name}")
    print(f"Paired delta vs Baseline CF: Delta NDCG@10 = {diff_mean:+.4f} [{ci_l:+.4f}, {ci_u:+.4f}], p = {p_val:.4f}")
    print("=" * 105)


if __name__ == "__main__":
    run_experiment_1()
