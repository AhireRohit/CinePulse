import math

import numpy as np


def compute_recall_at_k(recommended: list[int], ground_truth: set[int], k: int) -> float:
    """
    Computes Recall@K: fraction of ground truth relevant items retrieved in top-K.
    """
    if not ground_truth:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & ground_truth)
    return hits / float(len(ground_truth))


def compute_ndcg_at_k(recommended: list[int], ground_truth: set[int], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance.
    """
    if not ground_truth:
        return 0.0

    rec_k = recommended[:k]
    dcg = 0.0
    for rank_idx, item in enumerate(rec_k):
        if item in ground_truth:
            dcg += 1.0 / math.log2(rank_idx + 2)

    # Ideal DCG with binary relevance
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def compute_graded_ndcg_at_k(
    recommended: list[int],
    ground_truth_ratings: dict[int, float],
    k: int,
) -> float:
    """
    Computes Graded Normalized Discounted Cumulative Gain at K (Graded-NDCG@K).

    Mathematical Note on Leave-Last-1-Out:
      When each user has exactly 1 held-out item in ground truth (|GT| = 1) with rating r:
      - If r >= 3.0, the relevance gain is g = 2^rel(r) - 1 > 0.
      - If retrieved at rank i: DCG = g / log2(i + 2).
      - Ideal DCG: IDCG = g / log2(0 + 2) = g.
      - Therefore: NDCG = DCG / IDCG = (g / log2(i + 2)) / g = 1 / log2(i + 2).
      The scalar gain g cancels out identically in numerator and denominator for any |GT| = 1.
      Consequently, graded NDCG and binary NDCG are mathematically identical under
      leave-one-out evaluation.
      To observe graded differentiation, evaluation must be conducted in a multi-positive
      setting (|GT| >= 2 with varying ratings).

    Relevance grades:
      rating == 5.0 -> rel = 3 (gain = 2^3 - 1 = 7)
      rating == 4.0 -> rel = 2 (gain = 2^2 - 1 = 3)
      rating == 3.0 -> rel = 1 (gain = 2^1 - 1 = 1)
      rating <= 2.0 -> rel = 0 (gain = 0, explicitly zero gain for 1-2 star dislikes)
    """
    if not ground_truth_ratings:
        return 0.0

    def get_gain(rating: float) -> float:
        if rating >= 4.5:
            return 7.0
        elif rating >= 3.5:
            return 3.0
        elif rating >= 2.5:
            return 1.0
        return 0.0

    rec_k = recommended[:k]
    dcg = 0.0
    for rank_idx, item in enumerate(rec_k):
        if item in ground_truth_ratings:
            dcg += get_gain(ground_truth_ratings[item]) / math.log2(rank_idx + 2)

    all_gains = sorted([get_gain(r) for r in ground_truth_ratings.values()], reverse=True)
    ideal_gains = all_gains[:k]
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal_gains))

    return dcg / idcg if idcg > 0 else 0.0


def compute_paired_bootstrap_ci(
    values_a: list[float] | np.ndarray,
    values_b: list[float] | np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """
    Computes paired user-level bootstrap confidence interval for the difference (A - B).
    Used to rigorously test whether model A statistically significantly outperforms model B.
    Returns (mean_diff, ci_lower, ci_upper, p_value).
    """
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    if len(arr_a) != len(arr_b) or len(arr_a) == 0:
        return 0.0, 0.0, 0.0, 1.0

    diffs = arr_a - arr_b
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(n_bootstrap, len(diffs)))
    boot_means = np.mean(diffs[indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    # Empirical two-tailed bootstrap p-value with standard finite-sample adjustment (Davison & Hinkley)
    # Under H0, resampled means are centered around zero:
    centered_boot_means = boot_means - mean_diff
    extreme_count = int(np.sum(np.abs(centered_boot_means) >= np.abs(mean_diff)))
    p_val = float((extreme_count + 1) / (n_bootstrap + 1))

    return round(mean_diff, 4), round(lower, 4), round(upper, 4), round(p_val, 4)


def compute_bootstrap_ci(
    values: list[float] | np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Computes empirical user-level bootstrap confidence interval.
    Returns (mean, ci_lower, ci_upper).
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    boot_means = np.mean(arr[indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return float(np.mean(arr)), round(lower, 4), round(upper, 4)


def compute_mrr_at_k(recommended: list[int], ground_truth: set[int], k: int) -> float:
    """
    Computes Mean Reciprocal Rank at K (MRR@K).
    """
    if not ground_truth:
        return 0.0
    rec_k = recommended[:k]
    for rank_idx, item in enumerate(rec_k):
        if item in ground_truth:
            return 1.0 / (rank_idx + 1)
    return 0.0


def compute_catalog_coverage(all_recommendations: list[list[int]], catalog_size: int) -> float:
    """
    Computes the fraction of unique catalog items recommended across all users.
    """
    if catalog_size == 0:
        return 0.0
    unique_recommended = set()
    for recs in all_recommendations:
        unique_recommended.update(recs)
    return len(unique_recommended) / float(catalog_size)


def compute_intra_list_diversity(
    recommended: list[int],
    item_feature_matrix: np.ndarray,
) -> float:
    """
    Computes Intra-List Diversity (ILD) using average pairwise cosine distance (1 - similarity).
    Higher ILD indicates greater topic/genre diversity in the recommendation list.
    """
    valid_items = [idx for idx in recommended if idx < len(item_feature_matrix)]
    n = len(valid_items)
    if n <= 1:
        return 0.0

    vectors = item_feature_matrix[valid_items]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_vectors = vectors / norms

    sim_matrix = np.dot(norm_vectors, norm_vectors.T)
    # Distance is 1 - similarity
    dist_matrix = 1.0 - sim_matrix
    # Extract upper triangle without diagonal
    triu_indices = np.triu_indices(n, k=1)
    return float(np.mean(dist_matrix[triu_indices]))


def compute_novelty(
    recommended: list[int],
    item_counts: np.ndarray,
    total_ratings: int,
) -> float:
    """
    Computes average novelty (self-information in bits: -log2(P(item))).
    Higher novelty means recommending less mainstream / more unexpected items.
    """
    if not recommended or total_ratings == 0:
        return 0.0

    valid_items = [idx for idx in recommended if idx < len(item_counts)]
    if not valid_items:
        return 0.0

    novelties = []
    for idx in valid_items:
        p = max(item_counts[idx], 1) / float(total_ratings)
        novelties.append(-math.log2(p))

    return float(np.mean(novelties))
