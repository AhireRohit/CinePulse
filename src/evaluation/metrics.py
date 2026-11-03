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
