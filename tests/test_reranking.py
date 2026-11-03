import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_intra_list_diversity
from src.reranking.mmr import MMRReRanker


def test_mmr_reranking_diversity_tradeoff():
    """
    Verifies that lower lambda (higher diversity penalty) yields higher Intra-List Diversity (ILD).
    """
    movies_data = {
        "movie_idx": list(range(10)),
        "rating_count": [1000] * 5 + [10] * 5,
        "genre_Action": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "genre_Romance": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    }
    movies_df = pd.DataFrame(movies_data)
    genre_cols = ["genre_Action", "genre_Romance"]

    reranker = MMRReRanker(movies_df, genre_cols=genre_cols)

    # Action movies have higher relevance scores, Romance have lower
    candidate_scores = [
        (0, 0.95), (1, 0.90), (2, 0.88), (3, 0.85), (4, 0.82),
        (5, 0.70), (6, 0.68), (7, 0.65), (8, 0.62), (9, 0.60),
    ]

    # Pure relevance: lambda = 1.0 (should select all action movies first)
    pure_rel = reranker.rerank(candidate_scores, top_k=4, lambda_param=1.0, novelty_weight=0.0)
    rel_indices = [item["movie_idx"] for item in pure_rel]
    assert rel_indices == [0, 1, 2, 3], "Pure relevance should greedily pick highest scoring Action films"

    # Diverse rerank: lambda = 0.3
    diverse = reranker.rerank(candidate_scores, top_k=4, lambda_param=0.3, novelty_weight=0.0)
    div_indices = [item["movie_idx"] for item in diverse]

    # Diverse ranking should interleave genres to minimize similarity
    genres_matrix = movies_df[genre_cols].values.astype(np.float32)
    ild_pure = compute_intra_list_diversity(rel_indices, genres_matrix)
    ild_div = compute_intra_list_diversity(div_indices, genres_matrix)

    assert ild_div > ild_pure, f"Expected diverse ILD {ild_div} to be strictly greater than pure ILD {ild_pure}"


def test_mmr_novelty_boost():
    """
    Verifies that high novelty weight promotes low-count items.
    """
    movies_data = {
        "movie_idx": [0, 1],
        "rating_count": [10000, 5],  # 0 is mainstream, 1 is rare/obscure
        "genre_Action": [1, 1],
    }
    movies_df = pd.DataFrame(movies_data)
    genre_cols = ["genre_Action"]

    reranker = MMRReRanker(movies_df, genre_cols=genre_cols)

    # Item 0 has slightly higher relevance, but item 1 has vastly higher novelty
    candidates = [(0, 0.80), (1, 0.78)]

    # Without novelty boost: item 0 wins
    standard = reranker.rerank(candidates, top_k=1, lambda_param=1.0, novelty_weight=0.0)
    assert standard[0]["movie_idx"] == 0

    # With high novelty boost: item 1 wins
    boosted = reranker.rerank(candidates, top_k=1, lambda_param=1.0, novelty_weight=0.5)
    assert boosted[0]["movie_idx"] == 1
