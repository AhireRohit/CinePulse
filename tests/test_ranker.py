import numpy as np
import pandas as pd

from src.models.ranker import RANKER_FEATURE_COLS, LightGBMRanker
from src.ranking.feature_engineering import RankingFeatureExtractor


def test_ranking_feature_extractor():
    movies_data = {
        "movie_idx": [0, 1, 2],
        "movieId": [10, 20, 30],
        "release_year": [1995, 2001, 2010],
        "rating_count": [100, 50, 200],
        "rating_mean": [4.0, 3.5, 4.2],
        "bayesian_rating": [3.9, 3.5, 4.1],
        "genre_Action": [1, 0, 1],
        "genre_Sci-Fi": [0, 1, 1],
    }
    movies_df = pd.DataFrame(movies_data)
    genre_cols = ["genre_Action", "genre_Sci-Fi"]

    extractor = RankingFeatureExtractor(movies_df, genre_cols=genre_cols)

    candidates = [0, 1, 2]
    user_history = [0]
    tt_scores = {0: 0.9, 1: 0.4, 2: 0.75}

    features = extractor.extract_features(
        user_idx=1,
        user_history=user_history,
        candidate_items=candidates,
        two_tower_scores=tt_scores,
    )

    assert len(features) == 3
    for col in RANKER_FEATURE_COLS:
        assert col in features.columns
        assert not features[col].isna().any()


def test_lightgbm_ranker_fit_predict():
    rng = np.random.default_rng(42)
    n_queries = 20
    items_per_query = 5
    n_samples = n_queries * items_per_query

    data = {col: rng.normal(0, 1, size=n_samples).astype(np.float32) for col in RANKER_FEATURE_COLS}
    data["movie_idx"] = np.tile(np.arange(items_per_query), n_queries)
    df = pd.DataFrame(data)

    labels = rng.integers(0, 3, size=n_samples)
    groups = [items_per_query] * n_queries

    ranker = LightGBMRanker(n_estimators=10, learning_rate=0.1)
    ranker.fit(df, labels, groups)

    assert ranker.is_fitted
    scores = ranker.predict_scores(df.head(items_per_query))
    assert len(scores) == items_per_query

    ranked = ranker.rank(df.head(items_per_query))
    assert len(ranked) == items_per_query
    # Monotonicity
    assert all(ranked[i][1] >= ranked[i+1][1] for i in range(len(ranked)-1))
