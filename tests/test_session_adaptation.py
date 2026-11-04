import numpy as np
import pandas as pd

from src.pipeline.engine import RecommendationEngine


def test_session_adaptation_immediate_effect():
    """
    Verifies that when a user records a like or dismiss action,
    the recommendation response updates immediately without model retraining.
    """
    movies_data = {
        "movie_idx": list(range(20)),
        "movieId": [100 + i for i in range(20)],
        "clean_title": [f"Film {i}" for i in range(20)],
        "title": [f"Film {i} (2000)" for i in range(20)],
        "release_year": [2000] * 20,
        "genres": ["Action"] * 10 + ["Romance"] * 10,
        "rating_count": [100] * 20,
        "rating_mean": [4.0] * 20,
        "bayesian_rating": [3.9] * 20,
        "genre_Action": [1] * 10 + [0] * 10,
        "genre_Romance": [0] * 10 + [1] * 10,
    }
    movies_df = pd.DataFrame(movies_data)
    mappings = {
        "num_users": 10,
        "num_movies": 20,
        "global_rating_mean": 3.5,
    }
    user_histories = {1: [0, 1]}

    engine = RecommendationEngine(
        movies_df=movies_df,
        mappings=mappings,
        genre_cols=["genre_Action", "genre_Romance"],
        user_histories=user_histories,
    )

    # Initialize baselines & FAISS
    ratings_dummy = pd.DataFrame({
        "user_idx": [1, 1],
        "movie_idx": [0, 1],
        "rating": [4.5, 5.0],
    })
    engine.popularity_recommender.fit(ratings_dummy, num_movies=20)
    engine.item_cf.fit(ratings_dummy, num_users=10, num_movies=20)

    embeddings = np.random.normal(0, 1, size=(20, 64)).astype(np.float32)
    engine.faiss_index.build(embeddings)

    # 1. Baseline recommendations
    recs_initial = engine.recommend(user_idx=1, top_k=5)
    rec_indices_initial = [r["movie_idx"] for r in recs_initial["recommendations"]]

    # 2. Dismiss an item that was recommended
    target_dismiss = rec_indices_initial[0]
    engine.record_feedback(user_idx=1, movie_idx=target_dismiss, action="dismiss")

    # 3. Request updated recommendations
    recs_updated = engine.recommend(user_idx=1, top_k=5)
    rec_indices_updated = [r["movie_idx"] for r in recs_updated["recommendations"]]

    # The dismissed item must be strictly suppressed
    assert target_dismiss not in rec_indices_updated, f"Dismissed item {target_dismiss} must not be recommended"


def test_session_persistence_restoration_across_restarts():
    """
    Verifies that when user feedback and onboarding preferences are restored into
    a freshly initialized engine (simulating an app restart), the recommendations
    maintain suppression of dismissed/disliked items and preserve saved movies.
    """
    movies_data = {
        "movie_idx": list(range(20)),
        "movieId": [100 + i for i in range(20)],
        "clean_title": [f"Film {i}" for i in range(20)],
        "title": [f"Film {i} (2000)" for i in range(20)],
        "release_year": [2000] * 20,
        "genres": ["Action"] * 10 + ["Romance"] * 10,
        "rating_count": [100] * 20,
        "rating_mean": [4.0] * 20,
        "bayesian_rating": [3.9] * 20,
        "genre_Action": [1] * 10 + [0] * 10,
        "genre_Romance": [0] * 10 + [1] * 10,
    }
    movies_df = pd.DataFrame(movies_data)
    mappings = {"num_users": 10, "num_movies": 20, "global_rating_mean": 3.5}

    # Simulate fresh engine startup (Restart #1)
    engine1 = RecommendationEngine(
        movies_df=movies_df,
        mappings=mappings,
        genre_cols=["genre_Action", "genre_Romance"],
        user_histories={2: [0]},
    )
    engine1.popularity_recommender.fit(pd.DataFrame({"user_idx": [2], "movie_idx": [0], "rating": [5.0]}), num_movies=20)
    engine1.item_cf.fit(pd.DataFrame({"user_idx": [2], "movie_idx": [0], "rating": [5.0]}), num_users=10, num_movies=20)
    engine1.faiss_index.build(np.random.normal(0, 1, size=(20, 64)).astype(np.float32))

    # User 2 performs actions in engine 1
    engine1.record_feedback(user_idx=2, movie_idx=5, action="dislike")
    engine1.record_feedback(user_idx=2, movie_idx=8, action="save")
    engine1.record_feedback(user_idx=2, movie_idx=15, action="like")

    # Persisted interactions to be saved in DB
    persisted_feedback = [
        {"user_idx": 2, "movie_idx": 5, "action": "dislike", "rating": None},
        {"user_idx": 2, "movie_idx": 8, "action": "save", "rating": None},
        {"user_idx": 2, "movie_idx": 15, "action": "like", "rating": 5.0},
    ]

    # Simulate fresh engine startup (Restart #2)
    engine2 = RecommendationEngine(
        movies_df=movies_df,
        mappings=mappings,
        genre_cols=["genre_Action", "genre_Romance"],
        user_histories={2: [0]},
    )
    engine2.popularity_recommender.fit(pd.DataFrame({"user_idx": [2], "movie_idx": [0], "rating": [5.0]}), num_movies=20)
    engine2.item_cf.fit(pd.DataFrame({"user_idx": [2], "movie_idx": [0], "rating": [5.0]}), num_users=10, num_movies=20)
    engine2.faiss_index.build(np.random.normal(0, 1, size=(20, 64)).astype(np.float32))

    # Restore feedback into engine2
    for item in persisted_feedback:
        engine2.record_feedback(
            user_idx=item["user_idx"],
            movie_idx=item["movie_idx"],
            action=item["action"],
            rating=item["rating"],
        )

    # Check that state is completely preserved
    sess2 = engine2.get_user_session(2)
    assert 5 in sess2["dislikes"]
    assert 5 in sess2["dismissals"]
    assert 8 in sess2["saves"]
    assert 15 in sess2["likes"]

    recs2 = engine2.recommend(user_idx=2, top_k=10)
    rec_indices2 = [r["movie_idx"] for r in recs2["recommendations"]]
    assert 5 not in rec_indices2, "Disliked item 5 must remain suppressed after engine reload"

