"""
Product-Quality Safeguards Verification:
Validates user experience, demo personas, sparse/dense profiles,
cold-start onboarding, hero spotlight rating gates, and session adaptation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.config import settings
from src.models.two_tower import TwoTowerRecommender
from src.pipeline.engine import RecommendationEngine


@pytest.fixture(scope="module")
def setup_engine():
    data_dir = settings.data_processed_dir
    models_dir = settings.models_dir
    indices_dir = settings.indices_dir
    exp_models_dir = Path("artifacts/experiments/relevance_v2/models")

    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    train_ratings = pd.read_parquet(data_dir / "ratings_train.parquet")
    with open(data_dir / "mappings.json") as f:
        mappings = json.load(f)
    with open(data_dir / "user_train_histories.json") as f:
        user_histories = {int(k): v for k, v in json.load(f).items()}

    genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]
    num_users = mappings["num_users"]
    num_movies = mappings["num_movies"]

    user_train_ratings: dict[int, dict[int, float]] = {}
    for row in train_ratings[["user_idx", "movie_idx", "rating"]].itertuples(index=False):
        user_train_ratings.setdefault(int(row.user_idx), {})[int(row.movie_idx)] = float(row.rating)

    settings.model_version = "v2"
    engine = RecommendationEngine(
        movies_df=movies_df,
        mappings=mappings,
        genre_cols=genre_cols,
        user_histories=user_histories,
        user_train_ratings=user_train_ratings,
    )
    engine.popularity_recommender.fit(train_ratings, num_movies=num_movies)
    engine.item_cf.fit(train_ratings, num_users=num_users, num_movies=num_movies)

    two_tower_path = models_dir / "two_tower_best.pt"
    ckpt = torch.load(two_tower_path, map_location=engine.device, weights_only=True)
    tt_model = TwoTowerRecommender(
        num_users=num_users,
        num_movies=num_movies,
        num_genres=len(genre_cols),
        embedding_dim=settings.embedding_dim,
    ).to(engine.device)
    tt_model.load_state_dict(ckpt["model_state_dict"])
    tt_model.eval()
    engine.two_tower_model = tt_model

    engine.item_embeddings = np.load(models_dir / "item_embeddings.npy")
    engine.faiss_index.load(indices_dir)

    champion_ranker_path = exp_models_dir / "ranker_lgbm_exp3.txt"
    engine.ranker.load(champion_ranker_path)

    return engine


def test_demo_personas_quality(setup_engine):
    engine = setup_engine
    demo_users = [42, 100, 250, 500]

    for u in demo_users:
        res = engine.recommend(user_idx=u, top_k=20)
        recs = res["recommendations"]
        assert len(recs) == 20
        # Assert no duplicates
        rec_ids = [r["movie_idx"] for r in recs]
        assert len(rec_ids) == len(set(rec_ids)), f"Duplicate recommendations for user {u}"

        # Assert no consumed items recommended
        consumed = set(engine.user_histories.get(u, []))
        overlap = set(rec_ids) & consumed
        assert len(overlap) == 0, f"Consumed items recommended to user {u}: {overlap}"

        # Assert all recommendation cards have valid fields
        for r in recs:
            assert "title" in r and len(r["title"]) > 0
            assert "match_percentage" in r and 0 <= r["match_percentage"] <= 100
            assert "genres" in r and isinstance(r["genres"], list)
            assert "final_score" in r


def test_sparse_and_dense_users(setup_engine):
    engine = setup_engine
    hist_lens = {u: len(h) for u, h in engine.user_histories.items()}

    # In MovieLens 1M, find user with minimum history length and user with high history length (>= 200)
    min_len = min(hist_lens.values())
    sparse_user = next(u for u, length in hist_lens.items() if length == min_len)
    dense_user = next(u for u, length in hist_lens.items() if length >= 200)

    # Sparse user recommendations
    res_sparse = engine.recommend(user_idx=sparse_user, top_k=20)
    assert len(res_sparse["recommendations"]) == 20
    assert res_sparse["total_candidates"] >= 30

    # Dense user recommendations
    res_dense = engine.recommend(user_idx=dense_user, top_k=20)
    assert len(res_dense["recommendations"]) == 20
    assert res_dense["total_candidates"] >= 100


def test_cold_start_onboarding(setup_engine):
    engine = setup_engine
    cold_user_id = 999999  # New unregistered user

    # Cold start with preferred genres
    engine.user_sessions[cold_user_id] = {
        "history": [],
        "likes": set(),
        "dislikes": set(),
        "saves": set(),
        "dismissals": set(),
        "ratings": {},
        "preferred_genres": ["Sci-Fi", "Action"],
    }

    res = engine.recommend(user_idx=cold_user_id, top_k=20)
    recs = res["recommendations"]
    assert len(recs) == 20

    # Check that recommended movies have Sci-Fi or Action
    matching = 0
    for r in recs:
        genres = r.get("genres", [])
        if "Sci-Fi" in genres or "Action" in genres:
            matching += 1
    # At least 50% should match onboarding preferences
    assert matching >= 10, f"Expected >= 10 matching genres, got {matching}"


def test_hero_spotlight_quality_gate(setup_engine):
    engine = setup_engine
    for u in [10, 42, 100, 250, 500]:
        res = engine.recommend(user_idx=u, top_k=20)
        hero = res.get("hero_featured")
        if hero is not None:
            # Enforce community acclaim floor
            assert hero["bayesian_rating"] >= 3.20, f"Hero candidate {hero['title']} has bayesian rating < 3.20: {hero['bayesian_rating']}"
            assert hero["rating_mean"] >= 3.10, f"Hero candidate {hero['title']} has mean rating < 3.10: {hero['rating_mean']}"
            assert hero["rating_count"] >= 50, f"Hero candidate {hero['title']} has rating count < 50: {hero['rating_count']}"
            assert "Top Spotlight" in hero.get("reason", "")


def test_dynamic_session_adaptation(setup_engine):
    engine = setup_engine
    user_id = 888888  # Isolated test session

    engine.user_sessions[user_id] = {
        "history": [1, 2, 3],
        "likes": set(),
        "dislikes": set(),
        "saves": set(),
        "dismissals": set(),
        "ratings": {1: 4.5, 2: 4.0, 3: 5.0},
        "preferred_genres": [],
    }

    # Initial recommendation
    res1 = engine.recommend(user_idx=user_id, top_k=20)
    rec1_ids = [r["movie_idx"] for r in res1["recommendations"]]
    dislike_target = rec1_ids[0]

    # User dislikes the first recommendation
    engine.record_feedback(user_idx=user_id, movie_idx=dislike_target, action="dislike")

    # Second recommendation
    res2 = engine.recommend(user_idx=user_id, top_k=20)
    rec2_ids = [r["movie_idx"] for r in res2["recommendations"]]

    # Assert disliked movie is strictly excluded
    assert dislike_target not in rec2_ids, f"Disliked movie {dislike_target} was not excluded!"
    assert dislike_target in engine.user_sessions[user_id]["dislikes"]
    assert dislike_target in engine.user_sessions[user_id]["dismissals"]
