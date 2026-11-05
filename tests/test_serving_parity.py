"""
Automated Serving Parity Tests:
Verifies that the deployed-in-process inference pipeline (RecommendationEngine)
generates the EXACT same ranked movie IDs, 24-feature vectors, and model scores
as the experimental evaluator under identical user context, candidate eligibility,
and model settings.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import torch

from pipelines.exp3_ranker import EXPANDED_FEATURE_COLS, FastRankingFeatureExtractor
from src.core.config import settings
from src.models.two_tower import TwoTowerRecommender
from src.pipeline.engine import RecommendationEngine
from src.reranking.mmr import MMRReRanker


@pytest.fixture(scope="module")
def setup_parity_environment():
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

    # 1. Initialize RecommendationEngine in v2 mode
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

    # 2. Standalone experimental models
    exp_booster = lgb.Booster(model_file=str(champion_ranker_path))
    exp_extractor = FastRankingFeatureExtractor(
        movies_df=movies_df,
        genre_cols=genre_cols,
        global_mean_rating=mappings.get("global_rating_mean", 3.5),
    )
    exp_mmr = MMRReRanker(
        movies_df=movies_df,
        genre_cols=genre_cols,
        default_lambda=0.95,
        default_novelty_weight=0.01,
    )

    return {
        "engine": engine,
        "movies_df": movies_df,
        "user_histories": user_histories,
        "exp_booster": exp_booster,
        "exp_extractor": exp_extractor,
        "exp_mmr": exp_mmr,
    }


def test_24_features_schema_and_order(setup_parity_environment):
    env = setup_parity_environment
    engine = env["engine"]

    assert len(engine.ranker.feature_cols) == 24
    assert engine.ranker.feature_cols == EXPANDED_FEATURE_COLS
    assert engine.feature_extractor.use_expanded is True


@pytest.mark.parametrize("user_idx", [10, 42, 100, 250, 500, 1000])
def test_user_serving_parity(setup_parity_environment, user_idx):
    env = setup_parity_environment
    engine = env["engine"]
    exp_booster = env["exp_booster"]
    exp_extractor = env["exp_extractor"]
    exp_mmr = env["exp_mmr"]

    history = engine.user_histories.get(user_idx, [])
    assert len(history) > 0, f"User {user_idx} has empty history"

    # 1. Execute engine recommendation
    rec_result = engine.recommend(
        user_idx=user_idx,
        top_k=20,
        retrieval_k=200,
        lambda_param=0.95,
        novelty_weight=0.01,
    )
    engine_top20_ids = [r["movie_idx"] for r in rec_result["recommendations"]]
    assert len(engine_top20_ids) == 20

    # 2. Replicate candidate retrieval with identical context
    excluded = set(history)
    u_vec = engine.generate_user_vector(user_idx, history)

    # Retrieval
    c_sources = {}
    cf_map = {}
    for item_idx, _ in engine.faiss_index.search(u_vec, top_k=200, exclude_item_indices=excluded):
        c_sources.setdefault(item_idx, set()).add("two_tower")

    session = engine.get_user_session(user_idx)
    pos_hist = [
        m for m in history
        if m not in session["dislikes"] and m not in session["dismissals"]
        and session.get("ratings", {}).get(m, 4.0) >= 3.5
    ]
    if not pos_hist and history:
        pos_hist = [m for m in history if m not in session["dislikes"]]

    if pos_hist:
        for item_idx, score in engine.item_cf.recommend(pos_hist, top_k=100):
            if item_idx not in excluded:
                c_sources.setdefault(item_idx, set()).add("item_cf")
                cf_map[item_idx] = float(score)

    for item_idx, _ in engine.popularity_recommender.recommend(history, top_k=30):
        if item_idx not in excluded:
            c_sources.setdefault(item_idx, set()).add("popularity")

    c_list = list(c_sources.keys())
    assert len(c_list) == rec_result["total_candidates"]

    # Compute Two-Tower dot products
    cand_arr = np.array(c_list, dtype=np.int64)
    cand_embs = engine.item_embeddings[cand_arr]
    sims = np.dot(cand_embs, u_vec)
    tt_map = {int(c): float(s) for c, s in zip(c_list, sims)}

    # 3. Extract features via experimental extractor
    u_ratings = engine.user_train_ratings.get(user_idx, {})
    exp_matrix = exp_extractor.extract_matrix(
        user_idx=user_idx,
        user_history=history,
        user_ratings_map=u_ratings,
        candidate_items=c_list,
        two_tower_scores=tt_map,
        cf_scores=cf_map,
        retrieval_sources=c_sources,
        use_expanded=True,
    )

    # 4. Extract features via engine serving extractor
    serving_df = engine.feature_extractor.extract_features(
        user_idx=user_idx,
        user_history=history,
        candidate_items=c_list,
        two_tower_scores=tt_map,
        cf_scores=cf_map,
        retrieval_sources=c_sources,
        use_expanded=True,
        user_ratings=u_ratings,
    )
    serving_matrix = serving_df[EXPANDED_FEATURE_COLS].values.astype(np.float32)

    # Assert feature parity
    np.testing.assert_allclose(
        serving_matrix,
        exp_matrix,
        atol=1e-4,
        err_msg=f"Feature matrix mismatch for user {user_idx}!",
    )

    # 5. Predict scores
    serving_scores = engine.ranker.predict_scores(serving_df)
    exp_scores = exp_booster.predict(exp_matrix)
    np.testing.assert_allclose(
        serving_scores,
        exp_scores,
        atol=1e-5,
        err_msg=f"Model prediction scores mismatch for user {user_idx}!",
    )

    # 6. Ranked order
    exp_ranked = [(c_list[i], float(exp_scores[i])) for i in np.argsort(-exp_scores)]
    exp_reranked = exp_mmr.rerank(
        candidate_scores=exp_ranked,
        top_k=20,
        lambda_param=0.95,
        novelty_weight=0.01,
        excluded_item_indices=excluded,
    )
    exp_top20_ids = [item["movie_idx"] for item in exp_reranked]

    # Assert exact top-20 parity
    assert engine_top20_ids == exp_top20_ids, f"Top-20 recommendation IDs mismatch for user {user_idx}!"


def test_reproducibility(setup_parity_environment):
    env = setup_parity_environment
    engine = env["engine"]

    rec1 = engine.recommend(user_idx=42, top_k=20)
    rec2 = engine.recommend(user_idx=42, top_k=20)

    ids1 = [r["movie_idx"] for r in rec1["recommendations"]]
    ids2 = [r["movie_idx"] for r in rec2["recommendations"]]
    assert ids1 == ids2, "Recommendations must be deterministic across repeated calls"
