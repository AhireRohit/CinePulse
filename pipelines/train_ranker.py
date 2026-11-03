import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.logging import logger
from src.models.baselines import ItemCollaborativeFiltering, PopularityRecommender
from src.models.ranker import LightGBMRanker
from src.ranking.feature_engineering import RankingFeatureExtractor


def construct_ranking_dataset(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    genre_cols: list[str],
    user_histories: dict[int, list[int]],
    item_embeddings: np.ndarray,
    popularity_recommender: PopularityRecommender,
    item_cf: ItemCollaborativeFiltering,
    num_users_to_sample: int = 2500,
    negatives_per_positive: int = 9,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """
    Constructs query-grouped dataset for LightGBM ranking.
    Computes real feature values (two-tower similarity, CF score, popularity, genre affinity)
    without leakage.
    """
    feature_extractor = RankingFeatureExtractor(
        movies_df=movies_df,
        genre_cols=genre_cols,
        global_mean_rating=float(ratings_df["rating"].mean()),
    )

    unique_users = ratings_df["user_idx"].unique()
    if len(unique_users) > num_users_to_sample:
        sampled_users = np.random.choice(unique_users, size=num_users_to_sample, replace=False)
    else:
        sampled_users = unique_users

    all_features = []
    all_labels = []
    group_sizes = []

    for u_idx in sampled_users:
        u_idx = int(u_idx)
        user_ratings = ratings_df[ratings_df["user_idx"] == u_idx]
        history = list(user_histories.get(u_idx, []))
        if not history or user_ratings.empty:
            continue

        pos_row = user_ratings.iloc[-1]
        pos_item = int(pos_row["movie_idx"])
        pos_rating = float(pos_row["rating"])
        pos_label = 2 if pos_rating >= 4.0 else (1 if pos_rating >= 3.0 else 0)

        consumed = set(history) | {pos_item}

        # Select negative candidates from popularity and unobserved
        pop_candidates = [idx for idx, _ in popularity_recommender.recommend(history, top_k=20) if idx not in consumed]
        selected_negatives = []
        if pop_candidates:
            selected_negatives.extend(random.sample(pop_candidates, min(len(pop_candidates), negatives_per_positive // 2)))

        while len(selected_negatives) < negatives_per_positive:
            rand_idx = random.randint(0, len(movies_df) - 1)
            if rand_idx not in consumed and rand_idx not in selected_negatives:
                selected_negatives.append(rand_idx)

        # Pairs of (item, label)
        candidate_pairs = [(pos_item, pos_label)] + [(neg, 0) for neg in selected_negatives]
        # Shuffle candidates within the query group so candidate_rank is not a trivial shortcut
        random.shuffle(candidate_pairs)

        candidates = [item for item, _ in candidate_pairs]
        labels = [label for _, label in candidate_pairs]

        # Compute pseudo-user embedding from history for feature extraction
        valid_hist = [m for m in history if m < len(item_embeddings)]
        if valid_hist:
            user_vec = item_embeddings[valid_hist].mean(axis=0)
            u_norm = np.linalg.norm(user_vec)
            if u_norm > 0:
                user_vec = user_vec / u_norm
        else:
            user_vec = np.zeros(item_embeddings.shape[1], dtype=np.float32)

        two_tower_scores = {}
        for c in candidates:
            if c < len(item_embeddings):
                two_tower_scores[c] = float(np.dot(user_vec, item_embeddings[c]))
            else:
                two_tower_scores[c] = 0.0

        cf_scores = {}
        if history:
            cf_recs = dict(item_cf.recommend(history[-5:], top_k=30))
            for c in candidates:
                cf_scores[c] = float(cf_recs.get(c, 0.0))

        sources = {c: ("positive" if c == pos_item else "candidate") for c in candidates}

        feat_df = feature_extractor.extract_features(
            user_idx=u_idx,
            user_history=history,
            candidate_items=candidates,
            two_tower_scores=two_tower_scores,
            cf_scores=cf_scores,
            retrieval_sources=sources,
        )

        if not feat_df.empty and len(feat_df) == len(labels):
            all_features.append(feat_df)
            all_labels.extend(labels)
            group_sizes.append(len(labels))

    X = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    y = np.array(all_labels, dtype=np.int32)
    return X, y, group_sizes


def train_ranker(
    data_dir: Path = Path("data/processed"),
    output_dir: Path = Path("artifacts/models"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Stage C LightGBM Ranker training pipeline...")

    # Load data
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    train_ratings = pd.read_parquet(data_dir / "ratings_train.parquet")
    val_ratings = pd.read_parquet(data_dir / "ratings_val.parquet")
    item_embeddings = np.load(output_dir / "item_embeddings.npy")

    with open(data_dir / "mappings.json") as f:
        mappings = json.load(f)
    with open(data_dir / "user_train_histories.json") as f:
        user_histories = {int(k): v for k, v in json.load(f).items()}

    genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]
    num_users = mappings["num_users"]
    num_movies = mappings["num_movies"]

    # Initialize baselines
    logger.info("Initializing baseline retrieval models...")
    popularity = PopularityRecommender()
    popularity.fit(train_ratings, num_movies=num_movies)

    item_cf = ItemCollaborativeFiltering()
    item_cf.fit(train_ratings, num_users=num_users, num_movies=num_movies)

    # Build train and validation datasets
    logger.info("Constructing ranking training pairs...")
    X_train, y_train, group_train = construct_ranking_dataset(
        ratings_df=train_ratings,
        movies_df=movies_df,
        genre_cols=genre_cols,
        user_histories=user_histories,
        item_embeddings=item_embeddings,
        popularity_recommender=popularity,
        item_cf=item_cf,
        num_users_to_sample=2500,
        negatives_per_positive=9,
    )

    logger.info("Constructing ranking validation pairs...")
    X_val, y_val, group_val = construct_ranking_dataset(
        ratings_df=val_ratings,
        movies_df=movies_df,
        genre_cols=genre_cols,
        user_histories=user_histories,
        item_embeddings=item_embeddings,
        popularity_recommender=popularity,
        item_cf=item_cf,
        num_users_to_sample=600,
        negatives_per_positive=9,
    )

    # Train LightGBM Ranker
    ranker = LightGBMRanker(n_estimators=200, learning_rate=0.04)
    ranker.fit(
        X_train=X_train,
        y_train=y_train,
        group_train=group_train,
        X_val=X_val,
        y_val=y_val,
        group_val=group_val,
    )

    # Save model
    ranker_path = output_dir / "ranker_lgbm.txt"
    ranker.save(ranker_path)
    logger.info(f"Stage C LightGBM Ranker successfully saved to {ranker_path}")


if __name__ == "__main__":
    train_ranker()
