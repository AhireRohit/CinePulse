import json
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.core.config import settings
from src.core.logging import logger
from src.models.baselines import ItemCollaborativeFiltering, PopularityRecommender
from src.models.ranker import LightGBMRanker
from src.models.two_tower import TwoTowerRecommender
from src.ranking.feature_engineering import RankingFeatureExtractor
from src.retrieval.faiss_index import FaissCandidateIndex


def construct_ranking_dataset(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    genre_cols: list[str],
    user_histories: dict[int, list[int]],
    item_embeddings: np.ndarray,
    tt_model: TwoTowerRecommender,
    popularity_recommender: PopularityRecommender,
    item_cf: ItemCollaborativeFiltering,
    faiss_index: FaissCandidateIndex,
    device: torch.device,
    num_users_to_sample: int = 3000,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """
    Constructs high-fidelity multi-source candidate dataset for LightGBM ranking.
    Uses exact Two-Tower user tower representations and genuine candidate pools
    (TT Top-100 + Item-CF Top-100 + Popularity Top-30) matching production serving.
    """
    feature_extractor = RankingFeatureExtractor(
        movies_df=movies_df,
        genre_cols=genre_cols,
        global_mean_rating=float(ratings_df["rating"].mean()),
    )
    num_movies = len(movies_df)
    movie_genres = movies_df[genre_cols].values.astype(np.float32)

    unique_users = ratings_df["user_idx"].unique()
    if len(unique_users) > num_users_to_sample:
        np.random.seed(42)
        sampled_users = np.random.choice(unique_users, size=num_users_to_sample, replace=False)
    else:
        sampled_users = unique_users

    all_features = []
    all_labels = []
    group_sizes = []

    tt_model.eval()

    for u_idx in sampled_users:
        u_idx = int(u_idx)
        user_ratings = ratings_df[ratings_df["user_idx"] == u_idx]
        history = list(user_histories.get(u_idx, []))
        if not history or user_ratings.empty:
            continue

        pos_candidates = user_ratings[user_ratings["rating"] >= 3.0]
        if pos_candidates.empty:
            continue
        pos_row = pos_candidates.iloc[-1]
        pos_item = int(pos_row["movie_idx"])
        pos_rating = float(pos_row["rating"])

        # Graded relevance labels: 5★=3, 4★=2, 3★=1
        if pos_rating >= 4.5:
            pos_label = 3
        elif pos_rating >= 3.5:
            pos_label = 2
        else:
            pos_label = 1

        # History strictly excludes the target positive item (zero target leakage)
        history_clean = [m for m in history if m != pos_item]
        consumed = set(history_clean)

        # 1. Compute exact user vector using PyTorch User Tower
        padded_hist = np.full(settings.max_sequence_length, num_movies, dtype=np.int64)
        h_len = min(len(history_clean), settings.max_sequence_length)
        if h_len > 0:
            padded_hist[-h_len:] = history_clean[-h_len:]

        valid_hist = [m for m in history_clean if m < num_movies]
        if valid_hist:
            g_pref = movie_genres[valid_hist].mean(axis=0)
        else:
            g_pref = np.zeros(len(genre_cols), dtype=np.float32)

        with torch.no_grad():
            u_t = torch.tensor([u_idx], dtype=torch.long, device=device)
            h_t = torch.tensor(padded_hist, dtype=torch.long, device=device).unsqueeze(0)
            hl_t = torch.tensor([h_len], dtype=torch.long, device=device)
            gp_t = torch.tensor(g_pref, dtype=torch.float32, device=device).unsqueeze(0)
            user_vec = tt_model.user_tower(u_t, h_t, hl_t, gp_t).cpu().numpy()[0]

        # 2. Retrieve multi-source candidates (Matching serving pool: 100 TT + 100 CF + 30 Pop)
        candidate_sources: dict[int, set[str]] = {}
        cf_score_map: dict[int, float] = {}

        # 2a. Two-Tower candidates
        tt_cands = faiss_index.search(user_vec, top_k=100, exclude_item_indices=consumed)
        for rank_pos, (c, s) in enumerate(tt_cands):
            candidate_sources.setdefault(c, set()).add("two_tower")

        # 2b. Item-CF candidates (queried on positive history)
        cf_recs = item_cf.recommend(history_clean, top_k=100)
        for rank_pos, (c, s) in enumerate(cf_recs):
            if c not in consumed:
                candidate_sources.setdefault(c, set()).add("item_cf")
                cf_score_map[c] = float(s)

        # 2c. Popularity candidates
        for rank_pos, (c, _) in enumerate(popularity_recommender.recommend(history_clean, top_k=30)):
            if c not in consumed:
                candidate_sources.setdefault(c, set()).add("popularity")

        # 3. Ensure positive target item is included in candidate set
        if pos_item not in candidate_sources:
            candidate_sources.setdefault(pos_item, set()).add("target_pos")
            if pos_item in dict(cf_recs):
                cf_score_map[pos_item] = float(dict(cf_recs)[pos_item])

        candidate_list = list(candidate_sources.keys())
        labels = [pos_label if c == pos_item else 0 for c in candidate_list]

        # Compute genuine Two-Tower similarity for ALL candidates
        cand_arr = np.array(candidate_list, dtype=np.int64)
        valid_mask = cand_arr < len(item_embeddings)
        tt_scores_dict: dict[int, float] = {}
        if np.any(valid_mask):
            cand_embs = item_embeddings[cand_arr[valid_mask]]
            sims = np.dot(cand_embs, user_vec)
            for idx_c, sim in zip(cand_arr[valid_mask], sims):
                tt_scores_dict[int(idx_c)] = float(sim)
        for c in candidate_list:
            if c not in tt_scores_dict:
                tt_scores_dict[c] = 0.0

        # Shuffle candidates within group to avoid rank-order bias
        combined = list(zip(candidate_list, labels))
        random.shuffle(combined)
        shuffled_candidates = [c for c, _ in combined]
        shuffled_labels = [lbl for _, lbl in combined]

        two_tower_scores = {c: tt_scores_dict[c] for c in shuffled_candidates}
        cf_scores = {c: cf_score_map.get(c, 0.0) for c in shuffled_candidates}
        sources = {c: candidate_sources[c] for c in shuffled_candidates}

        feat_df = feature_extractor.extract_features(
            user_idx=u_idx,
            user_history=history_clean,
            candidate_items=shuffled_candidates,
            two_tower_scores=two_tower_scores,
            cf_scores=cf_scores,
            retrieval_sources=sources,
        )

        if not feat_df.empty and len(feat_df) == len(shuffled_labels):
            all_features.append(feat_df)
            all_labels.extend(shuffled_labels)
            group_sizes.append(len(shuffled_labels))

    X = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    y = np.array(all_labels, dtype=np.int32)
    return X, y, group_sizes


def train_ranker(
    data_dir: Path = Path("data/processed"),
    output_dir: Path = Path("artifacts/models"),
    num_train_users: int = 3500,
    num_val_users: int = 800,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not settings.force_cpu else "cpu")
    logger.info(f"Starting Stage C LightGBM Ranker training on {device}...")

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

    # Load trained Two-Tower model
    logger.info("Loading trained Two-Tower model for exact user representation extraction...")
    two_tower_path = output_dir / "two_tower_best.pt"
    tt_model = TwoTowerRecommender(
        num_users=num_users,
        num_movies=num_movies,
        num_genres=len(genre_cols),
        embedding_dim=settings.embedding_dim,
    ).to(device)
    ckpt = torch.load(two_tower_path, map_location=device, weights_only=True)
    tt_model.load_state_dict(ckpt["model_state_dict"])
    tt_model.eval()

    # Initialize baselines
    logger.info("Initializing baseline retrieval models...")
    popularity = PopularityRecommender()
    popularity.fit(train_ratings, num_movies=num_movies)

    item_cf = ItemCollaborativeFiltering()
    item_cf.fit(train_ratings, num_users=num_users, num_movies=num_movies)

    faiss_index = FaissCandidateIndex(dimension=item_embeddings.shape[1])
    indices_dir = output_dir.parent / "indices"
    faiss_index.load(indices_dir)

    # Build train and validation datasets
    logger.info("Constructing high-fidelity multi-source ranking training pairs...")
    X_train, y_train, group_train = construct_ranking_dataset(
        ratings_df=train_ratings,
        movies_df=movies_df,
        genre_cols=genre_cols,
        user_histories=user_histories,
        item_embeddings=item_embeddings,
        tt_model=tt_model,
        popularity_recommender=popularity,
        item_cf=item_cf,
        faiss_index=faiss_index,
        device=device,
        num_users_to_sample=num_train_users,
    )

    logger.info("Constructing multi-source ranking validation pairs...")
    X_val, y_val, group_val = construct_ranking_dataset(
        ratings_df=val_ratings,
        movies_df=movies_df,
        genre_cols=genre_cols,
        user_histories=user_histories,
        item_embeddings=item_embeddings,
        tt_model=tt_model,
        popularity_recommender=popularity,
        item_cf=item_cf,
        faiss_index=faiss_index,
        device=device,
        num_users_to_sample=num_val_users,
    )

    logger.info(
        f"Training set: {len(X_train):,} candidate rows across {len(group_train):,} user groups.\n"
        f"Validation set: {len(X_val):,} candidate rows across {len(group_val):,} user groups."
    )

    # Train LightGBM Ranker with LambdaRank
    ranker = LightGBMRanker(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
    )
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
