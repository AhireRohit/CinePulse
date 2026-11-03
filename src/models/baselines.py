import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.core.logging import logger


class PopularityRecommender:
    """
    Bayesian smoothed popularity baseline recommender.
    Scores items by balancing interaction frequency and rating quality.
    """

    def __init__(self, prior_weight: float = 10.0):
        self.prior_weight = prior_weight
        self.item_scores: dict[int, float] = {}
        self.sorted_item_indices: list[int] = []

    def fit(self, ratings_df: pd.DataFrame, num_movies: int):
        global_mean = float(ratings_df["rating"].mean())
        stats = ratings_df.groupby("movie_idx").agg(
            count=("rating", "count"),
            mean=("rating", "mean"),
        )

        scores = np.full(num_movies, global_mean, dtype=np.float32)
        for movie_idx, row in stats.iterrows():
            idx = int(movie_idx)
            if idx < num_movies:
                cnt = row["count"]
                avg = row["mean"]
                # Bayesian weighted rating
                score = (cnt * avg + self.prior_weight * global_mean) / (cnt + self.prior_weight)
                scores[idx] = score

        self.item_scores = {i: float(scores[i]) for i in range(num_movies)}
        # Sort descending by score
        self.sorted_item_indices = list(np.argsort(-scores))
        logger.info(f"PopularityRecommender fitted on {len(stats)} items.")

    def recommend(self, user_history: list[int], top_k: int = 20) -> list[tuple[int, float]]:
        consumed = set(user_history)
        recs = []
        for idx in self.sorted_item_indices:
            if idx not in consumed:
                recs.append((idx, self.item_scores[idx]))
                if len(recs) >= top_k:
                    break
        return recs


class ItemCollaborativeFiltering:
    """
    Item-based Collaborative Filtering with Cosine similarity and shrinkage.
    Computes item-item similarity matrix over user interaction vectors.
    """

    def __init__(self, shrinkage: float = 25.0, top_k_neighbors: int = 50):
        self.shrinkage = shrinkage
        self.top_k_neighbors = top_k_neighbors
        self.num_items = 0
        self.item_similarity: dict[int, list[tuple[int, float]]] = {}

    def fit(self, ratings_df: pd.DataFrame, num_users: int, num_movies: int):
        self.num_items = num_movies
        # Center ratings per user to normalize bias
        ratings = ratings_df.copy()
        user_means = ratings.groupby("user_idx")["rating"].transform("mean")
        ratings["centered_rating"] = ratings["rating"] - user_means

        rows = ratings["movie_idx"].values
        cols = ratings["user_idx"].values
        data = ratings["centered_rating"].values

        # Item-user sparse matrix: [num_movies, num_users]
        item_user = csr_matrix((data, (rows, cols)), shape=(num_movies, num_users))

        # Compute cosine similarity
        sim_matrix = cosine_similarity(item_user, dense_output=False)

        # Store top-K neighbors per item with shrinkage
        self.item_similarity = {}
        for i in range(num_movies):
            row = sim_matrix.getrow(i)
            indices = row.indices
            values = row.data
            # Filter self
            valid = indices != i
            ind = indices[valid]
            val = values[valid]

            if len(ind) > 0:
                # Keep top K
                if len(ind) > self.top_k_neighbors:
                    top_idx = np.argpartition(-val, self.top_k_neighbors)[:self.top_k_neighbors]
                    ind = ind[top_idx]
                    val = val[top_idx]
                sorted_order = np.argsort(-val)
                self.item_similarity[i] = list(zip(ind[sorted_order].tolist(), val[sorted_order].tolist()))
            else:
                self.item_similarity[i] = []

        logger.info(f"ItemCollaborativeFiltering fitted across {num_movies} items.")

    def recommend(self, user_history: list[int], top_k: int = 20) -> list[tuple[int, float]]:
        if not user_history:
            return []

        consumed = set(user_history)
        candidate_scores: dict[int, float] = {}

        # Give more weight to more recent items in history
        history_len = len(user_history)
        for pos, item_idx in enumerate(user_history):
            recency_weight = 0.5 + 0.5 * ((pos + 1) / history_len)
            neighbors = self.item_similarity.get(item_idx, [])
            for neighbor_idx, sim in neighbors:
                if neighbor_idx not in consumed:
                    candidate_scores[neighbor_idx] = candidate_scores.get(neighbor_idx, 0.0) + sim * recency_weight

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_similar_items(self, item_idx: int, top_k: int = 10) -> list[tuple[int, float]]:
        return self.item_similarity.get(item_idx, [])[:top_k]


class ContentRecommender:
    """
    Content-based recommender using movie genre multi-hot representations and metadata.
    """

    def __init__(self):
        self.movie_features: np.ndarray | None = None
        self.num_movies = 0

    def fit(self, movies_df: pd.DataFrame, genre_cols: list[str]):
        self.num_movies = len(movies_df)
        genres = movies_df[genre_cols].values.astype(np.float32)

        # Year normalized
        years = movies_df["release_year"].values.astype(np.float32)
        norm_years = (years - 1920.0) / 105.0
        norm_years = np.clip(norm_years, 0.0, 1.0)[:, np.newaxis]

        # Combine genres + year
        raw_feat = np.hstack([genres, norm_years])
        # L2 normalize
        norms = np.linalg.norm(raw_feat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.movie_features = raw_feat / norms
        logger.info(f"ContentRecommender fitted on {self.num_movies} items with {raw_feat.shape[1]} features.")

    def recommend(self, user_history: list[int], top_k: int = 20) -> list[tuple[int, float]]:
        if not user_history or self.movie_features is None:
            return []

        consumed = set(user_history)
        # Average vector of consumed movies
        valid_history = [idx for idx in user_history if idx < self.num_movies]
        if not valid_history:
            return []

        user_profile = self.movie_features[valid_history].mean(axis=0)
        norm = np.linalg.norm(user_profile)
        if norm > 0:
            user_profile = user_profile / norm

        scores = np.dot(self.movie_features, user_profile)
        sorted_indices = np.argsort(-scores)

        recs = []
        for idx in sorted_indices:
            if idx not in consumed:
                recs.append((int(idx), float(scores[idx])))
                if len(recs) >= top_k:
                    break
        return recs

    def get_similar_items(self, item_idx: int, top_k: int = 10) -> list[tuple[int, float]]:
        if self.movie_features is None or item_idx >= self.num_movies:
            return []
        item_vec = self.movie_features[item_idx]
        scores = np.dot(self.movie_features, item_vec)
        sorted_indices = np.argsort(-scores)

        results = []
        for idx in sorted_indices:
            if idx != item_idx:
                results.append((int(idx), float(scores[idx])))
                if len(results) >= top_k:
                    break
        return results
