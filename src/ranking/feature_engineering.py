import numpy as np
import pandas as pd


class RankingFeatureExtractor:
    """
    Extracts features for candidates passed to the Stage C ranking model.
    Avoids target leakage by computing features strictly from prior historical signals.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        genre_cols: list[str],
        global_mean_rating: float = 3.5,
    ):
        self.movies_df = movies_df.set_index("movie_idx") if "movie_idx" in movies_df.columns else movies_df
        self.genre_cols = genre_cols
        self.global_mean_rating = global_mean_rating
        self.num_genres = len(genre_cols)

        # Precompute movie genre arrays
        self.movie_genres = self.movies_df[self.genre_cols].values.astype(np.float32)
        self.movie_years = self.movies_df["release_year"].values.astype(np.float32)
        self.movie_counts = self.movies_df["rating_count"].values.astype(np.float32)
        self.movie_means = self.movies_df["rating_mean"].values.astype(np.float32)
        self.movie_bayesian = self.movies_df["bayesian_rating"].values.astype(np.float32)

    def extract_features(
        self,
        user_idx: int,
        user_history: list[int],
        candidate_items: list[int],
        two_tower_scores: dict[int, float] | None = None,
        cf_scores: dict[int, float] | None = None,
        retrieval_sources: dict[int, str] | None = None,
        user_genre_profile: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Builds feature matrix for a user and a list of candidate items.
        Returns a DataFrame ready for LightGBM scoring.
        """
        num_candidates = len(candidate_items)
        if num_candidates == 0:
            return pd.DataFrame()

        # Compute user genre preference vector if not provided
        if user_genre_profile is None:
            if len(user_history) > 0:
                valid_hist = [idx for idx in user_history if idx < len(self.movies_df)]
                if valid_hist:
                    user_genre_profile = self.movie_genres[valid_hist].mean(axis=0)
                else:
                    user_genre_profile = np.zeros(self.num_genres, dtype=np.float32)
            else:
                user_genre_profile = np.zeros(self.num_genres, dtype=np.float32)

        user_activity_count = float(len(user_history))
        two_tower_scores = two_tower_scores or {}
        cf_scores = cf_scores or {}
        retrieval_sources = retrieval_sources or {}

        features = []
        for rank_pos, item_idx in enumerate(candidate_items):
            if item_idx >= len(self.movies_df):
                continue

            # 1. Two-Tower similarity score
            tt_score = two_tower_scores.get(item_idx, 0.0)

            # 2. Collaborative filtering score
            cf_score = cf_scores.get(item_idx, 0.0)

            # 3. Item popularity & quality signals
            item_year = float(self.movie_years[item_idx])
            log_count = float(np.log1p(self.movie_counts[item_idx]))
            item_mean = float(self.movie_means[item_idx])
            bayesian_score = float(self.movie_bayesian[item_idx])

            # 4. Genre affinity (cosine similarity / dot product between user genre pref and item genres)
            item_genre_vec = self.movie_genres[item_idx]
            genre_overlap = float(np.dot(user_genre_profile, item_genre_vec))
            user_norm = np.linalg.norm(user_genre_profile)
            item_norm = np.linalg.norm(item_genre_vec)
            genre_cosine = float(genre_overlap / (user_norm * item_norm)) if (user_norm > 0 and item_norm > 0) else 0.0

            # 5. Retrieval metadata
            source = retrieval_sources.get(item_idx, "unknown")
            is_two_tower = 1.0 if source == "two_tower" else 0.0
            is_item_cf = 1.0 if source == "item_cf" else 0.0
            is_popularity = 1.0 if source == "popularity" else 0.0

            features.append({
                "candidate_rank": rank_pos,
                "two_tower_score": tt_score,
                "cf_score": cf_score,
                "genre_overlap": genre_overlap,
                "genre_cosine": genre_cosine,
                "log_rating_count": log_count,
                "rating_mean": item_mean,
                "bayesian_rating": bayesian_score,
                "release_year": item_year,
                "user_activity_count": user_activity_count,
                "is_source_two_tower": is_two_tower,
                "is_source_item_cf": is_item_cf,
                "is_source_popularity": is_popularity,
                "movie_idx": item_idx,
            })

        return pd.DataFrame(features)
