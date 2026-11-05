import numpy as np
import pandas as pd

from src.core.config import settings
from src.models.ranker import EXPANDED_FEATURE_COLS, RANKER_FEATURE_COLS


class RankingFeatureExtractor:
    """
    Extracts high-fidelity personalized cross-features for candidates passed to Stage C.
    Guarantees strict feature parity between training and production serving.
    Avoids target leakage by computing features strictly from prior historical signals.
    Supports both 17-feature v1 and 24-feature v2 (Config 8) feature sets.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        genre_cols: list[str],
        global_mean_rating: float = 3.5,
        use_expanded: bool | None = None,
    ):
        self.movies_df = movies_df.set_index("movie_idx") if "movie_idx" in movies_df.columns else movies_df
        self.genre_cols = genre_cols
        self.global_mean_rating = global_mean_rating
        self.num_genres = len(genre_cols)
        self.use_expanded = (settings.model_version == "v2") if use_expanded is None else use_expanded

        # Precompute movie feature arrays for sub-millisecond vectorized indexing
        self.movie_genres = self.movies_df[self.genre_cols].values.astype(np.float32)
        self.movie_years = self.movies_df["release_year"].values.astype(np.float32)
        self.movie_counts = self.movies_df["rating_count"].values.astype(np.float32)
        self.movie_means = self.movies_df["rating_mean"].values.astype(np.float32)
        self.movie_stds = (
            self.movies_df["rating_std"].values.astype(np.float32)
            if "rating_std" in self.movies_df.columns
            else np.zeros(len(self.movies_df), dtype=np.float32)
        )
        self.movie_bayesian = self.movies_df["bayesian_rating"].values.astype(np.float32)
        self.movie_norms = np.linalg.norm(self.movie_genres, axis=1) + 1e-6

    def extract_features(
        self,
        user_idx: int,
        user_history: list[int],
        candidate_items: list[int],
        two_tower_scores: dict[int, float] | None = None,
        cf_scores: dict[int, float] | None = None,
        retrieval_sources: dict[int, str | set] | None = None,
        user_genre_profile: np.ndarray | None = None,
        user_ratings: dict[int, float] | None = None,
        use_expanded: bool | None = None,
    ) -> pd.DataFrame:
        use_exp = self.use_expanded if use_expanded is None else use_expanded
        num_candidates = len(candidate_items)
        if num_candidates == 0:
            cols = list(EXPANDED_FEATURE_COLS if use_exp else RANKER_FEATURE_COLS) + ["movie_idx"]
            return pd.DataFrame(columns=cols)

        # Compute user genre preference vector and era preference from history
        valid_hist = [idx for idx in user_history if idx < len(self.movies_df)] if user_history else []
        if user_genre_profile is None:
            if valid_hist:
                user_genre_profile = self.movie_genres[valid_hist].mean(axis=0)
            else:
                user_genre_profile = np.zeros(self.num_genres, dtype=np.float32)

        user_mean_year = float(self.movie_years[valid_hist].mean()) if valid_hist else 1990.0
        user_activity_count = float(len(user_history))

        # Recent genre profile (last 5 interactions)
        recent_hist = valid_hist[-5:]
        if recent_hist:
            user_recent_genre = self.movie_genres[recent_hist].mean(axis=0)
        else:
            user_recent_genre = user_genre_profile
        u_rec_norm = float(np.linalg.norm(user_recent_genre)) + 1e-6

        # User mean rating
        if user_ratings:
            hist_ratings = [user_ratings.get(m, self.global_mean_rating) for m in valid_hist]
            user_mean_rating = float(np.mean(hist_ratings)) if hist_ratings else self.global_mean_rating
        else:
            user_mean_rating = self.global_mean_rating

        two_tower_scores = two_tower_scores or {}
        cf_scores = cf_scores or {}
        retrieval_sources = retrieval_sources or {}

        user_norm = float(np.linalg.norm(user_genre_profile)) + 1e-6

        # Precompute candidate score vectors for normalizations
        cf_vals = np.array([cf_scores.get(c, 0.0) for c in candidate_items], dtype=np.float32)
        tt_vals = np.array([two_tower_scores.get(c, 0.0) for c in candidate_items], dtype=np.float32)

        cf_min, cf_max = float(cf_vals.min()), float(cf_vals.max())
        cf_range = cf_max - cf_min
        tt_min, tt_max = float(tt_vals.min()), float(tt_vals.max())
        tt_range = tt_max - tt_min

        features = []
        for rank_pos, item_idx in enumerate(candidate_items):
            if item_idx >= len(self.movies_df):
                continue

            # 1. Retrieval scores
            tt_score = float(tt_vals[rank_pos])
            cf_score = float(cf_vals[rank_pos])

            # 2. Movie quality & catalog statistics
            item_year = float(self.movie_years[item_idx])
            log_count = float(np.log1p(self.movie_counts[item_idx]))
            item_mean = float(self.movie_means[item_idx])
            item_std = float(self.movie_stds[item_idx])
            bayesian_score = float(self.movie_bayesian[item_idx])
            year_diff = abs(item_year - user_mean_year)

            # 3. Genre affinity
            item_genre_vec = self.movie_genres[item_idx]
            genre_overlap = float(np.dot(user_genre_profile, item_genre_vec))
            item_norm = float(self.movie_norms[item_idx])
            genre_cosine = float(genre_overlap / (user_norm * item_norm)) if (user_norm > 0 and item_norm > 0) else 0.0

            # 4. Source indicators
            source_info = retrieval_sources.get(item_idx, "unknown")
            if isinstance(source_info, (set, list, tuple)):
                is_two_tower = 1.0 if "two_tower" in source_info else 0.0
                is_item_cf = 1.0 if "item_cf" in source_info else 0.0
                is_popularity = 1.0 if "popularity" in source_info else 0.0
            elif isinstance(source_info, dict):
                is_two_tower = float(source_info.get("two_tower", 0.0))
                is_item_cf = float(source_info.get("item_cf", 0.0))
                is_popularity = float(source_info.get("popularity", 0.0))
            else:
                is_two_tower = 1.0 if source_info == "two_tower" else 0.0
                is_item_cf = 1.0 if source_info == "item_cf" else 0.0
                is_popularity = 1.0 if source_info == "popularity" else 0.0

            # 5. Non-linear interaction cross-features
            cf_x_bayesian = cf_score * (bayesian_score - self.global_mean_rating)
            tt_x_genre = tt_score * genre_cosine

            row = {
                "candidate_rank": rank_pos,
                "two_tower_score": tt_score,
                "cf_score": cf_score,
                "genre_overlap": genre_overlap,
                "genre_cosine": genre_cosine,
                "log_rating_count": log_count,
                "rating_mean": item_mean,
                "rating_std": item_std,
                "bayesian_rating": bayesian_score,
                "release_year": item_year,
                "year_diff": year_diff,
                "user_activity_count": user_activity_count,
                "is_source_two_tower": is_two_tower,
                "is_source_item_cf": is_item_cf,
                "is_source_popularity": is_popularity,
                "cf_x_bayesian": cf_x_bayesian,
                "tt_x_genre": tt_x_genre,
            }

            if use_exp:
                # 7 New High-Fidelity Causal Features (exact parity with Config 8 training)
                source_count = is_two_tower + is_item_cf + is_popularity
                cf_norm = (cf_score - cf_min) / (cf_range + 1e-6) if cf_range > 0 else 0.0
                tt_norm = (tt_score - tt_min) / (tt_range + 1e-6) if tt_range > 0 else 0.0
                score_prod = cf_norm * tt_norm
                calib_diff = bayesian_score - user_mean_rating
                cnt_conf = log_count / (log_count + 1.0)
                rec_overlap = float(np.dot(user_recent_genre, item_genre_vec))
                rec_cosine = float(rec_overlap / (u_rec_norm * item_norm)) if (u_rec_norm > 0 and item_norm > 0) else 0.0

                row.update({
                    "source_count": source_count,
                    "cf_score_normalized": cf_norm,
                    "tt_score_normalized": tt_norm,
                    "score_product": score_prod,
                    "user_rating_calib_diff": calib_diff,
                    "rating_count_confidence": cnt_conf,
                    "recent_genre_cosine": rec_cosine,
                })

            row["movie_idx"] = item_idx
            features.append(row)

        return pd.DataFrame(features)

        return pd.DataFrame(features)
