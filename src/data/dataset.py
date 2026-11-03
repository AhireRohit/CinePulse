import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieLensRetrievalDataset(Dataset):
    """
    High-performance Dataset for training Two-Tower retrieval model.
    Uses pre-extracted numpy arrays for sub-microsecond indexing.
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        user_histories: dict[str | int, list[int]],
        max_seq_len: int = 20,
        genre_cols: list[str] | None = None,
    ):
        self.max_seq_len = max_seq_len
        self.num_movies = len(movies_df)

        # Fast flat numpy arrays for ratings
        self.user_indices = ratings_df["user_idx"].to_numpy(dtype=np.int64)
        self.target_indices = ratings_df["movie_idx"].to_numpy(dtype=np.int64)
        self.rating_vals = ratings_df["rating"].to_numpy(dtype=np.float32)
        self.length = len(self.user_indices)

        if genre_cols is None:
            genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]
        self.genre_cols = genre_cols

        # Movie genre matrix [M, num_genres]
        self.movie_genres = torch.tensor(
            movies_df[self.genre_cols].values, dtype=torch.float32
        )

        # Normalize release year and log popularity
        years = movies_df["release_year"].values.astype(np.float32)
        norm_years = (years - 1920.0) / (2025.0 - 1920.0)
        norm_years = np.clip(norm_years, 0.0, 1.0)

        log_counts = np.log1p(movies_df["rating_count"].values.astype(np.float32))
        max_count = log_counts.max() if log_counts.max() > 0 else 1.0
        norm_counts = log_counts / max_count

        norm_ratings = (movies_df["rating_mean"].values.astype(np.float32) - 1.0) / 4.0
        norm_ratings = np.clip(norm_ratings, 0.0, 1.0)

        # [M, 3]: year, count, mean rating
        self.movie_dense_features = torch.tensor(
            np.stack([norm_years, norm_counts, norm_ratings], axis=1),
            dtype=torch.float32,
        )

        # Precompute user history sequences as padded numpy arrays [num_users, max_seq_len]
        max_uid = int(self.user_indices.max()) + 1
        num_genres = len(self.genre_cols)

        self.padded_user_histories = np.zeros((max_uid, max_seq_len), dtype=np.int64)
        self.user_history_lens = np.zeros(max_uid, dtype=np.int64)
        self.user_genre_prefs = np.zeros((max_uid, num_genres), dtype=np.float32)

        genre_np = movies_df[self.genre_cols].values.astype(np.float32)

        for uid_raw, hist in user_histories.items():
            uid = int(uid_raw)
            if uid < max_uid:
                hist_arr = np.array(hist, dtype=np.int64)
                hist_len = min(len(hist_arr), max_seq_len)
                self.user_history_lens[uid] = hist_len
                if hist_len > 0:
                    self.padded_user_histories[uid, -hist_len:] = hist_arr[-hist_len:]
                    valid_items = hist_arr[hist_arr < self.num_movies]
                    if len(valid_items) > 0:
                        self.user_genre_prefs[uid] = genre_np[valid_items].mean(axis=0)

        self.padded_user_histories_tensor = torch.tensor(self.padded_user_histories, dtype=torch.long)
        self.user_history_lens_tensor = torch.tensor(self.user_history_lens, dtype=torch.long)
        self.user_genre_prefs_tensor = torch.tensor(self.user_genre_prefs, dtype=torch.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_idx = int(self.user_indices[idx])
        target_movie_idx = int(self.target_indices[idx])
        rating = float(self.rating_vals[idx])

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "target_movie_idx": torch.tensor(target_movie_idx, dtype=torch.long),
            "history_seq": self.padded_user_histories_tensor[user_idx],
            "history_len": self.user_history_lens_tensor[user_idx],
            "user_genre_pref": self.user_genre_prefs_tensor[user_idx],
            "target_movie_genres": self.movie_genres[target_movie_idx],
            "target_movie_dense": self.movie_dense_features[target_movie_idx],
            "rating": torch.tensor(rating, dtype=torch.float32),
        }
