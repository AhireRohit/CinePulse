import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieLensRetrievalDataset(Dataset):
    """
    High-performance Dataset for training Two-Tower retrieval model.
    Constructs strictly causal prefix interaction sequences:
    - Target is strictly chronologically subsequent to prefix history.
    - Zero target leakage (target item is NEVER in prefix history).
    - Zero future leakage (history contains only prior timestamps).
    - Positive-only thresholding (r >= 4.0).
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        user_histories: dict[str | int, list[int]] | None = None,
        max_seq_len: int = 20,
        genre_cols: list[str] | None = None,
        causal: bool = True,
        min_rating: float = 4.0,
    ):
        self.max_seq_len = max_seq_len
        self.num_movies = len(movies_df)

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

        movie_genres_np = movies_df[self.genre_cols].values.astype(np.float32)

        if causal:
            # Construct causal prefix sequences from sorted positive interactions
            pos_df = ratings_df[ratings_df["rating"] >= min_rating].sort_values(
                ["user_idx", "timestamp"]
            ).reset_index(drop=True)

            u_list = []
            tgt_list = []
            hist_list = []
            hlen_list = []
            gpref_list = []
            r_list = []

            for u, group in pos_df.groupby("user_idx"):
                m_arr = group["movie_idx"].values
                r_arr = group["rating"].values
                n_items = len(m_arr)
                if n_items < 2:
                    continue

                for k in range(1, n_items):
                    target = m_arr[k]
                    prefix = m_arr[max(0, k - max_seq_len):k]
                    h_len = len(prefix)

                    padded = np.full(max_seq_len, self.num_movies, dtype=np.int64)
                    padded[-h_len:] = prefix

                    valid_p = prefix[prefix < self.num_movies]
                    g_pref = (
                        movie_genres_np[valid_p].mean(axis=0)
                        if len(valid_p) > 0
                        else np.zeros(len(self.genre_cols), dtype=np.float32)
                    )

                    u_list.append(u)
                    tgt_list.append(target)
                    hist_list.append(padded)
                    hlen_list.append(h_len)
                    gpref_list.append(g_pref)
                    r_list.append(r_arr[k])

            self.user_indices = np.array(u_list, dtype=np.int64)
            self.target_indices = np.array(tgt_list, dtype=np.int64)
            self.history_seqs = np.array(hist_list, dtype=np.int64)
            self.history_lens = np.array(hlen_list, dtype=np.int64)
            self.genre_prefs = np.array(gpref_list, dtype=np.float32)
            self.rating_vals = np.array(r_list, dtype=np.float32)
            self.length = len(self.user_indices)

        else:
            # Non-causal evaluation mode (validation set where history comes from train)
            pos_df = ratings_df[ratings_df["rating"] >= min_rating].reset_index(drop=True)
            u_list = []
            tgt_list = []
            hist_list = []
            hlen_list = []
            gpref_list = []
            r_list = []

            for _, row in pos_df.iterrows():
                u = int(row["user_idx"])
                target = int(row["movie_idx"])
                r = float(row["rating"])

                hist = user_histories.get(u, []) if user_histories else []
                # Clean history strictly of target if somehow present
                hist_clean = [m for m in hist if m != target]
                h_len = min(len(hist_clean), max_seq_len)

                padded = np.full(max_seq_len, self.num_movies, dtype=np.int64)
                if h_len > 0:
                    padded[-h_len:] = hist_clean[-h_len:]

                valid_p = [m for m in hist_clean if m < self.num_movies]
                g_pref = (
                    movie_genres_np[valid_p].mean(axis=0)
                    if valid_p
                    else np.zeros(len(self.genre_cols), dtype=np.float32)
                )

                u_list.append(u)
                tgt_list.append(target)
                hist_list.append(padded)
                hlen_list.append(h_len)
                gpref_list.append(g_pref)
                r_list.append(r)

            self.user_indices = np.array(u_list, dtype=np.int64)
            self.target_indices = np.array(tgt_list, dtype=np.int64)
            self.history_seqs = np.array(hist_list, dtype=np.int64)
            self.history_lens = np.array(hlen_list, dtype=np.int64)
            self.genre_prefs = np.array(gpref_list, dtype=np.float32)
            self.rating_vals = np.array(r_list, dtype=np.float32)
            self.length = len(self.user_indices)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "user_idx": torch.tensor(self.user_indices[idx], dtype=torch.long),
            "target_movie_idx": torch.tensor(self.target_indices[idx], dtype=torch.long),
            "history_seq": torch.tensor(self.history_seqs[idx], dtype=torch.long),
            "history_len": torch.tensor(self.history_lens[idx], dtype=torch.long),
            "user_genre_pref": torch.tensor(self.genre_prefs[idx], dtype=torch.float32),
            "target_movie_genres": self.movie_genres[self.target_indices[idx]],
            "target_movie_dense": self.movie_dense_features[self.target_indices[idx]],
            "rating": torch.tensor(self.rating_vals[idx], dtype=torch.float32),
        }
