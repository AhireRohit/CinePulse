import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.logging import logger

GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


def extract_year(title: str) -> tuple[str, int | None]:
    """Extract release year from title e.g. 'Toy Story (1995)' -> ('Toy Story', 1995)."""
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        year = int(match.group(1))
        clean_title = title[:match.start()].strip()
        return clean_title, year
    return title.strip(), None


class MovieLensProcessor:
    def __init__(self, raw_data_dir: Path, output_dir: Path, min_user_ratings: int = 5, min_movie_ratings: int = 5):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.min_user_ratings = min_user_ratings
        self.min_movie_ratings = min_movie_ratings

        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.movie_to_idx: dict[int, int] = {}
        self.idx_to_movie: dict[int, int] = {}
        self.genres: list[str] = GENRE_LIST

    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        # Handle both standard CSV format (ml-latest-small, ml-25m, ml-32m) and .dat (ml-1m)
        movies_csv = self.raw_data_dir / "movies.csv"
        ratings_csv = self.raw_data_dir / "ratings.csv"
        tags_csv = self.raw_data_dir / "tags.csv"

        if movies_csv.exists() and ratings_csv.exists():
            logger.info("Loading CSV formatted MovieLens files...")
            movies_df = pd.read_csv(movies_csv)
            ratings_df = pd.read_csv(ratings_csv)
            tags_df = pd.read_csv(tags_csv) if tags_csv.exists() else None
        else:
            # Try .dat format (ml-1m)
            movies_dat = self.raw_data_dir / "movies.dat"
            ratings_dat = self.raw_data_dir / "ratings.dat"
            if movies_dat.exists() and ratings_dat.exists():
                logger.info("Loading DAT formatted MovieLens files...")
                movies_df = pd.read_csv(
                    movies_dat, sep="::", engine="python", header=None,
                    names=["movieId", "title", "genres"], encoding="latin-1"
                )
                ratings_df = pd.read_csv(
                    ratings_dat, sep="::", engine="python", header=None,
                    names=["userId", "movieId", "rating", "timestamp"], encoding="latin-1"
                )
                tags_df = None
            else:
                raise FileNotFoundError(f"Cannot find movie and rating files in {self.raw_data_dir}")

        return movies_df, ratings_df, tags_df

    def process(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        movies_df, ratings_df, tags_df = self.load_raw_data()

        logger.info(f"Raw counts: {len(ratings_df):,} ratings, {len(movies_df):,} movies, {ratings_df['userId'].nunique():,} users")

        # 1. Clean ratings
        ratings_df = ratings_df.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()
        ratings_df["userId"] = ratings_df["userId"].astype(int)
        ratings_df["movieId"] = ratings_df["movieId"].astype(int)
        ratings_df["rating"] = ratings_df["rating"].astype(float)
        ratings_df["timestamp"] = ratings_df["timestamp"].astype(int)

        # 2. Filter sparse users and movies if needed
        user_counts = ratings_df["userId"].value_counts()
        movie_counts = ratings_df["movieId"].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_ratings].index
        valid_movies = movie_counts[movie_counts >= self.min_movie_ratings].index

        ratings_df = ratings_df[
            ratings_df["userId"].isin(valid_users) & ratings_df["movieId"].isin(valid_movies)
        ].copy()

        # 3. Create continuous index mappings
        unique_users = sorted(ratings_df["userId"].unique())
        unique_movies = sorted(ratings_df["movieId"].unique())

        self.user_to_idx = {int(uid): idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: int(uid) for idx, uid in enumerate(unique_users)}
        self.movie_to_idx = {int(mid): idx for idx, mid in enumerate(unique_movies)}
        self.idx_to_movie = {idx: int(mid) for idx, mid in enumerate(unique_movies)}

        ratings_df["user_idx"] = ratings_df["userId"].map(self.user_to_idx)
        ratings_df["movie_idx"] = ratings_df["movieId"].map(self.movie_to_idx)

        # 4. Clean and parse movie metadata
        movies_df = movies_df[movies_df["movieId"].isin(unique_movies)].copy()
        movies_df["movie_idx"] = movies_df["movieId"].map(self.movie_to_idx)

        parsed_titles = [extract_year(t) for t in movies_df["title"]]
        movies_df["clean_title"] = [p[0] for p in parsed_titles]
        movies_df["release_year"] = [p[1] for p in parsed_titles]
        median_year = int(movies_df["release_year"].median(skipna=True)) if not movies_df["release_year"].isna().all() else 2000
        movies_df["release_year"] = movies_df["release_year"].fillna(median_year).astype(int)

        # Encode genres as multi-hot
        def encode_genres(genre_str: str) -> list[int]:
            if not isinstance(genre_str, str):
                return [0] * len(self.genres)
            g_set = set(genre_str.split("|"))
            return [1 if g in g_set else 0 for g in self.genres]

        genre_matrix = np.array([encode_genres(g) for g in movies_df["genres"]])
        for i, g_name in enumerate(self.genres):
            movies_df[f"genre_{g_name}"] = genre_matrix[:, i]

        # Aggregate movie rating statistics
        movie_stats = ratings_df.groupby("movie_idx").agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean"),
            rating_std=("rating", "std"),
        ).reset_index()
        movie_stats["rating_std"] = movie_stats["rating_std"].fillna(0.0)

        # Global Bayesian smoothing
        C = ratings_df["rating"].mean()
        m = 10.0  # prior weight
        movie_stats["bayesian_rating"] = (
            (movie_stats["rating_count"] * movie_stats["rating_mean"] + m * C) /
            (movie_stats["rating_count"] + m)
        )

        movies_df = movies_df.merge(movie_stats, on="movie_idx", how="left")
        movies_df["rating_count"] = movies_df["rating_count"].fillna(0).astype(int)
        movies_df["rating_mean"] = movies_df["rating_mean"].fillna(C).astype(float)
        movies_df["bayesian_rating"] = movies_df["bayesian_rating"].fillna(C).astype(float)

        # 5. Temporal Train / Validation / Test Split (Leave-Last-K-Out)
        logger.info("Performing temporal split (leave-last-2-out for val and test)...")
        ratings_df = ratings_df.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

        # Assign rank within user sequence: rank=1 is newest
        ratings_df["rev_rank"] = ratings_df.groupby("user_idx").cumcount(ascending=False) + 1

        test_df = ratings_df[ratings_df["rev_rank"] == 1].copy().drop(columns=["rev_rank"])
        val_df = ratings_df[ratings_df["rev_rank"] == 2].copy().drop(columns=["rev_rank"])
        train_df = ratings_df[ratings_df["rev_rank"] > 2].copy().drop(columns=["rev_rank"])

        logger.info(
            f"Split sizes: Train={len(train_df):,} ({len(train_df)/len(ratings_df):.1%}), "
            f"Val={len(val_df):,} ({len(val_df)/len(ratings_df):.1%}), "
            f"Test={len(test_df):,} ({len(test_df)/len(ratings_df):.1%})"
        )

        # 6. Build user sequence history from train split (prevent leakage!)
        # NOTE: We use ALL interactions (including low-rated ones) to build the full
        # interaction history list (used for exclusion / seen-item filtering).
        # However, we also save a separate "positive-only" history (rating >= 3.5)
        # used by the Two-Tower user tower history encoder so it encodes genuine
        # preferences rather than a mix of likes and explicit dislikes.
        user_histories = (
            train_df.sort_values(["user_idx", "timestamp"])
            .groupby("user_idx")["movie_idx"]
            .apply(list)
            .to_dict()
        )

        pos_train_df = train_df[train_df["rating"] >= 3.5]
        user_positive_histories = (
            pos_train_df.sort_values(["user_idx", "timestamp"])
            .groupby("user_idx")["movie_idx"]
            .apply(list)
            .to_dict()
        )
        # Fall back to full history for users who have no positives above threshold
        for uid, full_hist in user_histories.items():
            if uid not in user_positive_histories:
                user_positive_histories[uid] = full_hist

        # 7. Save Parquet and JSON artifacts
        logger.info(f"Saving processed artifacts to {self.output_dir}...")
        movies_df.to_parquet(self.output_dir / "movies.parquet", index=False)
        train_df.to_parquet(self.output_dir / "ratings_train.parquet", index=False)
        val_df.to_parquet(self.output_dir / "ratings_val.parquet", index=False)
        test_df.to_parquet(self.output_dir / "ratings_test.parquet", index=False)

        # Save user histories as JSON
        # Full history (all ratings) used for seen-item exclusion in retrieval
        with open(self.output_dir / "user_train_histories.json", "w") as f:
            json.dump({str(k): v for k, v in user_histories.items()}, f)
        # Positive-only history (rating >= 3.5) used by Two-Tower user tower encoder
        with open(self.output_dir / "user_positive_histories.json", "w") as f:
            json.dump({str(k): v for k, v in user_positive_histories.items()}, f)

        # Save ID mappings
        mappings = {
            "user_to_idx": self.user_to_idx,
            "movie_to_idx": self.movie_to_idx,
            "idx_to_user": self.idx_to_user,
            "idx_to_movie": self.idx_to_movie,
            "genres": self.genres,
            "num_users": len(unique_users),
            "num_movies": len(unique_movies),
            "global_rating_mean": float(C),
        }
        with open(self.output_dir / "mappings.json", "w") as f:
            json.dump(mappings, f)

        logger.info("Processing complete.")
        return {
            "num_users": len(unique_users),
            "num_movies": len(unique_movies),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "positive_history_users": len(user_positive_histories),
        }


if __name__ == "__main__":
    processor = MovieLensProcessor(
        raw_data_dir=Path("data/raw/ml-latest-small"),
        output_dir=Path("data/processed"),
    )
    processor.process()
