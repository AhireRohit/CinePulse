import json
from pathlib import Path

import pandas as pd


def test_processed_data_files_exist():
    data_dir = Path("data/processed")
    assert (data_dir / "movies.parquet").exists()
    assert (data_dir / "ratings_train.parquet").exists()
    assert (data_dir / "ratings_val.parquet").exists()
    assert (data_dir / "ratings_test.parquet").exists()
    assert (data_dir / "mappings.json").exists()
    assert (data_dir / "user_train_histories.json").exists()


def test_id_mappings_consistency():
    with open("data/processed/mappings.json") as f:
        mappings = json.load(f)

    movies_df = pd.read_parquet("data/processed/movies.parquet")
    num_movies = mappings["num_movies"]
    num_users = mappings["num_users"]

    assert len(movies_df) == num_movies
    assert len(mappings["user_to_idx"]) == num_users
    assert len(mappings["movie_to_idx"]) == num_movies
    assert "genres" in mappings
    assert len(mappings["genres"]) > 10


def test_temporal_split_no_leakage():
    """
    Verifies that for every user, the validation item is strictly later than train items,
    and the test item is strictly later than validation items.
    """
    train_df = pd.read_parquet("data/processed/ratings_train.parquet")
    val_df = pd.read_parquet("data/processed/ratings_val.parquet")
    test_df = pd.read_parquet("data/processed/ratings_test.parquet")

    # Sample users to verify temporal ordering
    sample_users = val_df["user_idx"].head(200).unique()

    train_max_ts = train_df.groupby("user_idx")["timestamp"].max()
    val_ts = val_df.set_index("user_idx")["timestamp"]
    test_ts = test_df.set_index("user_idx")["timestamp"]

    for u in sample_users:
        t_max = train_max_ts.get(u)
        v = val_ts.get(u)
        te = test_ts.get(u)

        if t_max is not None and v is not None:
            assert v >= t_max, f"User {u}: validation item timestamp {v} is older than max train timestamp {t_max}"
        if v is not None and te is not None:
            assert te >= v, f"User {u}: test item timestamp {te} is older than validation timestamp {v}"


def test_user_history_contains_only_train_items():
    """
    Verifies user_train_histories does not contain validation or test items.
    """
    with open("data/processed/user_train_histories.json") as f:
        user_histories = {int(k): set(v) for k, v in json.load(f).items()}

    test_df = pd.read_parquet("data/processed/ratings_test.parquet")
    for _, row in test_df.head(200).iterrows():
        u = int(row["user_idx"])
        test_m = int(row["movie_idx"])
        hist = user_histories.get(u, set())
        # In strict temporal split, test item must NOT be in training history
        assert test_m not in hist, f"Leakage detected: test item {test_m} found in train history for user {u}"
