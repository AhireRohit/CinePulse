import argparse
from pathlib import Path

from src.core.logging import logger
from src.data.download import download_and_extract
from src.data.processor import MovieLensProcessor


def main():
    parser = argparse.ArgumentParser(description="Prepare MovieLens Dataset for CinePulse Recommender")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-1m",
        choices=["ml-latest-small", "ml-1m", "ml-25m", "ml-32m"],
        help="MovieLens dataset to download and process",
    )
    parser.add_argument("--min-user-ratings", type=int, default=5, help="Minimum ratings per user")
    parser.add_argument("--min-movie-ratings", type=int, default=5, help="Minimum ratings per movie")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for processed parquet files")
    args = parser.parse_args()

    logger.info(f"=== Starting Data Pipeline for {args.dataset} ===")

    # 1. Download and extract
    raw_dir = download_and_extract(dataset_name=args.dataset, target_dir=Path("data/raw"))

    # 2. Process, clean, create continuous mappings, and split
    processor = MovieLensProcessor(
        raw_data_dir=raw_dir,
        output_dir=Path(args.output_dir),
        min_user_ratings=args.min_user_ratings,
        min_movie_ratings=args.min_movie_ratings,
    )
    stats = processor.process()

    logger.info("=== Data Preparation Finished Successfully! ===")
    logger.info(f"Users: {stats['num_users']:,} | Movies: {stats['num_movies']:,}")
    logger.info(f"Train samples: {stats['train_size']:,} | Val: {stats['val_size']:,} | Test: {stats['test_size']:,}")


if __name__ == "__main__":
    main()
