import urllib.request
import zipfile
from pathlib import Path

from src.core.logging import logger

DATASET_URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
}

EXPECTED_FILES = {
    "ml-latest-small": ["movies.csv", "ratings.csv", "tags.csv", "links.csv"],
    "ml-1m": ["movies.dat", "ratings.dat", "users.dat"],
    "ml-25m": ["movies.csv", "ratings.csv", "tags.csv", "links.csv"],
    "ml-32m": ["movies.csv", "ratings.csv", "tags.csv", "links.csv"],
}


def download_and_extract(dataset_name: str = "ml-latest-small", target_dir: Path | None = None) -> Path:
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: {list(DATASET_URLS.keys())}")

    if target_dir is None:
        target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = target_dir / dataset_name
    expected = EXPECTED_FILES.get(dataset_name, [])

    # Check if already downloaded and extracted
    if extracted_dir.exists() and all((extracted_dir / f).exists() for f in expected):
        logger.info(f"Dataset '{dataset_name}' already exists at {extracted_dir}. Skipping download.")
        return extracted_dir

    url = DATASET_URLS[dataset_name]
    zip_path = target_dir / f"{dataset_name}.zip"

    logger.info(f"Downloading {dataset_name} from {url}...")

    def reporthook(count, block_size, total_size):
        if total_size > 0 and count % 200 == 0:
            percent = min(100.0, count * block_size * 100.0 / total_size)
            logger.info(f"Download progress: {percent:.1f}% ({count * block_size / 1024 / 1024:.1f} MB)")

    urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
    logger.info(f"Downloaded to {zip_path}. File size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")

    logger.info(f"Extracting archive to {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    # Some zips extract into a subfolder with matching name
    if not (extracted_dir / expected[0]).exists():
        # Check subdirectories
        for item in target_dir.iterdir():
            if item.is_dir() and item.name.startswith(dataset_name):
                extracted_dir = item
                break

    logger.info(f"Extraction complete. Target directory: {extracted_dir}")
    return extracted_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download MovieLens dataset")
    parser.add_argument("--dataset", type=str, default="ml-latest-small", choices=list(DATASET_URLS.keys()))
    args = parser.parse_args()
    download_and_extract(args.dataset)
