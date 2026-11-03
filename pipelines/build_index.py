import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.logging import logger
from src.retrieval.faiss_index import FaissCandidateIndex


def build_faiss_index(
    embeddings_file: Path = Path("artifacts/models/item_embeddings.npy"),
    movies_file: Path = Path("data/processed/movies.parquet"),
    output_dir: Path = Path("artifacts/indices"),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not embeddings_file.exists():
        raise FileNotFoundError(f"Item embeddings not found at {embeddings_file}. Run train_retrieval first.")

    logger.info(f"Loading item embeddings from {embeddings_file}...")
    item_embeddings = np.load(embeddings_file)
    movies_df = pd.read_parquet(movies_file)

    num_items, dim = item_embeddings.shape
    logger.info(f"Building FAISS index for {num_items} items (dim={dim})...")

    index = FaissCandidateIndex(dimension=dim)
    movie_indices = movies_df["movie_idx"].values
    index.build(item_embeddings=item_embeddings, movie_indices=movie_indices)

    index.save(output_dir)
    logger.info(f"FAISS index successfully saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS Candidate Retrieval Index")
    parser.add_argument("--embeddings-file", type=str, default="artifacts/models/item_embeddings.npy")
    parser.add_argument("--output-dir", type=str, default="artifacts/indices")
    args = parser.parse_args()

    build_faiss_index(
        embeddings_file=Path(args.embeddings_file),
        output_dir=Path(args.output_dir),
    )
