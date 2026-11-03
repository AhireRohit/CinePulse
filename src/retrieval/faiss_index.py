from pathlib import Path

import faiss
import numpy as np

from src.core.logging import logger


class FaissCandidateIndex:
    """
    FAISS Approximate Nearest Neighbor (ANN) index for fast candidate retrieval
    using Inner Product (equivalent to Cosine Similarity for L2-normalized vectors).
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None
        self.idx_to_movie_idx: np.ndarray | None = None
        self.num_items: int = 0

    def build(self, item_embeddings: np.ndarray, movie_indices: np.ndarray | None = None):
        """
        Builds the FAISS index from an array of item embeddings.
        item_embeddings: [N, D] float32 numpy array
        movie_indices: [N] array of dataset movie indices
        """
        assert item_embeddings.ndim == 2, "item_embeddings must be 2D array [N, D]"
        assert item_embeddings.shape[1] == self.dimension, f"Expected dim {self.dimension}, got {item_embeddings.shape[1]}"

        # Ensure float32 and contiguous
        embeddings = np.ascontiguousarray(item_embeddings, dtype=np.float32)

        # Ensure L2 normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        self.num_items = embeddings.shape[0]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        if movie_indices is not None:
            self.idx_to_movie_idx = np.asarray(movie_indices, dtype=np.int64)
        else:
            self.idx_to_movie_idx = np.arange(self.num_items, dtype=np.int64)

        logger.info(f"Built FAISS IndexFlatIP with {self.num_items} items (dim={self.dimension}).")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
        exclude_item_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Searches the index with a single query embedding.
        query_embedding: [D] or [1, D] float32 array
        """
        if self.index is None:
            raise RuntimeError("FAISS index has not been built or loaded.")

        q_vec = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Fetch extra candidates to account for excluded items
        fetch_k = top_k
        if exclude_item_indices:
            fetch_k = min(self.num_items, top_k + len(exclude_item_indices))

        scores, indices = self.index.search(q_vec, fetch_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for rank_idx, faiss_idx in enumerate(indices):
            if faiss_idx < 0:
                continue
            movie_idx = int(self.idx_to_movie_idx[faiss_idx]) if self.idx_to_movie_idx is not None else int(faiss_idx)
            if exclude_item_indices and movie_idx in exclude_item_indices:
                continue
            results.append((movie_idx, float(scores[rank_idx])))
            if len(results) >= top_k:
                break

        return results

    def save(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            index_file = directory / "item_index.faiss"
            faiss.write_index(self.index, str(index_file))
            if self.idx_to_movie_idx is not None:
                np.save(directory / "faiss_movie_ids.npy", self.idx_to_movie_idx)
            logger.info(f"FAISS index saved to {directory}")

    def load(self, directory: Path):
        directory = Path(directory)
        index_file = directory / "item_index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found at {index_file}")

        self.index = faiss.read_index(str(index_file))
        id_file = directory / "faiss_movie_ids.npy"
        if id_file.exists():
            self.idx_to_movie_idx = np.load(id_file)
        else:
            self.idx_to_movie_idx = np.arange(self.index.ntotal, dtype=np.int64)
        self.num_items = self.index.ntotal
        self.dimension = self.index.d
        logger.info(f"Loaded FAISS index with {self.num_items} items (dim={self.dimension})")
