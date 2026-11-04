import numpy as np
import pandas as pd


class MMRReRanker:
    """
    Maximal Marginal Relevance (MMR) Re-Ranker with Novelty/Long-Tail Exploration.
    Balances predicted relevance, catalog diversity, and novelty.
    Uses calibrated sigmoid affinity normalization to eliminate artificial 100% match inflation.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        genre_cols: list[str],
        default_lambda: float = 0.85,
        default_novelty_weight: float = 0.03,
    ):
        self.movies_df = movies_df.set_index("movie_idx") if "movie_idx" in movies_df.columns else movies_df
        self.genre_cols = genre_cols
        self.default_lambda = default_lambda
        self.default_novelty_weight = default_novelty_weight

        # Precompute normalized item genre vectors for fast similarity computation
        genre_matrix = self.movies_df[self.genre_cols].values.astype(np.float32)
        norms = np.linalg.norm(genre_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normalized_genres = genre_matrix / norms

        # Precompute novelty score (inverse log-frequency in [0, 1])
        counts = self.movies_df["rating_count"].values.astype(np.float32)
        log_counts = np.log1p(counts)
        max_log = log_counts.max() if log_counts.max() > 0 else 1.0
        # High count -> low novelty, low count -> high novelty
        self.novelty_scores = 1.0 - (log_counts / max_log)

    def rerank(
        self,
        candidate_scores: list[tuple[int, float]],
        top_k: int = 20,
        lambda_param: float | None = None,
        novelty_weight: float | None = None,
        excluded_item_indices: set[int] | None = None,
    ) -> list[dict]:
        """
        Executes MMR greedy selection over ranked candidate pairs.
        candidate_scores: list of (movie_idx, relevance_score)
        returns list of dicts with movie_idx, final_score, relevance_score, novelty_score, diversity_penalty
        """
        if not candidate_scores:
            return []

        lam = self.default_lambda if lambda_param is None else lambda_param
        nov_wt = self.default_novelty_weight if novelty_weight is None else novelty_weight
        excluded = excluded_item_indices or set()

        # Filter out excluded items
        valid_candidates = [
            (idx, score) for idx, score in candidate_scores
            if idx not in excluded and idx < len(self.normalized_genres)
        ]

        if not valid_candidates:
            return []

        # Calibrated Sigmoid Normalization (Prevents artificial 100% Match inflation)
        raw_scores = np.array([score for _, score in valid_candidates], dtype=np.float32)
        mean_s = float(np.mean(raw_scores))
        std_s = float(np.std(raw_scores))
        std_s = max(std_s, 0.25)

        # Standardized sigmoid mapping to (0, 1)
        z_scores = (raw_scores - mean_s) / std_s
        norm_scores = 1.0 / (1.0 + np.exp(-z_scores))

        # Calibrated affinity percentage for UI presentation (bounded in realistic 70% - 95% range)
        affinity_percentages = np.clip(np.round(70.0 + 25.0 * norm_scores), 70, 96).astype(int)

        cand_indices = np.array([idx for idx, _ in valid_candidates], dtype=np.int64)
        n_cands = len(cand_indices)
        cand_vecs = self.normalized_genres[cand_indices]  # [N, num_genres]
        cand_novelty = self.novelty_scores[cand_indices]  # [N]

        # Vectorized incremental MMR
        max_sim = np.zeros(n_cands, dtype=np.float32)
        is_selected = np.zeros(n_cands, dtype=bool)
        selected_results: list[dict] = []

        num_to_select = min(top_k, n_cands)
        for step in range(num_to_select):
            if step == 0:
                mmr_scores = lam * norm_scores + nov_wt * cand_novelty
            else:
                mmr_scores = lam * norm_scores - (1.0 - lam) * max_sim + nov_wt * cand_novelty

            mmr_scores[is_selected] = -np.inf
            best_pos = int(np.argmax(mmr_scores))
            if mmr_scores[best_pos] == -np.inf:
                break

            is_selected[best_pos] = True
            best_item = int(cand_indices[best_pos])
            best_sim = float(max_sim[best_pos])

            selected_results.append({
                "movie_idx": best_item,
                "final_score": float(mmr_scores[best_pos]),
                "relevance_score": float(norm_scores[best_pos]),
                "match_percentage": int(affinity_percentages[best_pos]),
                "raw_relevance": float(raw_scores[best_pos]),
                "diversity_penalty": best_sim,
                "novelty_score": float(cand_novelty[best_pos]),
            })

            # Vectorized update of running maximum similarity to any selected item
            sims_to_new = np.dot(cand_vecs, cand_vecs[best_pos])
            max_sim = np.maximum(max_sim, sims_to_new)

        return selected_results
