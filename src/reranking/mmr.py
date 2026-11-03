import numpy as np
import pandas as pd


class MMRReRanker:
    """
    Maximal Marginal Relevance (MMR) Re-Ranker with Novelty/Long-Tail Exploration.
    Balances predicted relevance, catalog diversity, and novelty.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        genre_cols: list[str],
        default_lambda: float = 0.65,
        default_novelty_weight: float = 0.15,
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

        # Normalize relevance scores to [0, 1]
        raw_scores = np.array([score for _, score in valid_candidates], dtype=np.float32)
        min_s, max_s = float(raw_scores.min()), float(raw_scores.max())
        if min_s >= 0.0 and max_s <= 1.0:
            norm_scores = raw_scores
        elif max_s > min_s:
            norm_scores = (raw_scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.ones_like(raw_scores)

        candidate_map = {
            item_idx: {
                "norm_rel": float(norm_scores[i]),
                "raw_rel": float(score),
                "novelty": float(self.novelty_scores[item_idx]),
            }
            for i, (item_idx, score) in enumerate(valid_candidates)
        }

        unselected = set(candidate_map.keys())
        selected: list[int] = []
        selected_results: list[dict] = []

        while len(selected) < top_k and unselected:
            best_item = None
            best_mmr_score = -float("inf")
            best_metrics = {}

            for item_idx in unselected:
                rel = candidate_map[item_idx]["norm_rel"]
                novelty = candidate_map[item_idx]["novelty"]

                if not selected:
                    max_sim = 0.0
                else:
                    item_vec = self.normalized_genres[item_idx]
                    sel_vecs = self.normalized_genres[selected]
                    sims = np.dot(sel_vecs, item_vec)
                    max_sim = float(np.max(sims))

                # MMR formulation with novelty exploration
                mmr_score = lam * rel - (1.0 - lam) * max_sim + nov_wt * novelty

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_item = item_idx
                    best_metrics = {
                        "final_score": float(mmr_score),
                        "relevance_score": float(rel),
                        "raw_relevance": candidate_map[item_idx]["raw_rel"],
                        "diversity_penalty": float(max_sim),
                        "novelty_score": float(novelty),
                    }

            if best_item is None:
                break

            selected.append(best_item)
            unselected.remove(best_item)
            selected_results.append({
                "movie_idx": best_item,
                **best_metrics,
            })

        return selected_results
