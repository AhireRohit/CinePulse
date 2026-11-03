import time

import numpy as np
import pandas as pd
import torch

from src.core.config import settings
from src.core.logging import logger
from src.models.baselines import (
    ContentRecommender,
    ItemCollaborativeFiltering,
    PopularityRecommender,
)
from src.models.ranker import LightGBMRanker
from src.models.two_tower import TwoTowerRecommender
from src.ranking.feature_engineering import RankingFeatureExtractor
from src.reranking.mmr import MMRReRanker
from src.retrieval.faiss_index import FaissCandidateIndex


class RecommendationEngine:
    """
    Production Multi-Stage Recommendation Engine.
    Coordinates Candidate Retrieval -> Personalized Ranking -> MMR Diversity Re-Ranking -> Explainability.
    Supports real-time session updates and cold-start onboarding.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        mappings: dict,
        genre_cols: list[str],
        user_histories: dict[int, list[int]],
        device: torch.device | None = None,
    ):
        self.movies_df = movies_df
        self.mappings = mappings
        self.genre_cols = genre_cols
        self.user_histories = user_histories
        self.num_users = mappings["num_users"]
        self.num_movies = mappings["num_movies"]

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not settings.force_cpu else "cpu")
        else:
            self.device = device

        logger.info(f"Initializing RecommendationEngine on device: {self.device}")

        # In-memory session tracking: user_id -> dict(likes, dislikes, dismissals, session_history)
        self.user_sessions: dict[int, dict] = {}

        # Component models
        self.popularity_recommender = PopularityRecommender()
        self.item_cf = ItemCollaborativeFiltering()
        self.content_recommender = ContentRecommender()
        self.two_tower_model: TwoTowerRecommender | None = None
        self.faiss_index = FaissCandidateIndex(dimension=settings.embedding_dim)
        self.ranker = LightGBMRanker()
        self.feature_extractor = RankingFeatureExtractor(
            movies_df=self.movies_df,
            genre_cols=self.genre_cols,
            global_mean_rating=mappings.get("global_rating_mean", 3.5),
        )
        self.reranker = MMRReRanker(
            movies_df=self.movies_df,
            genre_cols=self.genre_cols,
            default_lambda=settings.default_mmr_lambda,
            default_novelty_weight=settings.novelty_weight,
        )

        # Precomputed movie item vectors
        self.item_embeddings: np.ndarray | None = None

    def get_user_session(self, user_idx: int) -> dict:
        if user_idx not in self.user_sessions:
            # Initialize from historical train data
            hist = list(self.user_histories.get(user_idx, []))
            self.user_sessions[user_idx] = {
                "history": hist,
                "likes": set(),
                "dislikes": set(),
                "saves": set(),
                "dismissals": set(),
                "ratings": {},
            }
        return self.user_sessions[user_idx]

    def record_feedback(self, user_idx: int, movie_idx: int, action: str, rating: float | None = None):
        """
        Records instant user feedback: 'like', 'dislike', 'save', 'dismiss', 'rate'.
        Updates session state in real time for dynamic adaptation.
        """
        session = self.get_user_session(user_idx)
        if action == "like":
            session["likes"].add(movie_idx)
            if movie_idx not in session["history"]:
                session["history"].append(movie_idx)
        elif action == "dislike":
            session["dislikes"].add(movie_idx)
            session["dismissals"].add(movie_idx)
        elif action == "save":
            session["saves"].add(movie_idx)
            if movie_idx not in session["history"]:
                session["history"].append(movie_idx)
        elif action == "dismiss":
            session["dismissals"].add(movie_idx)
        elif action == "rate" and rating is not None:
            session["ratings"][movie_idx] = rating
            if rating >= 3.5:
                session["likes"].add(movie_idx)
            else:
                session["dislikes"].add(movie_idx)
                session["dismissals"].add(movie_idx)
            if movie_idx not in session["history"]:
                session["history"].append(movie_idx)

        logger.info(f"User {user_idx} recorded feedback '{action}' on movie {movie_idx}.")

    def generate_user_vector(self, user_idx: int, history: list[int]) -> np.ndarray:
        """
        Generates user embedding vector for candidate retrieval.
        Supports cold-start / new users using dynamic genre preference and recent interactions.
        """
        if self.two_tower_model is None:
            # Fallback to genre-based vector
            return np.zeros(settings.embedding_dim, dtype=np.float32)

        self.two_tower_model.eval()
        with torch.no_grad():
            # Pad history
            padded_hist = np.zeros(settings.max_sequence_length, dtype=np.int64)
            hist_len = min(len(history), settings.max_sequence_length)
            if hist_len > 0:
                padded_hist[-hist_len:] = history[-hist_len:]

            # Genre preference
            if hist_len > 0:
                valid_m = [m for m in history if m < len(self.movies_df)]
                if valid_m:
                    genre_vals = self.movies_df.iloc[valid_m][self.genre_cols].values
                    genre_pref = torch.tensor(genre_vals.mean(axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    genre_pref = torch.zeros((1, len(self.genre_cols)), dtype=torch.float32, device=self.device)
            else:
                genre_pref = torch.zeros((1, len(self.genre_cols)), dtype=torch.float32, device=self.device)

            u_idx_tensor = torch.tensor([min(self.num_users, user_idx)], dtype=torch.long, device=self.device)
            h_seq_tensor = torch.tensor(padded_hist, dtype=torch.long, device=self.device).unsqueeze(0)
            h_len_tensor = torch.tensor([hist_len], dtype=torch.long, device=self.device)

            user_emb = self.two_tower_model.user_tower(
                user_idx=u_idx_tensor,
                history_seq=h_seq_tensor,
                history_len=h_len_tensor,
                user_genre_pref=genre_pref,
            )
            return user_emb.cpu().numpy()[0]

    def recommend(
        self,
        user_idx: int,
        top_k: int = 20,
        retrieval_k: int = 100,
        lambda_param: float | None = None,
        novelty_weight: float | None = None,
        force_category: str | None = None,
    ) -> dict:
        """
        Full End-to-End Recommendation Pipeline Execution:
        1. Retrieval: Two-Tower (ANN) + Item-CF + Popularity
        2. Ranking: LightGBM personalized scoring
        3. Re-ranking: MMR diversity + novelty + deduplication
        4. Explainability: rationale generation
        """
        start_time = time.perf_counter()

        session = self.get_user_session(user_idx)
        history = session["history"]
        excluded_items = set(history) | session["dismissals"]

        # Step 1: Candidate Retrieval
        candidate_set: dict[int, dict] = {}
        retrieval_times = {}

        # 1.1 Two-Tower ANN Retrieval
        t0 = time.perf_counter()
        user_vector = self.generate_user_vector(user_idx, history)
        two_tower_candidates = self.faiss_index.search(
            query_embedding=user_vector,
            top_k=retrieval_k,
            exclude_item_indices=excluded_items,
        )
        for rank_pos, (item_idx, score) in enumerate(two_tower_candidates):
            candidate_set[item_idx] = {
                "source": "two_tower",
                "two_tower_score": float(score),
                "cf_score": 0.0,
                "candidate_rank": rank_pos,
            }
        retrieval_times["two_tower_ms"] = (time.perf_counter() - t0) * 1000.0

        # 1.2 Item-CF Retrieval (Recent items)
        t0 = time.perf_counter()
        if history:
            cf_candidates = self.item_cf.recommend(history[-5:], top_k=50)
            for rank_pos, (item_idx, score) in enumerate(cf_candidates):
                if item_idx not in excluded_items:
                    if item_idx in candidate_set:
                        candidate_set[item_idx]["cf_score"] = float(score)
                    else:
                        candidate_set[item_idx] = {
                            "source": "item_cf",
                            "two_tower_score": 0.0,
                            "cf_score": float(score),
                            "candidate_rank": len(candidate_set),
                        }
        retrieval_times["item_cf_ms"] = (time.perf_counter() - t0) * 1000.0

        # 1.3 Popularity Fill (Cold start fallback & trending discovery)
        t0 = time.perf_counter()
        pop_candidates = self.popularity_recommender.recommend(history, top_k=30)
        for rank_pos, (item_idx, score) in enumerate(pop_candidates):
            if item_idx not in excluded_items and item_idx not in candidate_set:
                candidate_set[item_idx] = {
                    "source": "popularity",
                    "two_tower_score": 0.0,
                    "cf_score": 0.0,
                    "candidate_rank": len(candidate_set),
                }
        retrieval_times["popularity_ms"] = (time.perf_counter() - t0) * 1000.0

        candidate_indices = list(candidate_set.keys())

        # Step 2: Personalized Ranking (LightGBM)
        t0 = time.perf_counter()
        two_tower_scores = {k: v["two_tower_score"] for k, v in candidate_set.items()}
        cf_scores = {k: v["cf_score"] for k, v in candidate_set.items()}
        sources = {k: v["source"] for k, v in candidate_set.items()}

        features_df = self.feature_extractor.extract_features(
            user_idx=user_idx,
            user_history=history,
            candidate_items=candidate_indices,
            two_tower_scores=two_tower_scores,
            cf_scores=cf_scores,
            retrieval_sources=sources,
        )

        if self.ranker.is_fitted and not features_df.empty:
            ranked_pairs = self.ranker.rank(features_df)
        else:
            # Fallback to combined retrieval score
            ranked_pairs = [
                (idx, candidate_set[idx]["two_tower_score"] + candidate_set[idx]["cf_score"])
                for idx in candidate_indices
            ]
            ranked_pairs.sort(key=lambda x: x[1], reverse=True)

        ranking_time_ms = (time.perf_counter() - t0) * 1000.0

        # Step 3: MMR Diversity Re-ranking
        t0 = time.perf_counter()
        reranked_results = self.reranker.rerank(
            candidate_scores=ranked_pairs,
            top_k=top_k,
            lambda_param=lambda_param,
            novelty_weight=novelty_weight,
            excluded_item_indices=excluded_items,
        )
        reranking_time_ms = (time.perf_counter() - t0) * 1000.0

        # Step 4: Metadata Enrichment & Explainability
        t0 = time.perf_counter()
        final_recommendations = []
        for rank_pos, item in enumerate(reranked_results):
            m_idx = item["movie_idx"]
            movie_row = self.movies_df.iloc[m_idx]

            # Generate explainability reason
            reason = self._generate_explanation(
                movie_row=movie_row,
                source=sources.get(m_idx, "neural_retrieval"),
                cf_score=cf_scores.get(m_idx, 0.0),
                novelty_score=item.get("novelty_score", 0.0),
                history=history,
            )

            rec_item = {
                "rank": rank_pos + 1,
                "movie_idx": int(m_idx),
                "movieId": int(movie_row["movieId"]),
                "title": movie_row["clean_title"],
                "full_title": movie_row["title"],
                "release_year": int(movie_row["release_year"]),
                "genres": movie_row["genres"].split("|") if isinstance(movie_row["genres"], str) else [],
                "rating_mean": round(float(movie_row["rating_mean"]), 2),
                "rating_count": int(movie_row["rating_count"]),
                "final_score": round(item["final_score"], 4),
                "relevance_score": round(item["relevance_score"], 4),
                "diversity_penalty": round(item["diversity_penalty"], 4),
                "novelty_score": round(item["novelty_score"], 4),
                "retrieval_source": sources.get(m_idx, "two_tower"),
                "reason": reason,
            }
            final_recommendations.append(rec_item)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "user_idx": user_idx,
            "recommendations": final_recommendations,
            "total_candidates": len(candidate_indices),
            "retrieval_sources_breakdown": {
                "two_tower": sum(1 for v in sources.values() if v == "two_tower"),
                "item_cf": sum(1 for v in sources.values() if v == "item_cf"),
                "popularity": sum(1 for v in sources.values() if v == "popularity"),
            },
            "timings_ms": {
                **retrieval_times,
                "ranking_ms": ranking_time_ms,
                "reranking_ms": reranking_time_ms,
                "total_ms": total_latency_ms,
            },
        }

    def _generate_explanation(
        self,
        movie_row: pd.Series,
        source: str,
        cf_score: float,
        novelty_score: float,
        history: list[int],
    ) -> str:
        genres = movie_row["genres"] if isinstance(movie_row["genres"], str) else ""
        primary_genre = genres.split("|")[0] if genres else "Film"

        if source == "item_cf" and history:
            last_movie = self.movies_df.iloc[history[-1]]["clean_title"]
            return f"Because you watched {last_movie}"
        elif novelty_score > 0.65:
            return f"Hidden Gem in {primary_genre}"
        elif source == "two_tower":
            return f"Strong match for your {primary_genre} preference"
        else:
            return "Popular among film enthusiasts"
