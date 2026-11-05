import json
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
        user_train_ratings: dict[int, dict[int, float]] | None = None,
    ):
        self.movies_df = movies_df
        self.mappings = mappings
        self.genre_cols = genre_cols
        self.user_histories = user_histories
        self.user_train_ratings = user_train_ratings or {}
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

        # Model serving mode configuration (v1 = original production, v2 = Config 8)
        self.model_version = settings.model_version
        if self.model_version == "v2":
            logger.info("RecommendationEngine operating in v2 mode (Tournament Winner Config 8).")
            self.retrieval_two_tower_k = settings.retrieval_two_tower_k_v2
            self.default_lambda = settings.default_mmr_lambda_v2
            self.default_novelty_weight = settings.novelty_weight_v2
            self.item_cf.top_k_neighbors = 200
            self.item_cf.shrinkage = 25.0
            self.feature_extractor.use_expanded = True
        else:
            logger.info("RecommendationEngine operating in v1 mode (Original Production Baseline).")
            self.retrieval_two_tower_k = settings.retrieval_candidates_k
            self.default_lambda = settings.default_mmr_lambda
            self.default_novelty_weight = settings.novelty_weight
            self.item_cf.top_k_neighbors = 50
            self.item_cf.shrinkage = 25.0
            self.feature_extractor.use_expanded = False

        self.reranker = MMRReRanker(
            movies_df=self.movies_df,
            genre_cols=self.genre_cols,
            default_lambda=self.default_lambda,
            default_novelty_weight=self.default_novelty_weight,
        )

        # Precomputed movie item vectors
        self.item_embeddings: np.ndarray | None = None

        # Loaded movie posters mapping (movieId -> poster_url)
        self.movie_posters: dict[int, str] = {}
        posters_path = settings.data_processed_dir / "movie_posters.json"
        if posters_path.exists():
            try:
                with open(posters_path, encoding="utf-8") as f:
                    raw_posters = json.load(f)
                    self.movie_posters = {int(k): v for k, v in raw_posters.items()}
                logger.info(f"Loaded {len(self.movie_posters)} verified movie poster URLs.")
            except Exception as e:
                logger.warning(f"Could not load movie posters: {e}")

    def get_user_session(self, user_idx: int) -> dict:
        if user_idx not in self.user_sessions:
            # Initialize from historical train data
            hist = list(self.user_histories.get(user_idx, []))
            hist_ratings = dict(self.user_train_ratings.get(user_idx, {})) if self.user_train_ratings else {}
            self.user_sessions[user_idx] = {
                "history": hist,
                "likes": set(),
                "dislikes": set(),
                "saves": set(),
                "dismissals": set(),
                "ratings": hist_ratings,
                "preferred_genres": [],
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
            # Pad history (use num_movies as pad token)
            padded_hist = np.full(settings.max_sequence_length, self.num_movies, dtype=np.int64)
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
                session = self.get_user_session(user_idx)
                pref_genres = set(session.get("preferred_genres", []))
                if pref_genres:
                    g_vec = np.zeros(len(self.genre_cols), dtype=np.float32)
                    for idx_g, col in enumerate(self.genre_cols):
                        g_name = col.replace("genre_", "")
                        if g_name in pref_genres:
                            g_vec[idx_g] = 1.0
                    genre_pref = torch.tensor(g_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
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

        # Step 1: Candidate Retrieval (Multi-Source Union)
        candidate_sources: dict[int, set[str]] = {}
        cf_score_map: dict[int, float] = {}
        retrieval_times = {}

        # 1.1 Two-Tower ANN Retrieval
        t0 = time.perf_counter()
        user_vector = self.generate_user_vector(user_idx, history)
        eff_retrieval_k = (
            self.retrieval_two_tower_k
            if (retrieval_k == 100 or retrieval_k == settings.retrieval_candidates_k)
            else retrieval_k
        )
        two_tower_candidates = self.faiss_index.search(
            query_embedding=user_vector,
            top_k=eff_retrieval_k,
            exclude_item_indices=excluded_items,
        )
        for rank_pos, (item_idx, score) in enumerate(two_tower_candidates):
            if item_idx not in candidate_sources:
                candidate_sources[item_idx] = set()
            candidate_sources[item_idx].add("two_tower")
        retrieval_times["two_tower_ms"] = (time.perf_counter() - t0) * 1000.0

        # 1.2 Item-CF Retrieval (Anchored strictly on positive history, top 100)
        t0 = time.perf_counter()
        positive_history = [
            m for m in history
            if m not in session["dislikes"] and m not in session["dismissals"]
            and session.get("ratings", {}).get(m, 4.0) >= 3.5
        ]
        if not positive_history and history:
            positive_history = [m for m in history if m not in session["dislikes"]]

        if positive_history:
            cf_candidates = self.item_cf.recommend(positive_history, top_k=100)
            for rank_pos, (item_idx, score) in enumerate(cf_candidates):
                if item_idx not in excluded_items:
                    if item_idx not in candidate_sources:
                        candidate_sources[item_idx] = set()
                    candidate_sources[item_idx].add("item_cf")
                    cf_score_map[item_idx] = float(score)
        retrieval_times["item_cf_ms"] = (time.perf_counter() - t0) * 1000.0

        # 1.3 Popularity Fill (Cold start fallback & trending discovery)
        t0 = time.perf_counter()
        pop_candidates = self.popularity_recommender.recommend(history, top_k=30)
        for rank_pos, (item_idx, score) in enumerate(pop_candidates):
            if item_idx not in excluded_items:
                if item_idx not in candidate_sources:
                    candidate_sources[item_idx] = set()
                candidate_sources[item_idx].add("popularity")

        # 1.4 Cold-start Onboarding Genre Candidates
        pref_genres = session.get("preferred_genres", [])
        if not history and pref_genres:
            for g_name in pref_genres:
                g_col = f"genre_{g_name}"
                if g_col in self.movies_df.columns:
                    genre_matches = self.movies_df[self.movies_df[g_col] == 1].sort_values(
                        by="bayesian_rating", ascending=False
                    ).head(25)
                    for _, row in genre_matches.iterrows():
                        idx = int(row["movie_idx"])
                        if idx not in excluded_items:
                            candidate_sources.setdefault(idx, set()).add("onboarding_genre")
        retrieval_times["popularity_ms"] = (time.perf_counter() - t0) * 1000.0

        candidate_indices = list(candidate_sources.keys())

        # Vectorized computation of genuine Two-Tower scores for ALL candidates
        if self.item_embeddings is not None and len(candidate_indices) > 0:
            valid_cand_idx = [c for c in candidate_indices if c < len(self.item_embeddings)]
            if valid_cand_idx:
                cand_embs = self.item_embeddings[valid_cand_idx]
                dots = np.dot(cand_embs, user_vector)
                two_tower_scores = {c: float(dots[i]) for i, c in enumerate(valid_cand_idx)}
            else:
                two_tower_scores = {c: 0.0 for c in candidate_indices}
        else:
            # Fallback to ANN score if item_embeddings not in memory
            two_tower_scores = {item_idx: float(score) for item_idx, score in two_tower_candidates}
            for c in candidate_indices:
                if c not in two_tower_scores:
                    two_tower_scores[c] = 0.0

        cf_scores = {c: cf_score_map.get(c, 0.0) for c in candidate_indices}
        sources = {c: candidate_sources[c] for c in candidate_indices}

        # Step 2: Personalized Ranking (LightGBM)
        t0 = time.perf_counter()

        user_genre_profile = None
        if not history and pref_genres:
            pref_set = set(pref_genres)
            g_vec = np.zeros(len(self.genre_cols), dtype=np.float32)
            for idx_g, col in enumerate(self.genre_cols):
                if col.replace("genre_", "") in pref_set:
                    g_vec[idx_g] = 1.0
            user_genre_profile = g_vec

        features_df = self.feature_extractor.extract_features(
            user_idx=user_idx,
            user_history=history,
            candidate_items=candidate_indices,
            two_tower_scores=two_tower_scores,
            cf_scores=cf_scores,
            retrieval_sources=sources,
            user_genre_profile=user_genre_profile,
            user_ratings=session.get("ratings", {}),
            use_expanded=self.feature_extractor.use_expanded,
        )

        if self.ranker.is_fitted and not features_df.empty:
            ranked_pairs = self.ranker.rank(features_df)
        else:
            # Fallback to combined retrieval score
            ranked_pairs = [
                (idx, two_tower_scores.get(idx, 0.0) + cf_scores.get(idx, 0.0))
                for idx in candidate_indices
            ]
            ranked_pairs.sort(key=lambda x: x[1], reverse=True)

        ranking_time_ms = (time.perf_counter() - t0) * 1000.0

        # Step 3: MMR Diversity Re-ranking
        t0 = time.perf_counter()
        eff_lambda = self.default_lambda if lambda_param is None else lambda_param
        eff_novelty = self.default_novelty_weight if novelty_weight is None else novelty_weight
        reranked_results = self.reranker.rerank(
            candidate_scores=ranked_pairs,
            top_k=top_k,
            lambda_param=eff_lambda,
            novelty_weight=eff_novelty,
            excluded_item_indices=excluded_items,
        )
        reranking_time_ms = (time.perf_counter() - t0) * 1000.0

        # Step 4: Metadata Enrichment & Explainability
        t0 = time.perf_counter()
        final_recommendations = []
        for rank_pos, item in enumerate(reranked_results):
            m_idx = item["movie_idx"]
            movie_row = self.movies_df.iloc[m_idx]

            # Format retrieval source string
            raw_source = sources.get(m_idx, "two_tower")
            if isinstance(raw_source, (set, list, tuple)):
                source_str = " + ".join(sorted(raw_source)) if raw_source else "two_tower"
                primary_source = "item_cf" if "item_cf" in raw_source else ("two_tower" if "two_tower" in raw_source else "popularity")
            else:
                source_str = str(raw_source)
                primary_source = str(raw_source)

            # Generate explainability reason
            reason = self._generate_explanation(
                movie_row=movie_row,
                source=primary_source,
                cf_score=cf_scores.get(m_idx, 0.0),
                novelty_score=item.get("novelty_score", 0.0),
                history=history,
            )

            bayesian_rating = float(movie_row["bayesian_rating"]) if "bayesian_rating" in movie_row else float(movie_row["rating_mean"])
            match_pct = int(item.get("match_percentage", int(round(70 + 25 * item["relevance_score"]))))
            if rank_pos == 0:
                affinity_label = "Top Affinity Pick"
            elif rank_pos < 4:
                affinity_label = "High Affinity"
            else:
                affinity_label = "Curated Recommendation"

            m_id_int = int(movie_row["movieId"])
            poster_url = self.movie_posters.get(m_id_int)

            rec_item = {
                "rank": rank_pos + 1,
                "movie_idx": int(m_idx),
                "movieId": m_id_int,
                "title": movie_row["clean_title"],
                "full_title": movie_row["title"],
                "release_year": int(movie_row["release_year"]),
                "genres": movie_row["genres"].split("|") if isinstance(movie_row["genres"], str) else [],
                "rating_mean": round(float(movie_row["rating_mean"]), 2),
                "rating_count": int(movie_row["rating_count"]),
                "bayesian_rating": round(bayesian_rating, 2),
                "final_score": round(item["final_score"], 4),
                "relevance_score": round(item["relevance_score"], 4),
                "match_percentage": match_pct,
                "affinity_label": affinity_label,
                "diversity_penalty": round(item["diversity_penalty"], 4),
                "novelty_score": round(item["novelty_score"], 4),
                "retrieval_source": source_str,
                "reason": reason,
                "poster_url": poster_url,
            }
            final_recommendations.append(rec_item)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        def count_source(s_name: str) -> int:
            return sum(1 for v in sources.values() if (s_name in v if isinstance(v, (set, list, tuple)) else v == s_name))

        # Quality-Aware & Confidence-Weighted Hero Spotlight Selection:
        # 1. Enforce community acclaim floor: bayesian_rating >= 3.20, rating_mean >= 3.10, rating_count >= 50.
        # 2. Prioritize candidate with highest balanced spotlight merit:
        #    spotlight_merit = relevance_score * (bayesian_rating / 5.0) * confidence_weight
        # 3. If no personalized recommendation meets this acclaim floor, return None so the frontend
        #    renders a tasteful curated platform showcase rather than a deceptive pseudo-recommendation.
        eligible_spotlight = []
        for r in final_recommendations:
            b_rating = float(r.get("bayesian_rating") or r["rating_mean"])
            r_mean = float(r["rating_mean"])
            r_count = int(r.get("rating_count", 0))
            if b_rating >= 3.20 and r_mean >= 3.10 and r_count >= 50:
                confidence = min(1.0, float(np.log10(max(r_count, 1)) / np.log10(150)))
                merit = float(r["relevance_score"]) * (b_rating / 5.0) * confidence
                eligible_spotlight.append((merit, r))

        if eligible_spotlight:
            eligible_spotlight.sort(key=lambda x: x[0], reverse=True)
            best_hero = dict(eligible_spotlight[0][1])
            best_hero["reason"] = f"Top Spotlight: {best_hero['rating_mean']}/5.0 community rating with high personal relevance"
            hero_featured = best_hero
        else:
            hero_featured = None

        return {
            "user_idx": user_idx,
            "hero_featured": hero_featured,
            "recommendations": final_recommendations,
            "total_candidates": len(candidate_indices),
            "retrieval_sources_breakdown": {
                "two_tower": count_source("two_tower"),
                "item_cf": count_source("item_cf"),
                "popularity": count_source("popularity"),
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
