import json
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.api.routes import DEMO_USER_IDXS, router, set_recommendation_engine
from src.core.config import settings
from src.core.logging import logger
from src.db.database import AsyncSessionLocal, init_db
from src.db.models import Interaction, OnboardingPreference, SavedMovie, User
from src.models.two_tower import TwoTowerRecommender
from src.pipeline.engine import RecommendationEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing CinePulse application...")
    # 1. Initialize SQLite database and demo profiles
    await init_db()

    # 2. Check and load processed datasets
    movies_path = settings.data_processed_dir / "movies.parquet"
    train_ratings_path = settings.data_processed_dir / "ratings_train.parquet"
    mappings_path = settings.data_processed_dir / "mappings.json"
    histories_path = settings.data_processed_dir / "user_train_histories.json"

    if movies_path.exists() and mappings_path.exists():
        logger.info("Loading preprocessed dataset artifacts...")
        movies_df = pd.read_parquet(movies_path)
        train_ratings = pd.read_parquet(train_ratings_path)
        with open(mappings_path) as f:
            mappings = json.load(f)
        with open(histories_path) as f:
            user_histories = {int(k): v for k, v in json.load(f).items()}

        genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]
        num_users = mappings["num_users"]
        num_movies = mappings["num_movies"]

        user_train_ratings: dict[int, dict[int, float]] = {}
        for row in train_ratings[["user_idx", "movie_idx", "rating"]].itertuples(index=False):
            user_train_ratings.setdefault(int(row.user_idx), {})[int(row.movie_idx)] = float(row.rating)

        # Initialize Recommendation Engine
        engine = RecommendationEngine(
            movies_df=movies_df,
            mappings=mappings,
            genre_cols=genre_cols,
            user_histories=user_histories,
            user_train_ratings=user_train_ratings,
        )

        # Fit baselines
        logger.info("Fitting baseline models in engine...")
        engine.popularity_recommender.fit(train_ratings, num_movies=num_movies)
        engine.item_cf.fit(train_ratings, num_users=num_users, num_movies=num_movies)
        engine.content_recommender.fit(movies_df, genre_cols=genre_cols)

        # Load Two-Tower model if trained
        two_tower_path = settings.models_dir / "two_tower_best.pt"
        if two_tower_path.exists():
            logger.info("Loading trained Two-Tower model checkpoint...")
            ckpt = torch.load(two_tower_path, map_location=engine.device, weights_only=True)
            tt_model = TwoTowerRecommender(
                num_users=num_users,
                num_movies=num_movies,
                num_genres=len(genre_cols),
                embedding_dim=settings.embedding_dim,
            ).to(engine.device)
            tt_model.load_state_dict(ckpt["model_state_dict"])
            tt_model.eval()
            engine.two_tower_model = tt_model

        # Load Item Embeddings
        item_embeddings_path = settings.models_dir / "item_embeddings.npy"
        if item_embeddings_path.exists():
            logger.info("Loading precomputed item embeddings...")
            engine.item_embeddings = np.load(item_embeddings_path)

        # Load FAISS Index
        faiss_index_path = settings.indices_dir / "item_index.faiss"
        if faiss_index_path.exists():
            logger.info("Loading FAISS candidate index...")
            engine.faiss_index.load(settings.indices_dir)

        # Load LightGBM Ranker based on active model_version
        if settings.model_version == "v2" and settings.v2_ranker_path.exists():
            logger.info(f"Loading Config 8 LightGBM Ranker from {settings.v2_ranker_path}...")
            engine.ranker.load(settings.v2_ranker_path)
            engine.feature_extractor.use_expanded = True
        else:
            ranker_path = settings.models_dir / "ranker_lgbm.txt"
            if ranker_path.exists():
                logger.info(f"Loading Baseline LightGBM Ranker from {ranker_path}...")
                engine.ranker.load(ranker_path)
                engine.feature_extractor.use_expanded = False

        # Restore persistent user feedback and session state from SQLite
        logger.info("Restoring persisted user sessions and feedback from database...")
        async with AsyncSessionLocal() as db_session:
            # Restore users and onboarding preferences
            users_res = await db_session.execute(select(User))
            for u in users_res.scalars().all():
                sess = engine.get_user_session(u.user_idx)
                pref_res = await db_session.execute(
                    select(OnboardingPreference).where(OnboardingPreference.user_idx == u.user_idx)
                )
                pref = pref_res.scalar_one_or_none()
                if pref:
                    if pref.genres_json:
                        sess["preferred_genres"] = json.loads(pref.genres_json)
                    if pref.favorite_movies_json:
                        for m_idx in json.loads(pref.favorite_movies_json):
                            if m_idx not in sess["history"]:
                                sess["history"].append(m_idx)
                                sess["likes"].add(m_idx)

            # Restore all persistent feedback interactions (skipping demo user IDs to preserve baseline purity)
            inter_res = await db_session.execute(select(Interaction).order_by(Interaction.id))
            for inter in inter_res.scalars().all():
                if inter.user_idx not in DEMO_USER_IDXS:
                    engine.record_feedback(
                        user_idx=inter.user_idx,
                        movie_idx=inter.movie_idx,
                        action=inter.action,
                        rating=inter.rating,
                    )

            # Restore saved movies (skipping demo user IDs)
            saved_res = await db_session.execute(select(SavedMovie))
            for saved in saved_res.scalars().all():
                if saved.user_idx not in DEMO_USER_IDXS:
                    sess = engine.get_user_session(saved.user_idx)
                    sess["saves"].add(saved.movie_idx)
                    if saved.movie_idx not in sess["history"]:
                        sess["history"].append(saved.movie_idx)

        set_recommendation_engine(engine)
        logger.info("CinePulse Recommendation Engine successfully loaded and ready.")
    else:
        logger.warning("Processed data artifacts not found. Please run data preparation pipeline.")

    yield
    logger.info("Shutting down CinePulse application...")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window IP rate limiter and security header middleware.
    Prevents scraping, feedback flooding, and DoS attacks on public deployments.
    """
    def __init__(self, app, max_read_per_min: int = 180, max_write_per_min: int = 45):
        super().__init__(app)
        self.max_read = max_read_per_min
        self.max_write = max_write_per_min
        self.ip_timestamps: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Bypass rate limiting for static frontend assets
        path = request.url.path
        if path.startswith("/assets/") or path.endswith((".ico", ".png", ".jpg", ".js", ".css")):
            response = await call_next(request)
            return response

        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        window_start = now - 60.0

        # Purge timestamps older than 60s
        history = [t for t in self.ip_timestamps[client_ip] if t > window_start]
        self.ip_timestamps[client_ip] = history

        limit = self.max_write if request.method in ("POST", "PUT", "DELETE") else self.max_read
        if len(history) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please wait 60 seconds before making additional requests."},
                headers={"Retry-After": "60"},
            )

        self.ip_timestamps[client_ip].append(now)

        response = await call_next(request)

        # Standard Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-Stage Recommendation Platform (Two-Tower Retrieval, LightGBM Ranking, MMR Diversity, Session Personalization)",
    lifespan=lifespan,
)

# Security & Rate Limiting Middleware
app.add_middleware(RateLimitMiddleware)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Mount frontend build static files if present
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")
