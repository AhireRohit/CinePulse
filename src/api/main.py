import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router, set_recommendation_engine
from src.core.config import settings
from src.core.logging import logger
from src.db.database import init_db
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

        # Initialize Recommendation Engine
        engine = RecommendationEngine(
            movies_df=movies_df,
            mappings=mappings,
            genre_cols=genre_cols,
            user_histories=user_histories,
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

        # Load FAISS Index
        faiss_index_path = settings.indices_dir / "item_index.faiss"
        if faiss_index_path.exists():
            logger.info("Loading FAISS candidate index...")
            engine.faiss_index.load(settings.indices_dir)

        # Load LightGBM Ranker
        ranker_path = settings.models_dir / "ranker_lgbm.txt"
        if ranker_path.exists():
            logger.info("Loading LightGBM Ranker...")
            engine.ranker.load(ranker_path)

        set_recommendation_engine(engine)
        logger.info("CinePulse Recommendation Engine successfully loaded and ready.")
    else:
        logger.warning("Processed data artifacts not found. Please run data preparation pipeline.")

    yield
    logger.info("Shutting down CinePulse application...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-Grade Multi-Stage Recommendation Platform (Two-Tower Retrieval, LightGBM Ranking, MMR Diversity, Session Personalization)",
    lifespan=lifespan,
)

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
