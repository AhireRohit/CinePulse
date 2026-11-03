import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    CreateUserRequest,
    FeedbackRequest,
    FeedbackResponse,
    ModelEvaluationResponse,
    MovieDetailResponse,
    RecommendationFeedResponse,
    RecommendationItem,
    SearchResponse,
    UserInsightsResponse,
    UserProfileResponse,
)
from src.core.config import settings
from src.db.database import get_db
from src.db.models import Interaction, OnboardingPreference, SavedMovie, User

router = APIRouter()

# Global engine reference injected on startup
_recommendation_engine = None


def set_recommendation_engine(engine):
    global _recommendation_engine
    _recommendation_engine = engine


def get_engine():
    if _recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine is still initializing.")
    return _recommendation_engine


@router.get("/health")
async def health_check():
    engine_ready = _recommendation_engine is not None
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "engine_ready": engine_ready,
    }


@router.get("/api/users", response_model=list[UserProfileResponse])
async def list_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).order_by(User.id))
    users = result.scalars().all()
    profiles = []
    for u in users:
        pref_result = await db.execute(select(OnboardingPreference).where(OnboardingPreference.user_idx == u.user_idx))
        pref = pref_result.scalar_one_or_none()
        genres = json.loads(pref.genres_json) if pref and pref.genres_json else []
        profiles.append(
            UserProfileResponse(
                user_idx=u.user_idx,
                username=u.username,
                display_name=u.display_name,
                avatar_color=u.avatar_color,
                bio=u.bio,
                preferred_genres=genres,
            )
        )
    return profiles


@router.post("/api/users", response_model=UserProfileResponse)
async def create_user(req: CreateUserRequest, db: AsyncSession = Depends(get_db)):
    engine = get_engine()
    # Check username uniqueness
    existing = await db.execute(select(User).where(User.username == req.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already exists.")

    # Assign a new user_idx (above dataset user count to represent a new user)
    max_idx_result = await db.execute(select(User.user_idx).order_by(desc(User.user_idx)).limit(1))
    highest_idx = max_idx_result.scalar_one_or_none() or engine.num_users
    new_user_idx = max(highest_idx + 1, engine.num_users + 1)

    new_user = User(
        user_idx=new_user_idx,
        username=req.username,
        display_name=req.display_name,
        avatar_color=req.avatar_color,
        bio=req.bio,
    )
    db.add(new_user)
    await db.flush()

    pref = OnboardingPreference(
        user_idx=new_user_idx,
        genres_json=json.dumps(req.preferred_genres),
        favorite_movies_json=json.dumps(req.favorite_movies),
    )
    db.add(pref)
    await db.commit()

    # Seed the engine session with onboarding preferences
    engine.get_user_session(new_user_idx)
    for m_idx in req.favorite_movies:
        engine.record_feedback(new_user_idx, m_idx, "like")

    return UserProfileResponse(
        user_idx=new_user_idx,
        username=new_user.username,
        display_name=new_user.display_name,
        avatar_color=new_user.avatar_color,
        bio=new_user.bio,
        preferred_genres=req.preferred_genres,
    )


@router.get("/api/recommendations/{user_id}", response_model=RecommendationFeedResponse)
async def get_recommendations(
    user_id: int,
    k: int = Query(default=20, ge=5, le=50),
    diversity: float = Query(default=0.65, ge=0.0, le=1.0),
    novelty: float = Query(default=0.15, ge=0.0, le=0.5),
):
    engine = get_engine()
    # Generate primary personalized recommendations
    rec_result = engine.recommend(
        user_idx=user_id,
        top_k=k,
        retrieval_k=settings.retrieval_candidates_k,
        lambda_param=diversity,
        novelty_weight=novelty,
    )
    recs = rec_result["recommendations"]

    # Select hero featured item
    hero_item = recs[0] if recs else None

    # Categorized rails
    rails = {}
    rails["Top Picks For You"] = recs[:10]

    # Rail 2: Because you watched...
    session = engine.get_user_session(user_id)
    history = session["history"]
    if history:
        last_m = history[-1]
        last_title = engine.movies_df.iloc[last_m]["clean_title"]
        # Fetch items similar to last movie using Item-CF or Content
        similar = engine.item_cf.get_similar_items(last_m, top_k=8)
        consumed = set(history) | session["dismissals"]
        similar_items = []
        for idx, score in similar:
            if idx not in consumed:
                row = engine.movies_df.iloc[idx]
                similar_items.append(
                    RecommendationItem(
                        rank=len(similar_items) + 1,
                        movie_idx=idx,
                        movieId=int(row["movieId"]),
                        title=row["clean_title"],
                        full_title=row["title"],
                        release_year=int(row["release_year"]),
                        genres=row["genres"].split("|") if isinstance(row["genres"], str) else [],
                        rating_mean=round(float(row["rating_mean"]), 2),
                        rating_count=int(row["rating_count"]),
                        final_score=round(float(score), 4),
                        relevance_score=round(float(score), 4),
                        diversity_penalty=0.0,
                        novelty_score=0.2,
                        retrieval_source="item_cf",
                        reason=f"Similar to {last_title}",
                    )
                )
        if similar_items:
            rails[f"Because You Watched {last_title}"] = similar_items

    # Rail 3: Hidden Gems (High Novelty)
    hidden_gems = [r for r in recs if r["novelty_score"] > 0.55]
    if not hidden_gems:
        hidden_gems = sorted(recs, key=lambda x: x["novelty_score"], reverse=True)[:6]
    rails["Hidden Gems & Deep Cuts"] = hidden_gems[:8]

    # Rail 4: Trending & Critically Acclaimed
    trending_raw = engine.popularity_recommender.recommend(history, top_k=8)
    trending_items = []
    for rank_idx, (idx, score) in enumerate(trending_raw):
        row = engine.movies_df.iloc[idx]
        trending_items.append(
            RecommendationItem(
                rank=rank_idx + 1,
                movie_idx=idx,
                movieId=int(row["movieId"]),
                title=row["clean_title"],
                full_title=row["title"],
                release_year=int(row["release_year"]),
                genres=row["genres"].split("|") if isinstance(row["genres"], str) else [],
                rating_mean=round(float(row["rating_mean"]), 2),
                rating_count=int(row["rating_count"]),
                final_score=round(float(score), 4),
                relevance_score=round(float(score), 4),
                diversity_penalty=0.0,
                novelty_score=0.1,
                retrieval_source="popularity",
                reason="Critically acclaimed & widely watched",
            )
        )
    rails["Trending & Highly Rated"] = trending_items

    return RecommendationFeedResponse(
        user_idx=user_id,
        hero_featured=hero_item,
        rails=rails,
        total_candidates=rec_result["total_candidates"],
        retrieval_sources_breakdown=rec_result["retrieval_sources_breakdown"],
        timings_ms=rec_result["timings_ms"],
    )


@router.post("/api/feedback", response_model=FeedbackResponse)
async def post_feedback(req: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    engine = get_engine()
    if req.action not in ["like", "dislike", "save", "dismiss", "rate"]:
        raise HTTPException(status_code=400, detail="Invalid action.")

    # 1. Update in-memory session engine immediately
    engine.record_feedback(
        user_idx=req.user_idx,
        movie_idx=req.movie_idx,
        action=req.action,
        rating=req.rating,
    )

    # 2. Persist to database
    interaction = Interaction(
        user_idx=req.user_idx,
        movie_idx=req.movie_idx,
        action=req.action,
        rating=req.rating,
    )
    db.add(interaction)

    if req.action == "save":
        saved = SavedMovie(user_idx=req.user_idx, movie_idx=req.movie_idx)
        db.add(saved)

    await db.commit()

    session = engine.get_user_session(req.user_idx)
    summary = {
        "active_likes": len(session["likes"]),
        "active_saves": len(session["saves"]),
        "dismissed": len(session["dismissals"]),
        "recent_history_len": len(session["history"]),
    }

    return FeedbackResponse(
        success=True,
        user_idx=req.user_idx,
        movie_idx=req.movie_idx,
        action=req.action,
        message=f"Recorded '{req.action}'. Recommendations will adapt instantly.",
        session_summary=summary,
    )


@router.get("/api/movies/{movie_id}", response_model=MovieDetailResponse)
async def get_movie(movie_id: int):
    engine = get_engine()
    if movie_id < 0 or movie_id >= len(engine.movies_df):
        raise HTTPException(status_code=404, detail="Movie not found.")

    row = engine.movies_df.iloc[movie_id]
    similar_raw = engine.item_cf.get_similar_items(movie_id, top_k=6)
    similar_items = []
    for idx, score in similar_raw:
        sim_row = engine.movies_df.iloc[idx]
        similar_items.append(
            RecommendationItem(
                rank=len(similar_items) + 1,
                movie_idx=idx,
                movieId=int(sim_row["movieId"]),
                title=sim_row["clean_title"],
                full_title=sim_row["title"],
                release_year=int(sim_row["release_year"]),
                genres=sim_row["genres"].split("|") if isinstance(sim_row["genres"], str) else [],
                rating_mean=round(float(sim_row["rating_mean"]), 2),
                rating_count=int(sim_row["rating_count"]),
                final_score=round(float(score), 4),
                relevance_score=round(float(score), 4),
                diversity_penalty=0.0,
                novelty_score=0.2,
                retrieval_source="item_cf",
                reason="Similar themes & co-ratings",
            )
        )

    return MovieDetailResponse(
        movie_idx=movie_id,
        movieId=int(row["movieId"]),
        title=row["clean_title"],
        full_title=row["title"],
        release_year=int(row["release_year"]),
        genres=row["genres"].split("|") if isinstance(row["genres"], str) else [],
        rating_mean=round(float(row["rating_mean"]), 2),
        rating_count=int(row["rating_count"]),
        bayesian_rating=round(float(row["bayesian_rating"]), 2),
        overview=f"A critically noted {row['release_year']} release featuring {row['genres']}. Rated {round(float(row['rating_mean']), 1)}/5.0 across {int(row['rating_count']):,} community reviews.",
        similar_movies=similar_items,
    )


@router.get("/api/search", response_model=SearchResponse)
async def search_movies(q: str = Query(min_length=1)):
    engine = get_engine()
    query_lower = q.lower().strip()

    # Search in clean_title and genres
    matches = engine.movies_df[
        engine.movies_df["clean_title"].str.lower().str.contains(query_lower, na=False) |
        engine.movies_df["genres"].str.lower().str.contains(query_lower, na=False)
    ].head(30)

    results = []
    for rank_idx, (_, row) in enumerate(matches.iterrows()):
        idx = int(row["movie_idx"])
        results.append(
            RecommendationItem(
                rank=rank_idx + 1,
                movie_idx=idx,
                movieId=int(row["movieId"]),
                title=row["clean_title"],
                full_title=row["title"],
                release_year=int(row["release_year"]),
                genres=row["genres"].split("|") if isinstance(row["genres"], str) else [],
                rating_mean=round(float(row["rating_mean"]), 2),
                rating_count=int(row["rating_count"]),
                final_score=round(float(row["bayesian_rating"]), 4),
                relevance_score=round(float(row["rating_mean"]), 4),
                diversity_penalty=0.0,
                novelty_score=0.3,
                retrieval_source="search",
                reason="Search query match",
            )
        )

    return SearchResponse(query=q, total=len(results), results=results)


@router.get("/api/users/{user_id}/history")
async def get_user_history(user_id: int):
    engine = get_engine()
    session = engine.get_user_session(user_id)
    history_indices = session["history"][-30:]  # last 30
    items = []
    for idx in reversed(history_indices):
        if idx < len(engine.movies_df):
            row = engine.movies_df.iloc[idx]
            items.append({
                "movie_idx": idx,
                "movieId": int(row["movieId"]),
                "title": row["clean_title"],
                "release_year": int(row["release_year"]),
                "genres": row["genres"].split("|") if isinstance(row["genres"], str) else [],
                "rating_mean": round(float(row["rating_mean"]), 2),
            })
    return {"user_idx": user_id, "history": items}


@router.get("/api/users/{user_id}/saved")
async def get_user_saved(user_id: int):
    engine = get_engine()
    session = engine.get_user_session(user_id)
    saved_indices = list(session["saves"])
    items = []
    for idx in saved_indices:
        if idx < len(engine.movies_df):
            row = engine.movies_df.iloc[idx]
            items.append({
                "movie_idx": idx,
                "movieId": int(row["movieId"]),
                "title": row["clean_title"],
                "release_year": int(row["release_year"]),
                "genres": row["genres"].split("|") if isinstance(row["genres"], str) else [],
                "rating_mean": round(float(row["rating_mean"]), 2),
            })
    return {"user_idx": user_id, "saved": items}


@router.get("/api/users/{user_id}/insights", response_model=UserInsightsResponse)
async def get_user_insights(user_id: int):
    engine = get_engine()
    session = engine.get_user_session(user_id)
    history = session["history"]

    # Calculate genre affinity
    genre_counts = {}
    for idx in history:
        if idx < len(engine.movies_df):
            genres_str = engine.movies_df.iloc[idx]["genres"]
            if isinstance(genres_str, str):
                for g in genres_str.split("|"):
                    genre_counts[g] = genre_counts.get(g, 0) + 1

    total_g = sum(genre_counts.values()) if genre_counts else 1
    top_genres = [
        {"genre": g, "affinity": round(cnt / total_g, 3)}
        for g, cnt in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    ]

    rec_result = engine.recommend(user_id, top_k=10)

    return UserInsightsResponse(
        user_idx=user_id,
        total_interactions=len(history),
        active_likes_count=len(session["likes"]),
        active_saves_count=len(session["saves"]),
        dismissed_count=len(session["dismissals"]),
        top_affinity_genres=top_genres,
        retrieval_sources_breakdown=rec_result["retrieval_sources_breakdown"],
        timing_breakdown_ms=rec_result["timings_ms"],
    )


@router.get("/api/metrics", response_model=ModelEvaluationResponse)
async def get_offline_metrics():
    metrics_file = settings.metrics_dir / "evaluation_results.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not yet generated.")

    with open(metrics_file) as f:
        models_data = json.load(f)

    with open(settings.data_processed_dir / "mappings.json") as f:
        mappings = json.load(f)

    return ModelEvaluationResponse(
        dataset_name="MovieLens 1M (GroupLens Official)",
        num_users=mappings["num_users"],
        num_movies=mappings["num_movies"],
        test_samples=6040,
        models=models_data,
    )
