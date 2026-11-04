import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.core.logging import logger
from src.db.models import Base, OnboardingPreference, User

# Ensure parent directory for SQLite database exists
if "sqlite" in settings.database_url:
    db_path_str = settings.database_url.split("sqlite:///")[-1].split("sqlite+aiosqlite:///")[-1]
    if db_path_str:
        p = Path(db_path_str).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


DEMO_PROFILES = [
    {
        "user_idx": 12,
        "username": "elena_vance",
        "display_name": "Elena Vance",
        "avatar_color": "#3B82F6",
        "bio": "Curator of mind-bending Sci-Fi, artificial intelligence cinema, and cerebral mysteries.",
        "preferred_genres": ["Sci-Fi", "Mystery", "Thriller"],
    },
    {
        "user_idx": 45,
        "username": "marcus_sterling",
        "display_name": "Marcus Sterling",
        "avatar_color": "#D4AF37",
        "bio": "Aficionado of classic 1940s-70s Film-Noir, gritty crime sagas, and detective thrillers.",
        "preferred_genres": ["Film-Noir", "Crime", "Mystery"],
    },
    {
        "user_idx": 88,
        "username": "aria_thorne",
        "display_name": "Aria Thorne",
        "avatar_color": "#EC4899",
        "bio": "Drawn to quiet character studies, independent European dramas, and poignant visual storytelling.",
        "preferred_genres": ["Drama", "Romance", "Animation"],
    },
    {
        "user_idx": 120,
        "username": "leo_rivera",
        "display_name": "Leo Rivera",
        "avatar_color": "#EF4444",
        "bio": "High-octane blockbusters, cinematic space operas, and edge-of-your-seat action spectacles.",
        "preferred_genres": ["Action", "Adventure", "Sci-Fi"],
    },
]


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed demo users if not present
    async with AsyncSessionLocal() as session:
        for profile in DEMO_PROFILES:
            result = await session.execute(select(User).where(User.username == profile["username"]))
            existing = result.scalar_one_or_none()
            if not existing:
                user = User(
                    user_idx=profile["user_idx"],
                    username=profile["username"],
                    display_name=profile["display_name"],
                    avatar_color=profile["avatar_color"],
                    bio=profile["bio"],
                )
                session.add(user)
                await session.flush()

                pref = OnboardingPreference(
                    user_idx=profile["user_idx"],
                    genres_json=json.dumps(profile["preferred_genres"]),
                    favorite_movies_json="[]",
                )
                session.add(pref)

        await session.commit()
    logger.info("Database initialized and demo user profiles verified.")
