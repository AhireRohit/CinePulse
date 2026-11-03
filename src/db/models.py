from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_idx = Column(Integer, unique=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    display_name = Column(String(128), nullable=False)
    avatar_color = Column(String(32), default="#D4AF37")
    bio = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_idx = Column(Integer, index=True, nullable=False)
    movie_idx = Column(Integer, index=True, nullable=False)
    action = Column(String(32), index=True, nullable=False)  # 'like', 'dislike', 'save', 'dismiss', 'rate'
    rating = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SavedMovie(Base):
    __tablename__ = "saved_movies"

    id = Column(Integer, primary_key=True, index=True)
    user_idx = Column(Integer, index=True, nullable=False)
    movie_idx = Column(Integer, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class OnboardingPreference(Base):
    __tablename__ = "onboarding_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_idx = Column(Integer, unique=True, index=True, nullable=False)
    genres_json = Column(Text, default="[]")
    favorite_movies_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)
