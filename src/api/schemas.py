from pydantic import BaseModel, Field


class UserProfileResponse(BaseModel):
    user_idx: int
    username: str
    display_name: str
    avatar_color: str
    bio: str | None = None
    auth_token: str | None = None
    preferred_genres: list[str] = Field(default_factory=list)


class CreateUserRequest(BaseModel):
    username: str
    display_name: str
    preferred_genres: list[str] = Field(default_factory=list)
    favorite_movies: list[int] = Field(default_factory=list)
    avatar_color: str = "#D4AF37"
    bio: str | None = None


class RecommendationItem(BaseModel):
    rank: int
    movie_idx: int
    movieId: int
    title: str
    full_title: str
    release_year: int
    genres: list[str]
    rating_mean: float
    rating_count: int
    bayesian_rating: float | None = None
    final_score: float
    relevance_score: float
    match_percentage: int | None = None
    affinity_label: str | None = None
    diversity_penalty: float
    novelty_score: float
    retrieval_source: str
    reason: str
    poster_url: str | None = None
    backdrop_url: str | None = None


class RecommendationFeedResponse(BaseModel):
    user_idx: int
    hero_featured: RecommendationItem | None = None
    rails: dict[str, list[RecommendationItem]]
    total_candidates: int
    retrieval_sources_breakdown: dict[str, int]
    timings_ms: dict[str, float]


class FeedbackRequest(BaseModel):
    user_idx: int
    movie_idx: int
    action: str = Field(description="'like', 'dislike', 'save', 'dismiss', 'rate'")
    rating: float | None = None


class FeedbackResponse(BaseModel):
    success: bool
    user_idx: int
    movie_idx: int
    action: str
    message: str
    session_summary: dict


class MovieDetailResponse(BaseModel):
    movie_idx: int
    movieId: int
    title: str
    full_title: str
    release_year: int
    genres: list[str]
    rating_mean: float
    rating_count: int
    bayesian_rating: float
    poster_url: str | None = None
    backdrop_url: str | None = None
    overview: str | None = None
    similar_movies: list[RecommendationItem] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[RecommendationItem]


class GenreAffinityItem(BaseModel):
    genre: str
    affinity: float


class UserInsightsResponse(BaseModel):
    user_idx: int
    total_interactions: int
    active_likes_count: int
    active_saves_count: int
    dismissed_count: int
    top_affinity_genres: list[GenreAffinityItem]
    retrieval_sources_breakdown: dict[str, int]
    timing_breakdown_ms: dict[str, float]


class ModelEvaluationResponse(BaseModel):
    dataset_name: str
    num_users: int
    num_movies: int
    test_samples: int
    models: dict[str, dict]
