from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CinePulse"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_raw_dir: Path = base_dir / "data" / "raw"
    data_processed_dir: Path = base_dir / "data" / "processed"
    artifacts_dir: Path = base_dir / "artifacts"
    models_dir: Path = artifacts_dir / "models"
    indices_dir: Path = artifacts_dir / "indices"
    metrics_dir: Path = artifacts_dir / "metrics"

    # Database
    database_url: str = "sqlite+aiosqlite:///./cinepulse.db"

    # Model parameters
    embedding_dim: int = 64
    max_sequence_length: int = 20
    retrieval_candidates_k: int = 100
    ranking_top_k: int = 30
    default_mmr_lambda: float = 0.85
    novelty_weight: float = 0.03

    # TMDB API (Optional: enrich poster / backdrop / synopsis)
    tmdb_api_key: str | None = None
    tmdb_image_base_url: str = "https://image.tmdb.org/t/p/w500"

    # Device
    force_cpu: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
