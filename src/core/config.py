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

    # Model Version & Feature Flags
    # "v1" = original production (17 features, TT=100/CF=100/Pop=30, MMR lambda=0.85, beta=0.03)
    # "v2" = tournament winner Config 8 (24 features, TT=200/CF=100/Pop=30, ranker_lgbm_exp3, MMR lambda=0.95, beta=0.01)
    model_version: str = "v2"
    retrieval_two_tower_k_v2: int = 200
    default_mmr_lambda_v2: float = 0.95
    novelty_weight_v2: float = 0.01
    v2_ranker_path: Path = base_dir / "artifacts" / "models" / "ranker_lgbm_v2.txt"

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
