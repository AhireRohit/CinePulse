"""
CinePulse Local Serving Latency Benchmark:
Measures end-to-end P50, P90, P95, P99, and Mean recommendation latency across 100 queries.
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.core.config import settings
from src.core.logging import logger
from src.models.two_tower import TwoTowerRecommender
from src.pipeline.engine import RecommendationEngine


def run_latency_benchmark(num_queries: int = 100):
    logger.info("=== Running Local CinePulse Serving Latency Benchmark ===")
    data_dir = settings.data_processed_dir
    models_dir = settings.models_dir
    indices_dir = settings.indices_dir
    exp_models_dir = Path("artifacts/experiments/relevance_v2/models")

    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    train_ratings = pd.read_parquet(data_dir / "ratings_train.parquet")
    with open(data_dir / "mappings.json") as f:
        mappings = json.load(f)
    with open(data_dir / "user_train_histories.json") as f:
        user_histories = {int(k): v for k, v in json.load(f).items()}

    genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]
    num_users = mappings["num_users"]
    num_movies = mappings["num_movies"]

    user_train_ratings: dict[int, dict[int, float]] = {}
    for row in train_ratings[["user_idx", "movie_idx", "rating"]].itertuples(index=False):
        user_train_ratings.setdefault(int(row.user_idx), {})[int(row.movie_idx)] = float(row.rating)

    # Initialize v2 engine
    settings.model_version = "v2"
    engine = RecommendationEngine(
        movies_df=movies_df,
        mappings=mappings,
        genre_cols=genre_cols,
        user_histories=user_histories,
        user_train_ratings=user_train_ratings,
    )
    engine.popularity_recommender.fit(train_ratings, num_movies=num_movies)
    engine.item_cf.fit(train_ratings, num_users=num_users, num_movies=num_movies)

    two_tower_path = models_dir / "two_tower_best.pt"
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

    engine.item_embeddings = np.load(models_dir / "item_embeddings.npy")
    engine.faiss_index.load(indices_dir)

    champion_ranker_path = exp_models_dir / "ranker_lgbm_exp3.txt"
    engine.ranker.load(champion_ranker_path)

    # Warmup
    for u in [1, 2, 3, 4, 5]:
        engine.recommend(user_idx=u, top_k=20)

    # Sample users across different activity percentiles
    all_users = list(user_histories.keys())
    rng = np.random.default_rng(42)
    sample_users = rng.choice(all_users, size=num_queries, replace=False)

    latencies_ms = []
    logger.info(f"Measuring latency across {num_queries} end-to-end recommendation calls...")
    t_start = time.perf_counter()

    for u in sample_users:
        t0 = time.perf_counter()
        res = engine.recommend(user_idx=int(u), top_k=20)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        assert len(res["recommendations"]) == 20

    total_time = time.perf_counter() - t_start
    qps = num_queries / total_time

    p50 = float(np.percentile(latencies_ms, 50))
    p90 = float(np.percentile(latencies_ms, 90))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))

    benchmark_results = {
        "num_queries": num_queries,
        "total_time_seconds": round(total_time, 3),
        "throughput_qps": round(qps, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50, 2),
        "p90_latency_ms": round(p90, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "device": str(engine.device),
    }

    out_path = Path("artifacts/experiments/relevance_v2/latency_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n" + "=" * 60)
    print("      CINEPULSE LOCAL SERVING LATENCY BENCHMARK (V2)")
    print("=" * 60)
    print(f"Device:           {engine.device}")
    print(f"Total Queries:    {num_queries}")
    print(f"Throughput:       {qps:.2f} queries/sec")
    print(f"Mean Latency:     {mean_lat:.2f} ms")
    print(f"P50 Latency:      {p50:.2f} ms")
    print(f"P90 Latency:      {p90:.2f} ms")
    print(f"P95 Latency:      {p95:.2f} ms")
    print(f"P99 Latency:      {p99:.2f} ms")
    print("=" * 60)

    assert p95 < 50.0, f"P95 latency {p95:.2f} ms exceeds production SLA of 50.0 ms!"
    logger.info("Serving latency benchmark PASSED successfully!")


if __name__ == "__main__":
    run_latency_benchmark()
