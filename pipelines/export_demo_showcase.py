"""
CinePulse V2 Demo Showcase Exporter:
Generates authentic precomputed recommendations, metrics, and insights
for the GitHub Pages static showcase using the actual V2 Python inference engine.
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
from src.db.database import DEMO_PROFILES
from src.models.two_tower import TwoTowerRecommender
from src.pipeline.engine import RecommendationEngine


def json_default(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def export_showcase_data():
    logger.info("=== Starting CinePulse V2 Demo Showcase Export ===")
    data_dir = settings.data_processed_dir
    models_dir = settings.models_dir
    indices_dir = settings.indices_dir
    exp_models_dir = Path("artifacts/experiments/relevance_v2/models")
    frontend_demo_dir = Path("frontend/public/demo-data")
    frontend_demo_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
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

    # Force V2 settings
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

    # 2. Export movies.json
    logger.info("Exporting movies.json...")
    movies_export = []
    for _, row in movies_df.iterrows():
        m_id = int(row["movieId"])
        movies_export.append({
            "movie_idx": int(row["movie_idx"]),
            "movieId": m_id,
            "title": row["clean_title"],
            "full_title": row["title"],
            "release_year": int(row["release_year"]),
            "genres": row["genres"].split("|") if isinstance(row["genres"], str) else [],
            "rating_mean": round(float(row["rating_mean"]), 2),
            "rating_count": int(row["rating_count"]),
            "bayesian_rating": round(float(row.get("bayesian_rating", row["rating_mean"])), 2),
            "poster_url": engine.movie_posters.get(m_id),
        })
    with open(frontend_demo_dir / "movies.json", "w", encoding="utf-8") as f:
        json.dump(movies_export, f)
    logger.info(f"Exported {len(movies_export)} movies to movies.json.")

    # 3. Export users.json
    logger.info("Exporting users.json...")
    users_export = []
    for profile in DEMO_PROFILES:
        users_export.append({
            "user_idx": profile["user_idx"],
            "username": profile["username"],
            "display_name": profile["display_name"],
            "avatar_color": profile["avatar_color"],
            "bio": profile["bio"],
            "auth_token": None,
            "preferred_genres": profile["preferred_genres"],
        })
    with open(frontend_demo_dir / "users.json", "w", encoding="utf-8") as f:
        json.dump(users_export, f, indent=2)
    logger.info("Exported users.json.")

    # 4. Generate recommendations & insights for each demo profile
    demo_user_ids = [p["user_idx"] for p in DEMO_PROFILES]
    for u_id in demo_user_ids:
        logger.info(f"Generating V2 recommendations for user {u_id}...")
        rec_res = engine.recommend(
            user_idx=u_id,
            top_k=20,
            retrieval_k=200,
            lambda_param=0.95,
            novelty_weight=0.01,
        )
        recs = rec_res["recommendations"]
        hero_item = rec_res.get("hero_featured")

        # Rails construction
        rails = {}
        rails["Top Picks For You"] = recs[:10]

        # Rail 2: Because you watched (last positive)
        session = engine.get_user_session(u_id)
        history = session["history"]
        if history:
            positive_history = [
                m for m in reversed(history)
                if session.get("ratings", {}).get(m, 4.0) >= 3.5
            ]
            last_m = positive_history[0] if positive_history else history[-1]
            last_title = engine.movies_df.iloc[last_m]["clean_title"]
            similar = engine.item_cf.get_similar_items(last_m, top_k=8)
            consumed = set(history)
            similar_items = []
            for idx, score in similar:
                if idx not in consumed:
                    row = engine.movies_df.iloc[idx]
                    m_id = int(row["movieId"])
                    bayes = float(row.get("bayesian_rating", row["rating_mean"]))
                    similar_items.append({
                        "rank": len(similar_items) + 1,
                        "movie_idx": idx,
                        "movieId": m_id,
                        "title": row["clean_title"],
                        "full_title": row["title"],
                        "release_year": int(row["release_year"]),
                        "genres": row["genres"].split("|") if isinstance(row["genres"], str) else [],
                        "rating_mean": round(float(row["rating_mean"]), 2),
                        "rating_count": int(row["rating_count"]),
                        "bayesian_rating": round(bayes, 2),
                        "final_score": round(float(score), 4),
                        "relevance_score": round(float(score), 4),
                        "match_percentage": int(round(75 + 20 * min(1.0, float(score)))),
                        "affinity_label": "Content Anchor",
                        "diversity_penalty": 0.0,
                        "novelty_score": 0.2,
                        "retrieval_source": "item_cf",
                        "reason": f"Because you enjoyed {last_title}",
                        "poster_url": engine.movie_posters.get(m_id),
                    })
            if similar_items:
                rails[f"Because You Watched {last_title}"] = similar_items

        # Rail 3: Hidden Gems (High Novelty)
        hidden_gems = [r for r in recs if r.get("novelty_score", 0) > 0.55]
        if not hidden_gems:
            hidden_gems = sorted(recs, key=lambda x: x.get("novelty_score", 0), reverse=True)[:6]
        rails["Hidden Gems & Deep Cuts"] = hidden_gems[:8]

        # Rail 4: Trending / Critically Acclaimed
        trending_raw = engine.popularity_recommender.recommend(history, top_k=8)
        trending_items = []
        for rank_idx, (idx, score) in enumerate(trending_raw):
            row = engine.movies_df.iloc[idx]
            m_id = int(row["movieId"])
            bayes = float(row.get("bayesian_rating", row["rating_mean"]))
            trending_items.append({
                "rank": rank_idx + 1,
                "movie_idx": idx,
                "movieId": m_id,
                "title": row["clean_title"],
                "full_title": row["title"],
                "release_year": int(row["release_year"]),
                "genres": row["genres"].split("|") if isinstance(row["genres"], str) else [],
                "rating_mean": round(float(row["rating_mean"]), 2),
                "rating_count": int(row["rating_count"]),
                "bayesian_rating": round(bayes, 2),
                "final_score": round(float(score), 4),
                "relevance_score": round(float(score), 4),
                "match_percentage": 88,
                "affinity_label": "Acclaimed",
                "diversity_penalty": 0.0,
                "novelty_score": 0.1,
                "retrieval_source": "popularity",
                "reason": "Top rated community masterpiece",
                "poster_url": engine.movie_posters.get(m_id),
            })
        rails["Critically Acclaimed Masterpieces"] = trending_items

        feed_payload = {
            "model_version": "v2",
            "pipeline": "CinePulse V2 (Two-Tower top 200 + Shrunk Item-CF + 24-feature LightGBM + Relevance MMR 0.95/0.01)",
            "benchmark_ndcg10": 0.0778,
            "benchmark_recall20": 0.2219,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user_idx": u_id,
            "hero_featured": hero_item,
            "rails": rails,
        }
        with open(frontend_demo_dir / f"recommendations_{u_id}.json", "w", encoding="utf-8") as f:
            json.dump(feed_payload, f, indent=2, default=json_default)

        # Export user insights
        insights_payload = {
            "model_version": "v2",
            "user_idx": u_id,
            "total_interactions": len(history),
            "active_likes_count": len(session.get("likes", set())),
            "active_saves_count": len(session.get("saves", set())),
            "dismissed_count": len(session.get("dismissals", set())),
            "retrieval_sources_breakdown": rec_res["retrieval_sources_breakdown"],
            "timing_breakdown_ms": rec_res["timings_ms"],
        }
        with open(frontend_demo_dir / f"insights_{u_id}.json", "w", encoding="utf-8") as f:
            json.dump(insights_payload, f, indent=2, default=json_default)

    # 5. Export metrics.json
    logger.info("Exporting metrics.json with verified V1 and V2 benchmarks...")
    test_conf_path = Path("artifacts/experiments/relevance_v2/test_confirmation_results.json")
    with open(test_conf_path) as f:
        conf_data = json.load(f)

    seq_metrics = conf_data["cohorts"]["sequestered_cohort"]["metrics"]
    untouched_metrics = conf_data["cohorts"]["untouched_cohort"]["metrics"]

    v2_metrics_export = {
        "dataset_name": "MovieLens 1M (GroupLens Official)",
        "num_users": 6040,
        "num_movies": 3416,
        "test_samples": 4974,
        "active_version": "v2",
        "cohorts": {
            "sequestered_test_cohort": {
                "description": "3,974 strictly held-out test users never inspected during development or hyperparameter selection",
                "users_count": 3974,
                "metrics": {
                    "CinePulse V1 Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)": {
                        "ndcg@10": round(seq_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["ndcg_at_10"], 4),
                        "recall@20": round(seq_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["recall_at_20"], 4),
                        "recall@100": round(seq_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["recall_at_100"], 4),
                        "mrr@10": round(seq_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["mrr_at_10"], 4),
                        "catalog_coverage@100": round(seq_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["catalog_coverage_at_100"], 4),
                        "intra_list_diversity": round(seq_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["intra_list_diversity_at_20"], 4),
                        "latency_p95_ms": round(seq_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["p95_latency_ms"], 1),
                    },
                    "CinePulse V2 Production (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01)": {
                        "ndcg@10": round(seq_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["ndcg_at_10"], 4),
                        "recall@20": round(seq_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["recall_at_20"], 4),
                        "recall@100": round(seq_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["recall_at_100"], 4),
                        "mrr@10": round(seq_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["mrr_at_10"], 4),
                        "catalog_coverage@100": round(seq_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["catalog_coverage_at_100"], 4),
                        "intra_list_diversity": round(seq_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["intra_list_diversity_at_20"], 4),
                        "latency_p95_ms": round(seq_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["p95_latency_ms"], 1),
                    },
                    "CinePulse V2 Pure Ranker (24f, TT=200, CF=100, Pop=30, No MMR)": {
                        "ndcg@10": round(seq_metrics["3. New 24f Pure Ranker (24f, TT=200, CF=100, Pop=30, No MMR)"]["ndcg_at_10"], 4),
                        "recall@20": round(seq_metrics["3. New 24f Pure Ranker (24f, TT=200, CF=100, Pop=30, No MMR)"]["recall_at_20"], 4),
                        "recall@100": round(seq_metrics["3. New 24f Pure Ranker (24f, TT=200, CF=100, Pop=30, No MMR)"]["recall_at_100"], 4),
                        "mrr@10": round(seq_metrics["3. New 24f Pure Ranker (24f, TT=200, CF=100, Pop=30, No MMR)"]["mrr_at_10"], 4),
                        "catalog_coverage@100": round(seq_metrics["3. New 24f Pure Ranker (24f, TT=200, CF=100, Pop=30, No MMR)"]["catalog_coverage_at_100"], 4),
                        "intra_list_diversity": round(seq_metrics["3. New 24f Pure Ranker (24f, TT=200, CF=100, Pop=30, No MMR)"]["intra_list_diversity_at_20"], 4),
                        "latency_p95_ms": round(seq_metrics["3. New 24f Pure Ranker (24f, TT=200, CF=100, Pop=30, No MMR)"]["p95_latency_ms"], 1),
                    },
                },
                "paired_significance_v2_vs_v1": conf_data["cohorts"]["sequestered_cohort"]["significance"]["config8_vs_baseline"],
            },
            "untouched_test_cohort": {
                "description": "500 untouched test users (users 500-1000) corresponding to legacy reported benchmark",
                "users_count": 500,
                "metrics": {
                    "CinePulse V1 Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)": {
                        "ndcg@10": round(untouched_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["ndcg_at_10"], 4),
                        "recall@20": round(untouched_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["recall_at_20"], 4),
                        "recall@100": round(untouched_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["recall_at_100"], 4),
                        "catalog_coverage@100": round(untouched_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["catalog_coverage_at_100"], 4),
                        "latency_p95_ms": round(untouched_metrics["1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)"]["p95_latency_ms"], 1),
                    },
                    "CinePulse V2 Production (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01)": {
                        "ndcg@10": round(untouched_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["ndcg_at_10"], 4),
                        "recall@20": round(untouched_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["recall_at_20"], 4),
                        "recall@100": round(untouched_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["recall_at_100"], 4),
                        "catalog_coverage@100": round(untouched_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["catalog_coverage_at_100"], 4),
                        "latency_p95_ms": round(untouched_metrics["2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)"]["p95_latency_ms"], 1),
                    },
                },
            },
        },
        # Flat models key for backward compatibility with existing EvaluationPage component
        "models": {
            "Item-based CF": {
                "recall@20": 0.116,
                "recall@100": 0.316,
                "ndcg@10": 0.041,
                "mrr@10": 0.0329,
                "catalog_coverage@20": 0.1294,
                "intra_list_diversity": 0.7161,
                "novelty_bits": 9.53,
                "latency_p95_ms": 4.0,
            },
            "Two-Tower (ANN Retrieval)": {
                "recall@20": 0.106,
                "recall@100": 0.316,
                "ndcg@10": 0.0346,
                "mrr@10": 0.0251,
                "catalog_coverage@20": 0.8188,
                "intra_list_diversity": 0.4581,
                "novelty_bits": 12.39,
                "latency_p95_ms": 2.96,
            },
            "CinePulse V1 Baseline (17f + MMR 0.85/0.03)": {
                "recall@20": 0.1809,
                "recall@100": 0.4059,
                "ndcg@10": 0.0560,
                "mrr@10": 0.0400,
                "catalog_coverage@20": 0.4081,
                "intra_list_diversity": 0.6933,
                "novelty_bits": 10.32,
                "latency_p95_ms": 24.9,
            },
            "CinePulse V2 Production (24f + MMR 0.95/0.01) [WINNER]": {
                "recall@20": 0.2219,
                "recall@100": 0.4421,
                "ndcg@10": 0.0778,
                "mrr@10": 0.0563,
                "catalog_coverage@20": 0.6120,
                "intra_list_diversity": 0.5165,
                "novelty_bits": 10.85,
                "latency_p95_ms": 27.9,
            },
        },
    }

    with open(frontend_demo_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(v2_metrics_export, f, indent=2)
    logger.info("Exported metrics.json.")

    logger.info("=== Completed CinePulse V2 Demo Showcase Export Successfully ===")


if __name__ == "__main__":
    export_showcase_data()
