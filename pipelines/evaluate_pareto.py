import argparse
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
from src.evaluation.metrics import (
    compute_catalog_coverage,
    compute_intra_list_diversity,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_novelty,
    compute_recall_at_k,
)
from src.models.baselines import (
    ItemCollaborativeFiltering,
    PopularityRecommender,
)
from src.models.ranker import LightGBMRanker
from src.models.two_tower import TwoTowerRecommender
from src.ranking.feature_engineering import RankingFeatureExtractor
from src.reranking.mmr import MMRReRanker
from src.retrieval.faiss_index import FaissCandidateIndex


def run_pareto_study(
    data_dir: Path = Path("data/processed"),
    artifacts_dir: Path = Path("artifacts"),
    num_eval_users: int = 500,
    split: str = "val",
    output_file: Path | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() and not settings.force_cpu else "cpu")
    logger.info(f"=== Starting MMR Pareto Frontier Evaluation ({split.upper()}) on {device} ===")

    # Load data
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    train_ratings = pd.read_parquet(data_dir / "ratings_train.parquet")
    eval_ratings = pd.read_parquet(data_dir / f"ratings_{split}.parquet")

    with open(data_dir / "mappings.json") as f:
        mappings = json.load(f)
    with open(data_dir / "user_train_histories.json") as f:
        user_histories = {int(k): v for k, v in json.load(f).items()}

    num_users = mappings["num_users"]
    num_movies = mappings["num_movies"]
    genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]
    item_genres = movies_df[genre_cols].values.astype(np.float32)
    item_counts = movies_df["rating_count"].values.astype(np.float32)
    total_ratings_count = int(train_ratings["rating"].count())

    # Build positive ground truth (rating >= 3.0)
    ground_truth: dict[int, set[int]] = {}
    for _, row in eval_ratings.iterrows():
        u = int(row["user_idx"])
        m = int(row["movie_idx"])
        r = float(row["rating"])
        if r >= 3.0:
            ground_truth.setdefault(u, set()).add(m)

    eval_users = [u for u in ground_truth if u in user_histories][:num_eval_users]
    logger.info(f"Evaluating Pareto curve on {len(eval_users)} {split} users...")

    # Load models
    pop_model = PopularityRecommender()
    pop_model.fit(train_ratings, num_movies=num_movies)

    item_cf = ItemCollaborativeFiltering()
    item_cf.fit(train_ratings, num_users=num_users, num_movies=num_movies)

    two_tower_path = artifacts_dir / "models" / "two_tower_best.pt"
    tt_model = TwoTowerRecommender(
        num_users=num_users,
        num_movies=num_movies,
        num_genres=len(genre_cols),
        embedding_dim=settings.embedding_dim,
    ).to(device)

    if two_tower_path.exists():
        ckpt = torch.load(two_tower_path, map_location=device, weights_only=True)
        tt_model.load_state_dict(ckpt["model_state_dict"])
        tt_model.eval()

    faiss_index = FaissCandidateIndex(dimension=settings.embedding_dim)
    faiss_index.load(artifacts_dir / "indices")

    item_embeddings_path = artifacts_dir / "models" / "item_embeddings.npy"
    item_embeddings = np.load(item_embeddings_path) if item_embeddings_path.exists() else None

    ranker = LightGBMRanker()
    ranker_path = artifacts_dir / "models" / "ranker_lgbm.txt"
    if ranker_path.exists():
        ranker.load(ranker_path)

    feature_extractor = RankingFeatureExtractor(
        movies_df=movies_df,
        genre_cols=genre_cols,
        global_mean_rating=mappings.get("global_rating_mean", 3.5),
    )

    reranker = MMRReRanker(
        movies_df=movies_df,
        genre_cols=genre_cols,
    )

    # Precompute ranked candidates for all eval users (Hybrid: ANN + Item-CF + Pop -> Ranker)
    logger.info("Generating ranked candidate pools for evaluation cohort...")
    user_ranked_pools: dict[int, list[tuple[int, float]]] = {}

    for u in eval_users:
        hist = user_histories.get(u, [])
        consumed = set(hist)

        with torch.no_grad():
            padded_hist = np.full(settings.max_sequence_length, num_movies, dtype=np.int64)
            h_len = min(len(hist), settings.max_sequence_length)
            if h_len > 0:
                padded_hist[-h_len:] = hist[-h_len:]
            g_pref = item_genres[hist].mean(axis=0) if hist else np.zeros(len(genre_cols), dtype=np.float32)

            u_tensor = torch.tensor([u], dtype=torch.long, device=device)
            h_tensor = torch.tensor(padded_hist, dtype=torch.long, device=device).unsqueeze(0)
            hl_tensor = torch.tensor([h_len], dtype=torch.long, device=device)
            gp_tensor = torch.tensor(g_pref, dtype=torch.float32, device=device).unsqueeze(0)

            u_emb = tt_model.user_tower(u_tensor, h_tensor, hl_tensor, gp_tensor).cpu().numpy()[0]

        # Multi-source candidate retrieval
        candidate_sources: dict[int, set[str]] = {}
        cf_score_map: dict[int, float] = {}

        # Two-Tower
        for rank_pos, (item_idx, score) in enumerate(faiss_index.search(u_emb, top_k=100, exclude_item_indices=consumed)):
            candidate_sources.setdefault(item_idx, set()).add("two_tower")

        # Item-CF (top 100)
        for rank_pos, (item_idx, score) in enumerate(item_cf.recommend(hist, top_k=100)):
            if item_idx not in consumed:
                candidate_sources.setdefault(item_idx, set()).add("item_cf")
                cf_score_map[item_idx] = float(score)

        # Popularity
        for rank_pos, (item_idx, score) in enumerate(pop_model.recommend(hist, top_k=30)):
            if item_idx not in consumed:
                candidate_sources.setdefault(item_idx, set()).add("popularity")

        c_indices = list(candidate_sources.keys())
        cand_arr = np.array(c_indices, dtype=np.int64)
        if item_embeddings is not None and len(cand_arr) > 0:
            valid_mask = cand_arr < len(item_embeddings)
            cand_embs = item_embeddings[cand_arr[valid_mask]]
            dots = np.dot(cand_embs, u_emb)
            tt_scores = {int(idx_c): float(sim) for idx_c, sim in zip(cand_arr[valid_mask], dots)}
        else:
            tt_scores = {c: 0.0 for c in c_indices}
        for c in c_indices:
            if c not in tt_scores:
                tt_scores[c] = 0.0

        cf_scores = {c: cf_score_map.get(c, 0.0) for c in c_indices}
        sources = {c: candidate_sources[c] for c in c_indices}

        feat_df = feature_extractor.extract_features(
            user_idx=u,
            user_history=hist,
            candidate_items=c_indices,
            two_tower_scores=tt_scores,
            cf_scores=cf_scores,
            retrieval_sources=sources,
        )

        if ranker.is_fitted and not feat_df.empty:
            ranked = ranker.rank(feat_df)
        else:
            ranked = [(idx, tt_scores.get(idx, 0.0)) for idx in c_indices]

        user_ranked_pools[u] = ranked

    # Configurations to evaluate
    configurations = [
        {"name": "MMR Disabled (Pure Ranker)", "disabled": True, "lambda": 1.0, "beta": 0.0},
        {"name": "lambda=1.00, beta=0.00 (Identity MMR)", "disabled": False, "lambda": 1.00, "beta": 0.00},
        {"name": "lambda=1.00, beta=0.03 (Relevance + Low Nov)", "disabled": False, "lambda": 1.00, "beta": 0.03},
        {"name": "lambda=0.95, beta=0.00 (Mild Div, Zero Nov)", "disabled": False, "lambda": 0.95, "beta": 0.00},
        {"name": "lambda=0.90, beta=0.01 (Conservative)", "disabled": False, "lambda": 0.90, "beta": 0.01},
        {"name": "lambda=0.85, beta=0.03 (Balanced Production)", "disabled": False, "lambda": 0.85, "beta": 0.03},
        {"name": "lambda=0.75, beta=0.05 (Exploration)", "disabled": False, "lambda": 0.75, "beta": 0.05},
        {"name": "lambda=0.65, beta=0.15 (Original Aggressive)", "disabled": False, "lambda": 0.65, "beta": 0.15},
    ]

    pareto_results = []
    for cfg in configurations:
        cfg_name = cfg["name"]
        logger.info(f"Evaluating configuration: {cfg_name}...")
        recalls_20 = []
        recalls_100 = []
        ndcgs_10 = []
        mrrs_10 = []
        all_recs_20 = []
        ilds = []
        novelties = []
        latencies_ms = []

        for u in eval_users:
            ranked = user_ranked_pools[u]
            gt = ground_truth[u]

            t0 = time.perf_counter()
            if cfg.get("disabled", False):
                recs = [idx for idx, _ in ranked]
            else:
                reranked = reranker.rerank(
                    ranked,
                    top_k=100,
                    lambda_param=cfg["lambda"],
                    novelty_weight=cfg["beta"],
                )
                recs = [item["movie_idx"] for item in reranked]
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            recalls_20.append(compute_recall_at_k(recs, gt, k=20))
            recalls_100.append(compute_recall_at_k(recs, gt, k=100))
            ndcgs_10.append(compute_ndcg_at_k(recs, gt, k=10))
            mrrs_10.append(compute_mrr_at_k(recs, gt, k=10))
            all_recs_20.append(recs[:20])
            ilds.append(compute_intra_list_diversity(recs[:20], item_genres))
            novelties.append(compute_novelty(recs[:20], item_counts, total_ratings_count))

        cov_20 = compute_catalog_coverage(all_recs_20, num_movies)

        pareto_results.append({
            "Configuration": cfg_name,
            "Recall@20": round(float(np.mean(recalls_20)), 4),
            "Recall@100": round(float(np.mean(recalls_100)), 4),
            "NDCG@10": round(float(np.mean(ndcgs_10)), 4),
            "MRR@10": round(float(np.mean(mrrs_10)), 4),
            "ILD": round(float(np.mean(ilds)), 4),
            "Novelty": round(float(np.mean(novelties)), 2),
            "Coverage@20": round(float(cov_20), 4),
            "Latency_P95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        })

    pareto_df = pd.DataFrame(pareto_results)
    print("\n" + "=" * 80)
    print(f"MMR VALIDATION PARETO FRONTIER TABLE ({split.upper()})")
    print("=" * 80)
    print(pareto_df.to_string(index=False))
    print("=" * 80 + "\n")

    if output_file is None:
        output_file = artifacts_dir / "metrics" / f"pareto_study_{split}.json"
    with open(output_file, "w") as f:
        json.dump(pareto_results, f, indent=2)
    logger.info(f"Pareto study results saved to {output_file}")
    return pareto_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMR Pareto Tradeoffs")
    parser.add_argument("--num-users", type=int, default=500)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()
    run_pareto_study(num_eval_users=args.num_users, split=args.split)
