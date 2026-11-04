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
    compute_bootstrap_ci,
    compute_catalog_coverage,
    compute_graded_ndcg_at_k,
    compute_intra_list_diversity,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_novelty,
    compute_paired_bootstrap_ci,
    compute_recall_at_k,
)
from src.models.baselines import (
    ContentRecommender,
    ItemCollaborativeFiltering,
    PopularityRecommender,
)
from src.models.ranker import LightGBMRanker
from src.models.two_tower import TwoTowerRecommender
from src.ranking.feature_engineering import RankingFeatureExtractor
from src.reranking.mmr import MMRReRanker
from src.retrieval.faiss_index import FaissCandidateIndex


def run_evaluation(
    data_dir: Path = Path("data/processed"),
    artifacts_dir: Path = Path("artifacts"),
    num_eval_users: int = 500,
    split: str = "test",
    cohort: str = "dev",
    mmr_lambda: float = settings.default_mmr_lambda,
    mmr_novelty_weight: float = settings.novelty_weight,
):
    metrics_dir = artifacts_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not settings.force_cpu else "cpu")
    logger.info(f"=== Starting Comprehensive Multi-Stage Offline Evaluation ({split.upper()} - {cohort.upper()} COHORT) on {device} ===")

    # 1. Load data & ground truth
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    train_ratings = pd.read_parquet(data_dir / "ratings_train.parquet")
    target_ratings_file = data_dir / f"ratings_{split}.parquet"
    eval_ratings = pd.read_parquet(target_ratings_file)

    with open(data_dir / "mappings.json") as f:
        mappings = json.load(f)
    with open(data_dir / "user_train_histories.json") as f:
        user_histories = {int(k): v for k, v in json.load(f).items()}

    num_users = mappings["num_users"]
    num_movies = mappings["num_movies"]
    genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]

    # Multi-granularity ground truth sets:
    # a. Standard benchmark: rating >= 3.0 (matches prior literature & backward comparability)
    # b. Strict positive-only: rating >= 4.0 (endorsement benchmark, 1-3 stars NOT credited)
    # c. Graded ratings: dict[movie_idx, rating] for graded DCG (ratings 1-2 give 0 gain)
    ground_truth_std: dict[int, set[int]] = {}
    ground_truth_pos: dict[int, set[int]] = {}
    ground_truth_ratings: dict[int, dict[int, float]] = {}

    for _, row in eval_ratings.iterrows():
        u = int(row["user_idx"])
        m = int(row["movie_idx"])
        r = float(row["rating"])

        if u not in ground_truth_ratings:
            ground_truth_ratings[u] = {}
        ground_truth_ratings[u][m] = r

        if r >= 3.0:
            if u not in ground_truth_std:
                ground_truth_std[u] = set()
            ground_truth_std[u].add(m)

        if r >= 4.0:
            if u not in ground_truth_pos:
                ground_truth_pos[u] = set()
            ground_truth_pos[u].add(m)

    all_eligible = [u for u in ground_truth_std if u in user_histories]
    if cohort == "untouched":
        # Untouched evaluation cohort: distinct users 500-1000 never used during debugging
        eval_users = all_eligible[num_eval_users : 2 * num_eval_users]
    else:
        eval_users = all_eligible[:num_eval_users]

    logger.info(f"Evaluating across {len(eval_users)} {split} users ({cohort} cohort)...")

    # 2. Fit Baselines
    logger.info("Fitting baseline recommenders...")
    pop_model = PopularityRecommender()
    pop_model.fit(train_ratings, num_movies=num_movies)

    item_cf = ItemCollaborativeFiltering()
    item_cf.fit(train_ratings, num_users=num_users, num_movies=num_movies)

    content_model = ContentRecommender()
    content_model.fit(movies_df, genre_cols=genre_cols)

    # 3. Load Two-Tower Model & FAISS Index
    logger.info("Loading Two-Tower model and FAISS ANN index...")
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

    # 4. Load LightGBM Ranker
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
        default_lambda=0.65,
        default_novelty_weight=0.15,
    )

    # Precompute genre matrix for ILD
    item_genres = movies_df[genre_cols].values.astype(np.float32)
    item_counts = movies_df["rating_count"].values.astype(np.float32)
    total_ratings_count = len(train_ratings)

    # Models to benchmark
    model_names = [
        "Popularity Baseline",
        "Item-based CF",
        "Content-based",
        "Two-Tower (ANN Retrieval)",
        "Two-Tower + LightGBM Ranker",
        "Full Pipeline (Retrieval + Ranker + MMR)",
    ]

    results_by_model: dict[str, dict] = {name: {} for name in model_names}
    per_user_ndcgs: dict[str, list[float]] = {}
    per_user_recalls: dict[str, list[float]] = {}

    # Evaluate each approach
    for model_name in model_names:
        logger.info(f"Evaluating {model_name}...")
        recalls_20, recalls_100 = [], []
        recalls_20_pos = []
        ndcgs_10 = []
        graded_ndcgs_10 = []
        mrrs_10 = []
        all_recs_20 = []
        all_recs_100 = []
        ilds = []
        novelties = []
        latencies_ms = []

        for u in eval_users:
            hist = user_histories.get(u, [])
            consumed = set(hist)
            gt_std = ground_truth_std.get(u, set())
            gt_pos = ground_truth_pos.get(u, set())
            gt_ratings = ground_truth_ratings.get(u, {})

            t0 = time.perf_counter()

            if model_name == "Popularity Baseline":
                recs = [idx for idx, _ in pop_model.recommend(hist, top_k=100)]
            elif model_name == "Item-based CF":
                # Use full training history (not just last-10) for maximum CF signal
                recs = [idx for idx, _ in item_cf.recommend(hist, top_k=100)]
                if len(recs) < 100:
                    pop_fill = [idx for idx, _ in pop_model.recommend(hist, top_k=100) if idx not in set(recs)]
                    recs.extend(pop_fill[:100 - len(recs)])
            elif model_name == "Content-based":
                recs = [idx for idx, _ in content_model.recommend(hist, top_k=100)]
            elif model_name == "Two-Tower (ANN Retrieval)":
                # User embedding query
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
                recs = [idx for idx, _ in faiss_index.search(u_emb, top_k=100, exclude_item_indices=consumed)]
            elif model_name == "Two-Tower + LightGBM Ranker":
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

                # Hybrid retrieval: ANN + Item-CF + Popularity (matches production engine)
                candidate_sources: dict[int, set[str]] = {}
                cf_score_map: dict[int, float] = {}

                candidates = faiss_index.search(u_emb, top_k=100, exclude_item_indices=consumed)
                for rank_pos, (item_idx, score) in enumerate(candidates):
                    candidate_sources.setdefault(item_idx, set()).add("two_tower")

                # Item-CF candidates (matching 100 candidate pool in train_ranker & serving)
                cf_cands = item_cf.recommend(hist, top_k=100)
                for rank_pos, (item_idx, score) in enumerate(cf_cands):
                    if item_idx not in consumed:
                        candidate_sources.setdefault(item_idx, set()).add("item_cf")
                        cf_score_map[item_idx] = float(score)

                # Popularity fill
                for rank_pos, (item_idx, score) in enumerate(pop_model.recommend(hist, top_k=30)):
                    if item_idx not in consumed:
                        candidate_sources.setdefault(item_idx, set()).add("popularity")

                c_indices = list(candidate_sources.keys())
                # Genuine Two-Tower dot-product similarity
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

                cf_scores_dict = {c: cf_score_map.get(c, 0.0) for c in c_indices}
                sources = {c: candidate_sources[c] for c in c_indices}

                feat_df = feature_extractor.extract_features(
                    user_idx=u,
                    user_history=hist,
                    candidate_items=c_indices,
                    two_tower_scores=tt_scores,
                    cf_scores=cf_scores_dict,
                    retrieval_sources=sources,
                )
                if ranker.is_fitted and not feat_df.empty:
                    ranked = ranker.rank(feat_df)
                    recs = [idx for idx, _ in ranked]
                else:
                    recs = c_indices
            elif model_name == "Full Pipeline (Retrieval + Ranker + MMR)":
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

                # Hybrid retrieval (matches production engine)
                fp_sources: dict[int, set[str]] = {}
                fp_cf_map: dict[int, float] = {}

                fp_candidates = faiss_index.search(u_emb, top_k=100, exclude_item_indices=consumed)
                for rank_pos, (item_idx, score) in enumerate(fp_candidates):
                    fp_sources.setdefault(item_idx, set()).add("two_tower")

                for rank_pos, (item_idx, score) in enumerate(item_cf.recommend(hist, top_k=100)):
                    if item_idx not in consumed:
                        fp_sources.setdefault(item_idx, set()).add("item_cf")
                        fp_cf_map[item_idx] = float(score)

                for rank_pos, (item_idx, score) in enumerate(pop_model.recommend(hist, top_k=30)):
                    if item_idx not in consumed:
                        fp_sources.setdefault(item_idx, set()).add("popularity")

                fp_indices = list(fp_sources.keys())
                cand_arr = np.array(fp_indices, dtype=np.int64)
                if item_embeddings is not None and len(cand_arr) > 0:
                    valid_mask = cand_arr < len(item_embeddings)
                    cand_embs = item_embeddings[cand_arr[valid_mask]]
                    dots = np.dot(cand_embs, u_emb)
                    fp_tt_scores = {int(idx_c): float(sim) for idx_c, sim in zip(cand_arr[valid_mask], dots)}
                else:
                    fp_tt_scores = {c: 0.0 for c in fp_indices}
                for c in fp_indices:
                    if c not in fp_tt_scores:
                        fp_tt_scores[c] = 0.0

                fp_cf_scores = {c: fp_cf_map.get(c, 0.0) for c in fp_indices}
                sources = {c: fp_sources[c] for c in fp_indices}

                feat_df = feature_extractor.extract_features(
                    user_idx=u,
                    user_history=hist,
                    candidate_items=fp_indices,
                    two_tower_scores=fp_tt_scores,
                    cf_scores=fp_cf_scores,
                    retrieval_sources=sources,
                )
                ranked = ranker.rank(feat_df) if ranker.is_fitted and not feat_df.empty else [(idx, fp_tt_scores.get(idx, 0.0)) for idx in fp_indices]
                reranked = reranker.rerank(ranked, top_k=100, lambda_param=mmr_lambda, novelty_weight=mmr_novelty_weight)
                recs = [item["movie_idx"] for item in reranked]

            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            # Standard benchmark (rating >= 3.0)
            recalls_20.append(compute_recall_at_k(recs, gt_std, k=20))
            recalls_100.append(compute_recall_at_k(recs, gt_std, k=100))
            ndcgs_10.append(compute_ndcg_at_k(recs, gt_std, k=10))
            mrrs_10.append(compute_mrr_at_k(recs, gt_std, k=10))

            # Graded NDCG (5★=3, 4★=2, 3★=1, 1-2★=0)
            graded_ndcgs_10.append(compute_graded_ndcg_at_k(recs, gt_ratings, k=10))

            # Strict Positive-Only Recall (rating >= 4.0)
            if gt_pos:
                recalls_20_pos.append(compute_recall_at_k(recs, gt_pos, k=20))

            all_recs_20.append(recs[:20])
            all_recs_100.append(recs[:100])
            ilds.append(compute_intra_list_diversity(recs[:20], item_genres))
            novelties.append(compute_novelty(recs[:20], item_counts, total_ratings_count))

        coverage_20 = compute_catalog_coverage(all_recs_20, num_movies)
        coverage_100 = compute_catalog_coverage(all_recs_100, num_movies)

        per_user_ndcgs[model_name] = ndcgs_10
        per_user_recalls[model_name] = recalls_20

        # Compute empirical user-level 95% bootstrap confidence intervals
        ndcg_mean, ndcg_ci_l, ndcg_ci_u = compute_bootstrap_ci(ndcgs_10)
        rec20_mean, rec20_ci_l, rec20_ci_u = compute_bootstrap_ci(recalls_20)
        gndcg_mean, gndcg_ci_l, gndcg_ci_u = compute_bootstrap_ci(graded_ndcgs_10)

        results_by_model[model_name] = {
            "recall@20": round(float(np.mean(recalls_20)), 4),
            "recall@20_ci95": [rec20_ci_l, rec20_ci_u],
            "recall@100": round(float(np.mean(recalls_100)), 4),
            "ndcg@10": round(float(np.mean(ndcgs_10)), 4),
            "ndcg@10_ci95": [ndcg_ci_l, ndcg_ci_u],
            "graded_ndcg@10": round(float(np.mean(graded_ndcgs_10)), 4),
            "graded_ndcg@10_ci95": [gndcg_ci_l, gndcg_ci_u],
            "recall@20_positive_only": round(float(np.mean(recalls_20_pos)), 4) if recalls_20_pos else 0.0,
            "positive_only_eval_users": len(recalls_20_pos),
            "mrr@10": round(float(np.mean(mrrs_10)), 4),
            "catalog_coverage@20": round(float(coverage_20), 4),
            "catalog_coverage@100": round(float(coverage_100), 4),
            "intra_list_diversity": round(float(np.mean(ilds)), 4),
            "novelty_bits": round(float(np.mean(novelties)), 2),
            "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        }

    # Compute paired user-level bootstrap differences between Full Pipeline and Item-CF
    fp_key = "Full Pipeline (Retrieval + Ranker + MMR)"
    cf_key = "Item-based CF"
    if fp_key in per_user_ndcgs and cf_key in per_user_ndcgs:
        d_ndcg, d_ndcg_l, d_ndcg_u, p_ndcg = compute_paired_bootstrap_ci(
            per_user_ndcgs[fp_key], per_user_ndcgs[cf_key]
        )
        d_rec, d_rec_l, d_rec_u, p_rec = compute_paired_bootstrap_ci(
            per_user_recalls[fp_key], per_user_recalls[cf_key]
        )
        results_by_model[fp_key]["paired_delta_vs_item_cf"] = {
            "delta_ndcg@10": d_ndcg,
            "delta_ndcg@10_ci95": [d_ndcg_l, d_ndcg_u],
            "p_value_ndcg@10": p_ndcg,
            "delta_recall@20": d_rec,
            "delta_recall@20_ci95": [d_rec_l, d_rec_u],
            "p_value_recall@20": p_rec,
        }
        logger.info(
            f"=== Paired Bootstrap Analysis (Full Pipeline vs Item-CF) on {cohort.upper()} ===\n"
            f"  Delta NDCG@10 : {d_ndcg:+.4f} (95% CI: [{d_ndcg_l:+.4f}, {d_ndcg_u:+.4f}], p={p_ndcg:.4f})\n"
            f"  Delta Recall@20: {d_rec:+.4f} (95% CI: [{d_rec_l:+.4f}, {d_rec_u:+.4f}], p={p_rec:.4f})"
        )

    # Save detailed JSON metrics
    metrics_path = metrics_dir / f"evaluation_results_{split}_{cohort}.json"
    with open(metrics_path, "w") as f:
        json.dump(results_by_model, f, indent=2)
    if split == "test" and cohort == "dev":
        with open(metrics_dir / "evaluation_results.json", "w") as f:
            json.dump(results_by_model, f, indent=2)

    logger.info(f"=== Multi-Stage Offline Evaluation Results Saved to {metrics_path} ===")
    df_results = pd.DataFrame(results_by_model).T
    print("\n" + df_results.to_string() + "\n")
    return results_by_model


def run_multi_positive_evaluation(
    data_dir: Path = Path("data/processed"),
    artifacts_dir: Path = Path("artifacts"),
    num_eval_users: int = 300,
    holdout_k: int = 5,
):
    """
    Evaluates recommendation models on a multi-positive temporal holdout protocol.
    Under standard leave-one-out (|GT|=1), binary NDCG and graded NDCG are mathematically
    identical because the scalar relevance gain cancels out in DCG / IDCG.
    In this multi-positive protocol, the last 5 interactions per user (comprising diverse
    ratings from 1.0 to 5.0) are held out. Models are evaluated on their ability to rank
    5-star and 4-star items above 3-star and disliked items, revealing the genuine difference
    between binary NDCG and graded NDCG.
    """
    metrics_dir = artifacts_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not settings.force_cpu else "cpu")
    logger.info(f"=== Starting Multi-Positive Graded Evaluation (Holdout K={holdout_k}) on {device} ===")

    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    train_ratings = pd.read_parquet(data_dir / "ratings_train.parquet")
    test_ratings = pd.read_parquet(data_dir / "ratings_test.parquet")
    val_ratings = pd.read_parquet(data_dir / "ratings_val.parquet")

    # Combine all ratings to reconstruct chronologically complete user histories
    all_ratings = pd.concat([train_ratings, val_ratings, test_ratings], ignore_index=True)
    all_ratings.sort_values(by=["user_idx", "timestamp"], inplace=True)

    with open(data_dir / "mappings.json") as f:
        mappings = json.load(f)

    num_users = mappings["num_users"]
    num_movies = mappings["num_movies"]
    genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]
    item_genres = movies_df[genre_cols].values.astype(np.float32)

    # Filter users with at least 30 interactions for robust sequence split
    user_counts = all_ratings["user_idx"].value_counts()
    eligible_users = user_counts[user_counts >= 30].index.tolist()[:num_eval_users]

    # Pre-split each user's history into train_hist and multi-item holdout
    user_train_hists = {}
    user_gt_ratings = {}
    user_gt_std = {}
    user_gt_pos = {}

    grouped = all_ratings[all_ratings["user_idx"].isin(eligible_users)].groupby("user_idx")
    for u, group in grouped:
        m_list = group["movie_idx"].tolist()
        r_list = group["rating"].tolist()

        train_h = m_list[:-holdout_k]
        test_m = m_list[-holdout_k:]
        test_r = r_list[-holdout_k:]

        user_train_hists[u] = train_h
        gt_ratings_map = {int(m): float(r) for m, r in zip(test_m, test_r)}
        user_gt_ratings[u] = gt_ratings_map
        user_gt_std[u] = {m for m, r in gt_ratings_map.items() if r >= 3.0}
        user_gt_pos[u] = {m for m, r in gt_ratings_map.items() if r >= 4.0}

    # Baselines
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
        default_lambda=0.85,
        default_novelty_weight=0.03,
    )

    models_to_test = [
        "Item-based CF",
        "Two-Tower + LightGBM Ranker",
        "Full Pipeline (Retrieval + Ranker + MMR)",
    ]

    multi_results = {}
    for model_name in models_to_test:
        ndcgs_binary = []
        ndcgs_graded = []
        recalls_20 = []
        recalls_20_pos = []

        for u in eligible_users:
            hist = user_train_hists[u]
            consumed = set(hist)
            gt_std = user_gt_std[u]
            gt_pos = user_gt_pos[u]
            gt_ratings = user_gt_ratings[u]

            if model_name == "Item-based CF":
                recs = [idx for idx, _ in item_cf.recommend(hist, top_k=100)]
                if len(recs) < 100:
                    pop_fill = [idx for idx, _ in pop_model.recommend(hist, top_k=100) if idx not in set(recs)]
                    recs.extend(pop_fill[:100 - len(recs)])
            else:
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

                cand_sources: dict[int, set[str]] = {}
                cf_map: dict[int, float] = {}
                for idx_c, s in faiss_index.search(u_emb, top_k=100, exclude_item_indices=consumed):
                    cand_sources.setdefault(idx_c, set()).add("two_tower")
                for idx_c, s in item_cf.recommend(hist, top_k=100):
                    if idx_c not in consumed:
                        cand_sources.setdefault(idx_c, set()).add("item_cf")
                        cf_map[idx_c] = float(s)
                for idx_c, s in pop_model.recommend(hist, top_k=30):
                    if idx_c not in consumed:
                        cand_sources.setdefault(idx_c, set()).add("popularity")

                c_indices = list(cand_sources.keys())
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

                cf_scores_dict = {c: cf_map.get(c, 0.0) for c in c_indices}
                feat_df = feature_extractor.extract_features(
                    user_idx=u,
                    user_history=hist,
                    candidate_items=c_indices,
                    two_tower_scores=tt_scores,
                    cf_scores=cf_scores_dict,
                    retrieval_sources=cand_sources,
                )
                ranked = ranker.rank(feat_df) if ranker.is_fitted and not feat_df.empty else [(idx, tt_scores.get(idx, 0.0)) for idx in c_indices]

                if model_name == "Full Pipeline (Retrieval + Ranker + MMR)":
                    reranked = reranker.rerank(ranked, top_k=100, lambda_param=0.85, novelty_weight=0.03)
                    recs = [item["movie_idx"] for item in reranked]
                else:
                    recs = [idx for idx, _ in ranked]

            ndcgs_binary.append(compute_ndcg_at_k(recs, gt_std, k=10))
            ndcgs_graded.append(compute_graded_ndcg_at_k(recs, gt_ratings, k=10))
            recalls_20.append(compute_recall_at_k(recs, gt_std, k=20))
            if gt_pos:
                recalls_20_pos.append(compute_recall_at_k(recs, gt_pos, k=20))

        multi_results[model_name] = {
            "status": "COMPROMISED_60PCT_TRAIN_LEAKAGE",
            "binary_ndcg@10": round(float(np.mean(ndcgs_binary)), 4),
            "graded_ndcg@10": round(float(np.mean(ndcgs_graded)), 4),
            "graded_vs_binary_delta": round(float(np.mean(ndcgs_graded) - np.mean(ndcgs_binary)), 4),
            "recall@20_all_relevant": round(float(np.mean(recalls_20)), 4),
            "recall@20_positive_only": round(float(np.mean(recalls_20_pos)), 4) if recalls_20_pos else 0.0,
            "eval_users": len(eligible_users),
        }

    output_payload = {
        "_metadata": {
            "status": "INVALID_METHODOLOGICALLY_COMPROMISED",
            "leakage_rate": "60.0% (15,858 of 26,430 test pairs were observed in ratings_train.parquet)",
            "warning": "DO NOT USE FOR VALID BENCHMARK CLAIMS. Models were trained under leave-last-2-out.",
            "reproducible_fix_procedure": "To create a clean multi-positive benchmark: 1) Run data preparation with K=5 temporal holdout; 2) Retrain Two-Tower on the clean K=5 train split; 3) Rebuild FAISS index; 4) Retrain LightGBM on K=5 candidate pools; 5) Run evaluate.py against the genuine K=5 holdout.",
        },
        **multi_results,
    }

    save_path = metrics_dir / "multi_positive_graded_results.json"
    with open(save_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    logger.warning("=== WARNING: Multi-Positive Graded Benchmark is Methodologically Compromised (60% training overlap) ===")
    logger.info(f"=== Multi-Positive Graded Results Saved to {save_path} ===")
    df_multi = pd.DataFrame(multi_results).T
    print("\n[WARNING: 60% TRAINING CONTAMINATION - NOT A VALID HOLD-OUT BENCHMARK]")
    print(df_multi.to_string() + "\n")
    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Recommendation Models Offline")
    parser.add_argument("--num-users", type=int, default=500)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--cohort", type=str, default="dev", choices=["dev", "untouched"])
    parser.add_argument("--mmr-lambda", type=float, default=settings.default_mmr_lambda)
    parser.add_argument("--mmr-novelty", type=float, default=settings.novelty_weight)
    parser.add_argument("--multi-positive", action="store_true", help="Run multi-positive graded evaluation")
    args = parser.parse_args()

    if args.multi_positive:
        run_multi_positive_evaluation(num_eval_users=args.num_users)
    else:
        run_evaluation(
            num_eval_users=args.num_users,
            split=args.split,
            cohort=args.cohort,
            mmr_lambda=args.mmr_lambda,
            mmr_novelty_weight=args.mmr_novelty,
        )
