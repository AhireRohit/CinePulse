"""
CinePulse: Final Independent Confirmation on Sequestered Test Cohort.

Evaluates:
1. Published Production Baseline (TT=100, CF=100, Pop=30, 17f LGBM, MMR lambda=0.85, beta=0.03)
2. Promoted Config 8 (TT=200, CF=100, Pop=30, 24f LGBM, MMR lambda=0.95, beta=0.01, No Fusion)
3. New 24f Pure Ranker (TT=200, CF=100, Pop=30, 24f LGBM, No MMR)

Evaluates strictly on `ratings_test.parquet` (6,040 users) across:
- Sequestered Cohort: 3,974 users (all_eligible[1000:], never inspected in any prior dev/untouched run)
- All Test Cohort: 4,974 users (all_eligible, full test set)
- Historical Untouched Cohort: 500 users (all_eligible[500:1000])

Pre-records exact cohort user IDs before running evaluation.
Computes paired bootstrap significance with finite-sample p-value resolution (B=1000).
"""

import hashlib
import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from pipelines.exp1_item_cf import EnhancedItemCF
from pipelines.exp3_ranker import FastRankingFeatureExtractor
from src.core.config import settings
from src.core.logging import logger
from src.evaluation.metrics import (
    compute_catalog_coverage,
    compute_graded_ndcg_at_k,
    compute_intra_list_diversity,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_paired_bootstrap_ci,
    compute_recall_at_k,
)
from src.models.baselines import PopularityRecommender
from src.models.two_tower import TwoTowerRecommender
from src.reranking.mmr import MMRReRanker
from src.retrieval.faiss_index import FaissCandidateIndex


def run_confirmation():
    logger.info("=== Starting Final Independent Confirmation on Test Set ===")
    device = torch.device("cuda" if torch.cuda.is_available() and not settings.force_cpu else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = Path("data/processed")
    exp_dir = Path("artifacts/experiments/relevance_v2")
    models_dir = exp_dir / "models"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    train_ratings = pd.read_parquet(data_dir / "ratings_train.parquet")
    test_ratings = pd.read_parquet(data_dir / "ratings_test.parquet")

    with open(data_dir / "mappings.json") as f:
        mappings = json.load(f)
    with open(data_dir / "user_train_histories.json") as f:
        user_histories = {int(k): v for k, v in json.load(f).items()}

    num_users = mappings["num_users"]
    num_movies = mappings["num_movies"]
    genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]

    user_train_ratings: dict[int, dict[int, float]] = {}
    for row in train_ratings[["user_idx", "movie_idx", "rating"]].itertuples(index=False):
        user_train_ratings.setdefault(int(row.user_idx), {})[int(row.movie_idx)] = float(row.rating)

    # Ground truth mapping on test set
    ground_truth_std: dict[int, set[int]] = {}
    ground_truth_pos: dict[int, set[int]] = {}
    ground_truth_ratings: dict[int, dict[int, float]] = {}

    for row in test_ratings[["user_idx", "movie_idx", "rating"]].itertuples(index=False):
        u = int(row.user_idx)
        m = int(row.movie_idx)
        r = float(row.rating)
        ground_truth_ratings.setdefault(u, {})[m] = r
        if r >= 3.0:
            ground_truth_std.setdefault(u, set()).add(m)
        if r >= 4.0:
            ground_truth_pos.setdefault(u, set()).add(m)

    all_eligible = [u for u in ground_truth_std if u in user_histories]
    logger.info(f"Total eligible test users (rating >= 3.0): {len(all_eligible)} / {num_users}")

    # Cohort definitions
    dev_cohort = all_eligible[:500]
    untouched_cohort = all_eligible[500:1000]
    sequestered_cohort = all_eligible[1000:]
    all_test_cohort = all_eligible

    # Pre-record cohort manifest before running evaluation
    manifest = {
        "num_total_users": num_users,
        "num_test_interactions": len(test_ratings),
        "num_eligible_test_users_std": len(all_eligible),
        "cohort_counts": {
            "dev_cohort_500": len(dev_cohort),
            "untouched_cohort_500": len(untouched_cohort),
            "sequestered_cohort": len(sequestered_cohort),
            "all_test_cohort": len(all_test_cohort),
        },
        "cohort_checksums": {
            "dev_cohort_md5": hashlib.md5(str(dev_cohort).encode()).hexdigest(),
            "untouched_cohort_md5": hashlib.md5(str(untouched_cohort).encode()).hexdigest(),
            "sequestered_cohort_md5": hashlib.md5(str(sequestered_cohort).encode()).hexdigest(),
            "all_test_cohort_md5": hashlib.md5(str(all_test_cohort).encode()).hexdigest(),
        },
        "sequestered_user_ids": sequestered_cohort,
    }
    manifest_path = exp_dir / "test_cohort_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Pre-recorded test cohort manifest to {manifest_path}")

    # 2. Load models
    logger.info("Loading retrieval models and rankers...")
    pop_model = PopularityRecommender()
    pop_model.fit(train_ratings, num_movies=num_movies)

    cf_model = EnhancedItemCF(top_k_neighbors=200, shrinkage_lambda=25.0)
    cf_model.fit(train_ratings, num_users=num_users, num_movies=num_movies)

    two_tower_path = Path("artifacts/models/two_tower_best.pt")
    tt_model = TwoTowerRecommender(
        num_users=num_users,
        num_movies=num_movies,
        num_genres=len(genre_cols),
        embedding_dim=settings.embedding_dim,
    ).to(device)
    ckpt = torch.load(two_tower_path, map_location=device, weights_only=True)
    tt_model.load_state_dict(ckpt["model_state_dict"])
    tt_model.eval()

    faiss_index = FaissCandidateIndex(dimension=settings.embedding_dim)
    faiss_index.load(Path("artifacts/indices"))

    item_embeddings = np.load("artifacts/models/item_embeddings.npy")

    booster_base = lgb.Booster(model_file="artifacts/models/ranker_lgbm.txt")
    booster_exp = lgb.Booster(model_file=str(models_dir / "ranker_lgbm_exp3.txt"))

    fast_extractor = FastRankingFeatureExtractor(
        movies_df=movies_df,
        genre_cols=genre_cols,
        global_mean_rating=mappings.get("global_rating_mean", 3.5),
    )

    mmr_prod = MMRReRanker(movies_df, genre_cols, default_lambda=0.85, default_novelty_weight=0.03)
    mmr_rel_first = MMRReRanker(movies_df, genre_cols, default_lambda=0.95, default_novelty_weight=0.01)

    # 3. Precompute user representations on test users
    logger.info(f"Encoding Two-Tower embeddings for {len(all_eligible)} test users...")
    item_genres = movies_df[genre_cols].values.astype(np.float32)
    user_embs_dict = {}
    batch_size = 512
    with torch.no_grad():
        for start_idx in range(0, len(all_eligible), batch_size):
            batch_u = all_eligible[start_idx : start_idx + batch_size]
            B = len(batch_u)
            u_tensors = torch.tensor(batch_u, dtype=torch.long, device=device)
            h_seq = np.full((B, settings.max_sequence_length), num_movies, dtype=np.int64)
            h_len = np.zeros(B, dtype=np.int64)
            g_pref = np.zeros((B, len(genre_cols)), dtype=np.float32)

            for i, u_id in enumerate(batch_u):
                hist = user_histories.get(u_id, [])
                hlen = min(len(hist), settings.max_sequence_length)
                h_len[i] = hlen
                if hlen > 0:
                    h_seq[i, -hlen:] = hist[-hlen:]
                    valid_h = [m for m in hist if m < num_movies]
                    if valid_h:
                        g_pref[i] = item_genres[valid_h].mean(axis=0)

            user_embs = tt_model.user_tower(
                u_tensors,
                torch.tensor(h_seq, dtype=torch.long, device=device),
                torch.tensor(h_len, dtype=torch.long, device=device),
                torch.tensor(g_pref, dtype=torch.float32, device=device),
            ).cpu().numpy()

            for i, u_id in enumerate(batch_u):
                user_embs_dict[u_id] = user_embs[i]

    # 4. Generate recommendations across all 3 configurations
    configs = [
        "1. Published Production Baseline (17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)",
        "2. Promoted Config 8 (24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01, No Fusion)",
        "3. New 24f Pure Ranker (24f, TT=200, CF=100, Pop=30, No MMR)",
    ]
    recs_by_config = {c: {} for c in configs}
    lats_by_config = {c: [] for c in configs}

    logger.info(f"Generating recommendations for {len(all_eligible)} test users...")
    t_start = time.perf_counter()

    for idx_u, u in enumerate(all_eligible):
        if idx_u > 0 and idx_u % 1000 == 0:
            logger.info(f"Processed {idx_u}/{len(all_eligible)} users ({time.perf_counter() - t_start:.1f}s)...")

        h = user_histories.get(u, [])
        consumed = set(h)
        u_emb = user_embs_dict[u]
        u_ratings = user_train_ratings.get(u, {})

        # Candidate retrieval
        t0 = time.perf_counter()
        tt_search_results = faiss_index.search(u_emb, top_k=200, exclude_item_indices=consumed)
        cf_search_results = [
            (idx, float(score)) for idx, score in cf_model.recommend(h, top_k=100) if idx not in consumed
        ]
        pop_search_results = [
            (idx, float(score)) for idx, score in pop_model.recommend(h, top_k=30) if idx not in consumed
        ]

        # Base pool (TT=100, CF=100, Pop=30)
        base_sources = {}
        base_cf_map = {}
        for item_idx, _ in tt_search_results[:100]:
            base_sources.setdefault(item_idx, set()).add("two_tower")
        for item_idx, score in cf_search_results:
            base_sources.setdefault(item_idx, set()).add("item_cf")
            base_cf_map[item_idx] = score
        for item_idx, _ in pop_search_results:
            base_sources.setdefault(item_idx, set()).add("popularity")
        base_cand_list = list(base_sources.keys())

        # Expanded pool (TT=200, CF=100, Pop=30)
        exp_sources = {}
        exp_cf_map = {}
        for item_idx, _ in tt_search_results:
            exp_sources.setdefault(item_idx, set()).add("two_tower")
        for item_idx, score in cf_search_results:
            exp_sources.setdefault(item_idx, set()).add("item_cf")
            exp_cf_map[item_idx] = score
        for item_idx, _ in pop_search_results:
            exp_sources.setdefault(item_idx, set()).add("popularity")
        exp_cand_list = list(exp_sources.keys())

        # Vectorized TT dot products
        def get_tt_scores(c_list):
            cand_arr = np.array(c_list, dtype=np.int64)
            valid_mask = cand_arr < len(item_embeddings)
            tt_scores = {}
            if np.any(valid_mask):
                cand_embs = item_embeddings[cand_arr[valid_mask]]
                sims = np.dot(cand_embs, u_emb)
                for idx_c, sim in zip(cand_arr[valid_mask], sims):
                    tt_scores[int(idx_c)] = float(sim)
            for c in c_list:
                tt_scores.setdefault(c, 0.0)
            return tt_scores

        base_tt_scores = get_tt_scores(base_cand_list)
        exp_tt_scores = get_tt_scores(exp_cand_list)

        # Baseline 17 features
        base_mat_17 = fast_extractor.extract_matrix(
            u, h, u_ratings, base_cand_list, base_tt_scores, base_cf_map, base_sources, use_expanded=False
        )
        lgbm_base_scores = booster_base.predict(base_mat_17)
        base_ranked_pairs = [(base_cand_list[i], float(lgbm_base_scores[i])) for i in np.argsort(-lgbm_base_scores)]

        # Config 8: 24 features on expanded pool
        exp_mat_24 = fast_extractor.extract_matrix(
            u, h, u_ratings, exp_cand_list, exp_tt_scores, exp_cf_map, exp_sources, use_expanded=True
        )
        exp_lgbm_scores = booster_exp.predict(exp_mat_24)
        exp_ranked_pairs = [(exp_cand_list[i], float(exp_lgbm_scores[i])) for i in np.argsort(-exp_lgbm_scores)]

        # 1. Baseline: MMR 0.85/0.03
        recs_base = [x["movie_idx"] for x in mmr_prod.rerank(base_ranked_pairs, top_k=100, lambda_param=0.85, novelty_weight=0.03, excluded_item_indices=consumed)]
        recs_by_config[configs[0]][u] = recs_base
        lats_by_config[configs[0]].append((time.perf_counter() - t0) * 1000.0)

        # 2. Config 8: MMR 0.95/0.01
        recs_cfg8 = [x["movie_idx"] for x in mmr_rel_first.rerank(exp_ranked_pairs, top_k=100, lambda_param=0.95, novelty_weight=0.01, excluded_item_indices=consumed)]
        recs_by_config[configs[1]][u] = recs_cfg8
        lats_by_config[configs[1]].append((time.perf_counter() - t0) * 1000.0)

        # 3. New 24f Pure Ranker (No MMR)
        recs_pure = [c for c, _ in exp_ranked_pairs[:100]]
        recs_by_config[configs[2]][u] = recs_pure
        lats_by_config[configs[2]].append((time.perf_counter() - t0) * 1000.0)

    logger.info(f"Completed inference across all test users in {time.perf_counter() - t_start:.2f}s.")

    # 5. Metric Evaluation Helper
    def evaluate_cohort(user_subset: list[int], cohort_name: str) -> dict:
        results = {}
        pos_users = [u for u in user_subset if u in ground_truth_pos]

        for cfg in configs:
            recs_map = recs_by_config[cfg]
            rec20_list = [compute_recall_at_k(recs_map[u], ground_truth_std[u], 20) for u in user_subset]
            rec100_list = [compute_recall_at_k(recs_map[u], ground_truth_std[u], 100) for u in user_subset]
            ndcg10_list = [compute_ndcg_at_k(recs_map[u], ground_truth_std[u], 10) for u in user_subset]
            mrr10_list = [compute_mrr_at_k(recs_map[u], ground_truth_std[u], 10) for u in user_subset]
            graded10_list = [compute_graded_ndcg_at_k(recs_map[u], ground_truth_ratings[u], 10) for u in user_subset]
            pos_ndcg10_list = [compute_ndcg_at_k(recs_map[u], ground_truth_pos[u], 10) for u in pos_users]

            all_top100 = [recs_map[u][:100] for u in user_subset]
            cov100 = compute_catalog_coverage(all_top100, num_movies)

            ild_samples = [compute_intra_list_diversity(recs_map[u][:20], item_genres) for u in user_subset[:500]]
            mean_ild = float(np.mean(ild_samples))

            results[cfg] = {
                "recall_at_20": float(np.mean(rec20_list)),
                "recall_at_100": float(np.mean(rec100_list)),
                "ndcg_at_10": float(np.mean(ndcg10_list)),
                "graded_ndcg_at_10": float(np.mean(graded10_list)),
                "pos_ndcg_at_10": float(np.mean(pos_ndcg10_list)),
                "mrr_at_10": float(np.mean(mrr10_list)),
                "catalog_coverage_at_100": cov100,
                "intra_list_diversity_at_20": mean_ild,
                "p95_latency_ms": float(np.percentile(lats_by_config[cfg], 95)),
                "mean_latency_ms": float(np.mean(lats_by_config[cfg])),
                # Arrays for statistical testing
                "_user_ndcg10": ndcg10_list,
                "_user_rec20": rec20_list,
            }
        return results

    logger.info("Computing evaluation metrics across all cohorts...")
    results_sequestered = evaluate_cohort(sequestered_cohort, "Sequestered Cohort (3,974 users)")
    results_all_test = evaluate_cohort(all_test_cohort, "All Test Users (4,974 users)")
    results_untouched = evaluate_cohort(untouched_cohort, "Untouched Cohort (500 users)")

    # 6. Statistical Significance Tests on Sequestered and Full Test Sets
    def compute_significance(cohort_results: dict, cohort_size: int) -> dict:
        cfg_base = configs[0]
        cfg_winner = configs[1]
        cfg_pure = configs[2]

        # Config 8 vs Baseline
        diff_ndcg, low_ndcg, high_ndcg, p_ndcg = compute_paired_bootstrap_ci(
            cohort_results[cfg_winner]["_user_ndcg10"],
            cohort_results[cfg_base]["_user_ndcg10"],
            n_bootstrap=1000,
            seed=42,
        )
        diff_rec, low_rec, high_rec, p_rec = compute_paired_bootstrap_ci(
            cohort_results[cfg_winner]["_user_rec20"],
            cohort_results[cfg_base]["_user_rec20"],
            n_bootstrap=1000,
            seed=42,
        )

        # Config 8 vs Pure Ranker
        diff_pure_ndcg, low_pure_ndcg, high_pure_ndcg, p_pure_ndcg = compute_paired_bootstrap_ci(
            cohort_results[cfg_winner]["_user_ndcg10"],
            cohort_results[cfg_pure]["_user_ndcg10"],
            n_bootstrap=1000,
            seed=42,
        )

        return {
            "num_users": cohort_size,
            "bootstrap_resamples": 1000,
            "p_value_resolution_note": "For B=1000 resamples, minimum resolvable empirical p-value is 1/(B+1) ~ 0.001.",
            "config8_vs_baseline": {
                "ndcg_diff": diff_ndcg,
                "ndcg_ci_95": [low_ndcg, high_ndcg],
                "ndcg_p_value": p_ndcg,
                "recall_diff": diff_rec,
                "recall_ci_95": [low_rec, high_rec],
                "recall_p_value": p_rec,
            },
            "config8_vs_pure_ranker": {
                "ndcg_diff": diff_pure_ndcg,
                "ndcg_ci_95": [low_pure_ndcg, high_pure_ndcg],
                "ndcg_p_value": p_pure_ndcg,
            },
        }

    sig_sequestered = compute_significance(results_sequestered, len(sequestered_cohort))
    sig_all_test = compute_significance(results_all_test, len(all_test_cohort))
    sig_untouched = compute_significance(results_untouched, len(untouched_cohort))

    # Clean internal arrays before saving
    def clean_results(res_dict):
        cleaned = {}
        for k, v in res_dict.items():
            cleaned[k] = {ik: iv for ik, iv in v.items() if not ik.startswith("_")}
        return cleaned

    final_payload = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_dataset": "ratings_test.parquet",
        "total_test_users": len(all_eligible),
        "cohorts": {
            "sequestered_cohort": {
                "description": "3,974 test users (all_eligible[1000:]) never inspected during prior debugging or evaluation",
                "count": len(sequestered_cohort),
                "metrics": clean_results(results_sequestered),
                "significance": sig_sequestered,
            },
            "all_test_cohort": {
                "description": "Full set of 4,974 eligible test users (rating >= 3.0)",
                "count": len(all_test_cohort),
                "metrics": clean_results(results_all_test),
                "significance": sig_all_test,
            },
            "untouched_cohort": {
                "description": "500 untouched test users (all_eligible[500:1000]) corresponding to historical untouched benchmark",
                "count": len(untouched_cohort),
                "metrics": clean_results(results_untouched),
                "significance": sig_untouched,
            },
        },
    }

    out_file = exp_dir / "test_confirmation_results.json"
    with open(out_file, "w") as f:
        json.dump(final_payload, f, indent=2)
    logger.info(f"Saved independent test confirmation results to {out_file}")

    # Print summary table
    print("\n" + "=" * 95)
    print("       CINEPULSE INDEPENDENT TEST CONFIRMATION SUMMARY (SEQUESTERED COHORT: 3,974 USERS)")
    print("=" * 95)
    print(f"{'Configuration':<45} | {'Recall@20':<10} | {'NDCG@10':<10} | {'Pos NDCG':<10} | {'Coverage':<9} | {'P95 (ms)':<8}")
    print("-" * 95)
    for cfg in configs:
        m = results_sequestered[cfg]
        print(f"{cfg[:44]:<45} | {m['recall_at_20']:<10.4f} | {m['ndcg_at_10']:<10.4f} | {m['pos_ndcg_at_10']:<10.4f} | {m['catalog_coverage_at_100']*100:<8.1f}% | {m['p95_latency_ms']:<8.2f}")
    print("=" * 95)

    print("\n" + "=" * 95)
    print("       CINEPULSE INDEPENDENT TEST CONFIRMATION SUMMARY (ALL TEST USERS: 4,974 USERS)")
    print("=" * 95)
    print(f"{'Configuration':<45} | {'Recall@20':<10} | {'NDCG@10':<10} | {'Pos NDCG':<10} | {'Coverage':<9} | {'P95 (ms)':<8}")
    print("-" * 95)
    for cfg in configs:
        m = results_all_test[cfg]
        print(f"{cfg[:44]:<45} | {m['recall_at_20']:<10.4f} | {m['ndcg_at_10']:<10.4f} | {m['pos_ndcg_at_10']:<10.4f} | {m['catalog_coverage_at_100']*100:<8.1f}% | {m['p95_latency_ms']:<8.2f}")
    print("=" * 95)

    print("\n" + "=" * 95)
    print("       PAIRED BOOTSTRAP SIGNIFICANCE (CONFIG 8 vs PUBLISHED BASELINE, B=1000)")
    print("=" * 95)
    print(f"Sequestered (3,974 users): NDCG@10 diff = {sig_sequestered['config8_vs_baseline']['ndcg_diff']:+.4f} (95% CI: {sig_sequestered['config8_vs_baseline']['ndcg_ci_95']}, p = {sig_sequestered['config8_vs_baseline']['ndcg_p_value']:.4f})")
    print(f"Sequestered (3,974 users): Recall@20 diff = {sig_sequestered['config8_vs_baseline']['recall_diff']:+.4f} (95% CI: {sig_sequestered['config8_vs_baseline']['recall_ci_95']}, p = {sig_sequestered['config8_vs_baseline']['recall_p_value']:.4f})")
    print(f"All Test (4,974 users):    NDCG@10 diff = {sig_all_test['config8_vs_baseline']['ndcg_diff']:+.4f} (95% CI: {sig_all_test['config8_vs_baseline']['ndcg_ci_95']}, p = {sig_all_test['config8_vs_baseline']['ndcg_p_value']:.4f})")
    print(f"All Test (4,974 users):    Recall@20 diff = {sig_all_test['config8_vs_baseline']['recall_diff']:+.4f} (95% CI: {sig_all_test['config8_vs_baseline']['recall_ci_95']}, p = {sig_all_test['config8_vs_baseline']['recall_p_value']:.4f})")
    print("=" * 95)


if __name__ == "__main__":
    run_confirmation()
