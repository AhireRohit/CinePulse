"""
Experiment 3: LightGBM Ranking Optimization.
Controlled investigation of:
1. Feature Engineering Expansion (17 baseline features vs 24 expanded causal features):
   - Multi-source candidate agreement count (1, 2, 3)
   - Normalized CF score and Normalized Two-Tower score
   - Synergistic score product (norm_cf * norm_tt)
   - User rating calibration offset (bayesian_rating - user_hist_mean)
   - Rating count confidence shrinkage factor
   - User recent-interaction genre affinity
2. Hyperparameter optimization across tree depth, leaves, learning rate, and regularizations
3. Evaluates all variants on the frozen validation cohort (4,980 users)
4. Saves the best experimental model to artifacts/experiments/relevance_v2/models/ranker_lgbm_exp3.txt
"""

import json
import random
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
from pipelines.tournament_eval import TournamentEvaluator
from src.core.config import settings
from src.core.logging import logger
from src.evaluation.metrics import compute_paired_bootstrap_ci
from src.models.baselines import PopularityRecommender
from src.models.two_tower import TwoTowerRecommender
from src.retrieval.faiss_index import FaissCandidateIndex

EXPANDED_FEATURE_COLS = [
    # Baseline 17 features
    "candidate_rank",
    "two_tower_score",
    "cf_score",
    "genre_overlap",
    "genre_cosine",
    "log_rating_count",
    "rating_mean",
    "rating_std",
    "bayesian_rating",
    "release_year",
    "year_diff",
    "user_activity_count",
    "is_source_two_tower",
    "is_source_item_cf",
    "is_source_popularity",
    "cf_x_bayesian",
    "tt_x_genre",
    # 7 New High-Fidelity Causal Features
    "source_count",
    "cf_score_normalized",
    "tt_score_normalized",
    "score_product",
    "user_rating_calib_diff",
    "rating_count_confidence",
    "recent_genre_cosine",
]

BASELINE_FEATURE_COLS = EXPANDED_FEATURE_COLS[:17]


class FastRankingFeatureExtractor:
    """
    High-throughput vectorized feature extractor for ranking.
    Extracts all 24 causal features in < 0.2 ms per query.
    """

    def __init__(self, movies_df: pd.DataFrame, genre_cols: list[str], global_mean_rating: float = 3.5):
        self.movies_df = movies_df.set_index("movie_idx") if "movie_idx" in movies_df.columns else movies_df
        self.genre_cols = genre_cols
        self.num_genres = len(genre_cols)
        self.global_mean_rating = global_mean_rating

        # Vectorized precomputed numpy arrays
        self.movie_genres = self.movies_df[self.genre_cols].values.astype(np.float32)
        self.movie_years = self.movies_df["release_year"].values.astype(np.float32)
        self.movie_counts = self.movies_df["rating_count"].values.astype(np.float32)
        self.movie_means = self.movies_df["rating_mean"].values.astype(np.float32)
        self.movie_stds = (
            self.movies_df["rating_std"].values.astype(np.float32)
            if "rating_std" in self.movies_df.columns
            else np.zeros(len(self.movies_df), dtype=np.float32)
        )
        self.movie_bayesian = self.movies_df["bayesian_rating"].values.astype(np.float32)
        self.movie_norms = np.linalg.norm(self.movie_genres, axis=1) + 1e-6

    def extract_matrix(
        self,
        user_idx: int,
        user_history: list[int],
        user_ratings_map: dict[int, float],
        candidate_items: list[int],
        two_tower_scores: dict[int, float],
        cf_scores: dict[int, float],
        retrieval_sources: dict[int, set[str]],
        use_expanded: bool = True,
    ) -> np.ndarray:
        M = len(candidate_items)
        if M == 0:
            return np.empty((0, len(EXPANDED_FEATURE_COLS) if use_expanded else 17), dtype=np.float32)

        cand_arr = np.array(candidate_items, dtype=np.int64)

        # User profiles
        valid_hist = [m for m in user_history if m < len(self.movies_df)]
        if valid_hist:
            u_genre = self.movie_genres[valid_hist].mean(axis=0)
            u_mean_year = float(self.movie_years[valid_hist].mean())
            recent_hist = valid_hist[-5:]
            u_recent_genre = self.movie_genres[recent_hist].mean(axis=0)
            if user_ratings_map:
                hist_ratings = [user_ratings_map.get(m, self.global_mean_rating) for m in valid_hist]
                u_mean_rating = float(np.mean(hist_ratings)) if hist_ratings else self.global_mean_rating
            else:
                u_mean_rating = self.global_mean_rating
        else:
            u_genre = np.zeros(self.num_genres, dtype=np.float32)
            u_mean_year = 1990.0
            u_recent_genre = np.zeros(self.num_genres, dtype=np.float32)
            u_mean_rating = self.global_mean_rating

        u_norm = np.linalg.norm(u_genre) + 1e-6
        u_rec_norm = np.linalg.norm(u_recent_genre) + 1e-6
        user_activity = float(len(user_history))

        # Item features
        c_genres = self.movie_genres[cand_arr]
        c_years = self.movie_years[cand_arr]
        c_counts = self.movie_counts[cand_arr]
        c_means = self.movie_means[cand_arr]
        c_stds = self.movie_stds[cand_arr]
        c_bayesian = self.movie_bayesian[cand_arr]
        c_norms = self.movie_norms[cand_arr]

        # 1. Retrieval scores
        tt_vals = np.array([two_tower_scores.get(c, 0.0) for c in candidate_items], dtype=np.float32)
        cf_vals = np.array([cf_scores.get(c, 0.0) for c in candidate_items], dtype=np.float32)
        cand_ranks = np.arange(M, dtype=np.float32)

        # 2. Genre features
        g_overlap = np.dot(c_genres, u_genre)
        g_cosine = g_overlap / (u_norm * c_norms)

        # 3. Item quality & metadata
        log_counts = np.log1p(c_counts)
        year_diffs = np.abs(c_years - u_mean_year)

        # 4. Source indicators
        is_tt = np.zeros(M, dtype=np.float32)
        is_cf = np.zeros(M, dtype=np.float32)
        is_pop = np.zeros(M, dtype=np.float32)
        for i, c in enumerate(candidate_items):
            srcs = retrieval_sources.get(c, set())
            if "two_tower" in srcs:
                is_tt[i] = 1.0
            if "item_cf" in srcs:
                is_cf[i] = 1.0
            if "popularity" in srcs:
                is_pop[i] = 1.0

        # Cross features
        cf_x_bayesian = cf_vals * (c_bayesian - self.global_mean_rating)
        tt_x_genre = tt_vals * g_cosine

        base_mat = [
            cand_ranks,
            tt_vals,
            cf_vals,
            g_overlap,
            g_cosine,
            log_counts,
            c_means,
            c_stds,
            c_bayesian,
            c_years,
            year_diffs,
            np.full(M, user_activity, dtype=np.float32),
            is_tt,
            is_cf,
            is_pop,
            cf_x_bayesian,
            tt_x_genre,
        ]

        if not use_expanded:
            return np.column_stack(base_mat)

        # 7 New Expanded Causal Features
        source_count = is_tt + is_cf + is_pop

        cf_min, cf_max = cf_vals.min(), cf_vals.max()
        cf_norm = (cf_vals - cf_min) / (cf_max - cf_min + 1e-6)

        tt_min, tt_max = tt_vals.min(), tt_vals.max()
        tt_norm = (tt_vals - tt_min) / (tt_max - tt_min + 1e-6)

        score_prod = cf_norm * tt_norm
        calib_diff = c_bayesian - u_mean_rating
        cnt_conf = log_counts / (log_counts + 1.0)

        rec_overlap = np.dot(c_genres, u_recent_genre)
        rec_cosine = rec_overlap / (u_rec_norm * c_norms)

        expanded_mat = base_mat + [
            source_count,
            cf_norm,
            tt_norm,
            score_prod,
            calib_diff,
            cnt_conf,
            rec_cosine,
        ]
        return np.column_stack(expanded_mat)


def construct_fast_ranking_dataset(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    genre_cols: list[str],
    user_histories: dict[int, list[int]],
    item_embeddings: np.ndarray,
    tt_model: TwoTowerRecommender,
    pop_model: PopularityRecommender,
    cf_model: EnhancedItemCF,
    faiss_index: FaissCandidateIndex,
    device: torch.device,
    feature_extractor: FastRankingFeatureExtractor,
    num_users_to_sample: int = 3500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Constructs high-fidelity ranking dataset using FastRankingFeatureExtractor.
    Returns:
    - X_base: (N, 17) float32 matrix
    - X_exp:  (N, 24) float32 matrix
    - y:      (N,) int32 labels
    - group_sizes: list of candidate counts per user query
    """
    logger.info(f"Extracting ranking dataset from {len(ratings_df):,} ratings (sampling {num_users_to_sample} users)...")
    unique_users = ratings_df["user_idx"].unique()
    if len(unique_users) > num_users_to_sample:
        np.random.seed(seed)
        sampled_users = np.random.choice(unique_users, size=num_users_to_sample, replace=False)
    else:
        sampled_users = unique_users

    # Build user ratings map
    user_ratings_map: dict[int, dict[int, float]] = {}
    for row in ratings_df[["user_idx", "movie_idx", "rating"]].itertuples(index=False):
        u, m, r = int(row.user_idx), int(row.movie_idx), float(row.rating)
        user_ratings_map.setdefault(u, {})[m] = r

    # Precompute user embeddings in batches
    tt_model.eval()
    sampled_users_list = list(sampled_users)
    user_embs_dict = {}
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(sampled_users_list), batch_size):
            b_users = sampled_users_list[i : i + batch_size]
            b_size = len(b_users)
            u_arr = np.array(b_users, dtype=np.int64)
            padded_hist = np.full((b_size, settings.max_sequence_length), len(movies_df), dtype=np.int64)
            h_lens = np.zeros(b_size, dtype=np.int64)
            g_prefs = np.zeros((b_size, len(genre_cols)), dtype=np.float32)

            for idx, u in enumerate(b_users):
                hist = user_histories.get(u, [])
                h_len = min(len(hist), settings.max_sequence_length)
                h_lens[idx] = h_len
                if h_len > 0:
                    padded_hist[idx, -h_len:] = hist[-h_len:]
                    valid_h = [m for m in hist if m < len(movies_df)]
                    if valid_h:
                        g_prefs[idx] = feature_extractor.movie_genres[valid_h].mean(axis=0)

            u_tensor = torch.tensor(u_arr, dtype=torch.long, device=device)
            h_tensor = torch.tensor(padded_hist, dtype=torch.long, device=device)
            hl_tensor = torch.tensor(h_lens, dtype=torch.long, device=device)
            gp_tensor = torch.tensor(g_prefs, dtype=torch.float32, device=device)

            embs = tt_model.user_tower(u_tensor, h_tensor, hl_tensor, gp_tensor).cpu().numpy()
            for idx, u in enumerate(b_users):
                user_embs_dict[u] = embs[idx]

    all_matrices_base = []
    all_matrices_exp = []
    all_labels = []
    group_sizes = []

    for u_idx in sampled_users_list:
        u_ratings = user_ratings_map.get(u_idx, {})
        history = list(user_histories.get(u_idx, []))
        if not history or not u_ratings:
            continue

        # Find positive items with rating >= 3.0
        pos_items = [m for m, r in u_ratings.items() if r >= 3.0]
        if not pos_items:
            continue
        pos_item = pos_items[-1]
        pos_rating = u_ratings[pos_item]

        # Graded relevance labels: 5★=3, 4★=2, 3★=1
        if pos_rating >= 4.5:
            pos_label = 3
        elif pos_rating >= 3.5:
            pos_label = 2
        else:
            pos_label = 1

        # Zero target leakage: exclude positive item from history during feature extraction
        history_clean = [m for m in history if m != pos_item]
        consumed = set(history_clean)
        u_vec = user_embs_dict[u_idx]

        candidate_sources: dict[int, set[str]] = {}
        cf_score_map: dict[int, float] = {}

        # 1. Two-Tower candidates (100)
        for item_idx, score in faiss_index.search(u_vec, top_k=100, exclude_item_indices=consumed):
            candidate_sources.setdefault(item_idx, set()).add("two_tower")

        # 2. CF candidates (100)
        for item_idx, score in cf_model.recommend(history_clean, top_k=100):
            if item_idx not in consumed:
                candidate_sources.setdefault(item_idx, set()).add("item_cf")
                cf_score_map[item_idx] = float(score)

        # 3. Popularity candidates (30)
        for item_idx, score in pop_model.recommend(history_clean, top_k=30):
            if item_idx not in consumed:
                candidate_sources.setdefault(item_idx, set()).add("popularity")

        # 4. Guarantee positive item is present in candidate pool
        if pos_item not in candidate_sources:
            candidate_sources.setdefault(pos_item, set()).add("target_pos")
            if pos_item in dict(cf_model.recommend(history_clean, top_k=200)):
                cf_score_map[pos_item] = float(dict(cf_model.recommend(history_clean, top_k=200))[pos_item])

        cand_list = list(candidate_sources.keys())
        labels = [pos_label if c == pos_item else 0 for c in cand_list]

        # Compute Two-Tower dot products for all candidates
        cand_arr = np.array(cand_list, dtype=np.int64)
        valid_mask = cand_arr < len(item_embeddings)
        tt_scores_dict: dict[int, float] = {}
        if np.any(valid_mask):
            cand_embs = item_embeddings[cand_arr[valid_mask]]
            sims = np.dot(cand_embs, u_vec)
            for idx_c, sim in zip(cand_arr[valid_mask], sims):
                tt_scores_dict[int(idx_c)] = float(sim)
        for c in cand_list:
            tt_scores_dict.setdefault(c, 0.0)

        # Shuffle candidates within group to eliminate presentation bias
        combined = list(zip(cand_list, labels))
        random.seed(int(seed + u_idx))
        random.shuffle(combined)
        shuffled_cands = [c for c, _ in combined]
        shuffled_labels = [lbl for _, lbl in combined]

        # Extract feature matrices
        mat_base = feature_extractor.extract_matrix(
            user_idx=u_idx,
            user_history=history_clean,
            user_ratings_map=u_ratings,
            candidate_items=shuffled_cands,
            two_tower_scores=tt_scores_dict,
            cf_scores=cf_score_map,
            retrieval_sources=candidate_sources,
            use_expanded=False,
        )
        mat_exp = feature_extractor.extract_matrix(
            user_idx=u_idx,
            user_history=history_clean,
            user_ratings_map=u_ratings,
            candidate_items=shuffled_cands,
            two_tower_scores=tt_scores_dict,
            cf_scores=cf_score_map,
            retrieval_sources=candidate_sources,
            use_expanded=True,
        )

        all_matrices_base.append(mat_base)
        all_matrices_exp.append(mat_exp)
        all_labels.extend(shuffled_labels)
        group_sizes.append(len(shuffled_cands))

    X_base = np.vstack(all_matrices_base)
    X_exp = np.vstack(all_matrices_exp)
    y = np.array(all_labels, dtype=np.int32)
    logger.info(f"Dataset constructed: {len(X_base):,} candidate pairs across {len(group_sizes):,} groups.")
    return X_base, X_exp, y, group_sizes


def run_experiment_3():
    logger.info("=== Running Experiment 3: LightGBM Ranking Optimization ===")
    evaluator = TournamentEvaluator()
    device = evaluator.device

    # 1. Fit baselines & load models
    logger.info("Loading foundational retrieval models...")
    pop_model = PopularityRecommender()
    pop_model.fit(evaluator.train_ratings, num_movies=evaluator.num_movies)

    cf_model = EnhancedItemCF(top_k_neighbors=200, shrinkage_lambda=25.0)
    cf_model.fit(evaluator.train_ratings, num_users=evaluator.num_users, num_movies=evaluator.num_movies)

    two_tower_path = Path("artifacts/models/two_tower_best.pt")
    tt_model = TwoTowerRecommender(
        num_users=evaluator.num_users,
        num_movies=evaluator.num_movies,
        num_genres=len(evaluator.genre_cols),
        embedding_dim=settings.embedding_dim,
    ).to(device)
    ckpt = torch.load(two_tower_path, map_location=device, weights_only=True)
    tt_model.load_state_dict(ckpt["model_state_dict"])
    tt_model.eval()

    faiss_index = FaissCandidateIndex(dimension=settings.embedding_dim)
    faiss_index.load(Path("artifacts/indices"))

    item_embeddings = np.load("artifacts/models/item_embeddings.npy")

    fast_extractor = FastRankingFeatureExtractor(
        movies_df=evaluator.movies_df,
        genre_cols=evaluator.genre_cols,
        global_mean_rating=evaluator.mappings.get("global_rating_mean", 3.5),
    )

    # 2. Construct training & validation pairs
    X_train_base, X_train_exp, y_train, group_train = construct_fast_ranking_dataset(
        ratings_df=evaluator.train_ratings,
        movies_df=evaluator.movies_df,
        genre_cols=evaluator.genre_cols,
        user_histories=evaluator.user_histories,
        item_embeddings=item_embeddings,
        tt_model=tt_model,
        pop_model=pop_model,
        cf_model=cf_model,
        faiss_index=faiss_index,
        device=device,
        feature_extractor=fast_extractor,
        num_users_to_sample=3500,
        seed=42,
    )

    X_val_base, X_val_exp, y_val, group_val = construct_fast_ranking_dataset(
        ratings_df=pd.read_parquet("data/processed/ratings_val.parquet"),
        movies_df=evaluator.movies_df,
        genre_cols=evaluator.genre_cols,
        user_histories=evaluator.user_histories,
        item_embeddings=item_embeddings,
        tt_model=tt_model,
        pop_model=pop_model,
        cf_model=cf_model,
        faiss_index=faiss_index,
        device=device,
        feature_extractor=fast_extractor,
        num_users_to_sample=800,
        seed=101,
    )

    # Precompute validation user embeddings and candidate sets once for instant tournament evaluation
    logger.info("Precomputing validation user candidate sets for all 4,980 users...")
    eval_users = evaluator.std_users
    val_user_embs = {}
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(eval_users), batch_size):
            b_users = eval_users[i : i + batch_size]
            b_size = len(b_users)
            u_arr = np.array(b_users, dtype=np.int64)
            padded_hist = np.full((b_size, settings.max_sequence_length), evaluator.num_movies, dtype=np.int64)
            h_lens = np.zeros(b_size, dtype=np.int64)
            g_prefs = np.zeros((b_size, len(evaluator.genre_cols)), dtype=np.float32)

            for idx, u in enumerate(b_users):
                hist = evaluator.user_histories.get(u, [])
                h_len = min(len(hist), settings.max_sequence_length)
                h_lens[idx] = h_len
                if h_len > 0:
                    padded_hist[idx, -h_len:] = hist[-h_len:]
                    valid_h = [m for m in hist if m < evaluator.num_movies]
                    if valid_h:
                        g_prefs[idx] = fast_extractor.movie_genres[valid_h].mean(axis=0)

            u_tensor = torch.tensor(u_arr, dtype=torch.long, device=device)
            h_tensor = torch.tensor(padded_hist, dtype=torch.long, device=device)
            hl_tensor = torch.tensor(h_lens, dtype=torch.long, device=device)
            gp_tensor = torch.tensor(g_prefs, dtype=torch.float32, device=device)

            embs = tt_model.user_tower(u_tensor, h_tensor, hl_tensor, gp_tensor).cpu().numpy()
            for idx, u in enumerate(b_users):
                val_user_embs[u] = embs[idx]

    # Precompute candidate pools and matrices for all 4,980 validation users
    logger.info("Generating candidate matrices for validation evaluation...")
    val_user_candidates = {}
    val_user_mat_base = {}
    val_user_mat_exp = {}

    user_val_ratings_map: dict[int, dict[int, float]] = {}
    for row in evaluator.train_ratings[["user_idx", "movie_idx", "rating"]].itertuples(index=False):
        u, m, r = int(row.user_idx), int(row.movie_idx), float(row.rating)
        user_val_ratings_map.setdefault(u, {})[m] = r

    for u in eval_users:
        h = evaluator.user_histories.get(u, [])
        consumed = set(h)
        u_vec = val_user_embs[u]

        c_sources = {}
        cf_map = {}

        # Two-Tower top-100
        for item_idx, score in faiss_index.search(u_vec, top_k=100, exclude_item_indices=consumed):
            c_sources.setdefault(item_idx, set()).add("two_tower")

        # CF top-100
        for item_idx, score in cf_model.recommend(h, top_k=100):
            if item_idx not in consumed:
                c_sources.setdefault(item_idx, set()).add("item_cf")
                cf_map[item_idx] = float(score)

        # Popularity top-30
        for item_idx, score in pop_model.recommend(h, top_k=30):
            if item_idx not in consumed:
                c_sources.setdefault(item_idx, set()).add("popularity")

        c_list = list(c_sources.keys())
        cand_arr = np.array(c_list, dtype=np.int64)
        valid_mask = cand_arr < len(item_embeddings)
        tt_scores_dict = {}
        if np.any(valid_mask):
            cand_embs = item_embeddings[cand_arr[valid_mask]]
            sims = np.dot(cand_embs, u_vec)
            for idx_c, sim in zip(cand_arr[valid_mask], sims):
                tt_scores_dict[int(idx_c)] = float(sim)
        for c in c_list:
            tt_scores_dict.setdefault(c, 0.0)

        u_ratings = user_val_ratings_map.get(u, {})
        val_user_candidates[u] = c_list
        val_user_mat_base[u] = fast_extractor.extract_matrix(
            u, h, u_ratings, c_list, tt_scores_dict, cf_map, c_sources, use_expanded=False
        )
        val_user_mat_exp[u] = fast_extractor.extract_matrix(
            u, h, u_ratings, c_list, tt_scores_dict, cf_map, c_sources, use_expanded=True
        )

    # 3. Model Tournament Configurations
    ranker_variants = [
        {
            "name": "Ranker-V1 (17 Features, Default LightGBM)",
            "use_expanded": False,
            "params": {
                "n_estimators": 250,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 20,
            },
        },
        {
            "name": "Ranker-V2 (24 Expanded Features, Default LightGBM)",
            "use_expanded": True,
            "params": {
                "n_estimators": 250,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 20,
            },
        },
        {
            "name": "Ranker-V3 (24 Expanded Features, Deeper Trees)",
            "use_expanded": True,
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "max_depth": 8,
                "min_child_samples": 30,
            },
        },
        {
            "name": "Ranker-V4 (24 Expanded Features, Deep Regularized)",
            "use_expanded": True,
            "params": {
                "n_estimators": 350,
                "learning_rate": 0.025,
                "num_leaves": 63,
                "max_depth": 8,
                "min_child_samples": 30,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            },
        },
        {
            "name": "Ranker-V5 (24 Expanded Features, Conservative Regularized)",
            "use_expanded": True,
            "params": {
                "n_estimators": 250,
                "learning_rate": 0.035,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 25,
                "feature_fraction": 0.85,
                "reg_lambda": 0.5,
            },
        },
        {
            "name": "Ranker-V6 (24 Expanded Features, High Capacity)",
            "use_expanded": True,
            "params": {
                "n_estimators": 400,
                "learning_rate": 0.02,
                "num_leaves": 127,
                "max_depth": 10,
                "min_child_samples": 40,
                "reg_lambda": 2.0,
            },
        },
    ]

    exp3_results = {}
    exp3_per_user = {}
    trained_models = {}

    for var in ranker_variants:
        v_name = var["name"]
        use_exp = var["use_expanded"]
        params = var["params"]
        logger.info(f"Training and evaluating variant: {v_name}...")

        X_tr = X_train_exp if use_exp else X_train_base
        X_va = X_val_exp if use_exp else X_val_base

        t0_train = time.time()
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[5, 10, 20],
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
            **params,
        )
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        model.fit(
            X_tr,
            y_train,
            group=group_train,
            eval_set=[(X_va, y_val)],
            eval_group=[group_val],
            callbacks=callbacks,
        )
        train_duration = time.time() - t0_train
        trained_models[v_name] = model

        # Evaluate on all 4,980 validation users
        recs_dict = {}
        lats = []
        val_mats = val_user_mat_exp if use_exp else val_user_mat_base

        for u in eval_users:
            t_eval_start = time.perf_counter()
            c_list = val_user_candidates[u]
            feat_mat = val_mats[u]
            if len(c_list) > 0:
                scores = model.predict(feat_mat)
                order = np.argsort(-scores)
                r_list = [c_list[idx] for idx in order]
            else:
                r_list = []

            if len(r_list) < 100:
                h = evaluator.user_histories.get(u, [])
                pop_fill = [idx for idx, _ in pop_model.recommend(h, top_k=100) if idx not in set(r_list)]
                r_list.extend(pop_fill[: 100 - len(r_list)])

            t_eval_end = time.perf_counter()
            lats.append((t_eval_end - t_eval_start) * 1000.0)
            recs_dict[u] = r_list

        metrics, per_user = evaluator.evaluate_recommendations(
            recs_dict, v_name, latencies_ms=lats, train_time_sec=train_duration
        )
        exp3_results[v_name] = metrics
        exp3_per_user[v_name] = per_user

        logger.info(
            f"-> {v_name}: Recall@20 = {metrics['recall@20']:.4f}, NDCG@10 = {metrics['ndcg@10']:.4f}, "
            f"Pos NDCG@10 = {metrics['positive_only_ndcg@10']:.4f}, Train Runtime = {train_duration:.1f}s, "
            f"Rank Latency P95 = {metrics['latency_p95_ms']:.2f} ms"
        )

    # Save results
    output_dir = Path("artifacts/experiments/relevance_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "exp3_ranker_results.json", "w") as f:
        json.dump(exp3_results, f, indent=2)

    # Find the best variant
    best_variant_name = max(exp3_results.keys(), key=lambda k: exp3_results[k]["ndcg@10"])
    best_model = trained_models[best_variant_name]
    best_m = exp3_results[best_variant_name]

    # Save best model to artifacts/experiments/relevance_v2/models/ranker_lgbm_exp3.txt
    best_model_path = models_dir / "ranker_lgbm_exp3.txt"
    best_model.booster_.save_model(str(best_model_path))
    logger.info(f"Saved tournament winning ranker model to {best_model_path}")

    # Paired bootstrap test vs Baseline Pure Ranker
    with open(output_dir / "baseline_results_val.json") as f:
        baseline_results = json.load(f)
    baseline_ranker = baseline_results["Existing Pure Ranker"]

    with open(output_dir / "baseline_per_user_val.json") as f:
        baseline_user_data = json.load(f)["Existing Pure Ranker"]

    eval_u_list = evaluator.std_users
    base_ndcgs = [baseline_user_data["ndcg@10"][str(u)] for u in eval_u_list]
    best_ndcgs = [exp3_per_user[best_variant_name]["ndcg@10"][u] for u in eval_u_list]

    diff_mean, ci_l, ci_u, p_val = compute_paired_bootstrap_ci(best_ndcgs, base_ndcgs)
    significance_report = {
        "best_variant": best_variant_name,
        "delta_ndcg@10_vs_baseline_ranker": round(diff_mean, 4),
        "ci95_ndcg@10": [round(ci_l, 4), round(ci_u, 4)],
        "p_value": p_val,
        "baseline_ndcg@10": baseline_ranker["ndcg@10"],
        "best_ndcg@10": best_m["ndcg@10"],
        "baseline_recall@20": baseline_ranker["recall@20"],
        "best_recall@20": best_m["recall@20"],
    }
    with open(output_dir / "exp3_significance.json", "w") as f:
        json.dump(significance_report, f, indent=2)

    print("\n" + "=" * 115)
    print("EXPERIMENT 3: LIGHTGBM RANKING OPTIMIZATION ABLATION RESULTS (All 4,980 Val Users)")
    print("=" * 115)
    fmt = "{:<55} {:<12} {:<12} {:<12} {:<12} {:<10}"
    print(fmt.format("Configuration", "Recall@20", "NDCG@10", "Recall@100", "Pos NDCG@10", "P95 ms"))
    print("-" * 115)
    print(
        fmt.format(
            "Baseline Pure Ranker (17 Features)",
            f"{baseline_ranker['recall@20']:.4f}",
            f"{baseline_ranker['ndcg@10']:.4f}",
            f"{baseline_ranker['recall@100']:.4f}",
            f"{baseline_ranker['positive_only_ndcg@10']:.4f}",
            f"{baseline_ranker['latency_p95_ms']:.2f}",
        )
    )
    for name, m in exp3_results.items():
        print(
            fmt.format(
                name,
                f"{m['recall@20']:.4f}",
                f"{m['ndcg@10']:.4f}",
                f"{m['recall@100']:.4f}",
                f"{m['positive_only_ndcg@10']:.4f}",
                f"{m['latency_p95_ms']:.2f}",
            )
        )
    print("-" * 115)
    print(f"Tournament Best: {best_variant_name}")
    print(f"Paired delta vs Baseline Pure Ranker: Delta NDCG@10 = {diff_mean:+.4f} [{ci_l:+.4f}, {ci_u:+.4f}], p = {p_val:.4f}")
    print("=" * 115)


if __name__ == "__main__":
    run_experiment_3()

