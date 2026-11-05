from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.core.logging import logger

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
    # 7 New High-Fidelity Causal Features (Config 8)
    "source_count",
    "cf_score_normalized",
    "tt_score_normalized",
    "score_product",
    "user_rating_calib_diff",
    "rating_count_confidence",
    "recent_genre_cosine",
]

RANKER_FEATURE_COLS = EXPANDED_FEATURE_COLS[:17]


class LightGBMRanker:
    """
    LightGBM LambdaRank personalized ranking model.
    Reranks candidate items using cross-tower, CF, genre affinity, and item quality features.
    Supports both 17-feature v1 and 24-feature v2 (Config 8) models.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 15,
        objective: str = "lambdarank",
        use_expanded_features: bool = False,
    ):
        self.use_expanded_features = use_expanded_features
        self.feature_cols = EXPANDED_FEATURE_COLS if use_expanded_features else RANKER_FEATURE_COLS
        self.objective = objective
        self.model = lgb.LGBMRanker(
            objective=objective,
            metric="ndcg",
            ndcg_eval_at=[5, 10, 20],
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=42,
            importance_type="gain",
            n_jobs=-1,
        )
        self.booster: lgb.Booster | None = None
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        group_train: list[int] | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        group_val: list[int] | np.ndarray | None = None,
    ):
        logger.info(f"Training LightGBMRanker with {len(X_train)} samples across {len(group_train)} queries...")
        eval_set = []
        eval_group = []
        if X_val is not None and y_val is not None and group_val is not None:
            eval_set.append((X_val[self.feature_cols], y_val))
            eval_group.append(group_val)

        callbacks = [lgb.early_stopping(stopping_rounds=15, verbose=False)] if eval_set else None

        self.model.fit(
            X_train[self.feature_cols],
            y_train,
            group=group_train,
            eval_set=eval_set if eval_set else None,
            eval_group=eval_group if eval_group else None,
            callbacks=callbacks,
        )
        self.is_fitted = True
        logger.info("LightGBMRanker training completed.")

        # Log top feature importances
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(-importances)
        logger.info("Top Feature Importances (Gain):")
        for i in sorted_idx[:5]:
            logger.info(f"  {self.feature_cols[i]}: {importances[i]:.2f}")

    def predict_scores(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        if features_df.empty:
            return np.array([], dtype=np.float32)

        if self.booster is not None:
            return self.booster.predict(features_df[self.feature_cols].values)
        return self.model.predict(features_df[self.feature_cols])

    def rank(self, features_df: pd.DataFrame) -> list[tuple[int, float]]:
        """
        Scores and ranks candidate items.
        Returns list of (movie_idx, score) sorted descending by predicted relevance.
        """
        if features_df.empty:
            return []

        scores = self.predict_scores(features_df)
        movie_indices = features_df["movie_idx"].values

        ranked_pairs = list(zip(movie_indices.astype(int), scores.astype(float)))
        ranked_pairs.sort(key=lambda x: x[1], reverse=True)
        return ranked_pairs

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(filepath))
        logger.info(f"Saved LightGBM ranker model to {filepath}")

    def load(self, filepath: Path):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Ranker model file not found at {filepath}")
        # Normalize line endings to LF to prevent C++ LightGBM parsing errors on Windows CRLF
        with open(filepath, "rb") as f_in:
            raw_bytes = f_in.read()
        crlf = bytes([13, 10])
        lf = bytes([10])
        if crlf in raw_bytes:
            clean_str = raw_bytes.replace(crlf, lf).decode("utf-8")
            self.booster = lgb.Booster(model_str=clean_str)
        else:
            self.booster = lgb.Booster(model_file=str(filepath))
        num_features = self.booster.num_feature()
        if num_features == 24:
            self.feature_cols = EXPANDED_FEATURE_COLS
            self.use_expanded_features = True
        else:
            self.feature_cols = RANKER_FEATURE_COLS
            self.use_expanded_features = False
        self.model = lgb.LGBMRanker()
        self.model._Booster = self.booster
        self.model.fitted_ = True
        self.is_fitted = True
        logger.info(f"Loaded LightGBM ranker model from {filepath} ({num_features} features: {len(self.feature_cols)} configured)")
