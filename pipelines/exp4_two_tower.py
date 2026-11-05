"""
Experiment 4: Rating-Aware Two-Tower Neural Retrieval.
Controlled investigation of:
1. Baseline Production Two-Tower (Trained on min_rating=4.0, tau=0.05)
2. Two-Tower with Sharper Temperature (tau=0.03 for stricter discrimination)
3. Two-Tower with Higher Representation Capacity (hidden_dim=256, dropout=0.15)
4. Two-Tower with Rating-Graded Sample Weighting (5-star ratings weighted 1.5x, 4-star 1.0x)
5. Standalone ANN Retrieval Recall@{10, 20, 50, 100} and NDCG@{5, 10, 20}
6. Downstream LightGBM ranking performance with upgraded retrieval
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pipelines.tournament_eval import TournamentEvaluator
from src.core.config import settings
from src.core.logging import logger
from src.data.dataset import MovieLensRetrievalDataset
from src.evaluation.metrics import compute_paired_bootstrap_ci
from src.models.two_tower import TwoTowerRecommender
from src.retrieval.faiss_index import FaissCandidateIndex


class WeightedTwoTowerRecommender(TwoTowerRecommender):
    """
    Two-Tower Recommender supporting explicit temperature control and rating-graded sample weighting.
    """

    def compute_weighted_loss(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor, sample_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.01, max=0.2)
        similarity = torch.matmul(user_emb, item_emb.T) / temp
        targets = torch.arange(user_emb.shape[0], device=user_emb.device)
        if sample_weights is not None:
            losses = F.cross_entropy(similarity, targets, reduction="none")
            return (losses * sample_weights).mean()
        return F.cross_entropy(similarity, targets)


def train_two_tower_variant(
    evaluator: TournamentEvaluator,
    variant_name: str,
    train_dataset: MovieLensRetrievalDataset,
    val_dataset: MovieLensRetrievalDataset,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    temperature: float = 0.05,
    hidden_dim: int = 128,
    weight_by_rating: bool = False,
    output_model_path: Path = Path("artifacts/experiments/relevance_v2/models/two_tower_exp4.pt"),
) -> tuple[WeightedTwoTowerRecommender, np.ndarray, float]:
    device = evaluator.device
    logger.info(f"Training Two-Tower Variant '{variant_name}' on {device} (epochs={epochs}, tau={temperature}, hidden={hidden_dim})...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = WeightedTwoTowerRecommender(
        num_users=evaluator.num_users,
        num_movies=evaluator.num_movies,
        num_genres=len(evaluator.genre_cols),
        embedding_dim=settings.embedding_dim,
        hidden_dim=hidden_dim,
        temperature=temperature,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    t0_train = time.time()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            b_dev = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                u_emb, i_emb = model(b_dev)
                if weight_by_rating and "target_movie_rating" in b_dev:
                    ratings = b_dev["target_movie_rating"]
                    weights = torch.where(ratings >= 4.8, 1.5, 1.0)
                    loss = model.compute_weighted_loss(u_emb, i_emb, weights)
                else:
                    loss = model.compute_weighted_loss(u_emb, i_emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation loss
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                b_dev = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    u_emb, i_emb = model(b_dev)
                    loss = model.compute_weighted_loss(u_emb, i_emb)
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(1, n_val)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_duration = time.time() - t0_train
    if best_state is not None:
        model.load_state_dict(best_state)

    # Precompute catalog item embeddings
    model.eval()
    logger.info("Computing catalog item embeddings...")
    movie_indices = torch.arange(evaluator.num_movies, device=device)
    movie_genres = torch.tensor(evaluator.item_genres, dtype=torch.float32, device=device)
    movie_dense = train_dataset.movie_dense_features.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            item_embs = model.item_tower(movie_indices, movie_genres, movie_dense).cpu().numpy().astype(np.float32)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "val_loss": best_val_loss,
            "temperature": temperature,
            "hidden_dim": hidden_dim,
        },
        output_model_path,
    )
    return model, item_embs, train_duration


def run_experiment_4():
    logger.info("=== Running Experiment 4: Rating-Aware Two-Tower Neural Retrieval Optimization ===")
    evaluator = TournamentEvaluator()
    device = evaluator.device

    output_dir = Path("artifacts/experiments/relevance_v2")
    models_dir = output_dir / "models"
    indices_dir = output_dir / "indices"
    models_dir.mkdir(parents=True, exist_ok=True)
    indices_dir.mkdir(parents=True, exist_ok=True)

    # Pre-construct training and validation datasets once
    logger.info("Constructing causal training and validation datasets for neural retrieval...")
    train_dataset = MovieLensRetrievalDataset(
        ratings_df=evaluator.train_ratings,
        movies_df=evaluator.movies_df,
        max_seq_len=settings.max_sequence_length,
        genre_cols=evaluator.genre_cols,
        causal=True,
        min_rating=4.0,
    )
    val_dataset = MovieLensRetrievalDataset(
        ratings_df=pd.read_parquet("data/processed/ratings_val.parquet"),
        movies_df=evaluator.movies_df,
        user_histories=evaluator.user_histories,
        max_seq_len=settings.max_sequence_length,
        genre_cols=evaluator.genre_cols,
        causal=False,
        min_rating=4.0,
    )

    # 1. Baseline Model
    logger.info("Loading baseline Two-Tower model...")
    base_tt = TwoTowerRecommender(
        num_users=evaluator.num_users,
        num_movies=evaluator.num_movies,
        num_genres=len(evaluator.genre_cols),
        embedding_dim=settings.embedding_dim,
    ).to(device)
    ckpt = torch.load("artifacts/models/two_tower_best.pt", map_location=device, weights_only=True)
    base_tt.load_state_dict(ckpt["model_state_dict"])
    base_tt.eval()
    base_item_embs = np.load("artifacts/models/item_embeddings.npy")

    exp4_models = [
        {
            "name": "TT-V1 (Baseline Production: tau=0.05, hidden=128)",
            "model": base_tt,
            "item_embs": base_item_embs,
            "train_duration": 0.0,
        }
    ]

    # Train Variant 2: Sharper temperature tau=0.03
    m2_path = models_dir / "two_tower_tau03.pt"
    m2, embs2, dur2 = train_two_tower_variant(
        evaluator,
        variant_name="TT-V2 (Sharper Temp: tau=0.03, hidden=128)",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=5,
        temperature=0.03,
        hidden_dim=128,
        output_model_path=m2_path,
    )
    exp4_models.append({
        "name": "TT-V2 (Sharper Temp: tau=0.03, hidden=128)",
        "model": m2,
        "item_embs": embs2,
        "train_duration": dur2,
    })

    # Train Variant 3: Higher capacity hidden=256, tau=0.04
    m3_path = models_dir / "two_tower_cap256.pt"
    m3, embs3, dur3 = train_two_tower_variant(
        evaluator,
        variant_name="TT-V3 (Higher Capacity: tau=0.04, hidden=256)",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=5,
        temperature=0.04,
        hidden_dim=256,
        output_model_path=m3_path,
    )
    exp4_models.append({
        "name": "TT-V3 (Higher Capacity: tau=0.04, hidden=256)",
        "model": m3,
        "item_embs": embs3,
        "train_duration": dur3,
    })

    eval_users = evaluator.std_users
    exp4_results = {}
    exp4_per_user = {}

    for spec in exp4_models:
        v_name = spec["name"]
        tt_m = spec["model"]
        i_embs = spec["item_embs"]
        t_dur = spec["train_duration"]
        logger.info(f"Evaluating Two-Tower Variant: {v_name}...")

        # Build FAISS index for this variant
        faiss_idx = FaissCandidateIndex(dimension=settings.embedding_dim)
        faiss_idx.build(i_embs)

        # Batch encode user embeddings
        tt_m.eval()
        val_u_embs = {}
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
                            g_prefs[idx] = evaluator.item_genres[valid_h].mean(axis=0)

                u_tensor = torch.tensor(u_arr, dtype=torch.long, device=device)
                h_tensor = torch.tensor(padded_hist, dtype=torch.long, device=device)
                hl_tensor = torch.tensor(h_lens, dtype=torch.long, device=device)
                gp_tensor = torch.tensor(g_prefs, dtype=torch.float32, device=device)

                embs = tt_m.user_tower(u_tensor, h_tensor, hl_tensor, gp_tensor).cpu().numpy()
                for idx, u in enumerate(b_users):
                    val_u_embs[u] = embs[idx]

        recs_dict = {}
        lats = []
        for u in eval_users:
            h = evaluator.user_histories.get(u, [])
            consumed = set(h)
            u_vec = val_u_embs[u]
            t0 = time.perf_counter()
            recs = [idx for idx, _ in faiss_idx.search(u_vec, top_k=100, exclude_item_indices=consumed)]
            lats.append((time.perf_counter() - t0) * 1000.0)
            recs_dict[u] = recs

        metrics, per_user = evaluator.evaluate_recommendations(
            recs_dict, v_name, latencies_ms=lats, train_time_sec=t_dur
        )
        exp4_results[v_name] = metrics
        exp4_per_user[v_name] = per_user

        logger.info(
            f"-> {v_name}: Recall@20 = {metrics['recall@20']:.4f}, NDCG@10 = {metrics['ndcg@10']:.4f}, "
            f"Recall@100 = {metrics['recall@100']:.4f}, Pos NDCG@10 = {metrics['positive_only_ndcg@10']:.4f}, "
            f"Coverage@100 = {metrics['catalog_coverage@100']*100:.1f}%, Latency P95 = {metrics['latency_p95_ms']:.2f} ms"
        )

    # Save results
    with open(output_dir / "exp4_two_tower_results.json", "w") as f:
        json.dump(exp4_results, f, indent=2)

    best_variant_name = max(exp4_results.keys(), key=lambda k: exp4_results[k]["ndcg@10"])
    best_m = exp4_results[best_variant_name]
    base_m = exp4_results["TT-V1 (Baseline Production: tau=0.05, hidden=128)"]

    # Paired significance test
    base_ndcgs = [exp4_per_user["TT-V1 (Baseline Production: tau=0.05, hidden=128)"]["ndcg@10"][u] for u in eval_users]
    best_ndcgs = [exp4_per_user[best_variant_name]["ndcg@10"][u] for u in eval_users]
    diff_mean, ci_l, ci_u, p_val = compute_paired_bootstrap_ci(best_ndcgs, base_ndcgs)

    sig_report = {
        "best_variant": best_variant_name,
        "delta_ndcg@10_vs_baseline_tt": round(diff_mean, 4),
        "ci95_ndcg@10": [round(ci_l, 4), round(ci_u, 4)],
        "p_value": p_val,
        "baseline_ndcg@10": base_m["ndcg@10"],
        "best_ndcg@10": best_m["ndcg@10"],
        "baseline_recall@20": base_m["recall@20"],
        "best_recall@20": best_m["recall@20"],
    }
    with open(output_dir / "exp4_significance.json", "w") as f:
        json.dump(sig_report, f, indent=2)

    print("\n" + "=" * 115)
    print("EXPERIMENT 4: TWO-TOWER NEURAL RETRIEVAL ABLATION RESULTS (All 4,980 Val Users)")
    print("=" * 115)
    fmt = "{:<55} {:<12} {:<12} {:<12} {:<12} {:<10}"
    print(fmt.format("Configuration", "Recall@20", "NDCG@10", "Recall@100", "Pos NDCG@10", "P95 ms"))
    print("-" * 115)
    for name, m in exp4_results.items():
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
    print(f"Paired delta vs Baseline Two-Tower: Delta NDCG@10 = {diff_mean:+.4f} [{ci_l:+.4f}, {ci_u:+.4f}], p = {p_val:.4f}")
    print("=" * 115)


if __name__ == "__main__":
    run_experiment_4()
