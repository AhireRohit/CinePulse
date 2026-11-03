import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.core.config import settings
from src.core.logging import logger
from src.data.dataset import MovieLensRetrievalDataset
from src.models.two_tower import TwoTowerRecommender


def train_two_tower(
    data_dir: Path = Path("data/processed"),
    output_dir: Path = Path("artifacts/models"),
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not settings.force_cpu else "cpu")
    logger.info(f"Training Two-Tower model on device: {device}")

    # 1. Load data
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    train_ratings = pd.read_parquet(data_dir / "ratings_train.parquet")
    val_ratings = pd.read_parquet(data_dir / "ratings_val.parquet")

    with open(data_dir / "mappings.json") as f:
        mappings = json.load(f)
    with open(data_dir / "user_train_histories.json") as f:
        user_histories = json.load(f)

    num_users = mappings["num_users"]
    num_movies = mappings["num_movies"]
    genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]
    num_genres = len(genre_cols)

    logger.info(f"Dataset stats: {num_users:,} users, {num_movies:,} movies, {len(train_ratings):,} train ratings")

    # 2. Datasets & Loaders
    train_dataset = MovieLensRetrievalDataset(
        ratings_df=train_ratings,
        movies_df=movies_df,
        user_histories=user_histories,
        max_seq_len=settings.max_sequence_length,
        genre_cols=genre_cols,
    )
    val_dataset = MovieLensRetrievalDataset(
        ratings_df=val_ratings,
        movies_df=movies_df,
        user_histories=user_histories,
        max_seq_len=settings.max_sequence_length,
        genre_cols=genre_cols,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # 0 workers avoids Windows multiprocessing issues
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # 3. Model & Optimizer
    model = TwoTowerRecommender(
        num_users=num_users,
        num_movies=num_movies,
        num_genres=num_genres,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        temperature=0.07,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_model_path = output_dir / "two_tower_best.pt"

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        t0 = time.perf_counter()

        for batch in train_loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                user_emb, item_emb = model(batch_device)
                loss = model.compute_loss(user_emb, item_emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            train_batches += 1
            if train_batches % 500 == 0:
                logger.info(f"Epoch {epoch:02d} | Batch {train_batches}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, train_batches)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_device = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    user_emb, item_emb = model(batch_device)
                    loss = model.compute_loss(user_emb, item_emb)
                total_val_loss += loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / max(1, val_batches)
        elapsed = time.perf_counter() - t0

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "num_users": num_users,
                    "num_movies": num_movies,
                    "num_genres": num_genres,
                },
                best_model_path,
            )
            logger.info(f"  --> Saved new best checkpoint to {best_model_path}")

    # 5. Precompute item embeddings for all movies
    logger.info("Precomputing full catalog item embeddings...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Build full catalog tensors
    all_movie_indices = torch.arange(num_movies, dtype=torch.long, device=device)
    all_genres = train_dataset.movie_genres.to(device)
    all_dense = train_dataset.movie_dense_features.to(device)

    item_embeddings_list = []
    chunk_size = 1024
    with torch.no_grad():
        for i in range(0, num_movies, chunk_size):
            chunk_idx = all_movie_indices[i : i + chunk_size]
            chunk_g = all_genres[i : i + chunk_size]
            chunk_d = all_dense[i : i + chunk_size]
            emb = model.item_tower(chunk_idx, chunk_g, chunk_d)
            item_embeddings_list.append(emb.cpu().numpy())

    item_embeddings = np.vstack(item_embeddings_list)
    np.save(output_dir / "item_embeddings.npy", item_embeddings)
    logger.info(f"Item embeddings saved: shape {item_embeddings.shape} to {output_dir / 'item_embeddings.npy'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Two-Tower Candidate Retrieval Model")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    args = parser.parse_args()

    train_two_tower(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
    )
