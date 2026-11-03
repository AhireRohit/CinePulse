import numpy as np
import torch

from src.models.two_tower import TwoTowerRecommender
from src.retrieval.faiss_index import FaissCandidateIndex


def test_two_tower_model_forward():
    num_users = 100
    num_movies = 50
    num_genres = 18
    dim = 32

    model = TwoTowerRecommender(
        num_users=num_users,
        num_movies=num_movies,
        num_genres=num_genres,
        embedding_dim=dim,
    )

    batch_size = 8
    seq_len = 10
    batch = {
        "user_idx": torch.randint(0, num_users, (batch_size,)),
        "target_movie_idx": torch.randint(0, num_movies, (batch_size,)),
        "history_seq": torch.randint(0, num_movies, (batch_size, seq_len)),
        "history_len": torch.randint(1, seq_len, (batch_size,)),
        "user_genre_pref": torch.rand(batch_size, num_genres),
        "target_movie_genres": torch.randint(0, 2, (batch_size, num_genres)).float(),
        "target_movie_dense": torch.rand(batch_size, 3),
    }

    user_emb, item_emb = model(batch)
    assert user_emb.shape == (batch_size, dim)
    assert item_emb.shape == (batch_size, dim)

    # Check L2 normalized
    user_norms = torch.norm(user_emb, dim=1)
    item_norms = torch.norm(item_emb, dim=1)
    assert torch.allclose(user_norms, torch.ones_like(user_norms), atol=1e-4)
    assert torch.allclose(item_norms, torch.ones_like(item_norms), atol=1e-4)

    loss = model.compute_loss(user_emb, item_emb)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_faiss_candidate_index():
    num_items = 200
    dim = 64
    rng = np.random.default_rng(42)
    embeddings = rng.normal(0, 1, size=(num_items, dim)).astype(np.float32)

    index = FaissCandidateIndex(dimension=dim)
    index.build(item_embeddings=embeddings)

    assert index.num_items == num_items

    # Query with a random vector
    query_vec = rng.normal(0, 1, size=(dim,)).astype(np.float32)
    top_k = 15
    results = index.search(query_vec, top_k=top_k)

    assert len(results) == top_k
    # Check monotonic scores
    scores = [s for _, s in results]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    # Test exclusion filtering
    excluded = {results[0][0], results[1][0]}
    filtered_results = index.search(query_vec, top_k=top_k, exclude_item_indices=excluded)
    filtered_ids = {idx for idx, _ in filtered_results}
    assert not (filtered_ids & excluded), "Excluded items should not appear in search results"


def test_cold_start_user_representation():
    """
    Verifies that an unknown / cold-start user (with index above num_users)
    can generate a valid normalized query vector from genre preferences alone.
    """
    model = TwoTowerRecommender(
        num_users=50,
        num_movies=50,
        num_genres=18,
        embedding_dim=64,
    )
    model.eval()

    # Anonymous user with index 50 (padding/unknown index)
    u_idx = torch.tensor([50], dtype=torch.long)
    h_seq = torch.zeros((1, 10), dtype=torch.long)
    h_len = torch.tensor([0], dtype=torch.long)
    genre_pref = torch.zeros((1, 18), dtype=torch.float32)
    genre_pref[0, 0] = 1.0  # Prefers genre 0 (e.g. Sci-Fi)

    with torch.no_grad():
        emb = model.user_tower(u_idx, h_seq, h_len, genre_pref)

    assert emb.shape == (1, 64)
    norm = torch.norm(emb, dim=1).item()
    assert abs(norm - 1.0) < 1e-4
