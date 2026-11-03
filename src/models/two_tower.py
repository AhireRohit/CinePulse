import torch
import torch.nn.functional as F
from torch import nn


class UserTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        num_genres: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_users = num_users

        # User ID embedding (num_users is cold-start / unknown user index)
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=num_users)

        # Item embedding shared for sequence history
        self.item_history_embedding = nn.Embedding(num_movies + 1, embedding_dim, padding_idx=0)

        # Genre preference projection
        self.genre_proj = nn.Linear(num_genres, embedding_dim)

        # Attention layer for history sequence
        self.history_attn = nn.Linear(embedding_dim, 1)

        # User representation fusion
        # Concatenates: user_id (D) + history_repr (D) + genre_pref (D) -> 3*D
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_history(self, history_seq: torch.Tensor, history_len: torch.Tensor) -> torch.Tensor:
        """
        history_seq: [B, S]
        history_len: [B]
        """
        batch_size, seq_len = history_seq.shape
        # [B, S, D]
        item_embs = self.item_history_embedding(history_seq)

        # Mask padding (0 is pad)
        mask = (history_seq != 0).unsqueeze(-1)  # [B, S, 1]

        # Attention weights
        attn_logits = self.history_attn(item_embs)  # [B, S, 1]
        attn_logits = attn_logits.masked_fill(~mask, -1e4)
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, S, 1]

        # Handle all-zero padding rows
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted sum: [B, D]
        hist_repr = (item_embs * attn_weights).sum(dim=1)
        return hist_repr

    def forward(
        self,
        user_idx: torch.Tensor,
        history_seq: torch.Tensor,
        history_len: torch.Tensor,
        user_genre_pref: torch.Tensor,
    ) -> torch.Tensor:
        # [B, D]
        user_emb = self.user_embedding(user_idx)
        hist_repr = self.encode_history(history_seq, history_len)
        genre_repr = F.relu(self.genre_proj(user_genre_pref))

        combined = torch.cat([user_emb, hist_repr, genre_repr], dim=-1)
        output = self.fusion(combined)
        return F.normalize(output, p=2, dim=-1)


class ItemTower(nn.Module):
    def __init__(
        self,
        num_movies: int,
        num_genres: int,
        dense_dim: int = 3,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim, padding_idx=num_movies)
        self.genre_proj = nn.Linear(num_genres, embedding_dim)
        self.dense_proj = nn.Linear(dense_dim, embedding_dim // 2)

        fusion_in_dim = embedding_dim + embedding_dim + (embedding_dim // 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        movie_idx: torch.Tensor,
        movie_genres: torch.Tensor,
        movie_dense: torch.Tensor,
    ) -> torch.Tensor:
        movie_emb = self.movie_embedding(movie_idx)
        genre_repr = F.relu(self.genre_proj(movie_genres))
        dense_repr = F.relu(self.dense_proj(movie_dense))

        combined = torch.cat([movie_emb, genre_repr, dense_repr], dim=-1)
        output = self.fusion(combined)
        return F.normalize(output, p=2, dim=-1)


class TwoTowerRecommender(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        num_genres: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_tower = UserTower(
            num_users=num_users,
            num_movies=num_movies,
            num_genres=num_genres,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.item_tower = ItemTower(
            num_movies=num_movies,
            num_genres=num_genres,
            dense_dim=3,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_tower(
            user_idx=batch["user_idx"],
            history_seq=batch["history_seq"],
            history_len=batch["history_len"],
            user_genre_pref=batch["user_genre_pref"],
        )
        item_emb = self.item_tower(
            movie_idx=batch["target_movie_idx"],
            movie_genres=batch["target_movie_genres"],
            movie_dense=batch["target_movie_dense"],
        )
        return user_emb, item_emb

    def compute_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """
        In-batch symmetric contrastive loss (InfoNCE).
        Positive pairs on the diagonal; all other elements in batch are negative candidates.
        """
        temp = torch.clamp(self.temperature, min=0.01, max=0.5)
        # Cosine similarity matrix: [B, B]
        similarity = torch.matmul(user_emb, item_emb.T) / temp
        targets = torch.arange(user_emb.shape[0], device=user_emb.device)
        loss = F.cross_entropy(similarity, targets)
        return loss
