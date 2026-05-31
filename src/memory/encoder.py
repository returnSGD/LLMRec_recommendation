"""
Interaction → Compressed Vector Encoder.

Encodes raw interaction data into fixed-dim memory vectors for storage
in the long-term memory FAISS index. Uses a small MLP that can optionally
share the P5/T5 item/user embedding lookup.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEncoder(nn.Module):
    """
    Encodes a single interaction into a compressed memory vector.

    Input features (concatenated):
      - user_id embedding (from T5-shared or learned)
      - item_id embedding
      - action_type embedding (click/purchase/skip/return)
      - rating (normalized to [0,1])
      - timestamp delta (days since epoch, normalized)
    """

    def __init__(self, num_users: int, num_items: int,
                 num_actions: int = 4,
                 embed_dim: int = 128,
                 memory_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        self.user_embed = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
        self.item_embed = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.action_embed = nn.Embedding(num_actions, embed_dim // 2)

        # Scalar features: rating, time_delta
        input_dim = embed_dim * 2 + embed_dim // 2 + 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, memory_dim * 2),
            nn.LayerNorm(memory_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim * 2, memory_dim),
            nn.LayerNorm(memory_dim),
        )

        self.memory_dim = memory_dim

    def forward(self, user_ids: torch.LongTensor,
                item_ids: torch.LongTensor,
                action_types: torch.LongTensor,
                ratings: torch.FloatTensor,
                time_deltas: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            user_ids: (B,) user indices
            item_ids: (B,) item indices
            action_types: (B,) 0=click, 1=purchase, 2=skip, 3=return
            ratings: (B,) normalized [0,1]
            time_deltas: (B,) days since epoch, normalized

        Returns:
            memory_vectors: (B, memory_dim) L2-normalized
        """
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        a = self.action_embed(action_types)
        r = ratings.unsqueeze(-1)
        td = time_deltas.unsqueeze(-1)

        features = torch.cat([u, i, a, r, td], dim=-1)
        encoded = self.encoder(features)
        return F.normalize(encoded, p=2, dim=-1)


class MemoryEncoderPretrained(MemoryEncoder):
    """
    Variant that initializes user/item embeddings from a pretrained P5 model.
    """

    def __init__(self, p5_model, memory_dim: int = 256, dropout: float = 0.1):
        # Get vocab info from P5
        embed_dim = p5_model.config.d_model
        vocab_size = p5_model.config.vocab_size

        super().__init__(
            num_users=vocab_size,  # placeholder — will use user_id offsets
            num_items=vocab_size,
            embed_dim=embed_dim,
            memory_dim=memory_dim,
            dropout=dropout,
        )

        # Share with P5's shared embedding
        self.user_embed = p5_model.shared
        self.item_embed = p5_model.shared
