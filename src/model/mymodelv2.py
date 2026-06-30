import math
from typing import Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Injects information about token order so the Transformer is no longer
    permutation-invariant. Implemented as a non-persistent buffer: it adds no
    trainable parameters, is recomputed on load, and generalizes to any
    sequence length up to ``max_len``.
    """

    def __init__(self, embedding_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # slice div_term in case embedding_dim is odd
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])

        # shape [1, max_len, embedding_dim] so it broadcasts over the batch.
        # persistent=False -> not stored in state_dict (it is deterministic).
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, embedding_dim]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MyModelV2(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        num_layers,
        num_classes,
        max_len: int = 512,
        pad_token_id: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super(MyModelV2, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )
        self.pos_encoder = PositionalEncoding(
            embedding_dim, max_len=max_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(
            embedding_dim,
            out_features=num_classes,
        )

    def forward(self, x, attention_mask=None):
        # scale embeddings, then add positional information
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # masked mean pooling: average only over real tokens, ignoring padding
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            summed = (x * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            x = summed / counts
        else:
            x = torch.mean(x, dim=1)

        logits = self.fc(x)
        return logits
