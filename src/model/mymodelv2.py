from torch import nn
import torch

from src.utils import load_config

cfg = load_config()


class MyModelV2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, num_classes):
        super(MyModelV2, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(
            embedding_dim,
            out_features=num_classes,
        )

    def forward(self, x, attention_mask=None):
        x = self.embedding(x)

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        x = torch.mean(x, dim=1)
        logits = self.fc(x)
        return logits
