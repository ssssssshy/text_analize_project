from torch import nn
import torch

from src.utils import load_config

cfg = load_config()


class MyModelV2(nn.Module):
    def __init__(self, tokenizer):
        super(MyModelV2, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=cfg.mymodelv2_params.embedding_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.mymodelv2_params.embedding_dim,
            nhead=cfg.mymodelv2_params.num_heads,
            dim_feedforward=cfg.mymodelv2_params.embedding_dim * 4,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.mymodelv2_params.num_layers
        )

        self.fc = nn.Linear(
            cfg.mymodelv2_params.embedding_dim,
            out_features=cfg.mymodelv2_params.num_classes,
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
