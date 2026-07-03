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

        self.position_embedding = nn.Embedding(
            num_embeddings=cfg.mymodelv2_params.max_seq_length,
            embedding_dim=embedding_dim,
        )

        self.embedding_dropout = nn.Dropout(p=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.fc = nn.Linear(
            embedding_dim,
            out_features=num_classes,
        )

    def forward(self, x, attention_mask=None):
        word_embeddings = self.embedding(x)

        seq_len = x.size(1)
        position_embeddings = (
            self.position_embedding(torch.arange(seq_len, device=x.device))
            .unsqueeze(0)
            .expand(x.size(0), -1, -1)
        )
        x = word_embeddings + position_embeddings

        x = self.embedding_dropout(x)

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        if attention_mask is not None:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()

            sum_embeddings = torch.sum(x * input_mask_expanded, dim=1)

            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

            x = sum_embeddings / sum_mask
        else:
            x = torch.mean(x, dim=1)

        logits = self.fc(x)
        return logits
