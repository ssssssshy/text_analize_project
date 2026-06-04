import torch
import torch.nn as nn


class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        x = self.transformer(x)

        pooled = torch.mean(x, dim=1)

        logits = self.classifier(pooled)
        return logits
