import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

import torch
from src.utils import load_config
from torch.utils.data.dataset import Dataset


def load_data(
    path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load data from a CSV file.

    Args:
        path (str): The path to the CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: The loaded data.
    """
    df = pd.read_csv(path)

    cfg = load_config("config/default.yaml")

    X = df.drop(columns=[cfg.data.target_column])
    y = df[cfg.data.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.training.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=cfg.training.seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


class TextDataset(Dataset):
    """Tokenizes text lazily, WITHOUT padding.

    Padding is applied per-batch by ``DynamicPaddingCollator`` so each batch is
    only as long as its own longest example, instead of a fixed 512 tokens.
    ``max_length`` only truncates and is shared with inference via the config.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            # no padding here -- done dynamically per batch in the collator
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": int(self.labels[idx]),
        }


class DynamicPaddingCollator:
    """Pads each batch to its longest sequence using the tokenizer's pad token."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }
            for item in batch
        ]
        padded = self.tokenizer.pad(features, padding="longest", return_tensors="pt")
        padded["labels"] = torch.tensor(
            [item["labels"] for item in batch], dtype=torch.long
        )
        return padded
