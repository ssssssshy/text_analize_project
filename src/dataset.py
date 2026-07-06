import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

import torch
from src.utils import load_config
from torch.utils.data.dataset import Dataset


def load_data(
    path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load data from a CSV file."""
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
    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label,
        }


class DynamicPaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels,
        }
