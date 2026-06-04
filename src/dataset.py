import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
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
    """Custom Dataset for text data.
    Args:
        texts (pd.DataFrame): The input text data.
        labels (pd.Series): The target labels.
    """

    def __init__(self, texts: pd.DataFrame, labels: pd.Series):
        super().__init__()
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts.iloc[idx], self.labels.iloc[idx]
