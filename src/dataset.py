import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from src.utils import load_config


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
