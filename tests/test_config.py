from src.utils import load_config


def test_load_config():
    cfg = load_config("config/default.yaml")
    assert cfg.data.path == "data/processed/cleaned_dataset.csv"
    assert cfg.model.model_name == "cointegrated/rubert-tiny2"
