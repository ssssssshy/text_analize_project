from src.utils import load_config


def test_load_config():
    cfg = load_config("config/default.yaml")

    # values must match config/default.yaml
    assert cfg.data.path == "data/processed/cleaned_dataset.csv"
    assert cfg.model.model_name == "cointegrated/rubert-tiny2"

    # sanity checks on types / ranges
    assert isinstance(cfg.training.learning_rate, float)
    assert cfg.data.max_length > 0
    assert 0.0 < cfg.data.test_size < 1.0

    # training and inference must agree on where models live
    assert cfg.paths.bert_dir
    assert cfg.paths.mymodelv2_dir
