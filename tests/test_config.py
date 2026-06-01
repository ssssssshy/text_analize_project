from src.config_schema import AppConfig
from src.utils import load_config


def test_load_config():
    cfg = load_config("config/default.yaml")
    assert cfg.data.path == "/data"
    assert cfg.model.model_name == "ruBert-tiny"
