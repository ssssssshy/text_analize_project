import yaml

from src.config_schema import AppConfig


def load_config(config_path: str) -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return AppConfig(**config_dict)
