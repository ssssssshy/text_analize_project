from config_shema import AppConfig
import yaml


def laod_config(config_path: str) -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return AppConfig(**config_dict)
