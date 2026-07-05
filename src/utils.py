import yaml
import random
import numpy as np
import torch
import os

from src.config_schema import AppConfig


def load_config(config_path="config/default.yaml") -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return AppConfig(**config_dict)


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, model):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model

        if hasattr(base_model, "save_pretrained"):
            base_model.save_pretrained(os.path.dirname(self.save_path))
        else:
            torch.save(base_model.state_dict(), self.save_path)
