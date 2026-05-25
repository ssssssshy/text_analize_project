from pydantic import BaseModel


class Config(BaseModel):
    path: str
    target_column: str = "label"
    test_size: float = 0.2


class ModelConfig(BaseModel):
    model_name: str = "rubert"
    num_classes: int = 3


class TrainingConfig(BaseModel):
    seed: int = 42
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5


class AppConfig(BaseModel):
    config: Config
    model_config: ModelConfig
    training_config: TrainingConfig
