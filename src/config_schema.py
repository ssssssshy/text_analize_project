from pydantic import BaseModel


class Config(BaseModel):
    path: str
    target_column: str
    text_column: str
    test_size: float


class ModelConfig(BaseModel):
    model_name: str
    num_classes: int


class mymodelv2_params(BaseModel):
    embedding_dim: int
    num_heads: int
    num_layers: int
    num_classes: int


class TrainingConfig(BaseModel):
    seed: int
    batch_size: int
    num_epochs: int
    learning_rate: float


class AppConfig(BaseModel):
    data: Config
    model: ModelConfig
    training: TrainingConfig
    mymodelv2_params: mymodelv2_params
