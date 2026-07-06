from pydantic import BaseModel


class DataConfig(BaseModel):
    path: str
    target_column: str
    text_column: str
    test_size: float
    max_length: int


class ModelConfig(BaseModel):
    model_name: str
    num_classes: int


class MyModelV2Params(BaseModel):
    embedding_dim: int
    num_heads: int
    num_layers: int
    num_classes: int
    max_seq_length: int


class TrainingConfig(BaseModel):
    seed: int
    batch_size: int
    mymodel_epochs: int
    bert_epochs: int
    bert_lr: float
    mymodel_lr: float


class PathConfig(BaseModel):
    bert_dir: str
    mymodelv2_dir: str


class AppConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    mymodelv2_params: MyModelV2Params
    paths: PathConfig
