from pydantic import BaseModel


class Config(BaseModel):
    path: str
    target_column: str
    test_size: float


class ModelConfig(BaseModel):
    model_name: str
    num_classes: int


class ModelLogisticRegressionConfig(BaseModel):
    C: float
    max_iter: int
    class_weight: str
    random_state: int


class ModelSimpleTransformerConfig(BaseModel):
    d_model: int
    nhead: int
    num_layers: int
    num_classes: int
    vocab_size: int


class ModelTfidfConfig(BaseModel):
    max_features: int
    stop_words: str


class TrainingConfig(BaseModel):
    seed: int
    batch_size: int
    num_epochs: int
    learning_rate: float


class AppConfig(BaseModel):
    data: Config
    model: ModelConfig
    training: TrainingConfig
    tfidf: ModelTfidfConfig
    logistic_regression: ModelLogisticRegressionConfig
    my_model_params: ModelSimpleTransformerConfig
