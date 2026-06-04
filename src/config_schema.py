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
