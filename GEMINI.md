# Text Analysis Training Project

This is a machine learning project designed for training, evaluating, and serving text analysis models (e.g., toxicity detection).

## Architecture & Technologies

- **Language:** Python 3.12
- **Core ML:** PyTorch, Hugging Face Transformers, scikit-learn.
- **Experiment Tracking:** MLflow, Weights & Biases (W&B).
- **Inference:** FastAPI-based service.
- **Dependency Management:** Conda/Mamba (`environment.yml`).

## Project Structure

- `src/`: Source code.
  - `model/`: Custom model architectures (`mymodelv2.py`).
  - `train_*.py`: Training pipelines.
  - `predict.py`: Inference script.
  - `dataset.py`: Data loading and preprocessing.
- `config/`: Configuration files (YAML).
- `models/`: Storage for model weights (tracked via Git LFS).
- `notebooks/`: Exploratory Data Analysis (EDA) notebooks.
- `tests/`: Unit and integration tests.

## Development Workflows

### Environment Setup
- **Create Environment:** `mamba env create -f environment.yml`
- **Activate:** `mamba activate ml`

### Testing
- **Run Tests:** `python -m pytest tests/`
- *Note:* Always ensure all unit tests pass before submitting changes.

### Training & Inference
- **Inference:** `python -m src.predict`
- **Configuration:** Modify `config/default.yaml` before running training or inference scripts.

## Conventions

- **Coding Style:** Adhere to PEP 8 standards. Use `black` for formatting and `ruff` for linting.
- **Documentation:** Maintain `README.md`. Document all new models or significant changes to the pipeline.
- **Testing:** New features MUST be accompanied by unit tests in the `tests/` directory.
- **Experiment Tracking:** Use MLflow/WandB to log metrics and model artifacts for all training runs.
