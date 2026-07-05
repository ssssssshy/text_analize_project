# Text Analysis Training Project

This project provides a robust framework for training, evaluating, and serving text analysis models (e.g., toxicity detection) using PyTorch, Hugging Face Transformers, and scikit-learn.

## Features
- **Flexible Training:** Supports various architectures (BERT, custom models).
- **Experiment Tracking:** All training runs are logged to [Weights & Biases (W&B)](https://api.wandb.ai/links/petrosangosa2005-ssss/axeypnhm) for comprehensive metrics, model artifacts, and version control.
- **Inference:** FastAPI-based service for model serving.
- **Preprocessing:** Standardized pipelines for text data.

## Setup

### 1. Prerequisites
- Python 3.12
- Conda or Mamba

### 2. Environment Setup
```bash
mamba env create -f environment.yml
mamba activate ml
```

## Workflows

### Configuration
Adjust hyperparameters, data paths, and model settings in `config/default.yaml` before executing any pipelines.

### Training
Run training scripts directly from the source:
```bash
python -m src.train_bert  # Example for BERT
```
Training metrics, system logs, and model checkpoints are automatically synchronized to **Weights & Biases**. Ensure you are logged in (`wandb login`) before starting training.

### Inference
Use the prediction script to run inference on new data:
```bash
python -m src.predict
```

## Testing
To run the test suite and ensure project stability:
```bash
python -m pytest tests/
```
