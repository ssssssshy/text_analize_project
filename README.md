# Text Analysis Training Project

This project provides a framework for training and evaluating text analysis models (e.g., toxicity detection) using different approaches, including TF-IDF with Logistic Regression, BERT-based fine-tuning, and custom neural network architectures.

## Project Structure

- `config/`: Configuration files in YAML format.
- `data/`: Directory for raw and processed data.
- `src/`: Source code for data processing, training, and prediction.
  - `model/`: Custom model definitions.
  - `dataset.py`: Data loading and preprocessing.
  - `train_*.py`: Training scripts for different models (TF-IDF, BERT, Custom).
  - `predict.py`: Script to generate predictions using a trained model.
- `notebooks/`: Jupyter notebooks for EDA and experimentation.
- `tests/`: Unit tests for the project.

## Requirements

The project uses a variety of machine learning libraries including:
- PyTorch
- Hugging Face Transformers
- scikit-learn
- MLflow (for experiment tracking)
- FastAPI (for serving)

See `requirements.txt` for the full dependency list.

## Setup

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure the project by modifying `config/default.yaml`.
5. Run training scripts:
   - For TF-IDF: `python -m src.train_tfidf`
   - For BERT: `python -m src.train_bert`
   - For Custom Model: `python -m src.train_mymodel`

## Usage

### Training
Use the provided `train_*.py` scripts to train models. Configuration is loaded from `config/default.yaml`.

### Prediction
Use `src/predict.py` to run inference on new text data.

### Experiment Tracking
The project uses MLflow to track experiments. You can view them by running `mlflow ui` in the project root directory.
