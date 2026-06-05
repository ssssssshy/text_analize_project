# Text Analysis Training Project

This project provides a framework for training and evaluating text analysis models (e.g., toxicity detection) using different approaches, including TF-IDF with Logistic Regression, BERT-based fine-tuning, and custom neural network architectures.

## Project Structure

- `config/`: Configuration files in YAML format.
- `data/`: Directory for raw and processed data (ignored by Git).
- `models/`: Pre-trained model weights and configurations (managed via Git LFS):
  - `TF IDF/`: Saved TF-IDF vectorizer and Logistic Regression artifacts.
  - `bert_sequence_classification/`: Fine-tuned BERT model weights (`model.safetensors`) and tokenizer configs.
  - `custom_transformer_classification/`: Custom transformer weights (`model_weights.pth`) and tokenizer configs.
- `src/`: Source code for data processing, training, and prediction.
  - `model/`: Custom model architectures.
  - `dataset.py`: Data loading and preprocessing pipelines.
  - `train_*.py`: Training scripts for different models (TF-IDF, BERT, Custom).
  - `predict.py`: Script to generate predictions using a trained model.
- `notebooks/`: Jupyter notebooks for EDA and experimentation.
- `tests/`: Unit tests for the project components.

## Requirements

The project utilizes a comprehensive machine learning stack:
- PyTorch & Hugging Face Transformers
- scikit-learn
- MLflow (for experiment tracking)

---

## 🛠️ Local Development & Evaluation Setup

Follow these steps to clone the repository, restore the pre-trained models via Git LFS, and run the project locally.

### 1. Clone the Repository (with Git LFS)
Make sure you have **Git LFS** installed before cloning, so that the heavy model binaries are downloaded automatically:
```bash
git lfs install
git clone https://github.com/ssssssshy/text_analize_project.git
cd text_analize_project
git lfs pull

```

### 2. Environment Management (Conda / Mamba)

The environment dependencies are managed via Conda. We highly recommend using **Mamba** (a fast, drop-in alternative to conda) for significantly faster dependency resolution and setup.

1. **Create the environment from the configuration file:**
```bash
# Using Mamba (Recommended)
mamba env create -f environment.yml

# Alternatively, using standard Conda
conda env create -f environment.yml

```


2. **Activate the environment:**
```bash
# Using Mamba (Recommended)
mamba activate <environment_name>
# Alternatively, using standard Conda
conda activate <environment_name>

```


*(Note: Replace `<environment_name>` with the actual name defined at the top of your `environment.yml` file, e.g., `ml` or `text-analysis`)*

---

## 📊 Usage & Workflows

### 1. Configuration

Before running any scripts, adjust the parameters, data paths, and model selections in `config/default.yaml`.

### 2. Inference & Prediction

To test the pre-trained models without re-running the training pipelines, use the `src/predict.py` script to run inference on new data:

```bash
python -m src.predict

```

### 3. Training Pipelines

To retrain any of the models from scratch, execute the corresponding module from the project root:

* **TF-IDF + Logistic Regression:**
```bash
python -m src.train_tfidf

```


* **BERT Fine-Tuning:**
```bash
python -m src.train_bert

```


* **Custom Transformer:**
```bash
python -m src.train_mymodel

```



### 4. Experiment Tracking (MLflow)

Every training run automatically logs metrics, parameters, and loss curves. To inspect the training history and compare models, start the local MLflow server:

```bash
mlflow ui

```

Then, navigate to `http://127.0.0.1:5000` in your browser.

```

```
