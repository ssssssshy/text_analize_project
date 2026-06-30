# Text Analysis Training Project

This project provides a framework for training and evaluating text analysis models (e.g., toxicity detection) using different approaches, including TF-IDF with Logistic Regression, BERT-based fine-tuning, and custom neural network architectures.

## Project Structure

- `config/`: Configuration files in YAML format.
- `data/`: Directory for raw and processed data (ignored by Git).
- `models/`: Pre-trained model weights and configurations (managed via Git LFS):
  - `bert_sequence_classification/`: Fine-tuned BERT model weights (`model.safetensors`) and tokenizer configs.
  - `custom_modelv2/`: Custom transformer weights (`model_weights.pth`) and tokenizer configs.
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
- Weights & Biases (for experiment tracking)

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
mamba activate ml
#(Note: By default, the environment is named ml as specified in the environment.yml file).
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
#Тетстируем MymodelV2 и BERT
python -m src.predict

```
