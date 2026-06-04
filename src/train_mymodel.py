import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import mlflow
import tqdm
from sklearn.metrics import f1_score
from mlflow.pytorch import log_model

from src.model.mymodel import SimpleTransformerClassifier
from src.dataset import load_data, TextDataset
from src.utils import load_config


def train_my_model():
    """Train a custom transformer model for sequence classification.

    Logs hyper-parameters, metrics, and weights to MLflow, and saves
    the final checkpoints locally.
    """
    cfg = load_config("config/default.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство для обучения: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    model = SimpleTransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.model_params.d_model,
        nhead=cfg.model_params.nhead,
        num_layers=cfg.model_params.num_layers,
        num_classes=cfg.model_params.num_classes,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    x_train, x_val, x_test, y_train, y_val, y_test = load_data(
        "data/processed/labeled.csv"
    )
    train_dataset = TextDataset(
        x_train["comment"].tolist(), y_train.tolist(), tokenizer
    )
    val_dataset = TextDataset(x_val["comment"].tolist(), y_val.tolist(), tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg.training.learning_rate))

    mlflow.set_experiment("Custom_Transformer_Classification")

    with mlflow.start_run():
        mlflow.log_param("model_type", "Custom_SimpleTransformer")
        mlflow.log_param("batch_size", cfg.training.batch_size)
        mlflow.log_param("learning_rate", cfg.training.learning_rate)
        mlflow.log_param("num_epochs", cfg.training.num_epochs)

        for epoch in range(cfg.training.num_epochs):
            model.train()
            train_loss = 0

            for batch in tqdm.tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs} [Train]",
            ):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids=input_ids)
                loss = criterion(logits, labels)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = train_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")

            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm.tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs} [Val]",
                ):
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)

                    logits = model(input_ids=input_ids)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(all_labels, all_preds, average="weighted")

            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_f1_score", float(val_f1), step=epoch)

            print(
                f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}"
            )

        log_model(model, artifact_path="custom_transformer_model")

        save_dir = "models/custom_transformer_classification"
        os.makedirs(save_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))
        tokenizer.save_pretrained(save_dir)
        print(f"Модель и токенизатор успешно сохранены локально в папку: {save_dir}")


if __name__ == "__main__":
    train_my_model()
