import mlflow
import torch
import torch.optim as optim
import tqdm
from mlflow.transformers import log_model as log_bert_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.dataset import DynamicPaddingCollator, TextDataset, load_data
from src.utils import load_config


def train_bert():
    """
    Train Bert model(cointegrated/rubert-tiny2) for sequence classification and log parameters, metrics, and artifacts to MLflow.
        - Load configuration from YAML file.
        - Load and split the dataset.
        - Preprocess text data using Bert tokenizer.
        - Train a Bert model for sequence classification.
        - Evaluate the model and log the F1 score.
        - Save the trained model and tokenizer as artifacts in MLflow.
    Returns:
        None
    """
    cfg = load_config("config/default.yaml")

    device = (
        torch.accelerator.current_accelerator().type  # pyright: ignore[reportOptionalMemberAccess]
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Используется устройство для обучения: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name, num_labels=cfg.model.num_classes
    )
    model.to(device)

    x_train, x_val, x_test, y_train, y_val, y_test = load_data(cfg.data.path)
    train_dataset = TextDataset(
        x_train[cfg.data.text_column].tolist(),
        y_train.tolist(),
        tokenizer,
        max_length=cfg.data.max_length,
    )
    val_dataset = TextDataset(
        x_val[cfg.data.text_column].tolist(),
        y_val.tolist(),
        tokenizer,
        max_length=cfg.data.max_length,
    )

    collator = DynamicPaddingCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg.training.learning_rate))

    mlflow.set_experiment("BERT_Sequence_Classification")
    with mlflow.start_run():
        mlflow.log_param("model_name", cfg.model.model_name)
        mlflow.log_param("num_classes", cfg.model.num_classes)
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
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
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
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    val_loss += outputs.loss.item()

                    preds = torch.argmax(outputs.logits, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(all_labels, all_preds, average="weighted")

            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_f1_score", float(val_f1), step=epoch)

            print(
                f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}"
            )

        log_bert_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="bert_model",
            task="text-classification",
        )

        save_dir = cfg.paths.bert_dir
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Модель и токенизатор успешно сохранены локально в папку: {save_dir}")


if __name__ == "__main__":
    train_bert()
