import os
import wandb
import tqdm
import torch
import pandas as pd
import time
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model.mymodelv2 import MyModelV2
from src.utils import load_config, set_seed
from src.dataset import load_data, TextDataset


def train_mymodelv2():
    device = (
        torch.accelerator.current_accelerator().type  # pyright: ignore[reportOptionalMemberAccess]
        if torch.accelerator.is_available()
        else "cpu"
    )

    cfg = load_config("config/default.yaml")

    set_seed(cfg.training.seed)

    wandb.init(project="tat", name="mymodelv2-classification", config=dict(cfg))

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(cfg.data.path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    train_dataset = TextDataset(
        X_train[cfg.data.text_column].tolist(), y_train.tolist(), tokenizer
    )
    val_dataset = TextDataset(
        X_val[cfg.data.text_column].tolist(), y_val.tolist(), tokenizer
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)

    model = MyModelV2(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=cfg.mymodelv2_params.embedding_dim,
        num_heads=cfg.mymodelv2_params.num_heads,
        num_layers=cfg.mymodelv2_params.num_layers,
        num_classes=cfg.mymodelv2_params.num_classes,
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.mymodel_lr)

    for epoch in range(cfg.training.num_epochs):
        model.train()
        train_loss = 0
        epoch_start_time = time.time()

        for batch in tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs} [Train]",
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels).mean()
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        epoch_train_duration = time.time() - epoch_start_time
        wandb.log(
            {
                "train_loss": avg_train_loss,
                "epoch": epoch,
                "epoch_duration_seconds": epoch_train_duration,
            }
        )
        print(
            f"Epoch {epoch + 1}/{cfg.training.num_epochs} - Train Loss: {avg_train_loss:.4f}"
        )

        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm.tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs} [Validation]",
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask).detach()
                loss = criterion(logits, labels).mean()
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                class_1_probs = probs[:, 1]

                preds = torch.argmax(logits, dim=1).detach()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(class_1_probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        wandb.log({"val_loss": avg_val_loss, "val_f1_score": val_f1, "epoch": epoch})
        print(
            f"Epoch {epoch + 1}/{cfg.training.num_epochs} - Validation Loss: {avg_val_loss:.4f}"
        )

        print(
            f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

    save_dir = "models/mymodelv2_classification"
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(model, torch.nn.DataParallel):
        torch.save(
            model.module.state_dict(), os.path.join(save_dir, "model_weights.pth")
        )
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))
    tokenizer.save_pretrained(save_dir)
    print(f"\n Обучение завершено Модель сохранена в: {save_dir}")
    wandb.finish()


if __name__ == "__main__":
    train_mymodelv2()
