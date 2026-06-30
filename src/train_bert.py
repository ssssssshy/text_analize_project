import os
import time
import torch
import torch.optim as optim
import tqdm
import wandb
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.dataset import TextDataset, load_data
from src.utils import EarlyStopping, load_config, set_seed


def train_bert():
    """Train Bert model(cointegrated/rubert-tiny2) for sequence classification

    and log parameters, metrics, and artifacts to wandb.
    """
    cfg = load_config("config/default.yaml")
    set_seed(cfg.training.seed)

    wandb.init(
        project="tat",
        name="bert-sequence-classification-with-es-lrsh",
        config=dict(cfg),
    )

    device = (
        torch.accelerator.current_accelerator().type  # type: ignore
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
        x_train["comment"].tolist(), y_train.tolist(), tokenizer
    )
    val_dataset = TextDataset(x_val["comment"].tolist(), y_val.tolist(), tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg.training.bert_lr))

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=2,
        verbose=True,  # type: ignore
    )

    save_dir = "models/rubert_tiny2"
    early_stopping = EarlyStopping(
        patience=3, save_path=os.path.join(save_dir, "pytorch_model.bin")
    )

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

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss.mean()
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

                val_loss += outputs.loss.mean().item()

                logits = outputs.logits.detach()
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")

        wandb.log({"val_loss": avg_val_loss, "val_f1_score": val_f1, "epoch": epoch})
        print(
            f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"learning_rate": current_lr, "epoch": epoch})

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"ES на эпохе {epoch + 1}")
            break

    tokenizer.save_pretrained(save_dir)
    print(f"tokenizer successfully saved in {save_dir}")

    wandb.finish()


if __name__ == "__main__":
    train_bert()
