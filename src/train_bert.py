import os
import time
import torch
import torch.optim as optim
import tqdm
import wandb
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.dataset import TextDataset, load_data
from src.utils import EarlyStopping, load_config, set_seed


def train_bert():
    """Train Bert model (cointegrated/rubert-tiny2) for sequence classification."""
    cfg = load_config("config/default.yaml")
    set_seed(cfg.training.seed)

    wandb.init(
        project="tat",
        name="rubert-tiny2_baseline_v2",
        config={
            "architecture": "rubert-tiny2",
            "task": "classification",
            "dataset": "base-comments",
            "early_stopping": True,
            "lr_scheduler": "CosineWarmup",
            "weight_decay": 0.01,
            **dict(cfg),
        },
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

    decay_parameters = [
        n
        for n, p in model.named_parameters()
        if p.dim() > 1
        and not any(nd in n for nd in ["bias", "LayerNorm.weight", "LayerNorm.bias"])
    ]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in decay_parameters and p.requires_grad
            ],
            "weight_decay": 0.05,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in decay_parameters and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters, lr=float(cfg.training.bert_lr)
    )

    num_training_steps = len(train_loader) * cfg.training.bert_epochs

    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    save_dir = "models/rubert_tiny2"
    early_stopping = EarlyStopping(
        patience=3, save_path=os.path.join(save_dir, "pytorch_model.bin")
    )

    for epoch in range(cfg.training.bert_epochs):
        model.train()
        train_loss = 0
        epoch_start_time = time.time()

        for batch in tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.training.bert_epochs} [Train]",
        ):
            optimizer.zero_grad()

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        epoch_train_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm.tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{cfg.training.bert_epochs} [Val]",
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
        print(
            f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_f1_score": val_f1,
                "learning_rate": current_lr,
                "epoch_duration_seconds": epoch_train_duration,
            }
        )

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"ES на эпохе {epoch + 1}")
            break

    tokenizer.save_pretrained(save_dir)
    print(f"tokenizer successfully saved in {save_dir}")

    wandb.finish()


if __name__ == "__main__":
    train_bert()
