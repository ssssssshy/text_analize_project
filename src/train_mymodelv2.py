import os
import time
import torch
import tqdm
import wandb
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import lr_scheduler

from src.dataset import TextDataset, load_data
from src.model.mymodelv2 import MyModelV2
from src.utils import EarlyStopping, load_config, set_seed


def train_mymodelv2():
    device = (
        torch.accelerator.current_accelerator().type  # type: ignore
        if torch.accelerator.is_available()
        else "cpu"
    )

    cfg = load_config("config/default.yaml")
    set_seed(cfg.training.seed)

    wandb.init(
        project="tat",
        name="mymodelv2_pos-en_v2",
        config={
            "architecture": "mymodelv2",
            "task": "classification",
            "dataset": "pos-en",
            "early_stopping": True,
            "lr_scheduler": "CosineAnnealingLR",
            "weight_decay": 0.05,
            "dropout": 0.3,
            **dict(cfg),
        },
    )

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

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=cfg.training.mymodel_lr
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.mymodel_epochs, eta_min=1e-6
    )

    save_dir = "models/mymodelv2_classification"
    early_stopping = EarlyStopping(
        patience=3, save_path=os.path.join(save_dir, "model_weights.pth")
    )

    for epoch in range(cfg.training.mymodel_epochs):
        model.train()
        train_loss = 0
        epoch_start_time = time.time()

        for batch in tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.training.mymodel_epochs} [Train]",
        ):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels).mean()
            train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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
                desc=f"Epoch {epoch + 1}/{cfg.training.mymodel_epochs} [Validation]",
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask).detach()
                loss = criterion(logits, labels).mean()
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).detach()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        print(
            f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        scheduler.step()
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
    train_mymodelv2()
