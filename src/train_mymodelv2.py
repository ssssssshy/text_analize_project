from src.model.mymodelv2 import MyModelV2
from src.utils import load_config
from src.dataset import load_data, TextDataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
import tqdm
import os
from sklearn.metrics import f1_score


def train_mymodelv2():
    device = (
        torch.accelerator.current_accelerator().type  # pyright: ignore[reportOptionalMemberAccess]
        if torch.accelerator.is_available()
        else "cpu"
    )

    cfg = load_config("config/default.yaml")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(cfg.data.path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    train_dataset = TextDataset(
        X_train[cfg.data.target_column].tolist(), y_train.tolist(), tokenizer
    )
    val_dataset = TextDataset(
        X_val[cfg.data.target_column].tolist(), y_val.tolist(), tokenizer
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)

    model = MyModelV2(tokenizer)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

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

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{cfg.training.num_epochs} - Train Loss: {avg_train_loss:.4f}"
        )

        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm.tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs} [Validation]",
            ):
                input_ids = batch["input_ids"].to_device()
                attention_mask = batch["attention_mask"].to_device()
                labels = batch["labels"].to_device()

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1}/{cfg.training.num_epochs} - Validation Loss: {avg_val_loss:.4f}"
        )

        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        print(
            f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

    save_dir = "models/mymodelv2_classification"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))
    tokenizer.save_pretrained(save_dir)
    print(f"\n Обучение завершено! Модель сохранена в: {save_dir}")


if __name__ == "__main__":
    train_mymodelv2()
