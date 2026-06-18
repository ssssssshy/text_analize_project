import os
import warnings
import sklearn.exceptions
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.model.mymodelv2 import MyModelV2
from src.utils import load_config

warnings.filterwarnings(
    "ignore",
    category=sklearn.exceptions.InconsistentVersionWarning,  # type: ignore
)

cfg = load_config("config/default.yaml")


def predict_mymodelv2(text, model_path):
    device = (
        torch.accelerator.current_accelerator.type()
        if torch.accelerator.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = MyModelV2(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=cfg.mymodelv2_params.embedding_dim,
        num_heads=cfg.mymodelv2_params.num_heads,
        num_layers=cfg.mymodelv2_params.num_layers,
        num_classes=cfg.mymodelv2_params.num_classes,
    )

    weight_path = os.path.join(model_path, "model_weights.pth")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        print(f"Сырые логиты: {logits}")
        probs = torch.softmax(logits, dim=1)
        print(f"Вероятности: {probs}")
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0][pred_class].item()

    return pred_class, confidence


def predict_bert(text, model_path):
    device = (
        torch.accelerator.current_accelerator.type()
        if torch.accelerator.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        print(f"Сырые логиты: {logits}")
        probs = torch.softmax(logits, dim=1)
        print(f"Вероятности: {probs}")
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0][pred_class].item()

        return pred_class, confidence


def main():
    text = "Математика — это сложная наука"

    print("Тестируем MyModelV2...")
    print(f"Текст: {text}")
    model_v2_path = "./models/custom_modelv2"
    res_class, confidence = predict_mymodelv2(text, model_v2_path)
    print(f"Результат V2: Класс {res_class}, Уверенность: {confidence:.4f}\n")

    print("Тестируем Отфатюниный BERT...")
    print(f"Текст: {text}")
    model_bert_path = "./models/bert_sequence_classification"
    res_class, confidence = predict_bert(text, model_bert_path)
    print(f"Результат V2: Класс {res_class}, Уверенность: {confidence:.4f}\n")


if __name__ == "__main__":
    main()
