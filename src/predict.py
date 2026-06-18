import argparse
import os
import sys
import warnings

import joblib
import sklearn.exceptions
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model.mymodel import SimpleTransformerClassifier
from src.model.mymodelv2 import MyModelV2

warnings.filterwarnings(
    "ignore",
    category=sklearn.exceptions.InconsistentVersionWarning,  # type: ignore
)


def predict_mymodelv2(text, model_path):

    device = (
        torch.accelerator.current_accelerator.type()
        if torch.accelerator.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = MyModelV2(tokenizer=tokenizer)

    weight_path = os.path.join(model_path, "models/custom_modelv2/model_weight.pth")
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

        probs = torch.softmax(logits, dim=1)

        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0][pred_class].item()
    return pred_class, confidence


def predict_mymodel(text, model_path):
    device = (
        torch.accelerator.current_accelerator.type()
        if torch.accelerator.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = SimpleTransformerClassifier(tokenizer)
    weigth_path = os.path.join(
        model_path, "models/custom_transformers_classification/model_weight.pth"
    )
    model.load_state_dict(torch.load(weigth_path, map_location=device))
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0][pred_class].item()

    return pred_class, confidence


def main():
    model_path = "./models"
    text = "Очень плохой фильм"

    print("Тестируем MyModelV2")
    res_class, confidence = predict_mymodelv2(text, model_path)
    print(f"Результат V2: Класс {res_class}, Уверенность: {confidence:.4f}")

    print("Тестируем MyModel")
    res_class, confidence = predict_mymodel(text, model_path)
    print(f"Результат V1: Класс {res_class}, Уверенность: {confidence:.4f}")


if __name__ == "__main__":
    main()
