import argparse
import os
import sys
import warnings

import joblib
import sklearn.exceptions
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model.mymodel import SimpleTransformerClassifier

warnings.filterwarnings(
    "ignore",
    category=sklearn.exceptions.InconsistentVersionWarning,  # type: ignore
)


def predict_bert(text, model_path):
    device = (
        torch.accelerator.current_accelerator().type  # type: ignore
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
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0][pred_class].item()

    return pred_class, confidence


def predict_custom_transformer(text, model_path):
    device = (
        torch.accelerator.current_accelerator().type  # type: ignore
        if torch.accelerator.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = SimpleTransformerClassifier(vocab_size=tokenizer.vocab_size)

    weights_path = os.path.join(model_path, "model_weights.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0][pred_class].item()

    return pred_class, confidence


def predict_tfidf(text, model_path):
    vectorizer_path = os.path.join(model_path, "tfidf_vectorizer.joblib")
    model_path_joblib = os.path.join(model_path, "tfidf_logistic_model.joblib")

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path_joblib)

    text_vectorized = vectorizer.transform([text])
    pred_class = int(model.predict(text_vectorized)[0])

    probs = model.predict_proba(text_vectorized)[0]
    confidence = float(probs[pred_class])

    return pred_class, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for Text Analysis Project"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["bert", "custom", "tfidf"],
        help="Type of model to use for prediction (default: bert)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Немного глупый текст для теста модели",
        help="Text to analyze",
    )
    args = parser.parse_args()

    paths = {
        "bert": "models/bert_sequence_classification",
        "custom": "models/custom_transformer_classification",
        "tfidf": "models/TF IDF",
    }

    model_path = paths[args.model]

    if not os.path.exists(model_path):
        print(f"Ошибка: Директория {model_path} не найдена. Проверь 'git lfs pull'.")
        sys.exit(1)

    print(f"Используемая модель: {args.model.upper()}")
    print(f"Текст для анализа: '{args.text}'\n---")

    if args.model == "bert":
        label, conf = predict_bert(args.text, model_path)
    elif args.model == "custom":
        label, conf = predict_custom_transformer(args.text, model_path)
    elif args.model == "tfidf":
        label, conf = predict_tfidf(args.text, model_path)
    else:
        raise ValueError("Unknown model type")

    print("Результат проверки:")
    print(f"Предсказанный класс: {label} (1 — негативный, 0 — позитивный)")
    print(f"Уверенность модели: {conf:.4f}")


if __name__ == "__main__":
    main()
