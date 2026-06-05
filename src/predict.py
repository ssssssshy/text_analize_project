import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict(text, model_path="models/bert_sequence_classification"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Подготовка текста
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Предсказание
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0][pred_class].item()

    return pred_class, confidence


if __name__ == "__main__":
    import sys

    # Пример использования
    sample_text = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Немного глупый текст для теста модели"
    )
    print(f"Текст: {sample_text}")

    label, conf = predict(sample_text)
    print(f"Предсказанный класс: {label}, 1-негативный, 0-позитивный")
    print(f"Уверенность: {conf:.4f}")
