import pandas as pd
import re
import os


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"<.*?>", " ", text)  # Удаление HTML
    text = re.sub(r"([.,!?\"'()«»“”-])", r" \1 ", text)  # Отделяем пунктуацию пробелами
    return re.sub(r"\s+", " ", text).strip()  # Удаляем лишние пробелы


# 1. Загружаем сырые данные (укажите ваш реальный путь к файлу, например train.csv)
# Предположим, что у вас один файл, который вы делите, или уже готовые сплиты
raw_data_path = "data/raw/labeled.csv"  # Поменяйте на ваш путь
df = pd.read_csv(raw_data_path)

# 2. Очищаем колонку с текстом
# Замените 'text_column_name' на имя вашей колонки (например, 'comment' или 'text')
text_col = "comment"
df[text_col] = df[text_col].astype(str).apply(clean_text)

# 3. Создаем папку для очищенных данных и сохраняем
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/cleaned_dataset.csv", index=False)

print("Данные успешно очищены и сохранены в data/processed/cleaned_dataset.csv")
