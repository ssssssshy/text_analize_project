from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from src.dataset import load_data
from mlflow import log_metric, log_param, log_artifact
import nltk
from nltk.corpus import stopwords
from src.utils import load_config
import mlflow
import joblib


def train_tfidf():
    cfg = load_config("config/default.yaml")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        "data/processed/labeled.csv"
    )

    nltk.download("stopwords")
    russian_stopwords = stopwords.words("russian")

    vectorizer = TfidfVectorizer(
        max_features=cfg.tfidf.max_features, stop_words=russian_stopwords
    )

    X_train = vectorizer.fit_transform(X_train["comment"])
    X_val = vectorizer.transform(X_val["comment"])
    X_test = vectorizer.transform(X_test["comment"])

    mlflow.set_experiment("Tfidf_Logistic_Regression")
    with mlflow.start_run():
        log_param("max_features", cfg.tfidf.max_features)
        log_param("stop_words", cfg.tfidf.stop_words)
        log_param("C", cfg.logistic_regression.C)
        log_param("max_iter", cfg.logistic_regression.max_iter)
        log_param("random_state", cfg.logistic_regression.random_state)
        log_param("class_weight", cfg.logistic_regression.class_weight)

        model = LogisticRegression(
            C=cfg.logistic_regression.C,
            max_iter=cfg.logistic_regression.max_iter,
            random_state=cfg.logistic_regression.random_state,
            class_weight=cfg.logistic_regression.class_weight,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="weighted")
        print(classification_report(y_val, y_pred))
        print(f"F1 Score: {f1}")

        log_metric("f1_score", float(f1))

        model_path = "models/tfidf_logistic_model.joblib"
        vectorizer_path = "models/tfidf_vectorizer.joblib"

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        log_artifact(model_path, artifact_path="classifier")
        log_artifact(vectorizer_path, artifact_path="vectorizer")


if __name__ == "__main__":
    train_tfidf()
