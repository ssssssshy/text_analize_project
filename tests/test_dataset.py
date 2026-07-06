from src.dataset import load_data


def test_load_data():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        "data/processed/cleaned_dataset.csvv"
    )
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_val) > 0
    assert len(y_test) > 0
