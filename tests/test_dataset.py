import pandas as pd

from src.dataset import load_data


def test_load_data(tmp_path):
    # build a tiny dataset on the fly so the test does not depend on
    # data files that are git-ignored / absent in CI
    csv_path = tmp_path / "labeled.csv"
    df = pd.DataFrame(
        {
            "comment": [f"пример текста номер {i}" for i in range(20)],
            "toxic": [i % 2 for i in range(20)],
        }
    )
    df.to_csv(csv_path, index=False)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(str(csv_path))

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_val) > 0
    assert len(y_test) > 0

    # no rows are lost or duplicated across the three splits
    assert len(X_train) + len(X_val) + len(X_test) == len(df)
