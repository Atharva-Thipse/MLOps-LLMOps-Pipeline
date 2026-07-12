import pandas as pd

df = pd.read_csv("iris_train.csv")


def test_no_missing_values():
    assert df.isnull().sum().sum() == 0


def test_correct_columns():
    expected = {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    }
    
    assert expected.issubset(set(df.columns))


def test_feature_ranges():
    assert df["sepal_length"].between(4, 8).all()
    assert df["sepal_width"].between(2, 5).all()
    assert df["petal_length"].between(1, 7).all()
    assert df["petal_width"].between(0, 3).all()