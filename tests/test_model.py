import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

model = joblib.load("artifacts/model.joblib")

df = pd.read_csv("iris_eval.csv")

X = X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]

y = df["species"]

print(y.unique())

def test_accuracy():

    pred = model.predict(X)

    accuracy = accuracy_score(y, pred)

    assert accuracy >= 0.90

def test_precision():
    pred = model.predict(X)

    precision = precision_score(y, pred, average="macro")

    assert precision > 0.90