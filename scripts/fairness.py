import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from fairlearn.metrics import MetricFrame


DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "fairness_results.csv")

RANDOM_STATE = 42


def load_data():
    iris = pd.read_csv("data/iris.csv")

    df = pd.DataFrame(
        iris,
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    )

    df["target"] = iris['species']

    # Location is a sensitive attribute only.
    # It is intentionally NOT used as a model feature.
    rng = pd.Series(range(len(df))).sample(
        frac=1,
        random_state=RANDOM_STATE,
    )

    location_values = [0, 1] * (len(df) // 2 + 1)
    location_values = location_values[: len(df)]

    location_series = pd.Series(
        location_values,
        index=rng.index,
    )

    df["location"] = location_series.sort_index().values

    return df


def main():
    print("=" * 70)
    print("WEEK 9 - FAIRNESS ANALYSIS")
    print("=" * 70)

    df = load_data()

    features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    X = df[features]
    y = df["target"]
    sensitive_feature = df["location"]

    X_train, X_test, y_train, y_test, location_train, location_test = (
        train_test_split(
            X,
            y,
            sensitive_feature,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=4,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Features used:    {features}")
    print("Sensitive attribute: location")
    print("Location used for training: NO")

    overall_accuracy = accuracy_score(y_test, predictions)

    overall_precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    overall_recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\nOverall metrics:")
    print(f"Accuracy : {overall_accuracy:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall   : {overall_recall:.4f}")

    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall": lambda y_true, y_pred: recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=predictions,
        sensitive_features=location_test,
    )

    print("\nMetrics by location:")
    print(metric_frame.by_group)

    print("\nMetric differences between groups:")
    print(metric_frame.difference())

    results = metric_frame.by_group.reset_index()

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nResults saved to: {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("Fairness analysis completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()