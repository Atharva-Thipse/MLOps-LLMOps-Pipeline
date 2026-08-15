import os
import sys

import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data"

EXPERIMENT_NAME = "iris-week8-data-poisoning"

POISONING_LEVELS = [0, 5, 10, 50]

NUMBER_OF_REPETITIONS = 10

RANDOM_SEEDS = 42

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "target"


# ============================================================
# Utility functions
# ============================================================

def load_training_data(poisoning_level):
    """
    Load one of the poisoned training datasets.
    """

    file_path = os.path.join(
        DATA_DIR,
        f"train_{poisoning_level}pct_poisoned.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Training dataset not found: {file_path}\n"
            "Run poison_data.py first."
        )

    return pd.read_csv(file_path)


def load_test_data():
    """
    Load the fixed clean test set.

    This test set is identical for every poisoning experiment.
    """

    file_path = os.path.join(
        DATA_DIR,
        "test_clean.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Test dataset not found: {file_path}\n"
            "Run poison_data.py first."
        )

    return pd.read_csv(file_path)


def create_model(seed):
    """
    Create the classification pipeline.

    StandardScaler:
        Normalizes the four Iris features.

    LogisticRegression:
        Provides a relatively sensitive linear baseline
        for studying the effect of poisoned training data.
    """

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed
                )
            )
        ]
    )

    return model


def calculate_metrics(y_true, y_pred):
    """
    Calculate all required classification metrics.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    }


# ============================================================
# MLflow setup
# ============================================================

def setup_mlflow():

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


# ============================================================
# Run one experiment
# ============================================================

def run_experiment(
    train_df,
    test_df,
    poisoning_level,
    seed
):

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ].astype(int)

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET_COLUMN
    ].astype(int)

    poisoned_samples = int(
        train_df["is_poisoned"].sum()
    )

    model = create_model(seed)

    # --------------------------------------------------------
    # MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=f"poison_{poisoning_level}pct_seed_{seed}"
    ):

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "poisoning_level",
            poisoning_level
        )

        mlflow.log_param(
            "seed",
            seed
        )

        mlflow.log_param(
            "training_samples",
            len(train_df)
        )

        mlflow.log_param(
            "test_samples",
            len(test_df)
        )

        mlflow.log_param(
            "poisoned_samples",
            poisoned_samples
        )

        mlflow.log_param(
            "model",
            "LogisticRegression"
        )

        mlflow.log_param(
            "scaler",
            "StandardScaler"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Evaluate
        #
        # IMPORTANT:
        # Evaluation happens ONLY against clean test data.
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        metrics = calculate_metrics(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"]
        )

        mlflow.log_metric(
            "precision",
            metrics["precision"]
        )

        mlflow.log_metric(
            "recall",
            metrics["recall"]
        )

        mlflow.log_metric(
            "f1",
            metrics["f1"]
        )

        # ----------------------------------------------------
        # Log model
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="iris_model"
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            f"\nPoisoning: {poisoning_level}%"
        )

        print(
            f"Seed:      {seed}"
        )

        print(
            f"Poisoned:  {poisoned_samples}"
        )

        print(
            f"Accuracy:  {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall:    {metrics['recall']:.4f}"
        )

        print(
            f"F1:        {metrics['f1']:.4f}"
        )

    return {
        "poisoning_level": poisoning_level,
        "seed": seed,
        "poisoned_samples": poisoned_samples,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"]
    }


# ============================================================
# Main experiment
# ============================================================

def main():

    print("=" * 70)
    print("WEEK 8 - MLSECOPS DATA POISONING EXPERIMENT")
    print("=" * 70)

    setup_mlflow()

    # --------------------------------------------------------
    # Load fixed clean test set
    # --------------------------------------------------------

    test_df = load_test_data()

    print(
        f"\nFixed clean test samples: {len(test_df)}"
    )

    results = []

    # --------------------------------------------------------
    # Run one experiment for each poisoning level
    # --------------------------------------------------------

    for poisoning_level in POISONING_LEVELS:

        train_df = load_training_data(
            poisoning_level
        )

        print("\n" + "-" * 70)

        print(
            f"Running {poisoning_level}% poisoning experiment"
        )

        print(
            f"Training samples: {len(train_df)}"
        )

        print(
            f"Poisoned samples: "
            f"{int(train_df['is_poisoned'].sum())}"
        )

        # ----------------------------------------------------
        # Run experiment with fixed random seed
        # ----------------------------------------------------

        result = run_experiment(
            train_df=train_df,
            test_df=test_df,
            poisoning_level=poisoning_level,
            seed=42
        )

        results.append(result)

    # --------------------------------------------------------
    # Convert results to DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    detailed_results_file = os.path.join(
        DATA_DIR,
        "detailed_results.csv"
    )

    results_df.to_csv(
        detailed_results_file,
        index=False
    )

    # --------------------------------------------------------
    # Create summary
    # Since there is only one run per poisoning level,
    # the mean is simply the metric from that run.
    # --------------------------------------------------------

    summary_df = (
        results_df
        .groupby("poisoning_level")
        .agg(
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_file = os.path.join(
        DATA_DIR,
        "poisoning_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Display final results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nDetailed results:")
    print(detailed_results_file)

    print("\nSummary results:")
    print(summary_file)

    print("\nMLflow experiment:")
    print(EXPERIMENT_NAME)

    print("\n" + "=" * 70)
    print("Experiment completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()