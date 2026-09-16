import os

import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import ks_2samp


DATA_DIR = "data"
OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "drift_results.csv",
)

PLOT_FILE = os.path.join(
    DATA_DIR,
    "drift_comparison.png",
)

RANDOM_STATE = 42
DRIFT_OFFSET = 1.0
SIGNIFICANCE_LEVEL = 0.05


FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def load_training_data():
    iris = pd.read_csv("data/iris.csv")

    return iris


def create_production_data(training_df):
    production_df = training_df.copy()

    # Simulate production drift.
    production_df["petal_length"] += DRIFT_OFFSET

    return production_df


def detect_drift(training_df, production_df):
    results = []

    for feature in FEATURES:

        statistic, p_value = ks_2samp(
            training_df[feature],
            production_df[feature],
        )

        drift_detected = (
            p_value < SIGNIFICANCE_LEVEL
        )

        results.append(
            {
                "feature": feature,
                "ks_statistic": statistic,
                "p_value": p_value,
                "drift_detected": drift_detected,
            }
        )

    return pd.DataFrame(results)


def create_plot(training_df, production_df):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    axes = axes.flatten()

    for index, feature in enumerate(FEATURES):

        axes[index].hist(
            training_df[feature],
            bins=15,
            alpha=0.6,
            label="Training",
        )

        axes[index].hist(
            production_df[feature],
            bins=15,
            alpha=0.6,
            label="Production",
        )

        axes[index].set_title(
            feature
        )

        axes[index].legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=200,
    )

    plt.close()


def main():

    print("=" * 70)
    print("WEEK 9 - DATA DRIFT DETECTION")
    print("=" * 70)

    training_df = load_training_data()

    production_df = create_production_data(
        training_df
    )

    print(
        f"\nProduction simulation: "
        f"petal_length shifted by +{DRIFT_OFFSET}"
    )

    results_df = detect_drift(
        training_df,
        production_df,
    )

    print("\nDrift results:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    create_plot(
        training_df,
        production_df,
    )

    print(
        f"\nResults saved to: {OUTPUT_FILE}"
    )

    print(
        f"Distribution plot saved to: {PLOT_FILE}"
    )

    print("\n" + "=" * 70)
    print("Drift detection completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()