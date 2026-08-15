import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Configuration
# ============================================================

DATA_FILE = "data/iris.csv"

OUTPUT_DIR = "data/shap"

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


# ============================================================
# Load dataset
# ============================================================

def load_data():

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURE_COLUMNS].copy()

    y = df["species"].copy()

    return X, y


# ============================================================
# Convert SHAP output into:
#
#   samples x features x classes
#
# This handles different SHAP versions.
# ============================================================

def normalize_shap_values(shap_values, X, number_of_classes):

    # Newer SHAP versions may return an Explanation object
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    # Older SHAP versions return:
    #
    # [
    #   class_0_array,
    #   class_1_array,
    #   class_2_array
    # ]
    #
    if isinstance(shap_values, list):

        arrays = [
            np.asarray(values)
            for values in shap_values
        ]

        return np.stack(arrays, axis=2)

    values = np.asarray(shap_values)

    # Shape:
    #
    # samples x features x classes
    #
    if values.ndim == 3:

        # Most common newer SHAP format
        if values.shape[0] == len(X):

            return values

        # Possible alternative:
        # classes x samples x features
        if values.shape[1] == len(X):

            return np.transpose(
                values,
                (1, 2, 0)
            )

    raise ValueError(
        f"Unexpected SHAP shape: {values.shape}"
    )


# ============================================================
# Create signed SHAP bar plot
# ============================================================

def create_signed_bar_plot(
    shap_values,
    X,
    class_name,
    output_file,
):

    # --------------------------------------------------------
    # Calculate average SHAP contribution for each feature
    # across the COMPLETE dataset.
    #
    # Positive:
    #   pushes prediction toward this class
    #
    # Negative:
    #   pushes prediction away from this class
    # --------------------------------------------------------

    mean_shap = np.mean(
        shap_values,
        axis=0
    )

    # --------------------------------------------------------
    # Sort features by absolute impact
    # --------------------------------------------------------

    order = np.argsort(
        np.abs(mean_shap)
    )

    sorted_features = [
        FEATURE_COLUMNS[i]
        for i in order
    ]

    sorted_values = mean_shap[order]

    # --------------------------------------------------------
    # FIXED figure size
    #
    # This is important.
    #
    # Do NOT calculate figsize from feature names or SHAP
    # values. That caused the enormous image error.
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    y_positions = np.arange(
        len(sorted_features)
    )

    ax.barh(
        y_positions,
        sorted_values,
        height=0.6,
    )

    # --------------------------------------------------------
    # Zero line
    # --------------------------------------------------------

    ax.axvline(
        0,
        linewidth=1,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        sorted_features
    )

    ax.set_xlabel(
        "Mean SHAP value"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        f"SHAP Feature Impact - {class_name}"
    )

    # --------------------------------------------------------
    # Add explanation directly to graph
    # --------------------------------------------------------

    ax.text(
        0.01,
        0.02,
        "← Away from class",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )

    ax.text(
        0.99,
        0.02,
        "Toward class →",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Prevent layout problems
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.25,
        right=0.95,
        top=0.88,
        bottom=0.18,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Return values for terminal output
    # --------------------------------------------------------

    return pd.DataFrame(
        {
            "feature": sorted_features,
            "mean_shap": sorted_values,
            "direction": [
                "toward"
                if value > 0
                else "away"
                if value < 0
                else "neutral"
                for value in sorted_values
            ],
        }
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("WEEK 9 - SHAP EXPLAINABILITY")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load YOUR dataset
    # --------------------------------------------------------

    X, y = load_data()

    print(
        f"\nDataset samples: {len(X)}"
    )

    print(
        f"Features: {list(X.columns)}"
    )

    print(
        f"Classes: {sorted(y.unique())}"
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=4,
    )

    model.fit(
        X,
        y,
    )

    print(
        "\nDecision Tree trained successfully."
    )

    print(
        f"Model classes: {list(model.classes_)}"
    )

    # --------------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------------

    print(
        "\nCreating SHAP explainer..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_output = explainer(
        X
    )

    shap_values = normalize_shap_values(
        shap_output,
        X,
        len(model.classes_),
    )

    print(
        f"SHAP values shape: {shap_values.shape}"
    )

    # --------------------------------------------------------
    # Generate plots for every class
    # --------------------------------------------------------

    all_results = []

    for class_index, class_name in enumerate(
        model.classes_
    ):

        print(
            f"\nGenerating SHAP plot for: "
            f"{class_name}"
        )

        class_values = shap_values[
            :,
            :,
            class_index
        ]

        output_file = os.path.join(
            OUTPUT_DIR,
            f"shap_signed_{class_name}.png",
        )

        results = create_signed_bar_plot(
            shap_values=class_values,
            X=X,
            class_name=class_name,
            output_file=output_file,
        )

        results["class"] = class_name

        all_results.append(
            results
        )

        print(
            f"Saved: {output_file}"
        )

        print(
            results.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save numerical SHAP results
    # --------------------------------------------------------

    shap_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    results_file = os.path.join(
        OUTPUT_DIR,
        "shap_feature_impacts.csv",
    )

    shap_results.to_csv(
        results_file,
        index=False,
    )

    print(
        f"\nSHAP results saved to:"
        f"\n{results_file}"
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SHAP analysis completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()