import os
import sys
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

DATA_DIR = "data"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "target"

POISONING_LEVELS = [0, 5, 10, 50]


# ============================================================
# Utility functions
# ============================================================

def create_directories():
    """Create the output directory if it does not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def load_clean_iris():
    """
    Load the Iris dataset and return it as a DataFrame.
    """

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=FEATURE_COLUMNS
    )

    df[TARGET_COLUMN] = iris.target

    return df


def poison_dataset(
    train_df,
    poisoning_percentage,
    random_seed
):
    """
    Poison a percentage of the training samples.

    For every poisoned sample:
      1. Replace all four features with random values.
      2. Keep generated values within the original feature ranges.
      3. Replace the original class with a random incorrect class.

    The test set is NEVER poisoned.
    """

    rng = np.random.default_rng(random_seed)

    poisoned_df = train_df.copy().reset_index(drop=True)

    if poisoning_percentage == 0:
        poisoned_df["is_poisoned"] = 0
        return poisoned_df

    number_of_samples = len(poisoned_df)

    number_to_poison = int(
        round(
            number_of_samples
            * poisoning_percentage
            / 100
        )
    )

    # Select samples to poison.
    poison_indices = rng.choice(
        number_of_samples,
        size=number_to_poison,
        replace=False
    )

    # Obtain feature ranges from the CLEAN training data.
    feature_min = train_df[FEATURE_COLUMNS].min().values
    feature_max = train_df[FEATURE_COLUMNS].max().values

    # Generate random feature values within realistic Iris ranges.
    random_features = rng.uniform(
        low=feature_min,
        high=feature_max,
        size=(number_to_poison, len(FEATURE_COLUMNS))
    )

    poisoned_df.loc[
        poison_indices,
        FEATURE_COLUMNS
    ] = random_features

    # Generate random INCORRECT labels.
    number_of_classes = train_df[TARGET_COLUMN].nunique()

    original_labels = poisoned_df.loc[
        poison_indices,
        TARGET_COLUMN
    ].astype(int).values

    poisoned_labels = []

    for original_label in original_labels:

        possible_labels = [
            label
            for label in range(number_of_classes)
            if label != original_label
        ]

        new_label = rng.choice(possible_labels)

        poisoned_labels.append(new_label)

    poisoned_df.loc[
        poison_indices,
        TARGET_COLUMN
    ] = poisoned_labels

    # Mark poisoned rows so we can verify the experiment.
    poisoned_df["is_poisoned"] = 0

    poisoned_df.loc[
        poison_indices,
        "is_poisoned"
    ] = 1

    return poisoned_df


def save_dataset(df, poisoning_percentage):
    """
    Save a poisoned training dataset.
    """

    output_file = os.path.join(
        DATA_DIR,
        f"train_{poisoning_percentage}pct_poisoned.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved {poisoning_percentage}% dataset -> "
        f"{output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    create_directories()

    print("=" * 60)
    print("WEEK 8 - IRIS DATA POISONING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load clean data
    # --------------------------------------------------------

    df = load_clean_iris()

    print(f"\nTotal samples: {len(df)}")

    # --------------------------------------------------------
    # Fixed train/test split
    #
    # IMPORTANT:
    # The SAME clean test set is used for every experiment.
    # --------------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COLUMN]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Test set is always clean.
    test_df["is_poisoned"] = 0

    # --------------------------------------------------------
    # Save clean test set
    # --------------------------------------------------------

    test_file = os.path.join(
        DATA_DIR,
        "test_clean.csv"
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples:     {len(test_df)}")
    print(f"Clean test set:   {test_file}")

    # --------------------------------------------------------
    # Create poisoned training datasets
    # --------------------------------------------------------

    for level in POISONING_LEVELS:

        poisoned_train = poison_dataset(
            train_df=train_df,
            poisoning_percentage=level,
            random_seed=RANDOM_SEED + level
        )

        save_dataset(
            poisoned_train,
            level
        )

        poisoned_count = int(
            poisoned_train["is_poisoned"].sum()
        )

        print(
            f"{level:>2}% poisoning -> "
            f"{poisoned_count} poisoned samples"
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = pd.DataFrame({
        "total_samples": [len(df)],
        "training_samples": [len(train_df)],
        "test_samples": [len(test_df)],
        "random_seed": [RANDOM_SEED]
    })

    metadata.to_csv(
        os.path.join(
            DATA_DIR,
            "metadata.csv"
        ),
        index=False
    )

    print("\n" + "=" * 60)
    print("Poisoning experiment data created successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()