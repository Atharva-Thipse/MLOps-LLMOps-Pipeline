import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/iris.csv"

LOCAL_OUTPUT_DIR = "data/week10"

BUCKET = "gs://week-10-ga"
GCS_OUTPUT_DIR = f"{BUCKET}/fine-tuning/input"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# Helper functions
# ============================================================

def create_v1_input(row):
    """
    Raw numerical feature representation.
    """

    return (
        f"sepal_length: {row['sepal_length']}, "
        f"sepal_width: {row['sepal_width']}, "
        f"petal_length: {row['petal_length']}, "
        f"petal_width: {row['petal_width']}"
    )


def create_v2_input(row):
    """
    Natural language representation.
    """

    return (
        f"A flower specimen has a sepal length of "
        f"{row['sepal_length']} cm, sepal width of "
        f"{row['sepal_width']} cm, petal length of "
        f"{row['petal_length']} cm, and petal width of "
        f"{row['petal_width']} cm. Identify the iris species."
    )


def create_messages(user_input, output):
    """
    Create the Gemma supervised fine-tuning format.
    """

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify the flower into one of the following "
                    "species: [setosa, versicolor, virginica]"
                ),
            },
            {
                "role": "user",
                "content": user_input,
            },
            {
                "role": "assistant",
                "content": output,
            },
        ]
    }


def write_jsonl(records, output_file):
    """
    Write records as JSONL.
    """

    with open(output_file, "w", encoding="utf-8") as f:

        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("WEEK 10 - GEMMA DATASET PREPARATION")
    print("=" * 70)

    os.makedirs(
        LOCAL_OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load existing IRIS dataset
    # --------------------------------------------------------

    print(f"\nLoading dataset: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print(f"Total samples: {len(df)}")

    # --------------------------------------------------------
    # Create stratified train/test split
    # --------------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["species"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Training samples: {len(train_df)}")
    print(f"Evaluation samples: {len(test_df)}")

    # ========================================================
    # TRAINING DATA
    # ========================================================

    train_v1 = []
    train_v2 = []

    for _, row in train_df.iterrows():

        # -------------------------
        # v1 - Raw features
        # -------------------------

        v1_input = create_v1_input(row)

        train_v1.append(
            create_messages(
                user_input=v1_input,
                output=row["species"],
            )
        )

        # -------------------------
        # v2 - Natural language
        # -------------------------

        v2_input = create_v2_input(row)

        train_v2.append(
            create_messages(
                user_input=v2_input,
                output=f"This is Iris {row['species']}.",
            )
        )

    # ========================================================
    # EVALUATION DATA
    # ========================================================

    eval_v1 = []
    eval_v2 = []

    for _, row in test_df.iterrows():

        # -------------------------
        # v1 evaluation
        # -------------------------

        v1_input = create_v1_input(row)

        eval_v1.append(
            create_messages(
                user_input=v1_input,
                output=row["species"],
            )
        )

        # -------------------------
        # v2 evaluation
        # -------------------------

        v2_input = create_v2_input(row)

        eval_v2.append(
            create_messages(
                user_input=v2_input,
                output=f"This is Iris {row['species']}.",
            )
        )

    # ========================================================
    # Local output files
    # ========================================================

    train_v1_file = os.path.join(
        LOCAL_OUTPUT_DIR,
        "train_v1_gemma.jsonl"
    )

    train_v2_file = os.path.join(
        LOCAL_OUTPUT_DIR,
        "train_v2_gemma.jsonl"
    )

    eval_v1_file = os.path.join(
        LOCAL_OUTPUT_DIR,
        "eval_v1_gemma.jsonl"
    )

    eval_v2_file = os.path.join(
        LOCAL_OUTPUT_DIR,
        "eval_v2_gemma.jsonl"
    )

    # --------------------------------------------------------
    # Write files
    # --------------------------------------------------------

    write_jsonl(
        train_v1,
        train_v1_file
    )

    write_jsonl(
        train_v2,
        train_v2_file
    )

    write_jsonl(
        eval_v1,
        eval_v1_file
    )

    write_jsonl(
        eval_v2,
        eval_v2_file
    )

    print("\nLocal files created:")

    print(f"  {train_v1_file}")
    print(f"  {train_v2_file}")
    print(f"  {eval_v1_file}")
    print(f"  {eval_v2_file}")

    # ========================================================
    # Upload to GCS
    # ========================================================

    print("\nUploading files to GCS...")

    files = [
        train_v1_file,
        train_v2_file,
        eval_v1_file,
        eval_v2_file,
    ]

    for file in files:

        command = (
            f"gcloud storage cp "
            f'"{file}" '
            f'"{GCS_OUTPUT_DIR}/"'
        )

        print(f"\nUploading: {file}")

        exit_code = os.system(command)

        if exit_code != 0:
            raise RuntimeError(
                f"Failed to upload {file}"
            )

    # ========================================================
    # Verification
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 70)

    print("\nTraining datasets:")

    print(
        f"gs://week-10-ga/fine-tuning/input/"
        f"train_v1_gemma.jsonl"
    )

    print(
        f"gs://week-10-ga/fine-tuning/input/"
        f"train_v2_gemma.jsonl"
    )

    print("\nEvaluation datasets:")

    print(
        f"gs://week-10-ga/fine-tuning/input/"
        f"eval_v1_gemma.jsonl"
    )

    print(
        f"gs://week-10-ga/fine-tuning/input/"
        f"eval_v2_gemma.jsonl"
    )

    print("\nDataset sizes:")

    print(f"train_v1: {len(train_v1)}")
    print(f"train_v2: {len(train_v2)}")
    print(f"eval_v1:  {len(eval_v1)}")
    print(f"eval_v2:  {len(eval_v2)}")


if __name__ == "__main__":
    main()