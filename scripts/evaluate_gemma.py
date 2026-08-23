import json
import os
import re
import shutil
import subprocess

import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# Configuration
# ============================================================

BUCKET = "gs://week-10-ga"

V1_MODEL_GCS = (
    "gs://week-10-ga/fine-tuning/output/v1/"
    "gemma-3-1b-it-1787329747573-20260821100830/"
    "merged_model"
)

V2_MODEL_GCS = (
    "gs://week-10-ga/fine-tuning/output/v2/"
    "gemma-3-1b-it-1787330340855-20260821100949/"
    "merged_model"
)

V1_EVAL_GCS = (
    "gs://week-10-ga/fine-tuning/input/eval_v1_gemma.jsonl"
)

V2_EVAL_GCS = (
    "gs://week-10-ga/fine-tuning/input/eval_v2_gemma.jsonl"
)

LOCAL_V1_MODEL = "data/gemma_v1_model"
LOCAL_V2_MODEL = "data/gemma_v2_model"

LOCAL_EVAL_DIR = "data/week10_eval"

OUTPUT_DIR = "data/week10_results"

V1_PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "v1_predictions.csv",
)

V2_PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "v2_predictions.csv",
)

COMPARISON_FILE = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv",
)

CLASSIFICATION_REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "classification_report.csv",
)

SPECIES = [
    "setosa",
    "versicolor",
    "virginica",
]

RANDOM_STATE = 42


# ============================================================
# Utility functions
# ============================================================

def download_gcs_directory(gcs_path, local_path):
    """Download a complete GCS directory."""

    if os.path.exists(local_path):
        print(f"Removing existing directory: {local_path}")
        shutil.rmtree(local_path)

    os.makedirs(local_path, exist_ok=True)

    print(f"\nDownloading:")
    print(gcs_path)
    print(f"to:")
    print(local_path)

    subprocess.run(
        [
            "gcloud",
            "storage",
            "cp",
            "-r",
            gcs_path,
            local_path,
        ],
        check=True,
    )


def download_gcs_file(gcs_path, local_file):
    """Download one file from GCS."""

    os.makedirs(
        os.path.dirname(local_file),
        exist_ok=True,
    )

    print(f"\nDownloading:")
    print(gcs_path)

    subprocess.run(
        [
            "gcloud",
            "storage",
            "cp",
            gcs_path,
            local_file,
        ],
        check=True,
    )


# ============================================================
# Dataset loading
# ============================================================

def load_jsonl(path):
    """Load JSONL records."""

    records = []

    with open(path,"r",encoding="utf-8") as file:
        for line_number, line in enumerate(file,start=1):
            line = line.strip()

            if not line:
                continue
                
            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {error}"
                )

    return records


# ============================================================
# Gemma prompt
# ============================================================

def build_prompt(input_text):
    """Build the Gemma 3 instruction prompt."""

    prompt = (
        "<start_of_turn>system\n"
        "Classify the flower into one of the following species: "
        "[setosa, versicolor, virginica]\n"
        "<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{input_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    return prompt

# ============================================================
# Model loading
# ============================================================

def load_model(model_directory):
    """Load tokenizer and Gemma model."""

    # gcloud storage cp -r creates:
    #
    # model_directory/
    #     merged_model/
    #
    # Therefore locate the actual model directory.

    actual_model_directory = os.path.join(
        model_directory,
        "merged_model",
    )

    if not os.path.exists(actual_model_directory):
        actual_model_directory = model_directory

    print("\nLoading model:")
    print(actual_model_directory)

    tokenizer = AutoTokenizer.from_pretrained(actual_model_directory,use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        actual_model_directory,
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    print(f"Device: {device}")

    return tokenizer, model, device

# ============================================================
# Prediction
# ============================================================

def predict(tokenizer,model,device,input_text):
    """Generate a prediction."""

    prompt = build_prompt(input_text)
    inputs = tokenizer(prompt,return_tensors="pt")

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text.strip()


# ============================================================
# Classification extraction
# ============================================================

def extract_species(text):
    """
    Extract a species from model output.

    This is used ONLY for calculating classification
    accuracy/precision/recall.

    It is deliberately separate from format compliance.
    """

    if not text:
        return None

    text_lower = text.strip().lower()

    for species in SPECIES:
        if species in text_lower:
            return species

    return None

# ============================================================
# Format compliance
# ============================================================

def check_format_compliance(prediction,expected_species,version):
    """
    Check whether the model followed the required
    output format exactly.

    V1:
        setosa

    V2:
        This is Iris setosa.
    """

    prediction = prediction.strip()

    if version == "v1":
        expected = expected_species
        return prediction == expected

    if version == "v2":
        expected = (
            f"This is Iris "
            f"{expected_species}."
        )
        return prediction == expected

    raise ValueError(f"Unknown model version: {version}")


# ============================================================
# Evaluate one model
# ============================================================

def evaluate_model(version,tokenizer,model,device,evaluation_records):
    """Run the complete evaluation."""

    predictions = []

    y_true = []
    y_pred = []

    compliant_count = 0

    print("\n" + "-" * 70)

    print(
        f"Evaluating Gemma {version.upper()}"
    )

    print("-" * 70)

    for index, record in enumerate(evaluation_records,start=1):
        if version == "v1":
            input_text = record["messages"][1]["content"]
            expected = record["messages"][2]["content"].strip()

        else:
            input_text = record["messages"][1]["content"]
            expected = record["messages"][2]["content"].strip()

        prediction = predict(tokenizer=tokenizer,model=model,device=device,input_text=input_text)
        predicted_species = (extract_species(prediction))

        compliant = check_format_compliance(
            prediction=prediction,
            expected_species=expected
            if version == "v1"
            else extract_species(expected),
            version=version,
        )

        if compliant:
            compliant_count += 1

        y_true.append(
            expected
            if version == "v1"
            else extract_species(expected)
        )

        y_pred.append(predicted_species)

        predictions.append(
            {
                "sample_id": index,
                "input_text": input_text,
                "expected_output": expected,
                "raw_prediction": prediction,
                "predicted_species": (
                    predicted_species
                    if predicted_species
                    else ""
                ),
                "format_compliant": compliant,
            }
        )

        print(
            f"{index:3d}. "
            f"Expected: {expected:25s} "
            f"Predicted: {prediction}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_true,y_pred)

    precision, recall, _, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=SPECIES,
            average=None,
            zero_division=0,
        )
    )

    compliance_rate = (compliant_count/len(evaluation_records))

    metrics = {
        "model": version,
        "samples": len(evaluation_records),
        "accuracy": accuracy,
        "format_compliance_rate": (
            compliance_rate
        ),
    }

    for index, species in enumerate(SPECIES):
        metrics[f"{species}_precision"] = precision[index]
        metrics[f"{species}_recall"] = recall[index]

    return predictions, metrics


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("WEEK 10 - GEMMA FINE-TUNED MODEL EVALUATION")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR,exist_ok=True)
    os.makedirs(LOCAL_EVAL_DIR,exist_ok=True)

    # --------------------------------------------------------
    # Download models
    # --------------------------------------------------------

    print("\nSTEP 1 - Downloading fine-tuned models")

    download_gcs_directory(
        V1_MODEL_GCS,
        LOCAL_V1_MODEL,
    )

    download_gcs_directory(
        V2_MODEL_GCS,
        LOCAL_V2_MODEL,
    )

    # --------------------------------------------------------
    # Download evaluation datasets
    # --------------------------------------------------------

    print("\nSTEP 2 - Downloading held-out evaluation datasets")

    local_v1_eval = os.path.join(
        LOCAL_EVAL_DIR,
        "eval_v1_gemma.jsonl",
    )

    local_v2_eval = os.path.join(
        LOCAL_EVAL_DIR,
        "eval_v2_gemma.jsonl",
    )

    download_gcs_file(
        V1_EVAL_GCS,
        local_v1_eval,
    )

    download_gcs_file(
        V2_EVAL_GCS,
        local_v2_eval,
    )

    # --------------------------------------------------------
    # Load evaluation data
    # --------------------------------------------------------

    v1_records = load_jsonl(local_v1_eval)

    v2_records = load_jsonl(local_v2_eval)

    print(
        f"\nV1 evaluation samples: "
        f"{len(v1_records)}"
    )

    print(
        f"V2 evaluation samples: "
        f"{len(v2_records)}"
    )

    if len(v1_records) != len(v2_records):
        raise ValueError(
            "V1 and V2 evaluation datasets "
            "must contain the same number of samples."
        )

    # --------------------------------------------------------
    # Load V1 model
    # --------------------------------------------------------

    print(
        "\nSTEP 3 - Loading V1 model"
    )

    tokenizer_v1, model_v1, device_v1 = (
        load_model(LOCAL_V1_MODEL)
    )

    # --------------------------------------------------------
    # Evaluate V1
    # --------------------------------------------------------

    v1_predictions, v1_metrics = (
        evaluate_model(
            version="v1",
            tokenizer=tokenizer_v1,
            model=model_v1,
            device=device_v1,
            evaluation_records=v1_records,
        )
    )

    pd.DataFrame(
        v1_predictions
    ).to_csv(
        V1_PREDICTIONS_FILE,
        index=False,
    )

    # Free memory
    del model_v1
    del tokenizer_v1

    import gc
    gc.collect()

    # --------------------------------------------------------
    # Load V2 model
    # --------------------------------------------------------

    print("\nSTEP 4 - Loading V2 model")

    tokenizer_v2, model_v2, device_v2 = (load_model(LOCAL_V2_MODEL))

    # --------------------------------------------------------
    # Evaluate V2
    # --------------------------------------------------------

    v2_predictions, v2_metrics = (
        evaluate_model(
            version="v2",
            tokenizer=tokenizer_v2,
            model=model_v2,
            device=device_v2,
            evaluation_records=v2_records,
        )
    )

    pd.DataFrame(
        v2_predictions
    ).to_csv(
        V2_PREDICTIONS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        [
            v1_metrics,
            v2_metrics,
        ]
    )

    comparison_df.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_rows = []

    for version, metrics in [
        ("v1", v1_metrics),
        ("v2", v2_metrics),
    ]:

        for species in SPECIES:

            report_rows.append(
                {
                    "model": version,
                    "species": species,
                    "precision": metrics[
                        f"{species}_precision"
                    ],
                    "recall": metrics[
                        f"{species}_recall"
                    ],
                }
            )

    report_df = pd.DataFrame(
        report_rows
    )

    report_df.to_csv(
        CLASSIFICATION_REPORT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    display_columns = [
        "model",
        "samples",
        "accuracy",
        "format_compliance_rate",
    ]

    print(
        comparison_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nPer-class metrics:")

    print(
        report_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nOutput files:")

    print(
        f"V1 predictions: "
        f"{V1_PREDICTIONS_FILE}"
    )

    print(
        f"V2 predictions: "
        f"{V2_PREDICTIONS_FILE}"
    )

    print(
        f"Comparison: "
        f"{COMPARISON_FILE}"
    )

    print(
        f"Classification report: "
        f"{CLASSIFICATION_REPORT_FILE}"
    )

    print("\n" + "=" * 70)
    print("Evaluation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()