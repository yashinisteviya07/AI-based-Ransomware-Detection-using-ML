"""Reusable prediction interface for the final ransomware detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_ransomware_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"


def _load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def _validate_input_frame(input_frame: pd.DataFrame, required_features: list[str]) -> None:
    if input_frame.empty:
        raise ValueError("The CSV contains no data rows.")
    missing_features = [feature for feature in required_features if feature not in input_frame.columns]
    unexpected_features = [feature for feature in input_frame.columns if feature not in required_features]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    if unexpected_features:
        raise ValueError(f"Unexpected columns found: {unexpected_features}")
    non_numeric = input_frame[required_features].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"These features must be numeric: {non_numeric}")
    if input_frame[required_features].isna().any().any():
        raise ValueError("The input contains missing values (NaN).")
    if not np.isfinite(input_frame[required_features].to_numpy(dtype=float)).all():
        raise ValueError("The input contains infinite values.")


def predict_sample(sample: Mapping[str, object]) -> dict[str, object]:
    """Predict one sample containing the required 52 feature columns."""
    model, metadata = _load_model_and_metadata()
    required_features = metadata["feature_names"]
    missing_features = [feature for feature in required_features if feature not in sample]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    unexpected_features = [feature for feature in sample if feature not in required_features]
    if unexpected_features:
        raise ValueError(f"Unexpected columns found: {unexpected_features}")

    sample_frame = pd.DataFrame([{feature: sample[feature] for feature in required_features}])
    _validate_input_frame(sample_frame, required_features)
    prediction = model.predict(sample_frame)[0]
    result: dict[str, object] = {
        "prediction": str(prediction),
        "model_name": metadata["model_name"],
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample_frame)[0]
        class_probabilities = dict(zip(model.classes_, probabilities))
        result["probability"] = float(class_probabilities[prediction])
        result["class_probabilities"] = {
            str(label): float(value) for label, value in class_probabilities.items()
        }
    return result


def predict_csv(csv_path: str | Path) -> pd.DataFrame:
    """Predict every row in a CSV containing the required feature columns."""
    input_frame = pd.read_csv(csv_path)
    return predict_dataframe(input_frame)


def predict_dataframe(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Predict every row in a DataFrame containing the required feature columns."""
    model, metadata = _load_model_and_metadata()
    required_features = metadata["feature_names"]
    _validate_input_frame(input_frame, required_features)

    feature_frame = input_frame[required_features]
    predictions = model.predict(feature_frame)
    output = input_frame.copy()
    output["Prediction"] = predictions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)
        output["Confidence"] = probabilities.max(axis=1)
    output["Model"] = metadata["model_name"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict malware labels from a feature CSV.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    print(predict_csv(args.csv_path).to_string(index=False))
