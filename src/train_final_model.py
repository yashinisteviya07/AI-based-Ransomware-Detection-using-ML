"""Train and save the final Decision Tree ransomware detector."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "Obfuscated-MalMem2022.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_ransomware_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"
TARGET_COLUMN = "Class"
REMOVED_COLUMNS = {
    "Category": "High-cardinality descriptive text excluded from the ML feature matrix.",
    "pslist.nprocs64bit": "Constant/zero-variance feature excluded.",
    "handles.nport": "Constant/zero-variance feature excluded.",
    "svcscan.interactive_process_services": "Constant/zero-variance feature excluded.",
}
RANDOM_STATE = 42


def build_training_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the CSV and apply the exact Step 2 feature decisions."""
    dataframe = pd.read_csv(DATASET_PATH)
    if TARGET_COLUMN not in dataframe.columns:
        raise KeyError(f"Missing target column: {TARGET_COLUMN}")

    feature_frame = dataframe.drop(columns=[TARGET_COLUMN, *REMOVED_COLUMNS])
    if feature_frame.select_dtypes(exclude="number").shape[1] != 0:
        raise TypeError("All final model features must be numeric.")
    if feature_frame.isna().any().any():
        raise ValueError("Unexpected missing values found in final feature columns.")

    target = dataframe[TARGET_COLUMN]
    return feature_frame, target, feature_frame.columns.tolist()


def train_final_model() -> dict[str, object]:
    """Train, evaluate, serialize, and describe the final model."""
    features, target, feature_names = build_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                feature_names,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, pos_label="Malware", zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, pos_label="Malware", zero_division=0)),
        "F1-Score": float(f1_score(y_test, predictions, pos_label="Malware", zero_division=0)),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    metadata = {
        "model_name": "Decision Tree",
        "dataset_name": DATASET_PATH.name,
        "dataset_path": str(DATASET_PATH),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "target_column": TARGET_COLUMN,
        "target_classes": sorted(target.unique().tolist()),
        "positive_class": "Malware",
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "preprocessing": {
            "removed_columns": REMOVED_COLUMNS,
            "numeric_features": True,
            "imputation": "Median imputation fitted inside the training pipeline.",
            "scaling": "Not applied because the final model is tree-based.",
        },
        "test_metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "validation_caveat": "Performance is measured on CIC-MalMem2022 and may reflect dataset-specific separability. External or temporal validation is required before making real-world deployment claims.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    final_metadata = train_final_model()
    print(json.dumps(final_metadata, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
