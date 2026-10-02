"""Generate final poster tables, summaries, and visualizations from project outputs."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "Obfuscated-MalMem2022.csv"
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images"
MODEL_PATH = ROOT / "models" / "final_ransomware_model.joblib"
METADATA_PATH = ROOT / "models" / "best_model_metadata.json"
RANDOM_STATE = 42


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model_results = pd.read_csv(REPORTS / "model_results.csv")
    cv_results = pd.read_csv(REPORTS / "cross_validation_results.csv")
    alternative_results = pd.read_csv(REPORTS / "alternative_split_results.csv")
    dataframe = pd.read_csv(DATASET_PATH)

    metric_columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    final_table = model_results[metric_columns].copy()
    final_table.to_csv(REPORTS / "final_results_table.csv", index=False)
    final_table.round(4).to_csv(REPORTS / "poster_results_table.csv", index=False)

    model_order = [
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "Support Vector Machine",
        "Logistic Regression",
    ]
    plot_table = final_table.set_index("Model").loc[model_order].reset_index()
    long_table = plot_table.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=long_table, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_title("Final Model Performance on the Held-Out Test Split", fontsize=16)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", rotation=20)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=2, fontsize=8)
    ax.legend(title="Metric", fontsize=10, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(IMAGES / "final_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    feature_names = metadata["feature_names"]
    X = dataframe[feature_names]
    y = dataframe[metadata["target_column"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=metadata["test_size"],
        random_state=RANDOM_STATE,
        stratify=y,
    )
    final_model = joblib.load(MODEL_PATH)
    predictions = final_model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=["Benign", "Malware"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Benign", "Malware"],
        yticklabels=["Benign", "Malware"],
        annot_kws={"size": 16},
        ax=ax,
    )
    ax.set_title("Final Decision Tree Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMAGES / "final_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    estimator = final_model.named_steps["model"]
    importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": estimator.feature_importances_}
    ).sort_values("Importance", ascending=False)
    importance["Rank"] = range(1, len(importance) + 1)
    top15 = importance.head(15)
    top15.to_csv(REPORTS / "final_feature_importance.csv", index=False)
    plot_top15 = top15.sort_values("Importance")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(plot_top15["Feature"], plot_top15["Importance"], color="#2f6f9f")
    ax.set_title("Top 15 Decision Tree Features", fontsize=16)
    ax.set_xlabel("Feature importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMAGES / "final_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    class_counts = dataframe[metadata["target_column"]].value_counts().reindex(["Benign", "Malware"])
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(class_counts.index, class_counts.values, color=["#4c78a8", "#e45756"])
    ax.set_title("CIC-MalMem2022 Class Distribution", fontsize=16)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of records", fontsize=12)
    ax.bar_label(bars, labels=[f"{value:,}" for value in class_counts.values], padding=3)
    fig.tight_layout()
    fig.savefig(IMAGES / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    cv_plot = cv_results.set_index("Model").loc[["Decision Tree", "Random Forest", "XGBoost"]].reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cv_plot["Model"], cv_plot["Mean F1-Score"], color="#59a14f")
    ax.set_title("Five-Fold Cross-Validation Mean F1-Score", fontsize=16)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean F1-Score", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    fig.tight_layout()
    fig.savefig(IMAGES / "cross_validation_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    top_features = ", ".join(importance.head(4)["Feature"].tolist())
    best_cv = cv_results.sort_values("Mean F1-Score", ascending=False).iloc[0]
    summary = (
        f"On the CIC-MalMem2022 dataset, the project analyzed {len(dataframe):,} records and used "
        f"{len(feature_names)} final behavioral features after preprocessing. "
        f"The selected Decision Tree achieved {final_table.loc[final_table['Model'] == 'Decision Tree', 'Accuracy'].iloc[0]:.6f} accuracy, "
        f"{final_table.loc[final_table['Model'] == 'Decision Tree', 'Precision'].iloc[0]:.6f} precision, "
        f"{final_table.loc[final_table['Model'] == 'Decision Tree', 'Recall'].iloc[0]:.6f} recall, and "
        f"{final_table.loc[final_table['Model'] == 'Decision Tree', 'F1-Score'].iloc[0]:.6f} F1-score on the held-out test split. "
        f"The strongest five-fold cross-validation mean F1-score was {best_cv['Mean F1-Score']:.6f} for {best_cv['Model']}. "
        f"The most influential final Decision Tree features included {top_features}. "
        "Near-perfect performance was observed across validation experiments, but the dataset shows strong behavioral feature separability. "
        "These results should be described as dataset-specific, and external or temporal validation is required before real-world deployment claims."
    )
    (REPORTS / "final_results_summary.txt").write_text(summary, encoding="utf-8")

    experiment_summary = {
        "dataset_name": "CIC-MalMem2022",
        "dataset_file": DATASET_PATH.name,
        "records": len(dataframe),
        "original_columns": dataframe.shape[1],
        "final_features": len(feature_names),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "selected_model": metadata["model_name"],
        "test_metrics": metadata["test_metrics"],
        "cross_validation_metrics": cv_results.to_dict(orient="records"),
        "alternative_split_metrics": alternative_results.to_dict(orient="records"),
        "validation_caveat": metadata["validation_caveat"],
    }
    (REPORTS / "experiment_summary.json").write_text(
        json.dumps(experiment_summary, indent=2), encoding="utf-8"
    )
    print("Generated final tables, summary, and five poster-ready visualizations.")
    print("Final confusion matrix:", matrix.tolist())
    print("Top 10 features:", importance.head(10)["Feature"].tolist())


if __name__ == "__main__":
    main()
