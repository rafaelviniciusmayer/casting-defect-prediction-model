"""
Generate per-defect threshold analysis plots consistent with Table 28.

This script:
- Loads the saved model from models/best_model.pkl
- Recreates the same 80/20 train/test split used in train_model.py
- Loads the per-defect thresholds from model_analysis_report.json
- For each of the 28 defects, generates a 2x2 panel figure with:
  * Precision-Recall curve (using model probabilities)
  * F1-score vs. threshold
  * Precision and Recall vs. threshold
  * Probability distribution (histogram) for defective vs. non-defective samples

The vertical "Optimal" threshold line in the plots comes directly from
model_analysis_report.json, ensuring full consistency with Table 28.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_score, recall_score


def load_data_and_split():
    """Replicate the same 80/20 stratified split as in train_model.py."""
    df = pd.read_csv("aluminum_diecasting_dataset_with_features.csv")

    defect_prefixes = [
        "blisters",
        "surface",
        "die",
        "flow",
        "cold",
        "heat",
        "ejector",
        "low",
        "density",
        "incomplete",
        "flash",
        "warpage",
        "shrinkage",
        "volumetric",
        "dimensional",
        "gas",
        "internal",
        "cracks",
        "hard",
        "oxide",
    ]
    defect_cols = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in defect_prefixes)
    ]
    feature_cols = [
        col
        for col in df.columns
        if col not in defect_cols + ["id", "total_defects", "has_defect"]
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df[defect_cols].values.astype(np.float32)

    has_defect = (y.sum(axis=1) > 0).astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=has_defect
    )

    return X_test, y_test, defect_cols


def load_model_wrapper():
    """Load PyTorchModelWrapper artifacts from models/best_model.pkl."""
    # Import classes to make them available for unpickling
    import train_model  # noqa: F401
    import __main__ as main_module

    for attr in ("PyTorchModelWrapper", "DefectPredictionNN"):
        if hasattr(train_model, attr):
            setattr(main_module, attr, getattr(train_model, attr))

    with open("models/best_model.pkl", "rb") as f:
        artifacts = pickle.load(f)

    wrapper = artifacts["model"]
    defect_names = artifacts["defect_cols"]
    return wrapper, defect_names


def load_table28_thresholds():
    """Load per-defect thresholds from model_analysis_report.json."""
    with open("model_analysis_report.json", "r") as f:
        report = json.load(f)
    thresholds = report.get("thresholds", {})
    return thresholds


def compute_f1_for_thresholds(y_true, y_proba, thresholds_grid):
    precisions = []
    recalls = []
    f1_scores = []

    for thr in thresholds_grid:
        y_pred = (y_proba >= thr).astype(int)
        if y_pred.sum() == 0 and y_true.sum() == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            f1_scores.append(1.0)
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return np.array(precisions), np.array(recalls), np.array(f1_scores)


def plot_threshold_analysis_for_defect(
    defect_name,
    y_true,
    y_proba,
    table_threshold,
    output_dir,
):
    """Create the 2x2 threshold analysis figure for a single defect."""
    fixed_threshold = 0.5

    # Precision-Recall curve
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_proba)

    # F1 / Precision / Recall vs threshold grids
    thresholds_grid = np.linspace(0.01, 0.99, 99)
    grid_precision, grid_recall, grid_f1 = compute_f1_for_thresholds(
        y_true, y_proba, thresholds_grid
    )

    # Probability distribution
    probs_defect = y_proba[y_true == 1]
    probs_no_defect = y_proba[y_true == 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_pr, ax_f1), (ax_pr_rec, ax_hist) = axes

    # Top-left: Precision-Recall curve
    ax_pr.plot(pr_recall, pr_precision, label="PR Curve", color="blue")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_pr.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(True, alpha=0.3)

    # Top-right: F1-Score vs Threshold
    ax_f1.plot(
        thresholds_grid,
        grid_f1,
        color="green",
        label="F1-Score",
    )
    ax_f1.axvline(
        table_threshold,
        color="red",
        linestyle="--",
        label=f"Optimal: {table_threshold:.3f}",
    )
    ax_f1.axvline(
        fixed_threshold,
        color="orange",
        linestyle="--",
        label=f"Fixed: {fixed_threshold:.3f}",
    )
    ax_f1.set_title("F1-Score vs Threshold", fontsize=12, fontweight="bold")
    ax_f1.set_xlabel("Threshold")
    ax_f1.set_ylabel("F1-Score")
    ax_f1.set_ylim(0, 1.05)
    ax_f1.grid(True, alpha=0.3)
    ax_f1.legend(loc="lower left", fontsize=8)

    # Bottom-left: Precision and Recall vs Threshold
    ax_pr_rec.plot(
        thresholds_grid,
        grid_precision,
        color="blue",
        label="Precision",
    )
    ax_pr_rec.plot(
        thresholds_grid,
        grid_recall,
        color="red",
        label="Recall",
    )
    ax_pr_rec.axvline(
        table_threshold,
        color="green",
        linestyle="--",
        label=f"Optimal: {table_threshold:.3f}",
    )
    ax_pr_rec.axvline(
        fixed_threshold,
        color="orange",
        linestyle="--",
        label=f"Fixed: {fixed_threshold:.3f}",
    )
    ax_pr_rec.set_title("Precision and Recall vs Threshold", fontsize=12, fontweight="bold")
    ax_pr_rec.set_xlabel("Threshold")
    ax_pr_rec.set_ylabel("Score")
    ax_pr_rec.set_ylim(0, 1.05)
    ax_pr_rec.grid(True, alpha=0.3)
    ax_pr_rec.legend(loc="lower left", fontsize=8)

    # Bottom-right: Probability Distribution
    bins = np.linspace(0.0, 1.0, 50)
    ax_hist.hist(
        probs_no_defect,
        bins=bins,
        color="green",
        alpha=0.7,
        label="No Defect",
    )
    if len(probs_defect) > 0:
        ax_hist.hist(
            probs_defect,
            bins=bins,
            color="red",
            alpha=0.7,
            label="With Defect",
        )
    ax_hist.axvline(
        table_threshold,
        color="red",
        linestyle="--",
        label=f"Optimal: {table_threshold:.3f}",
    )
    ax_hist.axvline(
        fixed_threshold,
        color="orange",
        linestyle="--",
        label=f"Fixed: {fixed_threshold:.3f}",
    )
    ax_hist.set_title("Probability Distribution", fontsize=12, fontweight="bold")
    ax_hist.set_xlabel("Probability")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend(loc="upper right", fontsize=8)

    pretty_name = defect_name.replace("_", " ")
    fig.suptitle(
        f"Threshold Analysis: {pretty_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    safe_name = defect_name.replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"threshold_analysis_{safe_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    print("=" * 70)
    print("Generating per-defect threshold analysis plots (Table 28 consistent)")
    print("=" * 70)

    output_dir = Path("threshold_analysis")
    output_dir.mkdir(exist_ok=True)

    print("\n[1/4] Loading data and test split...")
    X_test, y_test, defect_cols = load_data_and_split()

    print("[2/4] Loading model wrapper...")
    wrapper, defect_names_model = load_model_wrapper()

    print("[3/4] Loading thresholds from model_analysis_report.json (Table 28)...")
    table_thresholds = load_table28_thresholds()

    # Predict probabilities for test set
    print("[4/4] Predicting probabilities on test set...")
    y_pred_proba = wrapper.predict_proba(X_test)

    # Align order between defect_cols, model outputs, and thresholds
    name_to_index = {name: idx for idx, name in enumerate(defect_names_model)}

    generated = 0
    for defect_name in defect_cols:
        if defect_name not in name_to_index:
            continue
        if defect_name not in table_thresholds:
            continue

        idx = name_to_index[defect_name]
        y_true_defect = y_test[:, idx]
        y_proba_defect = y_pred_proba[:, idx]
        table_thr = float(table_thresholds[defect_name])

        print(f"  - Generating plot for {defect_name} (threshold={table_thr:.3f})")
        plot_threshold_analysis_for_defect(
            defect_name,
            y_true_defect,
            y_proba_defect,
            table_thr,
            output_dir,
        )
        generated += 1

    print(f"\n[OK] Generated {generated} threshold analysis plots in '{output_dir}'")


if __name__ == "__main__":
    main()

