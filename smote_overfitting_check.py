"""
Item 1.4 — Check for overfitting caused by SMOTE.

Reviewer question: "Are the synthetic SMOTE samples inflating the
apparent performance (memorization of synthetic patterns)?"

Two analyses complementary to item 1.2:
  1. Train (post-SMOTE) vs Test: metrics of the final model computed on the
     balanced training set itself (with synthetic samples) vs on the
     held-out test set (never seen, no SMOTE). Large gap = overfitting,
     possibly inflated by the synthetic samples.
  2. Learning curves: validation recall/F1 as a function of the training
     set size (fractions of the dev set), with and without SMOTE. If the
     with-SMOTE curve saturates early with a persistent train-val gap, it
     suggests memorization of synthetic patterns.

Reuses the unified pipeline (unified_cv_pipeline) — does not reimplement
training/evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from unified_cv_pipeline import (
    FinalModelResult,
    SplitData,
    compute_multilabel_metrics,
    compute_pos_weights,
    dataframe_to_markdown,
    fit_fold_model,
    get_cv_splits,
    preprocess_fold,
    predict_proba_any,
)

LEARNING_CURVE_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
RANDOM_STATE = 42


def build_train_vs_test_table(
    final_results: Dict[str, FinalModelResult],
) -> pd.DataFrame:
    """Train (post-SMOTE) vs held-out test table, per model."""
    rows: List[Dict[str, Any]] = []
    for final in final_results.values():
        for metric_key, metric_label in (
            ("recall_micro", "Recall"),
            ("precision_micro", "Precision"),
            ("f1_micro", "F1"),
        ):
            train_val = final.train_metrics_on_balanced.get(metric_key, float("nan"))
            test_val = final.test_metrics[metric_key]
            rows.append(
                {
                    "Model": final.model_name,
                    "Metric": metric_label,
                    "Train (post-SMOTE)": f"{train_val:.4f}",
                    "Test (holdout)": f"{test_val:.4f}",
                    "Gap Train-Test": f"{train_val - test_val:.4f}",
                }
            )
    return pd.DataFrame(rows)


def compute_learning_curves(
    split: SplitData,
    model_types: List[str],
    use_smote: bool,
    fractions=LEARNING_CURVE_FRACTIONS,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Learning curves using the first CV fold as the fixed validation set.

    The training set is subsampled in a stratified way (has_defect)
    at increasing fractions; SMOTE (if enabled) is applied after the
    subsampling, to the training data only.
    """
    train_idx, val_idx = get_cv_splits(split.y_dev)[0]
    X_train_full = split.X_dev[train_idx]
    y_train_full = split.y_dev[train_idx]
    X_val = split.X_dev[val_idx]
    y_val = split.y_dev[val_idx]

    has_defect = (y_train_full.sum(axis=1) > 0).astype(int)
    rng = np.random.RandomState(RANDOM_STATE)

    results: List[Dict[str, Any]] = []
    for fraction in fractions:
        if fraction >= 1.0:
            subset_idx = np.arange(len(X_train_full))
        else:
            # Subsampling stratified by has_defect
            subset_parts = []
            for label in (0, 1):
                label_idx = np.where(has_defect == label)[0]
                n_take = max(1, int(round(len(label_idx) * fraction)))
                subset_parts.append(
                    rng.choice(label_idx, size=n_take, replace=False)
                )
            subset_idx = np.concatenate(subset_parts)
            rng.shuffle(subset_idx)

        X_sub = X_train_full[subset_idx]
        y_sub = y_train_full[subset_idx]
        pos_weights = compute_pos_weights(y_sub)

        X_sub_scaled, X_val_scaled, y_sub_balanced, scaler = preprocess_fold(
            X_sub,
            X_val,
            y_sub.copy(),
            split.defect_names,
            use_smote=use_smote,
            verbose=False,
        )
        X_sub_eval_scaled = scaler.transform(X_sub).astype(np.float32)

        for model_type in model_types:
            model, train_metrics, val_metrics, train_time = fit_fold_model(
                model_type,
                X_sub_scaled,
                y_sub_balanced,
                X_sub_eval_scaled,
                y_sub,
                X_val_scaled,
                y_val,
                pos_weights,
                split.defect_names,
            )
            # Metrics also on the balanced training set (with synthetic samples), to
            # expose inflation by SMOTE
            train_proba_balanced = predict_proba_any(model_type, model, X_sub_scaled)
            train_metrics_balanced = compute_multilabel_metrics(
                y_sub_balanced, train_proba_balanced
            )

            results.append(
                {
                    "model_type": model_type,
                    "use_smote": use_smote,
                    "fraction": fraction,
                    "n_train_original": int(len(X_sub)),
                    "n_train_effective": int(len(X_sub_scaled)),
                    "train_metrics_original": train_metrics,
                    "train_metrics_balanced": train_metrics_balanced,
                    "val_metrics": val_metrics,
                    "train_time_sec": train_time,
                }
            )
            if verbose:
                print(
                    f"    {model_type} — fraction {fraction:.0%} "
                    f"(n={len(X_sub):,}{' + SMOTE' if use_smote else ''}): "
                    f"Recall(val)={val_metrics['recall_micro']:.4f}, "
                    f"F1(val)={val_metrics['f1_micro']:.4f}, "
                    f"time={train_time:.1f}s"
                )
    return results


def plot_learning_curves(
    curves: List[Dict[str, Any]],
    model_display_names: Dict[str, str],
    output_path: str = "figures/phase1_learning_curves.png",
) -> str:
    """Generates the figure with learning curves (F1 and Recall) per model/condition."""
    model_types = sorted({c["model_type"] for c in curves})
    conditions = sorted({c["use_smote"] for c in curves})

    fig, axes = plt.subplots(
        2, len(model_types), figsize=(5.5 * len(model_types), 9), squeeze=False
    )

    for col, model_type in enumerate(model_types):
        for metric_row, (metric_key, metric_label) in enumerate(
            (("f1_micro", "F1-micro"), ("recall_micro", "Recall micro"))
        ):
            ax = axes[metric_row][col]
            for use_smote in conditions:
                points = sorted(
                    (
                        c
                        for c in curves
                        if c["model_type"] == model_type
                        and c["use_smote"] == use_smote
                    ),
                    key=lambda c: c["fraction"],
                )
                if not points:
                    continue
                x = [p["n_train_original"] for p in points]
                y_train = [p["train_metrics_original"][metric_key] for p in points]
                y_val = [p["val_metrics"][metric_key] for p in points]
                label_suffix = "with SMOTE" if use_smote else "without SMOTE"
                color = "tab:blue" if use_smote else "tab:orange"
                ax.plot(x, y_train, "--o", color=color, alpha=0.6,
                        label=f"Train ({label_suffix})")
                ax.plot(x, y_val, "-s", color=color,
                        label=f"Validation ({label_suffix})")

            ax.set_title(
                f"{model_display_names.get(model_type, model_type)} — {metric_label}"
            )
            ax.set_xlabel("Original training samples")
            ax.set_ylabel(metric_label)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    fig.suptitle(
        "Learning curves — SMOTE overfitting check (Phase 1.4)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    return output_path


def build_smote_overfitting_markdown(
    train_vs_test_df: pd.DataFrame,
    curves: List[Dict[str, Any]],
    figure_path: str,
) -> str:
    curve_rows = []
    for c in sorted(curves, key=lambda c: (c["model_type"], c["use_smote"], c["fraction"])):
        curve_rows.append(
            {
                "Model": c["model_type"],
                "Condition": "With SMOTE" if c["use_smote"] else "No SMOTE",
                "Train fraction": f"{c['fraction']:.0%}",
                "N train (orig.)": c["n_train_original"],
                "N train (effective)": c["n_train_effective"],
                "F1 train (orig.)": f"{c['train_metrics_original']['f1_micro']:.4f}",
                "F1 train (post-SMOTE)": f"{c['train_metrics_balanced']['f1_micro']:.4f}",
                "F1 validation": f"{c['val_metrics']['f1_micro']:.4f}",
                "Recall validation": f"{c['val_metrics']['recall_micro']:.4f}",
            }
        )
    curves_df = pd.DataFrame(curve_rows)

    lines = [
        "# Phase 1.4 — Overfitting Caused by SMOTE\n",
        "## Train (post-SMOTE) vs held-out Test\n",
        "Metrics of the final model computed on the balanced training set "
        "itself (containing synthetic samples) vs on the never-seen test set. "
        "A large gap indicates overfitting, possibly inflated by SMOTE.\n",
        dataframe_to_markdown(train_vs_test_df),
        "",
        "## Learning curves\n",
        f"![Learning curves]({Path(figure_path).name})\n",
        dataframe_to_markdown(curves_df),
        "",
    ]
    return "\n".join(lines)
