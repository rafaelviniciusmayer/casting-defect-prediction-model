"""
Item 1.2 — Formal overfitting analysis via 80/20 + cross-validation.

Reviewer question: "Are the models overfitting?"

Consolidates, per model, Recall/Precision/F1 in three columns:
  - Train (CV): mean over the training folds (evaluated on the ORIGINAL fold
    data, without SMOTE samples, so as not to inflate the metric artificially)
  - Validation (CV): mean over the validation folds
  - Final Test: model trained on the full 80%, evaluated on the held-out 20%

Large Train→Validation gaps indicate overfitting during training;
large Validation→Test gaps indicate overfitting/distribution shift.

This module does NOT train anything: it consumes the results already computed by the
unified pipeline (unified_cv_pipeline.run_cross_validation /
run_final_test_evaluation), orchestrated by run_phase1_validation.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from unified_cv_pipeline import (
    CVResult,
    FinalModelResult,
    dataframe_to_markdown,
    format_mean_std,
)

GAP_METRIC_KEYS = ("recall_micro", "precision_micro", "f1_micro")
METRIC_LABELS = (
    ("recall_micro", "Recall"),
    ("precision_micro", "Precision"),
    ("f1_micro", "F1"),
)


def build_overfitting_payload(
    cv_results: Dict[str, CVResult],
    final_results: Dict[str, FinalModelResult],
    use_smote: bool,
) -> Dict[str, Any]:
    """Builds the JSON payload of the analysis from already computed results."""
    models = []
    for model_type, cv_result in cv_results.items():
        final_result = final_results[model_type]
        gap_train_val = {
            key: cv_result.train_mean[key] - cv_result.val_mean[key]
            for key in GAP_METRIC_KEYS
        }
        gap_val_test = {
            key: cv_result.val_mean[key] - final_result.test_metrics[key]
            for key in GAP_METRIC_KEYS
        }
        models.append(
            {
                "model_type": model_type,
                "model_name": cv_result.model_name,
                "use_smote": use_smote,
                "cv_train_mean": cv_result.train_mean,
                "cv_train_std": cv_result.train_std,
                "cv_val_mean": cv_result.val_mean,
                "cv_val_std": cv_result.val_std,
                "test_metrics": final_result.test_metrics,
                "gap_train_val": gap_train_val,
                "gap_val_test": gap_val_test,
                "cv_total_train_time_sec": cv_result.total_train_time_sec,
                "final_train_time_sec": final_result.train_time_sec,
            }
        )
    return {"use_smote": use_smote, "models": models}


def build_overfitting_table(payload: Dict[str, Any]) -> pd.DataFrame:
    """Single table: Recall/Precision/F1 on Train (CV), Validation (CV) and Test."""
    rows: List[Dict[str, Any]] = []
    for model in payload["models"]:
        for metric_key, metric_label in METRIC_LABELS:
            rows.append(
                {
                    "Model": model["model_name"],
                    "Metric": metric_label,
                    "Train (CV)": format_mean_std(
                        model["cv_train_mean"][metric_key],
                        model["cv_train_std"][metric_key],
                    ),
                    "Validation (CV)": format_mean_std(
                        model["cv_val_mean"][metric_key],
                        model["cv_val_std"][metric_key],
                    ),
                    "Test": f"{model['test_metrics'][metric_key]:.4f}",
                    "Gap Train-Val": f"{model['gap_train_val'][metric_key]:.4f}",
                    "Gap Val-Test": f"{model['gap_val_test'][metric_key]:.4f}",
                }
            )
    return pd.DataFrame(rows)


def build_overfitting_markdown(df: pd.DataFrame, payload: Dict[str, Any]) -> str:
    smote_label = "with SMOTE" if payload["use_smote"] else "without SMOTE"
    lines = [
        f"# Phase 1.2 — Overfitting Analysis ({smote_label})\n",
        "Fixed threshold 0.5 across all columns for direct comparison between "
        "train (CV), validation (CV), and final test. Train metrics are computed "
        "on original fold data (no synthetic SMOTE samples).\n",
        "Gap Train-Val > 0 indicates possible overfitting within folds. "
        "Gap Val-Test > 0 indicates performance drop on the holdout set.\n",
        dataframe_to_markdown(df),
        "",
    ]
    return "\n".join(lines)
