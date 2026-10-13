"""
Item 2.1 (Phase 2) — Comparison with statistical and regression models.

Reviewer question: "Why only 'black-box' ML models? Compare with
classical statistical/regression models."

Adds to the comparison (besides PyTorch NN, XGBoost and Random Forest):
  - L2/Ridge Logistic Regression (class_weight='balanced') — interpretable linear
    baseline, via MultiOutputClassifier like XGBoost/RF.
  - L1/Lasso Logistic Regression — formal regularized statistical baseline.

Uses exactly the same pipeline as Phase 1 (unified_cv_pipeline):
same 80/20 split, same 5-fold CV, SMOTE according to the item 1.3 decision
(read automatically from reports/phase1_smote_comparison.json) and the same
per-defect threshold optimization (maximize recall) in the final evaluation.

Prerequisite: run run_phase1_validation.py first (it generates the SMOTE decision
and the CVs of the 3 original models, reused here without retraining).

Run: python compare_statistical_models.py

Outputs:
  reports/phase2_model_comparison.{json,md}
  reports/PHASE2_CHECKPOINT.md
  figures/table_phase2_model_comparison.csv  (equivalent to Table 8 of the article)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import f1_score, precision_score, recall_score

from compare_models import _optimize_thresholds_silent
from unified_cv_pipeline import (
    MODEL_DISPLAY_NAMES,
    MODEL_LOGISTIC_L1,
    MODEL_LOGISTIC_L2,
    MODEL_PYTORCH,
    MODEL_RANDOM_FOREST,
    MODEL_XGBOOST,
    available_model_types,
    cv_result_to_dict,
    dataframe_to_markdown,
    format_mean_std,
    load_pipeline_data,
    macro_f1_observed,
    predict_proba_any,
    prepare_split_data,
    run_cross_validation,
    run_final_test_evaluation,
)

PHASE1_DECISION_PATH = Path("reports/phase1_smote_comparison.json")
PHASE1_CV_PATH = Path("reports/phase1_cv_comparison.json")

NEW_MODEL_TYPES = [MODEL_LOGISTIC_L2, MODEL_LOGISTIC_L1]
OOF_DIR = Path("models/oof_proba")


def load_phase1_smote_decision() -> bool:
    """Reads the official Phase 1 decision (keep SMOTE or not)."""
    if not PHASE1_DECISION_PATH.exists():
        raise FileNotFoundError(
            "Run run_phase1_validation.py first: SMOTE decision "
            f"not found at {PHASE1_DECISION_PATH}"
        )
    with open(PHASE1_DECISION_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return bool(payload["decision"]["keep_smote"])


def load_phase1_cv_stats(use_smote: bool) -> Dict[str, Dict[str, Any]]:
    """
    Reuses the Phase 1 CVs for the 3 original models, under the official
    condition, avoiding retraining.
    """
    with open(PHASE1_DECISION_PATH, encoding="utf-8") as f:
        smote_payload = json.load(f)
    condition_key = "with_smote" if use_smote else "without_smote"
    stats = {}
    for model in smote_payload["models"]:
        cond = model[condition_key]
        stats[model["model_type"]] = {
            "model_name": model["model_name"],
            "cv_val_mean": cond["cv_val_mean"],
            "cv_val_std": cond["cv_val_std"],
            "cv_total_train_time_sec": cond["total_train_time_sec"],
        }
    return stats


def evaluate_final_with_thresholds(
    model_type: str,
    split,
    use_smote: bool,
) -> Dict[str, Any]:
    """
    Trains the final model on the 80% and evaluates on the test set with the SAME
    per-defect threshold optimization as the original pipeline (maximize recall), and also
    measures the training and inference time (ms per 100 samples).
    """
    final = run_final_test_evaluation(model_type, split, use_smote=use_smote, verbose=True)

    X_test_scaled = final.scaler.transform(split.X_test).astype(np.float32)
    y_pred_proba = predict_proba_any(model_type, final.model, X_test_scaled)

    oof_path = OOF_DIR / f"{model_type}.npy"
    if not oof_path.exists():
        raise FileNotFoundError(f"{oof_path} not found: run `python oof_predictions.py {model_type}` first")
    thresholds, _ = _optimize_thresholds_silent(
        split.y_dev, np.load(oof_path), split.defect_names
    )
    th = np.array([thresholds[n] for n in split.defect_names])
    y_pred_binary = (y_pred_proba >= th).astype(int)
    optimized_metrics = {
        "recall_micro": recall_score(split.y_test, y_pred_binary, average="micro", zero_division=0),
        "precision_micro": precision_score(split.y_test, y_pred_binary, average="micro", zero_division=0),
        "f1_micro": f1_score(split.y_test, y_pred_binary, average="micro", zero_division=0),
        "f1_macro": macro_f1_observed(split.y_test, y_pred_binary),
    }

    # Inference time: 100 repetitions over 100 samples
    X_bench = X_test_scaled[:100]
    t0 = time.perf_counter()
    if model_type == MODEL_PYTORCH:
        final.model.eval()
        with torch.no_grad():
            for _ in range(100):
                _ = torch.sigmoid(final.model(torch.FloatTensor(X_bench)))
    else:
        for _ in range(100):
            _ = final.model.predict_proba(X_bench)
    inference_ms_per_100 = (time.perf_counter() - t0) / 100 * 1000

    return {
        "model_type": model_type,
        "model_name": final.model_name,
        "use_smote": use_smote,
        "test_metrics_threshold_05": final.test_metrics,
        "test_metrics_optimized_threshold": optimized_metrics,
        "train_time_sec": final.train_time_sec,
        "inference_time_ms_per_100": inference_ms_per_100,
    }


def build_comparison_table(
    cv_stats: Dict[str, Dict[str, Any]],
    final_evals: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Table equivalent to Table 8 of the article, now with 5 models."""
    rows: List[Dict[str, Any]] = []
    for model_type, final in final_evals.items():
        cv = cv_stats.get(model_type)
        opt = final["test_metrics_optimized_threshold"]
        rows.append(
            {
                "Model": final["model_name"],
                "Recall (test)": f"{opt['recall_micro']:.4f}",
                "Precision (test)": f"{opt['precision_micro']:.4f}",
                "F1-micro (test)": f"{opt['f1_micro']:.4f}",
                "F1-macro (test)": f"{opt['f1_macro']:.4f}",
                "Recall (CV)": (
                    format_mean_std(
                        cv["cv_val_mean"]["recall_micro"], cv["cv_val_std"]["recall_micro"]
                    )
                    if cv
                    else "—"
                ),
                "F1-micro (CV)": (
                    format_mean_std(
                        cv["cv_val_mean"]["f1_micro"], cv["cv_val_std"]["f1_micro"]
                    )
                    if cv
                    else "—"
                ),
                "Train (s)": f"{final['train_time_sec']:.1f}",
                "Inference (ms/100)": f"{final['inference_time_ms_per_100']:.2f}",
            }
        )
    return pd.DataFrame(rows)


def build_phase2_markdown(df: pd.DataFrame, use_smote: bool) -> str:
    smote_label = "with SMOTE" if use_smote else "without SMOTE (cost-sensitive only)"
    lines = [
        "# Phase 2.1 — Comparison with statistical and regression models\n",
        f"Official Phase 1 configuration: **{smote_label}**. "
        "All models use the same 80/20 split, the same 5-fold CV, the same "
        "preprocessing and the same per-defect threshold optimization "
        "(maximize recall) in the final test evaluation.\n",
        "Statistical baselines added: Logistic Regression L2 (Ridge) and "
        "L1 (Lasso), both with `class_weight='balanced'` via `MultiOutputClassifier`.\n",
        dataframe_to_markdown(df),
        "",
    ]
    return "\n".join(lines)


def build_phase2_checkpoint_markdown(
    final_evals: Dict[str, Dict[str, Any]],
    use_smote: bool,
) -> str:
    best_recall = max(
        final_evals.values(),
        key=lambda e: e["test_metrics_optimized_threshold"]["recall_micro"],
    )
    best_f1 = max(
        final_evals.values(),
        key=lambda e: e["test_metrics_optimized_threshold"]["f1_micro"],
    )
    return "\n".join(
        [
            "# Checkpoint — Phase 2 (statistical models)\n",
            f"- Official condition: {'with' if use_smote else 'without'} SMOTE "
            "(inherited from Phase 1).",
            "- Models compared: PyTorch NN, XGBoost, Random Forest, "
            "Logistic Regression L2 (Ridge), Logistic Regression L1 (Lasso).",
            f"- Best test recall (optimized thresholds): "
            f"**{best_recall['model_name']}** "
            f"({best_recall['test_metrics_optimized_threshold']['recall_micro']:.4f}).",
            f"- Best test F1-micro: **{best_f1['model_name']}** "
            f"({best_f1['test_metrics_optimized_threshold']['f1_micro']:.4f}).",
            "- Phase 3 must use these 4-5 final models for feature "
            "importance (3.1) and the test probabilities for thresholds/ROC "
            "(3.2/3.3).",
            "",
        ]
    )


def main() -> None:
    t_start = time.perf_counter()
    print("=" * 70)
    print("PHASE 2 — COMPARISON WITH STATISTICAL MODELS")
    print("=" * 70)

    Path("reports").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    use_smote = load_phase1_smote_decision()
    print(f"\n[*] Phase 1 decision: SMOTE {'KEPT' if use_smote else 'REMOVED'} "
          f"(official condition: {'with' if use_smote else 'without'} SMOTE)")

    print("\n[*] Loading data (existing pipeline)...")
    data = load_pipeline_data(verbose=True)
    split = prepare_split_data(data, verbose=True)

    # CV only for the NEW models; the 3 original ones reuse Phase 1
    cv_stats = load_phase1_cv_stats(use_smote)
    new_cv_results = {}
    for model_type in NEW_MODEL_TYPES:
        cv_result = run_cross_validation(model_type, split, use_smote=use_smote, verbose=True)
        new_cv_results[model_type] = cv_result
        cv_stats[model_type] = {
            "model_name": cv_result.model_name,
            "cv_val_mean": cv_result.val_mean,
            "cv_val_std": cv_result.val_std,
            "cv_total_train_time_sec": cv_result.total_train_time_sec,
        }

    # Final evaluation (test, optimized thresholds) for the 5 models
    all_model_types = available_model_types() + NEW_MODEL_TYPES
    final_evals = {}
    for model_type in all_model_types:
        final_evals[model_type] = evaluate_final_with_thresholds(
            model_type, split, use_smote=use_smote
        )
        opt = final_evals[model_type]["test_metrics_optimized_threshold"]
        print(f"    [{MODEL_DISPLAY_NAMES[model_type]}] "
              f"Recall={opt['recall_micro']:.4f}, F1={opt['f1_micro']:.4f} "
              f"(optimized thresholds)")

    df = build_comparison_table(cv_stats, final_evals)
    df.to_csv("figures/table_phase2_model_comparison.csv", index=False)
    md = build_phase2_markdown(df, use_smote)
    Path("reports/phase2_model_comparison.md").write_text(md, encoding="utf-8")

    payload = {
        "phase": "2.1_statistical_models",
        "use_smote": use_smote,
        "cv_stats": cv_stats,
        "new_models_cv": {
            mt: cv_result_to_dict(cv) for mt, cv in new_cv_results.items()
        },
        "final_evaluations": final_evals,
    }
    with open("reports/phase2_model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    checkpoint = build_phase2_checkpoint_markdown(final_evals, use_smote)
    Path("reports/PHASE2_CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"[DONE] Phase 2 completed in {elapsed / 60:.1f} min")
    print("=" * 70)
    print(df.to_string(index=False))
    print("\nCheckpoint: reports/PHASE2_CHECKPOINT.md")


if __name__ == "__main__":
    main()
