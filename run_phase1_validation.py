"""
Phase 1 — Validation infrastructure (adjustments requested by the reviewers)
============================================================================

Orchestrates the Phase 1 items, running each CV only once per
(model, condition) and reusing the results across the analyses:

  1.1 Unified stratified CV (5 folds) for PyTorch NN, XGBoost and
      Random Forest — same CV as the neural network, SMOTE only in the training fold.
  1.2 Formal overfitting analysis: Train (CV) vs Validation (CV) vs
      Final Test (20% holdout), per model (overfitting_analysis.py).
  1.3 Ablation with/without SMOTE, cost-sensitive always on (smote_ablation.py).
  1.4 Overfitting caused by SMOTE: post-SMOTE train vs test + learning
      curves (smote_overfitting_check.py).

Run: python run_phase1_validation.py

Outputs:
  reports/phase1_cv_comparison.{json,md}           (1.1)
  reports/phase1_overfitting_analysis.{json,md}    (1.2)
  reports/phase1_smote_comparison.{json,md}        (1.3)
  reports/phase1_smote_overfitting.{json,md}       (1.4)
  reports/PHASE1_CHECKPOINT.md                     (phase decisions)
  figures/table_phase1_overfitting.csv
  figures/table_phase1_smote_comparison.csv
  figures/table_phase1_train_vs_test.csv
  figures/phase1_learning_curves.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

# The Windows console uses cp1252 by default; force UTF-8 output (Δ, ±, −)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from unified_cv_pipeline import (
    MODEL_DISPLAY_NAMES,
    available_model_types,
    cv_result_to_dict,
    format_mean_std,
    load_pipeline_data,
    prepare_split_data,
    run_cross_validation,
    run_final_test_evaluation,
)
from overfitting_analysis import (
    build_overfitting_markdown,
    build_overfitting_payload,
    build_overfitting_table,
)
from smote_ablation import (
    build_smote_comparison_payload,
    build_smote_comparison_table,
    build_smote_markdown,
    build_smote_recommendations,
    decide_official_smote,
)
from smote_overfitting_check import (
    build_smote_overfitting_markdown,
    build_train_vs_test_table,
    compute_learning_curves,
    plot_learning_curves,
)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_text(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def _build_cv_comparison_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Phase 1.1 — Cross-Validation Comparison (5 folds)\n",
        "Mean ± standard deviation of the metrics on the **validation** part of each fold "
        "(fixed threshold 0.5). SMOTE applied only to the training part of each fold; "
        "the final test set (20%) remains held out.\n",
        "| Model | Recall (val) | Precision (val) | F1-micro (val) | F1-macro (val) | CV Time (s) |",
        "|--------|--------------|-----------------|----------------|----------------|--------------|",
    ]
    for model in payload["models"]:
        vm = model["val_mean"]
        vs = model["val_std"]
        lines.append(
            f"| {model['model_name']} | "
            f"{format_mean_std(vm['recall_micro'], vs['recall_micro'])} | "
            f"{format_mean_std(vm['precision_micro'], vs['precision_micro'])} | "
            f"{format_mean_std(vm['f1_micro'], vs['f1_micro'])} | "
            f"{format_mean_std(vm['f1_macro'], vs['f1_macro'])} | "
            f"{model['total_train_time_sec']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def _build_checkpoint_markdown(
    decision: Dict[str, Any],
    cv_payload: Dict[str, Any],
) -> str:
    smote_status = "KEPT" if decision["keep_smote"] else "REMOVED"
    best_recall = max(
        cv_payload["models"], key=lambda m: m["val_mean"]["recall_micro"]
    )
    best_f1 = max(cv_payload["models"], key=lambda m: m["val_mean"]["f1_micro"])
    lines = [
        "# Checkpoint — Phase 1 (Validation infrastructure)\n",
        "## Decisions taken\n",
        f"- **SMOTE: {smote_status}** in the official pipeline configuration "
        f"(mean F1-micro gain {decision['avg_f1_micro_gain']:+.4f}, "
        f"recall {decision['avg_recall_gain']:+.4f}, "
        f"train-val gap increase {decision['avg_train_val_gap_increase']:+.4f}).",
        "- **Unified CV**: the 3 models now use the same stratified CV "
        "with 5 folds (random_state=42) over the dev set (80%), with SMOTE "
        "(when enabled) applied only to the training fold.",
        "- **Cost-sensitive learning** active in all models: pos_weight "
        "in the NN, class_weight='balanced' in RF, scale_pos_weight in XGBoost "
        "(new in this phase — XGBoost previously had no weighting).",
        f"- Best recall (CV validation, with SMOTE): **{best_recall['model_name']}** "
        f"({best_recall['val_mean']['recall_micro']:.4f}).",
        f"- Best F1-micro (CV validation, with SMOTE): **{best_f1['model_name']}** "
        f"({best_f1['val_mean']['f1_micro']:.4f}).",
        "",
        "## Reference for the next phases\n",
        "- Phase 2 (new models) must use `unified_cv_pipeline.run_cross_validation` "
        f"with `use_smote={decision['keep_smote']}`.",
        "- Phase 3 (thresholds/ROC) must start from the final models trained via "
        "`unified_cv_pipeline.run_final_test_evaluation` under the same condition.",
        "",
        "## Generated files\n",
        "- `reports/phase1_cv_comparison.md` (1.1)",
        "- `reports/phase1_overfitting_analysis.md` (1.2)",
        "- `reports/phase1_smote_comparison.md` (1.3)",
        "- `reports/phase1_smote_overfitting.md` (1.4)",
        "- `figures/table_phase1_overfitting.csv`",
        "- `figures/table_phase1_smote_comparison.csv`",
        "- `figures/table_phase1_train_vs_test.csv`",
        "- `figures/phase1_learning_curves.png`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    t_start = time.perf_counter()
    print("=" * 70)
    print("PHASE 1 — VALIDATION INFRASTRUCTURE")
    print("=" * 70)

    Path("reports").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    print("\n[*] Loading data (existing pipeline)...")
    data = load_pipeline_data(verbose=True)
    split = prepare_split_data(data, verbose=True)
    model_types = available_model_types()

    # ------------------------------------------------------------------
    # Training: each CV runs ONCE per (model, condition) and is reused
    # ------------------------------------------------------------------
    cv_with: Dict[str, Any] = {}
    cv_without: Dict[str, Any] = {}
    final_with: Dict[str, Any] = {}

    print("\n" + "=" * 70)
    print("TRAINING — 5-fold CV per model and condition (with/without SMOTE)")
    print("=" * 70)
    for model_type in model_types:
        cv_with[model_type] = run_cross_validation(
            model_type, split, use_smote=True, verbose=True
        )
        cv_without[model_type] = run_cross_validation(
            model_type, split, use_smote=False, verbose=True
        )
        final_with[model_type] = run_final_test_evaluation(
            model_type, split, use_smote=True, verbose=True
        )
        final_with[model_type].cv_val_mean = cv_with[model_type].val_mean

    # ------------------------------------------------------------------
    # 1.1 — Unified CV comparison (current official condition: with SMOTE)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1.1 — Unified CV report")
    print("=" * 70)
    cv_payload = {
        "phase": "1.1_cv_comparison",
        "use_smote": True,
        "n_folds": 5,
        "split": "80/20 stratified (random_state=42)",
        "models": [cv_result_to_dict(cv_with[m]) for m in model_types],
    }
    _save_json("reports/phase1_cv_comparison.json", cv_payload)
    _write_text(
        "reports/phase1_cv_comparison.md", _build_cv_comparison_markdown(cv_payload)
    )
    print("[OK] reports/phase1_cv_comparison.{json,md}")

    # ------------------------------------------------------------------
    # 1.2 — Formal overfitting analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1.2 — Formal overfitting analysis")
    print("=" * 70)
    overfit_payload = build_overfitting_payload(cv_with, final_with, use_smote=True)
    overfit_df = build_overfitting_table(overfit_payload)
    overfit_df.to_csv("figures/table_phase1_overfitting.csv", index=False)
    overfit_md = build_overfitting_markdown(overfit_df, overfit_payload)
    _write_text("reports/phase1_overfitting_analysis.md", overfit_md)
    _write_text("figures/table_phase1_overfitting.md", overfit_md)
    _save_json("reports/phase1_overfitting_analysis.json", overfit_payload)
    print(overfit_df.to_string(index=False))
    print("[OK] reports/phase1_overfitting_analysis.{json,md}")

    # ------------------------------------------------------------------
    # 1.3 — SMOTE ablation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1.3 — SMOTE contribution (with vs without)")
    print("=" * 70)
    smote_payload = build_smote_comparison_payload(cv_without, cv_with)
    smote_df = build_smote_comparison_table(smote_payload)
    smote_df.to_csv("figures/table_phase1_smote_comparison.csv", index=False)
    recommendations = build_smote_recommendations(smote_payload)
    decision = decide_official_smote(smote_payload)
    smote_md = build_smote_markdown(smote_df, recommendations, decision)
    _write_text("reports/phase1_smote_comparison.md", smote_md)
    _write_text("figures/table_phase1_smote_comparison.md", smote_md)
    smote_payload["recommendations"] = recommendations
    smote_payload["decision"] = decision
    _save_json("reports/phase1_smote_comparison.json", smote_payload)
    print(smote_df.to_string(index=False))
    print(f"\n[DECISION] SMOTE {'kept' if decision['keep_smote'] else 'removed'} "
          f"in the official configuration.")
    print("[OK] reports/phase1_smote_comparison.{json,md}")

    # ------------------------------------------------------------------
    # 1.4 — Overfitting caused by SMOTE (post-SMOTE train vs test + curves)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1.4 — Overfitting caused by SMOTE")
    print("=" * 70)
    train_vs_test_df = build_train_vs_test_table(final_with)
    train_vs_test_df.to_csv("figures/table_phase1_train_vs_test.csv", index=False)
    print(train_vs_test_df.to_string(index=False))

    print("\n[*] Learning curves (fold 1 as the fixed validation set)...")
    curves = []
    curves += compute_learning_curves(split, model_types, use_smote=True, verbose=True)
    curves += compute_learning_curves(split, model_types, use_smote=False, verbose=True)
    figure_path = plot_learning_curves(curves, MODEL_DISPLAY_NAMES)
    print(f"    [OK] {figure_path}")

    smote_overfit_md = build_smote_overfitting_markdown(
        train_vs_test_df, curves, figure_path
    )
    _write_text("reports/phase1_smote_overfitting.md", smote_overfit_md)
    _save_json(
        "reports/phase1_smote_overfitting.json",
        {
            "train_vs_test": train_vs_test_df.to_dict(orient="records"),
            "learning_curves": curves,
        },
    )
    print("[OK] reports/phase1_smote_overfitting.{json,md}")

    # ------------------------------------------------------------------
    # Phase 1 checkpoint
    # ------------------------------------------------------------------
    checkpoint_md = _build_checkpoint_markdown(decision, cv_payload)
    _write_text("reports/PHASE1_CHECKPOINT.md", checkpoint_md)

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"[DONE] Phase 1 completed in {elapsed / 60:.1f} min")
    print("=" * 70)
    print("Checkpoint: reports/PHASE1_CHECKPOINT.md")


if __name__ == "__main__":
    main()
