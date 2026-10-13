"""
Consolidated report — Adjustments requested by the review committee (Phases 1-3)
================================================================================

Concatenates the markdown reports of the three phases into a single document
(reports/CONSOLIDATED_REPORT.md), in phase order, with the decision
checkpoints at the beginning.

Thresholds: the official thresholds come from threshold_selection_oof.py
(selected on the out-of-fold predictions of the development set, without
using the test set); the corresponding section is generated from
reports/oof_threshold_results.json. The former reports/phase3_threshold_tradeoff.md
is now reports/phase3_threshold_tradeoff_test_selected.md (legacy, thresholds
chosen on the test set) and is not included in the report.

Also offers --run-all to run the three phases in sequence before
consolidating (useful to reproduce everything from scratch):
    python generate_consolidated_report.py --run-all

All paths are resolved from this script's directory (ROOT).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

# Marker of the section generated from the OOF thresholds JSON
OOF_THRESHOLDS_SECTION = "<oof_threshold_results>"
OOF_RESULTS_PATH = ROOT / "reports" / "oof_threshold_results.json"
LEGACY_TRADEOFF_MD = "reports/phase3_threshold_tradeoff_test_selected.md"

REPORT_SECTIONS = [
    ("Decisions (checkpoints)", [
        "reports/PHASE1_CHECKPOINT.md",
        "reports/PHASE2_CHECKPOINT.md",
        "reports/PHASE3_CHECKPOINT.md",
    ]),
    ("Phase 1 — Validation infrastructure", [
        "reports/phase1_cv_comparison.md",
        "reports/phase1_overfitting_analysis.md",
        "reports/phase1_smote_comparison.md",
        "reports/phase1_smote_overfitting.md",
    ]),
    ("Phase 2 — Statistical models", [
        "reports/phase2_model_comparison.md",
    ]),
    ("Phase 3 — Diagnostics and interpretation", [
        OOF_THRESHOLDS_SECTION,
        "reports/phase3_feature_importance.md",
        "reports/phase3_roc_analysis.md",
        "reports/phase3_predicted_vs_actual.md",
    ]),
]

OUTPUT_PATH = ROOT / "reports" / "CONSOLIDATED_REPORT.md"

METRIC_LABELS = ["Recall (micro)", "Precision (micro)", "F1 (micro)", "F1 (macro)"]


def run_all_phases() -> None:
    """Run the three phases in sequence (full reproduction)."""
    import subprocess

    for script in (
        "run_phase1_validation.py",
        "compare_statistical_models.py",
        "run_phase3_diagnostics.py",  # includes threshold_selection_oof.py (official thresholds)
    ):
        print(f"\n{'=' * 70}\nRunning {script}...\n{'=' * 70}")
        result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"{script} failed (exit code {result.returncode})")


def oof_thresholds_markdown() -> str | None:
    """Official thresholds section built from reports/oof_threshold_results.json."""
    if not OOF_RESULTS_PATH.exists():
        return None
    data = json.loads(OOF_RESULTS_PATH.read_text(encoding="utf-8"))
    off = data["official"]
    lines = [
        "## Official thresholds (out-of-fold selection, test set not used)",
        "",
        "Source: `threshold_selection_oof.py` → `reports/oof_threshold_results.json`. "
        "Per-defect thresholds selected on the out-of-fold predictions of the "
        "development set (5-fold CV) and applied, unchanged, to the test set.",
        "",
        f"- Official strategy: `{data['official_strategy']}`",
        f"- Threshold range (defects with positives in the test set): "
        f"{off['thr_min']:.2f} – {off['thr_max']:.2f}",
        "",
        "### Neural network — test set (95% bootstrap CI)",
        "",
        "| Metric | Value | 95% CI |",
        "|--------|-------|--------|",
    ]
    for label, pt, lo, hi in zip(METRIC_LABELS, off["point"], off["lo"], off["hi"]):
        lines.append(f"| {label} | {pt:.4f} | [{lo:.4f}, {hi:.4f}] |")

    conf = data.get("overall_confusion")
    if conf:
        lines += [
            "",
            f"Aggregated confusion matrix (28 defects × test set): TP = {conf['tp']}, "
            f"TN = {conf['tn']}, FP = {conf['fp']}, FN = {conf['fn']}.",
        ]

    dev = data.get("dev_f2_by_strategy_nn")
    if dev:
        lines += [
            "",
            "### F2-micro on the development set (OOF) by strategy — neural network",
            "",
            "| Strategy | F2-micro (OOF) |",
            "|----------|----------------|",
        ]
        for strategy, value in dev.items():
            lines.append(f"| {strategy} | {value:.4f} |")

    all_models = data.get("official_all_models")
    if all_models:
        lines += [
            "",
            "### All models — official strategy, test set",
            "",
            "| Model | " + " | ".join(METRIC_LABELS) + " |",
            "|-------|" + "|".join("---" for _ in METRIC_LABELS) + "|",
        ]
        for model, values in all_models.items():
            lines.append(f"| {model} | " + " | ".join(f"{v:.4f}" for v in values) + " |")

    lines += [
        "",
        f"*Legacy:* `{LEGACY_TRADEOFF_MD}` (threshold_tradeoff_analysis.py) contains the "
        "previous procedure, with thresholds chosen on the test set; it is not "
        "used in the official results.",
    ]
    return "\n".join(lines)


def build_report() -> None:
    parts = [
        "# Consolidated report — Adjustments requested by the review committee\n",
        f"Generated on {datetime.now():%Y-%m-%d %H:%M}. "
        "Pipeline: stratified 80/20 split (random_state=42), unified 5-fold "
        "CV, official configuration defined in Phase 1.\n",
        "---\n",
    ]
    for section_title, files in REPORT_SECTIONS:
        parts.append(f"\n# {section_title}\n")
        for file in files:
            if file == OOF_THRESHOLDS_SECTION:
                content = oof_thresholds_markdown()
                if content is None:
                    parts.append("\n> [PENDING] reports/oof_threshold_results.json not "
                                 "found — run threshold_selection_oof.py.\n")
                    continue
            else:
                path = ROOT / file
                if not path.exists():
                    parts.append(f"\n> [PENDING] {file} not found — "
                                 "run the corresponding phase.\n")
                    continue
                content = path.read_text(encoding="utf-8")
            # Demote headings by one level to fit the hierarchy
            content = "\n".join(
                ("#" + line) if line.startswith("#") else line
                for line in content.splitlines()
            )
            parts.append(content)
            parts.append("\n---\n")

    OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] Consolidated report: {OUTPUT_PATH}")


def main() -> None:
    if "--run-all" in sys.argv:
        run_all_phases()
    build_report()


if __name__ == "__main__":
    main()
