"""
Item 3.4 (Phase 3) — Predicted vs. actual plot (visual validation).

Review committee question: "Show predicted vs. actual as a validation of the
methodology."

Since the problem is multi-label binary classification (not regression), the
"predicted vs. actual" view is adapted into two complementary views:
  1. Calibration diagram (reliability diagram): predicted probability
     (bins) vs. observed defect frequency — aggregated (all defects) per
     model, and per frequent defect for the production model.
  2. Normalized confusion matrix per defect (most frequent defects),
     using the official thresholds defined in item 3.2.

Outputs:
  figures/phase3_calibration_by_model.png
  figures/phase3_calibration_top_defects_nn.png
  figures/phase3_confusion_normalized_nn.png
  figures/table_phase3_calibration_bins.csv
  reports/phase3_predicted_vs_actual.{json,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sklearn.calibration import calibration_curve

from phase3_utils import build_or_load_final_models, most_frequent_defects
from unified_cv_pipeline import MODEL_DISPLAY_NAMES, MODEL_PYTORCH, dataframe_to_markdown

OFFICIAL_THRESHOLDS_PATH = Path("reports/phase3_official_thresholds.json")
N_BINS = 10
TOP_DEFECTS = 3


def plot_calibration_by_model(
    cache: Dict[str, Any],
    path: str = "figures/phase3_calibration_by_model.png",
) -> pd.DataFrame:
    """Aggregated reliability diagram (28 defects stacked) per model."""
    y_test = cache["y_test"]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    bin_rows: List[Dict[str, Any]] = []
    for model_type, entry in cache["models"].items():
        y_true_flat = y_test.ravel()
        proba_flat = entry["test_proba"].ravel()
        label = MODEL_DISPLAY_NAMES.get(model_type, entry["model_name"])
        frac_pos, mean_pred = calibration_curve(
            y_true_flat, proba_flat, n_bins=N_BINS, strategy="uniform"
        )
        ax.plot(mean_pred, frac_pos, "-o", ms=4, label=label)
        for mp, fp_ in zip(mean_pred, frac_pos):
            bin_rows.append(
                {
                    "model": label,
                    "mean_predicted_prob": round(float(mp), 4),
                    "observed_positive_freq": round(float(fp_), 4),
                }
            )

    ax.set_xlabel("Mean predicted probability (bin center)")
    ax.set_ylabel("Observed defect frequency")
    ax.set_title("Predicted vs. Actual — aggregate calibration (28 defects, test set)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return pd.DataFrame(bin_rows)


def plot_calibration_top_defects_nn(
    cache: Dict[str, Any],
    path: str = "figures/phase3_calibration_top_defects_nn.png",
) -> None:
    """Reliability diagram per frequent defect, production model (NN)."""
    y_test = cache["y_test"]
    defect_names = cache["defect_names"]
    proba = cache["models"][MODEL_PYTORCH]["test_proba"]
    top_idx = most_frequent_defects(y_test, defect_names, TOP_DEFECTS)

    fig, axes = plt.subplots(1, len(top_idx), figsize=(5.5 * len(top_idx), 5))
    if len(top_idx) == 1:
        axes = [axes]
    for ax, defect_idx in zip(axes, top_idx):
        frac_pos, mean_pred = calibration_curve(
            y_test[:, defect_idx], proba[:, defect_idx],
            n_bins=N_BINS, strategy="uniform"
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(mean_pred, frac_pos, "-o", color="tab:blue")
        ax.set_title(defect_names[defect_idx].replace("_", " ").title())
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.grid(alpha=0.3)
    fig.suptitle("Calibration by defect — PyTorch NN (test set)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_confusion_normalized_nn(
    cache: Dict[str, Any],
    official: Dict[str, Any],
    path: str = "figures/phase3_confusion_normalized_nn.png",
) -> List[Dict[str, Any]]:
    """Row-normalized confusion matrices for the most frequent defects,
    using the official threshold from item 3.2."""
    y_test = cache["y_test"]
    defect_names = cache["defect_names"]
    proba = cache["models"][MODEL_PYTORCH]["test_proba"]
    strategy = official["official_strategy"]
    thresholds = official["thresholds_by_model"][MODEL_PYTORCH][strategy]
    top_idx = most_frequent_defects(y_test, defect_names, TOP_DEFECTS)

    fig, axes = plt.subplots(1, len(top_idx), figsize=(5.5 * len(top_idx), 5))
    if len(top_idx) == 1:
        axes = [axes]

    stats = []
    for ax, defect_idx in zip(axes, top_idx):
        defect = defect_names[defect_idx]
        th = float(thresholds[defect])
        y_true = y_test[:, defect_idx]
        y_pred = (proba[:, defect_idx] >= th).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        cm = np.array([[tn, fp], [fn, tp]], dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm_norm[r, c]:.2%}\n(n={int(cm[r, c])})",
                        ha="center", va="center",
                        color="white" if cm_norm[r, c] > 0.5 else "black",
                        fontsize=11)
        ax.set_xticks([0, 1], ["No defect", "Defect"])
        ax.set_yticks([0, 1], ["No defect", "Defect"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{defect.replace('_', ' ').title()}\n(threshold={th:.2f})")
        stats.append(
            {
                "defect": defect,
                "threshold": th,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else None,
                "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else None,
            }
        )

    fig.suptitle("Predicted vs. Actual — normalized confusion matrices (NN, official thresholds)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return stats


def main() -> None:
    print("=" * 70)
    print("PHASE 3.4 — PREDICTED × ACTUAL")
    print("=" * 70)
    Path("reports").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    cache = build_or_load_final_models(verbose=True)
    with open(OFFICIAL_THRESHOLDS_PATH, encoding="utf-8") as f:
        official = json.load(f)
    print(f"[*] Official strategy (3.2): {official['official_strategy_name']}")

    bins_df = plot_calibration_by_model(cache)
    bins_df.to_csv("figures/table_phase3_calibration_bins.csv", index=False)
    print("    [OK] figures/phase3_calibration_by_model.png")

    plot_calibration_top_defects_nn(cache)
    print("    [OK] figures/phase3_calibration_top_defects_nn.png")

    confusion_stats = plot_confusion_normalized_nn(cache, official)
    print("    [OK] figures/phase3_confusion_normalized_nn.png")

    md = "\n".join([
        "# Phase 3.4 — Predicted × Actual\n",
        "Adaptation for multi-label classification: (1) calibration diagrams "
        "(predicted probability vs. observed frequency) and (2) normalized "
        "confusion matrices with the official thresholds from item 3.2.\n",
        "## Confusion matrices (NN, official thresholds)\n",
        dataframe_to_markdown(pd.DataFrame(confusion_stats)),
        "",
        "![Calibration by model](phase3_calibration_by_model.png)",
        "![Calibration by defect](phase3_calibration_top_defects_nn.png)",
        "![Normalized confusion](phase3_confusion_normalized_nn.png)",
        "",
    ])
    Path("reports/phase3_predicted_vs_actual.md").write_text(md, encoding="utf-8")

    with open("reports/phase3_predicted_vs_actual.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "confusion_stats": confusion_stats,
                "calibration_bins": bins_df.to_dict(orient="records"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("[OK] reports/phase3_predicted_vs_actual.{json,md}")


if __name__ == "__main__":
    main()
