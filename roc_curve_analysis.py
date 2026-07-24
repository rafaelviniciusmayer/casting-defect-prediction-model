"""
Item 3.3 (Fase 3) — Curvas ROC e AUC.

Pergunta da banca (Leonardo): "Como o threshold escolhido se relaciona com o
trade-off TPR/FPR? Reportar curva ROC e AUC."

Reaproveita as probabilidades de teste do cache da Fase 3 e os thresholds da
estratégia oficial definida no item 3.2 (reports/phase3_official_thresholds.json).

Gera:
  - Curva ROC por defeito para os defeitos mais frequentes, com o ponto de
    operação do threshold oficial marcado.
  - Curva ROC micro-average agregada por modelo.
  - Tabela de AUC-ROC por defeito (todos os 28) e por modelo.

Saídas:
  figures/phase3_roc_curves_top_defects.png
  figures/phase3_roc_micro_average.png
  figures/table_phase3_auc_by_defect.csv
  reports/phase3_roc_analysis.{json,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import auc, roc_auc_score, roc_curve

from phase3_utils import build_or_load_final_models, most_frequent_defects
from unified_cv_pipeline import MODEL_DISPLAY_NAMES, MODEL_PYTORCH, dataframe_to_markdown

OFFICIAL_THRESHOLDS_PATH = Path("reports/phase3_official_thresholds.json")
TOP_DEFECTS_FOR_PLOT = 3


def load_official_thresholds() -> Dict[str, Any]:
    if not OFFICIAL_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            "Execute threshold_tradeoff_analysis.py antes (item 3.2): "
            f"{OFFICIAL_THRESHOLDS_PATH} não encontrado"
        )
    with open(OFFICIAL_THRESHOLDS_PATH, encoding="utf-8") as f:
        return json.load(f)


def plot_roc_top_defects(
    cache: Dict[str, Any],
    official: Dict[str, Any],
    path: str = "figures/phase3_roc_curves_top_defects.png",
) -> str:
    """ROC por defeito (defeitos mais frequentes) para o modelo em produção (NN),
    com o ponto de operação do threshold oficial marcado."""
    y_test = cache["y_test"]
    defect_names = cache["defect_names"]
    top_idx = most_frequent_defects(y_test, defect_names, TOP_DEFECTS_FOR_PLOT)

    strategy = official["official_strategy"]
    nn_entry = cache["models"][MODEL_PYTORCH]
    proba = nn_entry["test_proba"]
    nn_thresholds = official["thresholds_by_model"][MODEL_PYTORCH][strategy]

    fig, axes = plt.subplots(1, len(top_idx), figsize=(6 * len(top_idx), 5.5))
    if len(top_idx) == 1:
        axes = [axes]

    for ax, defect_idx in zip(axes, top_idx):
        defect = defect_names[defect_idx]
        y_true = y_test[:, defect_idx]
        y_score = proba[:, defect_idx]

        fpr, tpr, ths = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="tab:blue", lw=2,
                label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatório")

        # Ponto de operação do threshold oficial
        th_official = float(nn_thresholds[defect])
        y_pred = (y_score >= th_official).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        tpr_op = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_op = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ax.scatter([fpr_op], [tpr_op], color="tab:red", s=120, zorder=5,
                   label=f"Official threshold = {th_official:.2f}\n"
                         f"(TPR={tpr_op:.3f}, FPR={fpr_op:.3f})")

        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title(defect.replace("_", " ").title())
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("ROC curves — most frequent defects (PyTorch NN, test set)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_roc_micro_average(
    cache: Dict[str, Any],
    path: str = "figures/phase3_roc_micro_average.png",
) -> Dict[str, float]:
    """ROC micro-average agregada, uma curva por modelo."""
    y_test = cache["y_test"]
    fig, ax = plt.subplots(figsize=(8, 7))
    micro_aucs: Dict[str, float] = {}

    for model_type, entry in cache["models"].items():
        proba = entry["test_proba"]
        fpr, tpr, _ = roc_curve(y_test.ravel(), proba.ravel())
        roc_auc = auc(fpr, tpr)
        label = MODEL_DISPLAY_NAMES.get(model_type, entry["model_name"])
        micro_aucs[label] = float(roc_auc)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("Micro-average ROC — 28 defects aggregated (test set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return micro_aucs


def build_auc_table(cache: Dict[str, Any]) -> pd.DataFrame:
    """AUC-ROC por defeito × modelo."""
    y_test = cache["y_test"]
    defect_names = cache["defect_names"]
    counts = y_test.sum(axis=0).astype(int)

    rows = []
    for i, defect in enumerate(defect_names):
        row: Dict[str, Any] = {
            "defect": defect,
            "test_positives": int(counts[i]),
        }
        for model_type, entry in cache["models"].items():
            col = MODEL_DISPLAY_NAMES.get(model_type, entry["model_name"])
            if counts[i] == 0:
                row[col] = None
            else:
                row[col] = round(
                    roc_auc_score(y_test[:, i], entry["test_proba"][:, i]), 4
                )
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("test_positives", ascending=False)
    return df.reset_index(drop=True)


def main() -> None:
    print("=" * 70)
    print("FASE 3.3 — CURVAS ROC E AUC")
    print("=" * 70)
    Path("reports").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    cache = build_or_load_final_models(verbose=True)
    official = load_official_thresholds()
    print(f"[*] Estratégia oficial (3.2): {official['official_strategy_name']}")

    fig1 = plot_roc_top_defects(cache, official)
    print(f"    [OK] {fig1}")
    micro_aucs = plot_roc_micro_average(cache)
    print(f"    [OK] figures/phase3_roc_micro_average.png")

    auc_df = build_auc_table(cache)
    auc_df.to_csv("figures/table_phase3_auc_by_defect.csv", index=False)

    model_cols = [
        MODEL_DISPLAY_NAMES.get(mt, cache["models"][mt]["model_name"])
        for mt in cache["models"]
    ]
    mean_aucs = {col: round(float(auc_df[col].dropna().mean()), 4) for col in model_cols}

    md = "\n".join([
        "# Fase 3.3 — Curvas ROC e AUC\n",
        f"Estratégia oficial de threshold (item 3.2): "
        f"**{official['official_strategy_name']}** — ponto de operação marcado "
        "nas curvas dos defeitos mais frequentes.\n",
        "## AUC micro-average por modelo\n",
        dataframe_to_markdown(pd.DataFrame([
            {"Model": name, "AUC micro": f"{v:.4f}",
             "Mean AUC per defect": f"{mean_aucs[name]:.4f}"}
            for name, v in micro_aucs.items()
        ])),
        "\n## AUC-ROC por defeito (10 mais frequentes)\n",
        dataframe_to_markdown(auc_df.head(10)),
        "",
        "Tabela completa: `figures/table_phase3_auc_by_defect.csv`\n",
        "![ROC defeitos frequentes](phase3_roc_curves_top_defects.png)",
        "![ROC micro-average](phase3_roc_micro_average.png)",
        "",
    ])
    Path("reports/phase3_roc_analysis.md").write_text(md, encoding="utf-8")

    with open("reports/phase3_roc_analysis.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "micro_average_auc": micro_aucs,
                "mean_auc_by_defect": mean_aucs,
                "auc_by_defect": auc_df.to_dict(orient="records"),
                "official_strategy": official["official_strategy"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n[*] AUC micro-average:")
    for name, v in micro_aucs.items():
        print(f"    {name:<35} {v:.4f}")
    print("[OK] reports/phase3_roc_analysis.{json,md}")


if __name__ == "__main__":
    main()
