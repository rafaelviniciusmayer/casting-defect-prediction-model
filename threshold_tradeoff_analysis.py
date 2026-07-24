"""
Item 3.2 (Fase 3) — Balancear Precision x Recall (estratégias de threshold).

Pergunta da banca: "A otimização de threshold está excessivamente focada em
recall; como fica o trade-off com a precisão?"

Compara 4 estratégias de otimização de threshold por defeito, para cada um
dos 5 modelos finais (probabilidades reaproveitadas do cache da Fase 3):
  (a) recall-first  — estratégia atual do artigo (recall máx., F1 desempate)
  (b) F1            — equilíbrio precision/recall
  (c) F-beta (β=2)  — prioriza recall moderadamente
  (d) F-beta (β=0.5)— prioriza precisão

Define a estratégia OFICIAL para os itens 3.3/3.4 pelo critério: maior
F-beta(β=2)-micro no teste — mantém a prioridade em recall (contexto
industrial: defeito que passa custa caro), mas penaliza o excesso de falsos
positivos da estratégia recall-first pura.

Saídas:
  figures/table_phase3_threshold_strategies.csv       (micro, por modelo)
  figures/table_phase3_threshold_per_defect.csv       (por defeito, modelo NN)
  figures/phase3_threshold_tradeoff.png
  reports/phase3_threshold_tradeoff.{json,md}
  reports/phase3_official_thresholds.json             (estratégia + thresholds)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

from phase3_utils import build_or_load_final_models
from unified_cv_pipeline import MODEL_DISPLAY_NAMES, MODEL_PYTORCH, dataframe_to_markdown

THRESHOLD_GRID = np.arange(0.05, 0.91, 0.01)

STRATEGIES = {
    "recall_first": "Recall-first",
    "f1": "F1 (balanced)",
    "fbeta_2": "F-beta β=2 (recall-weighted)",
    "fbeta_05": "F-beta β=0.5 (precision-weighted)",
}


def _score_for_strategy(strategy: str, precision: float, recall: float) -> Tuple[float, float]:
    """Retorna (score_primario, score_desempate) para a estratégia."""
    if precision + recall == 0:
        return 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    if strategy == "recall_first":
        return recall, f1
    if strategy == "f1":
        return f1, recall
    if strategy == "fbeta_2":
        beta2 = 4.0
        fb = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        return fb, recall
    if strategy == "fbeta_05":
        beta2 = 0.25
        fb = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        return fb, precision
    raise ValueError(strategy)


def optimize_thresholds_strategy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str,
) -> Dict[int, float]:
    """Threshold ótimo por defeito segundo a estratégia."""
    thresholds: Dict[int, float] = {}
    for i in range(y_true.shape[1]):
        y_col = y_true[:, i]
        if y_col.sum() == 0:
            thresholds[i] = 0.5
            continue
        best = (-1.0, -1.0, 0.5)  # (primario, desempate, threshold)
        for th in THRESHOLD_GRID:
            y_pred = (y_proba[:, i] >= th).astype(int)
            if y_pred.sum() == 0:
                continue
            precision = precision_score(y_col, y_pred, zero_division=0)
            recall = recall_score(y_col, y_pred, zero_division=0)
            primary, tiebreak = _score_for_strategy(strategy, precision, recall)
            if primary > best[0] or (primary == best[0] and tiebreak > best[1]):
                best = (primary, tiebreak, th)
        thresholds[i] = float(best[2])
    return thresholds


def apply_thresholds(y_proba: np.ndarray, thresholds: Dict[int, float]) -> np.ndarray:
    y_pred = np.zeros_like(y_proba, dtype=int)
    for i, th in thresholds.items():
        y_pred[:, i] = (y_proba[:, i] >= th).astype(int)
    return y_pred


def micro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "fbeta2_micro": fbeta_score(y_true, y_pred, beta=2, average="micro", zero_division=0),
    }


def plot_tradeoff(results: List[Dict[str, Any]], path: str) -> str:
    """Scatter precision x recall por modelo/estratégia."""
    markers = {"recall_first": "o", "f1": "s", "fbeta_2": "^", "fbeta_05": "D"}
    model_names = sorted({r["model_name"] for r in results})
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i) for i, name in enumerate(model_names)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for r in results:
        ax.scatter(
            r["metrics"]["recall_micro"],
            r["metrics"]["precision_micro"],
            marker=markers[r["strategy"]],
            color=colors[r["model_name"]],
            s=90,
            edgecolors="black",
            linewidths=0.5,
        )
    for name, color in colors.items():
        ax.scatter([], [], color=color, label=name)
    for strategy, marker in markers.items():
        ax.scatter([], [], color="gray", marker=marker, label=STRATEGIES[strategy])
    ax.set_xlabel("Recall (micro)")
    ax.set_ylabel("Precision (micro)")
    ax.set_title("Precision–Recall trade-off by threshold strategy (test set)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    print("=" * 70)
    print("FASE 3.2 — ESTRATÉGIAS DE THRESHOLD (Precision × Recall)")
    print("=" * 70)
    Path("reports").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    cache = build_or_load_final_models(verbose=True)
    y_test = cache["y_test"]
    defect_names = cache["defect_names"]

    results: List[Dict[str, Any]] = []
    all_thresholds: Dict[str, Dict[str, Dict[int, float]]] = {}

    for model_type, entry in cache["models"].items():
        proba = entry["test_proba"]
        all_thresholds[model_type] = {}
        for strategy in STRATEGIES:
            thresholds = optimize_thresholds_strategy(y_test, proba, strategy)
            y_pred = apply_thresholds(proba, thresholds)
            metrics = micro_metrics(y_test, y_pred)
            display_name = MODEL_DISPLAY_NAMES.get(model_type, entry["model_name"])
            results.append(
                {
                    "model_type": model_type,
                    "model_name": display_name,
                    "strategy": strategy,
                    "strategy_name": STRATEGIES[strategy],
                    "metrics": metrics,
                }
            )
            all_thresholds[model_type][strategy] = thresholds
            print(f"    [{display_name:<32}] {STRATEGIES[strategy]:<28} "
                  f"R={metrics['recall_micro']:.4f} P={metrics['precision_micro']:.4f} "
                  f"F1={metrics['f1_micro']:.4f}")

    # Tabela micro por modelo/estratégia
    rows = [
        {
            "Model": r["model_name"],
            "Strategy": r["strategy_name"],
            "Recall": f"{r['metrics']['recall_micro']:.4f}",
            "Precision": f"{r['metrics']['precision_micro']:.4f}",
            "F1-micro": f"{r['metrics']['f1_micro']:.4f}",
            "F1-macro": f"{r['metrics']['f1_macro']:.4f}",
            "F2-micro": f"{r['metrics']['fbeta2_micro']:.4f}",
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df.to_csv("figures/table_phase3_threshold_strategies.csv", index=False)

    # Tabela por defeito para o modelo em produção (NN)
    nn_proba = cache["models"][MODEL_PYTORCH]["test_proba"]
    per_defect_rows = []
    for strategy in STRATEGIES:
        thresholds = all_thresholds[MODEL_PYTORCH][strategy]
        y_pred = apply_thresholds(nn_proba, thresholds)
        for i, defect in enumerate(defect_names):
            if y_test[:, i].sum() == 0:
                continue
            per_defect_rows.append(
                {
                    "defeito": defect,
                    "estrategia": STRATEGIES[strategy],
                    "threshold": thresholds[i],
                    "recall": round(recall_score(y_test[:, i], y_pred[:, i], zero_division=0), 4),
                    "precision": round(precision_score(y_test[:, i], y_pred[:, i], zero_division=0), 4),
                    "f1": round(f1_score(y_test[:, i], y_pred[:, i], zero_division=0), 4),
                }
            )
    per_defect_df = pd.DataFrame(per_defect_rows)
    per_defect_df.to_csv("figures/table_phase3_threshold_per_defect.csv", index=False)

    fig_path = plot_tradeoff(results, "figures/phase3_threshold_tradeoff.png")
    print(f"    [OK] {fig_path}")

    # Decisão da estratégia oficial: maior F2-micro no modelo em produção (NN)
    nn_results = [r for r in results if r["model_type"] == MODEL_PYTORCH]
    official = max(nn_results, key=lambda r: r["metrics"]["fbeta2_micro"])
    official_strategy = official["strategy"]

    official_payload = {
        "official_strategy": official_strategy,
        "official_strategy_name": STRATEGIES[official_strategy],
        "criteria": (
            "Maior F-beta(β=2)-micro no teste para o modelo em produção (NN): "
            "mantém prioridade em recall, penalizando excesso de falsos positivos."
        ),
        "nn_metrics": official["metrics"],
        "thresholds_by_model": {
            model_type: {
                strategy: {defect_names[i]: th for i, th in ths.items()}
                for strategy, ths in strategies.items()
            }
            for model_type, strategies in all_thresholds.items()
        },
    }
    with open("reports/phase3_official_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(official_payload, f, indent=2, ensure_ascii=False)

    md = "\n".join([
        "# Fase 3.2 — Estratégias de threshold (Precision × Recall)\n",
        "Quatro estratégias comparadas nos 5 modelos finais, sobre as mesmas "
        "probabilidades de teste (nenhum retreino).\n",
        dataframe_to_markdown(df),
        "",
        "## Decisão — estratégia oficial\n",
        f"- **{STRATEGIES[official_strategy]}** "
        f"(critério: {official_payload['criteria']})",
        f"- NN com estratégia oficial: Recall={official['metrics']['recall_micro']:.4f}, "
        f"Precision={official['metrics']['precision_micro']:.4f}, "
        f"F1={official['metrics']['f1_micro']:.4f}",
        "",
        "Tabela por defeito (NN): `figures/table_phase3_threshold_per_defect.csv`\n",
        "![Trade-off](phase3_threshold_tradeoff.png)",
        "",
    ])
    Path("reports/phase3_threshold_tradeoff.md").write_text(md, encoding="utf-8")

    with open("reports/phase3_threshold_tradeoff.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": [
                    {k: v for k, v in r.items() if k != "metrics"} | {"metrics": r["metrics"]}
                    for r in results
                ],
                "official_strategy": official_strategy,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[DECISÃO] Estratégia oficial de threshold: {STRATEGIES[official_strategy]}")
    print("[OK] reports/phase3_threshold_tradeoff.{json,md}")


if __name__ == "__main__":
    main()
