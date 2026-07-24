"""
Item 2.1 (Fase 2) — Comparação com modelos estatísticos e de regressão.

Pergunta da banca: "Por que só modelos de ML 'caixa-preta'? Comparar com
modelos estatísticos/de regressão clássicos."

Adiciona ao comparativo (além de PyTorch NN, XGBoost e Random Forest):
  - Regressão Logística L2/Ridge (class_weight='balanced') — baseline linear
    interpretável, via MultiOutputClassifier como XGBoost/RF.
  - Regressão Logística L1/Lasso — baseline estatístico regularizado formal.

Usa exatamente o mesmo pipeline da Fase 1 (unified_cv_pipeline):
mesmo split 80/20, mesma CV 5-fold, SMOTE conforme decisão do item 1.3
(lida automaticamente de reports/phase1_smote_comparison.json) e mesma
otimização de threshold por defeito (maximizar recall) na avaliação final.

Pré-requisito: rodar run_phase1_validation.py antes (gera a decisão do SMOTE
e as CVs dos 3 modelos originais, reaproveitadas aqui sem retreinar).

Execute: python compare_statistical_models.py

Saídas:
  reports/phase2_model_comparison.{json,md}
  reports/PHASE2_CHECKPOINT.md
  figures/table_phase2_model_comparison.csv  (equivalente à Tabela 8 do artigo)
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
    predict_proba_any,
    prepare_split_data,
    run_cross_validation,
    run_final_test_evaluation,
)

PHASE1_DECISION_PATH = Path("reports/phase1_smote_comparison.json")
PHASE1_CV_PATH = Path("reports/phase1_cv_comparison.json")

NEW_MODEL_TYPES = [MODEL_LOGISTIC_L2, MODEL_LOGISTIC_L1]


def load_phase1_smote_decision() -> bool:
    """Lê a decisão oficial da Fase 1 (manter SMOTE ou não)."""
    if not PHASE1_DECISION_PATH.exists():
        raise FileNotFoundError(
            "Execute run_phase1_validation.py antes: decisão do SMOTE "
            f"não encontrada em {PHASE1_DECISION_PATH}"
        )
    with open(PHASE1_DECISION_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return bool(payload["decision"]["keep_smote"])


def load_phase1_cv_stats(use_smote: bool) -> Dict[str, Dict[str, Any]]:
    """
    Reaproveita as CVs da Fase 1 para os 3 modelos originais, na condição
    oficial, evitando retreino.
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
    Treina o modelo final nos 80% e avalia no teste com a MESMA otimização de
    threshold por defeito do pipeline original (maximizar recall), além de
    medir tempo de treino e de inferência (ms por 100 amostras).
    """
    final = run_final_test_evaluation(model_type, split, use_smote=use_smote, verbose=True)

    X_test_scaled = final.scaler.transform(split.X_test).astype(np.float32)
    y_pred_proba = predict_proba_any(model_type, final.model, X_test_scaled)

    _, y_pred_binary = _optimize_thresholds_silent(
        split.y_test, y_pred_proba, split.defect_names
    )
    optimized_metrics = {
        "recall_micro": recall_score(split.y_test, y_pred_binary, average="micro", zero_division=0),
        "precision_micro": precision_score(split.y_test, y_pred_binary, average="micro", zero_division=0),
        "f1_micro": f1_score(split.y_test, y_pred_binary, average="micro", zero_division=0),
        "f1_macro": f1_score(split.y_test, y_pred_binary, average="macro", zero_division=0),
    }

    # Tempo de inferência: 100 repetições sobre 100 amostras
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
    """Tabela equivalente à Tabela 8 do artigo, agora com 5 modelos."""
    rows: List[Dict[str, Any]] = []
    for model_type, final in final_evals.items():
        cv = cv_stats.get(model_type)
        opt = final["test_metrics_optimized_threshold"]
        rows.append(
            {
                "Modelo": final["model_name"],
                "Recall (teste)": f"{opt['recall_micro']:.4f}",
                "Precision (teste)": f"{opt['precision_micro']:.4f}",
                "F1-micro (teste)": f"{opt['f1_micro']:.4f}",
                "F1-macro (teste)": f"{opt['f1_macro']:.4f}",
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
                "Treino (s)": f"{final['train_time_sec']:.1f}",
                "Inferência (ms/100)": f"{final['inference_time_ms_per_100']:.2f}",
            }
        )
    return pd.DataFrame(rows)


def build_phase2_markdown(df: pd.DataFrame, use_smote: bool) -> str:
    smote_label = "com SMOTE" if use_smote else "sem SMOTE (apenas cost-sensitive)"
    lines = [
        "# Fase 2.1 — Comparação com modelos estatísticos e de regressão\n",
        f"Configuração oficial da Fase 1: **{smote_label}**. "
        "Todos os modelos usam o mesmo split 80/20, a mesma CV 5-fold, o mesmo "
        "pré-processamento e a mesma otimização de threshold por defeito "
        "(maximizar recall) na avaliação final de teste.\n",
        "Baselines estatísticos adicionados: Regressão Logística L2 (Ridge) e "
        "L1 (Lasso), ambas com `class_weight='balanced'` via `MultiOutputClassifier`.\n",
        dataframe_to_markdown(df),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    t_start = time.perf_counter()
    print("=" * 70)
    print("FASE 2 — COMPARAÇÃO COM MODELOS ESTATÍSTICOS")
    print("=" * 70)

    Path("reports").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    use_smote = load_phase1_smote_decision()
    print(f"\n[*] Decisão da Fase 1: SMOTE {'MANTIDO' if use_smote else 'REMOVIDO'} "
          f"(condição oficial: {'com' if use_smote else 'sem'} SMOTE)")

    print("\n[*] Carregando dados (pipeline existente)...")
    data = load_pipeline_data(verbose=True)
    split = prepare_split_data(data, verbose=True)

    # CV apenas para os modelos NOVOS; os 3 originais reaproveitam a Fase 1
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

    # Avaliação final (teste, thresholds otimizados) para os 5 modelos
    all_model_types = available_model_types() + NEW_MODEL_TYPES
    final_evals = {}
    for model_type in all_model_types:
        final_evals[model_type] = evaluate_final_with_thresholds(
            model_type, split, use_smote=use_smote
        )
        opt = final_evals[model_type]["test_metrics_optimized_threshold"]
        print(f"    [{MODEL_DISPLAY_NAMES[model_type]}] "
              f"Recall={opt['recall_micro']:.4f}, F1={opt['f1_micro']:.4f} "
              f"(thresholds otimizados)")

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

    best_recall = max(
        final_evals.values(),
        key=lambda e: e["test_metrics_optimized_threshold"]["recall_micro"],
    )
    best_f1 = max(
        final_evals.values(),
        key=lambda e: e["test_metrics_optimized_threshold"]["f1_micro"],
    )
    checkpoint = "\n".join(
        [
            "# Checkpoint — Fase 2 (modelos estatísticos)\n",
            f"- Condição oficial: {'com' if use_smote else 'sem'} SMOTE "
            "(herdada da Fase 1).",
            "- Modelos comparados: PyTorch NN, XGBoost, Random Forest, "
            "Regressão Logística L2 (Ridge), Regressão Logística L1 (Lasso).",
            f"- Melhor recall no teste (thresholds otimizados): "
            f"**{best_recall['model_name']}** "
            f"({best_recall['test_metrics_optimized_threshold']['recall_micro']:.4f}).",
            f"- Melhor F1-micro no teste: **{best_f1['model_name']}** "
            f"({best_f1['test_metrics_optimized_threshold']['f1_micro']:.4f}).",
            "- A Fase 3 deve usar estes 4-5 modelos finais para importância de "
            "features (3.1) e as probabilidades de teste para thresholds/ROC "
            "(3.2/3.3).",
            "",
        ]
    )
    Path("reports/PHASE2_CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"[CONCLUÍDO] Fase 2 executada em {elapsed / 60:.1f} min")
    print("=" * 70)
    print(df.to_string(index=False))
    print("\nCheckpoint: reports/PHASE2_CHECKPOINT.md")


if __name__ == "__main__":
    main()
