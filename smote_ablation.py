"""
Item 1.3 — Resultados com e sem SMOTE (ablation study).

Pergunta da banca: "Qual é a real contribuição do SMOTE? A combinação
SMOTE + cost-sensitive learning é de fato melhor que cost-sensitive sozinho?"

Compara, para cada modelo, duas condições no MESMO pipeline de CV (5 folds):
  (a) sem SMOTE — apenas cost-sensitive learning
      (pos_weight na NN, class_weight='balanced' no RF, scale_pos_weight no XGBoost)
  (b) com SMOTE — SMOTE dentro do fold de treino + cost-sensitive learning

Métricas comparadas: recall, precisão, F1-micro, F1-macro e tempo de treino.

A decisão (manter ou remover SMOTE da configuração oficial) é registrada no
checkpoint da Fase 1, gerado por run_phase1_validation.py.

Este módulo NÃO treina nada: consome CVResults já calculados pelo pipeline
unificado, orquestrados por run_phase1_validation.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from unified_cv_pipeline import (
    CVResult,
    METRIC_KEYS,
    dataframe_to_markdown,
    format_mean_std,
)


def build_smote_comparison_payload(
    cv_results_without: Dict[str, CVResult],
    cv_results_with: Dict[str, CVResult],
) -> Dict[str, Any]:
    """Monta o payload da comparação a partir de CVResults já computados."""
    comparison = []
    for model_type, cv_with in cv_results_with.items():
        cv_without = cv_results_without[model_type]

        def _condition_dict(cv: CVResult) -> Dict[str, Any]:
            return {
                "use_smote": cv.use_smote,
                "cv_val_mean": cv.val_mean,
                "cv_val_std": cv.val_std,
                "cv_train_mean": cv.train_mean,
                "total_train_time_sec": cv.total_train_time_sec,
            }

        without = _condition_dict(cv_without)
        with_sm = _condition_dict(cv_with)
        delta = {
            key: with_sm["cv_val_mean"][key] - without["cv_val_mean"][key]
            for key in METRIC_KEYS
        }
        delta["train_time_sec"] = (
            with_sm["total_train_time_sec"] - without["total_train_time_sec"]
        )

        comparison.append(
            {
                "model_type": model_type,
                "model_name": cv_with.model_name,
                "without_smote": without,
                "with_smote": with_sm,
                "delta_with_minus_without": delta,
            }
        )
    return {"models": comparison}


def build_smote_comparison_table(payload: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model in payload["models"]:
        name = model["model_name"]
        for condition_key, condition_label in (
            ("without_smote", "No SMOTE"),
            ("with_smote", "With SMOTE"),
        ):
            data = model[condition_key]
            vm = data["cv_val_mean"]
            vs = data["cv_val_std"]
            rows.append(
                {
                    "Model": name,
                    "Condition": condition_label,
                    "Recall (CV)": format_mean_std(vm["recall_micro"], vs["recall_micro"]),
                    "Precision (CV)": format_mean_std(
                        vm["precision_micro"], vs["precision_micro"]
                    ),
                    "F1-micro (CV)": format_mean_std(vm["f1_micro"], vs["f1_micro"]),
                    "F1-macro (CV)": format_mean_std(vm["f1_macro"], vs["f1_macro"]),
                    "CV Time (s)": f"{data['total_train_time_sec']:.1f}",
                }
            )
        delta = model["delta_with_minus_without"]
        rows.append(
            {
                "Model": name,
                "Condition": "Delta (With - No)",
                "Recall (CV)": f"{delta['recall_micro']:+.4f}",
                "Precision (CV)": f"{delta['precision_micro']:+.4f}",
                "F1-micro (CV)": f"{delta['f1_micro']:+.4f}",
                "F1-macro (CV)": f"{delta['f1_macro']:+.4f}",
                "CV Time (s)": f"{delta['train_time_sec']:+.1f}",
            }
        )
    return pd.DataFrame(rows)


def build_smote_recommendations(payload: Dict[str, Any]) -> List[str]:
    """Heurística para recomendar manter ou remover SMOTE, por modelo."""
    recommendations = []
    for model in payload["models"]:
        delta = model["delta_with_minus_without"]
        recall_gain = delta["recall_micro"]
        f1_gain = delta["f1_micro"]
        gap_with = (
            model["with_smote"]["cv_train_mean"]["f1_micro"]
            - model["with_smote"]["cv_val_mean"]["f1_micro"]
        )
        gap_without = (
            model["without_smote"]["cv_train_mean"]["f1_micro"]
            - model["without_smote"]["cv_val_mean"]["f1_micro"]
        )
        overfit_increase = gap_with - gap_without

        if f1_gain >= 0.01 or recall_gain >= 0.02:
            if overfit_increase <= 0.05:
                rec = (
                    f"**{model['model_name']}**: manter SMOTE "
                    f"(ΔF1={f1_gain:+.4f}, ΔRecall={recall_gain:+.4f}, "
                    f"aumento de gap treino-val={overfit_increase:+.4f})."
                )
            else:
                rec = (
                    f"**{model['model_name']}**: SMOTE melhora métricas "
                    f"(ΔF1={f1_gain:+.4f}), mas aumenta o gap treino-val "
                    f"({overfit_increase:+.4f}) — avaliar trade-off."
                )
        else:
            rec = (
                f"**{model['model_name']}**: SMOTE sem ganho relevante "
                f"(ΔF1={f1_gain:+.4f}, ΔRecall={recall_gain:+.4f}) — "
                f"considerar apenas cost-sensitive learning."
            )
        recommendations.append(rec)
    return recommendations


def decide_official_smote(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decisão global da Fase 1: manter SMOTE na configuração oficial?

    Critério: SMOTE é mantido se, em média entre os modelos, trouxer ganho
    de F1-micro >= 0.005 ou de recall >= 0.01 na validação da CV, sem
    aumento médio do gap treino-val acima de 0.05.
    """
    f1_gains = []
    recall_gains = []
    gap_increases = []
    for model in payload["models"]:
        delta = model["delta_with_minus_without"]
        f1_gains.append(delta["f1_micro"])
        recall_gains.append(delta["recall_micro"])
        gap_with = (
            model["with_smote"]["cv_train_mean"]["f1_micro"]
            - model["with_smote"]["cv_val_mean"]["f1_micro"]
        )
        gap_without = (
            model["without_smote"]["cv_train_mean"]["f1_micro"]
            - model["without_smote"]["cv_val_mean"]["f1_micro"]
        )
        gap_increases.append(gap_with - gap_without)

    avg_f1_gain = sum(f1_gains) / len(f1_gains)
    avg_recall_gain = sum(recall_gains) / len(recall_gains)
    avg_gap_increase = sum(gap_increases) / len(gap_increases)

    keep_smote = (
        (avg_f1_gain >= 0.005 or avg_recall_gain >= 0.01)
        and avg_gap_increase <= 0.05
    )

    return {
        "keep_smote": bool(keep_smote),
        "avg_f1_micro_gain": avg_f1_gain,
        "avg_recall_gain": avg_recall_gain,
        "avg_train_val_gap_increase": avg_gap_increase,
        "criteria": (
            "Manter SMOTE se ganho médio de F1-micro >= 0.005 ou de recall >= 0.01 "
            "na validação (CV), com aumento médio de gap treino-val <= 0.05."
        ),
    }


def build_smote_markdown(
    df: pd.DataFrame,
    recommendations: List[str],
    decision: Dict[str, Any],
) -> str:
    decision_label = (
        "**SMOTE MANTIDO** na configuração oficial"
        if decision["keep_smote"]
        else "**SMOTE REMOVIDO** — configuração oficial passa a usar apenas cost-sensitive learning"
    )
    lines = [
        "# Fase 1.3 — Contribuição do SMOTE (ablation)\n",
        "Comparação lado a lado usando o mesmo pipeline de CV (5 folds). "
        "Em ambas as condições, cost-sensitive learning permanece ativo "
        "(pos_weight na NN, class_weight='balanced' no RF, scale_pos_weight no XGBoost).\n",
        dataframe_to_markdown(df),
        "",
        "## Recomendações por modelo\n",
    ]
    lines.extend(f"- {r}" for r in recommendations)
    lines += [
        "",
        "## Decisão da Fase 1\n",
        f"- {decision_label}",
        f"- Ganho médio de F1-micro: {decision['avg_f1_micro_gain']:+.4f}",
        f"- Ganho médio de recall: {decision['avg_recall_gain']:+.4f}",
        f"- Aumento médio do gap treino-val: {decision['avg_train_val_gap_increase']:+.4f}",
        f"- Critério: {decision['criteria']}",
        "",
    ]
    return "\n".join(lines)
