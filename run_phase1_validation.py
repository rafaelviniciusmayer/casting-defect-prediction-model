"""
Fase 1 — Infraestrutura de validação (ajustes solicitados pela banca)
=====================================================================

Orquestra os itens da Fase 1, executando cada CV uma única vez por
(modelo, condição) e reutilizando os resultados entre as análises:

  1.1 CV estratificada unificada (5 folds) para PyTorch NN, XGBoost e
      Random Forest — mesma CV da rede neural, SMOTE só no fold de treino.
  1.2 Análise formal de overfitting: Treino (CV) vs Validação (CV) vs
      Teste Final (20% isolado), por modelo (overfitting_analysis.py).
  1.3 Ablation com/sem SMOTE, cost-sensitive sempre ativo (smote_ablation.py).
  1.4 Overfitting causado pelo SMOTE: treino pós-SMOTE vs teste + curvas
      de aprendizado (smote_overfitting_check.py).

Execute: python run_phase1_validation.py

Saídas:
  reports/phase1_cv_comparison.{json,md}           (1.1)
  reports/phase1_overfitting_analysis.{json,md}    (1.2)
  reports/phase1_smote_comparison.{json,md}        (1.3)
  reports/phase1_smote_overfitting.{json,md}       (1.4)
  reports/PHASE1_CHECKPOINT.md                     (decisões da fase)
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

# Console do Windows usa cp1252 por padrão; garante saída UTF-8 (Δ, ±, −)
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
        "# Fase 1.1 — Comparação por Cross-Validation (5 folds)\n",
        "Métricas médias ± desvio-padrão na **validação** de cada fold "
        "(threshold fixo 0.5). SMOTE aplicado apenas no treino de cada fold; "
        "teste final (20%) permanece isolado.\n",
        "| Modelo | Recall (val) | Precision (val) | F1-micro (val) | F1-macro (val) | Tempo CV (s) |",
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
    smote_status = "MANTIDO" if decision["keep_smote"] else "REMOVIDO"
    best_recall = max(
        cv_payload["models"], key=lambda m: m["val_mean"]["recall_micro"]
    )
    best_f1 = max(cv_payload["models"], key=lambda m: m["val_mean"]["f1_micro"])
    lines = [
        "# Checkpoint — Fase 1 (Infraestrutura de validação)\n",
        "## Decisões tomadas\n",
        f"- **SMOTE: {smote_status}** na configuração oficial do pipeline "
        f"(ganho médio F1-micro {decision['avg_f1_micro_gain']:+.4f}, "
        f"recall {decision['avg_recall_gain']:+.4f}, "
        f"aumento de gap treino-val {decision['avg_train_val_gap_increase']:+.4f}).",
        "- **CV unificada**: os 3 modelos passam a usar a mesma CV estratificada "
        "em 5 folds (random_state=42) sobre o dev set (80%), com SMOTE "
        "(quando habilitado) aplicado apenas no fold de treino.",
        "- **Cost-sensitive learning** ativo em todos os modelos: pos_weight "
        "na NN, class_weight='balanced' no RF, scale_pos_weight no XGBoost "
        "(novidade desta fase — antes o XGBoost não tinha ponderação).",
        f"- Melhor recall (validação CV, com SMOTE): **{best_recall['model_name']}** "
        f"({best_recall['val_mean']['recall_micro']:.4f}).",
        f"- Melhor F1-micro (validação CV, com SMOTE): **{best_f1['model_name']}** "
        f"({best_f1['val_mean']['f1_micro']:.4f}).",
        "",
        "## Referência para as próximas fases\n",
        "- Fase 2 (novos modelos) deve usar `unified_cv_pipeline.run_cross_validation` "
        f"com `use_smote={decision['keep_smote']}`.",
        "- Fase 3 (thresholds/ROC) deve partir dos modelos finais treinados via "
        "`unified_cv_pipeline.run_final_test_evaluation` na mesma condição.",
        "",
        "## Arquivos gerados\n",
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
    print("FASE 1 — INFRAESTRUTURA DE VALIDAÇÃO")
    print("=" * 70)

    Path("reports").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    print("\n[*] Carregando dados (pipeline existente)...")
    data = load_pipeline_data(verbose=True)
    split = prepare_split_data(data, verbose=True)
    model_types = available_model_types()

    # ------------------------------------------------------------------
    # Treinos: cada CV roda UMA vez por (modelo, condição) e é reutilizada
    # ------------------------------------------------------------------
    cv_with: Dict[str, Any] = {}
    cv_without: Dict[str, Any] = {}
    final_with: Dict[str, Any] = {}

    print("\n" + "=" * 70)
    print("TREINOS — CV 5-fold por modelo e condição (com/sem SMOTE)")
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
    # 1.1 — Comparação por CV unificada (condição oficial atual: com SMOTE)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FASE 1.1 — Relatório de CV unificada")
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
    # 1.2 — Análise formal de overfitting
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FASE 1.2 — Análise formal de overfitting")
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
    # 1.3 — Ablation SMOTE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FASE 1.3 — Contribuição do SMOTE (com vs sem)")
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
    print(f"\n[DECISÃO] SMOTE {'mantido' if decision['keep_smote'] else 'removido'} "
          f"na configuração oficial.")
    print("[OK] reports/phase1_smote_comparison.{json,md}")

    # ------------------------------------------------------------------
    # 1.4 — Overfitting causado pelo SMOTE (treino pós-SMOTE vs teste + curvas)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FASE 1.4 — Overfitting causado pelo SMOTE")
    print("=" * 70)
    train_vs_test_df = build_train_vs_test_table(final_with)
    train_vs_test_df.to_csv("figures/table_phase1_train_vs_test.csv", index=False)
    print(train_vs_test_df.to_string(index=False))

    print("\n[*] Curvas de aprendizado (fold 1 como validação fixa)...")
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
    # Checkpoint da Fase 1
    # ------------------------------------------------------------------
    checkpoint_md = _build_checkpoint_markdown(decision, cv_payload)
    _write_text("reports/PHASE1_CHECKPOINT.md", checkpoint_md)

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"[CONCLUÍDO] Fase 1 executada em {elapsed / 60:.1f} min")
    print("=" * 70)
    print("Checkpoint: reports/PHASE1_CHECKPOINT.md")


if __name__ == "__main__":
    main()
