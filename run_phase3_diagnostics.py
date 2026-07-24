"""
Fase 3 — Diagnósticos e interpretação (ajustes solicitados pela banca)
======================================================================

Orquestra os itens da Fase 3 na ordem correta:
  3.1 Importância de features (feature_importance_analysis.py)
  3.2 Estratégias de threshold Precision × Recall (threshold_tradeoff_analysis.py)
  3.3 Curvas ROC e AUC (roc_curve_analysis.py) — usa thresholds do 3.2
  3.4 Previsto × realizado (predicted_vs_actual.py) — usa thresholds do 3.2

Os 5 modelos finais são treinados UMA única vez (phase3_utils, config oficial
da Fase 1) e cacheados em models/phase3_final_models.pkl; os 4 itens
reutilizam o cache.

Execute: python run_phase3_diagnostics.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import feature_importance_analysis
import predicted_vs_actual
import roc_curve_analysis
import threshold_tradeoff_analysis
from phase3_utils import build_or_load_final_models


def main() -> None:
    t_start = time.perf_counter()
    print("=" * 70)
    print("FASE 3 — DIAGNÓSTICOS E INTERPRETAÇÃO")
    print("=" * 70)

    # Garante que o cache de modelos finais existe (treina se necessário)
    build_or_load_final_models(verbose=True)

    threshold_tradeoff_analysis.main()   # 3.2 primeiro: define thresholds oficiais
    feature_importance_analysis.main()   # 3.1 (independe de threshold)
    roc_curve_analysis.main()            # 3.3 (usa thresholds do 3.2)
    predicted_vs_actual.main()           # 3.4 (usa thresholds do 3.2)

    # Checkpoint da fase
    import json
    with open("reports/phase3_official_thresholds.json", encoding="utf-8") as f:
        official = json.load(f)
    with open("reports/phase3_feature_importance.json", encoding="utf-8") as f:
        importance = json.load(f)

    checkpoint = "\n".join([
        "# Checkpoint — Fase 3 (diagnósticos e interpretação)\n",
        f"- **Estratégia oficial de threshold: {official['official_strategy_name']}** "
        f"({official['criteria']})",
        f"- Importância de features: {importance['answer']}",
        "- Curvas ROC/AUC e previsto×realizado gerados com os thresholds oficiais.",
        "",
        "## Arquivos gerados\n",
        "- `reports/phase3_feature_importance.md` (3.1)",
        "- `reports/phase3_threshold_tradeoff.md` (3.2)",
        "- `reports/phase3_roc_analysis.md` (3.3)",
        "- `reports/phase3_predicted_vs_actual.md` (3.4)",
        "- Figuras e CSVs correspondentes em `figures/`",
        "",
    ])
    Path("reports/PHASE3_CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"[CONCLUÍDO] Fase 3 executada em {elapsed / 60:.1f} min")
    print("=" * 70)
    print("Checkpoint: reports/PHASE3_CHECKPOINT.md")


if __name__ == "__main__":
    main()
