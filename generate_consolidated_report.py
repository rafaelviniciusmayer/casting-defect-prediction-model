"""
Relatório consolidado — Ajustes solicitados pela banca (Fases 1-3)
==================================================================

Concatena os relatórios markdown das três fases em um único documento
(reports/RELATORIO_CONSOLIDADO_BANCA.md), na ordem das fases, com os
checkpoints de decisão no início.

Também oferece --run-all para executar as três fases em sequência antes de
consolidar (útil para reproduzir tudo do zero):
    python generate_consolidated_report.py --run-all
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPORT_SECTIONS = [
    ("Decisões (checkpoints)", [
        "reports/PHASE1_CHECKPOINT.md",
        "reports/PHASE2_CHECKPOINT.md",
        "reports/PHASE3_CHECKPOINT.md",
    ]),
    ("Fase 1 — Infraestrutura de validação", [
        "reports/phase1_cv_comparison.md",
        "reports/phase1_overfitting_analysis.md",
        "reports/phase1_smote_comparison.md",
        "reports/phase1_smote_overfitting.md",
    ]),
    ("Fase 2 — Modelos estatísticos", [
        "reports/phase2_model_comparison.md",
    ]),
    ("Fase 3 — Diagnósticos e interpretação", [
        "reports/phase3_threshold_tradeoff.md",
        "reports/phase3_feature_importance.md",
        "reports/phase3_roc_analysis.md",
        "reports/phase3_predicted_vs_actual.md",
    ]),
]

OUTPUT_PATH = Path("reports/RELATORIO_CONSOLIDADO_BANCA.md")


def run_all_phases() -> None:
    """Executa as três fases em sequência (reprodução completa)."""
    import subprocess

    for script in (
        "run_phase1_validation.py",
        "compare_statistical_models.py",
        "run_phase3_diagnostics.py",
    ):
        print(f"\n{'=' * 70}\nExecutando {script}...\n{'=' * 70}")
        result = subprocess.run([sys.executable, "-X", "utf8", script])
        if result.returncode != 0:
            raise RuntimeError(f"{script} falhou (exit code {result.returncode})")


def build_report() -> None:
    parts = [
        "# Relatório consolidado — Ajustes solicitados pela banca\n",
        f"Gerado em {datetime.now():%Y-%m-%d %H:%M}. "
        "Pipeline: split 80/20 estratificado (random_state=42), CV 5-fold "
        "unificada, configuração oficial definida na Fase 1.\n",
        "---\n",
    ]
    for section_title, files in REPORT_SECTIONS:
        parts.append(f"\n# {section_title}\n")
        for file in files:
            path = Path(file)
            if not path.exists():
                parts.append(f"\n> [PENDENTE] {file} não encontrado — "
                             "execute a fase correspondente.\n")
                continue
            content = path.read_text(encoding="utf-8")
            # Rebaixa títulos em um nível para caber na hierarquia
            content = "\n".join(
                ("#" + line) if line.startswith("#") else line
                for line in content.splitlines()
            )
            parts.append(content)
            parts.append("\n---\n")

    OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] Relatório consolidado: {OUTPUT_PATH}")


def main() -> None:
    if "--run-all" in sys.argv:
        run_all_phases()
    build_report()


if __name__ == "__main__":
    main()
