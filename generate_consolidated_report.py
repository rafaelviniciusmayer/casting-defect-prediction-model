"""
Relatório consolidado — Ajustes solicitados pela banca (Fases 1-3)
==================================================================

Concatena os relatórios markdown das três fases em um único documento
(reports/RELATORIO_CONSOLIDADO_BANCA.md), na ordem das fases, com os
checkpoints de decisão no início.

Thresholds: os thresholds oficiais vêm de threshold_selection_oof.py
(selecionados nas predições out-of-fold do conjunto de desenvolvimento, sem
usar o teste); a seção correspondente é gerada a partir de
reports/oof_threshold_results.json. O antigo reports/phase3_threshold_tradeoff.md
agora é reports/phase3_threshold_tradeoff_test_selected.md (legado, thresholds
escolhidos no teste) e não é incorporado ao relatório.

Também oferece --run-all para executar as três fases em sequência antes de
consolidar (útil para reproduzir tudo do zero):
    python generate_consolidated_report.py --run-all

Todos os caminhos são resolvidos a partir do diretório deste script (ROOT).
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

# Marcador de seção gerada a partir do JSON de thresholds OOF
OOF_THRESHOLDS_SECTION = "<oof_threshold_results>"
OOF_RESULTS_PATH = ROOT / "reports" / "oof_threshold_results.json"
LEGACY_TRADEOFF_MD = "reports/phase3_threshold_tradeoff_test_selected.md"

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
        OOF_THRESHOLDS_SECTION,
        "reports/phase3_feature_importance.md",
        "reports/phase3_roc_analysis.md",
        "reports/phase3_predicted_vs_actual.md",
    ]),
]

OUTPUT_PATH = ROOT / "reports" / "RELATORIO_CONSOLIDADO_BANCA.md"

METRIC_LABELS = ["Recall (micro)", "Precisão (micro)", "F1 (micro)", "F1 (macro)"]


def run_all_phases() -> None:
    """Executa as três fases em sequência (reprodução completa)."""
    import subprocess

    for script in (
        "run_phase1_validation.py",
        "compare_statistical_models.py",
        "run_phase3_diagnostics.py",  # inclui threshold_selection_oof.py (thresholds oficiais)
    ):
        print(f"\n{'=' * 70}\nExecutando {script}...\n{'=' * 70}")
        result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"{script} falhou (exit code {result.returncode})")


def oof_thresholds_markdown() -> str | None:
    """Seção de thresholds oficiais a partir de reports/oof_threshold_results.json."""
    if not OOF_RESULTS_PATH.exists():
        return None
    data = json.loads(OOF_RESULTS_PATH.read_text(encoding="utf-8"))
    off = data["official"]
    lines = [
        "## Thresholds oficiais (seleção out-of-fold, sem uso do teste)",
        "",
        "Fonte: `threshold_selection_oof.py` → `reports/oof_threshold_results.json`. "
        "Thresholds por defeito selecionados nas predições out-of-fold do conjunto de "
        "desenvolvimento (CV 5-fold) e aplicados, inalterados, ao conjunto de teste.",
        "",
        f"- Estratégia oficial: `{data['official_strategy']}`",
        f"- Faixa de thresholds (defeitos com positivos no teste): "
        f"{off['thr_min']:.2f} – {off['thr_max']:.2f}",
        "",
        "### Rede neural — teste (IC 95% bootstrap)",
        "",
        "| Métrica | Valor | IC 95% |",
        "|---------|-------|--------|",
    ]
    for label, pt, lo, hi in zip(METRIC_LABELS, off["point"], off["lo"], off["hi"]):
        lines.append(f"| {label} | {pt:.4f} | [{lo:.4f}, {hi:.4f}] |")

    conf = data.get("overall_confusion")
    if conf:
        lines += [
            "",
            f"Matriz de confusão agregada (28 defeitos × teste): TP = {conf['tp']}, "
            f"TN = {conf['tn']}, FP = {conf['fp']}, FN = {conf['fn']}.",
        ]

    dev = data.get("dev_f2_by_strategy_nn")
    if dev:
        lines += [
            "",
            "### F2-micro no desenvolvimento (OOF) por estratégia — rede neural",
            "",
            "| Estratégia | F2-micro (OOF) |",
            "|------------|----------------|",
        ]
        for strategy, value in dev.items():
            lines.append(f"| {strategy} | {value:.4f} |")

    all_models = data.get("official_all_models")
    if all_models:
        lines += [
            "",
            "### Todos os modelos — estratégia oficial, teste",
            "",
            "| Modelo | " + " | ".join(METRIC_LABELS) + " |",
            "|--------|" + "|".join("---" for _ in METRIC_LABELS) + "|",
        ]
        for model, values in all_models.items():
            lines.append(f"| {model} | " + " | ".join(f"{v:.4f}" for v in values) + " |")

    lines += [
        "",
        f"*Legado:* `{LEGACY_TRADEOFF_MD}` (threshold_tradeoff_analysis.py) contém o "
        "procedimento anterior, com thresholds escolhidos no conjunto de teste; não é "
        "usado nos resultados oficiais.",
    ]
    return "\n".join(lines)


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
            if file == OOF_THRESHOLDS_SECTION:
                content = oof_thresholds_markdown()
                if content is None:
                    parts.append("\n> [PENDENTE] reports/oof_threshold_results.json não "
                                 "encontrado — execute threshold_selection_oof.py.\n")
                    continue
            else:
                path = ROOT / file
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
