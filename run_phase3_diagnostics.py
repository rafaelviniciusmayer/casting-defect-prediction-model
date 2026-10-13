"""
Phase 3 — Diagnostics and interpretation (adjustments requested by the review committee)
========================================================================================

Orchestrates the Phase 3 items in the correct order:
  3.1 Feature importance (feature_importance_analysis.py)
  3.2 Precision × Recall threshold strategies (threshold_selection_oof.py,
      thresholds chosen on the development OOF predictions, without using the test set)
  3.3 ROC curves and AUC (roc_curve_analysis.py) — uses the thresholds from 3.2
  3.4 Predicted × actual (predicted_vs_actual.py) — uses the thresholds from 3.2

The 5 final models are trained only ONCE (phase3_utils, official Phase 1
config) and cached in models/phase3_final_models.pkl; the 4 items reuse
the cache.

Run: python run_phase3_diagnostics.py
"""

from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import feature_importance_analysis
import predicted_vs_actual
import roc_curve_analysis
from phase3_utils import build_or_load_final_models


def main() -> None:
    t_start = time.perf_counter()
    print("=" * 70)
    print("PHASE 3 — DIAGNOSTICS AND INTERPRETATION")
    print("=" * 70)

    # Ensure the final models cache exists (trains if necessary)
    build_or_load_final_models(verbose=True)

    runpy.run_path("threshold_selection_oof.py")   # 3.2 first: official thresholds (OOF)
    feature_importance_analysis.main()   # 3.1 (independent of threshold)
    roc_curve_analysis.main()            # 3.3 (uses the thresholds from 3.2)
    predicted_vs_actual.main()           # 3.4 (uses the thresholds from 3.2)

    # Phase checkpoint
    import json
    with open("reports/phase3_official_thresholds.json", encoding="utf-8") as f:
        official = json.load(f)
    with open("reports/phase3_feature_importance.json", encoding="utf-8") as f:
        importance = json.load(f)

    checkpoint = "\n".join([
        "# Checkpoint — Phase 3 (diagnostics and interpretation)\n",
        f"- **Official threshold strategy: {official['official_strategy_name']}** "
        f"({official['criteria']})",
        f"- Feature importance: {importance['answer']}",
        "- ROC/AUC curves and predicted×actual generated with the official thresholds.",
        "",
        "## Generated files\n",
        "- `reports/phase3_feature_importance.md` (3.1)",
        "- `reports/oof_threshold_results.json` and `reports/phase3_official_thresholds.json` (3.2)",
        "- `reports/phase3_roc_analysis.md` (3.3)",
        "- `reports/phase3_predicted_vs_actual.md` (3.4)",
        "- Corresponding figures and CSVs in `figures/`",
        "",
    ])
    Path("reports/PHASE3_CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"[DONE] Phase 3 completed in {elapsed / 60:.1f} min")
    print("=" * 70)
    print("Checkpoint: reports/PHASE3_CHECKPOINT.md")


if __name__ == "__main__":
    main()
