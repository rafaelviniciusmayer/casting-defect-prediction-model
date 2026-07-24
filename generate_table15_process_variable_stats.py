"""
Generate Table 15: Summary statistics for key process variables.

Uses aluminum_diecasting_dataset.csv and defect-free ranges from
aluminum_diecasting_dataset_metadata.json (generation_config.variable_ranges).

Output:
- figures/table15_summary_process_variables.csv
- figures/table15_summary_process_variables.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "aluminum_diecasting_dataset.csv"
META_PATH = ROOT / "aluminum_diecasting_dataset_metadata.json"

# Column key -> display label for manuscript (Table 15)
VARIABLES = [
    ("metal_velocity_gate", "Metal velocity at gate (m/s)"),
    ("fill_time", "Fill time (ms)"),
    ("intensification_pressure", "Intensification pressure (MPa)"),
    ("solidification_time", "Solidification time (s)"),
]


def main():
    df = pd.read_csv(CSV_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ranges = meta["generation_config"]["variable_ranges"]

    rows = []
    for col, label in VARIABLES:
        if col not in df.columns:
            raise KeyError(f"Missing column {col} in dataset")
        lo, hi = ranges[col]["defect_free_range"]
        x = df[col]
        in_zone = ((x >= lo) & (x <= hi)).mean() * 100.0
        rows.append(
            {
                "Variable": label,
                "Mean": round(float(x.mean()), 2),
                "Std Dev": round(float(x.std(ddof=0)), 2),
                "Min": round(float(x.min()), 2),
                "Max": round(float(x.max()), 2),
                "% in Defect-free Zone": round(in_zone, 1),
                "_range_lo": lo,
                "_range_hi": hi,
            }
        )

    out_df = pd.DataFrame(rows)
    export = out_df[
        ["Variable", "Mean", "Std Dev", "Min", "Max", "% in Defect-free Zone"]
    ]

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    csv_path = fig_dir / "table15_summary_process_variables.csv"
    md_path = fig_dir / "table15_summary_process_variables.md"
    export.to_csv(csv_path, index=False)

    n = len(df)
    md_lines = [
        f"# Table 15. Summary Statistics for Key Process Variables (*n* = {n:,})",
        "",
        "*Defect-free zone: interval from `aluminum_diecasting_dataset_metadata.json` (`defect_free_range`) for each variable.*",
        "",
        "| Variable | Mean | Std Dev | Min | Max | % in Defect-free Zone |",
        "|----------|------|---------|-----|-----|------------------------|",
    ]
    for _, r in export.iterrows():
        md_lines.append(
            f"| {r['Variable']} | {r['Mean']} | {r['Std Dev']} | {r['Min']} | {r['Max']} | {r['% in Defect-free Zone']:.1f}% |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"  [OK] {csv_path}")
    print(f"  [OK] {md_path}")
    for r in rows:
        print(
            f"    {r['Variable']}: [{r['_range_lo']}, {r['_range_hi']}] -> "
            f"{r['% in Defect-free Zone']:.1f}% in zone"
        )


if __name__ == "__main__":
    main()
