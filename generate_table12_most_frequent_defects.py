"""
Build Table 12: confusion matrix metrics for the k most frequent defects (test set).

Frequency = number of positive instances in the test split (TP + FN), matching
the same evaluation as generate_table_confusion_matrix_metrics.py.

Outputs:
- figures/table12_most_frequent_confusion_matrix_metrics.csv
- figures/table12_most_frequent_confusion_matrix_metrics.md

The overall (micro-avg) row always aggregates all 28 defect types (from full table10).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLE10 = ROOT / "figures" / "table10_confusion_matrix_metrics.csv"
TOP_K = 8


def main():
    df = pd.read_csv(TABLE10)
    overall = df[df["Defect Type"].str.strip() == "Overall (micro-avg)"].copy()
    per_defect = df[df["Defect Type"].str.strip() != "Overall (micro-avg)"].copy()

    per_defect = per_defect.assign(
        _positives_test=per_defect["TP"] + per_defect["FN"]
    )
    per_defect = per_defect.sort_values(
        ["_positives_test", "Defect Type"], ascending=[False, True]
    )
    top = per_defect.drop(columns=["_positives_test"]).head(TOP_K)

    out_csv = ROOT / "figures" / "table12_most_frequent_confusion_matrix_metrics.csv"
    out_md = ROOT / "figures" / "table12_most_frequent_confusion_matrix_metrics.md"

    # CSV: top defects + overall
    pd.concat([top, overall], ignore_index=True).to_csv(out_csv, index=False)

    lines = [
        f"# Table 12. Confusion Matrix Metrics — {TOP_K} Most Frequent Defect Types (Test Set, *n* = 5,000)",
        "",
        f"*Selection criterion:* highest number of **positive** test instances (**TP + FN**) among the 28 labels. "
        f"Full per-defect table (all 28 + overall): `figures/table10_confusion_matrix_metrics.csv`.*",
        "",
        "| Defect Type | TP | TN | FP | FN | Recall | Precision |",
        "|-------------|----|----|----|----|--------|-----------|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['Defect Type']} | {int(r['TP'])} | {int(r['TN'])} | "
            f"{int(r['FP'])} | {int(r['FN'])} | {r['Recall']:.3f} | {r['Precision']:.3f} |"
        )
    if not overall.empty:
        r = overall.iloc[0]
        lines.append(
            f"| **Overall (micro-avg, all 28 defects)** | {int(r['TP'])} | {int(r['TN'])} | "
            f"{int(r['FP'])} | {int(r['FN'])} | {r['Recall']:.3f} | {r['Precision']:.3f} |"
        )

    lines.append("")
    lines.append(
        "*The overall row sums TP/TN/FP/FN across **all 28** binary defect heads (5,000 × 28 decisions), "
        "not only the eight rows above.*"
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {out_csv}")
    print(f"  [OK] {out_md}")
    print(f"\n  Top {TOP_K} by test positives (TP+FN):")
    for _, r in per_defect.head(TOP_K).iterrows():
        print(f"    {r['Defect Type']}: {int(r['_positives_test'])}")


if __name__ == "__main__":
    main()
