"""
Build Table 12: confusion matrix metrics for the k most frequent defects (test set).

Source: figures/table10_confusion_matrix_metrics.csv as written by
threshold_selection_oof.py (official per-defect thresholds selected on the
out-of-fold development predictions, evaluated once on the test set).
Schema: defect,n_pos,threshold,TP,TN,FP,FN,recall,precision (recall/precision
empty when undefined; no overall row).

Frequency = number of positive instances in the test split (n_pos = TP + FN).

Outputs:
- figures/table12_most_frequent_confusion_matrix_metrics.csv
- figures/table12_most_frequent_confusion_matrix_metrics.md

The overall (micro-avg) row always aggregates all 28 defect types (summed
TP/TN/FP/FN of the full table10).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLE10 = ROOT / "figures" / "table10_confusion_matrix_metrics.csv"
TOP_K = 8


def defect_display_name(name: str) -> str:
    """Convert snake_case defect name to Title Case for the table."""
    return name.replace("_", " ").title()


def fmt(v) -> str:
    """Format a metric with 3 decimals, or 'n/a' when undefined."""
    return "n/a" if pd.isna(v) else f"{v:.3f}"


def main():
    df = pd.read_csv(TABLE10)
    for col in ("TP", "TN", "FP", "FN", "n_pos"):
        df[col] = df[col].astype(int)

    per_defect = pd.DataFrame({
        "Defect Type": df["defect"].map(defect_display_name),
        "TP": df["TP"],
        "TN": df["TN"],
        "FP": df["FP"],
        "FN": df["FN"],
        "Recall": df["recall"],
        "Precision": df["precision"],
        "_positives_test": df["n_pos"],
    })

    # Overall micro-average over all 28 defects, from summed counts
    tp, tn, fp, fn = (int(df[c].sum()) for c in ("TP", "TN", "FP", "FN"))
    overall = pd.DataFrame([{
        "Defect Type": "Overall (micro-avg)",
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Recall": round(tp / (tp + fn), 3) if (tp + fn) > 0 else float("nan"),
        "Precision": round(tp / (tp + fp), 3) if (tp + fp) > 0 else float("nan"),
    }])

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
        f"Thresholds: official per-defect thresholds selected on out-of-fold development predictions "
        f"(`threshold_selection_oof.py`), never on the test set. "
        f"Full per-defect table (all 28): `figures/table10_confusion_matrix_metrics.csv`.*",
        "",
        "| Defect Type | TP | TN | FP | FN | Recall | Precision |",
        "|-------------|----|----|----|----|--------|-----------|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['Defect Type']} | {int(r['TP'])} | {int(r['TN'])} | "
            f"{int(r['FP'])} | {int(r['FN'])} | {fmt(r['Recall'])} | {fmt(r['Precision'])} |"
        )
    if not overall.empty:
        r = overall.iloc[0]
        lines.append(
            f"| **Overall (micro-avg, all 28 defects)** | {int(r['TP'])} | {int(r['TN'])} | "
            f"{int(r['FP'])} | {int(r['FN'])} | {fmt(r['Recall'])} | {fmt(r['Precision'])} |"
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
