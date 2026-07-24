"""
Generate Table 7: Defect Frequencies and SMOTE Balancing Effect (Training Set).

Replicates the exact pipeline used in train_model.py for the final model:
- load_and_prepare_data() -> aluminum_diecasting_dataset_with_features.csv
- train_test_split 80/20, random_state=42, stratify=has_defect
- StandardScaler fit on development (training) set
- apply_smote_balancing(X_dev_scaled, y_dev, defect_names)

Outputs:
- figures/table7_defect_frequencies_smote.csv
- figures/table7_defect_frequencies_smote.md

Note: After SMOTE, the training matrix grows (multi-label); each row's label
vector may have multiple defects. Frequencies are (positives for defect j) / n_rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_model import load_and_prepare_data, apply_smote_balancing  # noqa: E402


# Same eight defect types as the methodology table (ordered by original frequency, descending)
DEFECT_KEYS = [
    "gas_porosity",
    "density_deviation",
    "cold_shut",
    "gas_bubbles",
    "incomplete_fill",
    "blisters_post_treatment",
    "flow_lines",
    "shrinkage_porosity",
]

DISPLAY_NAMES = {
    "gas_porosity": "Gas Porosity",
    "density_deviation": "Density Deviation",
    "cold_shut": "Cold Shut",
    "gas_bubbles": "Gas Bubbles",
    "incomplete_fill": "Incomplete Fill",
    "blisters_post_treatment": "Blisters Post Treatment",
    "flow_lines": "Flow Lines",
    "shrinkage_porosity": "Shrinkage Porosity",
}


def format_ratio_one_to_x(neg_per_pos: float) -> str:
    """
    Express imbalance as '1:X' = 1 positive (minority) : X negatives (majority),
    matching manuscript style (e.g. 1:34 → 1:1.5).
    """
    if neg_per_pos is None or (isinstance(neg_per_pos, float) and np.isnan(neg_per_pos)):
        return "—"
    x = float(neg_per_pos)
    if x >= 10:
        return f"1:{round(x)}"
    return f"1:{x:.1f}"


def ratio_before_after_str(neg_per_pos_before: float | None, neg_per_pos_after: float | None) -> str:
    if neg_per_pos_before is None or neg_per_pos_after is None:
        return "—"
    return f"{format_ratio_one_to_x(neg_per_pos_before)} → {format_ratio_one_to_x(neg_per_pos_after)}"


def main():
    X, y, _feature_names, defect_names, _pos_weights = load_and_prepare_data()
    has_defect = (y.sum(axis=1) > 0).astype(int)
    X_dev, _X_test, y_dev, _y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=has_defect
    )

    n_before = len(y_dev)
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev).astype(np.float32)

    y_before = y_dev.copy()
    X_after, y_after = apply_smote_balancing(
        X_dev_scaled.copy(), y_dev.copy(), defect_names
    )
    n_after = len(y_after)

    name_to_idx = {n: i for i, n in enumerate(defect_names)}

    rows = []
    for key in DEFECT_KEYS:
        i = name_to_idx[key]
        pos_b = int(y_before[:, i].sum())
        pos_a = int(y_after[:, i].sum())
        neg_b = n_before - pos_b
        neg_a = n_after - pos_a

        freq_b = 100.0 * pos_b / n_before
        freq_a = 100.0 * pos_a / n_after

        # Negatives per positive (imbalance ratio); avoids div by zero
        ratio_b = neg_b / pos_b if pos_b > 0 else float("nan")
        ratio_a = neg_a / pos_a if pos_a > 0 else float("nan")

        rows.append(
            {
                "Defect Type": DISPLAY_NAMES[key],
                "defect_key": key,
                "Original Count": pos_b,
                "Original Freq (%)": round(freq_b, 2),
                "After SMOTE Count": pos_a,
                "After SMOTE Freq (%)": round(freq_a, 2),
                "Neg per pos (before)": round(ratio_b, 1) if pos_b > 0 else None,
                "Neg per pos (after)": round(ratio_a, 1) if pos_a > 0 else None,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = ROOT / "figures" / "table7_defect_frequencies_smote.csv"
    out_md = ROOT / "figures" / "table7_defect_frequencies_smote.md"
    out_csv.parent.mkdir(exist_ok=True)

    export = df[
        [
            "Defect Type",
            "Original Count",
            "Original Freq (%)",
            "After SMOTE Count",
            "After SMOTE Freq (%)",
            "Neg per pos (before)",
            "Neg per pos (after)",
        ]
    ].copy()
    export["Ratio Before → After"] = export.apply(
        lambda r: ratio_before_after_str(
            r["Neg per pos (before)"], r["Neg per pos (after)"]
        ),
        axis=1,
    )
    export = export.drop(columns=["Neg per pos (before)", "Neg per pos (after)"])
    export = export.rename(
        columns={
            "Original Freq (%)": "Original Freq. (%)",
            "After SMOTE Freq (%)": "After SMOTE Freq. (%)",
        }
    )
    # Consistent decimal formatting for percentages in CSV
    export["Original Freq. (%)"] = export["Original Freq. (%)"].map(lambda x: f"{x:.2f}")
    export["After SMOTE Freq. (%)"] = export["After SMOTE Freq. (%)"].map(lambda x: f"{x:.2f}")
    export.to_csv(out_csv, index=False)

    # Markdown (LaTeX-friendly caption)
    lines = [
        f"# Table 7. Defect Frequencies and SMOTE Balancing Effect (Training Set, n = {n_before:,})",
        "",
        f"After SMOTE augmentation, the training matrix has **n = {n_after:,}** rows (same pipeline as `train_model.py`). "
        "Frequencies after SMOTE are **positives for that defect / n_after** (multi-label rows may count in several defects).",
        "",
        "| Defect Type | Original Count | Original Freq. (%) | After SMOTE Count | After SMOTE Freq. (%) | Ratio Before → After |",
        "|-------------|----------------|--------------------|-------------------|-----------------------|----------------------|",
    ]
    for _, r in df.iterrows():
        rb = r["Neg per pos (before)"]
        ra = r["Neg per pos (after)"]
        ratio_str = ratio_before_after_str(rb, ra)
        lines.append(
            f"| {r['Defect Type']} | {int(r['Original Count'])} | {r['Original Freq (%)']:.2f} | "
            f"{int(r['After SMOTE Count'])} | {r['After SMOTE Freq (%)']:.2f} | {ratio_str} |"
        )

    lines.append("")
    lines.append(
        "*Ratio Before → After: expressed as **1 positive : X negatives** for the binary view of each defect "
        "(minority : majority), before SMOTE vs. after SMOTE.*"
    )
    lines.append("")
    lines.append(
        f"*Synthetic rows added: {n_after - n_before:,} (total training size after SMOTE: {n_after:,}).*"
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"  [OK] {out_csv}")
    print(f"  [OK] {out_md}")
    print(f"\n  Training before SMOTE: n = {n_before:,}")
    print(f"  Training after SMOTE:  n = {n_after:,}")


if __name__ == "__main__":
    main()
