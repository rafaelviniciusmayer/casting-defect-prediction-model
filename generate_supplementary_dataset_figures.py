"""
Supplementary dataset figures (Figures S1-S4), drawn from the dataset used in the study
(aluminum_diecasting_dataset_with_features.csv) and the operating ranges of the metadata JSON.

  figures/supplementary/figS1_distribution_process_variables.png
  figures/supplementary/figS2_defect_occurrence_rates.png
  figures/supplementary/figS3_scatter_matrix.png
  figures/supplementary/figS4_variables_out_of_range.png

Run from the repository root:  python generate_supplementary_dataset_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "supplementary"
BLUE, ORANGE = "#4682b4", "#ff7f50"

SHORT_NAMES = {
    "piston_velocity_phase1": "Piston velocity (Phase 1)",
    "metal_velocity_gate": "Metal velocity at gate",
    "fill_time": "Cavity fill time",
    "phase_transition_position": "Phase transition position",
    "intensification_time_phase3": "Intensification time",
    "intensification_pressure": "Intensification pressure",
    "solidification_time": "Solidification time",
    "cycle_time": "Cycle time",
    "sleeve_fill_percentage": "Sleeve fill percentage",
    "sleeve_diameter": "Sleeve diameter",
    "sleeve_length": "Sleeve length",
    "plunger_sleeve_clearance": "Plunger-sleeve clearance",
    "sleeve_temperature": "Sleeve temperature",
    "plunger_temperature": "Plunger temperature",
}
SCATTER_VARS = ["metal_velocity_gate", "fill_time", "intensification_pressure", "solidification_time"]
# Design proportions of the generator (Supplementary Eq. 4); the 3% of ">= 4 variables" is split 40/30/20/10
EXPECTED = {1: 65.0, 2: 27.0, 3: 5.0, 4: 1.2, 5: 0.9, 6: 0.6, 7: 0.3}


def load():
    df = pd.read_csv(ROOT / "aluminum_diecasting_dataset_with_features.csv")
    with open(ROOT / "aluminum_diecasting_dataset_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    ranges = meta["generation_config"]["variable_ranges"]
    return df, meta["columns"]["defects"], ranges


def n_vars_out(df):
    """Number of process variables outside the defect-free zone (14 continuous + lubrication)."""
    return df["n_vars_out_of_range"] + (df["plunger_lubricant"] != 0)


def fig_s1(df, ranges):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 5, figsize=(22, 14))
    axes = axes.ravel()
    for ax, (var, name) in zip(axes, SHORT_NAMES.items()):
        lo, hi = ranges[var]["defect_free_range"]
        x = df[var]
        inside = (x >= lo) & (x <= hi)
        bins = np.linspace(x.min(), x.max(), 51)
        ax.hist(x[inside], bins=bins, color=BLUE, alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.hist(x[~inside], bins=bins, color=ORANGE, alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.axvline(lo, color="green", linestyle="--", linewidth=1.8)
        ax.axvline(hi, color="green", linestyle="--", linewidth=1.8)
        ax.set_yscale("log")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel(ranges[var]["unit"].replace("�", "°"), fontsize=9)
    axes[-1].axis("off")
    axes[-1].legend(
        handles=[Patch(color=BLUE, alpha=0.75, label="Defect-free zone"),
                 Patch(color=ORANGE, alpha=0.75, label="Defect-prone zone"),
                 Line2D([0], [0], color="green", linestyle="--", label="Defect-free boundaries")],
        loc="center", fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figS1_distribution_process_variables.png", dpi=100, facecolor="white")
    plt.close(fig)


def fig_s2(df, defects):
    rates = (df[defects].mean() * 100).sort_values()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 10.5))
    colors = plt.get_cmap("RdYlGn_r")(np.linspace(0.15, 0.85, len(rates)))
    bars = ax.barh([d.replace("_", " ").title() for d in rates.index], rates.values, color=colors,
                   edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in rates.values], padding=10, fontsize=9)
    ax.set_xlabel("Occurrence Rate (%)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, rates.max() * 1.12)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "figS2_defect_occurrence_rates.png", dpi=100, facecolor="white")
    plt.close(fig)


def fig_s3(df, ranges):
    all_ok = n_vars_out(df) == 0
    plt.style.use("seaborn-v0_8-whitegrid")
    n = len(SCATTER_VARS)
    fig, axes = plt.subplots(n, n, figsize=(13, 13))
    for i, vy in enumerate(SCATTER_VARS):
        for j, vx in enumerate(SCATTER_VARS):
            ax = axes[i, j]
            if i == j:
                ax.axis("off")
                ax.text(0.5, 0.5, SHORT_NAMES[vx].replace(" ", "\n"), ha="center", va="center",
                        fontsize=11, fontweight="bold", transform=ax.transAxes)
                continue
            ax.scatter(df.loc[~all_ok, vx], df.loc[~all_ok, vy], s=18, color="#ff8c00", alpha=0.7,
                       edgecolors="#8b4500", linewidths=0.4, rasterized=True)
            ax.scatter(df.loc[all_ok, vx], df.loc[all_ok, vy], s=18, color=BLUE, alpha=0.6,
                       edgecolors="#1f3b57", linewidths=0.3, rasterized=True)
            ax.set_xlim(*ranges[vx]["defect_prone_range"])
            ax.set_ylim(*ranges[vy]["defect_prone_range"])
            ax.tick_params(labelsize=7)
            if i == n - 1 or (i == n - 2 and j == n - 1):
                ax.set_xlabel(f"{SHORT_NAMES[vx]}\n({ranges[vx]['unit']})", fontsize=8)
            if j == 0 or (j == 1 and i == 0):
                ax.set_ylabel(f"{SHORT_NAMES[vy]}\n({ranges[vy]['unit']})", fontsize=8)
    fig.legend(handles=[Line2D([0], [0], marker="o", linestyle="", color=BLUE, label="All variables in defect-free zones"),
                        Line2D([0], [0], marker="o", linestyle="", color="#ff8c00", label="At least one variable in a defect-prone zone")],
               loc="upper center", ncol=2, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "figS3_scatter_matrix.png", dpi=100, facecolor="white")
    plt.close(fig)


def fig_s4(df):
    n = n_vars_out(df)
    scenarios = n[n > 0]
    observed = scenarios.value_counts(normalize=True).sort_index() * 100
    ks = sorted(EXPECTED)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11.7, 7))
    w = 0.4
    b1 = ax.bar(np.array(ks) - w / 2, [observed.get(k, 0.0) for k in ks], w, color=BLUE, edgecolor="black",
                linewidth=1.3, label=f"Observed (n = {len(scenarios):,})")
    b2 = ax.bar(np.array(ks) + w / 2, [EXPECTED[k] for k in ks], w, color=ORANGE, edgecolor="black",
                linewidth=1.3, label="Design (Eq. 4)")
    ax.bar_label(b1, fmt="%.1f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=9, padding=2)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of Variables Outside Defect-free Range (k)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Percentage of Defect Scenarios (%)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "figS4_variables_out_of_range.png", dpi=100, facecolor="white")
    plt.close(fig)
    return observed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, defects, ranges = load()
    fig_s1(df, ranges)
    fig_s2(df, defects)
    fig_s3(df, ranges)
    observed = fig_s4(df)
    print("[OK] figures written to", OUT)
    print("Variables outside defect-free zone (% of defect scenarios):", observed.round(1).to_dict())


if __name__ == "__main__":
    main()
