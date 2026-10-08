"""
Redraw Figure 7 (top-25 feature importance) with reader-friendly feature names.
Run from project root: python figures/redraw_feature_importance_readable.py
Reads figures/table_phase3_top_features.csv; writes figures/phase3_feature_importance_top25_readable.png
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BASE = {
    "piston_velocity_phase1": "Piston injection velocity",
    "metal_velocity_gate": "Metal velocity at gate",
    "fill_time": "Cavity fill time",
    "phase_transition_position": "Phase transition position",
    "intensification_pressure": "Intensification pressure",
    "intensification_time_phase3": "Pressure rise time",
    "solidification_time": "Solidification time",
    "cycle_time": "Total cycle time",
    "sleeve_diameter": "Shot sleeve diameter",
    "sleeve_length": "Shot sleeve length",
    "sleeve_temperature": "Sleeve temperature",
    "plunger_temperature": "Plunger temperature",
    "plunger_lubricant": "Plunger lubrication condition",
    "sleeve_fill_percentage": "Shot sleeve fill percentage",
    "plunger_sleeve_clearance": "Plunger-sleeve clearance",
}
SPECIAL = {
    "n_vars_in_range": "Number of variables in range",
    "n_vars_out_of_range": "Number of variables out of range",
    "max_distance_from_ideal": "Maximum distance from ideal",
    "avg_distance_from_ideal": "Average distance from ideal",
    "solidification_ratio": "Solidification ratio",
    "pressure_time_ratio": "Pressure-time ratio",
    "temp_ratio": "Temperature ratio",
    "velocity_distance": "Piston travel (velocity \u00d7 fill time)",
    "intensification_energy": "Intensification energy",
}

def pretty(name: str) -> str:
    if name in SPECIAL:
        return SPECIAL[name]
    for suffix, label in (("_distance_from_range", "distance from ideal range"),
                          ("_in_range", "within ideal range")):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            return f"{BASE.get(base, base)} ({label})"
    return BASE.get(name, name.replace("_", " ").capitalize())

COLORS = {"Injection": "tab:red", "Intensification": "tab:orange", "Cooling": "tab:blue",
          "Configuration/Maintenance": "tab:green", "Multiple phases": "tab:purple",
          "Global/Aggregation": "tab:gray"}

df = pd.read_csv(ROOT / "figures" / "table_phase3_top_features.csv").head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(11, 9.5))
ax.barh([pretty(f) for f in df["feature"]], df["importancia_media"],
        color=[COLORS.get(p, "tab:gray") for p in df["process_phase"]])
ax.set_xlabel("Mean normalized importance (5 models)", fontsize=11)
ax.set_title("Top 25 most important features \u2014 mean across models", fontsize=12)
ax.tick_params(axis="y", labelsize=10.5)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()], list(COLORS),
          title="Process phase", loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
out = ROOT / "figures" / "phase3_feature_importance_top25_readable.png"
fig.savefig(out, dpi=170, facecolor="white")
print("Saved:", out)
