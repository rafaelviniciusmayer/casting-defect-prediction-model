"""
Redraw Figure 2 with value labels on each bar, using saved recall data.
Run from project root: python figures/redraw_figure2_with_labels.py
Requires: figures/feature_engineering_recall_data.json (from generate_figure_feature_engineering.py)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Run from project root
ROOT = Path(__file__).resolve().parent.parent
DATA_JSON = ROOT / 'figures' / 'feature_engineering_recall_data.json'
OUT_PATH = ROOT / 'figures' / 'feature_engineering_recall_comparison.png'


def main():
    if not DATA_JSON.exists():
        print("Data file not found. Run first: python generate_figure_feature_engineering.py")
        sys.exit(1)
    with open(DATA_JSON) as f:
        data = json.load(f)
    recall_15 = data['recall_15']
    recall_110 = data['recall_110']
    top8_labels = data['labels']
    n_110 = data['n_110']

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(top8_labels))
    width = 0.35
    vals_15 = [r * 100 for r in recall_15]
    vals_110 = [r * 100 for r in recall_110]
    bars1 = ax.bar(x - width / 2, vals_15, width, label='Original 15 features', color='#1f77b4')
    bars2 = ax.bar(x + width / 2, vals_110, width, label=f'Engineered {n_110} features', color='#ff7f0e')

    # Rótulos de valor em cada barra (acima da barra para boa legibilidade)
    ax.bar_label(bars1, labels=[f'{v:.0f}%' for v in vals_15], label_type='edge', fontsize=9, padding=2)
    ax.bar_label(bars2, labels=[f'{v:.0f}%' for v in vals_110], label_type='edge', fontsize=9, padding=2)

    ax.set_ylabel('Recall (%)', fontsize=11)
    ax.set_xlabel('Defect type', fontsize=11)
    ax.set_title('Impact of feature engineering on model recall', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top8_labels, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
