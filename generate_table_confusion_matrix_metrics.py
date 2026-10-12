"""
Table 10 (Markdown) - confusion-matrix metrics per defect type on the test set.
======================================================================
Computes TP, TN, FP, FN, Recall, and Precision for each defect from the
deployment model models/best_model.pkl (built by train_model.py with the
leak-free network and the official per-defect thresholds selected on
out-of-fold development predictions), then writes the Markdown table.

The CSV version (figures/table10_confusion_matrix_metrics.csv) is written by
threshold_selection_oof.py from the same model and thresholds; this script
does not overwrite it.

Run after train_model.py. Uses same 80/20 split (random_state=42).
Output: figures/table10_confusion_matrix_metrics.md
Recall/Precision are NaN ("n/a" in Markdown) when undefined (no positives /
no positive predictions, e.g. ejector_pin_marks).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# For unpickling PyTorch model
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_data_and_split():
    """Same 80/20 stratified split as train_model / regenerate_figures."""
    df = pd.read_csv(ROOT / 'aluminum_diecasting_dataset_with_features.csv')
    defect_prefixes = [
        'blisters', 'surface', 'die', 'flow', 'cold', 'heat', 'ejector',
        'low', 'density', 'incomplete', 'flash', 'warpage', 'shrinkage',
        'volumetric', 'dimensional', 'gas', 'internal', 'cracks', 'hard', 'oxide'
    ]
    defect_cols = [c for c in df.columns
                  if any(c.startswith(p) for p in defect_prefixes)]
    feature_cols = [c for c in df.columns
                    if c not in defect_cols + ['id', 'total_defects', 'has_defect']]
    X = df[feature_cols].values.astype(np.float32)
    y = df[defect_cols].values.astype(np.float32)
    has_defect = (y.sum(axis=1) > 0).astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=has_defect
    )
    return X_test, y_test, defect_cols


def defect_display_name(name: str) -> str:
    """Convert snake_case defect name to Title Case for table."""
    return name.replace('_', ' ').title()


def _fmt(v) -> str:
    """3-decimal metric, or 'n/a' when undefined (NaN)."""
    return 'n/a' if v != v else f'{v:.3f}'


def main():
    # Load model (need PyTorch classes in __main__ for pickle)
    import train_model as tm
    import __main__ as main_module
    for attr in ('PyTorchModelWrapper', 'DefectPredictionNN'):
        if hasattr(tm, attr):
            setattr(main_module, attr, getattr(tm, attr))

    with open(ROOT / 'models' / 'best_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)

    model = artifacts['model']
    defect_names = artifacts['defect_cols']

    X_test, y_test, _ = load_data_and_split()
    n_test = X_test.shape[0]

    y_pred = model.predict(X_test, defect_names=defect_names)

    rows = []
    for i, defect_name in enumerate(defect_names):
        y_true = y_test[:, i]
        y_pred_d = y_pred[:, i]
        cm = confusion_matrix(y_true, y_pred_d, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Edge case: only one class present
            tn = int((y_true == 0).sum())
            fp = fn = tp = 0
            if y_true.sum() > 0:
                fn = int(y_true.sum())
            else:
                fp = int((y_pred_d == 1).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')

        rows.append({
            'defect_type': defect_display_name(defect_name),
            'defect_key': defect_name,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'Recall': round(recall, 3),
            'Precision': round(precision, 3),
        })

    # Micro-average (aggregate over all defect dimensions)
    total_tp = sum(r['TP'] for r in rows)
    total_tn = sum(r['TN'] for r in rows)
    total_fp = sum(r['FP'] for r in rows)
    total_fn = sum(r['FN'] for r in rows)
    recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float('nan')
    precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float('nan')

    rows.append({
        'defect_type': 'Overall (micro-avg)',
        'defect_key': '',
        'TP': total_tp,
        'TN': total_tn,
        'FP': total_fp,
        'FN': total_fn,
        'Recall': round(recall_micro, 3),
        'Precision': round(precision_micro, 3),
    })

    # DataFrame for CSV (without defect_key in export if you prefer)
    df_out = pd.DataFrame(rows)
    df_export = df_out[['defect_type', 'TP', 'TN', 'FP', 'FN', 'Recall', 'Precision']].rename(columns={'defect_type': 'Defect Type'})

    out_dir = ROOT / 'figures'
    out_dir.mkdir(exist_ok=True)
    print(df_export.to_string(index=False))

    # Markdown table with caption and acronym definitions
    md_path = out_dir / 'table10_confusion_matrix_metrics.md'
    md_lines = [
        '# Table 10. Confusion Matrix Metrics per Defect Type (Test Set, n=' + f'{n_test:,}' + ')',
        '',
        '*Per-defect thresholds of the official strategy, selected on out-of-fold development '
        'predictions (the test set is not used to choose them).*',
        '',
        '**Acronyms:**',
        '- **TP (True Positives):** Correctly predicted defects.',
        '- **TN (True Negatives):** Correctly predicted non-defects.',
        '- **FP (False Positives):** Type I error — non-defects predicted as defects.',
        '- **FN (False Negatives):** Type II error — defects predicted as non-defects.',
        '- **Recall:** TP / (TP + FN); proportion of actual defects correctly identified (sensitivity).',
        '- **Precision:** TP / (TP + FP); proportion of positive predictions that are correct (positive predictive value).',
        '',
        '| Defect Type | TP | TN | FP | FN | Recall | Precision |',
        '|-------------|----|----|----|----|--------|-----------|',
    ]
    for r in rows:
        md_lines.append(
            f"| {r['defect_type']} | {r['TP']} | {r['TN']} | {r['FP']} | {r['FN']} | {_fmt(r['Recall'])} | {_fmt(r['Precision'])} |"
        )
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"  [OK] {md_path}")

    print(f"\n  Test set size: n = {n_test:,}")
    print(f"  Defects: {len(defect_names)}")
    print(f"  Micro Recall: {recall_micro:.3f}, Micro Precision: {precision_micro:.3f}")


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[INFO] Model or data not found. Legacy table: figures/table10_confusion_matrix_metrics_legacy_test_selected.*")
        print(f"       Run train_model.py, then run this script to fill with real metrics.\n")
        raise
