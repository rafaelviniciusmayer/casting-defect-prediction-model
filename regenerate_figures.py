"""
Regenerate figures without modifying the trained model.
======================================================

This script regenerates all visualization figures (RESULTS.md and Supplementary)
using the existing trained model. It does NOT retrain or alter the model.

- Loads saved model from models/best_model.pkl
- Uses same train/test split (random_state=42) for consistency
- Regenerates: confusion matrices, and runs generate_visualizations.py

All figure labels and text are in English.
"""

import pickle
import sys
# Required for unpickling PyTorchModelWrapper from train_model
sys.path.insert(0, str(__file__).rsplit('/', 1)[0] if '/' in __file__ else '.')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import subprocess
import sys


def load_data_and_split():
    """Load dataset and replicate the same 80/20 stratified split as train_model.py."""
    df = pd.read_csv('aluminum_diecasting_dataset_with_features.csv')
    
    defect_prefixes = [
        'blisters', 'surface', 'die', 'flow', 'cold', 'heat', 'ejector',
        'low', 'density', 'incomplete', 'flash', 'warpage', 'shrinkage',
        'volumetric', 'dimensional', 'gas', 'internal', 'cracks', 'hard', 'oxide'
    ]
    defect_cols = [col for col in df.columns 
                  if any(col.startswith(prefix) for prefix in defect_prefixes)]
    feature_cols = [col for col in df.columns 
                   if col not in defect_cols + ['id', 'total_defects', 'has_defect']]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[defect_cols].values.astype(np.float32)
    
    has_defect = (y.sum(axis=1) > 0).astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=has_defect
    )
    
    return X_test, y_test, defect_cols


def regenerate_confusion_matrices():
    """Regenerate confusion matrix figures using the saved model (no retraining)."""
    print("="*60)
    print("Regenerating figures (model data unchanged)")
    print("="*60)
    
    # Load saved model (make classes available for unpickling from train_model)
    import train_model
    import __main__ as main_module
    for attr in ('PyTorchModelWrapper', 'DefectPredictionNN'):
        if hasattr(train_model, attr):
            setattr(main_module, attr, getattr(train_model, attr))
    with open('models/best_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    wrapper = artifacts['model']
    defect_names = artifacts['defect_cols']
    
    # Load data with same split
    X_test, y_test, _ = load_data_and_split()
    
    # Predict using saved model (same thresholds)
    y_pred_binary = wrapper.predict(X_test, defect_names=defect_names)
    
    # Regenerate confusion matrices
    Path('confusion_matrices').mkdir(exist_ok=True)
    print("\nRegenerating confusion matrices (English labels)...")
    
    for i, defect_name in enumerate(defect_names):
        y_true_defect = y_test[:, i]
        y_pred_defect = y_pred_binary[:, i]
        
        cm = confusion_matrix(y_true_defect, y_pred_defect)
        
        if cm.shape == (2, 2):
            fig, ax = plt.subplots(figsize=(8, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                         display_labels=['No Defect', 'Defect'])
            disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
            for text in ax.texts:
                text.set_fontsize(14)
            title_name = defect_name.replace('_', ' ').title()
            ax.set_title(f'Confusion Matrix: {title_name}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            ax.tick_params(labelsize=10)
            
            tn, fp, fn, tp = cm.ravel()
            precision_defect = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_defect = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            textstr = f'Precision: {precision_defect:.3f}\nRecall: {recall_defect:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='bottom', horizontalalignment='right', bbox=props)
            
            plt.tight_layout()
            safe_name = defect_name.replace('/', '_').replace('\\', '_')
            plt.savefig(f'confusion_matrices/confusion_matrix_{safe_name}.png', 
                       dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
    
    print(f"  [OK] {len(defect_names)} confusion matrices saved")


def main():
    # 1. Regenerate confusion matrices (uses saved model, no retraining)
    try:
        regenerate_confusion_matrices()
    except FileNotFoundError as e:
        print(f"\n[ERROR] Model not found. Run train_model.py first: {e}")
        sys.exit(1)
    
    # 2. Regenerate all other figures (boxplots, histograms, correlation)
    print("\nRegenerating EDA and supplementary figures...")
    subprocess.run([sys.executable, 'generate_visualizations.py'], check=True)
    
    print("\n" + "="*60)
    print("All figures regenerated successfully (model unchanged)")
    print("="*60)


if __name__ == "__main__":
    main()
