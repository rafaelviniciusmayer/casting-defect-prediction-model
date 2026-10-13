"""
Comment #91 (Figure 11) - Calibration curves WITH vs WITHOUT SMOTE.

How to run (inside the local clone of the
casting-defect-prediction-model repository, with the environment already
used to train the original models):

    pip install imbalanced-learn   # if not installed yet
    python generate_calibration_with_without_smote.py

What the script does:
  1. Loads the official data split (same random_state=42, 80/20,
     stratified) via unified_cv_pipeline.load_pipeline_data /
     prepare_split_data.
  2. For the 5 models, trains WITH SMOTE (use_smote=True) on the full
     development set and evaluates on the test set (run_final_test_evaluation),
     exactly as the project pipeline itself already does.
  3. The predictions WITHOUT SMOTE are already saved in
     models/phase3_final_models.pkl (models[<type>]['test_proba']) -
     that part does not need retraining.
  4. Builds calibration curves (reliability diagram, micro-average over
     the 28 defects, 10 bins) for the two conditions, in the same style
     as the current Figure 11, and saves them to
     figures/phase3_calibration_with_vs_without_smote.png

Expected time (running locally, ordinary CPU): XGBoost and Logistic
Regression L2 finish in 1-2 min each; Random Forest and Logistic
Regression L1 may take more than 5 min (the liblinear solver gets
slower with the extra volume of synthetic SMOTE samples); the neural
network (200 epochs) should take a few minutes. In the sandbox
environment used for the rest of the edits this script exceeded the
per-command time limit before finishing all 5 models -
so it was not run to completion there, but it should run without
problems in the project environment, which already has everything
installed and no such limit.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import unified_cv_pipeline as ucv

MODEL_TYPES = [
    ucv.MODEL_PYTORCH,
    ucv.MODEL_XGBOOST,
    ucv.MODEL_RANDOM_FOREST,
    ucv.MODEL_LOGISTIC_L2,
    ucv.MODEL_LOGISTIC_L1,
]

def get_without_smote_proba():
    with open('models/phase3_final_models.pkl', 'rb') as f:
        d = pickle.load(f)
    return {mt: d['models'][mt]['test_proba'] for mt in MODEL_TYPES}, d['y_test']

def get_with_smote_proba(split):
    out = {}
    for model_type in MODEL_TYPES:
        print(f"[*] Training {model_type} WITH SMOTE...")
        res = ucv.run_final_test_evaluation(model_type, split, use_smote=True, verbose=True)
        proba = ucv.predict_proba_any(model_type, res.model, res.scaler.transform(split.X_test))
        out[model_type] = proba
    return out

def reliability_curve(y_true, proba, n_bins=10):
    y_true = y_true.ravel()
    proba = proba.ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(proba, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    mean_pred, frac_pos = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        mean_pred.append(proba[mask].mean())
        frac_pos.append(y_true[mask].mean())
    return np.array(mean_pred), np.array(frac_pos)

def main():
    data = ucv.load_pipeline_data(verbose=False)
    split = ucv.prepare_split_data(data, verbose=False)

    proba_without, y_test = get_without_smote_proba()
    proba_with = get_with_smote_proba(split)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, proba_dict, title in [
        (axes[0], proba_without, "Without SMOTE (official pipeline)"),
        (axes[1], proba_with, "With SMOTE"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
        for mt in MODEL_TYPES:
            mp, fp = reliability_curve(y_test, proba_dict[mt])
            ax.plot(mp, fp, marker="o", label=ucv.MODEL_DISPLAY_NAMES[mt])
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Observed frequency")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Calibration curves by model architecture (micro-average): with vs. without SMOTE")
    fig.tight_layout()
    fig.savefig("figures/phase3_calibration_with_vs_without_smote.png", dpi=150, facecolor="white")
    print("Saved to figures/phase3_calibration_with_vs_without_smote.png")

if __name__ == "__main__":
    main()
