"""
Calibration with vs. without SMOTE (v2): same procedure as gerar_calibracao_com_sem_smote.py, but
(1) saves the with-SMOTE test probabilities to disk (one .npy per model) so the figure can be redrawn
    without retraining, and (2) draws the figure at manuscript size and reports ECE.
Usage (repo root):
    python calibration_smote_v2.py --train logistic_regression_l1
    python calibration_smote_v2.py --train pytorch_nn xgboost random_forest logistic_regression_l2
    python calibration_smote_v2.py --plot
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import unified_cv_pipeline as ucv

OUT = Path("models/smote_proba"); OUT.mkdir(parents=True, exist_ok=True)
MODEL_TYPES = [ucv.MODEL_PYTORCH, ucv.MODEL_XGBOOST, ucv.MODEL_RANDOM_FOREST,
               ucv.MODEL_LOGISTIC_L2, ucv.MODEL_LOGISTIC_L1]

def train(models):
    data = ucv.load_pipeline_data(verbose=False); split = ucv.prepare_split_data(data, verbose=False)
    for mt in models:
        print(f"[*] {mt} WITH SMOTE...", flush=True)
        res = ucv.run_final_test_evaluation(mt, split, use_smote=True, verbose=True)
        p = ucv.predict_proba_any(mt, res.model, res.scaler.transform(split.X_test))
        np.save(OUT / f"{mt}.npy", np.asarray(p)); print(f"[OK] saved {mt}", flush=True)

def reliability(y, p, n_bins=10):
    y = y.ravel(); p = p.ravel()
    ids = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)) - 1, 0, n_bins - 1)
    mp, fp, w = [], [], []
    for b in range(n_bins):
        m = ids == b
        if m.any(): mp.append(p[m].mean()); fp.append(y[m].mean()); w.append(m.mean())
    return np.array(mp), np.array(fp), np.array(w)

def plot():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = pickle.load(open("models/phase3_final_models.pkl", "rb")); y = d["y_test"]
    without = {mt: d["models"][mt]["test_proba"] for mt in MODEL_TYPES}
    with_ = {mt: np.load(OUT / f"{mt}.npy") for mt in MODEL_TYPES}
    ece = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for ax, pd_, title, key in [(axes[0], without, "(a) Without SMOTE", "without"), (axes[1], with_, "(b) With SMOTE", "with")]:
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="Perfectly calibrated")
        for mt in MODEL_TYPES:
            mp, fp, w = reliability(y, pd_[mt])
            ece.setdefault(mt, {})[key] = float((w * np.abs(mp - fp)).sum())
            ax.plot(mp, fp, marker="o", ms=3, lw=1.2, label=ucv.MODEL_DISPLAY_NAMES[mt])
        ax.set_title(title, fontsize=9); ax.set_xlabel("Mean predicted probability", fontsize=8)
        ax.tick_params(labelsize=7.5); ax.grid(alpha=0.3); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Observed frequency", fontsize=8)
    axes[1].legend(fontsize=6, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig("figures/phase3_calibration_with_vs_without_smote_v2.png", dpi=300, facecolor="white")
    json.dump(ece, open("reports/calibration_ece_with_vs_without_smote.json", "w"), indent=2)
    for mt, v in ece.items(): print(f"{ucv.MODEL_DISPLAY_NAMES[mt]:34s} ECE without={v['without']:.3f} with={v['with']:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", nargs="*"); ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.train: train(a.train)
    if a.plot: plot()
