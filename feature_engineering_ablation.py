"""
Feature-engineering ablation: neural network trained on the 15 original process variables vs. the
110 engineered features, under the official leak-free protocol.

Both models use the same 80/20 split, cost-sensitive learning (no SMOTE), early stopping on an internal
hold-out of the training data (nn_noleak) and per-defect F-beta (beta=2) thresholds selected on
out-of-fold development predictions; the test set is used only for the final metrics.
The 110-feature results are the official ones (models/phase3_final_models.pkl, models/oof_proba/).

Run from the repository root:  python feature_engineering_ablation.py          (train + plot)
                               python feature_engineering_ablation.py --plot   (redraw from the saved JSON)
Output: figures/feature_engineering_recall_data.json  (also read by figures/redraw_figure2_with_labels.py)
        figures/feature_engineering_overall_performance.png
"""
import json
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import unified_cv_pipeline as ucv
from feature_engineering import RAW_VARS
from threshold_tradeoff_analysis import apply_thresholds, optimize_thresholds_strategy

STRATEGY = "fbeta_2"
MT = ucv.MODEL_PYTORCH
TOP_N = 8


def metrics(y, pred):
    return {
        "f1_micro": float(f1_score(y, pred, average="micro", zero_division=0)),
        "f1_macro": float(ucv.macro_f1_observed(y, pred)),
        "precision_micro": float(precision_score(y, pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y, pred, average="micro", zero_division=0)),
    }


def evaluate(y_dev, oof, y_test, test_proba):
    thr = optimize_thresholds_strategy(y_dev, oof, STRATEGY)
    pred = apply_thresholds(test_proba, thr)
    per_defect_recall = [
        float(recall_score(y_test[:, k], pred[:, k], zero_division=0)) for k in range(y_test.shape[1])
    ]
    return metrics(y_test, pred), per_defect_recall


def plot_overall(out, path="figures/feature_engineering_overall_performance.png"):
    keys = [("f1_micro", "F1-Micro"), ("f1_macro", "F1-Macro"), ("precision_micro", "Precision"), ("recall_micro", "Recall")]
    v15 = [out["metrics_15"][k] * 100 for k, _ in keys]
    v110 = [out["metrics_110"][k] * 100 for k, _ in keys]
    x, w = np.arange(len(keys)), 0.35
    fig, ax = plt.subplots(figsize=(9.7, 6))
    b1 = ax.bar(x - w / 2, v15, w, label="Original 15 features", color="#1f77b4")
    b2 = ax.bar(x + w / 2, v110, w, label=f"Engineered {out['n_110']} features", color="#ff7f0e")
    ax.bar_label(b1, fmt="%.1f%%", padding=2, fontsize=10)
    ax.bar_label(b2, fmt="%.1f%%", padding=2, fontsize=10)
    ax.set_xticks(x, [name for _, name in keys], fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.legend(loc="lower left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main():
    if "--plot" in sys.argv:
        with open("figures/feature_engineering_recall_data.json") as f:
            plot_overall(json.load(f))
        return
    data = ucv.load_pipeline_data(verbose=False)
    split = ucv.prepare_split_data(data, verbose=False)
    cols = [split.feature_names.index(v) for v in RAW_VARS]
    y_dev, y_test = split.y_dev.astype(int), split.y_test.astype(int)
    X_dev, X_test = split.X_dev[:, cols], split.X_test[:, cols]

    # 15 features: out-of-fold predictions on the development set
    oof = np.zeros(y_dev.shape, dtype=np.float32)
    for k, (tr, va) in enumerate(ucv.get_cv_splits(split.y_dev), start=1):
        Xtr, Xva, ytr, scaler = ucv.preprocess_fold(
            X_dev[tr], X_dev[va], split.y_dev[tr].copy(), split.defect_names, use_smote=False, verbose=False
        )
        pw = ucv.compute_pos_weights(split.y_dev[tr])
        model, *_ = ucv.fit_fold_model(MT, Xtr, ytr, Xtr, split.y_dev[tr], Xva, split.y_dev[va], pw, split.defect_names)
        oof[va] = ucv.predict_proba_any(MT, model, Xva)
        print(f"[15 features] fold {k}/5 done", flush=True)

    # 15 features: final model on the whole development set
    Xd, Xt, yd, _ = ucv.preprocess_fold(X_dev, X_test, split.y_dev.copy(), split.defect_names, use_smote=False, verbose=False)
    pw = ucv.compute_pos_weights(split.y_dev)
    final, _ = ucv.train_single_model(Xd, yd, None, None, pw, Xd.shape[1], yd.shape[1], verbose=False)
    m15, r15 = evaluate(y_dev, oof, y_test, ucv.predict_proba_any(MT, final, Xt.astype(np.float32)))

    # 110 features: official out-of-fold predictions and final model
    with open("models/phase3_final_models.pkl", "rb") as f:
        cache = pickle.load(f)
    assert (cache["y_test"].astype(int) == y_test).all()
    m110, r110 = evaluate(y_dev, np.load(f"models/oof_proba/{MT}.npy"), y_test, cache["models"][MT]["test_proba"])

    top = np.argsort(-y_test.sum(axis=0), kind="stable")[:TOP_N]
    out = {
        "protocol": "leak-free NN, F-beta(2) thresholds selected on out-of-fold development predictions, test set",
        "labels": [split.defect_names[i].replace("_", " ").title() for i in top],
        "recall_15": [r15[i] for i in top],
        "recall_110": [r110[i] for i in top],
        "n_110": len(split.feature_names),
        "metrics_15": m15,
        "metrics_110": m110,
    }
    with open("figures/feature_engineering_recall_data.json", "w") as f:
        json.dump(out, f, indent=2)
    plot_overall(out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
