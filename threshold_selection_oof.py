"""
Threshold selection WITHOUT test-set leakage.

Per-defect decision thresholds are chosen on the DEVELOPMENT set only, using out-of-fold (OOF) probabilities
from the 5-fold stratified CV (oof_predictions.py), and are then applied, unchanged, to the held-out test set
probabilities of the final models (models/phase3_final_models.pkl). The official strategy is also selected
on the OOF predictions (highest F-beta(2)-micro of the production neural network), never on the test set.

Run from the repository root:
    python oof_predictions.py <model types>      # once (slow for the L1 model)
    python threshold_selection_oof.py

Outputs
    reports/oof_threshold_results.json           all numbers used in the manuscript
    reports/phase3_official_thresholds.json      (same layout as before; old file kept as *_test_selected.json)
    figures/table_phase3_threshold_strategies.csv
    figures/table_phase2_model_comparison_oof.csv
    figures/table10_confusion_matrix_metrics.csv
    figures/phase3_threshold_tradeoff.png
"""
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

from compare_models import _optimize_thresholds_silent
from threshold_tradeoff_analysis import (STRATEGIES, apply_thresholds, micro_metrics,
                                         optimize_thresholds_strategy, plot_tradeoff)
from unified_cv_pipeline import MODEL_DISPLAY_NAMES, MODEL_PYTORCH

import os
MODELS = os.environ.get("MODELS", "pytorch_nn,xgboost,random_forest,logistic_regression_l2,logistic_regression_l1").split(",")
N_BOOT, SEED = 2000, 42

cache = pickle.load(open("models/phase3_final_models.pkl", "rb"))
y_test = cache["y_test"].astype(int)
names = list(cache["defect_names"])
y_dev = np.load("models/oof_proba/y_dev.npy").astype(int)
assert (y_dev == cache["y_dev"].astype(int)).all(), "OOF labels do not match the cached development set"
oof = {m: np.load(f"models/oof_proba/{m}.npy") for m in MODELS}
test_proba = {m: cache["models"][m]["test_proba"] for m in MODELS}


def metrics4(y, p):
    """micro recall, micro precision, micro F1, macro F1"""
    return [recall_score(y, p, average="micro", zero_division=0), precision_score(y, p, average="micro", zero_division=0),
            f1_score(y, p, average="micro", zero_division=0), f1_score(y, p, average="macro", zero_division=0)]


def fast_metrics(y, p):
    tp = (y & p).sum(); fp = ((1 - y) & p).sum(); fn = (y & (1 - p)).sum()
    r = tp / (tp + fn) if tp + fn else 0.0; pr = tp / (tp + fp) if tp + fp else 0.0
    f1m = 2 * r * pr / (r + pr) if r + pr else 0.0
    tpk = (y & p).sum(0); fpk = ((1 - y) & p).sum(0); fnk = (y & (1 - p)).sum(0)
    den = 2 * tpk + fpk + fnk
    f1k = np.where(den > 0, 2 * tpk / np.maximum(den, 1), 0.0)
    return np.array([r, pr, f1m, f1k.mean()])


rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, len(y_test), len(y_test)) for _ in range(N_BOOT)]


def ci(y, p, fn=fast_metrics):
    pt = fn(y, p)
    bs = np.array([fn(y[i], p[i]) for i in boot_idx])
    return pt.tolist(), np.percentile(bs, 2.5, axis=0).tolist(), np.percentile(bs, 97.5, axis=0).tolist()


out = {"table5": {}, "strategies": {}, "thresholds": {}}

# --- Table 5: "recall-first" thresholds of the original comparison (compare_models logic), now chosen on OOF dev
for m in MODELS:
    thr, _ = _optimize_thresholds_silent(y_dev, oof[m], names)
    th = np.array([thr[n] for n in names])
    pred = (test_proba[m] >= th).astype(int)
    pt, lo, hi = ci(y_test, pred)
    out["table5"][m] = {"point": pt, "lo": lo, "hi": hi, "thresholds": {n: float(t) for n, t in zip(names, th)}}

# --- four strategies, thresholds from OOF dev, evaluated on test
dev_f2 = {}
for m in MODELS:
    out["strategies"][m] = {}
    out["thresholds"][m] = {}
    for s in STRATEGIES:
        thr = optimize_thresholds_strategy(y_dev, oof[m], s)
        out["thresholds"][m][s] = {names[i]: float(t) for i, t in thr.items()}
        mt = micro_metrics(y_test, apply_thresholds(test_proba[m], thr))
        out["strategies"][m][s] = mt
        if m == MODEL_PYTORCH:
            dev_f2[s] = fbeta_score(y_dev, apply_thresholds(oof[m], thr), beta=2, average="micro", zero_division=0)

official = max(dev_f2, key=dev_f2.get)           # chosen on development (OOF) data only
out["official_strategy"] = official
out["dev_f2_by_strategy_nn"] = dev_f2
th_off = np.array([out["thresholds"][MODEL_PYTORCH][official][n] for n in names])
pred_off = (test_proba[MODEL_PYTORCH] >= th_off).astype(int)
pt, lo, hi = ci(y_test, pred_off)
out["official"] = {"point": pt, "lo": lo, "hi": hi,
                   "thr_min": float(th_off[y_test.sum(0) > 0].min()), "thr_max": float(th_off[y_test.sum(0) > 0].max())}

# --- per-defect table + confusion counts for the official operating point (NN)
def met1(yy, pp):
    tp = (yy & pp).sum(); fp = ((1 - yy) & pp).sum(); fn = (yy & (1 - pp)).sum()
    r = tp / (tp + fn) if tp + fn else np.nan; pr = tp / (tp + fp) if tp + fp else np.nan
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    return np.array([r, pr, f])

per = {}
rows = []
for k, nm in enumerate(names):
    yk, pk = y_test[:, k], pred_off[:, k]
    tp = int((yk & pk).sum()); fp = int(((1 - yk) & pk).sum()); fn = int((yk & (1 - pk)).sum()); tn = int(((1 - yk) & (1 - pk)).sum())
    pt = met1(yk, pk)
    bs = np.array([met1(yk[i], pk[i]) for i in boot_idx])
    with np.errstate(all="ignore"):
        lo = np.nanpercentile(bs, 2.5, axis=0); hi = np.nanpercentile(bs, 97.5, axis=0)
    per[nm] = {"npos": int(yk.sum()), "thr": float(th_off[k]), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
               "point": pt.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    rows.append({"defect": nm, "n_pos": int(yk.sum()), "threshold": round(float(th_off[k]), 2), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                 "recall": None if np.isnan(pt[0]) else round(float(pt[0]), 3), "precision": None if np.isnan(pt[1]) else round(float(pt[1]), 3)})
out["per_defect"] = per
tot = {k: int(sum(per[n][k] for n in names)) for k in ("tp", "tn", "fp", "fn")}
out["overall_confusion"] = tot

# official operating point for every model (for completeness)
out["official_all_models"] = {}
for m in MODELS:
    thm = np.array([out["thresholds"][m][official][n] for n in names])
    out["official_all_models"][m] = fast_metrics(y_test, (test_proba[m] >= thm).astype(int)).tolist()

json.dump(out, open("reports/oof_threshold_results.json", "w"), indent=2)

# --- legacy-format files consumed by predicted_vs_actual.py / roc_curve_analysis.py
old = Path("reports/phase3_official_thresholds.json")
if old.exists() and not Path("reports/phase3_official_thresholds_test_selected.json").exists():
    shutil.copy(old, "reports/phase3_official_thresholds_test_selected.json")
json.dump({"official_strategy": official, "official_strategy_name": STRATEGIES[official],
           "criteria": "Highest F-beta(2)-micro of the production neural network on out-of-fold development predictions; thresholds chosen on development data only.",
           "nn_metrics": dict(zip(["recall_micro", "precision_micro", "f1_micro", "f1_macro"], out["official"]["point"])),
           "thresholds_by_model": out["thresholds"]},
          open("reports/phase3_official_thresholds.json", "w"), indent=2)

# --- CSV / figure outputs
pd.DataFrame(rows).to_csv("figures/table10_confusion_matrix_metrics.csv", index=False)
srows = [{"Model": MODEL_DISPLAY_NAMES[m], "Strategy": STRATEGIES[s], "Recall": f"{v['recall_micro']:.4f}", "Precision": f"{v['precision_micro']:.4f}",
          "F1-micro": f"{v['f1_micro']:.4f}", "F1-macro": f"{v['f1_macro']:.4f}", "F2-micro": f"{v['fbeta2_micro']:.4f}"}
         for m in MODELS for s, v in out["strategies"][m].items()]
pd.DataFrame(srows).to_csv("figures/table_phase3_threshold_strategies.csv", index=False)
results = [{"model_type": m, "model_name": MODEL_DISPLAY_NAMES[m], "strategy": s, "metrics": v} for m in MODELS for s, v in out["strategies"][m].items()]
plot_tradeoff(results, "figures/phase3_threshold_tradeoff.png")

print("official strategy (chosen on OOF dev):", official, {k: round(v, 4) for k, v in dev_f2.items()})
print("NN official, test: R/P/F1/F1macro =", [round(x, 4) for x in out["official"]["point"]], "thr range", out["official"]["thr_min"], out["official"]["thr_max"])
for m in MODELS:
    print(f"{m:26s} table5 R/P/F1/F1macro", [round(x, 4) for x in out["table5"][m]["point"]])
print("overall confusion", tot)
