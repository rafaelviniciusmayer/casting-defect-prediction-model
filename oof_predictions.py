"""
Out-of-fold (OOF) probabilities on the development set (5-fold stratified CV, no SMOTE, cost-sensitive learning),
used to choose per-defect decision thresholds WITHOUT touching the held-out test set.
Usage: python oof_predictions.py <model_type> [<model_type> ...]
Output: models/oof_proba/<model_type>.npy  (shape = y_dev.shape; row i = probabilities for dev sample i)
"""
import sys, time
from pathlib import Path
import numpy as np
import unified_cv_pipeline as ucv

OUT = Path("models/oof_proba"); OUT.mkdir(parents=True, exist_ok=True)

def run(model_type):
    data = ucv.load_pipeline_data(verbose=False)
    split = ucv.prepare_split_data(data, verbose=False)
    oof = np.zeros(split.y_dev.shape, dtype=np.float32)
    for k, (tr, va) in enumerate(ucv.get_cv_splits(split.y_dev), start=1):
        t0 = time.time()
        X_tr, y_tr = split.X_dev[tr], split.y_dev[tr]
        X_va, y_va = split.X_dev[va], split.y_dev[va]
        pw = ucv.compute_pos_weights(y_tr)
        Xtr_s, Xva_s, ytr_b, scaler = ucv.preprocess_fold(X_tr, X_va, y_tr.copy(), split.defect_names, use_smote=False, verbose=False)
        Xtr_eval = scaler.transform(X_tr).astype(np.float32)
        model, *_ = ucv.fit_fold_model(model_type, Xtr_s, ytr_b, Xtr_eval, y_tr, Xva_s, y_va, pw, split.defect_names)
        oof[va] = ucv.predict_proba_any(model_type, model, Xva_s)
        print(f"[{model_type}] fold {k}/5 done in {time.time()-t0:.0f}s", flush=True)
    np.save(OUT / f"{model_type}.npy", oof)
    np.save(OUT / "y_dev.npy", split.y_dev)
    print(f"[OK] {model_type}", flush=True)

if __name__ == "__main__":
    for m in sys.argv[1:]:
        run(m)
