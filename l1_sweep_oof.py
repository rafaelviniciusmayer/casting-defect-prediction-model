"""Regularization-strength sensitivity of the logistic baselines, thresholds chosen on OOF development predictions only.
Usage: python l1_sweep_oof.py <l1|l2> <C>     -> models/sweep_oof/<penalty>_<C>.json"""
import sys, json, time, pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
import unified_cv_pipeline as ucv
from compare_models import _optimize_thresholds_silent

pen, C = sys.argv[1], float(sys.argv[2]); solver = 'liblinear' if pen == 'l1' else 'lbfgs'
out = Path('models/sweep_oof'); out.mkdir(parents=True, exist_ok=True)

def fit_predict(Xtr, ytr, Xev):
    P = np.zeros((len(Xev), ytr.shape[1])); nz = tot = 0
    for k in range(ytr.shape[1]):
        if len(np.unique(ytr[:, k])) < 2:
            P[:, k] = float(ytr[0, k]); continue
        clf = LogisticRegression(penalty=pen, C=C, class_weight='balanced', solver=solver, max_iter=2000, random_state=42).fit(Xtr, ytr[:, k])
        P[:, k] = clf.predict_proba(Xev)[:, 1]; nz += int((np.abs(clf.coef_) > 1e-8).sum()); tot += clf.coef_.size
    return P, nz, tot

data = ucv.load_pipeline_data(verbose=False); split = ucv.prepare_split_data(data, verbose=False)
oof = np.zeros(split.y_dev.shape)
t_fold = []
for tr, va in ucv.get_cv_splits(split.y_dev):
    Xtr, Xva, ytr, _ = ucv.preprocess_fold(split.X_dev[tr], split.X_dev[va], split.y_dev[tr].copy(), split.defect_names, use_smote=False)
    t0 = time.time(); oof[va], _, _ = fit_predict(Xtr.astype(np.float64), ytr.astype(int), Xva.astype(np.float64)); t_fold.append(time.time() - t0)
Xd, Xt, yd, sc = ucv.preprocess_fold(split.X_dev, split.X_test, split.y_dev.copy(), split.defect_names, use_smote=False)
t0 = time.time(); Pt, nz, tot = fit_predict(Xd.astype(np.float64), yd.astype(int), Xt.astype(np.float64)); t_final = time.time() - t0
thr, _ = _optimize_thresholds_silent(split.y_dev.astype(int), oof, split.defect_names)
pred = (Pt >= np.array([thr[n] for n in split.defect_names])).astype(int); y = split.y_test.astype(int)
r = dict(penalty=pen, C=C, time=t_final, nonzero=nz, total=tot,
         recall=recall_score(y, pred, average='micro'), precision=precision_score(y, pred, average='micro'),
         f1=f1_score(y, pred, average='micro'), f1_macro=f1_score(y, pred, average='macro', zero_division=0))
json.dump(r, open(out / f'{pen}_{C}.json', 'w')); print(r, flush=True)
