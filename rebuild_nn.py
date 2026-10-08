"""Rebuild the neural-network results without any test-set influence on training.
python rebuild_nn.py final | oof | smote"""
import sys, time, json, pickle, shutil
from pathlib import Path
import numpy as np, torch
import nn_noleak; nn_noleak.install()
import unified_cv_pipeline as ucv

stage = sys.argv[1]; MT = ucv.MODEL_PYTORCH
data = ucv.load_pipeline_data(verbose=False); split = ucv.prepare_split_data(data, verbose=False)

if stage == "final":
    res = ucv.run_final_test_evaluation(MT, split, use_smote=False, verbose=True)
    Xt = res.scaler.transform(split.X_test).astype(np.float32)
    proba = ucv.predict_proba_any(MT, res.model, Xt)
    Xb = Xt[:100]; res.model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(1000): torch.sigmoid(res.model(torch.FloatTensor(Xb)))
    inf_ms = (time.perf_counter() - t0) / 1000 * 1000
    cp = Path("models/phase3_final_models.pkl"); bk = Path("models/phase3_final_models_test_early_stopping.pkl")
    if not bk.exists(): shutil.copy(cp, bk)
    cache = pickle.load(open(bk, "rb"))
    cache["models"][MT] = {"model": res.model, "model_name": res.model_name, "test_proba": proba,
                           "test_metrics_threshold_05": res.test_metrics, "train_time_sec": res.train_time_sec, "inference_ms_per_100": inf_ms}
    pickle.dump(cache, open(cp, "wb"))
    json.dump({"train_time_sec": res.train_time_sec, "inference_ms_per_100": inf_ms, "test_metrics_threshold_05": res.test_metrics}, open("reports/nn_noleak_final.json", "w"), indent=2)
    print("[OK] final NN", res.train_time_sec, inf_ms, res.test_metrics)

elif stage == "oof":
    out = Path("models/oof_proba"); bk = out / "pytorch_nn_test_early_stopping.npy"
    if not bk.exists(): shutil.copy(out / "pytorch_nn.npy", bk)
    oof = np.zeros(split.y_dev.shape, dtype=np.float32)
    for k, (tr, va) in enumerate(ucv.get_cv_splits(split.y_dev), start=1):
        Xtr, Xva, ytr, sc = ucv.preprocess_fold(split.X_dev[tr], split.X_dev[va], split.y_dev[tr].copy(), split.defect_names, use_smote=False)
        pw = ucv.compute_pos_weights(split.y_dev[tr])
        model, _ = nn_noleak.train_single_model_noleak(Xtr, ytr, None, None, pw, Xtr.shape[1], ytr.shape[1])
        oof[va] = ucv.predict_proba_any(MT, model, Xva); print("fold", k, flush=True)
    np.save(out / "pytorch_nn.npy", oof); print("[OK] oof NN")

elif stage == "smote":
    res = ucv.run_final_test_evaluation(MT, split, use_smote=True, verbose=True)
    Xt = res.scaler.transform(split.X_test).astype(np.float32)
    np.save("models/smote_proba/pytorch_nn.npy", ucv.predict_proba_any(MT, res.model, Xt))
    json.dump(res.test_metrics, open("reports/nn_noleak_smote_test05.json", "w"), indent=2)
    print("[OK] smote NN", res.test_metrics)
