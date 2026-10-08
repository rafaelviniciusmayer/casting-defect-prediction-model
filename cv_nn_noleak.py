"""5-fold CV of the neural network with leak-free early stopping (internal hold-out), with and without SMOTE.
python cv_nn_noleak.py nosmote|smote  ->  reports/nn_noleak_cv_<cond>.json"""
import sys, json
import nn_noleak; nn_noleak.install()
import unified_cv_pipeline as ucv
cond = sys.argv[1]
data = ucv.load_pipeline_data(verbose=False); split = ucv.prepare_split_data(data, verbose=False)
r = ucv.run_cross_validation(ucv.MODEL_PYTORCH, split, use_smote=(cond == "smote"), verbose=True)
json.dump({"val_mean": r.val_mean, "val_std": r.val_std, "train_mean": r.train_mean, "train_std": r.train_std,
           "total_train_time_sec": r.total_train_time_sec,
           "folds": [{"train": f.train_metrics, "val": f.val_metrics, "time": f.train_time_sec} for f in r.fold_metrics]},
          open(f"reports/nn_noleak_cv_{cond}.json", "w"), indent=2)
print("[OK]", cond)
