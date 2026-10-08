"""Per-defect threshold analysis plots using the SAVED neural network (test probabilities) and the official thresholds
selected on out-of-fold development data (reports/phase3_official_thresholds.json). Output: threshold_analysis_oof/."""
import json, pickle
from pathlib import Path
import numpy as np
from generate_threshold_analysis_plots import plot_threshold_analysis_for_defect

cache = pickle.load(open("models/phase3_final_models.pkl", "rb"))
off = json.load(open("reports/phase3_official_thresholds.json"))
thr = off["thresholds_by_model"]["pytorch_nn"][off["official_strategy"]]
y, p = cache["y_test"], cache["models"]["pytorch_nn"]["test_proba"]
out = Path("threshold_analysis_oof"); out.mkdir(exist_ok=True)
n = 0
for k, name in enumerate(cache["defect_names"]):
    if y[:, k].sum() == 0:
        continue
    plot_threshold_analysis_for_defect(name, y[:, k], p[:, k], float(thr[name]), out); n += 1
print("generated", n)
