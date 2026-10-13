# Casting Defect Prediction Model

Neural network for multilabel defect prediction in high-pressure aluminum die casting, plus a Streamlit dashboard for interactive inference.

The training pipeline (`train_model.py`) covers data preparation, feature engineering, and model training/selection. The web app (`streamlit_app.py`) loads the saved model and exposes the 15 process variables in a 2×2 dashboard (process parameters, defect probabilities with thresholds, out-of-range analysis, and recommendations).

## Requirements

- Python 3.10+ recommended
- Dataset present in the repo root (prefer the engineered CSV):
  - `aluminum_diecasting_dataset_with_features.csv` (preferred; ~110 features)
  - or `aluminum_diecasting_dataset.csv` (15 process variables; used as fallback)

### Python dependencies

```bash
pip install -r requirements.txt
```

Optional (GPU PyTorch): install the build that matches your CUDA setup from [pytorch.org](https://pytorch.org).

## Regenerate the model locally

From the repository root:

```bash
python train_model.py
```

What this does:

1. **Data preparation** — loads `aluminum_diecasting_dataset_with_features.csv`, separates the 110 process/engineered features from the 28 defect labels, and computes class weights (same pipeline as `unified_cv_pipeline.py`).
2. **Training** — stratified 80/20 split, PyTorch MLP (`input → 128 → 64 → 32 → 28`) with cost-sensitive loss and no SMOTE (Phase 1 decision), trained with the leak-free routine of `nn_noleak.py` (early stopping on an internal 10 % hold-out of the development data).
3. **Thresholds** — the per-defect thresholds of the official strategy, selected on out-of-fold development predictions (`reports/phase3_official_thresholds.json`, written by `threshold_selection_oof.py`). The test set is used only for the final report, so run `oof_predictions.py` / `threshold_selection_oof.py` first.
4. **Artifacts** written to disk:
   - `models/best_model.pkl` — primary artifact used by Streamlit
   - `models/pytorch_stable_model.pkl` — same payload (stable copy)
   - `optimal_thresholds.pkl` — per-defect thresholds
   - `model_metrics.json` — summary metrics (F1, precision, recall, accuracy)
   - `confusion_matrices/` — per-defect confusion matrix plots

Training can take several minutes depending on CPU/GPU. When it finishes, confirm that `models/best_model.pkl` exists.

> **Note:** The Streamlit UI collects only the **15 original process variables**; the model expects the full engineered feature vector (110). The app builds it with `feature_engineering.compute_engineered_features`, which reproduces the engineered columns of `aluminum_diecasting_dataset_with_features.csv` exactly (ranges read from `aluminum_diecasting_dataset_metadata.json`). Always regenerate with `train_model.py` before running the app if you changed data or training code, so `models/best_model.pkl` stays in sync with `train_model.py` / `streamlit_app.py`.

## Run the Streamlit app

After the model artifacts exist:

```bash
streamlit run streamlit_app.py
```

The browser opens the **Die Casting Defect Predictor** dashboard:

1. Adjust the **15 process variables** in the sidebar.
2. Click **CHECK DEFECT PROBABILITY**.
3. Inspect the 2×2 layout:
   - process parameters vs ideal ranges
   - predicted probabilities for 28 defect types (with threshold markers)
   - variables outside ideal ranges (severity)
   - parameter adjustment recommendations

If the model file is missing, the app stops with a message to run `python train_model.py` first.

### Typical local workflow

```bash
# 1) Install deps (once)
pip install -r requirements.txt

# 2) Train / regenerate artifacts
python train_model.py

# 3) Launch the UI
streamlit run streamlit_app.py
```

## Project layout (main files)

| Path | Role |
|------|------|
| `train_model.py` | End-to-end training; saves `models/best_model.pkl` |
| `streamlit_app.py` | Interactive inference UI |
| `aluminum_diecasting_dataset_with_features.csv` | Preferred training dataset |
| `models/best_model.pkl` | Deployed model + scaler + thresholds + metadata |
| `model_metrics.json` | Last training metrics |

## Model summary

- **Task:** multilabel classification (28 defect types)
- **Architecture:** PyTorch MLP with BatchNorm and Dropout
- **Operating point:** per-defect thresholds of the official strategy (F-beta, β = 2, favoring recall), selected on out-of-fold development predictions
- **Reference metrics** (test set, `model_metrics.json`): recall-micro ≈ 0.97, precision-micro ≈ 0.49, F1-micro ≈ 0.65, subset accuracy ≈ 0.94

## Reproducibility notes (added during the manuscript revision)

* **Data actually used in the study.** The models are trained on `aluminum_diecasting_dataset_with_features.csv`
  (25,000 records, 1,401 defective = 5.60 %). `aluminum_diecasting_dataset_used_in_study.csv` contains exactly its
  non-engineered columns (process variables, 28 defect labels, `total_defects`, `has_defect`). The older
  `aluminum_diecasting_dataset.csv` is a different random realization of the generator (1,408 defective records) and is
  **not** the data used in the article.
* **Decision thresholds are chosen without the test set.** `oof_predictions.py` produces out-of-fold probabilities on the
  development set; `threshold_selection_oof.py` selects the per-defect thresholds on them (and the official strategy) and
  evaluates once on the test set. The previous procedure (thresholds optimized on the test set) is kept only in
  `reports/phase3_official_thresholds_test_selected.json`.
* **Neural network without test leakage.** `nn_noleak.py` replaces the early-stopping set (previously the test set in the
  final model and the validation fold in CV) by an internal stratified 10 % hold-out of the training data and fixes the
  random seeds. `rebuild_nn.py` and `cv_nn_noleak.py` regenerate the final model, its out-of-fold predictions and the CV
  metrics with this training function. The previous cache is kept as `models/phase3_final_models_test_early_stopping.pkl`.
* `calibration_smote_v2.py`, `l1_sweep_oof.py`, `threshold_analysis_oof.py` and
  `figures/redraw_feature_importance_readable.py` regenerate the remaining new figures/tables of the revision.
* **Deployment artifacts regenerated.** `models/best_model.pkl`, `optimal_thresholds.pkl`, `model_metrics.json` and
  `confusion_matrices/` now come from `train_model.py`, i.e. the leak-free network with the OOF-selected official
  thresholds (the same model and operating point reported in the article). The original `train_model()` routine, which
  used the test set for early stopping and threshold selection, is kept in the file only for reference.
* **Legacy, test-selected outputs** are kept with the suffix `_test_selected` (`threshold_tradeoff_analysis.py` writes
  only such files and never touches `reports/phase3_official_thresholds.json`). `l1_sweep.py`, `compare_models.py` and
  `generate_calibration_with_without_smote.py` are also legacy (test-selected thresholds / superseded by `l1_sweep_oof.py` and
  `calibration_smote_v2.py`).
* **F1-macro** is averaged over the defect types that occur in the evaluated set: `ejector_pin_marks` has no positive
  record in the whole dataset, so including it would add a constant F1 = 0 (macro over 27 types = macro over 28 × 28/27).
  The stored L1/L2 sweep results (`models/sweep_oof/`) and `reports/nn_noleak_final.json` /
  `nn_noleak_smote_test05.json` were updated with this definition without retraining (exact conversion / recomputation
  from the saved probabilities).
* **Engineered features.** `feature_engineering.py` reproduces every engineered column of the study CSV from the 15
  process variables (max. abs. difference < 1e-10); the Streamlit app uses it to build the model input. Note that
  `fill_time_ratio` divides `fill_time` (ms) by `cycle_time` (s), so `process_efficiency` = `fill_time_ratio` / 1000;
  the definitions are kept as they are because the trained models depend on them.
* **Synthetic generator.** Running `generate_synthetic_dataset.py` reproduces `aluminum_diecasting_dataset.csv`
  byte-for-byte, not the study CSV (which came from an earlier run/version of the generator and is the canonical data
  of the article). The metadata `statistics` therefore describe the generated file, not the study data.
* **Random Forest / XGBoost reproducibility (open issue).** Re-running the Phase 1/Phase 2 scripts in the current
  environment (scikit-learn 1.9.1; last re-run stored in `reports/phase1_*` and `reports/phase2_*`) reproduces the
  neural network exactly but NOT the Random Forest values reported in the article (CV recall without SMOTE 0.944
  instead of 0.820; with SMOTE 0.713 instead of 0.624), and gives slightly different XGBoost / logistic-regression
  values. Refitting the cached Random Forest of `models/phase3_final_models.pkl` with identical parameters and data
  also gives a different model, so the published values come from an earlier environment (the legacy scaler was
  pickled with scikit-learn 1.7.2, but the exact cause was not confirmed). The official tables are computed from the
  cached models; the code selects the logistic-regression penalty in a way that is valid for old and new versions.
* **Scripts added in the last revision round.** `feature_engineering_ablation.py` (15 vs. 110 features under the
  leak-free protocol; Figure 2), `generate_supplementary_dataset_figures.py` (Figures S1-S4 from the study dataset) and
  `figures/figure4_streamlit_dashboard.png` (screenshot of the current app). In the study dataset the defect-free
  values follow a normal distribution centered on the zone (sd = width/6, limited to the zone), whereas
  `generate_synthetic_dataset.py` draws them uniformly.
* **Known modelling choices** (documented, not changed): early stopping is patience-based and returns the weights of the
  last epoch (not the best one); with SMOTE, the internal early-stopping hold-out is drawn after oversampling; OOF
  thresholds come from fold models and are applied to the model refitted on the whole development set.
