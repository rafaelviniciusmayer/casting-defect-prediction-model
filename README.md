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
pip install pandas numpy scikit-learn torch matplotlib imbalanced-learn streamlit
```

Optional (GPU PyTorch): install the build that matches your CUDA setup from [pytorch.org](https://pytorch.org).

## Regenerate the model locally

From the repository root:

```bash
python train_model.py
```

What this does:

1. **Data preparation** — loads the CSV, separates process/engineered features from the 28 defect labels, and computes class weights.
2. **Feature engineering** — keeps the engineered feature set when the `*_with_features.csv` is available.
3. **Training** — stratified 80/20 split, 5-fold CV, SMOTE (if `imbalanced-learn` is installed), PyTorch MLP (`input → 128 → 64 → 32 → 28`), per-defect threshold optimization for recall-oriented operation.
4. **Artifacts** written to disk:
   - `models/best_model.pkl` — primary artifact used by Streamlit
   - `models/pytorch_stable_model.pkl` — same payload (stable copy)
   - `optimal_thresholds.pkl` — per-defect thresholds
   - `model_metrics.json` — summary metrics (F1, precision, recall, accuracy)
   - `confusion_matrices/` — per-defect confusion matrix plots

Training can take several minutes depending on CPU/GPU. When it finishes, confirm that `models/best_model.pkl` exists.

> **Note:** The Streamlit UI collects only the **15 original process variables**. The saved model may expect the full engineered feature vector (~110). Always regenerate with `train_model.py` before running the app if you changed data or training code, so `models/best_model.pkl` stays in sync with `train_model.py` / `streamlit_app.py`.

## Run the Streamlit app

After the model artifacts exist:

```bash
streamlit run streamlit_app.py
```

The browser opens the **Die Casting Defect Predictor** dashboard:

1. Adjust the **15 process variables** in the sidebar.
2. Click **VERIFICAR PROBABILIDADE DE DEFEITOS**.
3. Inspect the 2×2 layout:
   - process parameters vs ideal ranges
   - predicted probabilities for 28 defect types (with threshold markers)
   - variables outside ideal ranges (severity)
   - parameter adjustment recommendations

If the model file is missing, the app stops with a message to run `python train_model.py` first.

### Typical local workflow

```bash
# 1) Install deps (once)
pip install pandas numpy scikit-learn torch matplotlib imbalanced-learn streamlit

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
- **Operating point:** per-defect thresholds favoring recall (defect escape risk)
- **Reference metrics** (last saved run in `model_metrics.json`): F1-micro ≈ 0.66, recall-micro ≈ 0.97, accuracy ≈ 0.94
