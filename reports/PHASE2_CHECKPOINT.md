# Checkpoint — Phase 2 (statistical models)

- Official condition: without SMOTE (inherited from Phase 1).
- Models compared: PyTorch NN, XGBoost, Random Forest, Logistic Regression L2 (Ridge), Logistic Regression L1 (Lasso).
- Best test recall (optimized thresholds): **Random Forest** (0.9952).
- Best test F1-micro: **Logistic Regression (L2/Ridge)** (0.6495).
- Phase 3 must use these 4-5 final models for feature importance (3.1) and the test probabilities for thresholds/ROC (3.2/3.3).
