# Consolidated report — Adjustments requested by the review committee

Generated on 2026-10-13 14:30. Pipeline: stratified 80/20 split (random_state=42), unified 5-fold CV, official configuration defined in Phase 1.

---


# Decisions (checkpoints)

## Checkpoint — Phase 1 (Validation infrastructure)

### Decisions taken

- **SMOTE: REMOVED** in the official pipeline configuration (mean F1-micro gain -0.0076, recall -0.1450, train-val gap increase +0.0693).
- **Unified CV**: the 3 models now use the same stratified CV with 5 folds (random_state=42) over the dev set (80%), with SMOTE (when enabled) applied only to the training fold.
- **Cost-sensitive learning** active in all models: pos_weight in the NN, class_weight='balanced' in RF, scale_pos_weight in XGBoost (new in this phase — XGBoost previously had no weighting).
- Best recall (CV validation, with SMOTE): **Neural network (MLP)** (0.8665).
- Best F1-micro (CV validation, with SMOTE): **Neural network (MLP)** (0.6737).

### Reference for the next phases

- Phase 2 (new models) must use `unified_cv_pipeline.run_cross_validation` with `use_smote=False`.
- Phase 3 (thresholds/ROC) must start from the final models trained via `unified_cv_pipeline.run_final_test_evaluation` under the same condition.

### Generated files

- `reports/phase1_cv_comparison.md` (1.1)
- `reports/phase1_overfitting_analysis.md` (1.2)
- `reports/phase1_smote_comparison.md` (1.3)
- `reports/phase1_smote_overfitting.md` (1.4)
- `figures/table_phase1_overfitting.csv`
- `figures/table_phase1_smote_comparison.csv`
- `figures/table_phase1_train_vs_test.csv`
- `figures/phase1_learning_curves.png`

---

## Checkpoint — Phase 2 (statistical models)

- Official condition: without SMOTE (inherited from Phase 1).
- Models compared: PyTorch NN, XGBoost, Random Forest, Logistic Regression L2 (Ridge), Logistic Regression L1 (Lasso).
- Best test recall (optimized thresholds): **Random Forest** (0.9952).
- Best test F1-micro: **Logistic Regression (L2/Ridge)** (0.6495).
- Phase 3 must use these 4-5 final models for feature importance (3.1) and the test probabilities for thresholds/ROC (3.2/3.3).

---

## Checkpoint — Phase 3 (diagnostics and interpretation)

- **Official threshold strategy: F-beta β=2 (recall-weighted)** (Highest F-beta(2)-micro of the production neural network on out-of-fold development predictions; thresholds chosen on development data only.)
- Feature importance: The phase with the highest aggregated importance is **Global/Aggregation**. The injection phase accounts for 25.2% of the total importance, averaged across the 5 models.
- ROC/AUC curves and predicted×actual generated with the official thresholds.

### Generated files

- `reports/phase3_feature_importance.md` (3.1)
- `reports/oof_threshold_results.json` and `reports/phase3_official_thresholds.json` (3.2)
- `reports/phase3_roc_analysis.md` (3.3)
- `reports/phase3_predicted_vs_actual.md` (3.4)
- Corresponding figures and CSVs in `figures/`

---


# Phase 1 — Validation infrastructure

## Phase 1.1 — Cross-Validation Comparison (5 folds)

Mean ± standard deviation of the metrics on the **validation** part of each fold (fixed threshold 0.5). SMOTE applied only to the training part of each fold; the final test set (20%) remains held out.

| Model | Recall (val) | Precision (val) | F1-micro (val) | F1-macro (val) | CV Time (s) |
|--------|--------------|-----------------|----------------|----------------|--------------|
| Neural network (MLP) | 0.8665 ± 0.0090 | 0.5512 ± 0.0089 | 0.6737 ± 0.0082 | 0.6321 ± 0.0130 | 2266.3 |
| XGBoost | 0.7377 ± 0.0093 | 0.5950 ± 0.0087 | 0.6587 ± 0.0084 | 0.5985 ± 0.0130 | 315.1 |
| Random Forest | 0.7127 ± 0.0116 | 0.5990 ± 0.0064 | 0.6508 ± 0.0024 | 0.5907 ± 0.0149 | 420.9 |

---

## Phase 1.2 — Overfitting Analysis (with SMOTE)

Fixed threshold 0.5 across all columns for direct comparison between train (CV), validation (CV), and final test. Train metrics are computed on original fold data (no synthetic SMOTE samples).

Gap Train-Val > 0 indicates possible overfitting within folds. Gap Val-Test > 0 indicates performance drop on the holdout set.

| Model | Metric | Train (CV) | Validation (CV) | Test | Gap Train-Val | Gap Val-Test |
| --- | --- | --- | --- | --- | --- | --- |
| Neural network (MLP) | Recall | 0.9782 ± 0.0024 | 0.8665 ± 0.0090 | 0.8497 | 0.1117 | 0.0168 |
| Neural network (MLP) | Precision | 0.6008 ± 0.0040 | 0.5512 ± 0.0089 | 0.5391 | 0.0496 | 0.0121 |
| Neural network (MLP) | F1 | 0.7444 ± 0.0028 | 0.6737 ± 0.0082 | 0.6597 | 0.0707 | 0.0140 |
| XGBoost | Recall | 0.9804 ± 0.0009 | 0.7377 ± 0.0093 | 0.7619 | 0.2427 | -0.0242 |
| XGBoost | Precision | 0.8174 ± 0.0026 | 0.5950 ± 0.0087 | 0.5929 | 0.2224 | 0.0021 |
| XGBoost | F1 | 0.8915 ± 0.0017 | 0.6587 ± 0.0084 | 0.6668 | 0.2328 | -0.0081 |
| Random Forest | Recall | 0.9948 ± 0.0007 | 0.7127 ± 0.0116 | 0.7182 | 0.2822 | -0.0055 |
| Random Forest | Precision | 0.8184 ± 0.0035 | 0.5990 ± 0.0064 | 0.5877 | 0.2194 | 0.0113 |
| Random Forest | F1 | 0.8980 ± 0.0022 | 0.6508 ± 0.0024 | 0.6465 | 0.2472 | 0.0044 |

---

## Phase 1.3 — SMOTE Contribution (ablation)

Side-by-side comparison using the same CV pipeline (5 folds). In both conditions, cost-sensitive learning remains active (pos_weight in the NN, class_weight='balanced' in RF, scale_pos_weight in XGBoost).

| Model | Condition | Recall (CV) | Precision (CV) | F1-micro (CV) | F1-macro (CV) | CV Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Neural network (MLP) | No SMOTE | 0.9513 ± 0.0102 | 0.5127 ± 0.0133 | 0.6662 ± 0.0122 | 0.6286 ± 0.0142 | 350.8 |
| Neural network (MLP) | With SMOTE | 0.8665 ± 0.0090 | 0.5512 ± 0.0089 | 0.6737 ± 0.0082 | 0.6321 ± 0.0130 | 2266.3 |
| Neural network (MLP) | Delta (With - No) | -0.0849 | +0.0385 | +0.0075 | +0.0035 | +1915.5 |
| XGBoost | No SMOTE | 0.8569 ± 0.0102 | 0.5491 ± 0.0119 | 0.6692 ± 0.0103 | 0.6231 ± 0.0157 | 157.7 |
| XGBoost | With SMOTE | 0.7377 ± 0.0093 | 0.5950 ± 0.0087 | 0.6587 ± 0.0084 | 0.5985 ± 0.0130 | 315.1 |
| XGBoost | Delta (With - No) | -0.1192 | +0.0459 | -0.0105 | -0.0246 | +157.4 |
| Random Forest | No SMOTE | 0.9436 ± 0.0041 | 0.5200 ± 0.0106 | 0.6705 ± 0.0093 | 0.6319 ± 0.0147 | 57541.6 |
| Random Forest | With SMOTE | 0.7127 ± 0.0116 | 0.5990 ± 0.0064 | 0.6508 ± 0.0024 | 0.5907 ± 0.0149 | 420.9 |
| Random Forest | Delta (With - No) | -0.2309 | +0.0790 | -0.0196 | -0.0412 | -57120.6 |

### Recommendations per model

- **Neural network (MLP)**: SMOTE brings no relevant gain (ΔF1=+0.0075, ΔRecall=-0.0849) — consider cost-sensitive learning only.
- **XGBoost**: SMOTE brings no relevant gain (ΔF1=-0.0105, ΔRecall=-0.1192) — consider cost-sensitive learning only.
- **Random Forest**: SMOTE brings no relevant gain (ΔF1=-0.0196, ΔRecall=-0.2309) — consider cost-sensitive learning only.

### Phase 1 decision

- **SMOTE REMOVED** — the official configuration now uses cost-sensitive learning only
- Mean F1-micro gain: -0.0076
- Mean recall gain: -0.1450
- Mean train-val gap increase: +0.0693
- Criterion: Keep SMOTE if the mean gain of F1-micro >= 0.005 or of recall >= 0.01 on validation (CV), with a mean train-val gap increase <= 0.05.

---

## Phase 1.4 — Overfitting Caused by SMOTE

### Train (post-SMOTE) vs held-out Test

Metrics of the final model computed on the balanced training set itself (containing synthetic samples) vs on the never-seen test set. A large gap indicates overfitting, possibly inflated by SMOTE.

| Model | Metric | Train (post-SMOTE) | Test (holdout) | Gap Train-Test |
| --- | --- | --- | --- | --- |
| Neural network (MLP) | Recall | 0.9958 | 0.8497 | 0.1462 |
| Neural network (MLP) | Precision | 0.6064 | 0.5391 | 0.0672 |
| Neural network (MLP) | F1 | 0.7538 | 0.6597 | 0.0941 |
| XGBoost | Recall | 0.9882 | 0.7619 | 0.2264 |
| XGBoost | Precision | 0.7689 | 0.5929 | 0.1760 |
| XGBoost | F1 | 0.8649 | 0.6668 | 0.1980 |
| Random Forest | Recall | 0.9981 | 0.7182 | 0.2798 |
| Random Forest | Precision | 0.7921 | 0.5877 | 0.2044 |
| Random Forest | F1 | 0.8833 | 0.6465 | 0.2368 |

### Learning curves

![Learning curves](phase1_learning_curves.png)

| Model | Condition | Train fraction | N train (orig.) | N train (effective) | F1 train (orig.) | F1 train (post-SMOTE) | F1 validation | Recall validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pytorch_nn | No SMOTE | 10% | 1600 | 1600 | 0.6729 | 0.6729 | 0.5870 | 0.8630 |
| pytorch_nn | No SMOTE | 25% | 4000 | 4000 | 0.7077 | 0.7077 | 0.6581 | 0.8811 |
| pytorch_nn | No SMOTE | 50% | 8000 | 8000 | 0.6986 | 0.6986 | 0.6647 | 0.9296 |
| pytorch_nn | No SMOTE | 75% | 12000 | 12000 | 0.6976 | 0.6976 | 0.6764 | 0.9457 |
| pytorch_nn | No SMOTE | 100% | 16000 | 16000 | 0.6794 | 0.6794 | 0.6725 | 0.9703 |
| pytorch_nn | With SMOTE | 10% | 1600 | 3451 | 0.8463 | 0.8087 | 0.5964 | 0.6434 |
| pytorch_nn | With SMOTE | 25% | 4000 | 8266 | 0.8090 | 0.7983 | 0.6244 | 0.6809 |
| pytorch_nn | With SMOTE | 50% | 8000 | 17027 | 0.7743 | 0.7591 | 0.6642 | 0.8017 |
| pytorch_nn | With SMOTE | 75% | 12000 | 25485 | 0.7597 | 0.7671 | 0.6736 | 0.8398 |
| pytorch_nn | With SMOTE | 100% | 16000 | 33706 | 0.7473 | 0.7595 | 0.6780 | 0.8760 |
| random_forest | No SMOTE | 10% | 1600 | 1600 | 0.8526 | 0.8526 | 0.6104 | 0.7119 |
| random_forest | No SMOTE | 25% | 4000 | 4000 | 0.8100 | 0.8100 | 0.6571 | 0.8172 |
| random_forest | No SMOTE | 50% | 8000 | 8000 | 0.7544 | 0.7544 | 0.6716 | 0.9121 |
| random_forest | No SMOTE | 75% | 12000 | 12000 | 0.7377 | 0.7377 | 0.6755 | 0.9354 |
| random_forest | No SMOTE | 100% | 16000 | 16000 | 0.7264 | 0.7264 | 0.6813 | 0.9503 |
| random_forest | With SMOTE | 10% | 1600 | 3451 | 0.9726 | 0.9270 | 0.4810 | 0.3882 |
| random_forest | With SMOTE | 25% | 4000 | 8266 | 0.9320 | 0.9159 | 0.5490 | 0.5084 |
| random_forest | With SMOTE | 50% | 8000 | 17027 | 0.9168 | 0.8934 | 0.6138 | 0.6253 |
| random_forest | With SMOTE | 75% | 12000 | 25485 | 0.9077 | 0.8974 | 0.6419 | 0.6964 |
| random_forest | With SMOTE | 100% | 16000 | 33706 | 0.9017 | 0.8876 | 0.6479 | 0.7119 |
| xgboost | No SMOTE | 10% | 1600 | 1600 | 0.9968 | 0.9968 | 0.5354 | 0.5640 |
| xgboost | No SMOTE | 25% | 4000 | 4000 | 0.9979 | 0.9979 | 0.6014 | 0.6492 |
| xgboost | No SMOTE | 50% | 8000 | 8000 | 0.9923 | 0.9923 | 0.6573 | 0.7694 |
| xgboost | No SMOTE | 75% | 12000 | 12000 | 0.9629 | 0.9629 | 0.6791 | 0.8495 |
| xgboost | No SMOTE | 100% | 16000 | 16000 | 0.9359 | 0.9359 | 0.6815 | 0.8702 |
| xgboost | With SMOTE | 10% | 1600 | 3451 | 0.9857 | 0.9730 | 0.4796 | 0.3844 |
| xgboost | With SMOTE | 25% | 4000 | 8266 | 0.9440 | 0.9376 | 0.5580 | 0.5161 |
| xgboost | With SMOTE | 50% | 8000 | 17027 | 0.9264 | 0.8939 | 0.6315 | 0.6505 |
| xgboost | With SMOTE | 75% | 12000 | 25485 | 0.9048 | 0.8876 | 0.6626 | 0.7332 |
| xgboost | With SMOTE | 100% | 16000 | 33706 | 0.8942 | 0.8737 | 0.6623 | 0.7461 |

---


# Phase 2 — Statistical models

## Phase 2.1 — Comparison with statistical and regression models

Official Phase 1 configuration: **without SMOTE (cost-sensitive only)**. All models use the same 80/20 split, the same 5-fold CV, the same preprocessing and the same per-defect threshold optimization (maximize recall) in the final test evaluation.

Statistical baselines added: Logistic Regression L2 (Ridge) and L1 (Lasso), both with `class_weight='balanced'` via `MultiOutputClassifier`.

| Model | Recall (test) | Precision (test) | F1-micro (test) | F1-macro (test) | Recall (CV) | F1-micro (CV) | Train (s) | Inference (ms/100) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Neural network (MLP) | 0.9865 | 0.4648 | 0.6319 | 0.5923 | 0.9513 ± 0.0102 | 0.6662 ± 0.0122 | 171.8 | 0.50 |
| XGBoost | 0.9795 | 0.4789 | 0.6433 | 0.6190 | 0.8569 ± 0.0102 | 0.6692 ± 0.0103 | 31.3 | 35.04 |
| Random Forest | 0.9952 | 0.4400 | 0.6102 | 0.5903 | 0.9436 ± 0.0041 | 0.6705 ± 0.0093 | 44.5 | 1801.89 |
| Logistic Regression (L2/Ridge) | 0.9811 | 0.4855 | 0.6495 | 0.6253 | 0.9392 ± 0.0059 | 0.6725 ± 0.0086 | 8.3 | 5.64 |
| Logistic Regression (L1/Lasso) | 0.9806 | 0.4826 | 0.6469 | 0.6177 | 0.9452 ± 0.0063 | 0.6730 ± 0.0084 | 615.6 | 7.22 |

---


# Phase 3 — Diagnostics and interpretation

### Official thresholds (out-of-fold selection, test set not used)

Source: `threshold_selection_oof.py` → `reports/oof_threshold_results.json`. Per-defect thresholds selected on the out-of-fold predictions of the development set (5-fold CV) and applied, unchanged, to the test set.

- Official strategy: `fbeta_2`
- Threshold range (defects with positives in the test set): 0.13 – 0.66

#### Neural network — test set (95% bootstrap CI)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Recall (micro) | 0.9725 | [0.9617, 0.9818] |
| Precision (micro) | 0.4928 | [0.4687, 0.5153] |
| F1 (micro) | 0.6541 | [0.6321, 0.6736] |
| F1 (macro) | 0.6151 | [0.5884, 0.6370] |

Aggregated confusion matrix (28 defects × test set): TP = 1805, TN = 136286, FP = 1858, FN = 51.

#### F2-micro on the development set (OOF) by strategy — neural network

| Strategy | F2-micro (OOF) |
|----------|----------------|
| recall_first | 0.7987 |
| f1 | 0.7896 |
| fbeta_2 | 0.8206 |
| fbeta_05 | 0.6498 |

#### All models — official strategy, test set

| Model | Recall (micro) | Precision (micro) | F1 (micro) | F1 (macro) |
|-------|---|---|---|---|
| pytorch_nn | 0.9725 | 0.4928 | 0.6541 | 0.6151 |
| xgboost | 0.9634 | 0.5020 | 0.6600 | 0.6329 |
| random_forest | 0.8922 | 0.5170 | 0.6547 | 0.6159 |
| logistic_regression_l2 | 0.9709 | 0.5021 | 0.6619 | 0.6361 |
| logistic_regression_l1 | 0.9671 | 0.5035 | 0.6622 | 0.6307 |

*Legacy:* `reports/phase3_threshold_tradeoff_test_selected.md` (threshold_tradeoff_analysis.py) contains the previous procedure, with thresholds chosen on the test set; it is not used in the official results.

---

## Phase 3.1 — Feature importance

Review committee question: *are the injection-stage variables the ones with the greatest impact?*

**Answer:** The phase with the highest aggregated importance is **Global/Aggregation**. The injection phase accounts for 25.2% of the total importance, averaged across the 5 models.

### Aggregated importance by process phase

| process_phase | Neural network (MLP) | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) | mean_importance | n_features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Global/Aggregation | 0.046 | 0.6635000109672546 | 0.5719 | 0.053 | 0.064 | 0.2797 | 4 |
| Injection | 0.3818 | 0.10890000313520432 | 0.1571 | 0.3122 | 0.2996 | 0.2519 | 36 |
| Configuration/Maintenance | 0.1331 | 0.09359999746084213 | 0.0658 | 0.2695 | 0.2795 | 0.1683 | 35 |
| Intensification | 0.2537 | 0.06549999862909317 | 0.0844 | 0.1685 | 0.1666 | 0.1477 | 16 |
| Cooling | 0.1814 | 0.05939999967813492 | 0.0996 | 0.1392 | 0.1396 | 0.1238 | 14 |
| Multiple phases | 0.004 | 0.009200000204145908 | 0.0211 | 0.0577 | 0.0507 | 0.0285 | 5 |

### Aggregated importance by feature engineering category

| feature_category | Neural network (MLP) | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) | mean_importance | n_features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ideal-range distance | 0.5204 | 0.11659999936819077 | 0.1978 | 0.279 | 0.3009 | 0.283 | 28 |
| Statistical aggregation | 0.046 | 0.6635000109672546 | 0.5719 | 0.053 | 0.064 | 0.2797 | 4 |
| Range flag | 0.3859 | 0.15039999783039093 | 0.1044 | 0.2663 | 0.2536 | 0.2321 | 42 |
| Original | 0.0303 | 0.03139999881386757 | 0.0412 | 0.151 | 0.1426 | 0.0793 | 15 |
| Ratio | 0.0043 | 0.011099999770522118 | 0.0295 | 0.103 | 0.101 | 0.0498 | 6 |
| Difference | 0.0025 | 0.007499999832361937 | 0.0162 | 0.0472 | 0.0545 | 0.0256 | 4 |
| Domain-specific | 0.0043 | 0.009499999694526196 | 0.017 | 0.0433 | 0.0318 | 0.0212 | 5 |
| Product | 0.0 | 0.004100000020116568 | 0.0088 | 0.0294 | 0.0321 | 0.0149 | 2 |
| Mathematical transform | 0.0061 | 0.005900000222027302 | 0.0133 | 0.0278 | 0.0194 | 0.0145 | 4 |

### Top 20 features (mean across models)

| feature | process_phase | feature_category | mean_importance | Neural network (MLP) | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n_vars_out_of_range | Global/Aggregation | Statistical aggregation | 0.09845 | 0.00964 | 0.2995699942111969 | 0.16818 | 0.00698 | 0.00789 |
| n_vars_in_range | Global/Aggregation | Statistical aggregation | 0.09557 | 0.02896 | 0.2989799976348877 | 0.13458 | 0.00698 | 0.00836 |
| max_distance_from_ideal | Global/Aggregation | Statistical aggregation | 0.05886 | 0.00543 | 0.04019000008702278 | 0.16906 | 0.03344 | 0.04618 |
| cycle_time_distance_from_range | Cooling | Ideal-range distance | 0.02881 | 0.08169 | 0.01906999945640564 | 0.01915 | 0.01101 | 0.01314 |
| avg_distance_from_ideal | Global/Aggregation | Statistical aggregation | 0.02679 | 0.00193 | 0.024720000103116035 | 0.1001 | 0.00561 | 0.00156 |
| metal_velocity_gate_distance_from_range | Injection | Ideal-range distance | 0.02508 | 0.07448 | 0.005570000037550926 | 0.01113 | 0.01623 | 0.01798 |
| intensification_pressure_distance_from_range | Intensification | Ideal-range distance | 0.02355 | 0.08168 | 0.0033400000538676977 | 0.01228 | 0.00991 | 0.01053 |
| intensification_pressure_in_range | Intensification | Range flag | 0.01896 | 0.03637 | 0.02490999922156334 | 0.00963 | 0.00805 | 0.01582 |
| intensification_time_phase3_distance_from_range | Intensification | Ideal-range distance | 0.01611 | 0.04224 | 0.006169999949634075 | 0.00716 | 0.01127 | 0.0137 |
| fill_time_distance_from_range | Injection | Ideal-range distance | 0.01595 | 0.02275 | 0.012769999913871288 | 0.0144 | 0.01309 | 0.01674 |
| phase_transition_position_distance_from_range | Injection | Ideal-range distance | 0.01591 | 0.05381 | 0.0041600000113248825 | 0.0058 | 0.0079 | 0.0079 |
| cycle_time_in_range | Cooling | Range flag | 0.01581 | 0.0284 | 0.009990000165998936 | 0.01602 | 0.0096 | 0.01506 |
| phase_transition_position_in_range | Injection | Range flag | 0.0153 | 0.04807 | 0.005249999929219484 | 0.00672 | 0.00654 | 0.00991 |
| metal_velocity_gate_in_range | Injection | Range flag | 0.01411 | 0.02373 | 0.01083999965339899 | 0.01097 | 0.00928 | 0.01575 |
| solidification_ratio | Cooling | Ratio | 0.01238 | 0.00072 | 0.001509999972768128 | 0.00495 | 0.026 | 0.02872 |
| intensification_time_phase3_in_range | Intensification | Range flag | 0.01234 | 0.03181 | 0.003379999892786145 | 0.00666 | 0.00776 | 0.0121 |
| pressure_time_ratio | Intensification | Ratio | 0.01184 | 0.00072 | 0.0017600000137463212 | 0.00675 | 0.02584 | 0.02414 |
| fill_time_in_range | Injection | Range flag | 0.01178 | 0.01067 | 0.020320000126957893 | 0.01049 | 0.00681 | 0.01064 |
| plunger_lubricant | Configuration/Maintenance | Original | 0.01174 | 0.01756 | 0.010259999893605709 | 0.00863 | 0.01029 | 0.01195 |
| solidification_time_distance_from_range | Cooling | Ideal-range distance | 0.01171 | 0.02117 | 0.004170000087469816 | 0.00615 | 0.0133 | 0.01378 |

![Top 25 features](phase3_feature_importance_top25.png)
![By phase](phase3_importance_by_phase.png)

---

## Phase 3.3 — ROC curves and AUC

Official threshold strategy (item 3.2): **F-beta β=2 (recall-weighted)** — operating point marked on the curves of the most frequent defects.

### Micro-average AUC per model

| Model | AUC micro | Mean AUC per defect |
| --- | --- | --- |
| Neural network (MLP) | 0.9956 | 0.9950 |
| XGBoost | 0.9952 | 0.9947 |
| Random Forest | 0.9945 | 0.9940 |
| Logistic Regression (L2/Ridge) | 0.9929 | 0.9929 |
| Logistic Regression (L1/Lasso) | 0.9932 | 0.9893 |

### AUC-ROC per defect (10 most frequent)

| defect | test_positives | Neural network (MLP) | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) |
| --- | --- | --- | --- | --- | --- | --- |
| gas_porosity | 141 | 0.9946 | 0.994 | 0.9904 | 0.9937 | 0.9937 |
| density_deviation | 140 | 0.993 | 0.9922 | 0.9904 | 0.9945 | 0.9945 |
| cold_shut | 138 | 0.9943 | 0.9938 | 0.9913 | 0.9959 | 0.9959 |
| gas_bubbles | 129 | 0.9938 | 0.993 | 0.9909 | 0.9947 | 0.9948 |
| incomplete_fill | 105 | 0.9939 | 0.9939 | 0.9931 | 0.9948 | 0.9948 |
| surface_blisters | 93 | 0.991 | 0.9906 | 0.9887 | 0.9925 | 0.9925 |
| blisters_post_treatment | 92 | 0.9951 | 0.9946 | 0.9937 | 0.9954 | 0.9954 |
| volumetric_variation | 84 | 0.9966 | 0.9962 | 0.9965 | 0.9851 | 0.9871 |
| low_elongation | 84 | 0.9941 | 0.9937 | 0.9928 | 0.9938 | 0.9939 |
| dimensional_deviation | 83 | 0.9967 | 0.9964 | 0.9961 | 0.9963 | 0.9964 |

Full table: `figures/table_phase3_auc_by_defect.csv`

![ROC most frequent defects](phase3_roc_curves_top_defects.png)
![ROC micro-average](phase3_roc_micro_average.png)

---

## Phase 3.4 — Predicted × Actual

Adaptation for multi-label classification: (1) calibration diagrams (predicted probability vs. observed frequency) and (2) normalized confusion matrices with the official thresholds from item 3.2.

### Confusion matrices (NN, official thresholds)

| defect | threshold | tn | fp | fn | tp | recall | precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gas_porosity | 0.17000000000000004 | 4741 | 118 | 0 | 141 | 1.0 | 0.5444 |
| density_deviation | 0.4100000000000001 | 4763 | 97 | 0 | 140 | 1.0 | 0.5907 |
| cold_shut | 0.34 | 4771 | 91 | 4 | 134 | 0.971 | 0.5956 |

![Calibration by model](phase3_calibration_by_model.png)
![Calibration by defect](phase3_calibration_top_defects_nn.png)
![Normalized confusion](phase3_confusion_normalized_nn.png)

---
