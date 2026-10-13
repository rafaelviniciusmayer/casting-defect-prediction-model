# Checkpoint — Phase 1 (Validation infrastructure)

## Decisions taken

- **SMOTE: REMOVED** in the official pipeline configuration (mean F1-micro gain -0.0076, recall -0.1450, train-val gap increase +0.0693).
- **Unified CV**: the 3 models now use the same stratified CV with 5 folds (random_state=42) over the dev set (80%), with SMOTE (when enabled) applied only to the training fold.
- **Cost-sensitive learning** active in all models: pos_weight in the NN, class_weight='balanced' in RF, scale_pos_weight in XGBoost (new in this phase — XGBoost previously had no weighting).
- Best recall (CV validation, with SMOTE): **Neural network (MLP)** (0.8665).
- Best F1-micro (CV validation, with SMOTE): **Neural network (MLP)** (0.6737).

## Reference for the next phases

- Phase 2 (new models) must use `unified_cv_pipeline.run_cross_validation` with `use_smote=False`.
- Phase 3 (thresholds/ROC) must start from the final models trained via `unified_cv_pipeline.run_final_test_evaluation` under the same condition.

## Generated files

- `reports/phase1_cv_comparison.md` (1.1)
- `reports/phase1_overfitting_analysis.md` (1.2)
- `reports/phase1_smote_comparison.md` (1.3)
- `reports/phase1_smote_overfitting.md` (1.4)
- `figures/table_phase1_overfitting.csv`
- `figures/table_phase1_smote_comparison.csv`
- `figures/table_phase1_train_vs_test.csv`
- `figures/phase1_learning_curves.png`
