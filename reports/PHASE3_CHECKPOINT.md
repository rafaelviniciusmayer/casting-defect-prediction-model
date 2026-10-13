# Checkpoint — Phase 3 (diagnostics and interpretation)

- **Official threshold strategy: F-beta β=2 (recall-weighted)** (Highest F-beta(2)-micro of the production neural network on out-of-fold development predictions; thresholds chosen on development data only.)
- Feature importance: The phase with the highest aggregated importance is **Global/Aggregation**. The injection phase accounts for 25.2% of the total importance, averaged across the 5 models.
- ROC/AUC curves and predicted×actual generated with the official thresholds.

## Generated files

- `reports/phase3_feature_importance.md` (3.1)
- `reports/oof_threshold_results.json` and `reports/phase3_official_thresholds.json` (3.2)
- `reports/phase3_roc_analysis.md` (3.3)
- `reports/phase3_predicted_vs_actual.md` (3.4)
- Corresponding figures and CSVs in `figures/`
