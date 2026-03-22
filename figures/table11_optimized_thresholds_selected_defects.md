# Table 11. Optimized Thresholds and Performance for Selected Defects (Test Set, *n* = 5,000)

**Optimal threshold** values come from `model_analysis_report.json` (same recall-oriented optimization as `train_model.py`). **Recall**, **Precision**, and **F1-Score** are computed from the test-set confusion matrix with those thresholds applied (`figures/table10_confusion_matrix_metrics.csv`).  
F1 = 2·Precision·Recall / (Precision + Recall). Ejector pin marks has no positive test instances; precision/recall/F1 are not defined for that label.

| Defect Type | Optimal Threshold | Recall | Precision | F1-Score |
|-------------|-------------------|--------|-----------|----------|
| Gas Porosity | 0.21 | 0.993 | 0.538 | 0.698 |
| Density Deviation | 0.18 | 1.000 | 0.601 | 0.751 |
| Cold Shut | 0.31 | 1.000 | 0.592 | 0.744 |
| Gas Bubbles | 0.11 | 1.000 | 0.547 | 0.707 |
| Incomplete Fill | 0.24 | 0.990 | 0.515 | 0.677 |
| Shrinkage Porosity | 0.10 | 0.850 | 0.486 | 0.618 |
| Low Tensile Strength | 0.21 | 0.962 | 0.517 | 0.672 |
| Warpage | 0.15 | 0.946 | 0.321 | 0.479 |
| Ejector Pin Marks | 0.50 | 0.000 | 0.000 | — |

*Note: Threshold 0.50 for ejector pin marks is the default when no positive samples exist for that defect in the split used for optimization.*
