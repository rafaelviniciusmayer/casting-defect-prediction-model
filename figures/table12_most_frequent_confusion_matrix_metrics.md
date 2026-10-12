# Table 12. Confusion Matrix Metrics — 8 Most Frequent Defect Types (Test Set, *n* = 5,000)

*Selection criterion:* highest number of **positive** test instances (**TP + FN**) among the 28 labels. Thresholds: official per-defect thresholds selected on out-of-fold development predictions (`threshold_selection_oof.py`), never on the test set. Full per-defect table (all 28): `figures/table10_confusion_matrix_metrics.csv`.*

| Defect Type | TP | TN | FP | FN | Recall | Precision |
|-------------|----|----|----|----|--------|-----------|
| Gas Porosity | 141 | 4741 | 118 | 0 | 1.000 | 0.544 |
| Density Deviation | 140 | 4763 | 97 | 0 | 1.000 | 0.591 |
| Cold Shut | 134 | 4771 | 91 | 4 | 0.971 | 0.596 |
| Gas Bubbles | 129 | 4769 | 102 | 0 | 1.000 | 0.558 |
| Incomplete Fill | 100 | 4802 | 93 | 5 | 0.952 | 0.518 |
| Surface Blisters | 88 | 4794 | 113 | 5 | 0.946 | 0.438 |
| Blisters Post Treatment | 91 | 4813 | 95 | 1 | 0.989 | 0.489 |
| Low Elongation | 82 | 4843 | 73 | 2 | 0.976 | 0.529 |
| **Overall (micro-avg, all 28 defects)** | 1805 | 136286 | 1858 | 51 | 0.973 | 0.493 |

*The overall row sums TP/TN/FP/FN across **all 28** binary defect heads (5,000 × 28 decisions), not only the eight rows above.*