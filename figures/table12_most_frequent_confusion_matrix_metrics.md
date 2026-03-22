# Table 12. Confusion Matrix Metrics — 8 Most Frequent Defect Types (Test Set, *n* = 5,000)

*Selection criterion:* highest number of **positive** test instances (**TP + FN**) among the 28 labels. Full per-defect table (all 28 + overall): `figures/table10_confusion_matrix_metrics.csv`.*

| Defect Type | TP | TN | FP | FN | Recall | Precision |
|-------------|----|----|----|----|--------|-----------|
| Gas Porosity | 140 | 4739 | 120 | 1 | 0.993 | 0.538 |
| Density Deviation | 140 | 4767 | 93 | 0 | 1.000 | 0.601 |
| Cold Shut | 138 | 4767 | 95 | 0 | 1.000 | 0.592 |
| Gas Bubbles | 129 | 4764 | 107 | 0 | 1.000 | 0.547 |
| Incomplete Fill | 104 | 4797 | 98 | 1 | 0.990 | 0.515 |
| Surface Blisters | 91 | 4795 | 112 | 2 | 0.978 | 0.448 |
| Blisters Post Treatment | 92 | 4815 | 93 | 0 | 1.000 | 0.497 |
| Low Elongation | 82 | 4845 | 71 | 2 | 0.976 | 0.536 |
| **Overall (micro-avg, all 28 defects)** | 1809 | 136356 | 1788 | 47 | 0.975 | 0.503 |

*The overall row sums TP/TN/FP/FN across **all 28** binary defect heads (5,000 × 28 decisions), not only the eight rows above.*