# Table 10. Confusion Matrix Metrics per Defect Type (Test Set, n=5,000)

*Per-defect thresholds of the official strategy, selected on out-of-fold development predictions (the test set is not used to choose them).*

**Acronyms:**
- **TP (True Positives):** Correctly predicted defects.
- **TN (True Negatives):** Correctly predicted non-defects.
- **FP (False Positives):** Type I error — non-defects predicted as defects.
- **FN (False Negatives):** Type II error — defects predicted as non-defects.
- **Recall:** TP / (TP + FN); proportion of actual defects correctly identified (sensitivity).
- **Precision:** TP / (TP + FP); proportion of positive predictions that are correct (positive predictive value).

| Defect Type | TP | TN | FP | FN | Recall | Precision |
|-------------|----|----|----|----|--------|-----------|
| Blisters Post Treatment | 91 | 4813 | 95 | 1 | 0.989 | 0.489 |
| Surface Blisters | 88 | 4794 | 113 | 5 | 0.946 | 0.438 |
| Die Sticking | 55 | 4859 | 86 | 0 | 1.000 | 0.390 |
| Flow Lines | 49 | 4889 | 61 | 1 | 0.980 | 0.445 |
| Surface Streaks | 6 | 4976 | 17 | 1 | 0.857 | 0.261 |
| Cold Shut | 134 | 4771 | 91 | 4 | 0.971 | 0.596 |
| Heat Cracks | 8 | 4962 | 30 | 0 | 1.000 | 0.211 |
| Ejector Pin Marks | 0 | 5000 | 0 | 0 | n/a | n/a |
| Die Soldering | 67 | 4858 | 75 | 0 | 1.000 | 0.472 |
| Surface Oxide Inclusions | 33 | 4914 | 46 | 7 | 0.825 | 0.418 |
| Low Tensile Strength | 80 | 4842 | 78 | 0 | 1.000 | 0.506 |
| Low Elongation | 82 | 4843 | 73 | 2 | 0.976 | 0.529 |
| Low Ultimate Strength | 70 | 4845 | 85 | 0 | 1.000 | 0.452 |
| Low Fatigue Resistance | 76 | 4854 | 70 | 0 | 1.000 | 0.521 |
| Low Surface Hardness | 63 | 4889 | 46 | 2 | 0.969 | 0.578 |
| Density Deviation | 140 | 4763 | 97 | 0 | 1.000 | 0.591 |
| Incomplete Fill | 100 | 4802 | 93 | 5 | 0.952 | 0.518 |
| Flash | 48 | 4857 | 94 | 1 | 0.980 | 0.338 |
| Warpage | 37 | 4881 | 82 | 0 | 1.000 | 0.311 |
| Shrinkage Porosity | 17 | 4960 | 20 | 3 | 0.850 | 0.459 |
| Volumetric Variation | 82 | 4863 | 53 | 2 | 0.976 | 0.607 |
| Dimensional Deviation | 82 | 4858 | 59 | 1 | 0.988 | 0.582 |
| Gas Porosity | 141 | 4741 | 118 | 0 | 1.000 | 0.544 |
| Gas Bubbles | 129 | 4769 | 102 | 0 | 1.000 | 0.558 |
| Internal Shrinkage | 38 | 4914 | 44 | 4 | 0.905 | 0.463 |
| Cracks | 31 | 4936 | 33 | 0 | 1.000 | 0.484 |
| Hard Inclusions | 13 | 4944 | 35 | 8 | 0.619 | 0.271 |
| Oxide Inclusions | 45 | 4889 | 62 | 4 | 0.918 | 0.421 |
| Overall (micro-avg) | 1805 | 136286 | 1858 | 51 | 0.973 | 0.493 |