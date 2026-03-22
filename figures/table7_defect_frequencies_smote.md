# Table 7. Defect Frequencies and SMOTE Balancing Effect (Training Set, n = 20,000)

After SMOTE augmentation, the training matrix has **n = 42,350** rows (same pipeline as `train_model.py`). Frequencies after SMOTE are **positives for that defect / n_after** (multi-label rows may count in several defects).

| Defect Type | Original Count | Original Freq. (%) | After SMOTE Count | After SMOTE Freq. (%) | Ratio Before → After |
|-------------|----------------|--------------------|-------------------|-----------------------|----------------------|
| Gas Porosity | 644 | 3.22 | 15650 | 36.95 | 1:30 → 1:1.7 |
| Density Deviation | 596 | 2.98 | 14450 | 34.12 | 1:33 → 1:1.9 |
| Cold Shut | 560 | 2.80 | 12981 | 30.65 | 1:35 → 1:2.3 |
| Gas Bubbles | 497 | 2.48 | 11755 | 27.76 | 1:39 → 1:2.6 |
| Incomplete Fill | 416 | 2.08 | 9357 | 22.09 | 1:47 → 1:3.5 |
| Blisters Post Treatment | 380 | 1.90 | 9054 | 21.38 | 1:52 → 1:3.7 |
| Flow Lines | 213 | 1.06 | 5399 | 12.75 | 1:93 → 1:6.8 |
| Shrinkage Porosity | 73 | 0.36 | 1653 | 3.90 | 1:273 → 1:25 |

*Ratio Before → After: expressed as **1 positive : X negatives** for the binary view of each defect (minority : majority), before SMOTE vs. after SMOTE.*

*Synthetic rows added: 22,350 (total training size after SMOTE: 42,350).*