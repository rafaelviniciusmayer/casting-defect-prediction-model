# 4.4.5. Classification Performance Results (revised — aligned with `table10_confusion_matrix_metrics.csv`)

**Important:** The numbers below match **`figures/table10_confusion_matrix_metrics.csv`**, produced by `generate_table_confusion_matrix_metrics.py` (saved model, test set *n* = 5,000, same split as `train_model.py`).  

If your **Table 12** in Word/PDF still shows rows such as Density Deviation TP = 132 or Gas Bubbles TP = 112, that version is **not** the same as the current CSV in the repository—**re-export Table 12 from this file** to avoid contradictions.

---

### 4.4.5. Classification Performance Results

The confusion matrix analysis for the most frequent defects demonstrates the trade-off between recall and precision that characterizes the optimized model (Cuartas et al., 2021; Dettori et al., 2024; Sala et al., 2023a). **Table 12** presents confusion matrix metrics for **selected high-frequency defect types** on the test set (full per-defect metrics for all **28** types are available in **Appendix III** / `table10_confusion_matrix_metrics.csv`). The metrics include: **True Positives (TP)**, defective parts correctly identified as defective; **True Negatives (TN)**, defect-free parts correctly identified as defect-free; **False Positives (FP)**, defect-free parts incorrectly flagged as defective; and **False Negatives (FN)**, defective parts incorrectly classified as defect-free.

For **gas porosity** (the most frequent defect with **141** occurrences in the test set), the model detected **140** actual defects (**recall 0.993**), missing only one. Of **260** samples classified as defective for this label, **140** were correct and **120** were false positives (**precision 0.538**). The **4,739** true negatives represent **94.8%** of the test set correctly identified as defect-free for this specific defect type.

The **overall** row in the full confusion-matrix table presents **micro-averaged** metrics aggregated across **all 28** defect types and all **5,000** test samples. Micro-averaging sums TP, FP, and FN across all defect dimensions before computing precision and recall, weighting each binary decision equally (Sokolova and Lapalme, 2009). In the current export, this yields **TP = 1,809**, **TN = 136,356**, **FP = 1,788**, **FN = 47**, **recall = 0.975**, and **precision = 0.503**.

For **cold shut** (**138** occurrences in the test set), the model achieved **perfect recall (1.000)**, identifying all **138** defects with **zero** false negatives. Among **233** samples predicted as cold-shut defective, **138** were correct and **95** were false positives (**precision 0.592**). The perfect recall indicates strong separation for this label under the chosen threshold, consistent with process variables that correlate with incomplete filling and cold shut (Gupta et al., 2021; Lee et al., 2018).

For **incomplete fill** (**105** occurrences in the test set), the model detected **104** defects (**recall 0.990**), missing one. Of **202** samples classified as defective, **104** were correct and **98** were false positives (**precision 0.515**). The high recall with moderate precision reflects the operational priority of avoiding missed defects (Sala, Deyne, et al., 2023).

For **gas bubbles** (**129** positives in the test set; **626** in the full engineered dataset), the model achieved **recall 1.000** and **precision 0.547** (**129** TP, **0** FN, **107** FP). For **internal shrinkage** (**42** positives in the test set), the model achieved **recall 1.000** and **precision 0.452** (**42** TP, **0** FN, **51** FP). These two labels illustrate that **very high recall** can still coexist with **moderate precision**, depending on score separation and threshold; they **do not** match the older illustrative values (e.g., recall **0.973** / **0.952**) sometimes copied from an outdated table draft.

The pattern is consistent across defect types: thresholds are chosen to **prioritize recall**, so precision reflects class imbalance and the cost structure favoring detection over false alarms. Derived from the same per-defect confusion matrix export, **macro-averaged F1** (unweighted mean of per-defect F1 scores) is approximately **0.609**, while **micro-averaged F1** implied by the overall micro recall/precision above is approximately **0.664**.  

*Optional alignment with `reports/model_comparison_report.json` (PyTorch NN on the same test protocol): micro recall **0.9763**, micro precision **0.5054**, F1-micro **0.6661**, F1-macro **0.6116**—these can differ slightly from the confusion-matrix CSV footers after retraining or rounding; use **one** official source per submission version.*

The difference between micro and macro F1 indicates that performance is somewhat stronger for frequent defects than for rare ones, as expected when positive supervision is uneven (Sokolova and Lapalme, 2009).

---

### Checklist vs. your previous draft

| Claim | Previous text / image table | Correct (current `table10`) |
|-------|-----------------------------|------------------------------|
| Gas porosity | OK | OK |
| Cold shut | OK | OK |
| Incomplete fill | OK | OK |
| Gas bubbles test *n* | ~115 | **129** positives |
| Gas bubbles R / P | 0.973 / 0.487 | **1.000 / 0.547** |
| Internal shrinkage R / P | 0.952 / 0.421 | **1.000 / 0.452** |
| Overall micro TP… | 1,374 / … (subset) | **1,809 / 136,356 / 1,788 / 47** |
| Overall R / P | 0.976 / 0.505 (subset) | **0.975 / 0.503** (full 28 labels) |
