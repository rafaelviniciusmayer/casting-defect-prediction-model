# Phase 3.4 — Predicted × Actual

Adaptation for multi-label classification: (1) calibration diagrams (predicted probability vs. observed frequency) and (2) normalized confusion matrices with the official thresholds from item 3.2.

## Confusion matrices (NN, official thresholds)

| defect | threshold | tn | fp | fn | tp | recall | precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gas_porosity | 0.17000000000000004 | 4741 | 118 | 0 | 141 | 1.0 | 0.5444 |
| density_deviation | 0.4100000000000001 | 4763 | 97 | 0 | 140 | 1.0 | 0.5907 |
| cold_shut | 0.34 | 4771 | 91 | 4 | 134 | 0.971 | 0.5956 |

![Calibration by model](phase3_calibration_by_model.png)
![Calibration by defect](phase3_calibration_top_defects_nn.png)
![Normalized confusion](phase3_confusion_normalized_nn.png)
