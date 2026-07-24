# Fase 3.4 — Previsto × Realizado

Adaptação para classificação multi-label: (1) diagramas de calibração (probabilidade prevista vs frequência real) e (2) matrizes de confusão normalizadas com os thresholds oficiais do item 3.2.

## Matrizes de confusão (NN, thresholds oficiais)

| defect | threshold | tn | fp | fn | tp | recall | precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gas_porosity | 0.05 | 4739 | 120 | 0 | 141 | 1.0 | 0.5402 |
| density_deviation | 0.1 | 4765 | 95 | 0 | 140 | 1.0 | 0.5957 |
| cold_shut | 0.21000000000000002 | 4766 | 96 | 0 | 138 | 1.0 | 0.5897 |

![Calibração por modelo](phase3_calibration_by_model.png)
![Calibração por defeito](phase3_calibration_top_defects_nn.png)
![Confusão normalizada](phase3_confusion_normalized_nn.png)
