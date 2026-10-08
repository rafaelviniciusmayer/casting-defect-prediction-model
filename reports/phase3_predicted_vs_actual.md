# Fase 3.4 — Previsto × Realizado

Adaptação para classificação multi-label: (1) diagramas de calibração (probabilidade prevista vs frequência real) e (2) matrizes de confusão normalizadas com os thresholds oficiais do item 3.2.

## Matrizes de confusão (NN, thresholds oficiais)

| defect | threshold | tn | fp | fn | tp | recall | precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gas_porosity | 0.17000000000000004 | 4741 | 118 | 0 | 141 | 1.0 | 0.5444 |
| density_deviation | 0.4100000000000001 | 4763 | 97 | 0 | 140 | 1.0 | 0.5907 |
| cold_shut | 0.34 | 4771 | 91 | 4 | 134 | 0.971 | 0.5956 |

![Calibração por modelo](phase3_calibration_by_model.png)
![Calibração por defeito](phase3_calibration_top_defects_nn.png)
![Confusão normalizada](phase3_confusion_normalized_nn.png)
