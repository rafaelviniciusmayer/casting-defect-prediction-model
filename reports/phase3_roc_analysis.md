# Fase 3.3 — Curvas ROC e AUC

Estratégia oficial de threshold (item 3.2): **F-beta β=2 (recall-weighted)** — ponto de operação marcado nas curvas dos defeitos mais frequentes.

## AUC micro-average por modelo

| Model | AUC micro | Mean AUC per defect |
| --- | --- | --- |
| PyTorch NN | 0.9955 | 0.9951 |
| XGBoost | 0.9952 | 0.9947 |
| Random Forest | 0.9945 | 0.9940 |
| Logistic Regression (L2/Ridge) | 0.9929 | 0.9929 |
| Logistic Regression (L1/Lasso) | 0.9932 | 0.9893 |

## AUC-ROC por defeito (10 mais frequentes)

| defect | test_positives | PyTorch NN | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) |
| --- | --- | --- | --- | --- | --- | --- |
| gas_porosity | 141 | 0.9941 | 0.994 | 0.9904 | 0.9937 | 0.9937 |
| density_deviation | 140 | 0.9932 | 0.9922 | 0.9904 | 0.9945 | 0.9945 |
| cold_shut | 138 | 0.9951 | 0.9938 | 0.9913 | 0.9959 | 0.9959 |
| gas_bubbles | 129 | 0.9938 | 0.993 | 0.9909 | 0.9947 | 0.9948 |
| incomplete_fill | 105 | 0.9942 | 0.9939 | 0.9931 | 0.9948 | 0.9948 |
| surface_blisters | 93 | 0.9916 | 0.9906 | 0.9887 | 0.9925 | 0.9925 |
| blisters_post_treatment | 92 | 0.9951 | 0.9946 | 0.9937 | 0.9954 | 0.9954 |
| volumetric_variation | 84 | 0.9961 | 0.9962 | 0.9965 | 0.9851 | 0.9871 |
| low_elongation | 84 | 0.9941 | 0.9937 | 0.9928 | 0.9938 | 0.9939 |
| dimensional_deviation | 83 | 0.9967 | 0.9964 | 0.9961 | 0.9963 | 0.9964 |

Tabela completa: `figures/table_phase3_auc_by_defect.csv`

![ROC defeitos frequentes](phase3_roc_curves_top_defects.png)
![ROC micro-average](phase3_roc_micro_average.png)
