# Fase 3.1 — Importância de features

Pergunta da banca: *as variáveis da etapa de injeção são as que mais impactam?*

**Resposta:** A fase com maior importância agregada é **Agregada/Global**. A fase de injeção concentra 24.6% da importância total média entre os 5 modelos.

## Importância agregada por fase do processo

| fase_processo | PyTorch NN | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) | importancia_media | n_features | process_phase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agregada/Global | 0.0556 | 0.7056000232696533 | 0.6276 | 0.053 | 0.064 | 0.3012 | 4 | Global/Aggregation |
| Injeção | 0.3788 | 0.09080000221729279 | 0.1481 | 0.3122 | 0.2996 | 0.2459 | 36 | Injection |
| Configuração/Manutenção | 0.1556 | 0.0966000035405159 | 0.0779 | 0.2695 | 0.2795 | 0.1758 | 35 | Configuration/Maintenance |
| Intensificação | 0.2564 | 0.04969999939203262 | 0.0699 | 0.1685 | 0.1666 | 0.1422 | 16 | Intensification |
| Resfriamento | 0.1536 | 0.042500000447034836 | 0.061 | 0.1392 | 0.1396 | 0.1072 | 14 | Cooling |
| Múltiplas fases | 0.0 | 0.01489999983459711 | 0.0155 | 0.0577 | 0.0507 | 0.0277 | 5 | Multiple phases |

## Importância agregada por categoria de feature engineering

| categoria_fe | PyTorch NN | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) | importancia_media | n_features | feature_category |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agregação estatística | 0.0556 | 0.7056000232696533 | 0.6276 | 0.053 | 0.064 | 0.3012 | 4 | Statistical aggregation |
| Distância de faixa ideal | 0.5572 | 0.1103999987244606 | 0.1833 | 0.279 | 0.3009 | 0.2862 | 28 | Ideal-range distance |
| Binária de faixa | 0.3632 | 0.07649999856948853 | 0.0854 | 0.2663 | 0.2536 | 0.209 | 42 | Range flag |
| Original | 0.0174 | 0.0544000007212162 | 0.0377 | 0.151 | 0.1426 | 0.0806 | 15 | Original |
| Ratio | 0.0043 | 0.016899999231100082 | 0.0229 | 0.103 | 0.101 | 0.0496 | 6 | Ratio |
| Diferença | 0.0 | 0.010900000110268593 | 0.0126 | 0.0472 | 0.0545 | 0.025 | 4 | Difference |
| Específica de domínio | 0.0016 | 0.014499999582767487 | 0.0132 | 0.0433 | 0.0318 | 0.0209 | 5 | Domain-specific |
| Produto | 0.0008 | 0.006599999964237213 | 0.0066 | 0.0294 | 0.0321 | 0.0151 | 2 | Product |
| Transformação matemática | 0.0 | 0.00430000014603138 | 0.0106 | 0.0278 | 0.0194 | 0.0124 | 4 | Mathematical transform |

## Top 20 features (média entre modelos)

| feature | fase_processo | categoria_fe | importancia_media | PyTorch NN | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n_vars_in_range | Agregada/Global | Agregação estatística | 0.11357 | 0.01682 | 0.4000900089740753 | 0.13559 | 0.00698 | 0.00836 |
| n_vars_out_of_range | Agregada/Global | Agregação estatística | 0.08524 | 0.03856 | 0.1998099982738495 | 0.17297 | 0.00698 | 0.00789 |
| max_distance_from_ideal | Agregada/Global | Agregação estatística | 0.07114 | 0.0 | 0.05948000028729439 | 0.21659 | 0.03344 | 0.04618 |
| avg_distance_from_ideal | Agregada/Global | Agregação estatística | 0.03121 | 0.00019 | 0.04619999974966049 | 0.10251 | 0.00561 | 0.00156 |
| metal_velocity_gate_distance_from_range | Injeção | Distância de faixa ideal | 0.02498 | 0.07809 | 0.0035099999513477087 | 0.0091 | 0.01623 | 0.01798 |
| intensification_pressure_distance_from_range | Intensificação | Distância de faixa ideal | 0.02485 | 0.08944 | 0.003659999929368496 | 0.01071 | 0.00991 | 0.01053 |
| cycle_time_distance_from_range | Resfriamento | Distância de faixa ideal | 0.02443 | 0.07546 | 0.011289999820291996 | 0.01124 | 0.01101 | 0.01314 |
| intensification_pressure_in_range | Intensificação | Binária de faixa | 0.02007 | 0.0555 | 0.012620000168681145 | 0.00836 | 0.00805 | 0.01582 |
| intensification_time_phase3_distance_from_range | Intensificação | Distância de faixa ideal | 0.01746 | 0.05286 | 0.003599999938160181 | 0.00589 | 0.01127 | 0.0137 |
| fill_time_distance_from_range | Injeção | Distância de faixa ideal | 0.0166 | 0.03276 | 0.008009999990463257 | 0.01239 | 0.01309 | 0.01674 |
| phase_transition_position_distance_from_range | Injeção | Distância de faixa ideal | 0.01628 | 0.05579 | 0.0037899999879300594 | 0.00605 | 0.0079 | 0.0079 |
| phase_transition_position_in_range | Injeção | Binária de faixa | 0.01319 | 0.04072 | 0.00267999991774559 | 0.00606 | 0.00654 | 0.00991 |
| solidification_time_distance_from_range | Resfriamento | Distância de faixa ideal | 0.01269 | 0.03063 | 0.002309999894350767 | 0.00346 | 0.0133 | 0.01378 |
| solidification_ratio | Resfriamento | Ratio | 0.01234 | 0.00156 | 0.002139999996870756 | 0.00329 | 0.026 | 0.02872 |
| piston_velocity_phase1_distance_from_range | Injeção | Distância de faixa ideal | 0.0123 | 0.03823 | 0.0011699999449774623 | 0.00484 | 0.00875 | 0.00851 |
| plunger_lubricant | Configuração/Manutenção | Original | 0.01213 | 0.01425 | 0.017100000753998756 | 0.00705 | 0.01029 | 0.01195 |
| sleeve_length_distance_from_range | Configuração/Manutenção | Distância de faixa ideal | 0.01174 | 0.03362 | 0.003280000062659383 | 0.00229 | 0.00934 | 0.01018 |
| fill_time_in_range | Injeção | Binária de faixa | 0.01167 | 0.02041 | 0.01018999982625246 | 0.01029 | 0.00681 | 0.01064 |
| pressure_time_ratio | Intensificação | Ratio | 0.01162 | 0.0 | 0.00343000004068017 | 0.0047 | 0.02584 | 0.02414 |
| cycle_time_in_range | Resfriamento | Binária de faixa | 0.01122 | 0.01722 | 0.005150000099092722 | 0.00906 | 0.0096 | 0.01506 |

![Top 25 features](phase3_feature_importance_top25.png)
![Por fase](phase3_importance_by_phase.png)
