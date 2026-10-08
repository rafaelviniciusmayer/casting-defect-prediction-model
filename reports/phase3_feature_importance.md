# Fase 3.1 — Importância de features

Pergunta da banca: *as variáveis da etapa de injeção são as que mais impactam?*

**Resposta:** A fase com maior importância agregada é **Agregada/Global**. A fase de injeção concentra 25.2% da importância total média entre os 5 modelos.

## Importância agregada por fase do processo

| fase_processo | PyTorch NN | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) | importancia_media | n_features | process_phase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agregada/Global | 0.046 | 0.6635000109672546 | 0.5719 | 0.053 | 0.064 | 0.2797 | 4 | Global/Aggregation |
| Injeção | 0.3818 | 0.10890000313520432 | 0.1571 | 0.3122 | 0.2996 | 0.2519 | 36 | Injection |
| Configuração/Manutenção | 0.1331 | 0.09359999746084213 | 0.0658 | 0.2695 | 0.2795 | 0.1683 | 35 | Configuration/Maintenance |
| Intensificação | 0.2537 | 0.06549999862909317 | 0.0844 | 0.1685 | 0.1666 | 0.1477 | 16 | Intensification |
| Resfriamento | 0.1814 | 0.05939999967813492 | 0.0996 | 0.1392 | 0.1396 | 0.1238 | 14 | Cooling |
| Múltiplas fases | 0.004 | 0.009200000204145908 | 0.0211 | 0.0577 | 0.0507 | 0.0285 | 5 | Multiple phases |

## Importância agregada por categoria de feature engineering

| categoria_fe | PyTorch NN | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) | importancia_media | n_features | feature_category |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Distância de faixa ideal | 0.5204 | 0.11659999936819077 | 0.1978 | 0.279 | 0.3009 | 0.283 | 28 | Ideal-range distance |
| Agregação estatística | 0.046 | 0.6635000109672546 | 0.5719 | 0.053 | 0.064 | 0.2797 | 4 | Statistical aggregation |
| Binária de faixa | 0.3859 | 0.15039999783039093 | 0.1044 | 0.2663 | 0.2536 | 0.2321 | 42 | Range flag |
| Original | 0.0303 | 0.03139999881386757 | 0.0412 | 0.151 | 0.1426 | 0.0793 | 15 | Original |
| Ratio | 0.0043 | 0.011099999770522118 | 0.0295 | 0.103 | 0.101 | 0.0498 | 6 | Ratio |
| Diferença | 0.0025 | 0.007499999832361937 | 0.0162 | 0.0472 | 0.0545 | 0.0256 | 4 | Difference |
| Específica de domínio | 0.0043 | 0.009499999694526196 | 0.017 | 0.0433 | 0.0318 | 0.0212 | 5 | Domain-specific |
| Produto | 0.0 | 0.004100000020116568 | 0.0088 | 0.0294 | 0.0321 | 0.0149 | 2 | Product |
| Transformação matemática | 0.0061 | 0.005900000222027302 | 0.0133 | 0.0278 | 0.0194 | 0.0145 | 4 | Mathematical transform |

## Top 20 features (média entre modelos)

| feature | fase_processo | categoria_fe | importancia_media | PyTorch NN | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n_vars_out_of_range | Agregada/Global | Agregação estatística | 0.09845 | 0.00964 | 0.2995699942111969 | 0.16818 | 0.00698 | 0.00789 |
| n_vars_in_range | Agregada/Global | Agregação estatística | 0.09557 | 0.02896 | 0.2989799976348877 | 0.13458 | 0.00698 | 0.00836 |
| max_distance_from_ideal | Agregada/Global | Agregação estatística | 0.05886 | 0.00543 | 0.04019000008702278 | 0.16906 | 0.03344 | 0.04618 |
| cycle_time_distance_from_range | Resfriamento | Distância de faixa ideal | 0.02881 | 0.08169 | 0.01906999945640564 | 0.01915 | 0.01101 | 0.01314 |
| avg_distance_from_ideal | Agregada/Global | Agregação estatística | 0.02679 | 0.00193 | 0.024720000103116035 | 0.1001 | 0.00561 | 0.00156 |
| metal_velocity_gate_distance_from_range | Injeção | Distância de faixa ideal | 0.02508 | 0.07448 | 0.005570000037550926 | 0.01113 | 0.01623 | 0.01798 |
| intensification_pressure_distance_from_range | Intensificação | Distância de faixa ideal | 0.02355 | 0.08168 | 0.0033400000538676977 | 0.01228 | 0.00991 | 0.01053 |
| intensification_pressure_in_range | Intensificação | Binária de faixa | 0.01896 | 0.03637 | 0.02490999922156334 | 0.00963 | 0.00805 | 0.01582 |
| intensification_time_phase3_distance_from_range | Intensificação | Distância de faixa ideal | 0.01611 | 0.04224 | 0.006169999949634075 | 0.00716 | 0.01127 | 0.0137 |
| fill_time_distance_from_range | Injeção | Distância de faixa ideal | 0.01595 | 0.02275 | 0.012769999913871288 | 0.0144 | 0.01309 | 0.01674 |
| phase_transition_position_distance_from_range | Injeção | Distância de faixa ideal | 0.01591 | 0.05381 | 0.0041600000113248825 | 0.0058 | 0.0079 | 0.0079 |
| cycle_time_in_range | Resfriamento | Binária de faixa | 0.01581 | 0.0284 | 0.009990000165998936 | 0.01602 | 0.0096 | 0.01506 |
| phase_transition_position_in_range | Injeção | Binária de faixa | 0.0153 | 0.04807 | 0.005249999929219484 | 0.00672 | 0.00654 | 0.00991 |
| metal_velocity_gate_in_range | Injeção | Binária de faixa | 0.01411 | 0.02373 | 0.01083999965339899 | 0.01097 | 0.00928 | 0.01575 |
| solidification_ratio | Resfriamento | Ratio | 0.01238 | 0.00072 | 0.001509999972768128 | 0.00495 | 0.026 | 0.02872 |
| intensification_time_phase3_in_range | Intensificação | Binária de faixa | 0.01234 | 0.03181 | 0.003379999892786145 | 0.00666 | 0.00776 | 0.0121 |
| pressure_time_ratio | Intensificação | Ratio | 0.01184 | 0.00072 | 0.0017600000137463212 | 0.00675 | 0.02584 | 0.02414 |
| fill_time_in_range | Injeção | Binária de faixa | 0.01178 | 0.01067 | 0.020320000126957893 | 0.01049 | 0.00681 | 0.01064 |
| plunger_lubricant | Configuração/Manutenção | Original | 0.01174 | 0.01756 | 0.010259999893605709 | 0.00863 | 0.01029 | 0.01195 |
| solidification_time_distance_from_range | Resfriamento | Distância de faixa ideal | 0.01171 | 0.02117 | 0.004170000087469816 | 0.00615 | 0.0133 | 0.01378 |

![Top 25 features](phase3_feature_importance_top25.png)
![Por fase](phase3_importance_by_phase.png)
