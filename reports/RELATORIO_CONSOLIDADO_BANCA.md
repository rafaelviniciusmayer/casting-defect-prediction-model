# Relatório consolidado — Ajustes solicitados pela banca

Gerado em 2026-10-09 10:43. Pipeline: split 80/20 estratificado (random_state=42), CV 5-fold unificada, configuração oficial definida na Fase 1.

---


# Decisões (checkpoints)

## Checkpoint — Fase 1 (Infraestrutura de validação)

### Decisões tomadas

- **SMOTE: REMOVIDO** na configuração oficial do pipeline (ganho médio F1-micro -0.0076, recall -0.1450, aumento de gap treino-val +0.0693).
- **CV unificada**: os 3 modelos passam a usar a mesma CV estratificada em 5 folds (random_state=42) sobre o dev set (80%), com SMOTE (quando habilitado) aplicado apenas no fold de treino.
- **Cost-sensitive learning** ativo em todos os modelos: pos_weight na NN, class_weight='balanced' no RF, scale_pos_weight no XGBoost (novidade desta fase — antes o XGBoost não tinha ponderação).
- Melhor recall (validação CV, com SMOTE): **Neural network (MLP)** (0.8665).
- Melhor F1-micro (validação CV, com SMOTE): **Neural network (MLP)** (0.6737).

### Referência para as próximas fases

- Fase 2 (novos modelos) deve usar `unified_cv_pipeline.run_cross_validation` com `use_smote=False`.
- Fase 3 (thresholds/ROC) deve partir dos modelos finais treinados via `unified_cv_pipeline.run_final_test_evaluation` na mesma condição.

### Arquivos gerados

- `reports/phase1_cv_comparison.md` (1.1)
- `reports/phase1_overfitting_analysis.md` (1.2)
- `reports/phase1_smote_comparison.md` (1.3)
- `reports/phase1_smote_overfitting.md` (1.4)
- `figures/table_phase1_overfitting.csv`
- `figures/table_phase1_smote_comparison.csv`
- `figures/table_phase1_train_vs_test.csv`
- `figures/phase1_learning_curves.png`

---

## Checkpoint — Fase 2 (modelos estatísticos)

- Condição oficial: sem SMOTE (herdada da Fase 1).
- Modelos comparados: PyTorch NN, XGBoost, Random Forest, Regressão Logística L2 (Ridge), Regressão Logística L1 (Lasso).
- Melhor recall no teste (thresholds otimizados): **Random Forest** (0.9952).
- Melhor F1-micro no teste: **Logistic Regression (L2/Ridge)** (0.6495).
- A Fase 3 deve usar estes 4-5 modelos finais para importância de features (3.1) e as probabilidades de teste para thresholds/ROC (3.2/3.3).

---

## Checkpoint — Fase 3 (diagnósticos e interpretação)

- **Estratégia oficial de threshold: F-beta β=2 (recall-weighted)** (Highest F-beta(2)-micro of the production neural network on out-of-fold development predictions; thresholds chosen on development data only.)
- Importância de features: A fase com maior importância agregada é **Agregada/Global**. A fase de injeção concentra 25.2% da importância total média entre os 5 modelos.
- Curvas ROC/AUC e previsto×realizado gerados com os thresholds oficiais.

### Arquivos gerados

- `reports/phase3_feature_importance.md` (3.1)
- `reports/oof_threshold_results.json` e `reports/phase3_official_thresholds.json` (3.2)
- `reports/phase3_roc_analysis.md` (3.3)
- `reports/phase3_predicted_vs_actual.md` (3.4)
- Figuras e CSVs correspondentes em `figures/`

---


# Fase 1 — Infraestrutura de validação

## Fase 1.1 — Comparação por Cross-Validation (5 folds)

Métricas médias ± desvio-padrão na **validação** de cada fold (threshold fixo 0.5). SMOTE aplicado apenas no treino de cada fold; teste final (20%) permanece isolado.

| Modelo | Recall (val) | Precision (val) | F1-micro (val) | F1-macro (val) | Tempo CV (s) |
|--------|--------------|-----------------|----------------|----------------|--------------|
| Neural network (MLP) | 0.8665 ± 0.0090 | 0.5512 ± 0.0089 | 0.6737 ± 0.0082 | 0.6321 ± 0.0130 | 2266.3 |
| XGBoost | 0.7377 ± 0.0093 | 0.5950 ± 0.0087 | 0.6587 ± 0.0084 | 0.5985 ± 0.0130 | 315.1 |
| Random Forest | 0.7127 ± 0.0116 | 0.5990 ± 0.0064 | 0.6508 ± 0.0024 | 0.5907 ± 0.0149 | 420.9 |

---

## Phase 1.2 — Overfitting Analysis (with SMOTE)

Fixed threshold 0.5 across all columns for direct comparison between train (CV), validation (CV), and final test. Train metrics are computed on original fold data (no synthetic SMOTE samples).

Gap Train-Val > 0 indicates possible overfitting within folds. Gap Val-Test > 0 indicates performance drop on the holdout set.

| Model | Metric | Train (CV) | Validation (CV) | Test | Gap Train-Val | Gap Val-Test |
| --- | --- | --- | --- | --- | --- | --- |
| Neural network (MLP) | Recall | 0.9782 ± 0.0024 | 0.8665 ± 0.0090 | 0.8497 | 0.1117 | 0.0168 |
| Neural network (MLP) | Precision | 0.6008 ± 0.0040 | 0.5512 ± 0.0089 | 0.5391 | 0.0496 | 0.0121 |
| Neural network (MLP) | F1 | 0.7444 ± 0.0028 | 0.6737 ± 0.0082 | 0.6597 | 0.0707 | 0.0140 |
| XGBoost | Recall | 0.9804 ± 0.0009 | 0.7377 ± 0.0093 | 0.7619 | 0.2427 | -0.0242 |
| XGBoost | Precision | 0.8174 ± 0.0026 | 0.5950 ± 0.0087 | 0.5929 | 0.2224 | 0.0021 |
| XGBoost | F1 | 0.8915 ± 0.0017 | 0.6587 ± 0.0084 | 0.6668 | 0.2328 | -0.0081 |
| Random Forest | Recall | 0.9948 ± 0.0007 | 0.7127 ± 0.0116 | 0.7182 | 0.2822 | -0.0055 |
| Random Forest | Precision | 0.8184 ± 0.0035 | 0.5990 ± 0.0064 | 0.5877 | 0.2194 | 0.0113 |
| Random Forest | F1 | 0.8980 ± 0.0022 | 0.6508 ± 0.0024 | 0.6465 | 0.2472 | 0.0044 |

---

## Fase 1.3 — Contribuição do SMOTE (ablation)

Comparação lado a lado usando o mesmo pipeline de CV (5 folds). Em ambas as condições, cost-sensitive learning permanece ativo (pos_weight na NN, class_weight='balanced' no RF, scale_pos_weight no XGBoost).

| Model | Condition | Recall (CV) | Precision (CV) | F1-micro (CV) | F1-macro (CV) | CV Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Neural network (MLP) | No SMOTE | 0.9513 ± 0.0102 | 0.5127 ± 0.0133 | 0.6662 ± 0.0122 | 0.6286 ± 0.0142 | 350.8 |
| Neural network (MLP) | With SMOTE | 0.8665 ± 0.0090 | 0.5512 ± 0.0089 | 0.6737 ± 0.0082 | 0.6321 ± 0.0130 | 2266.3 |
| Neural network (MLP) | Delta (With - No) | -0.0849 | +0.0385 | +0.0075 | +0.0035 | +1915.5 |
| XGBoost | No SMOTE | 0.8569 ± 0.0102 | 0.5491 ± 0.0119 | 0.6692 ± 0.0103 | 0.6231 ± 0.0157 | 157.7 |
| XGBoost | With SMOTE | 0.7377 ± 0.0093 | 0.5950 ± 0.0087 | 0.6587 ± 0.0084 | 0.5985 ± 0.0130 | 315.1 |
| XGBoost | Delta (With - No) | -0.1192 | +0.0459 | -0.0105 | -0.0246 | +157.4 |
| Random Forest | No SMOTE | 0.9436 ± 0.0041 | 0.5200 ± 0.0106 | 0.6705 ± 0.0093 | 0.6319 ± 0.0147 | 57541.6 |
| Random Forest | With SMOTE | 0.7127 ± 0.0116 | 0.5990 ± 0.0064 | 0.6508 ± 0.0024 | 0.5907 ± 0.0149 | 420.9 |
| Random Forest | Delta (With - No) | -0.2309 | +0.0790 | -0.0196 | -0.0412 | -57120.6 |

### Recomendações por modelo

- **Neural network (MLP)**: SMOTE sem ganho relevante (ΔF1=+0.0075, ΔRecall=-0.0849) — considerar apenas cost-sensitive learning.
- **XGBoost**: SMOTE sem ganho relevante (ΔF1=-0.0105, ΔRecall=-0.1192) — considerar apenas cost-sensitive learning.
- **Random Forest**: SMOTE sem ganho relevante (ΔF1=-0.0196, ΔRecall=-0.2309) — considerar apenas cost-sensitive learning.

### Decisão da Fase 1

- **SMOTE REMOVIDO** — configuração oficial passa a usar apenas cost-sensitive learning
- Ganho médio de F1-micro: -0.0076
- Ganho médio de recall: -0.1450
- Aumento médio do gap treino-val: +0.0693
- Critério: Manter SMOTE se ganho médio de F1-micro >= 0.005 ou de recall >= 0.01 na validação (CV), com aumento médio de gap treino-val <= 0.05.

---

## Fase 1.4 — Overfitting causado pelo SMOTE

### Treino (pós-SMOTE) vs Teste isolado

Métricas do modelo final calculadas no próprio conjunto de treino balanceado (contendo amostras sintéticas) vs no teste nunca visto. Gap grande indica overfitting, possivelmente inflado pelo SMOTE.

| Model | Metric | Train (post-SMOTE) | Test (holdout) | Gap Train-Test |
| --- | --- | --- | --- | --- |
| Neural network (MLP) | Recall | 0.9958 | 0.8497 | 0.1462 |
| Neural network (MLP) | Precision | 0.6064 | 0.5391 | 0.0672 |
| Neural network (MLP) | F1 | 0.7538 | 0.6597 | 0.0941 |
| XGBoost | Recall | 0.9882 | 0.7619 | 0.2264 |
| XGBoost | Precision | 0.7689 | 0.5929 | 0.1760 |
| XGBoost | F1 | 0.8649 | 0.6668 | 0.1980 |
| Random Forest | Recall | 0.9981 | 0.7182 | 0.2798 |
| Random Forest | Precision | 0.7921 | 0.5877 | 0.2044 |
| Random Forest | F1 | 0.8833 | 0.6465 | 0.2368 |

### Curvas de aprendizado

![Curvas de aprendizado](phase1_learning_curves.png)

| Modelo | Condição | Fração treino | N treino (orig.) | N treino (efetivo) | F1 treino (orig.) | F1 treino (pós-SMOTE) | F1 validação | Recall validação |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pytorch_nn | Sem SMOTE | 10% | 1600 | 1600 | 0.6729 | 0.6729 | 0.5870 | 0.8630 |
| pytorch_nn | Sem SMOTE | 25% | 4000 | 4000 | 0.7077 | 0.7077 | 0.6581 | 0.8811 |
| pytorch_nn | Sem SMOTE | 50% | 8000 | 8000 | 0.6986 | 0.6986 | 0.6647 | 0.9296 |
| pytorch_nn | Sem SMOTE | 75% | 12000 | 12000 | 0.6976 | 0.6976 | 0.6764 | 0.9457 |
| pytorch_nn | Sem SMOTE | 100% | 16000 | 16000 | 0.6794 | 0.6794 | 0.6725 | 0.9703 |
| pytorch_nn | Com SMOTE | 10% | 1600 | 3451 | 0.8463 | 0.8087 | 0.5964 | 0.6434 |
| pytorch_nn | Com SMOTE | 25% | 4000 | 8266 | 0.8090 | 0.7983 | 0.6244 | 0.6809 |
| pytorch_nn | Com SMOTE | 50% | 8000 | 17027 | 0.7743 | 0.7591 | 0.6642 | 0.8017 |
| pytorch_nn | Com SMOTE | 75% | 12000 | 25485 | 0.7597 | 0.7671 | 0.6736 | 0.8398 |
| pytorch_nn | Com SMOTE | 100% | 16000 | 33706 | 0.7473 | 0.7595 | 0.6780 | 0.8760 |
| random_forest | Sem SMOTE | 10% | 1600 | 1600 | 0.8526 | 0.8526 | 0.6104 | 0.7119 |
| random_forest | Sem SMOTE | 25% | 4000 | 4000 | 0.8100 | 0.8100 | 0.6571 | 0.8172 |
| random_forest | Sem SMOTE | 50% | 8000 | 8000 | 0.7544 | 0.7544 | 0.6716 | 0.9121 |
| random_forest | Sem SMOTE | 75% | 12000 | 12000 | 0.7377 | 0.7377 | 0.6755 | 0.9354 |
| random_forest | Sem SMOTE | 100% | 16000 | 16000 | 0.7264 | 0.7264 | 0.6813 | 0.9503 |
| random_forest | Com SMOTE | 10% | 1600 | 3451 | 0.9726 | 0.9270 | 0.4810 | 0.3882 |
| random_forest | Com SMOTE | 25% | 4000 | 8266 | 0.9320 | 0.9159 | 0.5490 | 0.5084 |
| random_forest | Com SMOTE | 50% | 8000 | 17027 | 0.9168 | 0.8934 | 0.6138 | 0.6253 |
| random_forest | Com SMOTE | 75% | 12000 | 25485 | 0.9077 | 0.8974 | 0.6419 | 0.6964 |
| random_forest | Com SMOTE | 100% | 16000 | 33706 | 0.9017 | 0.8876 | 0.6479 | 0.7119 |
| xgboost | Sem SMOTE | 10% | 1600 | 1600 | 0.9968 | 0.9968 | 0.5354 | 0.5640 |
| xgboost | Sem SMOTE | 25% | 4000 | 4000 | 0.9979 | 0.9979 | 0.6014 | 0.6492 |
| xgboost | Sem SMOTE | 50% | 8000 | 8000 | 0.9923 | 0.9923 | 0.6573 | 0.7694 |
| xgboost | Sem SMOTE | 75% | 12000 | 12000 | 0.9629 | 0.9629 | 0.6791 | 0.8495 |
| xgboost | Sem SMOTE | 100% | 16000 | 16000 | 0.9359 | 0.9359 | 0.6815 | 0.8702 |
| xgboost | Com SMOTE | 10% | 1600 | 3451 | 0.9857 | 0.9730 | 0.4796 | 0.3844 |
| xgboost | Com SMOTE | 25% | 4000 | 8266 | 0.9440 | 0.9376 | 0.5580 | 0.5161 |
| xgboost | Com SMOTE | 50% | 8000 | 17027 | 0.9264 | 0.8939 | 0.6315 | 0.6505 |
| xgboost | Com SMOTE | 75% | 12000 | 25485 | 0.9048 | 0.8876 | 0.6626 | 0.7332 |
| xgboost | Com SMOTE | 100% | 16000 | 33706 | 0.8942 | 0.8737 | 0.6623 | 0.7461 |

---


# Fase 2 — Modelos estatísticos

## Fase 2.1 — Comparação com modelos estatísticos e de regressão

Configuração oficial da Fase 1: **sem SMOTE (apenas cost-sensitive)**. Todos os modelos usam o mesmo split 80/20, a mesma CV 5-fold, o mesmo pré-processamento e a mesma otimização de threshold por defeito (maximizar recall) na avaliação final de teste.

Baselines estatísticos adicionados: Regressão Logística L2 (Ridge) e L1 (Lasso), ambas com `class_weight='balanced'` via `MultiOutputClassifier`.

| Modelo | Recall (teste) | Precision (teste) | F1-micro (teste) | F1-macro (teste) | Recall (CV) | F1-micro (CV) | Treino (s) | Inferência (ms/100) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Neural network (MLP) | 0.9865 | 0.4648 | 0.6319 | 0.5923 | 0.9513 ± 0.0102 | 0.6662 ± 0.0122 | 171.8 | 0.50 |
| XGBoost | 0.9795 | 0.4789 | 0.6433 | 0.6190 | 0.8569 ± 0.0102 | 0.6692 ± 0.0103 | 31.3 | 35.04 |
| Random Forest | 0.9952 | 0.4400 | 0.6102 | 0.5903 | 0.9436 ± 0.0041 | 0.6705 ± 0.0093 | 44.5 | 1801.89 |
| Logistic Regression (L2/Ridge) | 0.9811 | 0.4855 | 0.6495 | 0.6253 | 0.9392 ± 0.0059 | 0.6725 ± 0.0086 | 8.3 | 5.64 |
| Logistic Regression (L1/Lasso) | 0.9806 | 0.4826 | 0.6469 | 0.6177 | 0.9452 ± 0.0063 | 0.6730 ± 0.0084 | 615.6 | 7.22 |

---


# Fase 3 — Diagnósticos e interpretação

### Thresholds oficiais (seleção out-of-fold, sem uso do teste)

Fonte: `threshold_selection_oof.py` → `reports/oof_threshold_results.json`. Thresholds por defeito selecionados nas predições out-of-fold do conjunto de desenvolvimento (CV 5-fold) e aplicados, inalterados, ao conjunto de teste.

- Estratégia oficial: `fbeta_2`
- Faixa de thresholds (defeitos com positivos no teste): 0.13 – 0.66

#### Rede neural — teste (IC 95% bootstrap)

| Métrica | Valor | IC 95% |
|---------|-------|--------|
| Recall (micro) | 0.9725 | [0.9617, 0.9818] |
| Precisão (micro) | 0.4928 | [0.4687, 0.5153] |
| F1 (micro) | 0.6541 | [0.6321, 0.6736] |
| F1 (macro) | 0.6151 | [0.5884, 0.6370] |

Matriz de confusão agregada (28 defeitos × teste): TP = 1805, TN = 136286, FP = 1858, FN = 51.

#### F2-micro no desenvolvimento (OOF) por estratégia — rede neural

| Estratégia | F2-micro (OOF) |
|------------|----------------|
| recall_first | 0.7987 |
| f1 | 0.7896 |
| fbeta_2 | 0.8206 |
| fbeta_05 | 0.6498 |

#### Todos os modelos — estratégia oficial, teste

| Modelo | Recall (micro) | Precisão (micro) | F1 (micro) | F1 (macro) |
|--------|---|---|---|---|
| pytorch_nn | 0.9725 | 0.4928 | 0.6541 | 0.6151 |
| xgboost | 0.9634 | 0.5020 | 0.6600 | 0.6329 |
| random_forest | 0.8922 | 0.5170 | 0.6547 | 0.6159 |
| logistic_regression_l2 | 0.9709 | 0.5021 | 0.6619 | 0.6361 |
| logistic_regression_l1 | 0.9671 | 0.5035 | 0.6622 | 0.6307 |

*Legado:* `reports/phase3_threshold_tradeoff_test_selected.md` (threshold_tradeoff_analysis.py) contém o procedimento anterior, com thresholds escolhidos no conjunto de teste; não é usado nos resultados oficiais.

---

## Fase 3.1 — Importância de features

Pergunta da banca: *as variáveis da etapa de injeção são as que mais impactam?*

**Resposta:** A fase com maior importância agregada é **Agregada/Global**. A fase de injeção concentra 25.2% da importância total média entre os 5 modelos.

### Importância agregada por fase do processo

| fase_processo | Neural network (MLP) | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) | importancia_media | n_features | process_phase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agregada/Global | 0.046 | 0.6635000109672546 | 0.5719 | 0.053 | 0.064 | 0.2797 | 4 | Global/Aggregation |
| Injeção | 0.3818 | 0.10890000313520432 | 0.1571 | 0.3122 | 0.2996 | 0.2519 | 36 | Injection |
| Configuração/Manutenção | 0.1331 | 0.09359999746084213 | 0.0658 | 0.2695 | 0.2795 | 0.1683 | 35 | Configuration/Maintenance |
| Intensificação | 0.2537 | 0.06549999862909317 | 0.0844 | 0.1685 | 0.1666 | 0.1477 | 16 | Intensification |
| Resfriamento | 0.1814 | 0.05939999967813492 | 0.0996 | 0.1392 | 0.1396 | 0.1238 | 14 | Cooling |
| Múltiplas fases | 0.004 | 0.009200000204145908 | 0.0211 | 0.0577 | 0.0507 | 0.0285 | 5 | Multiple phases |

### Importância agregada por categoria de feature engineering

| categoria_fe | Neural network (MLP) | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) | importancia_media | n_features | feature_category |
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

### Top 20 features (média entre modelos)

| feature | fase_processo | categoria_fe | importancia_media | Neural network (MLP) | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) |
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

---

## Fase 3.3 — Curvas ROC e AUC

Estratégia oficial de threshold (item 3.2): **F-beta β=2 (recall-weighted)** — ponto de operação marcado nas curvas dos defeitos mais frequentes.

### AUC micro-average por modelo

| Model | AUC micro | Mean AUC per defect |
| --- | --- | --- |
| Neural network (MLP) | 0.9956 | 0.9950 |
| XGBoost | 0.9952 | 0.9947 |
| Random Forest | 0.9945 | 0.9940 |
| Logistic Regression (L2/Ridge) | 0.9929 | 0.9929 |
| Logistic Regression (L1/Lasso) | 0.9932 | 0.9893 |

### AUC-ROC por defeito (10 mais frequentes)

| defect | test_positives | Neural network (MLP) | XGBoost | Random Forest | Logistic Regression (L2/Ridge) | Logistic Regression (L1/Lasso) |
| --- | --- | --- | --- | --- | --- | --- |
| gas_porosity | 141 | 0.9946 | 0.994 | 0.9904 | 0.9937 | 0.9937 |
| density_deviation | 140 | 0.993 | 0.9922 | 0.9904 | 0.9945 | 0.9945 |
| cold_shut | 138 | 0.9943 | 0.9938 | 0.9913 | 0.9959 | 0.9959 |
| gas_bubbles | 129 | 0.9938 | 0.993 | 0.9909 | 0.9947 | 0.9948 |
| incomplete_fill | 105 | 0.9939 | 0.9939 | 0.9931 | 0.9948 | 0.9948 |
| surface_blisters | 93 | 0.991 | 0.9906 | 0.9887 | 0.9925 | 0.9925 |
| blisters_post_treatment | 92 | 0.9951 | 0.9946 | 0.9937 | 0.9954 | 0.9954 |
| volumetric_variation | 84 | 0.9966 | 0.9962 | 0.9965 | 0.9851 | 0.9871 |
| low_elongation | 84 | 0.9941 | 0.9937 | 0.9928 | 0.9938 | 0.9939 |
| dimensional_deviation | 83 | 0.9967 | 0.9964 | 0.9961 | 0.9963 | 0.9964 |

Tabela completa: `figures/table_phase3_auc_by_defect.csv`

![ROC defeitos frequentes](phase3_roc_curves_top_defects.png)
![ROC micro-average](phase3_roc_micro_average.png)

---

## Fase 3.4 — Previsto × Realizado

Adaptação para classificação multi-label: (1) diagramas de calibração (probabilidade prevista vs frequência real) e (2) matrizes de confusão normalizadas com os thresholds oficiais do item 3.2.

### Matrizes de confusão (NN, thresholds oficiais)

| defect | threshold | tn | fp | fn | tp | recall | precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gas_porosity | 0.17000000000000004 | 4741 | 118 | 0 | 141 | 1.0 | 0.5444 |
| density_deviation | 0.4100000000000001 | 4763 | 97 | 0 | 140 | 1.0 | 0.5907 |
| cold_shut | 0.34 | 4771 | 91 | 4 | 134 | 0.971 | 0.5956 |

![Calibração por modelo](phase3_calibration_by_model.png)
![Calibração por defeito](phase3_calibration_top_defects_nn.png)
![Confusão normalizada](phase3_confusion_normalized_nn.png)

---
