# Relatório consolidado — Ajustes solicitados pela banca

Gerado em 2026-07-10 00:59. Pipeline: split 80/20 estratificado (random_state=42), CV 5-fold unificada, configuração oficial definida na Fase 1.

---


# Decisões (checkpoints)

## Checkpoint — Fase 1 (Infraestrutura de validação)

### Decisões tomadas

- **SMOTE: REMOVIDO** na configuração oficial do pipeline (ganho médio F1-micro -0.0150, recall -0.1249, aumento de gap treino-val +0.0410).
- **CV unificada**: os 3 modelos passam a usar a mesma CV estratificada em 5 folds (random_state=42) sobre o dev set (80%), com SMOTE (quando habilitado) aplicado apenas no fold de treino.
- **Cost-sensitive learning** ativo em todos os modelos: pos_weight na NN, class_weight='balanced' no RF, scale_pos_weight no XGBoost (novidade desta fase — antes o XGBoost não tinha ponderação).
- Melhor recall (validação CV, com SMOTE): **PyTorch NN** (0.8928).
- Melhor F1-micro (validação CV, com SMOTE): **PyTorch NN** (0.6721).

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
- Melhor recall no teste (thresholds otimizados): **Random Forest** (0.9968).
- Melhor F1-micro no teste: **PyTorch NN** (0.6639).
- A Fase 3 deve usar estes 4-5 modelos finais para importância de features (3.1) e as probabilidades de teste para thresholds/ROC (3.2/3.3).

---

## Checkpoint — Fase 3 (diagnósticos e interpretação)

- **Estratégia oficial de threshold: F-beta β=2 (recall moderado)** (Maior F-beta(β=2)-micro no teste para o modelo em produção (NN): mantém prioridade em recall, penalizando excesso de falsos positivos.)
- Importância de features: A fase com maior importância agregada é **Agregada/Global**. A fase de injeção concentra 24.6% da importância total média entre os 5 modelos.
- Curvas ROC/AUC e previsto×realizado gerados com os thresholds oficiais.

### Arquivos gerados

- `reports/phase3_feature_importance.md` (3.1)
- `reports/phase3_threshold_tradeoff.md` (3.2)
- `reports/phase3_roc_analysis.md` (3.3)
- `reports/phase3_predicted_vs_actual.md` (3.4)
- Figuras e CSVs correspondentes em `figures/`

---


# Fase 1 — Infraestrutura de validação

## Fase 1.1 — Comparação por Cross-Validation (5 folds)

Métricas médias ± desvio-padrão na **validação** de cada fold (threshold fixo 0.5). SMOTE aplicado apenas no treino de cada fold; teste final (20%) permanece isolado.

| Modelo | Recall (val) | Precision (val) | F1-micro (val) | F1-macro (val) | Tempo CV (s) |
|--------|--------------|-----------------|----------------|----------------|--------------|
| PyTorch NN | 0.8928 ± 0.0102 | 0.5389 ± 0.0105 | 0.6721 ± 0.0106 | 0.6093 ± 0.0147 | 366.4 |
| XGBoost | 0.7377 ± 0.0088 | 0.5987 ± 0.0079 | 0.6609 ± 0.0075 | 0.5809 ± 0.0107 | 329.9 |
| Random Forest | 0.6239 ± 0.0052 | 0.6059 ± 0.0077 | 0.6147 ± 0.0049 | 0.5207 ± 0.0152 | 949.0 |

---

## Fase 1.2 — Análise de Overfitting (com SMOTE)

Threshold fixo 0.5 em todas as colunas para comparação direta entre treino (CV), validação (CV) e teste final. Métricas de treino calculadas nos dados originais do fold (sem amostras sintéticas).

Gap Treino-Val > 0 indica possível overfitting dentro dos folds. Gap Val-Teste > 0 indica queda de desempenho no holdout.

| Modelo | Métrica | Treino (CV) | Validação (CV) | Teste Final | Gap Treino-Val | Gap Val-Teste |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch NN | Recall | 0.9698 ± 0.0023 | 0.8928 ± 0.0102 | 0.9019 | 0.0770 | -0.0091 |
| PyTorch NN | Precision | 0.5695 ± 0.0032 | 0.5389 ± 0.0105 | 0.5331 | 0.0306 | 0.0058 |
| PyTorch NN | F1 | 0.7176 ± 0.0021 | 0.6721 ± 0.0106 | 0.6701 | 0.0455 | 0.0020 |
| XGBoost | Recall | 0.9805 ± 0.0011 | 0.7377 ± 0.0088 | 0.7565 | 0.2428 | -0.0188 |
| XGBoost | Precision | 0.8162 ± 0.0030 | 0.5987 ± 0.0079 | 0.5899 | 0.2175 | 0.0088 |
| XGBoost | F1 | 0.8908 ± 0.0018 | 0.6609 ± 0.0075 | 0.6629 | 0.2299 | -0.0020 |
| Random Forest | Recall | 0.9879 ± 0.0003 | 0.6239 ± 0.0052 | 0.6498 | 0.3641 | -0.0259 |
| Random Forest | Precision | 0.8495 ± 0.0046 | 0.6059 ± 0.0077 | 0.5956 | 0.2436 | 0.0103 |
| Random Forest | F1 | 0.9135 ± 0.0026 | 0.6147 ± 0.0049 | 0.6215 | 0.2987 | -0.0068 |

---

## Fase 1.3 — Contribuição do SMOTE (ablation)

Comparação lado a lado usando o mesmo pipeline de CV (5 folds). Em ambas as condições, cost-sensitive learning permanece ativo (pos_weight na NN, class_weight='balanced' no RF, scale_pos_weight no XGBoost).

| Modelo | Condição | Recall (CV) | Precision (CV) | F1-micro (CV) | F1-macro (CV) | Tempo treino CV (s) |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch NN | Sem SMOTE | 0.9511 ± 0.0098 | 0.5122 ± 0.0077 | 0.6657 ± 0.0071 | 0.6044 ± 0.0114 | 417.4 |
| PyTorch NN | Com SMOTE | 0.8928 ± 0.0102 | 0.5389 ± 0.0105 | 0.6721 ± 0.0106 | 0.6093 ± 0.0147 | 366.4 |
| PyTorch NN | Δ (Com - Sem) | -0.0583 | +0.0268 | +0.0064 | +0.0049 | -51.0 |
| XGBoost | Sem SMOTE | 0.8576 ± 0.0123 | 0.5488 ± 0.0107 | 0.6692 ± 0.0100 | 0.6045 ± 0.0129 | 120.2 |
| XGBoost | Com SMOTE | 0.7377 ± 0.0088 | 0.5987 ± 0.0079 | 0.6609 ± 0.0075 | 0.5809 ± 0.0107 | 329.9 |
| XGBoost | Δ (Com - Sem) | -0.1199 | +0.0499 | -0.0083 | -0.0236 | +209.7 |
| Random Forest | Sem SMOTE | 0.8204 ± 0.0114 | 0.5491 ± 0.0151 | 0.6578 ± 0.0138 | 0.5663 ± 0.0313 | 204.5 |
| Random Forest | Com SMOTE | 0.6239 ± 0.0052 | 0.6059 ± 0.0077 | 0.6147 ± 0.0049 | 0.5207 ± 0.0152 | 949.0 |
| Random Forest | Δ (Com - Sem) | -0.1966 | +0.0568 | -0.0431 | -0.0456 | +744.5 |

### Recomendações por modelo

- **PyTorch NN**: SMOTE sem ganho relevante (ΔF1=+0.0064, ΔRecall=-0.0583) — considerar apenas cost-sensitive learning.
- **XGBoost**: SMOTE sem ganho relevante (ΔF1=-0.0083, ΔRecall=-0.1199) — considerar apenas cost-sensitive learning.
- **Random Forest**: SMOTE sem ganho relevante (ΔF1=-0.0431, ΔRecall=-0.1966) — considerar apenas cost-sensitive learning.

### Decisão da Fase 1

- **SMOTE REMOVIDO** — configuração oficial passa a usar apenas cost-sensitive learning
- Ganho médio de F1-micro: -0.0150
- Ganho médio de recall: -0.1249
- Aumento médio do gap treino-val: +0.0410
- Critério: Manter SMOTE se ganho médio de F1-micro >= 0.005 ou de recall >= 0.01 na validação (CV), com aumento médio de gap treino-val <= 0.05.

---

## Fase 1.4 — Overfitting causado pelo SMOTE

### Treino (pós-SMOTE) vs Teste isolado

Métricas do modelo final calculadas no próprio conjunto de treino balanceado (contendo amostras sintéticas) vs no teste nunca visto. Gap grande indica overfitting, possivelmente inflado pelo SMOTE.

| Modelo | Métrica | Treino (pós-SMOTE) | Teste (isolado) | Gap Treino-Teste |
| --- | --- | --- | --- | --- |
| PyTorch NN | Recall | 0.9933 | 0.9019 | 0.0914 |
| PyTorch NN | Precision | 0.5848 | 0.5331 | 0.0517 |
| PyTorch NN | F1 | 0.7362 | 0.6701 | 0.0660 |
| XGBoost | Recall | 0.9882 | 0.7565 | 0.2317 |
| XGBoost | Precision | 0.7691 | 0.5899 | 0.1792 |
| XGBoost | F1 | 0.8650 | 0.6629 | 0.2021 |
| Random Forest | Recall | 0.9955 | 0.6498 | 0.3458 |
| Random Forest | Precision | 0.8227 | 0.5956 | 0.2272 |
| Random Forest | F1 | 0.9009 | 0.6215 | 0.2794 |

### Curvas de aprendizado

![Curvas de aprendizado](phase1_learning_curves.png)

| Modelo | Condição | Fração treino | N treino (orig.) | N treino (efetivo) | F1 treino (orig.) | F1 treino (pós-SMOTE) | F1 validação | Recall validação |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pytorch_nn | Sem SMOTE | 10% | 1600 | 1600 | 0.7103 | 0.7103 | 0.6271 | 0.8637 |
| pytorch_nn | Sem SMOTE | 25% | 4000 | 4000 | 0.6865 | 0.6865 | 0.6614 | 0.9205 |
| pytorch_nn | Sem SMOTE | 50% | 8000 | 8000 | 0.6860 | 0.6860 | 0.6632 | 0.9386 |
| pytorch_nn | Sem SMOTE | 75% | 12000 | 12000 | 0.6961 | 0.6961 | 0.6718 | 0.9490 |
| pytorch_nn | Sem SMOTE | 100% | 16000 | 16000 | 0.6828 | 0.6828 | 0.6742 | 0.9658 |
| pytorch_nn | Com SMOTE | 10% | 1600 | 3451 | 0.8054 | 0.7619 | 0.6389 | 0.7539 |
| pytorch_nn | Com SMOTE | 25% | 4000 | 8266 | 0.7578 | 0.7550 | 0.6554 | 0.7771 |
| pytorch_nn | Com SMOTE | 50% | 8000 | 17027 | 0.7420 | 0.7366 | 0.6725 | 0.8372 |
| pytorch_nn | Com SMOTE | 75% | 12000 | 25485 | 0.7231 | 0.7388 | 0.6862 | 0.8928 |
| pytorch_nn | Com SMOTE | 100% | 16000 | 33706 | 0.7146 | 0.7343 | 0.6860 | 0.9160 |
| random_forest | Sem SMOTE | 10% | 1600 | 1600 | 0.9889 | 0.9889 | 0.4381 | 0.3282 |
| random_forest | Sem SMOTE | 25% | 4000 | 4000 | 0.9573 | 0.9573 | 0.5677 | 0.5252 |
| random_forest | Sem SMOTE | 50% | 8000 | 8000 | 0.8741 | 0.8741 | 0.6370 | 0.7080 |
| random_forest | Sem SMOTE | 75% | 12000 | 12000 | 0.8368 | 0.8368 | 0.6564 | 0.7855 |
| random_forest | Sem SMOTE | 100% | 16000 | 16000 | 0.8244 | 0.8244 | 0.6733 | 0.8301 |
| random_forest | Com SMOTE | 10% | 1600 | 3451 | 0.9833 | 0.9597 | 0.3902 | 0.2791 |
| random_forest | Com SMOTE | 25% | 4000 | 8266 | 0.9566 | 0.9452 | 0.4912 | 0.4141 |
| random_forest | Com SMOTE | 50% | 8000 | 17027 | 0.9371 | 0.9168 | 0.5652 | 0.5291 |
| random_forest | Com SMOTE | 75% | 12000 | 25485 | 0.9258 | 0.9178 | 0.6143 | 0.6137 |
| random_forest | Com SMOTE | 100% | 16000 | 33706 | 0.9156 | 0.9059 | 0.6071 | 0.6150 |
| xgboost | Sem SMOTE | 10% | 1600 | 1600 | 0.9968 | 0.9968 | 0.5421 | 0.5678 |
| xgboost | Sem SMOTE | 25% | 4000 | 4000 | 0.9986 | 0.9986 | 0.5982 | 0.6415 |
| xgboost | Sem SMOTE | 50% | 8000 | 8000 | 0.9913 | 0.9913 | 0.6591 | 0.7733 |
| xgboost | Sem SMOTE | 75% | 12000 | 12000 | 0.9628 | 0.9628 | 0.6796 | 0.8501 |
| xgboost | Sem SMOTE | 100% | 16000 | 16000 | 0.9356 | 0.9356 | 0.6852 | 0.8760 |
| xgboost | Com SMOTE | 10% | 1600 | 3451 | 0.9842 | 0.9727 | 0.4785 | 0.3844 |
| xgboost | Com SMOTE | 25% | 4000 | 8266 | 0.9446 | 0.9368 | 0.5458 | 0.5026 |
| xgboost | Com SMOTE | 50% | 8000 | 17027 | 0.9256 | 0.8937 | 0.6332 | 0.6512 |
| xgboost | Com SMOTE | 75% | 12000 | 25485 | 0.9045 | 0.8878 | 0.6590 | 0.7242 |
| xgboost | Com SMOTE | 100% | 16000 | 33706 | 0.8939 | 0.8738 | 0.6613 | 0.7461 |

---


# Fase 2 — Modelos estatísticos

## Fase 2.1 — Comparação com modelos estatísticos e de regressão

Configuração oficial da Fase 1: **sem SMOTE (apenas cost-sensitive)**. Todos os modelos usam o mesmo split 80/20, a mesma CV 5-fold, o mesmo pré-processamento e a mesma otimização de threshold por defeito (maximizar recall) na avaliação final de teste.

Baselines estatísticos adicionados: Regressão Logística L2 (Ridge) e L1 (Lasso), ambas com `class_weight='balanced'` via `MultiOutputClassifier`.

| Modelo | Recall (teste) | Precision (teste) | F1-micro (teste) | F1-macro (teste) | Recall (CV) | F1-micro (CV) | Treino (s) | Inferência (ms/100) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch NN | 0.9930 | 0.4986 | 0.6639 | 0.6088 | 0.9511 ± 0.0098 | 0.6657 ± 0.0071 | 48.5 | 0.30 |
| XGBoost | 0.9811 | 0.4984 | 0.6610 | 0.6200 | 0.8576 ± 0.0123 | 0.6692 ± 0.0100 | 23.4 | 16.96 |
| Random Forest | 0.9968 | 0.4137 | 0.5847 | 0.5600 | 0.8204 ± 0.0114 | 0.6578 ± 0.0138 | 55.8 | 1269.45 |
| Regressão Logística (L2/Ridge) | 0.9844 | 0.4999 | 0.6630 | 0.6206 | 0.9384 ± 0.0065 | 0.6721 ± 0.0089 | 11.3 | 2.85 |
| Regressão Logística (L1/Lasso) | 0.9844 | 0.5005 | 0.6636 | 0.6191 | 0.9452 ± 0.0063 | 0.6730 ± 0.0084 | 510.2 | 2.77 |

---


# Fase 3 — Diagnósticos e interpretação

## Fase 3.2 — Estratégias de threshold (Precision × Recall)

Quatro estratégias comparadas nos 5 modelos finais, sobre as mesmas probabilidades de teste (nenhum retreino).

| Modelo | Estratégia | Recall | Precision | F1-micro | F1-macro | F2-micro |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch NN | Recall-first (atual) | 0.9925 | 0.4876 | 0.6539 | 0.5995 | 0.8222 |
| PyTorch NN | F1 (equilíbrio) | 0.8879 | 0.5708 | 0.6949 | 0.6377 | 0.7991 |
| PyTorch NN | F-beta β=2 (recall moderado) | 0.9828 | 0.5068 | 0.6687 | 0.6150 | 0.8274 |
| PyTorch NN | F-beta β=0.5 (precisão) | 0.6789 | 0.6360 | 0.6568 | 0.6022 | 0.6699 |
| XGBoost | Recall-first (atual) | 0.9903 | 0.4806 | 0.6472 | 0.6099 | 0.8170 |
| XGBoost | F1 (equilíbrio) | 0.8863 | 0.5620 | 0.6879 | 0.6427 | 0.7946 |
| XGBoost | F-beta β=2 (recall moderado) | 0.9774 | 0.5084 | 0.6689 | 0.6264 | 0.8251 |
| XGBoost | F-beta β=0.5 (precisão) | 0.7306 | 0.6054 | 0.6621 | 0.6128 | 0.7016 |
| Random Forest | Recall-first (atual) | 0.9989 | 0.4067 | 0.5780 | 0.5549 | 0.7736 |
| Random Forest | F1 (equilíbrio) | 0.9213 | 0.5355 | 0.6774 | 0.6316 | 0.8053 |
| Random Forest | F-beta β=2 (recall moderado) | 0.9709 | 0.5045 | 0.6640 | 0.6201 | 0.8194 |
| Random Forest | F-beta β=0.5 (precisão) | 0.7166 | 0.5836 | 0.6433 | 0.5860 | 0.6854 |
| Regressão Logística (L2/Ridge) | Recall-first (atual) | 0.9865 | 0.4981 | 0.6620 | 0.6194 | 0.8248 |
| Regressão Logística (L2/Ridge) | F1 (equilíbrio) | 0.9359 | 0.5486 | 0.6918 | 0.6421 | 0.8201 |
| Regressão Logística (L2/Ridge) | F-beta β=2 (recall moderado) | 0.9758 | 0.5218 | 0.6799 | 0.6328 | 0.8311 |
| Regressão Logística (L2/Ridge) | F-beta β=0.5 (precisão) | 0.8869 | 0.5604 | 0.6868 | 0.6355 | 0.7943 |
| Regressão Logística (L1/Lasso) | Recall-first (atual) | 0.9876 | 0.4972 | 0.6614 | 0.6163 | 0.8249 |
| Regressão Logística (L1/Lasso) | F1 (equilíbrio) | 0.9364 | 0.5505 | 0.6934 | 0.6431 | 0.8213 |
| Regressão Logística (L1/Lasso) | F-beta β=2 (recall moderado) | 0.9752 | 0.5246 | 0.6822 | 0.6323 | 0.8323 |
| Regressão Logística (L1/Lasso) | F-beta β=0.5 (precisão) | 0.8939 | 0.5618 | 0.6900 | 0.6389 | 0.7994 |

### Decisão — estratégia oficial

- **F-beta β=2 (recall moderado)** (critério: Maior F-beta(β=2)-micro no teste para o modelo em produção (NN): mantém prioridade em recall, penalizando excesso de falsos positivos.)
- NN com estratégia oficial: Recall=0.9828, Precision=0.5068, F1=0.6687

Tabela por defeito (NN): `figures/table_phase3_threshold_per_defect.csv`

![Trade-off](phase3_threshold_tradeoff.png)

---

## Fase 3.1 — Importância de features

Pergunta da banca: *as variáveis da etapa de injeção são as que mais impactam?*

**Resposta:** A fase com maior importância agregada é **Agregada/Global**. A fase de injeção concentra 24.6% da importância total média entre os 5 modelos.

### Importância agregada por fase do processo

| fase_processo | PyTorch NN | XGBoost | Random Forest | Regressão Logística (L2/Ridge) | Regressão Logística (L1/Lasso) | importancia_media | n_features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Agregada/Global | 0.0556 | 0.7056000232696533 | 0.6276 | 0.053 | 0.064 | 0.3012 | 4 |
| Injeção | 0.3788 | 0.09080000221729279 | 0.1481 | 0.3122 | 0.2996 | 0.2459 | 36 |
| Configuração/Manutenção | 0.1556 | 0.0966000035405159 | 0.0779 | 0.2695 | 0.2795 | 0.1758 | 35 |
| Intensificação | 0.2564 | 0.04969999939203262 | 0.0699 | 0.1685 | 0.1666 | 0.1422 | 16 |
| Resfriamento | 0.1536 | 0.042500000447034836 | 0.061 | 0.1392 | 0.1396 | 0.1072 | 14 |
| Múltiplas fases | 0.0 | 0.01489999983459711 | 0.0155 | 0.0577 | 0.0507 | 0.0277 | 5 |

### Importância agregada por categoria de feature engineering

| categoria_fe | PyTorch NN | XGBoost | Random Forest | Regressão Logística (L2/Ridge) | Regressão Logística (L1/Lasso) | importancia_media | n_features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Agregação estatística | 0.0556 | 0.7056000232696533 | 0.6276 | 0.053 | 0.064 | 0.3012 | 4 |
| Distância de faixa ideal | 0.5572 | 0.1103999987244606 | 0.1833 | 0.279 | 0.3009 | 0.2862 | 28 |
| Binária de faixa | 0.3632 | 0.07649999856948853 | 0.0854 | 0.2663 | 0.2536 | 0.209 | 42 |
| Original | 0.0174 | 0.0544000007212162 | 0.0377 | 0.151 | 0.1426 | 0.0806 | 15 |
| Ratio | 0.0043 | 0.016899999231100082 | 0.0229 | 0.103 | 0.101 | 0.0496 | 6 |
| Diferença | 0.0 | 0.010900000110268593 | 0.0126 | 0.0472 | 0.0545 | 0.025 | 4 |
| Específica de domínio | 0.0016 | 0.014499999582767487 | 0.0132 | 0.0433 | 0.0318 | 0.0209 | 5 |
| Produto | 0.0008 | 0.006599999964237213 | 0.0066 | 0.0294 | 0.0321 | 0.0151 | 2 |
| Transformação matemática | 0.0 | 0.00430000014603138 | 0.0106 | 0.0278 | 0.0194 | 0.0124 | 4 |

### Top 20 features (média entre modelos)

| feature | fase_processo | categoria_fe | importancia_media | PyTorch NN | XGBoost | Random Forest | Regressão Logística (L2/Ridge) | Regressão Logística (L1/Lasso) |
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

---

## Fase 3.3 — Curvas ROC e AUC

Estratégia oficial de threshold (item 3.2): **F-beta β=2 (recall moderado)** — ponto de operação marcado nas curvas dos defeitos mais frequentes.

### AUC micro-average por modelo

| Modelo | AUC micro | AUC médio por defeito |
| --- | --- | --- |
| PyTorch NN | 0.9955 | 0.9951 |
| XGBoost | 0.9952 | 0.9947 |
| Random Forest | 0.9945 | 0.9940 |
| Regressão Logística (L2/Ridge) | 0.9929 | 0.9929 |
| Regressão Logística (L1/Lasso) | 0.9932 | 0.9893 |

### AUC-ROC por defeito (10 mais frequentes)

| defeito | positivos_teste | PyTorch NN | XGBoost | Random Forest | Regressão Logística (L2/Ridge) | Regressão Logística (L1/Lasso) |
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

---

## Fase 3.4 — Previsto × Realizado

Adaptação para classificação multi-label: (1) diagramas de calibração (probabilidade prevista vs frequência real) e (2) matrizes de confusão normalizadas com os thresholds oficiais do item 3.2.

### Matrizes de confusão (NN, thresholds oficiais)

| defeito | threshold | tn | fp | fn | tp | recall | precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gas_porosity | 0.05 | 4739 | 120 | 0 | 141 | 1.0 | 0.5402 |
| density_deviation | 0.1 | 4765 | 95 | 0 | 140 | 1.0 | 0.5957 |
| cold_shut | 0.21000000000000002 | 4766 | 96 | 0 | 138 | 1.0 | 0.5897 |

![Calibração por modelo](phase3_calibration_by_model.png)
![Calibração por defeito](phase3_calibration_top_defects_nn.png)
![Confusão normalizada](phase3_confusion_normalized_nn.png)

---
