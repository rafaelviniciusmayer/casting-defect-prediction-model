# Fase 1.2 — Análise de Overfitting (com SMOTE)

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
