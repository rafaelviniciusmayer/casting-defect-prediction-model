# Fase 2.1 — Comparação com modelos estatísticos e de regressão

Configuração oficial da Fase 1: **sem SMOTE (apenas cost-sensitive)**. Todos os modelos usam o mesmo split 80/20, a mesma CV 5-fold, o mesmo pré-processamento e a mesma otimização de threshold por defeito (maximizar recall) na avaliação final de teste.

Baselines estatísticos adicionados: Regressão Logística L2 (Ridge) e L1 (Lasso), ambas com `class_weight='balanced'` via `MultiOutputClassifier`.

| Modelo | Recall (teste) | Precision (teste) | F1-micro (teste) | F1-macro (teste) | Recall (CV) | F1-micro (CV) | Treino (s) | Inferência (ms/100) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Neural network (MLP) | 0.9865 | 0.4648 | 0.6319 | 0.5923 | 0.9513 ± 0.0102 | 0.6662 ± 0.0122 | 171.8 | 0.50 |
| XGBoost | 0.9795 | 0.4789 | 0.6433 | 0.6190 | 0.8569 ± 0.0102 | 0.6692 ± 0.0103 | 31.3 | 35.04 |
| Random Forest | 0.9952 | 0.4400 | 0.6102 | 0.5903 | 0.9436 ± 0.0041 | 0.6705 ± 0.0093 | 44.5 | 1801.89 |
| Logistic Regression (L2/Ridge) | 0.9811 | 0.4855 | 0.6495 | 0.6253 | 0.9392 ± 0.0059 | 0.6725 ± 0.0086 | 8.3 | 5.64 |
| Logistic Regression (L1/Lasso) | 0.9806 | 0.4826 | 0.6469 | 0.6177 | 0.9452 ± 0.0063 | 0.6730 ± 0.0084 | 615.6 | 7.22 |
