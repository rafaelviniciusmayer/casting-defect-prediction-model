# Fase 2.1 — Comparação com modelos estatísticos e de regressão

Configuração oficial da Fase 1: **sem SMOTE (apenas cost-sensitive)**. Todos os modelos usam o mesmo split 80/20, a mesma CV 5-fold, o mesmo pré-processamento e a mesma otimização de threshold por defeito (maximizar recall) na avaliação final de teste.

Baselines estatísticos adicionados: Regressão Logística L2 (Ridge) e L1 (Lasso), ambas com `class_weight='balanced'` via `MultiOutputClassifier`.

| Modelo | Recall (teste) | Precision (teste) | F1-micro (teste) | F1-macro (teste) | Recall (CV) | F1-micro (CV) | Treino (s) | Inferência (ms/100) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch NN | 0.9930 | 0.4986 | 0.6639 | 0.6088 | 0.9511 ± 0.0098 | 0.6657 ± 0.0071 | 48.5 | 0.30 |
| XGBoost | 0.9811 | 0.4984 | 0.6610 | 0.6200 | 0.8576 ± 0.0123 | 0.6692 ± 0.0100 | 23.4 | 16.96 |
| Random Forest | 0.9968 | 0.4137 | 0.5847 | 0.5600 | 0.8204 ± 0.0114 | 0.6578 ± 0.0138 | 55.8 | 1269.45 |
| Regressão Logística (L2/Ridge) | 0.9844 | 0.4999 | 0.6630 | 0.6206 | 0.9384 ± 0.0065 | 0.6721 ± 0.0089 | 11.3 | 2.85 |
| Regressão Logística (L1/Lasso) | 0.9844 | 0.5005 | 0.6636 | 0.6191 | 0.9452 ± 0.0063 | 0.6730 ± 0.0084 | 510.2 | 2.77 |
