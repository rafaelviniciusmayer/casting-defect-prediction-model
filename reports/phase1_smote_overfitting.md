# Fase 1.4 — Overfitting causado pelo SMOTE

## Treino (pós-SMOTE) vs Teste isolado

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

## Curvas de aprendizado

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
