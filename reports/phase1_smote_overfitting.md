# Fase 1.4 — Overfitting causado pelo SMOTE

## Treino (pós-SMOTE) vs Teste isolado

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

## Curvas de aprendizado

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
