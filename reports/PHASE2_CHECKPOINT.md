# Checkpoint — Fase 2 (modelos estatísticos)

- Condição oficial: sem SMOTE (herdada da Fase 1).
- Modelos comparados: PyTorch NN, XGBoost, Random Forest, Regressão Logística L2 (Ridge), Regressão Logística L1 (Lasso).
- Melhor recall no teste (thresholds otimizados): **Random Forest** (0.9968).
- Melhor F1-micro no teste: **PyTorch NN** (0.6639).
- A Fase 3 deve usar estes 4-5 modelos finais para importância de features (3.1) e as probabilidades de teste para thresholds/ROC (3.2/3.3).
