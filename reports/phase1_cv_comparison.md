# Fase 1.1 — Comparação por Cross-Validation (5 folds)

Métricas médias ± desvio-padrão na **validação** de cada fold (threshold fixo 0.5). SMOTE aplicado apenas no treino de cada fold; teste final (20%) permanece isolado.

| Modelo | Recall (val) | Precision (val) | F1-micro (val) | F1-macro (val) | Tempo CV (s) |
|--------|--------------|-----------------|----------------|----------------|--------------|
| PyTorch NN | 0.8928 ± 0.0102 | 0.5389 ± 0.0105 | 0.6721 ± 0.0106 | 0.6093 ± 0.0147 | 366.4 |
| XGBoost | 0.7377 ± 0.0088 | 0.5987 ± 0.0079 | 0.6609 ± 0.0075 | 0.5809 ± 0.0107 | 329.9 |
| Random Forest | 0.6239 ± 0.0052 | 0.6059 ± 0.0077 | 0.6147 ± 0.0049 | 0.5207 ± 0.0152 | 949.0 |
