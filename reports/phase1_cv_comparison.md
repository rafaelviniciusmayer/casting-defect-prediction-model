# Fase 1.1 — Comparação por Cross-Validation (5 folds)

Métricas médias ± desvio-padrão na **validação** de cada fold (threshold fixo 0.5). SMOTE aplicado apenas no treino de cada fold; teste final (20%) permanece isolado.

| Modelo | Recall (val) | Precision (val) | F1-micro (val) | F1-macro (val) | Tempo CV (s) |
|--------|--------------|-----------------|----------------|----------------|--------------|
| Neural network (MLP) | 0.8665 ± 0.0090 | 0.5512 ± 0.0089 | 0.6737 ± 0.0082 | 0.6321 ± 0.0130 | 2266.3 |
| XGBoost | 0.7377 ± 0.0093 | 0.5950 ± 0.0087 | 0.6587 ± 0.0084 | 0.5985 ± 0.0130 | 315.1 |
| Random Forest | 0.7127 ± 0.0116 | 0.5990 ± 0.0064 | 0.6508 ± 0.0024 | 0.5907 ± 0.0149 | 420.9 |
