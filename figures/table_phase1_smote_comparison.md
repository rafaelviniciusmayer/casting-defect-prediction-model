# Fase 1.3 — Contribuição do SMOTE (ablation)

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

## Recomendações por modelo

- **PyTorch NN**: SMOTE sem ganho relevante (ΔF1=+0.0064, ΔRecall=-0.0583) — considerar apenas cost-sensitive learning.
- **XGBoost**: SMOTE sem ganho relevante (ΔF1=-0.0083, ΔRecall=-0.1199) — considerar apenas cost-sensitive learning.
- **Random Forest**: SMOTE sem ganho relevante (ΔF1=-0.0431, ΔRecall=-0.1966) — considerar apenas cost-sensitive learning.

## Decisão da Fase 1

- **SMOTE REMOVIDO** — configuração oficial passa a usar apenas cost-sensitive learning
- Ganho médio de F1-micro: -0.0150
- Ganho médio de recall: -0.1249
- Aumento médio do gap treino-val: +0.0410
- Critério: Manter SMOTE se ganho médio de F1-micro >= 0.005 ou de recall >= 0.01 na validação (CV), com aumento médio de gap treino-val <= 0.05.
