# Checkpoint — Fase 1 (Infraestrutura de validação)

## Decisões tomadas

- **SMOTE: REMOVIDO** na configuração oficial do pipeline (ganho médio F1-micro -0.0150, recall -0.1249, aumento de gap treino-val +0.0410).
- **CV unificada**: os 3 modelos passam a usar a mesma CV estratificada em 5 folds (random_state=42) sobre o dev set (80%), com SMOTE (quando habilitado) aplicado apenas no fold de treino.
- **Cost-sensitive learning** ativo em todos os modelos: pos_weight na NN, class_weight='balanced' no RF, scale_pos_weight no XGBoost (novidade desta fase — antes o XGBoost não tinha ponderação).
- Melhor recall (validação CV, com SMOTE): **PyTorch NN** (0.8928).
- Melhor F1-micro (validação CV, com SMOTE): **PyTorch NN** (0.6721).

## Referência para as próximas fases

- Fase 2 (novos modelos) deve usar `unified_cv_pipeline.run_cross_validation` com `use_smote=False`.
- Fase 3 (thresholds/ROC) deve partir dos modelos finais treinados via `unified_cv_pipeline.run_final_test_evaluation` na mesma condição.

## Arquivos gerados

- `reports/phase1_cv_comparison.md` (1.1)
- `reports/phase1_overfitting_analysis.md` (1.2)
- `reports/phase1_smote_comparison.md` (1.3)
- `reports/phase1_smote_overfitting.md` (1.4)
- `figures/table_phase1_overfitting.csv`
- `figures/table_phase1_smote_comparison.csv`
- `figures/table_phase1_train_vs_test.csv`
- `figures/phase1_learning_curves.png`
