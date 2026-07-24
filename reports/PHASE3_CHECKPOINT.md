# Checkpoint — Fase 3 (diagnósticos e interpretação)

- **Estratégia oficial de threshold: F-beta β=2 (recall moderado)** (Maior F-beta(β=2)-micro no teste para o modelo em produção (NN): mantém prioridade em recall, penalizando excesso de falsos positivos.)
- Importância de features: A fase com maior importância agregada é **Agregada/Global**. A fase de injeção concentra 24.6% da importância total média entre os 5 modelos.
- Curvas ROC/AUC e previsto×realizado gerados com os thresholds oficiais.

## Arquivos gerados

- `reports/phase3_feature_importance.md` (3.1)
- `reports/phase3_threshold_tradeoff.md` (3.2)
- `reports/phase3_roc_analysis.md` (3.3)
- `reports/phase3_predicted_vs_actual.md` (3.4)
- Figuras e CSVs correspondentes em `figures/`
