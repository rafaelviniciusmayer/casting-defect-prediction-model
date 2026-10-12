# Checkpoint — Fase 3 (diagnósticos e interpretação)

- **Estratégia oficial de threshold: F-beta β=2 (recall-weighted)** (Highest F-beta(2)-micro of the production neural network on out-of-fold development predictions; thresholds chosen on development data only.)
- Importância de features: A fase com maior importância agregada é **Agregada/Global**. A fase de injeção concentra 25.2% da importância total média entre os 5 modelos.
- Curvas ROC/AUC e previsto×realizado gerados com os thresholds oficiais.

## Arquivos gerados

- `reports/phase3_feature_importance.md` (3.1)
- `reports/oof_threshold_results.json` e `reports/phase3_official_thresholds.json` (3.2)
- `reports/phase3_roc_analysis.md` (3.3)
- `reports/phase3_predicted_vs_actual.md` (3.4)
- Figuras e CSVs correspondentes em `figures/`
