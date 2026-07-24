# Relatório de Comparação de Modelos

## Objetivo

Justificar a escolha do modelo para predição de defeitos em fundição de alumínio, comparando PyTorch NN (modelo atual), XGBoost e Random Forest.

## Metodologia

- Mesmo pipeline de dados (load_and_prepare_data, apply_feature_engineering)
- Mesmo split 80/20 estratificado (random_state=42)
- SMOTE aplicado ao treino para todos os modelos
- Otimização de thresholds por defeito (maximizar Recall)

## Resultados

| Modelo | F1-Micro | F1-Macro | Precision | Recall | Treino (s) | Inf. (ms/100) |
|--------|----------|----------|-----------|--------|------------|---------------|
| PyTorch NN | 0.6661 | 0.6116 | 0.5054 | 0.9763 | 70.74 | 0.33 |
| XGBoost | 0.6833 | 0.6328 | 0.5408 | 0.9278 | 53.67 | 19.82 |
| Random Forest | 0.6669 | 0.6163 | 0.5044 | 0.9838 | 210.26 | 1232.20 |

## Conclusão

- **Melhor F1-Micro:** XGBoost
- **Melhor Recall:** Random Forest

- Modelo alternativo (XGBoost) apresentou melhor desempenho. Considere reavaliar a escolha do modelo em produção.
