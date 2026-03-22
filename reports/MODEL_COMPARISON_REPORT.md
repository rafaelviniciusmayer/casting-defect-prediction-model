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
| PyTorch NN | 0.6708 | 0.6194 | 0.5085 | 0.9855 | 68.55 | 0.29 |
| XGBoost | 0.6833 | 0.6328 | 0.5408 | 0.9278 | 48.91 | 19.76 |
| Random Forest | 0.6669 | 0.6163 | 0.5044 | 0.9838 | 204.11 | 1346.37 |

## Conclusão

- **Melhor F1-Micro:** XGBoost
- **Melhor Recall:** PyTorch NN

- O PyTorch NN apresentou o maior Recall, essencial para minimizar falsos negativos (defeitos que passam despercebidos).
