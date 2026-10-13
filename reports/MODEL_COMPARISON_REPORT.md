# Model Comparison Report

## Objective

Justify the choice of model for defect prediction in aluminum die casting by comparing the PyTorch NN (current model), XGBoost and Random Forest.

## Methodology

- Same data pipeline (load_and_prepare_data, apply_feature_engineering)
- Same stratified 80/20 split (random_state=42)
- SMOTE applied to the training data for all models
- Per-defect threshold optimization (maximize recall)

## Results

| Model | F1-Micro | F1-Macro | Precision | Recall | Training (s) | Inf. (ms/100) |
|--------|----------|----------|-----------|--------|------------|---------------|
| PyTorch NN | 0.6661 | 0.6116 | 0.5054 | 0.9763 | 70.74 | 0.33 |
| XGBoost | 0.6833 | 0.6328 | 0.5408 | 0.9278 | 53.67 | 19.82 |
| Random Forest | 0.6669 | 0.6163 | 0.5044 | 0.9838 | 210.26 | 1232.20 |

## Conclusion

- **Best F1-Micro:** XGBoost
- **Best Recall:** Random Forest

- An alternative model (XGBoost) performed better. Consider re-evaluating the choice of the production model.
