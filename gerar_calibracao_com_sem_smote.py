"""
Comentario #91 (Figura 11) - Curvas de calibracao COM vs SEM SMOTE.

Como rodar (dentro do clone local do repositorio
casting-defect-prediction-model, com o ambiente ja usado para treinar
os modelos originais):

    pip install imbalanced-learn   # se ainda nao tiver
    python gerar_calibracao_com_sem_smote.py

O que o script faz:
  1. Carrega o split de dados oficial (mesmo random_state=42, 80/20,
     estratificado) via unified_cv_pipeline.load_pipeline_data /
     prepare_split_data.
  2. Para os 5 modelos, treina COM SMOTE (use_smote=True) no conjunto de
     desenvolvimento completo e avalia no teste (run_final_test_evaluation),
     igual ao que o proprio pipeline de voces ja faz.
  3. As predicoes SEM SMOTE ja estao salvas em
     models/phase3_final_models.pkl (models[<tipo>]['test_proba']) -
     nao precisa retreinar essa parte.
  4. Monta curvas de calibracao (reliability diagram, micro-media entre
     os 28 defeitos, 10 bins) para as duas condicoes, no mesmo estilo
     da Figura 11 atual, e salva em
     figures/phase3_calibration_with_vs_without_smote.png

Tempo esperado (rodando local, CPU comum): XGBoost e Logistic
Regression L2 ficam prontos em 1-2 min cada; Random Forest e Logistic
Regression L1 podem passar de 5 min (o solver liblinear fica mais
lento com o volume extra de amostras sinteticas do SMOTE); a rede
neural (200 epocas) deve ficar na faixa de poucos minutos. No ambiente
de sandbox usado para o restante das edicoes este script ultrapassou o
limite de tempo por comando antes de terminar todos os 5 modelos -
por isso ele nao foi executado ate o fim la, mas deve rodar sem
problema no ambiente de voces, que ja tem tudo instalado e sem esse
limite.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import unified_cv_pipeline as ucv

MODEL_TYPES = [
    ucv.MODEL_PYTORCH,
    ucv.MODEL_XGBOOST,
    ucv.MODEL_RANDOM_FOREST,
    ucv.MODEL_LOGISTIC_L2,
    ucv.MODEL_LOGISTIC_L1,
]

def get_without_smote_proba():
    with open('models/phase3_final_models.pkl', 'rb') as f:
        d = pickle.load(f)
    return {mt: d['models'][mt]['test_proba'] for mt in MODEL_TYPES}, d['y_test']

def get_with_smote_proba(split):
    out = {}
    for model_type in MODEL_TYPES:
        print(f"[*] Treinando {model_type} COM SMOTE...")
        res = ucv.run_final_test_evaluation(model_type, split, use_smote=True, verbose=True)
        proba = ucv.predict_proba_any(model_type, res.model, res.scaler.transform(split.X_test))
        out[model_type] = proba
    return out

def reliability_curve(y_true, proba, n_bins=10):
    y_true = y_true.ravel()
    proba = proba.ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(proba, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    mean_pred, frac_pos = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        mean_pred.append(proba[mask].mean())
        frac_pos.append(y_true[mask].mean())
    return np.array(mean_pred), np.array(frac_pos)

def main():
    data = ucv.load_pipeline_data(verbose=False)
    split = ucv.prepare_split_data(data, verbose=False)

    proba_without, y_test = get_without_smote_proba()
    proba_with = get_with_smote_proba(split)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, proba_dict, title in [
        (axes[0], proba_without, "Without SMOTE (official pipeline)"),
        (axes[1], proba_with, "With SMOTE"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
        for mt in MODEL_TYPES:
            mp, fp = reliability_curve(y_test, proba_dict[mt])
            ax.plot(mp, fp, marker="o", label=ucv.MODEL_DISPLAY_NAMES[mt])
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Observed frequency")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Calibration curves by model architecture (micro-average): with vs. without SMOTE")
    fig.tight_layout()
    fig.savefig("figures/phase3_calibration_with_vs_without_smote.png", dpi=150, facecolor="white")
    print("Salvo em figures/phase3_calibration_with_vs_without_smote.png")

if __name__ == "__main__":
    main()
