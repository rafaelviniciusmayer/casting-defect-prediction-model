"""
Comparação de Modelos para Predição de Defeitos em Fundição
===========================================================

Script standalone que compara 3 modelos no mesmo pipeline:
1. PyTorch Neural Network (modelo atual em produção)
2. XGBoost
3. Random Forest

Objetivo: Justificar a escolha do melhor modelo com evidências empíricas.
Não altera train_model.py - importa funções existentes e adiciona apenas
a lógica de comparação.

Execute: python compare_models.py

Requisitos adicionais: pip install xgboost
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Importar do train_model existente (sem alterá-lo)
from train_model import (
    apply_feature_engineering,
    apply_smote_balancing,
    load_and_prepare_data,
    train_single_model,
)
from train_model import DefectPredictionNN

import torch
from sklearn.model_selection import train_test_split

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[AVISO] xgboost não instalado. Execute: pip install xgboost")


def _get_train_test_split():
    """Carrega dados e retorna split idêntico ao train_model.py."""
    X, y, feature_names, defect_names, pos_weights = load_and_prepare_data()
    X, feature_names = apply_feature_engineering(X, feature_names)

    has_defect = (y.sum(axis=1) > 0).astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=has_defect
    )
    return X_dev, X_test, y_dev, y_test, feature_names, defect_names, pos_weights


def _optimize_thresholds_silent(y_test, y_pred_proba, defect_names):
    """Mesma lógica de optimize_thresholds do train_model, sem prints."""
    optimal_thresholds = {}
    y_pred_binary = np.zeros_like(y_pred_proba)
    for i, defect_name in enumerate(defect_names):
        y_true_defect = y_test[:, i]
        if y_true_defect.sum() == 0:
            optimal_thresholds[defect_name] = 0.5
            continue
        best_recall, best_threshold, best_precision, best_f1 = 0, 0.5, 0, 0
        for threshold in np.arange(0.1, 0.91, 0.01):
            y_pred_defect = (y_pred_proba[:, i] >= threshold).astype(int)
            if y_pred_defect.sum() > 0:
                precision = precision_score(y_true_defect, y_pred_defect, zero_division=0)
                recall = recall_score(y_true_defect, y_pred_defect, zero_division=0)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                if recall > best_recall:
                    best_recall, best_threshold, best_precision, best_f1 = recall, threshold, precision, f1
                elif recall == best_recall and f1 > best_f1:
                    best_threshold, best_precision, best_f1 = threshold, precision, f1
        if best_recall < 0.80:
            for threshold in np.arange(0.05, 0.11, 0.01):
                y_pred_defect = (y_pred_proba[:, i] >= threshold).astype(int)
                if y_pred_defect.sum() > 0:
                    precision = precision_score(y_true_defect, y_pred_defect, zero_division=0)
                    recall = recall_score(y_true_defect, y_pred_defect, zero_division=0)
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    if recall > best_recall:
                        best_recall, best_threshold, best_precision, best_f1 = recall, threshold, precision, f1
                    elif recall == best_recall and f1 > best_f1:
                        best_threshold, best_precision, best_f1 = threshold, precision, f1
        optimal_thresholds[defect_name] = best_threshold
        y_pred_binary[:, i] = (y_pred_proba[:, i] >= best_threshold).astype(int)
    return optimal_thresholds, y_pred_binary


def _optimize_and_evaluate(y_test, y_pred_proba, defect_names, model_name):
    """Aplica otimização de thresholds e retorna métricas."""
    optimal_thresholds, y_pred_binary = _optimize_thresholds_silent(
        y_test, y_pred_proba, defect_names
    )
    metrics = {
        'f1_micro': f1_score(y_test, y_pred_binary, average='micro'),
        'f1_macro': f1_score(y_test, y_pred_binary, average='macro'),
        'precision_micro': precision_score(y_test, y_pred_binary, average='micro'),
        'recall_micro': recall_score(y_test, y_pred_binary, average='micro'),
        'accuracy': accuracy_score(y_test, y_pred_binary),
    }
    return metrics, optimal_thresholds


def train_and_evaluate_pytorch(X_train, X_test, y_train, y_test, pos_weights, defect_names):
    """Treina e avalia o modelo PyTorch (mesma lógica do train_model.py)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    X_train_scaled, y_train = apply_smote_balancing(
        X_train_scaled, y_train, defect_names
    )

    t0 = time.perf_counter()
    model, _ = train_single_model(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        pos_weights, X_train_scaled.shape[1], y_train.shape[1]
    )
    train_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled)
        logits = model(X_tensor)
        y_pred_proba = torch.sigmoid(logits).numpy()

    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = torch.sigmoid(model(torch.FloatTensor(X_test_scaled[:100])))
    inference_time_ms = (time.perf_counter() - t0) / 100 * 1000  # ms por 100 amostras

    metrics, thresholds = _optimize_and_evaluate(
        y_test, y_pred_proba, defect_names, 'PyTorch NN'
    )
    return metrics, train_time, inference_time_ms, thresholds


def _extract_proba_positive(proba_list, model):
    """Extrai P(classe=1) de MultiOutputClassifier, tratando caso de classe única."""
    proba_cols = []
    for i, p in enumerate(proba_list):
        if p.shape[1] == 2:
            proba_cols.append(p[:, 1])
        else:
            est = model.estimators_[i]
            proba_cols.append(p[:, 0] if (1 in est.classes_) else np.zeros(p.shape[0]))
    return np.column_stack(proba_cols)


def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test, defect_names):
    """Treina e avalia XGBoost com MultiOutputClassifier."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled, y_train = apply_smote_balancing(
        X_train_scaled, y_train, defect_names
    )

    base_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
    )
    model = MultiOutputClassifier(base_clf, n_jobs=-1)

    t0 = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - t0

    proba_list = model.predict_proba(X_test_scaled)
    y_pred_proba = _extract_proba_positive(proba_list, model)

    t0 = time.perf_counter()
    for _ in range(100):
        _ = model.predict_proba(X_test_scaled[:100])
    inference_time_ms = (time.perf_counter() - t0) / 100 * 1000

    metrics, thresholds = _optimize_and_evaluate(
        y_test, y_pred_proba, defect_names, 'XGBoost'
    )
    return metrics, train_time, inference_time_ms, thresholds


def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, defect_names):
    """Treina e avalia Random Forest com MultiOutputClassifier."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled, y_train = apply_smote_balancing(
        X_train_scaled, y_train, defect_names
    )

    base_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputClassifier(base_clf, n_jobs=-1)

    t0 = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - t0

    proba_list = model.predict_proba(X_test_scaled)
    y_pred_proba = _extract_proba_positive(proba_list, model)

    t0 = time.perf_counter()
    for _ in range(100):
        _ = model.predict_proba(X_test_scaled[:100])
    inference_time_ms = (time.perf_counter() - t0) / 100 * 1000

    metrics, thresholds = _optimize_and_evaluate(
        y_test, y_pred_proba, defect_names, 'Random Forest'
    )
    return metrics, train_time, inference_time_ms, thresholds


def main():
    print("=" * 70)
    print("COMPARAÇÃO DE MODELOS - Predição de Defeitos em Fundição")
    print("=" * 70)

    if not XGBOOST_AVAILABLE:
        print("\n[ERRO] xgboost não está instalado. Execute: pip install xgboost")
        return

    print("\n[*] Carregando dados (mesmo pipeline do train_model.py)...")
    X_dev, X_test, y_dev, y_test, feature_names, defect_names, pos_weights = _get_train_test_split()
    print(f"    Treino: {X_dev.shape[0]:,} | Teste: {X_test.shape[0]:,} | Features: {X_dev.shape[1]} | Defeitos: {y_dev.shape[1]}")

    results = {}

    # 1. PyTorch NN
    print("\n" + "-" * 70)
    print("1/3 Treinando PyTorch Neural Network...")
    print("-" * 70)
    metrics_pytorch, train_time_pytorch, inf_time_pytorch, _ = train_and_evaluate_pytorch(
        X_dev, X_test, y_dev, y_test, pos_weights, defect_names
    )
    results['PyTorch NN'] = {
        'metrics': metrics_pytorch,
        'train_time_sec': round(train_time_pytorch, 2),
        'inference_time_ms_per_100': round(inf_time_pytorch, 2),
    }
    print(f"    F1-Micro: {metrics_pytorch['f1_micro']:.4f} | Recall: {metrics_pytorch['recall_micro']:.4f} | "
          f"Precision: {metrics_pytorch['precision_micro']:.4f} | Treino: {train_time_pytorch:.1f}s")

    # 2. XGBoost
    print("\n" + "-" * 70)
    print("2/3 Treinando XGBoost...")
    print("-" * 70)
    metrics_xgb, train_time_xgb, inf_time_xgb, _ = train_and_evaluate_xgboost(
        X_dev, X_test, y_dev, y_test, defect_names
    )
    results['XGBoost'] = {
        'metrics': metrics_xgb,
        'train_time_sec': round(train_time_xgb, 2),
        'inference_time_ms_per_100': round(inf_time_xgb, 2),
    }
    print(f"    F1-Micro: {metrics_xgb['f1_micro']:.4f} | Recall: {metrics_xgb['recall_micro']:.4f} | "
          f"Precision: {metrics_xgb['precision_micro']:.4f} | Treino: {train_time_xgb:.1f}s")

    # 3. Random Forest
    print("\n" + "-" * 70)
    print("3/3 Treinando Random Forest...")
    print("-" * 70)
    metrics_rf, train_time_rf, inf_time_rf, _ = train_and_evaluate_random_forest(
        X_dev, X_test, y_dev, y_test, defect_names
    )
    results['Random Forest'] = {
        'metrics': metrics_rf,
        'train_time_sec': round(train_time_rf, 2),
        'inference_time_ms_per_100': round(inf_time_rf, 2),
    }
    print(f"    F1-Micro: {metrics_rf['f1_micro']:.4f} | Recall: {metrics_rf['recall_micro']:.4f} | "
          f"Precision: {metrics_rf['precision_micro']:.4f} | Treino: {train_time_rf:.1f}s")

    # Resumo e justificativa
    print("\n" + "=" * 70)
    print("RESUMO DA COMPARAÇÃO")
    print("=" * 70)

    # Tabela
    print("\n{:<18} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
        "Modelo", "F1-Micro", "F1-Macro", "Precision", "Recall", "Treino (s)"
    ))
    print("-" * 72)
    for name, data in results.items():
        m = data['metrics']
        print("{:<18} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.2f}".format(
            name, m['f1_micro'], m['f1_macro'], m['precision_micro'],
            m['recall_micro'], data['train_time_sec']
        ))

    # Melhor modelo por critério
    best_f1 = max(results.items(), key=lambda x: x[1]['metrics']['f1_micro'])
    best_recall = max(results.items(), key=lambda x: x[1]['metrics']['recall_micro'])

    print("\n[*] Melhor F1-Micro:", best_f1[0])
    print("[*] Melhor Recall:", best_recall[0])

    # Justificativa
    justification = []
    if best_f1[0] == 'PyTorch NN':
        justification.append(
            "O modelo PyTorch NN obteve o melhor F1-Score, indicando melhor "
            "balanceamento entre Precision e Recall na tarefa multi-label."
        )
    if best_recall[0] == 'PyTorch NN':
        justification.append(
            "O PyTorch NN apresentou o maior Recall, essencial para minimizar "
            "falsos negativos (defeitos que passam despercebidos)."
        )
    if justification:
        print("\n[Justificativa] " + " ".join(justification))
    else:
        print("\n[Justificativa] O modelo alternativo", best_f1[0], "ou", best_recall[0],
              "apresentou desempenho superior. Considere reavaliar a escolha do modelo.")

    # Salvar relatório
    report = {
        'comparison': {k: {**v, 'metrics': {mk: float(mv) for mk, mv in v['metrics'].items()}}
                      for k, v in results.items()},
        'best_f1_micro': best_f1[0],
        'best_recall': best_recall[0],
        'justification': justification if justification else [
            f"Modelo alternativo ({best_f1[0]}) apresentou melhor desempenho. "
            "Considere reavaliar a escolha do modelo em produção."
        ],
        'config': {
            'train_test_split': '80/20',
            'random_state': 42,
            'threshold_optimization': 'maximize_recall_per_defect',
        },
    }

    Path('reports').mkdir(exist_ok=True)
    with open('reports/model_comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Markdown
    md_path = Path('reports/MODEL_COMPARISON_REPORT.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Comparação de Modelos\n\n")
        f.write("## Objetivo\n\n")
        f.write("Justificar a escolha do modelo para predição de defeitos em fundição de alumínio, ")
        f.write("comparando PyTorch NN (modelo atual), XGBoost e Random Forest.\n\n")
        f.write("## Metodologia\n\n")
        f.write("- Mesmo pipeline de dados (load_and_prepare_data, apply_feature_engineering)\n")
        f.write("- Mesmo split 80/20 estratificado (random_state=42)\n")
        f.write("- SMOTE aplicado ao treino para todos os modelos\n")
        f.write("- Otimização de thresholds por defeito (maximizar Recall)\n\n")
        f.write("## Resultados\n\n")
        f.write("| Modelo | F1-Micro | F1-Macro | Precision | Recall | Treino (s) | Inf. (ms/100) |\n")
        f.write("|--------|----------|----------|-----------|--------|------------|---------------|\n")
        for name, data in results.items():
            m = data['metrics']
            f.write(f"| {name} | {m['f1_micro']:.4f} | {m['f1_macro']:.4f} | "
                    f"{m['precision_micro']:.4f} | {m['recall_micro']:.4f} | "
                    f"{data['train_time_sec']} | {data['inference_time_ms_per_100']:.2f} |\n")
        f.write("\n## Conclusão\n\n")
        f.write(f"- **Melhor F1-Micro:** {best_f1[0]}\n")
        f.write(f"- **Melhor Recall:** {best_recall[0]}\n\n")
        for j in report['justification']:
            f.write(f"- {j}\n")

    print(f"\n[OK] Relatório salvo em reports/model_comparison_report.json")
    print(f"[OK] Relatório Markdown salvo em {md_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
