"""
Pipeline unificado de validação cruzada para predição de defeitos em fundição.

Centraliza split 80/20, CV estratificada em 5 folds, SMOTE dentro do fold de
treino, treino dos 3 modelos comparados (PyTorch NN, XGBoost, Random Forest)
e geração de métricas comparáveis entre modelos e condições (com/sem SMOTE).

Compatível com o pipeline existente:
- Mesmo dataset e features (load_and_prepare_data + apply_feature_engineering)
- Mesmo split 80/20 estratificado (random_state=42, stratify=has_defect)
- Mesma CV estratificada em 5 folds (random_state=42)
- SMOTE aplicado apenas no fold de treino (nunca na validação)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from train_model import (
    DefectPredictionNN,
    apply_feature_engineering,
    apply_smote_balancing,
    load_and_prepare_data,
    train_single_model,
)

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.2
METRIC_KEYS = ("recall_micro", "precision_micro", "f1_micro", "f1_macro")

MODEL_PYTORCH = "pytorch_nn"
MODEL_XGBOOST = "xgboost"
MODEL_RANDOM_FOREST = "random_forest"
# Fase 2 — baselines estatísticos (item 2.1)
MODEL_LOGISTIC_L2 = "logistic_regression_l2"
MODEL_LOGISTIC_L1 = "logistic_regression_l1"

MODEL_DISPLAY_NAMES = {
    MODEL_PYTORCH: "PyTorch NN",
    MODEL_XGBOOST: "XGBoost",
    MODEL_RANDOM_FOREST: "Random Forest",
    MODEL_LOGISTIC_L2: "Logistic Regression (L2/Ridge)",
    MODEL_LOGISTIC_L1: "Logistic Regression (L1/Lasso)",
}


@dataclass
class PipelineData:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    defect_names: List[str]
    pos_weights: List[float]


@dataclass
class SplitData:
    X_dev: np.ndarray
    X_test: np.ndarray
    y_dev: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    defect_names: List[str]
    pos_weights: List[float]


@dataclass
class FoldMetrics:
    fold_idx: int
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    train_time_sec: float


@dataclass
class CVResult:
    model_type: str
    model_name: str
    use_smote: bool
    fold_metrics: List[FoldMetrics] = field(default_factory=list)
    train_mean: Dict[str, float] = field(default_factory=dict)
    train_std: Dict[str, float] = field(default_factory=dict)
    val_mean: Dict[str, float] = field(default_factory=dict)
    val_std: Dict[str, float] = field(default_factory=dict)
    total_train_time_sec: float = 0.0


@dataclass
class FinalModelResult:
    model_type: str
    model_name: str
    use_smote: bool
    test_metrics: Dict[str, float]
    cv_val_mean: Dict[str, float]
    train_time_sec: float
    model: Any = None
    scaler: Optional[StandardScaler] = None
    # Métricas no próprio conjunto de treino usado (pós-SMOTE, se habilitado).
    # Usado no item 1.4 (overfitting inflado por amostras sintéticas).
    train_metrics_on_balanced: Dict[str, float] = field(default_factory=dict)


def load_pipeline_data(verbose: bool = True) -> PipelineData:
    """Carrega dados com o mesmo pipeline do train_model.py."""
    X, y, feature_names, defect_names, pos_weights = load_and_prepare_data()
    X, feature_names = apply_feature_engineering(X, feature_names)
    if not verbose:
        return PipelineData(X, y, feature_names, defect_names, pos_weights)
    return PipelineData(X, y, feature_names, defect_names, pos_weights)


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split 80/20 estratificado por has_defect."""
    has_defect = (y.sum(axis=1) > 0).astype(int)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=has_defect
    )


def prepare_split_data(data: PipelineData, verbose: bool = True) -> SplitData:
    """Prepara split dev/test idêntico ao pipeline existente."""
    X_dev, X_test, y_dev, y_test = stratified_split(data.X, data.y)
    if verbose:
        print(f"    Desenvolvimento: {X_dev.shape[0]:,} amostras (80%)")
        print(f"    Teste final: {X_test.shape[0]:,} amostras (20%)")
    return SplitData(
        X_dev=X_dev,
        X_test=X_test,
        y_dev=y_dev,
        y_test=y_test,
        feature_names=data.feature_names,
        defect_names=data.defect_names,
        pos_weights=data.pos_weights,
    )


def get_cv_splits(
    y_dev: np.ndarray,
    n_splits: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Retorna índices da CV estratificada em 5 folds sobre o dev set."""
    has_defect_dev = (y_dev.sum(axis=1) > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(y_dev, has_defect_dev))


def compute_pos_weights(y_train: np.ndarray) -> List[float]:
    """Calcula pesos de classe a partir do fold/conjunto de treino."""
    pos_weights = []
    for i in range(y_train.shape[1]):
        pos_count = y_train[:, i].sum()
        neg_count = len(y_train) - pos_count
        if pos_count > 0:
            weight = np.sqrt(neg_count / pos_count)
            weight = min(weight, 10.0)
        else:
            weight = 1.0
        pos_weights.append(weight)
    return pos_weights


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Métricas micro/macro com threshold fixo (sem otimização)."""
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    return {
        "recall_micro": recall_score(y_true, y_pred_binary, average="micro", zero_division=0),
        "precision_micro": precision_score(
            y_true, y_pred_binary, average="micro", zero_division=0
        ),
        "f1_micro": f1_score(y_true, y_pred_binary, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred_binary, average="macro", zero_division=0),
    }


def preprocess_fold(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    y_train: np.ndarray,
    defect_names: List[str],
    use_smote: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Escala (fit no treino) e aplica SMOTE opcionalmente apenas no treino."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_eval_scaled = scaler.transform(X_eval).astype(np.float32)

    if use_smote:
        X_train_scaled, y_train = apply_smote_balancing(
            X_train_scaled, y_train, defect_names, verbose=verbose
        )

    return X_train_scaled, X_eval_scaled, y_train, scaler


def _extract_proba_positive(proba_list, model) -> np.ndarray:
    """Extrai P(classe=1) de MultiOutputClassifier."""
    proba_cols = []
    for i, p in enumerate(proba_list):
        if p.shape[1] == 2:
            proba_cols.append(p[:, 1])
        else:
            est = model.estimators_[i]
            proba_cols.append(p[:, 0] if (1 in est.classes_) else np.zeros(p.shape[0]))
    return np.column_stack(proba_cols)


class MultiLabelXGBoost:
    """XGBoost multi-label com scale_pos_weight por defeito."""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.estimators_: List[Any] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MultiLabelXGBoost":
        self.estimators_ = []
        for defect_idx in range(y_train.shape[1]):
            pos_count = y_train[:, defect_idx].sum()
            neg_count = len(y_train) - pos_count
            scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
            clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
            )
            clf.fit(X_train, y_train[:, defect_idx])
            self.estimators_.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        return [est.predict_proba(X) for est in self.estimators_]


def _build_xgboost_model(
    y_train: np.ndarray,
    use_cost_sensitive: bool = True,
) -> MultiLabelXGBoost | MultiOutputClassifier:
    """Cria XGBoost multi-label; scale_pos_weight por defeito quando solicitado."""
    if use_cost_sensitive:
        return MultiLabelXGBoost(random_state=RANDOM_STATE)

    base_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )
    return MultiOutputClassifier(base_clf, n_jobs=-1)


class _ConstantProbaEstimator:
    """Fallback para defeitos com classe única no treino (defeitos raríssimos)."""

    def __init__(self, constant_class: int):
        self.classes_ = np.array([constant_class])
        self._proba = 1.0 if constant_class == 1 else 0.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((X.shape[0], 1), 1.0)


class MultiLabelLogisticRegression:
    """
    Regressão Logística multi-label com class_weight='balanced' (item 2.1).

    Implementação one-vs-rest própria (em vez de MultiOutputClassifier) para
    tolerar defeitos sem exemplos positivos no fold de treino.
    """

    def __init__(self, penalty: str = "l2", random_state: int = RANDOM_STATE):
        self.penalty = penalty
        self.random_state = random_state
        self.estimators_: List[Any] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MultiLabelLogisticRegression":
        solver = "liblinear" if self.penalty == "l1" else "lbfgs"
        self.estimators_ = []
        for defect_idx in range(y_train.shape[1]):
            y_col = y_train[:, defect_idx].astype(int)
            if len(np.unique(y_col)) < 2:
                self.estimators_.append(_ConstantProbaEstimator(int(y_col[0])))
                continue
            clf = LogisticRegression(
                penalty=self.penalty,
                C=1.0,
                class_weight="balanced",
                solver=solver,
                max_iter=2000,
                random_state=self.random_state,
            )
            clf.fit(X_train, y_col)
            self.estimators_.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        return [est.predict_proba(X) for est in self.estimators_]


def _build_logistic_model(penalty: str = "l2") -> MultiLabelLogisticRegression:
    return MultiLabelLogisticRegression(penalty=penalty)


def _build_random_forest_model() -> MultiOutputClassifier:
    base_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return MultiOutputClassifier(base_clf, n_jobs=-1)


def _build_sklearn_model(model_type: str, y_train: np.ndarray) -> Any:
    """Fábrica de modelos sklearn-like por tipo (exceto PyTorch)."""
    if model_type == MODEL_XGBOOST:
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost não instalado. Execute: pip install xgboost")
        return _build_xgboost_model(y_train, use_cost_sensitive=True)
    if model_type == MODEL_RANDOM_FOREST:
        return _build_random_forest_model()
    if model_type == MODEL_LOGISTIC_L2:
        return _build_logistic_model(penalty="l2")
    if model_type == MODEL_LOGISTIC_L1:
        return _build_logistic_model(penalty="l1")
    raise ValueError(f"Modelo desconhecido: {model_type}")


def _predict_pytorch(model: DefectPredictionNN, X_scaled: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_scaled))
        return torch.sigmoid(logits).numpy()


def predict_proba_any(model_type: str, model: Any, X_scaled: np.ndarray) -> np.ndarray:
    """Probabilidades P(defeito=1) para qualquer um dos 3 modelos."""
    if model_type == MODEL_PYTORCH:
        return _predict_pytorch(model, X_scaled)
    return _extract_proba_positive(model.predict_proba(X_scaled), model)


def fit_fold_model(
    model_type: str,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_train_eval_scaled: np.ndarray,
    y_train_eval: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    pos_weights: List[float],
    defect_names: List[str],
) -> Tuple[Any, Dict[str, float], Dict[str, float], float]:
    """
    Treina um modelo em um fold e retorna métricas de treino e validação.

    Métricas de treino são calculadas no fold original (sem amostras SMOTE).
    """
    t0 = time.perf_counter()

    if model_type == MODEL_PYTORCH:
        model, _ = train_single_model(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            pos_weights,
            X_train_scaled.shape[1],
            y_train.shape[1],
            verbose=False,
        )
        train_proba = _predict_pytorch(model, X_train_eval_scaled)
        val_proba = _predict_pytorch(model, X_val_scaled)

    else:
        model = _build_sklearn_model(model_type, y_train)
        model.fit(X_train_scaled, y_train)
        train_proba = _extract_proba_positive(
            model.predict_proba(X_train_eval_scaled), model
        )
        val_proba = _extract_proba_positive(model.predict_proba(X_val_scaled), model)

    train_time = time.perf_counter() - t0
    train_metrics = compute_multilabel_metrics(y_train_eval, train_proba)
    val_metrics = compute_multilabel_metrics(y_val, val_proba)
    return model, train_metrics, val_metrics, train_time


def fit_final_model(
    model_type: str,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    pos_weights: List[float],
    defect_names: List[str],
) -> Tuple[Any, Dict[str, float], float]:
    """Treina modelo final nos 80% completos e avalia no teste (threshold 0.5)."""
    t0 = time.perf_counter()

    if model_type == MODEL_PYTORCH:
        model, _ = train_single_model(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            pos_weights,
            X_train_scaled.shape[1],
            y_train.shape[1],
            verbose=False,
        )
        test_proba = _predict_pytorch(model, X_test_scaled)

    else:
        model = _build_sklearn_model(model_type, y_train)
        model.fit(X_train_scaled, y_train)
        test_proba = _extract_proba_positive(model.predict_proba(X_test_scaled), model)

    train_time = time.perf_counter() - t0
    test_metrics = compute_multilabel_metrics(y_test, test_proba)
    return model, test_metrics, train_time


def _aggregate_fold_metrics(
    fold_metrics: List[FoldMetrics],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Calcula média e desvio-padrão das métricas de treino e validação."""
    train_mean: Dict[str, float] = {}
    train_std: Dict[str, float] = {}
    val_mean: Dict[str, float] = {}
    val_std: Dict[str, float] = {}

    for key in METRIC_KEYS:
        train_values = [fold.train_metrics[key] for fold in fold_metrics]
        val_values = [fold.val_metrics[key] for fold in fold_metrics]
        train_mean[key] = float(np.mean(train_values))
        train_std[key] = float(np.std(train_values))
        val_mean[key] = float(np.mean(val_values))
        val_std[key] = float(np.std(val_values))

    return train_mean, train_std, val_mean, val_std


def run_cross_validation(
    model_type: str,
    split: SplitData,
    use_smote: bool = True,
    verbose: bool = True,
) -> CVResult:
    """
    Executa CV estratificada em 5 folds sobre o dev set para um modelo.

    SMOTE (se habilitado) é aplicado apenas dentro de cada fold de treino.
    """
    model_name = MODEL_DISPLAY_NAMES[model_type]
    cv_splits = get_cv_splits(split.y_dev)
    fold_results: List[FoldMetrics] = []

    if verbose:
        smote_label = "com SMOTE" if use_smote else "sem SMOTE"
        print(f"\n[*] CV 5-fold — {model_name} ({smote_label})")

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_train_fold = split.X_dev[train_idx]
        y_train_fold = split.y_dev[train_idx]
        X_val_fold = split.X_dev[val_idx]
        y_val_fold = split.y_dev[val_idx]

        pos_weights = compute_pos_weights(y_train_fold)

        X_train_scaled, X_val_scaled, y_train_balanced, _ = preprocess_fold(
            X_train_fold,
            X_val_fold,
            y_train_fold.copy(),
            split.defect_names,
            use_smote=use_smote,
            verbose=False,
        )

        scaler_eval = StandardScaler()
        X_train_eval_scaled = scaler_eval.fit_transform(X_train_fold).astype(np.float32)

        _, train_metrics, val_metrics, train_time = fit_fold_model(
            model_type,
            X_train_scaled,
            y_train_balanced,
            X_train_eval_scaled,
            y_train_fold,
            X_val_scaled,
            y_val_fold,
            pos_weights,
            split.defect_names,
        )

        fold_results.append(
            FoldMetrics(
                fold_idx=fold_idx + 1,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                train_time_sec=train_time,
            )
        )

        if verbose:
            print(
                f"    Fold {fold_idx + 1}/5 — "
                f"Recall(val)={val_metrics['recall_micro']:.4f}, "
                f"Precision(val)={val_metrics['precision_micro']:.4f}, "
                f"F1(val)={val_metrics['f1_micro']:.4f}, "
                f"tempo={train_time:.1f}s"
            )

    train_mean, train_std, val_mean, val_std = _aggregate_fold_metrics(fold_results)
    total_time = sum(f.train_time_sec for f in fold_results)

    if verbose:
        print(f"    Média CV (validação): Recall={val_mean['recall_micro']:.4f} "
              f"(±{val_std['recall_micro']:.4f}), "
              f"Precision={val_mean['precision_micro']:.4f} "
              f"(±{val_std['precision_micro']:.4f}), "
              f"F1={val_mean['f1_micro']:.4f} (±{val_std['f1_micro']:.4f})")

    return CVResult(
        model_type=model_type,
        model_name=model_name,
        use_smote=use_smote,
        fold_metrics=fold_results,
        train_mean=train_mean,
        train_std=train_std,
        val_mean=val_mean,
        val_std=val_std,
        total_train_time_sec=total_time,
    )


def run_final_test_evaluation(
    model_type: str,
    split: SplitData,
    use_smote: bool = True,
    verbose: bool = True,
) -> FinalModelResult:
    """Treina nos 80% completos e avalia no teste final (20%), threshold 0.5."""
    model_name = MODEL_DISPLAY_NAMES[model_type]
    pos_weights = compute_pos_weights(split.y_dev)

    X_dev_scaled, X_test_scaled, y_dev_balanced, scaler = preprocess_fold(
        split.X_dev,
        split.X_test,
        split.y_dev.copy(),
        split.defect_names,
        use_smote=use_smote,
        verbose=False,
    )

    if verbose:
        smote_label = "com SMOTE" if use_smote else "sem SMOTE"
        print(f"\n[*] Avaliação final — {model_name} ({smote_label})")

    model, test_metrics, train_time = fit_final_model(
        model_type,
        X_dev_scaled,
        y_dev_balanced,
        X_test_scaled,
        split.y_test,
        pos_weights,
        split.defect_names,
    )

    # Métricas no conjunto de treino efetivo (pós-SMOTE) — item 1.4
    train_proba = predict_proba_any(model_type, model, X_dev_scaled)
    train_metrics_on_balanced = compute_multilabel_metrics(y_dev_balanced, train_proba)

    if verbose:
        print(
            f"    Teste: Recall={test_metrics['recall_micro']:.4f}, "
            f"Precision={test_metrics['precision_micro']:.4f}, "
            f"F1={test_metrics['f1_micro']:.4f}, tempo={train_time:.1f}s"
        )

    return FinalModelResult(
        model_type=model_type,
        model_name=model_name,
        use_smote=use_smote,
        test_metrics=test_metrics,
        cv_val_mean={},
        train_time_sec=train_time,
        model=model,
        scaler=scaler,
        train_metrics_on_balanced=train_metrics_on_balanced,
    )


def run_overfitting_analysis(
    split: SplitData,
    use_smote: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Análise formal de overfitting: métricas de treino (CV), validação (CV)
    e teste final para cada um dos 3 modelos.
    """
    results = []
    for model_type in (MODEL_PYTORCH, MODEL_XGBOOST, MODEL_RANDOM_FOREST):
        if model_type == MODEL_XGBOOST and not XGBOOST_AVAILABLE:
            continue

        cv_result = run_cross_validation(
            model_type, split, use_smote=use_smote, verbose=verbose
        )
        final_result = run_final_test_evaluation(
            model_type, split, use_smote=use_smote, verbose=verbose
        )
        final_result.cv_val_mean = cv_result.val_mean

        gap_train_val = {
            key: cv_result.train_mean[key] - cv_result.val_mean[key]
            for key in ("recall_micro", "precision_micro", "f1_micro")
        }
        gap_val_test = {
            key: cv_result.val_mean[key] - final_result.test_metrics[key]
            for key in ("recall_micro", "precision_micro", "f1_micro")
        }

        results.append(
            {
                "model_type": model_type,
                "model_name": cv_result.model_name,
                "use_smote": use_smote,
                "cv_train_mean": cv_result.train_mean,
                "cv_train_std": cv_result.train_std,
                "cv_val_mean": cv_result.val_mean,
                "cv_val_std": cv_result.val_std,
                "test_metrics": final_result.test_metrics,
                "gap_train_val": gap_train_val,
                "gap_val_test": gap_val_test,
                "cv_total_train_time_sec": cv_result.total_train_time_sec,
                "final_train_time_sec": final_result.train_time_sec,
            }
        )

    return {"use_smote": use_smote, "models": results}


def run_smote_comparison(
    split: SplitData,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compara cada modelo com e sem SMOTE usando o mesmo pipeline de CV.
    """
    comparison = []
    for model_type in (MODEL_PYTORCH, MODEL_XGBOOST, MODEL_RANDOM_FOREST):
        if model_type == MODEL_XGBOOST and not XGBOOST_AVAILABLE:
            continue

        model_name = MODEL_DISPLAY_NAMES[model_type]
        condition_results = {}

        for use_smote in (False, True):
            cv_result = run_cross_validation(
                model_type, split, use_smote=use_smote, verbose=verbose
            )
            condition_label = "with_smote" if use_smote else "without_smote"
            condition_results[condition_label] = {
                "use_smote": use_smote,
                "cv_val_mean": cv_result.val_mean,
                "cv_val_std": cv_result.val_std,
                "cv_train_mean": cv_result.train_mean,
                "total_train_time_sec": cv_result.total_train_time_sec,
            }

        without = condition_results["without_smote"]
        with_sm = condition_results["with_smote"]
        delta = {
            key: with_sm["cv_val_mean"][key] - without["cv_val_mean"][key]
            for key in METRIC_KEYS
        }
        delta["train_time_sec"] = (
            with_sm["total_train_time_sec"] - without["total_train_time_sec"]
        )

        comparison.append(
            {
                "model_type": model_type,
                "model_name": model_name,
                "without_smote": without,
                "with_smote": with_sm,
                "delta_with_minus_without": delta,
            }
        )

    return {"models": comparison}


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def dataframe_to_markdown(df) -> str:
    """Converte DataFrame para tabela markdown sem depender de tabulate."""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def available_model_types() -> List[str]:
    """Modelos disponíveis no ambiente (exclui XGBoost se não instalado)."""
    models = [MODEL_PYTORCH, MODEL_XGBOOST, MODEL_RANDOM_FOREST]
    if not XGBOOST_AVAILABLE:
        models.remove(MODEL_XGBOOST)
    return models


def cv_result_to_dict(cv_result: CVResult) -> Dict[str, Any]:
    return {
        "model_type": cv_result.model_type,
        "model_name": cv_result.model_name,
        "use_smote": cv_result.use_smote,
        "train_mean": cv_result.train_mean,
        "train_std": cv_result.train_std,
        "val_mean": cv_result.val_mean,
        "val_std": cv_result.val_std,
        "total_train_time_sec": cv_result.total_train_time_sec,
        "folds": [asdict(f) for f in cv_result.fold_metrics],
    }
