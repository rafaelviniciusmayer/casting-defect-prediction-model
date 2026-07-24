"""
Utilitários compartilhados da Fase 3 — Diagnósticos e interpretação.

Responsabilidades:
  - Treinar UMA vez os 5 modelos finais na configuração oficial da Fase 1
    (sem SMOTE, apenas cost-sensitive) e cachear modelos + probabilidades de
    teste em disco, para que os itens 3.1-3.4 não retreinem nada.
  - Mapear cada feature para a fase do processo (injeção, intensificação,
    resfriamento, configuração/manutenção) e para a categoria de feature
    engineering (ratio, produto, diferença, distância, binária de faixa,
    agregação, domínio, transformação matemática) — usado no item 3.1.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from unified_cv_pipeline import (
    MODEL_DISPLAY_NAMES,
    MODEL_LOGISTIC_L1,
    MODEL_LOGISTIC_L2,
    MODEL_PYTORCH,
    MODEL_RANDOM_FOREST,
    MODEL_XGBOOST,
    available_model_types,
    load_pipeline_data,
    predict_proba_any,
    prepare_split_data,
    run_final_test_evaluation,
)

CACHE_PATH = Path("models/phase3_final_models.pkl")
PHASE1_DECISION_PATH = Path("reports/phase1_smote_comparison.json")

PHASE3_MODEL_TYPES = [
    MODEL_PYTORCH,
    MODEL_XGBOOST,
    MODEL_RANDOM_FOREST,
    MODEL_LOGISTIC_L2,
    MODEL_LOGISTIC_L1,
]

# ---------------------------------------------------------------------------
# Mapeamento fase do processo / categoria de feature engineering (item 3.1)
# ---------------------------------------------------------------------------

PHASE_INJECTION = "Injeção"
PHASE_INTENSIFICATION = "Intensificação"
PHASE_COOLING = "Resfriamento"
PHASE_CONFIG = "Configuração/Manutenção"
PHASE_MULTIPLE = "Múltiplas fases"
PHASE_AGGREGATE = "Agregada/Global"

BASE_VAR_PHASES = {
    "piston_velocity_phase1": PHASE_INJECTION,
    "metal_velocity_gate": PHASE_INJECTION,
    "fill_time": PHASE_INJECTION,
    "phase_transition_position": PHASE_INJECTION,
    "sleeve_fill_percentage": PHASE_INJECTION,
    "intensification_time_phase3": PHASE_INTENSIFICATION,
    "intensification_pressure": PHASE_INTENSIFICATION,
    "solidification_time": PHASE_COOLING,
    "cycle_time": PHASE_COOLING,
    "sleeve_diameter": PHASE_CONFIG,
    "sleeve_length": PHASE_CONFIG,
    "plunger_lubricant": PHASE_CONFIG,
    "plunger_sleeve_clearance": PHASE_CONFIG,
    "sleeve_temperature": PHASE_CONFIG,
    "plunger_temperature": PHASE_CONFIG,
}

# Features derivadas sem correspondência direta por substring
EXPLICIT_PHASE_OVERRIDES = {
    "velocity_ratio": PHASE_INJECTION,
    "velocity_distance": PHASE_INJECTION,
    "weighted_velocity": PHASE_INJECTION,
    "fill_volume_ratio": PHASE_MULTIPLE,
    "temp_ratio": PHASE_CONFIG,
    "temp_diff": PHASE_CONFIG,
    "temp_solidification": PHASE_MULTIPLE,
    "cycle_fill_diff": PHASE_MULTIPLE,
    "pressure_time_ratio": PHASE_INTENSIFICATION,
    "pressure_deviation": PHASE_INTENSIFICATION,
    "intensification_energy": PHASE_INTENSIFICATION,
    "sleeve_aspect_ratio": PHASE_CONFIG,
    "sleeve_volume_estimate": PHASE_CONFIG,
    "solidification_ratio": PHASE_COOLING,
    "fill_time_ratio": PHASE_INJECTION,
    "total_process_time": PHASE_MULTIPLE,
    "process_efficiency": PHASE_MULTIPLE,
    "n_vars_in_range": PHASE_AGGREGATE,
    "n_vars_out_of_range": PHASE_AGGREGATE,
    "avg_distance_from_ideal": PHASE_AGGREGATE,
    "max_distance_from_ideal": PHASE_AGGREGATE,
}

CATEGORY_ORIGINAL = "Original"
CATEGORY_RATIO = "Ratio"
CATEGORY_PRODUCT = "Produto"
CATEGORY_DIFFERENCE = "Diferença"
CATEGORY_DISTANCE = "Distância de faixa ideal"
CATEGORY_RANGE_FLAG = "Binária de faixa"
CATEGORY_AGGREGATION = "Agregação estatística"
CATEGORY_DOMAIN = "Específica de domínio"
CATEGORY_MATH = "Transformação matemática"

PHASE_LABELS_EN = {
    PHASE_INJECTION: "Injection",
    PHASE_INTENSIFICATION: "Intensification",
    PHASE_COOLING: "Cooling",
    PHASE_CONFIG: "Configuration/Maintenance",
    PHASE_MULTIPLE: "Multiple phases",
    PHASE_AGGREGATE: "Global/Aggregation",
}

CATEGORY_LABELS_EN = {
    CATEGORY_ORIGINAL: "Original",
    CATEGORY_RATIO: "Ratio",
    CATEGORY_PRODUCT: "Product",
    CATEGORY_DIFFERENCE: "Difference",
    CATEGORY_DISTANCE: "Ideal-range distance",
    CATEGORY_RANGE_FLAG: "Range flag",
    CATEGORY_AGGREGATION: "Statistical aggregation",
    CATEGORY_DOMAIN: "Domain-specific",
    CATEGORY_MATH: "Mathematical transform",
}


def phase_label_en(phase: str) -> str:
    return PHASE_LABELS_EN.get(phase, phase)


def category_label_en(category: str) -> str:
    return CATEGORY_LABELS_EN.get(category, category)


DOMAIN_FEATURES = {
    "sleeve_volume_estimate",
    "fill_volume_ratio",
    "weighted_velocity",
    "total_process_time",
    "process_efficiency",
}
PRODUCT_FEATURES = {"intensification_energy", "temp_solidification"}
AGGREGATION_FEATURES = {
    "n_vars_in_range",
    "n_vars_out_of_range",
    "avg_distance_from_ideal",
    "max_distance_from_ideal",
}


def get_feature_phase(feature_name: str) -> str:
    """Fase do processo à qual a feature pertence."""
    if feature_name in EXPLICIT_PHASE_OVERRIDES:
        return EXPLICIT_PHASE_OVERRIDES[feature_name]
    if feature_name in BASE_VAR_PHASES:
        return BASE_VAR_PHASES[feature_name]
    # Derivadas: casa por substring da variável base (nomes mais longos primeiro
    # para evitar que 'fill_time' case com 'fill_time_ratio' errado etc.)
    matched_phases = set()
    for base_var in sorted(BASE_VAR_PHASES, key=len, reverse=True):
        if base_var in feature_name:
            matched_phases.add(BASE_VAR_PHASES[base_var])
    if len(matched_phases) == 1:
        return matched_phases.pop()
    if len(matched_phases) > 1:
        return PHASE_MULTIPLE
    return PHASE_AGGREGATE


def get_feature_category(feature_name: str) -> str:
    """Categoria de feature engineering."""
    if feature_name in BASE_VAR_PHASES:
        return CATEGORY_ORIGINAL
    if feature_name in DOMAIN_FEATURES:
        return CATEGORY_DOMAIN
    if feature_name in PRODUCT_FEATURES:
        return CATEGORY_PRODUCT
    if feature_name in AGGREGATION_FEATURES:
        return CATEGORY_AGGREGATION
    if feature_name.endswith(("_log", "_squared")):
        return CATEGORY_MATH
    if "distance_from" in feature_name:
        return CATEGORY_DISTANCE
    if feature_name.endswith(("_in_range", "_above_range", "_below_range")):
        return CATEGORY_RANGE_FLAG
    if "ratio" in feature_name:
        return CATEGORY_RATIO
    if "diff" in feature_name or "deviation" in feature_name or "distance" in feature_name:
        return CATEGORY_DIFFERENCE
    return CATEGORY_DOMAIN


# ---------------------------------------------------------------------------
# Cache de modelos finais + probabilidades (evita retreino nos itens 3.1-3.4)
# ---------------------------------------------------------------------------

def load_official_use_smote() -> bool:
    with open(PHASE1_DECISION_PATH, encoding="utf-8") as f:
        return bool(json.load(f)["decision"]["keep_smote"])


def build_or_load_final_models(force: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """
    Treina os 5 modelos finais (config oficial da Fase 1) e cacheia em disco.

    Retorna dict com:
      'use_smote', 'feature_names', 'defect_names', 'y_test', 'y_dev',
      'X_test_scaled', 'X_dev_scaled', 'scaler',
      'models': {model_type: {'model', 'test_proba', 'train_time_sec'}}
    """
    if CACHE_PATH.exists() and not force:
        if verbose:
            print(f"[*] Carregando cache de modelos finais: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    use_smote = load_official_use_smote()
    if verbose:
        print(f"[*] Treinando modelos finais (config oficial: "
              f"{'com' if use_smote else 'sem'} SMOTE)...")

    data = load_pipeline_data(verbose=verbose)
    split = prepare_split_data(data, verbose=verbose)

    cache: Dict[str, Any] = {
        "use_smote": use_smote,
        "feature_names": split.feature_names,
        "defect_names": split.defect_names,
        "y_test": split.y_test,
        "y_dev": split.y_dev,
        "models": {},
    }

    for model_type in PHASE3_MODEL_TYPES:
        t0 = time.perf_counter()
        final = run_final_test_evaluation(
            model_type, split, use_smote=use_smote, verbose=verbose
        )
        X_test_scaled = final.scaler.transform(split.X_test).astype(np.float32)
        test_proba = predict_proba_any(model_type, final.model, X_test_scaled)

        if "X_test_scaled" not in cache:
            cache["X_test_scaled"] = X_test_scaled
            cache["X_dev_scaled"] = final.scaler.transform(split.X_dev).astype(np.float32)
            cache["scaler"] = final.scaler

        cache["models"][model_type] = {
            "model": final.model,
            "model_name": final.model_name,
            "test_proba": test_proba,
            "test_metrics_threshold_05": final.test_metrics,
            "train_time_sec": final.train_time_sec,
        }
        if verbose:
            print(f"    [{MODEL_DISPLAY_NAMES[model_type]}] pronto "
                  f"({time.perf_counter() - t0:.1f}s)")

    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    if verbose:
        print(f"[OK] Cache salvo em {CACHE_PATH}")
    return cache


def most_frequent_defects(y: np.ndarray, defect_names: List[str], top_n: int = 3) -> List[int]:
    """Índices dos defeitos mais frequentes (para ROC/calibração)."""
    counts = y.sum(axis=0)
    return list(np.argsort(counts)[::-1][:top_n])
