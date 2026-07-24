"""
Item 3.1 (Fase 3) — Análise de importância de features.

Pergunta da banca: "As variáveis da etapa de injeção são as que mais
impactam a ocorrência de defeitos?"

Métodos por modelo:
  - Random Forest / XGBoost: feature_importances_ nativo (média entre os 28
    estimadores one-vs-rest) + SHAP TreeExplainer (média |SHAP| em amostra
    do teste, média entre defeitos mais frequentes).
  - PyTorch NN: permutation importance (queda de F1-micro ao permutar cada
    feature no teste).
  - Regressão Logística: |coeficiente| médio entre defeitos (features já
    padronizadas pelo StandardScaler, então os coeficientes são comparáveis).

Agregações: por fase do processo (injeção, intensificação, resfriamento,
configuração/manutenção) e por categoria de feature engineering.

Saídas:
  figures/phase3_feature_importance_top25.png
  figures/phase3_importance_by_phase.png
  figures/table_phase3_importance_by_phase.csv
  figures/table_phase3_importance_by_category.csv
  figures/table_phase3_top_features.csv
  reports/phase3_feature_importance.{json,md}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import f1_score

from phase3_utils import (
    PHASE_AGGREGATE,
    PHASE_CONFIG,
    PHASE_COOLING,
    PHASE_INJECTION,
    PHASE_INTENSIFICATION,
    PHASE_MULTIPLE,
    build_or_load_final_models,
    category_label_en,
    get_feature_category,
    get_feature_phase,
    most_frequent_defects,
    phase_label_en,
)
from unified_cv_pipeline import (
    MODEL_DISPLAY_NAMES,
    MODEL_LOGISTIC_L1,
    MODEL_LOGISTIC_L2,
    MODEL_PYTORCH,
    MODEL_RANDOM_FOREST,
    MODEL_XGBOOST,
    dataframe_to_markdown,
)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

RNG = np.random.RandomState(42)
SHAP_SAMPLE_SIZE = 300
SHAP_TOP_DEFECTS = 5
PERMUTATION_SAMPLE_SIZE = 2000


def _normalize(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    return values / total if total > 0 else values


def importance_tree_native(model: Any) -> np.ndarray:
    """Média de feature_importances_ entre os estimadores one-vs-rest."""
    importances = [est.feature_importances_ for est in model.estimators_
                   if hasattr(est, "feature_importances_")]
    return _normalize(np.mean(importances, axis=0))


def importance_logistic_coef(model: Any) -> np.ndarray:
    """|coeficiente| médio entre defeitos (features padronizadas)."""
    coefs = [np.abs(est.coef_[0]) for est in model.estimators_
             if hasattr(est, "coef_")]
    return _normalize(np.mean(coefs, axis=0))


def importance_shap_tree(
    model: Any,
    X_sample: np.ndarray,
    defect_indices: List[int],
) -> np.ndarray:
    """Média de |SHAP| (TreeExplainer) nos defeitos mais frequentes."""
    all_importances = []
    for defect_idx in defect_indices:
        est = model.estimators_[defect_idx]
        explainer = shap.TreeExplainer(est)
        shap_values = explainer.shap_values(X_sample)
        # Para binário, shap_values pode ser lista [classe0, classe1] ou array
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        all_importances.append(np.abs(shap_values).mean(axis=0))
    return _normalize(np.mean(all_importances, axis=0))


def importance_permutation_nn(
    model: Any,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    sample_size: int = PERMUTATION_SAMPLE_SIZE,
) -> np.ndarray:
    """Permutation importance para a NN: queda de F1-micro por feature."""
    from unified_cv_pipeline import predict_proba_any, MODEL_PYTORCH as MT

    idx = RNG.choice(len(X_test_scaled), size=min(sample_size, len(X_test_scaled)),
                     replace=False)
    X_sub = X_test_scaled[idx].copy()
    y_sub = y_test[idx]

    baseline_proba = predict_proba_any(MT, model, X_sub)
    baseline_f1 = f1_score(y_sub, (baseline_proba >= 0.5).astype(int),
                           average="micro", zero_division=0)

    n_features = X_sub.shape[1]
    drops = np.zeros(n_features)
    for j in range(n_features):
        X_perm = X_sub.copy()
        X_perm[:, j] = RNG.permutation(X_perm[:, j])
        proba = predict_proba_any(MT, model, X_perm)
        f1 = f1_score(y_sub, (proba >= 0.5).astype(int),
                      average="micro", zero_division=0)
        drops[j] = max(0.0, baseline_f1 - f1)
    return _normalize(drops)


def compute_all_importances(cache: Dict[str, Any], verbose: bool = True) -> Dict[str, np.ndarray]:
    """Importância normalizada (soma=1) por modelo."""
    X_test_scaled = cache["X_test_scaled"]
    y_test = cache["y_test"]
    defect_indices = most_frequent_defects(y_test, cache["defect_names"], SHAP_TOP_DEFECTS)

    sample_idx = RNG.choice(len(X_test_scaled),
                            size=min(SHAP_SAMPLE_SIZE, len(X_test_scaled)),
                            replace=False)
    X_sample = X_test_scaled[sample_idx].astype(np.float64)

    importances: Dict[str, np.ndarray] = {}

    for model_type, entry in cache["models"].items():
        t0 = time.perf_counter()
        model = entry["model"]
        if model_type == MODEL_PYTORCH:
            importances[model_type] = importance_permutation_nn(
                model, X_test_scaled, y_test
            )
            method = "permutation importance"
        elif model_type in (MODEL_XGBOOST, MODEL_RANDOM_FOREST):
            native = importance_tree_native(model)
            if SHAP_AVAILABLE:
                try:
                    shap_imp = importance_shap_tree(model, X_sample, defect_indices)
                    importances[model_type] = _normalize((native + shap_imp) / 2)
                    method = "feature_importances_ + SHAP (média)"
                except Exception as exc:
                    importances[model_type] = native
                    method = f"feature_importances_ (SHAP falhou: {exc})"
            else:
                importances[model_type] = native
                method = "feature_importances_"
        elif model_type in (MODEL_LOGISTIC_L2, MODEL_LOGISTIC_L1):
            importances[model_type] = importance_logistic_coef(model)
            method = "|coeficientes| padronizados"
        else:
            continue
        if verbose:
            print(f"    [{entry['model_name']}] {method} "
                  f"({time.perf_counter() - t0:.1f}s)")

    return importances


def build_feature_table(
    importances: Dict[str, np.ndarray],
    cache: Dict[str, Any],
) -> pd.DataFrame:
    """Tabela por feature com importância de cada modelo + média + fase/categoria."""
    feature_names = cache["feature_names"]
    df = pd.DataFrame({"feature": feature_names})
    model_cols = []
    for model_type, imp in importances.items():
        col = MODEL_DISPLAY_NAMES.get(model_type, cache["models"][model_type]["model_name"])
        df[col] = imp
        model_cols.append(col)
    df["importancia_media"] = df[model_cols].mean(axis=1)
    df["fase_processo"] = df["feature"].map(get_feature_phase)
    df["categoria_fe"] = df["feature"].map(get_feature_category)
    return df.sort_values("importancia_media", ascending=False).reset_index(drop=True)


def plot_top_features(df: pd.DataFrame, top_n: int = 25,
                      path: str = "figures/phase3_feature_importance_top25.png") -> str:
    top = df.head(top_n).iloc[::-1]
    phase_colors = {
        PHASE_INJECTION: "tab:red",
        PHASE_INTENSIFICATION: "tab:orange",
        PHASE_COOLING: "tab:blue",
        PHASE_CONFIG: "tab:green",
        PHASE_MULTIPLE: "tab:purple",
        PHASE_AGGREGATE: "tab:gray",
    }
    phase_legend = {phase: phase_label_en(phase) for phase in phase_colors}
    colors = [phase_colors.get(p, "tab:gray") for p in top["fase_processo"]]

    fig, ax = plt.subplots(figsize=(10, 0.35 * top_n + 2))
    ax.barh(top["feature"], top["importancia_media"], color=colors)
    ax.set_xlabel("Mean normalized importance (5 models)")
    ax.set_title(f"Top {top_n} most important features — mean across models")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in phase_colors.values()]
    ax.legend(handles, phase_legend.values(), title="Process phase",
              loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def aggregate_importance(df: pd.DataFrame, by: str, model_cols: List[str]) -> pd.DataFrame:
    agg = df.groupby(by)[model_cols + ["importancia_media"]].sum()
    agg["n_features"] = df.groupby(by).size()
    agg = agg.sort_values("importancia_media", ascending=False).reset_index()
    for col in model_cols + ["importancia_media"]:
        agg[col] = agg[col].round(4)
    return agg


def plot_importance_by_phase(agg_df: pd.DataFrame,
                             path: str = "figures/phase3_importance_by_phase.png") -> str:
    labels = [phase_label_en(p) for p in agg_df["fase_processo"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, agg_df["importancia_media"], color="tab:red")
    for i, (v, n) in enumerate(zip(agg_df["importancia_media"], agg_df["n_features"])):
        ax.text(i, v + 0.005, f"{v:.3f}\n({n} feat.)", ha="center", fontsize=9)
    ax.set_ylabel("Total importance (mean across models)")
    ax.set_title("Aggregated feature importance by process phase")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    print("=" * 70)
    print("FASE 3.1 — IMPORTÂNCIA DE FEATURES")
    print("=" * 70)
    Path("reports").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    cache = build_or_load_final_models(verbose=True)

    print("\n[*] Calculando importâncias por modelo...")
    importances = compute_all_importances(cache, verbose=True)

    df = build_feature_table(importances, cache)
    model_cols = [
        MODEL_DISPLAY_NAMES.get(mt, cache["models"][mt]["model_name"])
        for mt in importances
    ]

    df_out = df.copy()
    for col in model_cols + ["importancia_media"]:
        df_out[col] = df_out[col].round(5)
    df_out["process_phase"] = df_out["fase_processo"].map(phase_label_en)
    df_out["feature_category"] = df_out["categoria_fe"].map(category_label_en)
    export_cols = ["feature"] + model_cols + ["importancia_media", "process_phase", "feature_category"]
    df_out[export_cols].to_csv("figures/table_phase3_top_features.csv", index=False)

    by_phase = aggregate_importance(df, "fase_processo", model_cols)
    by_category = aggregate_importance(df, "categoria_fe", model_cols)
    by_phase["process_phase"] = by_phase["fase_processo"].map(phase_label_en)
    by_category["feature_category"] = by_category["categoria_fe"].map(category_label_en)
    by_phase[["process_phase"] + model_cols + ["importancia_media", "n_features"]].to_csv(
        "figures/table_phase3_importance_by_phase.csv", index=False
    )
    by_category[["feature_category"] + model_cols + ["importancia_media", "n_features"]].to_csv(
        "figures/table_phase3_importance_by_category.csv", index=False
    )

    fig1 = plot_top_features(df)
    fig2 = plot_importance_by_phase(by_phase)
    print(f"    [OK] {fig1}")
    print(f"    [OK] {fig2}")

    injection_share = float(
        by_phase.loc[by_phase["fase_processo"] == PHASE_INJECTION, "importancia_media"].sum()
    )
    top_phase = by_phase.iloc[0]["fase_processo"]
    answer = (
        f"A fase com maior importância agregada é **{top_phase}**. "
        f"A fase de injeção concentra {injection_share:.1%} da importância total "
        f"média entre os 5 modelos."
    )

    md = "\n".join([
        "# Fase 3.1 — Importância de features\n",
        "Pergunta da banca: *as variáveis da etapa de injeção são as que mais "
        "impactam?*\n",
        f"**Resposta:** {answer}\n",
        "## Importância agregada por fase do processo\n",
        dataframe_to_markdown(by_phase),
        "\n## Importância agregada por categoria de feature engineering\n",
        dataframe_to_markdown(by_category),
        "\n## Top 20 features (média entre modelos)\n",
        dataframe_to_markdown(
            df_out.head(20)[["feature", "fase_processo", "categoria_fe",
                             "importancia_media"] + model_cols]
        ),
        "",
        f"![Top 25 features](phase3_feature_importance_top25.png)",
        f"![Por fase](phase3_importance_by_phase.png)",
        "",
    ])
    Path("reports/phase3_feature_importance.md").write_text(md, encoding="utf-8")

    payload = {
        "answer": answer,
        "injection_share": injection_share,
        "by_phase": by_phase.to_dict(orient="records"),
        "by_category": by_category.to_dict(orient="records"),
        "top_30_features": df_out.head(30).to_dict(orient="records"),
    }
    with open("reports/phase3_feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n[RESPOSTA À BANCA] {answer}")
    print("[OK] reports/phase3_feature_importance.{json,md}")


if __name__ == "__main__":
    main()
