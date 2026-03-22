"""
Script de Análise de Resultados do Modelo
==========================================

Analisa os resultados do treinamento focando em MAXIMIZAR DETECÇÃO DE DEFEITOS.
Objetivo: Identificar a maior quantidade possível de peças com defeito.

Os valores de Recall, F1-Score e Precision são SAÍDAS do processo de otimização,
não critérios de entrada.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

def load_artifacts_safely():
    """Carrega artifacts do modelo de forma segura."""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except (AttributeError, ModuleNotFoundError) as e:
        print(f"[AVISO] Erro ao carregar modelo completo: {e}")
        print("[INFO] Tentando carregar apenas métricas e thresholds...")
        
        thresholds = {}
        try:
            with open('optimal_thresholds.pkl', 'rb') as f:
                thresholds = pickle.load(f)
        except FileNotFoundError:
            pass
        
        artifacts = {
            'metrics': {},
            'optimal_thresholds': thresholds,
            'process_vars': [],
            'defect_cols': []
        }
        return artifacts

def analyze_model_results():
    """Analisa os resultados do modelo treinado focando em detecção máxima de defeitos."""
    
    print("="*70)
    print("ANÁLISE DE RESULTADOS - FOCO EM MAXIMIZAR DETECÇÃO DE DEFEITOS")
    print("="*70)
    
    # Carregar modelo
    try:
        artifacts = load_artifacts_safely()
        print("\n[OK] Dados do modelo carregados")
    except Exception as e:
        print(f"\n[ERRO] Erro ao carregar modelo: {e}")
        print("[INFO] Execute train_model.py primeiro para gerar o modelo.")
        return
    
    # Tentar carregar métricas de JSON primeiro
    metrics = {}
    thresholds = {}
    feature_names = []
    defect_names = []
    
    try:
        with open('model_metrics.json', 'r') as f:
            metrics_json = json.load(f)
            metrics = {
                'f1_micro': metrics_json.get('f1_micro', 0),
                'f1_macro': metrics_json.get('f1_macro', 0),
                'precision_micro': metrics_json.get('precision_micro', 0),
                'recall_micro': metrics_json.get('recall_micro', 0),
                'accuracy': metrics_json.get('accuracy', 0)
            }
            thresholds = metrics_json.get('thresholds', {})
            print("\n[OK] Métricas carregadas de model_metrics.json")
    except FileNotFoundError:
        # Tentar extrair do artifacts
        metrics = artifacts.get('metrics', {})
        thresholds = artifacts.get('optimal_thresholds', {})
        feature_names = artifacts.get('process_vars', [])
        defect_names = artifacts.get('defect_cols', [])
        
        if not metrics:
            print("\n[AVISO] Métricas não encontradas.")
            print("[INFO] Execute train_model.py para gerar métricas completas.")
            return
    
    # =============================================================================
    # 1. MÉTRICAS OBTIDAS (SAÍDAS DO PROCESSO DE OTIMIZAÇÃO)
    # =============================================================================
    print("\n" + "="*70)
    print("1. MÉTRICAS OBTIDAS (Resultados da Otimização)")
    print("="*70)
    
    recall = metrics.get('recall_micro', 0)
    precision = metrics.get('precision_micro', 0)
    f1_micro = metrics.get('f1_micro', 0)
    f1_macro = metrics.get('f1_macro', 0)
    accuracy = metrics.get('accuracy', 0)
    
    print(f"\nMétricas no Conjunto de Teste:")
    print(f"  Recall (Sensibilidade):     {recall:.4f} ({recall*100:.2f}%)")
    print(f"  Precision (Precisão):       {precision:.4f} ({precision*100:.2f}%)")
    print(f"  F1-Score (Micro):           {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    print(f"  F1-Score (Macro):           {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  Accuracy (Acurácia):         {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # =============================================================================
    # 2. ANÁLISE DE DETECÇÃO DE DEFEITOS (FOCO PRINCIPAL)
    # =============================================================================
    print("\n" + "="*70)
    print("2. ANÁLISE DE DETECÇÃO DE DEFEITOS")
    print("="*70)
    
    # Taxa de falsos negativos (defeitos que passam - CRÍTICO)
    false_negative_rate = 1 - recall
    true_positive_rate = recall
    
    print(f"\n[*] Detecção de Defeitos Reais:")
    print(f"    Taxa de Detecção (Recall):     {recall:.4f} ({recall*100:.2f}%)")
    print(f"    Taxa de Falsos Negativos:      {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
    
    # Estimar impacto em produção
    print(f"\n[*] Impacto Estimado em Produção (por 1000 peças):")
    print(f"    Defeitos Detectados:           ~{recall*1000:.0f} peças")
    print(f"    Defeitos que Passam (FN):       ~{false_negative_rate*1000:.0f} peças")
    
    # Interpretação do Recall
    if recall >= 0.90:
        recall_status = "EXCELENTE - Modelo detecta mais de 90% dos defeitos"
        recall_color = "[OK]"
    elif recall >= 0.80:
        recall_status = "MUITO BOM - Modelo detecta mais de 80% dos defeitos"
        recall_color = "[OK]"
    elif recall >= 0.70:
        recall_status = "BOM - Modelo detecta mais de 70% dos defeitos"
        recall_color = "[OK]"
    elif recall >= 0.60:
        recall_status = "REGULAR - Modelo detecta mais de 60% dos defeitos"
        recall_color = "[!]"
    else:
        recall_status = "BAIXO - Modelo detecta menos de 60% dos defeitos"
        recall_color = "[X]"
    
    print(f"\n    {recall_color} {recall_status}")
    
    # =============================================================================
    # 3. ANÁLISE DE PRECISION (TRADE-OFF)
    # =============================================================================
    print("\n" + "="*70)
    print("3. ANÁLISE DE PRECISION (Trade-off com Recall)")
    print("="*70)
    
    false_positive_rate = 1 - precision
    
    print(f"\n[*] Precisão das Predições:")
    print(f"    Precision:                     {precision:.4f} ({precision*100:.2f}%)")
    print(f"    Taxa de Falsos Positivos:      {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
    
    print(f"\n[*] Impacto Estimado em Produção (por 1000 peças):")
    print(f"    Peças Boas Rejeitadas (FP):   ~{false_positive_rate*1000:.0f} peças")
    
    # Interpretação do Precision
    if precision >= 0.80:
        precision_status = "ALTA - Poucos falsos positivos"
        precision_color = "[OK]"
    elif precision >= 0.70:
        precision_status = "MODERADA - Alguns falsos positivos"
        precision_color = "[!]"
    elif precision >= 0.60:
        precision_status = "BAIXA - Muitos falsos positivos"
        precision_color = "[!]"
    else:
        precision_status = "MUITO BAIXA - Muitos falsos positivos"
        precision_color = "[X]"
    
    print(f"\n    {precision_color} {precision_status}")
    
    # =============================================================================
    # 4. BALANCEAMENTO (F1-SCORE)
    # =============================================================================
    print("\n" + "="*70)
    print("4. BALANCEAMENTO ENTRE RECALL E PRECISION (F1-Score)")
    print("="*70)
    
    print(f"\n[*] F1-Score (Média Harmônica):")
    print(f"    F1-Score (Micro):             {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    print(f"    F1-Score (Macro):             {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    
    # Interpretação do F1-Score
    if f1_micro >= 0.80:
        f1_status = "EXCELENTE balanceamento"
        f1_color = "[OK]"
    elif f1_micro >= 0.70:
        f1_status = "BOM balanceamento"
        f1_color = "[OK]"
    elif f1_micro >= 0.60:
        f1_status = "REGULAR balanceamento"
        f1_color = "[!]"
    else:
        f1_status = "BAIXO balanceamento"
        f1_color = "[X]"
    
    print(f"\n    {f1_color} {f1_status}")
    
    # =============================================================================
    # 5. ANÁLISE DE THRESHOLDS OTIMIZADOS
    # =============================================================================
    print("\n" + "="*70)
    print("5. ANÁLISE DE THRESHOLDS OTIMIZADOS")
    print("="*70)
    
    if thresholds:
        threshold_values = list(thresholds.values())
        avg_threshold = np.mean(threshold_values)
        min_threshold = np.min(threshold_values)
        max_threshold = np.max(threshold_values)
        median_threshold = np.median(threshold_values)
        
        print(f"\n[*] Estatísticas dos Thresholds:")
        print(f"    Média:   {avg_threshold:.3f}")
        print(f"    Mediana: {median_threshold:.3f}")
        print(f"    Mínimo:  {min_threshold:.3f}")
        print(f"    Máximo:  {max_threshold:.3f}")
        
        # Distribuição de thresholds
        low_thresholds = sum(1 for t in threshold_values if t < 0.3)
        medium_thresholds = sum(1 for t in threshold_values if 0.3 <= t < 0.6)
        high_thresholds = sum(1 for t in threshold_values if t >= 0.6)
        
        print(f"\n[*] Distribuição:")
        print(f"    Thresholds Baixos (< 0.3):    {low_thresholds} defeitos (alta sensibilidade)")
        print(f"    Thresholds Médios (0.3-0.6):  {medium_thresholds} defeitos")
        print(f"    Thresholds Altos (>= 0.6):    {high_thresholds} defeitos (baixa sensibilidade)")
        
        if avg_threshold < 0.4:
            threshold_strategy = "Estratégia: Thresholds baixos para maximizar Recall"
        elif avg_threshold < 0.6:
            threshold_strategy = "Estratégia: Thresholds moderados para balanceamento"
        else:
            threshold_strategy = "Estratégia: Thresholds altos (pode estar limitando Recall)"
        
        print(f"\n    {threshold_strategy}")
        
        # Mostrar top 10 defeitos com thresholds mais baixos (mais sensíveis)
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])
        print(f"\n[*] Top 10 Defeitos com Maior Sensibilidade (thresholds mais baixos):")
        for i, (defect, threshold) in enumerate(sorted_thresholds[:10], 1):
            print(f"    {i:2d}. {defect:<35}: {threshold:.3f}")
    
    # =============================================================================
    # 6. RESUMO E INTERPRETAÇÃO FINAL
    # =============================================================================
    print("\n" + "="*70)
    print("6. RESUMO E INTERPRETAÇÃO FINAL")
    print("="*70)
    
    print(f"\n[*] Objetivo Principal: Maximizar Detecção de Defeitos")
    print(f"    Recall Obtido: {recall:.4f} ({recall*100:.2f}%)")
    
    print(f"\n[*] Trade-offs:")
    print(f"    Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"    F1-Score:  {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    
    # Recomendação baseada nos resultados obtidos
    print(f"\n[*] Interpretação dos Resultados:")
    
    if recall >= 0.85:
        print(f"    [OK] EXCELENTE deteccao de defeitos ({recall*100:.1f}%)")
        print(f"    [OK] O modelo esta capturando a grande maioria dos defeitos reais")
        if precision >= 0.70:
            print(f"    [OK] Boa precisao mantida ({precision*100:.1f}%)")
            recommendation = "Modelo EXCELENTE para uso em producao. Alta deteccao de defeitos com boa precisao."
        else:
            print(f"    [!] Precision baixa ({precision*100:.1f}%) - muitas pecas boas serao rejeitadas")
            recommendation = "Modelo BOM para deteccao, mas com muitos falsos positivos. Considere se o custo de rejeicao e aceitavel."
    elif recall >= 0.75:
        print(f"    [OK] BOM deteccao de defeitos ({recall*100:.1f}%)")
        print(f"    [OK] O modelo esta capturando a maioria dos defeitos reais")
        if precision >= 0.70:
            recommendation = "Modelo BOM para uso em producao. Boa deteccao com precisao aceitavel."
        else:
            recommendation = "Modelo ACEITAVEL. Boa deteccao mas com trade-off em precisao."
    elif recall >= 0.65:
        print(f"    [!] REGULAR deteccao de defeitos ({recall*100:.1f}%)")
        print(f"    [!] Alguns defeitos podem passar sem deteccao")
        recommendation = "Modelo REGULAR. Considere ajustar hiperparametros ou coletar mais dados para melhorar Recall."
    else:
        print(f"    [X] BAIXA deteccao de defeitos ({recall*100:.1f}%)")
        print(f"    [X] Muitos defeitos podem passar sem deteccao")
        recommendation = "Modelo precisa de MELHORIAS. Recall muito baixo para uso em producao."
    
    print(f"\n[*] Recomendação:")
    print(f"    {recommendation}")
    
    # Salvar relatório
    report = {
        'metrics': metrics,
        'thresholds': thresholds,
        'recall': float(recall),
        'precision': float(precision),
        'f1_micro': float(f1_micro),
        'f1_macro': float(f1_macro),
        'accuracy': float(accuracy),
        'false_negative_rate': float(false_negative_rate),
        'false_positive_rate': float(false_positive_rate),
        'recommendation': recommendation,
        'recall_status': recall_status,
        'precision_status': precision_status
    }
    
    with open('model_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[OK] Relatório completo salvo em 'model_analysis_report.json'")
    
    return report


if __name__ == "__main__":
    analyze_model_results()
