"""
Model Results Analysis Script
==========================================

Analyzes training results focusing on MAXIMIZING DEFECT DETECTION.
Goal: Identify the largest possible number of defective parts.

Recall, F1-Score, and Precision values are OUTPUTS of the optimization process,
not input criteria.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

def load_artifacts_safely():
    """Loads model artifacts safely."""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except (AttributeError, ModuleNotFoundError) as e:
        print(f"[WARNING] Error loading full model: {e}")
        print("[INFO] Attempting to load metrics and thresholds only...")
        
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
    """Analyzes trained model results focusing on maximum defect detection."""
    
    print("="*70)
    print("RESULTS ANALYSIS - FOCUS ON MAXIMIZING DEFECT DETECTION")
    print("="*70)
    
    # Load model
    try:
        artifacts = load_artifacts_safely()
        print("\n[OK] Model data loaded")
    except Exception as e:
        print(f"\n[ERROR] Error loading model: {e}")
        print("[INFO] Run train_model.py first to generate the model.")
        return
    
    # Try loading metrics from JSON first
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
            print("\n[OK] Metrics loaded from model_metrics.json")
    except FileNotFoundError:
        # Try extracting from artifacts
        metrics = artifacts.get('metrics', {})
        thresholds = artifacts.get('optimal_thresholds', {})
        feature_names = artifacts.get('process_vars', [])
        defect_names = artifacts.get('defect_cols', [])
        
        if not metrics:
            print("\n[WARNING] Metrics not found.")
            print("[INFO] Run train_model.py to generate complete metrics.")
            return
    
    # =============================================================================
    # 1. OBTAINED METRICS (OUTPUTS OF THE OPTIMIZATION PROCESS)
    # =============================================================================
    print("\n" + "="*70)
    print("1. OBTAINED METRICS (Optimization Results)")
    print("="*70)
    
    recall = metrics.get('recall_micro', 0)
    precision = metrics.get('precision_micro', 0)
    f1_micro = metrics.get('f1_micro', 0)
    f1_macro = metrics.get('f1_macro', 0)
    accuracy = metrics.get('accuracy', 0)
    
    print(f"\nTest Set Metrics:")
    print(f"  Recall (Sensitivity):       {recall:.4f} ({recall*100:.2f}%)")
    print(f"  Precision:                  {precision:.4f} ({precision*100:.2f}%)")
    print(f"  F1-Score (Micro):           {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    print(f"  F1-Score (Macro):           {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  Accuracy:                   {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # =============================================================================
    # 2. DEFECT DETECTION ANALYSIS (MAIN FOCUS)
    # =============================================================================
    print("\n" + "="*70)
    print("2. DEFECT DETECTION ANALYSIS")
    print("="*70)
    
    # False negative rate (defects that slip through - CRITICAL)
    false_negative_rate = 1 - recall
    true_positive_rate = recall
    
    print(f"\n[*] Real Defect Detection:")
    print(f"    Detection Rate (Recall):     {recall:.4f} ({recall*100:.2f}%)")
    print(f"    False Negative Rate:         {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
    
    # Estimate production impact
    print(f"\n[*] Estimated Production Impact (per 1000 parts):")
    print(f"    Defects Detected:              ~{recall*1000:.0f} parts")
    print(f"    Defects That Slip Through (FN): ~{false_negative_rate*1000:.0f} parts")
    
    # Recall interpretation
    if recall >= 0.90:
        recall_status = "EXCELLENT - Model detects more than 90% of defects"
        recall_color = "[OK]"
    elif recall >= 0.80:
        recall_status = "VERY GOOD - Model detects more than 80% of defects"
        recall_color = "[OK]"
    elif recall >= 0.70:
        recall_status = "GOOD - Model detects more than 70% of defects"
        recall_color = "[OK]"
    elif recall >= 0.60:
        recall_status = "FAIR - Model detects more than 60% of defects"
        recall_color = "[!]"
    else:
        recall_status = "LOW - Model detects less than 60% of defects"
        recall_color = "[X]"
    
    print(f"\n    {recall_color} {recall_status}")
    
    # =============================================================================
    # 3. PRECISION ANALYSIS (TRADE-OFF)
    # =============================================================================
    print("\n" + "="*70)
    print("3. PRECISION ANALYSIS (Trade-off with Recall)")
    print("="*70)
    
    false_positive_rate = 1 - precision
    
    print(f"\n[*] Prediction Precision:")
    print(f"    Precision:                     {precision:.4f} ({precision*100:.2f}%)")
    print(f"    False Positive Rate:           {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
    
    print(f"\n[*] Estimated Production Impact (per 1000 parts):")
    print(f"    Good Parts Rejected (FP):      ~{false_positive_rate*1000:.0f} parts")
    
    # Precision interpretation
    if precision >= 0.80:
        precision_status = "HIGH - Few false positives"
        precision_color = "[OK]"
    elif precision >= 0.70:
        precision_status = "MODERATE - Some false positives"
        precision_color = "[!]"
    elif precision >= 0.60:
        precision_status = "LOW - Many false positives"
        precision_color = "[!]"
    else:
        precision_status = "VERY LOW - Many false positives"
        precision_color = "[X]"
    
    print(f"\n    {precision_color} {precision_status}")
    
    # =============================================================================
    # 4. BALANCE (F1-SCORE)
    # =============================================================================
    print("\n" + "="*70)
    print("4. BALANCE BETWEEN RECALL AND PRECISION (F1-Score)")
    print("="*70)
    
    print(f"\n[*] F1-Score (Harmonic Mean):")
    print(f"    F1-Score (Micro):             {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    print(f"    F1-Score (Macro):             {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    
    # F1-Score interpretation
    if f1_micro >= 0.80:
        f1_status = "EXCELLENT balance"
        f1_color = "[OK]"
    elif f1_micro >= 0.70:
        f1_status = "GOOD balance"
        f1_color = "[OK]"
    elif f1_micro >= 0.60:
        f1_status = "FAIR balance"
        f1_color = "[!]"
    else:
        f1_status = "LOW balance"
        f1_color = "[X]"
    
    print(f"\n    {f1_color} {f1_status}")
    
    # =============================================================================
    # 5. OPTIMIZED THRESHOLDS ANALYSIS
    # =============================================================================
    print("\n" + "="*70)
    print("5. OPTIMIZED THRESHOLDS ANALYSIS")
    print("="*70)
    
    if thresholds:
        threshold_values = list(thresholds.values())
        avg_threshold = np.mean(threshold_values)
        min_threshold = np.min(threshold_values)
        max_threshold = np.max(threshold_values)
        median_threshold = np.median(threshold_values)
        
        print(f"\n[*] Threshold Statistics:")
        print(f"    Mean:    {avg_threshold:.3f}")
        print(f"    Median:  {median_threshold:.3f}")
        print(f"    Minimum: {min_threshold:.3f}")
        print(f"    Maximum: {max_threshold:.3f}")
        
        # Threshold distribution
        low_thresholds = sum(1 for t in threshold_values if t < 0.3)
        medium_thresholds = sum(1 for t in threshold_values if 0.3 <= t < 0.6)
        high_thresholds = sum(1 for t in threshold_values if t >= 0.6)
        
        print(f"\n[*] Distribution:")
        print(f"    Low Thresholds (< 0.3):       {low_thresholds} defects (high sensitivity)")
        print(f"    Medium Thresholds (0.3-0.6):  {medium_thresholds} defects")
        print(f"    High Thresholds (>= 0.6):     {high_thresholds} defects (low sensitivity)")
        
        if avg_threshold < 0.4:
            threshold_strategy = "Strategy: Low thresholds to maximize Recall"
        elif avg_threshold < 0.6:
            threshold_strategy = "Strategy: Moderate thresholds for balance"
        else:
            threshold_strategy = "Strategy: High thresholds (may be limiting Recall)"
        
        print(f"\n    {threshold_strategy}")
        
        # Show top 10 defects with lowest thresholds (most sensitive)
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])
        print(f"\n[*] Top 10 Defects with Highest Sensitivity (lowest thresholds):")
        for i, (defect, threshold) in enumerate(sorted_thresholds[:10], 1):
            print(f"    {i:2d}. {defect:<35}: {threshold:.3f}")
    
    # =============================================================================
    # 6. SUMMARY AND FINAL INTERPRETATION
    # =============================================================================
    print("\n" + "="*70)
    print("6. SUMMARY AND FINAL INTERPRETATION")
    print("="*70)
    
    print(f"\n[*] Primary Goal: Maximize Defect Detection")
    print(f"    Recall Obtained: {recall:.4f} ({recall*100:.2f}%)")
    
    print(f"\n[*] Trade-offs:")
    print(f"    Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"    F1-Score:  {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    
    # Recommendation based on obtained results
    print(f"\n[*] Results Interpretation:")
    
    if recall >= 0.85:
        print(f"    [OK] EXCELLENT defect detection ({recall*100:.1f}%)")
        print(f"    [OK] The model is capturing the vast majority of real defects")
        if precision >= 0.70:
            print(f"    [OK] Good precision maintained ({precision*100:.1f}%)")
            recommendation = "EXCELLENT model for production use. High defect detection with good precision."
        else:
            print(f"    [!] Low precision ({precision*100:.1f}%) - many good parts will be rejected")
            recommendation = "GOOD model for detection, but with many false positives. Consider whether rejection cost is acceptable."
    elif recall >= 0.75:
        print(f"    [OK] GOOD defect detection ({recall*100:.1f}%)")
        print(f"    [OK] The model is capturing most real defects")
        if precision >= 0.70:
            recommendation = "GOOD model for production use. Good detection with acceptable precision."
        else:
            recommendation = "ACCEPTABLE model. Good detection but with a precision trade-off."
    elif recall >= 0.65:
        print(f"    [!] FAIR defect detection ({recall*100:.1f}%)")
        print(f"    [!] Some defects may slip through undetected")
        recommendation = "FAIR model. Consider adjusting hyperparameters or collecting more data to improve Recall."
    else:
        print(f"    [X] LOW defect detection ({recall*100:.1f}%)")
        print(f"    [X] Many defects may slip through undetected")
        recommendation = "Model needs IMPROVEMENTS. Recall too low for production use."
    
    print(f"\n[*] Recommendation:")
    print(f"    {recommendation}")
    
    # Save report
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
    
    print(f"\n[OK] Full report saved to 'model_analysis_report.json'")
    
    return report


if __name__ == "__main__":
    analyze_model_results()
