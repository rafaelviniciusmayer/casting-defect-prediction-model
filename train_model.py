"""
ML-Based Workflow for Defect Prediction in Casting Processes
============================================================

This script implements the complete ML workflow for defect prediction in casting:
- 3.3. Data Preparation: Data preparation and cleaning
- 3.5. Feature Engineering: Feature engineering
- 3.6. Model Training and Selection: Model training and selection

Objective: Maximize prediction accuracy to reduce industrial risks.
Each defective part that passes through represents a risk to the customer.

Execute: python train_model.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# SMOTE for class balancing
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARNING] imbalanced-learn not installed. Run: pip install imbalanced-learn")


# =============================================================================
# 3.3. DATA PREPARATION - Data Preparation and Cleaning
# =============================================================================

def load_and_prepare_data():
    """
    Step 3.3: Load and prepare raw data.
    
    Output: Clean and structured dataset
    """
    print("\n" + "="*70)
    print("3.3. DATA PREPARATION - Data Preparation and Cleaning")
    print("="*70)
    
    print("\n[*] Loading dataset...")
    
    # Try to load dataset with features first
    try:
        df = pd.read_csv('aluminum_diecasting_dataset_with_features.csv')
        print("    [OK] Dataset with features loaded")
        
        # Identify defects
        defect_prefixes = [
            'blisters', 'surface', 'die', 'flow', 'cold', 'heat', 'ejector',
            'low', 'density', 'incomplete', 'flash', 'warpage', 'shrinkage',
            'volumetric', 'dimensional', 'gas', 'internal', 'cracks', 'hard', 'oxide'
        ]
        defect_cols = [col for col in df.columns 
                      if any(col.startswith(prefix) for prefix in defect_prefixes)]
        
        # Features are all columns except defects and metadata
        feature_cols = [col for col in df.columns 
                       if col not in defect_cols + ['id', 'total_defects', 'has_defect']]
        
    except FileNotFoundError:
        print("    [INFO] Dataset with features not found, using original dataset")
        df = pd.read_csv('aluminum_diecasting_dataset.csv')
        
        # Original process variables
        process_vars = [
            'piston_velocity_phase1', 'metal_velocity_gate', 'fill_time',
            'phase_transition_position', 'intensification_time_phase3',
            'intensification_pressure', 'solidification_time', 'cycle_time',
            'sleeve_fill_percentage', 'sleeve_diameter', 'sleeve_length',
            'plunger_lubricant', 'plunger_sleeve_clearance', 'sleeve_temperature',
            'plunger_temperature'
        ]
        
        # Defects
        defect_cols = [col for col in df.columns 
                      if col not in process_vars + ['id', 'total_defects', 'has_defect']]
        feature_cols = process_vars
    
    # Extract features and labels
    X = df[feature_cols].values.astype(np.float32)
    y = df[defect_cols].values.astype(np.float32)
    
    # Basic data validation
    assert not np.isnan(X).any(), "Data contains NaN in features"
    assert not np.isnan(y).any(), "Data contains NaN in labels"
    assert X.shape[0] == y.shape[0], "Inconsistent number of samples"
    
    print(f"\n[*] Dataset prepared:")
    print(f"    Features: {X.shape[1]}")
    print(f"    Defects: {y.shape[1]}")
    print(f"    Samples: {X.shape[0]:,}")
    
    # Calculate weights for class balancing
    pos_weights = []
    print("\n[*] Calculating weights for class balancing:")
    for i, defect_name in enumerate(defect_cols):
        pos_count = y[:, i].sum()
        neg_count = len(y) - pos_count
        if pos_count > 0:
            weight = np.sqrt(neg_count / pos_count)
            weight = min(weight, 10.0)  # Limit maximum weight
        else:
            weight = 1.0
        pos_weights.append(weight)
        
        if i < 5:  # Show only first 5
            print(f"    {defect_name:<30}: weight={weight:.2f} (defects={int(pos_count)})")
    
    print("\n[OK] Data Preparation complete - Clean and structured dataset ready")
    
    return X, y, feature_cols, defect_cols, pos_weights


# =============================================================================
# 3.5. FEATURE ENGINEERING - Feature Engineering
# =============================================================================

def apply_feature_engineering(X, feature_names):
    """
    Step 3.5: Feature engineering (if needed).
    
    Input: Clean and structured dataset
    Output: Final dataset
    """
    print("\n" + "="*70)
    print("3.5. FEATURE ENGINEERING - Feature Engineering")
    print("="*70)
    
    print("\n[*] Applying feature engineering...")
    
    # If we already have derived features, no additional work is needed
    # Otherwise, we could add derived features here
    # For now, we keep the original or already derived features
    
    print(f"    Final features: {X.shape[1]}")
    print("\n[OK] Feature Engineering complete - Final dataset ready")
    
    return X, feature_names


# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================

class DefectPredictionNN(nn.Module):
    """Neural Network for defect prediction in aluminum die casting."""
    
    def __init__(self, input_size, num_defects):
        super(DefectPredictionNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, num_defects)
        )
    
    def forward(self, x):
        return self.network(x)


class PyTorchModelWrapper:
    """Wrapper for compatibility with sklearn and pickle."""
    
    def __init__(self, pytorch_model, scaler, thresholds=None):
        self.pytorch_model = pytorch_model
        self.scaler = scaler
        self.thresholds = thresholds
    
    def predict(self, X, defect_names=None):
        """Binary prediction using adaptive thresholds or fixed 0.5."""
        proba = self.predict_proba(X)
        
        if self.thresholds is not None and defect_names is not None:
            binary_pred = np.zeros_like(proba)
            for i, defect_name in enumerate(defect_names):
                threshold = self.thresholds.get(defect_name, 0.5)
                binary_pred[:, i] = (proba[:, i] >= threshold).astype(int)
            return binary_pred
        else:
            return (proba > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Probability prediction."""
        X_scaled = self.scaler.transform(X).astype(np.float32)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.pytorch_model.eval()
        with torch.no_grad():
            logits = self.pytorch_model(X_tensor)
            proba = torch.sigmoid(logits).numpy()
        
        return proba


# =============================================================================
# 3.6. MODEL TRAINING AND SELECTION - Model Training and Selection
# =============================================================================

def apply_smote_balancing(X_train, y_train, defect_names, verbose=True):
    """Apply SMOTE for artificial class balancing."""
    if not SMOTE_AVAILABLE:
        return X_train, y_train
    
    if verbose:
        print("\n[*] Applying SMOTE for class balancing...")
    
    synthetic_samples = []
    synthetic_labels = []
    
    for defect_idx, defect_name in enumerate(defect_names):
        y_defect = y_train[:, defect_idx]
        pos_count = int(y_defect.sum())
        
        if pos_count < 5:
            continue
        
        neg_count = len(y_defect) - pos_count
        target_ratio = 1.5
        n_target_pos = int(neg_count / target_ratio) if target_ratio > 0 else pos_count
        n_synthetic_needed = max(0, n_target_pos - pos_count)
        
        if n_synthetic_needed > 0:
            try:
                pos_mask = y_defect == 1
                X_pos = X_train[pos_mask]
                
                if len(X_pos) < 2:
                    continue
                
                y_binary = y_defect.astype(int)
                k_neighbors = min(5, len(X_pos) - 1)
                if k_neighbors < 1:
                    continue
                
                n_synthetic_needed = min(n_synthetic_needed, pos_count * 3)
                
                smote = SMOTE(
                    random_state=42 + defect_idx,
                    k_neighbors=k_neighbors,
                    sampling_strategy={1: pos_count + n_synthetic_needed}
                )
                
                X_balanced, y_balanced = smote.fit_resample(X_train, y_binary)
                original_size = len(X_train)
                synthetic_mask = np.arange(len(X_balanced)) >= original_size
                
                if synthetic_mask.sum() > 0:
                    X_synthetic = X_balanced[synthetic_mask]
                    from sklearn.neighbors import NearestNeighbors
                    knn = NearestNeighbors(n_neighbors=min(3, len(X_pos)))
                    knn.fit(X_pos)
                    _, neighbor_indices = knn.kneighbors(X_synthetic)
                    
                    for i, neighbors in enumerate(neighbor_indices):
                        neighbor_original_indices = np.where(pos_mask)[0][neighbors]
                        neighbor_labels = y_train[neighbor_original_indices]
                        synthetic_label = (neighbor_labels.mean(axis=0) > 0.5).astype(float)
                        synthetic_label[defect_idx] = 1.0
                        synthetic_samples.append(X_synthetic[i])
                        synthetic_labels.append(synthetic_label)
            
            except Exception:
                continue
    
    if len(synthetic_samples) > 0:
        X_synthetic_array = np.array(synthetic_samples).astype(np.float32)
        y_synthetic_array = np.array(synthetic_labels).astype(np.float32)
        X_train = np.vstack([X_train, X_synthetic_array]).astype(np.float32)
        y_train = np.vstack([y_train, y_synthetic_array]).astype(np.float32)
        if verbose:
            print(f"    [OK] {len(synthetic_samples)} synthetic samples added")
    
    return X_train, y_train


def train_single_model(X_train, y_train, X_val, y_val, pos_weights, input_size, num_defects, verbose=True):
    """Train a single model."""
    # Prepare data
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Create model
    model = DefectPredictionNN(input_size, num_defects)
    
    # Weighted loss
    pos_weight_tensor = torch.FloatTensor(pos_weights)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7)
    
    # Training
    model.train()
    best_val_loss = float('inf')
    patience = 0
    max_epochs = 200
    
    for epoch in range(max_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        if batch_count > 0:
            avg_train_loss = epoch_loss / batch_count
            scheduler.step(avg_train_loss)
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
            
            model.train()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
            
            if verbose and epoch % 25 == 0:
                print(f"      Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            if patience >= 25:
                break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_val_tensor)
        y_pred_proba = torch.sigmoid(logits).numpy()
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'f1_micro': f1_score(y_val, y_pred_binary, average='micro'),
        'f1_macro': f1_score(y_val, y_pred_binary, average='macro'),
        'precision_micro': precision_score(y_val, y_pred_binary, average='micro'),
        'recall_micro': recall_score(y_val, y_pred_binary, average='micro'),
    }
    
    return model, metrics


def optimize_thresholds(y_test, y_pred_proba, defect_names):
    """
    Optimize thresholds to MAXIMIZE RECALL - identify as many defective parts as possible.
    
    Primary Objective: Minimize false negatives (defects that pass through).
    Strategy: Test multiple thresholds and choose the one that maximizes Recall,
    while maintaining a reasonable balance with Precision through F1-Score.
    """
    print("\n[*] Optimizing thresholds to MAXIMIZE DEFECT DETECTION (Recall)...")
    print("    Objective: Identify as many defective parts as possible")
    
    optimal_thresholds = {}
    y_pred_binary = np.zeros_like(y_pred_proba)
    
    for i, defect_name in enumerate(defect_names):
        y_true_defect = y_test[:, i]
        
        if y_true_defect.sum() == 0:
            # If there are no defects in the test set, use a conservative threshold
            optimal_thresholds[defect_name] = 0.5
            continue
        
        best_recall = 0
        best_threshold = 0.5
        best_precision = 0
        best_f1 = 0
        
        # Strategy: Test thresholds from 0.1 to 0.9 in steps of 0.01
        # Lower thresholds = more sensitive = higher Recall
        for threshold in np.arange(0.1, 0.91, 0.01):
            y_pred_defect = (y_pred_proba[:, i] >= threshold).astype(int)
            
            if y_pred_defect.sum() > 0:
                precision = precision_score(y_true_defect, y_pred_defect, zero_division=0)
                recall = recall_score(y_true_defect, y_pred_defect, zero_division=0)
                
                # Calculate F1-Score for balance
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0
                
                # Prioritize maximum Recall, but use F1-Score as tiebreaker
                # to avoid extremely low thresholds that generate too many false positives
                if recall > best_recall:
                    # New best Recall found
                    best_recall = recall
                    best_threshold = threshold
                    best_precision = precision
                    best_f1 = f1
                elif recall == best_recall and f1 > best_f1:
                    # Same Recall, but better F1-Score (more balanced)
                    best_threshold = threshold
                    best_precision = precision
                    best_f1 = f1
        
        # If a good Recall was still not found, try even lower thresholds
        if best_recall < 0.80:
            for threshold in np.arange(0.05, 0.11, 0.01):
                y_pred_defect = (y_pred_proba[:, i] >= threshold).astype(int)
                if y_pred_defect.sum() > 0:
                    precision = precision_score(y_true_defect, y_pred_defect, zero_division=0)
                    recall = recall_score(y_true_defect, y_pred_defect, zero_division=0)
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0
                    
                    if recall > best_recall:
                        best_recall = recall
                        best_threshold = threshold
                        best_precision = precision
                        best_f1 = f1
                    elif recall == best_recall and f1 > best_f1:
                        best_threshold = threshold
                        best_precision = precision
                        best_f1 = f1
        
        optimal_thresholds[defect_name] = best_threshold
        y_pred_binary[:, i] = (y_pred_proba[:, i] >= best_threshold).astype(int)
        
        if i < 10:
            print(f"    {defect_name:<30}: threshold={best_threshold:.3f}, "
                  f"Recall={best_recall:.3f}, Precision={best_precision:.3f}, F1={best_f1:.3f}")
    
    print("    [OK] Thresholds optimized to maximize defect detection")
    return optimal_thresholds, y_pred_binary


def train_model(X, y, pos_weights, feature_names, defect_names):
    """
    Step 3.6: Train and select the best model.
    
    Input: Final dataset
    Output: Selected model
    """
    print("\n" + "="*70)
    print("3.6. MODEL TRAINING AND SELECTION - Model Training and Selection")
    print("="*70)
    
    # Initial split: development (80%) and final test (20%)
    has_defect = (y.sum(axis=1) > 0).astype(int)
    X_dev, X_test_final, y_dev, y_test_final = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=has_defect
    )
    
    print(f"\n[*] Data split:")
    print(f"    Development: {X_dev.shape[0]:,} samples (80%)")
    print(f"    Final Test: {X_test_final.shape[0]:,} samples (20%)")
    
    # Cross-Validation for model selection
    print(f"\n[*] Running 5-Fold Cross-Validation...")
    has_defect_dev = (y_dev.sum(axis=1) > 0).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(skf.split(X_dev, has_defect_dev))
    
    cv_metrics = []
    fold_models = []
    fold_scalers = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n  Fold {fold_idx + 1}/5:")
        X_train_fold = X_dev[train_idx]
        y_train_fold = y_dev[train_idx]
        X_val_fold = X_dev[val_idx]
        y_val_fold = y_dev[val_idx]
        
        # Normalization (fit on training data only)
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold).astype(np.float32)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold).astype(np.float32)
        
        # Class balancing with SMOTE
        X_train_fold_scaled, y_train_fold = apply_smote_balancing(
            X_train_fold_scaled, y_train_fold, defect_names
        )
        
        # Train model
        model_fold, metrics_fold = train_single_model(
            X_train_fold_scaled, y_train_fold,
            X_val_fold_scaled, y_val_fold,
            pos_weights, X_train_fold_scaled.shape[1], y_train_fold.shape[1]
        )
        
        cv_metrics.append(metrics_fold)
        fold_models.append(model_fold)
        fold_scalers.append(scaler_fold)
        
        print(f"    F1-Score (Micro): {metrics_fold['f1_micro']:.4f}")
        print(f"    Precision: {metrics_fold['precision_micro']:.4f}")
        print(f"    Recall: {metrics_fold['recall_micro']:.4f}")
    
    # Average Cross-Validation metrics
    print(f"\n[*] Cross-Validation results (5 folds):")
    avg_metrics = {}
    std_metrics = {}
    for key in cv_metrics[0].keys():
        values = [m[key] for m in cv_metrics]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)
        print(f"    {key}: {avg_metrics[key]:.4f} (+/- {std_metrics[key]:.4f})")
    
    # Train final model with ALL development data
    print(f"\n[*] Training final model with all development data...")
    
    scaler_final = StandardScaler()
    X_dev_scaled = scaler_final.fit_transform(X_dev).astype(np.float32)
    X_test_final_scaled = scaler_final.transform(X_test_final).astype(np.float32)
    
    # Class balancing with SMOTE
    X_dev_scaled, y_dev = apply_smote_balancing(
        X_dev_scaled, y_dev, defect_names
    )
    
    # Train final model
    model_final, _ = train_single_model(
        X_dev_scaled, y_dev,
        X_test_final_scaled, y_test_final,
        pos_weights, X_dev_scaled.shape[1], y_dev.shape[1]
    )
    
    # Evaluation on final test set
    model_final.eval()
    X_test_tensor = torch.FloatTensor(X_test_final_scaled)
    y_test_tensor = torch.FloatTensor(y_test_final)
    
    with torch.no_grad():
        logits = model_final(X_test_tensor)
        y_pred_proba = torch.sigmoid(logits).numpy()
    
    # Optimize thresholds to maximize Recall
    optimal_thresholds, y_pred_binary = optimize_thresholds(
        y_test_final, y_pred_proba, defect_names
    )
    
    # Final metrics
    metrics = {
        'f1_micro': f1_score(y_test_final, y_pred_binary, average='micro'),
        'f1_macro': f1_score(y_test_final, y_pred_binary, average='macro'),
        'precision_micro': precision_score(y_test_final, y_pred_binary, average='micro'),
        'recall_micro': recall_score(y_test_final, y_pred_binary, average='micro'),
        'accuracy': accuracy_score(y_test_final, y_pred_binary)
    }
    
    print(f"\n[*] Metrics on final test set:")
    print(f"    F1-Score (Micro): {metrics['f1_micro']:.4f}")
    print(f"    F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"    Precision: {metrics['precision_micro']:.4f}")
    print(f"    Recall: {metrics['recall_micro']:.4f}")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    
    # Generate confusion matrices
    print(f"\n[*] Generating confusion matrices...")
    Path('confusion_matrices').mkdir(exist_ok=True)
    
    confusion_matrices = {}
    for i, defect_name in enumerate(defect_names):
        y_true_defect = y_test_final[:, i]
        y_pred_defect = y_pred_binary[:, i]
        
        cm = confusion_matrix(y_true_defect, y_pred_defect)
        confusion_matrices[defect_name] = cm
        
        # Save visualization
        if cm.shape == (2, 2):
            fig, ax = plt.subplots(figsize=(8, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                         display_labels=['No Defect', 'Defect'])
            disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
            # Increase font sizes for readability
            for text in ax.texts:
                text.set_fontsize(14)
            title_name = defect_name.replace('_', ' ').title()
            ax.set_title(f'Confusion Matrix: {title_name}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            ax.tick_params(labelsize=10)
            
            tn, fp, fn, tp = cm.ravel()
            precision_defect = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_defect = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            textstr = f'Precision: {precision_defect:.3f}\nRecall: {recall_defect:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='bottom', horizontalalignment='right', bbox=props)
            
            plt.tight_layout()
            safe_name = defect_name.replace('/', '_').replace('\\', '_')
            plt.savefig(f'confusion_matrices/confusion_matrix_{safe_name}.png', 
                       dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
    
    print(f"    [OK] {len(confusion_matrices)} confusion matrices saved")
    
    print("\n[OK] Model Training and Selection complete - Selected model ready")
    
    return {
        'model': model_final,
        'scaler': scaler_final,
        'metrics': metrics,
        'precision_thresholds': optimal_thresholds,  # Kept for compatibility
        'optimal_thresholds': optimal_thresholds,
        'confusion_matrices': confusion_matrices,
        'pos_weights': pos_weights
    }


def save_model(model_info, feature_names, defect_names):
    """Save trained model."""
    print("\n[*] Saving model...")
    
    Path('models').mkdir(exist_ok=True)
    
    # Create wrapper
    wrapper = PyTorchModelWrapper(
        model_info['model'], 
        model_info['scaler'], 
        thresholds=model_info['precision_thresholds']
    )
    
    artifacts = {
        'model': wrapper,
        'scaler': model_info['scaler'],
        'process_vars': feature_names,
        'defect_cols': defect_names,
        'metrics': model_info['metrics'],
        'pos_weights': model_info['pos_weights'],
        'optimal_thresholds': model_info.get('optimal_thresholds', model_info.get('precision_thresholds', {})),
        'model_name': 'pytorch_defect_prediction',
        'model_type': 'pytorch_neural_network',
        'version': '1.0'
    }
    
    with open('models/pytorch_stable_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    # Save thresholds separately
    with open('optimal_thresholds.pkl', 'wb') as f:
        pickle.dump(model_info['precision_thresholds'], f)
    
    # Save metrics in JSON for easy analysis
    import json
    metrics_json = {
        'f1_micro': float(model_info['metrics']['f1_micro']),
        'f1_macro': float(model_info['metrics']['f1_macro']),
        'precision_micro': float(model_info['metrics']['precision_micro']),
        'recall_micro': float(model_info['metrics']['recall_micro']),
        'accuracy': float(model_info['metrics']['accuracy']),
        'thresholds': {k: float(v) for k, v in model_info['precision_thresholds'].items()},
        'feature_count': len(feature_names),
        'defect_count': len(defect_names)
    }
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print("    [OK] Model saved:")
    print("        - models/pytorch_stable_model.pkl")
    print("        - models/best_model.pkl")
    print("        - optimal_thresholds.pkl")
    print("        - model_metrics.json")


def main():
    """Main function - Runs the complete ML workflow."""
    print("="*70)
    print("ML-BASED WORKFLOW FOR DEFECT PREDICTION IN CASTING PROCESSES")
    print("="*70)
    
    try:
        # 3.3. Data Preparation
        X, y, feature_names, defect_names, pos_weights = load_and_prepare_data()
        
        # 3.5. Feature Engineering
        X, feature_names = apply_feature_engineering(X, feature_names)
        
        # 3.6. Model Training and Selection
        model_info = train_model(X, y, pos_weights, feature_names, defect_names)
        
        # Save model
        save_model(model_info, feature_names, defect_names)
        
        print("\n" + "="*70)
        print("[COMPLETED] ML WORKFLOW EXECUTED SUCCESSFULLY!")
        print("="*70)
        print(f"\nFinal Summary:")
        print(f"  F1-Score (Micro): {model_info['metrics']['f1_micro']:.4f}")
        print(f"  Precision: {model_info['metrics']['precision_micro']:.4f}")
        print(f"  Recall: {model_info['metrics']['recall_micro']:.4f}")
        print(f"  Accuracy: {model_info['metrics']['accuracy']:.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
