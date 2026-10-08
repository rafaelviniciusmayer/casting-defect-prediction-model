"""
Leak-free neural-network training.

The original train_single_model (train_model.py) receives a 'validation' set that is used for early stopping and for
the LR scheduler's sibling criterion. In unified_cv_pipeline.fit_final_model that set was the TEST set, and in the CV
folds it was the fold-validation set. Here the early-stopping set is an internal, stratified 10% hold-out taken from
the training data passed to the function, so neither the test set nor the CV validation fold influences training.
Seeds are fixed (torch, numpy) so that results are reproducible.

Usage:  import nn_noleak; nn_noleak.install()   (before calling unified_cv_pipeline functions)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import train_model
from train_model import DefectPredictionNN

SEED = int(os.environ.get("NN_SEED", "42"))


def train_single_model_noleak(X_train, y_train, X_val, y_val, pos_weights, input_size, num_defects, verbose=True):
    """Same architecture/hyper-parameters as train_model.train_single_model; X_val/y_val are IGNORED."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    has_defect = (np.asarray(y_train).sum(axis=1) > 0).astype(int)
    idx_tr, idx_va = train_test_split(np.arange(len(X_train)), test_size=0.10, random_state=SEED, stratify=has_defect)
    Xt, yt = torch.FloatTensor(X_train[idx_tr]), torch.FloatTensor(y_train[idx_tr])
    Xv, yv = torch.FloatTensor(X_train[idx_va]), torch.FloatTensor(y_train[idx_va])
    g = torch.Generator(); g.manual_seed(SEED)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True, generator=g)
    model = DefectPredictionNN(input_size, num_defects)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(pos_weights))
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7)
    model.train(); best, patience = float("inf"), 0
    for epoch in range(200):
        tot = n = 0
        for bx, by in loader:
            optimizer.zero_grad(); loss = criterion(model(bx), by)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            tot += loss.item(); n += 1
        if n:
            scheduler.step(tot / n)
            model.eval()
            with torch.no_grad():
                vl = criterion(model(Xv), yv).item()
            model.train()
            if vl < best: best, patience = vl, 0
            else: patience += 1
            if patience >= 25:
                break
    model.eval()
    return model, {"stopped_epoch": epoch + 1, "internal_val_loss": best}


def install():
    import unified_cv_pipeline as ucv
    ucv.train_single_model = train_single_model_noleak
    train_model.train_single_model = train_single_model_noleak
