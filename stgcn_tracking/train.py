"""
Training script for STGCN football tracking.

Usage:
    python -m stgcn_tracking.train
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stgcn_tracking import constants as C
from stgcn_tracking.Dataset import TrackingDataset, TrackingDatasetV2
from stgcn_tracking.model import build_model, build_model_dynamic


# ── Hyperparàmetres ──────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 10          # early stopping
CHECKPOINT   = os.path.join(C.OUTPUT_DIR, 'best_model.pt')


# ── Mètriques ────────────────────────────────────────────────────────────────
def mae_meters(pred: torch.Tensor, target: torch.Tensor, mean_std: np.ndarray) -> float:
    """
    MAE en metres (desnormalitzat).
    pred, target: [B, N_PRED, N_NODES, N_FEAT]  (normalitzats)
    """
    mean = torch.tensor(mean_std[0], dtype=torch.float32, device=pred.device)
    std  = torch.tensor(mean_std[1], dtype=torch.float32, device=pred.device)
    pred_m   = pred   * std + mean
    target_m = target * std + mean
    return (pred_m - target_m).abs().mean().item()


# ── Bucle d'entrenament ──────────────────────────────────────────────────────
def train(dynamic = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dades
    if dynamic:
        train_loader = DataLoader(
            TrackingDatasetV2('seq_train.npy'),
            batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(
            TrackingDatasetV2('seq_val.npy'),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            TrackingDataset('seq_train.npy'),
            batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(
            TrackingDataset('seq_val.npy'),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
        )

    mean_std = np.load(os.path.join(C.OUTPUT_DIR, 'mean_std.npy'))  # [2, 2]

    # Model, loss, optimitzador
    model     = build_model_dynamic(device, 6) if dynamic else build_model(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paràmetres entrenables: {n_params:,}")

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)           # [B, N_PRED, N_NODES, N_FEAT]
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validació ──
        model.eval()
        val_loss = 0.0
        val_mae  = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                val_mae  += mae_meters(pred, y, mean_std)
        val_loss /= len(val_loader)
        val_mae  /= len(val_loader)

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  "
            f"val_MAE={val_mae:.2f}m  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"({elapsed:.1f}s)"
        )

        # ── Early stopping + checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Millor model guardat ({CHECKPOINT})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping a l'epoch {epoch}.")
                break

    print(f"\nEntrenament finalitzat. Millor val_loss: {best_val_loss:.5f}")


if __name__ == '__main__':
    train()
