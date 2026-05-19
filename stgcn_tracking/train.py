"""
Training script for STGCN football tracking.

Usage:
    python -m stgcn_tracking.train                     # variant estàtica
    python -m stgcn_tracking.train --dynamic           # variant signada/dinàmica
    python -m stgcn_tracking.train --epochs 100 --batch-size 64
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stgcn_tracking import constants as C
from stgcn_tracking.Dataset import TrackingDataset, TrackingDatasetV2
from stgcn_tracking.model import build_model, build_model_dynamic


# ── Hyperparàmetres per defecte ─────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE_ES  = 10        # early stopping
PATIENCE_LR  = 5         # scheduler ReduceLROnPlateau
LR_FACTOR    = 0.5
GRAD_CLIP    = 5.0


# ── Mètriques ────────────────────────────────────────────────────────────────

def mae_meters(pred: torch.Tensor, target: torch.Tensor,
               mean_std: np.ndarray) -> float:
    """
    MAE en metres (desnormalitzat).
    pred, target: [B, N_PRED, N_NODES, N_FEAT]  (normalitzats)
    """
    mean = torch.tensor(mean_std[0], dtype=torch.float32, device=pred.device)
    std  = torch.tensor(mean_std[1], dtype=torch.float32, device=pred.device)
    pred_m   = pred   * std + mean
    target_m = target * std + mean
    return (pred_m - target_m).abs().mean().item()


def final_step_err_meters(pred: torch.Tensor, target: torch.Tensor,
                          mean_std: np.ndarray) -> float:
    """
    Distància euclidiana mitjana en metres al **darrer step predit**
    (= horitzó més llunyà). Útil com a mètrica robusta del cas pitjor.
    """
    mean = torch.tensor(mean_std[0], dtype=torch.float32, device=pred.device)
    std  = torch.tensor(mean_std[1], dtype=torch.float32, device=pred.device)
    pred_last   = pred[:, -1]   * std + mean   # [B, N, F]
    target_last = target[:, -1] * std + mean
    diff = (pred_last - target_last)[..., :2]   # només (x, y)
    return diff.pow(2).sum(-1).sqrt().mean().item()


# ── Bucle d'una època ───────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, mean_std,
              optimizer=None, grad_clip=GRAD_CLIP):
    is_train = optimizer is not None
    model.train(is_train)

    sum_loss     = 0.0
    sum_mae      = 0.0
    sum_final_m  = 0.0
    n_batches    = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss = criterion(pred, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        sum_loss    += loss.item()
        sum_mae     += mae_meters(pred.detach(), y, mean_std)
        sum_final_m += final_step_err_meters(pred.detach(), y, mean_std)
        n_batches   += 1

    return {
        "loss":         sum_loss    / max(n_batches, 1),
        "mae_m":        sum_mae     / max(n_batches, 1),
        "final_err_m":  sum_final_m / max(n_batches, 1),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device: {device}")
    print(f"[i] Variant: {'dynamic (signed)' if args.dynamic else 'static'}")

    # Directori de sortida específic per variant
    variant = "dynamic" if args.dynamic else "static"
    out_dir = Path(args.out_dir) / f"stgcn_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] Sortida: {out_dir}")

    # ── Dades ──────────────────────────────────────────────────────────────
    DatasetCls = TrackingDatasetV2 if args.dynamic else TrackingDataset
    train_loader = DataLoader(
        DatasetCls("seq_train.npy"),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        DatasetCls("seq_val.npy"),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        DatasetCls("seq_test.npy"),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    mean_std = np.load(os.path.join(C.OUTPUT_DIR, "mean_std.npy"))   # [2, 2]
    print(f"[i] Mostres: train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    # ── Model + opt + scheduler + loss ─────────────────────────────────────
    model = (
        build_model_dynamic(device, c_in=6) if args.dynamic
        else build_model(device)
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=PATIENCE_LR,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[i] Paràmetres entrenables: {n_params:,}")

    # ── CSV logging ────────────────────────────────────────────────────────
    csv_path = out_dir / "train_log.csv"
    log_keys = [
        "epoch", "lr",
        "train_loss", "train_mae_m", "train_final_err_m",
        "val_loss",   "val_mae_m",   "val_final_err_m",
        "elapsed_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(log_keys)

    # ── Loop principal ─────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    no_improve       = 0
    best_path        = out_dir / "best_model.pt"
    last_path        = out_dir / "last_model.pt"

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, criterion, device,
                                mean_std, optimizer, args.grad_clip)
        val_stats   = run_epoch(model, val_loader, criterion, device,
                                mean_std, optimizer=None)
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_stats["loss"])

        # Stdout
        print(
            f"[{ep:03d}/{args.epochs:03d}] {elapsed:5.1f}s  lr={lr:.1e}  "
            f"train_loss={train_stats['loss']:.5f}  "
            f"val_loss={val_stats['loss']:.5f}  "
            f"val_MAE={val_stats['mae_m']:.2f}m  "
            f"val_final_err={val_stats['final_err_m']:.2f}m"
        )

        # CSV
        row = {
            "epoch": ep, "lr": lr,
            "train_loss":         train_stats["loss"],
            "train_mae_m":        train_stats["mae_m"],
            "train_final_err_m":  train_stats["final_err_m"],
            "val_loss":           val_stats["loss"],
            "val_mae_m":          val_stats["mae_m"],
            "val_final_err_m":    val_stats["final_err_m"],
            "elapsed_s":          elapsed,
        }
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k, "") for k in log_keys])

        # Checkpoint últim (estat complet per reprendre)
        torch.save({
            "epoch":           ep,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss":        val_stats["loss"],
            "variant":         variant,
        }, last_path)

        # Best + early stopping
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            no_improve    = 0
            torch.save(model.state_dict(), best_path)
            print(f"      ★ nou best (val_loss={best_val_loss:.5f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE_ES:
                print(f"[!] Early stopping a època {ep} "
                      f"({PATIENCE_ES} èpoques sense millora)")
                break

    # ── Test final amb el millor model ─────────────────────────────────────
    print(f"\n[i] Carregant best model des de {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_stats = run_epoch(model, test_loader, criterion, device, mean_std)

    print(f"\n[Test final]")
    for k, v in test_stats.items():
        print(f"  {k:14s} = {v:.5f}")

    # Resultats de test al CSV
    with (out_dir / "test_results.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(test_stats.keys()))
        w.writerow([f"{v:.6f}" for v in test_stats.values()])

    print(f"\n[OK] Entrenament finalitzat. Millor val_loss = {best_val_loss:.5f}")
    print(f"[OK] Sortides a: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic",      action="store_true",
                        help="Entrena la variant signada/dinàmica.")
    parser.add_argument("--out-dir",      type=str,   default=C.OUTPUT_DIR,
                        help="Directori arrel de sortides (s'hi crea un "
                             "subdirectori stgcn_static/ o stgcn_dynamic/).")
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip",    type=float, default=GRAD_CLIP)
    parser.add_argument("--num-workers",  type=int,   default=2)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
