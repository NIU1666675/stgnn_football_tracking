"""
Bucle d'entrenament per al model multi-head.

Ús:
    python -m multi_head_model.train                 # entrena amb defaults
    python -m multi_head_model.train --epochs 1      # smoke test ràpid

Sortida:
    multi_head_output/
      ├── train_log.csv      # mètriques per època
      ├── best_model.pt      # pesos del millor model (val_total mínim)
      └── last_model.pt      # checkpoint complet de la darrera època
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .constants import (
    DATA_DIR,
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    GRAD_CLIP,
    PATIENCE_ES,
    PATIENCE_LR,
    LR_FACTOR,
    N_TRAIN_MATCHES,
    N_VAL_MATCHES,
    N_TEST_MATCHES,
)
from .dataset import PhaseDataset
from .model   import MultiHeadModel
from .losses  import MultiHeadLoss, compute_metrics


# ── Reproductibilitat ───────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Partició de partits ─────────────────────────────────────────────────────

def split_match_dirs(
    data_dir: str,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Selecció aleatòria amb seed dels partits per train / val / test."""
    all_dirs = sorted(p for p in Path(data_dir).iterdir() if p.is_dir())
    rng = random.Random(seed)
    rng.shuffle(all_dirs)
    n_tr, n_va, n_te = N_TRAIN_MATCHES, N_VAL_MATCHES, N_TEST_MATCHES
    train = all_dirs[:n_tr]
    val   = all_dirs[n_tr:n_tr + n_va]
    test  = all_dirs[n_tr + n_va:n_tr + n_va + n_te]
    return train, val, test


# ── Helpers ─────────────────────────────────────────────────────────────────

def to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def aggregate(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = records[0].keys()
    return {k: sum(r[k] for r in records) / len(records) for k in keys}


# ── Loop d'una època ────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = GRAD_CLIP,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    loss_recs:   List[Dict[str, float]] = []
    metric_recs: List[Dict[str, float]] = []

    for batch in loader:
        batch = to_device(batch, device)

        with torch.set_grad_enabled(is_train):
            preds   = model(batch)
            losses  = criterion(preds, batch)
            metrics = compute_metrics(preds, batch)

        if is_train:
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        loss_recs.append({k: v.item() for k, v in losses.items()})
        metric_recs.append(metrics)

    return {**aggregate(loss_recs), **aggregate(metric_recs)}


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",     type=str,   default=DATA_DIR)
    parser.add_argument("--out-dir",      type=str,   default="multi_head_output")
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip",    type=float, default=GRAD_CLIP)
    parser.add_argument("--num-workers",  type=int,   default=0)
    parser.add_argument("--cache-size",   type=int,   default=None,
                        help="Mida del cache LRU de partits per dataset. "
                             "Per defecte = nombre de partits del split (tots "
                             "a RAM, ~200 MB cadascun).")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       type=str,   default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[i] Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Splits ─────────────────────────────────────────────────────────────
    train_dirs, val_dirs, test_dirs = split_match_dirs(args.data_dir, seed=args.seed)
    print(f"[i] Train ({len(train_dirs)}): {[d.name for d in train_dirs]}")
    print(f"[i] Val   ({len(val_dirs)}): {[d.name for d in val_dirs]}")
    print(f"[i] Test  ({len(test_dirs)}): {[d.name for d in test_dirs]}")

    # ── Datasets / Loaders ────────────────────────────────────────────────
    train_ds = PhaseDataset(train_dirs, random_t=True,  cache_size=args.cache_size)
    val_ds   = PhaseDataset(val_dirs,   random_t=False, cache_size=args.cache_size)
    test_ds  = PhaseDataset(test_dirs,  random_t=False, cache_size=args.cache_size)
    print(f"[i] Mostres: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    loader_kw = dict(
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    # ── Model / Optimitzador / Scheduler / Loss ───────────────────────────
    model = MultiHeadModel().to(device)
    print(f"[i] Paràmetres: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=PATIENCE_LR,
    )
    criterion = MultiHeadLoss()

    # ── CSV logging ────────────────────────────────────────────────────────
    csv_path = out_dir / "train_log.csv"
    log_keys = [
        "epoch", "lr",
        "train_total", "train_event", "train_time", "train_pause", "train_traj",
        "train_event_acc", "train_pause_acc", "train_time_mae_s",
        "train_traj_err_m", "train_final_err_m",
        "val_total", "val_event", "val_time", "val_pause", "val_traj",
        "val_event_acc", "val_pause_acc", "val_time_mae_s",
        "val_traj_err_m", "val_final_err_m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(log_keys)

    # ── Loop principal ─────────────────────────────────────────────────────
    best_val   = float("inf")
    no_improve = 0
    best_path  = out_dir / "best_model.pt"
    last_path  = out_dir / "last_model.pt"

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer, args.grad_clip)
        val_stats   = run_epoch(model, val_loader,   criterion, device, None,      args.grad_clip)
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_stats["total"])

        # Stdout
        print(
            f"[{ep:03d}/{args.epochs:03d}] {elapsed:5.1f}s  lr={lr:.1e}  "
            f"train={train_stats['total']:.3f}  val={val_stats['total']:.3f}  "
            f"event_acc={val_stats['event_acc']:.3f}  "
            f"time_mae={val_stats['time_mae_s']:.2f}s  "
            f"state_err={val_stats['state_err_m']:.2f}m"
        )

        # CSV
        row = {"epoch": ep, "lr": lr}
        for k, v in train_stats.items(): row[f"train_{k}"] = v
        for k, v in val_stats.items():   row[f"val_{k}"]   = v
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k, "") for k in log_keys])

        # Checkpoints
        torch.save({
            "epoch":           ep,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss":        val_stats["total"],
        }, last_path)

        if val_stats["total"] < best_val:
            best_val   = val_stats["total"]
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"      ★ nou best (val_total={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE_ES:
                print(f"[!] Early stopping a època {ep} ({PATIENCE_ES} èpoques sense millora)")
                break

    # ── Avaluació final amb el millor model ───────────────────────────────
    print(f"\n[i] Carregant best model des de {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_stats = run_epoch(model, test_loader, criterion, device, None, args.grad_clip)

    print(f"\n[Test final]")
    for k, v in test_stats.items():
        print(f"  {k:18s} = {v:.4f}")

    # També al CSV
    with (out_dir / "test_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(test_stats.keys()))
        w.writerow([f"{v:.6f}" for v in test_stats.values()])


if __name__ == "__main__":
    main()
