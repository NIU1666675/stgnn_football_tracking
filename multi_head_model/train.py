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
    VAL_T_FRACTIONS,
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
    parser.add_argument("--samples-per-phase", type=int, default=3,
                        help="Nombre de mostres per fase a train (augmentació "
                             "estocàstica explícita). Cada còpia rebrà un t "
                             "aleatori independent. No afecta val/test.")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Desactiva el càlcul automàtic de pesos de "
                             "classe (per a baseline sense pesos).")
    parser.add_argument("--class-weight-min-count", type=int, default=5,
                        help="Mínim 'fictici' de mostres per classe en el "
                             "càlcul de class_weights (per evitar pesos "
                             "infinits a classes rares).")
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
    # Train: random_t (augmentació estocàstica). Si samples_per_phase > 1,
    # cada fase es replica N vegades amb t aleatoris independents.
    # Val / Test: multipoint val_t_fractions per cobrir horitzons diversos.
    train_ds = PhaseDataset(
        train_dirs,
        random_t=True,
        samples_per_phase=args.samples_per_phase,
        cache_size=args.cache_size,
    )
    val_ds = PhaseDataset(
        val_dirs,
        random_t=False,
        val_t_fractions=VAL_T_FRACTIONS,
        cache_size=args.cache_size,
    )
    test_ds = PhaseDataset(
        test_dirs,
        random_t=False,
        val_t_fractions=VAL_T_FRACTIONS,
        cache_size=args.cache_size,
    )
    print(f"[i] Mostres: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"[i] Train: samples_per_phase={args.samples_per_phase} → {len(train_ds)//args.samples_per_phase} fases × {args.samples_per_phase}")
    print(f"[i] Val/test t_fractions: {VAL_T_FRACTIONS}")

    # Pre-carrega tots els partits al cache. Així el primer batch no es queda
    # bloquejat carregant JSONL durant minuts sense feedback.
    print(f"[i] Pre-carregant partits a memòria...")
    t0 = time.time()
    train_ds.warm_cache()
    val_ds.warm_cache()
    test_ds.warm_cache()
    print(f"[i] Cache preparada en {time.time() - t0:.1f}s")

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

    # ── Càlcul de pesos de classe sobre el train ───────────────────────────
    event_cw, pause_pw, poss_pw = None, None, None
    if not args.no_class_weights:
        print(f"[i] Calculant pesos de classe sobre el train_ds...")
        t0 = time.time()
        stats = train_ds.compute_class_stats(verbose=True)
        print(f"[i] Estadístiques computades en {time.time()-t0:.1f}s")

        # PauseHead: pos_weight = N_neg / N_pos
        pause_pw = float(stats["pause"][0]) / max(int(stats["pause"][1]), 1)

        # PossessionChangeHead: pos_weight = N_neg / N_pos
        poss_pw = float(stats["poss"][0]) / max(int(stats["poss"][1]), 1)

        # EventHead: class_weights = N / (K · n_c), amb clip mínim per estabilitat
        ev = stats["event"].astype(np.float64)
        ev_clipped = np.maximum(ev, args.class_weight_min_count)
        K = len(ev)
        N = ev.sum()
        event_cw = torch.tensor(
            N / (K * ev_clipped), dtype=torch.float32, device=device,
        )
        print(f"[i] Pesos finals:")
        print(f"    pause_pos_weight   = {pause_pw:.2f}")
        print(f"    poss_pos_weight    = {poss_pw:.2f}")
        print(f"    event_class_weights= {[round(float(w), 2) for w in event_cw]}")
    else:
        print(f"[i] --no-class-weights actiu: cap pes aplicat (baseline pur)")

    criterion = MultiHeadLoss(
        event_class_weights=event_cw,
        pause_pos_weight=pause_pw,
        poss_pos_weight=poss_pw,
    ).to(device)

    # ── CSV logging ────────────────────────────────────────────────────────
    csv_path = out_dir / "train_log.csv"
    log_keys = [
        "epoch", "lr",
        "train_total", "train_event", "train_time", "train_pause",
        "train_poss", "train_traj",
        "train_event_acc", "train_pause_acc", "train_poss_acc",
        "train_time_mae_s", "train_traj_err_m", "train_final_err_m",
        "val_total", "val_event", "val_time", "val_pause",
        "val_poss", "val_traj",
        "val_event_acc", "val_pause_acc", "val_poss_acc",
        "val_time_mae_s", "val_traj_err_m", "val_final_err_m",
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
            f"poss_acc={val_stats['poss_acc']:.3f}  "
            f"time_mae={val_stats['time_mae_s']:.2f}s  "
            f"traj_err={val_stats['traj_err_m']:.2f}m  "
            f"final_err={val_stats['final_err_m']:.2f}m"
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
