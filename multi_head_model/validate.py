"""
Validació qualitativa del millor model multi-head sobre el conjunt de test.

Genera 5 visualitzacions a `multi_head_output/figures/`:
  1. confusion_matrix_event.png — matriu de confusió de l'event head
  2. time_calibration.png       — scatter Δt predit vs real (mediana log-normal)
  3. pause_distribution.png     — sigmoid(pause_logit) per classe real
  4. trajectory_examples.png    — predicció vs target en el camp, 6 mostres
  5. trajectory_err_by_step.png — error euclidià mitjà com a funció del step k

Ús:
    python -m multi_head_model.validate
    python -m multi_head_model.validate --split val
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import (
    DATA_DIR, BATCH_SIZE, T_PRED_MAX, STRIDE, FPS,
    N_PHASE_CLASSES, PHASE_TYPES, N_PLAYERS,
    VAL_T_FRACTIONS,
)
from .dataset import PhaseDataset
from .model   import MultiHeadModel
from .train   import split_match_dirs, to_device


# ── Config visual ───────────────────────────────────────────────────────────

PITCH_LEN, PITCH_WID = 105.0, 68.0
HOME_COLOR, AWAY_COLOR, BALL_COLOR = "#1976D2", "#D32F2F", "#FFFFFF"
PRED_COLOR, TARGET_COLOR = "#FF9800", "#FFFFFF"
BG, FG = "#0d1117", "white"


def _style(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#cfd8dc", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#37474f")


def _draw_pitch(ax):
    ax.set_facecolor("#2e8b57")
    ax.set_xlim(-PITCH_LEN/2 - 3, PITCH_LEN/2 + 3)
    ax.set_ylim(-PITCH_WID/2 - 3, PITCH_WID/2 + 3)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    hl, hw = PITCH_LEN/2, PITCH_WID/2
    ax.plot([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw], "w-", lw=1)
    ax.plot([0, 0], [-hw, hw], "w-", lw=1)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="white", lw=1))
    for s in (-1, 1):
        ax.plot([s*hl, s*(hl-16.5), s*(hl-16.5), s*hl],
                [-20.16, -20.16, 20.16, 20.16], "w-", lw=1)


# ── Recol·lecció de prediccions ─────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()
    chunks: Dict[str, List[np.ndarray]] = {
        "event_logits": [], "phase_target": [],
        "time_mix_logits": [], "time_mu": [], "time_log_sigma": [],
        "delta_proper": [],
        "pause_logit": [], "is_long_pause": [],
        "poss_logit": [], "possession_change": [],
        "traj_pred": [], "target_traj": [], "target_mask": [],
        "current_pos": [], "team_idx": [],
    }
    for batch in loader:
        batch_d = to_device(batch, device)
        preds = model(batch_d)
        for k in ("event_logits", "time_mix_logits",
                  "time_mu", "time_log_sigma",
                  "pause_logit", "poss_logit", "traj_pred"):
            chunks[k].append(preds[k].cpu().numpy())
        for k in ("phase_target", "delta_proper", "is_long_pause",
                  "possession_change",
                  "target_traj", "target_mask"):
            chunks[k].append(batch[k].cpu().numpy())

        # current_pos al frame de predicció (per a les visualitzacions)
        nn_ = batch["node_numeric"].numpy()                    # [B, T, N, 8]
        fm = batch["frame_mask"].numpy()                       # [B, T]
        B, T = fm.shape
        idx_t = (fm.astype(np.int64) * np.arange(T)).argmax(axis=1)
        cp = np.stack([nn_[b, idx_t[b], :, :2] for b in range(B)], axis=0)
        chunks["current_pos"].append(cp)
        team = np.stack([nn_[b, idx_t[b], :, 7] for b in range(B)], axis=0)
        chunks["team_idx"].append(team)

    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


# ── 1. Matriu de confusió per event ─────────────────────────────────────────

def plot_confusion_event(data, out_path: Path):
    y_true = data["phase_target"]
    y_pred = data["event_logits"].argmax(axis=1)
    K = N_PHASE_CLASSES
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            txt = f"{cm_norm[i,j]:.2f}\n({cm[i,j]})"
            color = "white" if cm_norm[i, j] < 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=7)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(PHASE_TYPES, rotation=45, ha="right",
                       color=FG, fontsize=8)
    ax.set_yticklabels(PHASE_TYPES, color=FG, fontsize=8)
    ax.set_xlabel("predicció", color=FG, fontsize=10)
    ax.set_ylabel("classe real", color=FG, fontsize=10)
    acc = (y_true == y_pred).mean()
    ax.set_title(
        f"Confusion matrix — Event head   ·   accuracy = {acc*100:.1f}%",
        color=FG, fontsize=11, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="proporció per fila")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
    print(f"[OK] {out_path}")


# ── 2. Calibració del Time head ─────────────────────────────────────────────

def _mixture_mean_np(mix_logits, mu, log_sigma):
    """Implementació NumPy de mixture_lognormal_mean."""
    # mix_logits, mu, log_sigma: [N, K]
    pi = np.exp(mix_logits - mix_logits.max(axis=-1, keepdims=True))
    pi = pi / pi.sum(axis=-1, keepdims=True)
    sigma2 = np.exp(2.0 * log_sigma)
    mean_per_k = np.exp(mu + 0.5 * sigma2)
    return (pi * mean_per_k).sum(axis=-1)


def plot_time_calibration(data, out_path: Path):
    valid = data["is_long_pause"] == 0
    real = data["delta_proper"][valid]
    # Predicció puntual = mitjana de la mixture
    pred = _mixture_mean_np(
        data["time_mix_logits"][valid],
        data["time_mu"][valid],
        data["time_log_sigma"][valid],
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)

    # Scatter
    ax = axs[0]
    _style(ax)
    ax.scatter(real, pred, s=12, c="#42a5f5", alpha=0.5, edgecolor="none")
    mx = max(real.max(), pred.max())
    ax.plot([0, mx], [0, mx], color="#ffeb3b", lw=1, ls="--",
            label="predicció perfecta")
    ax.set_xlabel("Δt real (s)", color=FG)
    ax.set_ylabel("Δt predit = exp(μ) (s)", color=FG)
    ax.set_title(f"Calibració temporal  ·  N={valid.sum()} mostres",
                 color=FG, fontweight="bold")
    ax.legend(facecolor="#1a1a1a", labelcolor=FG, fontsize=8)
    ax.set_xlim(0, mx*1.05); ax.set_ylim(0, mx*1.05)

    # Histogrames superposats
    ax = axs[1]
    _style(ax)
    bins = np.linspace(0, max(real.max(), pred.max()), 40)
    ax.hist(real, bins=bins, alpha=0.55, color="#42a5f5", label="real")
    ax.hist(pred, bins=bins, alpha=0.55, color="#ef5350", label="predit")
    ax.set_xlabel("Δt (s)", color=FG); ax.set_ylabel("freqüència", color=FG)
    ax.set_title("Distribució de Δt", color=FG, fontweight="bold")
    ax.legend(facecolor="#1a1a1a", labelcolor=FG, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
    print(f"[OK] {out_path}")


# ── 3. Distribució del pause logit ──────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def plot_pause_distribution(data, out_path: Path):
    probs = sigmoid(data["pause_logit"])
    is_pause = data["is_long_pause"].astype(bool)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG); _style(ax)
    bins = np.linspace(0, 1, 30)
    ax.hist(probs[~is_pause], bins=bins, color="#42a5f5", alpha=0.6,
            label=f"no pause real (N={(~is_pause).sum()})")
    ax.hist(probs[is_pause],  bins=bins, color="#ef5350", alpha=0.6,
            label=f"long pause real (N={is_pause.sum()})")
    ax.axvline(0.5, color="#ffeb3b", ls="--", lw=1, label="llindar 0.5")
    ax.set_xlabel("P(long_pause) predita", color=FG)
    ax.set_ylabel("freqüència (log)", color=FG)
    ax.set_yscale("log")
    acc = ((probs > 0.5) == is_pause).mean()
    pred_pos = (probs > 0.5).sum()
    real_pos = is_pause.sum()
    tp = ((probs > 0.5) & is_pause).sum()
    prec = tp / max(pred_pos, 1)
    rec  = tp / max(real_pos, 1)
    ax.set_title(
        f"Pause head  ·  acc={acc*100:.1f}%  prec={prec*100:.1f}%  "
        f"rec={rec*100:.1f}%  (N={len(probs)})",
        color=FG, fontweight="bold",
    )
    ax.legend(facecolor="#1a1a1a", labelcolor=FG, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
    print(f"[OK] {out_path}")


# ── 3b. Distribució del PossessionChange logit ──────────────────────────────

def plot_possession_distribution(data, out_path: Path):
    valid = (data["is_long_pause"] == 0) & (data["possession_change"] >= 0)
    probs = sigmoid(data["poss_logit"][valid])
    is_change = data["possession_change"][valid].astype(bool)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG); _style(ax)
    bins = np.linspace(0, 1, 30)
    ax.hist(probs[~is_change], bins=bins, color="#42a5f5", alpha=0.6,
            label=f"manteniment real (N={(~is_change).sum()})")
    ax.hist(probs[is_change],  bins=bins, color="#ef5350", alpha=0.6,
            label=f"canvi de possessió real (N={is_change.sum()})")
    ax.axvline(0.5, color="#ffeb3b", ls="--", lw=1, label="llindar 0.5")
    ax.set_xlabel("P(canvi de possessió) predita", color=FG)
    ax.set_ylabel("freqüència", color=FG)
    acc = ((probs > 0.5) == is_change).mean()
    pred_pos = (probs > 0.5).sum()
    real_pos = is_change.sum()
    tp = ((probs > 0.5) & is_change).sum()
    prec = tp / max(pred_pos, 1)
    rec  = tp / max(real_pos, 1)
    ax.set_title(
        f"Possession-change head  ·  acc={acc*100:.1f}%  "
        f"prec={prec*100:.1f}%  rec={rec*100:.1f}%  (N={len(probs)})",
        color=FG, fontweight="bold",
    )
    ax.legend(facecolor="#1a1a1a", labelcolor=FG, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
    print(f"[OK] {out_path}")


# ── 4. Trajectòries al camp ─────────────────────────────────────────────────

def plot_trajectories(data, out_path: Path, n_samples: int = 6, seed: int = 0):
    valid_idx = np.where(
        (data["is_long_pause"] == 0)
        & (data["target_mask"].sum(axis=1) >= 5)        # almenys 5 steps reals
    )[0]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_idx, size=min(n_samples, len(valid_idx)),
                        replace=False)

    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 9))
    fig.patch.set_facecolor(BG)

    for ax, idx in zip(axs.flat, chosen):
        _draw_pitch(ax)
        cp = data["current_pos"][idx]                  # [N, 2]
        team = data["team_idx"][idx]                   # [N]
        target = data["target_traj"][idx]              # [K, N, 2]
        pred = data["traj_pred"][idx]                  # [K, N, 2]
        mask = data["target_mask"][idx].astype(bool)   # [K]
        n_pred = int(mask.sum())

        # Posicions inicials (frame t)
        BALL = N_PLAYERS
        for n in range(N_PLAYERS):
            col = HOME_COLOR if team[n] < 0.5 else AWAY_COLOR
            ax.plot(cp[n, 0], cp[n, 1], "o", color=col, ms=6,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=4)
        ax.plot(cp[BALL, 0], cp[BALL, 1], "o", color=BALL_COLOR, ms=7,
                markeredgecolor="black", markeredgewidth=0.7, zorder=5)

        # Trajectòries: target (línia contínua blanca)
        for n in range(N_PLAYERS + 1):
            if n == BALL:
                col_t = "#FFFFFF"; col_p = "#FFC107"; lw_t = 1.6; lw_p = 1.4
            else:
                col_t = HOME_COLOR if team[n] < 0.5 else AWAY_COLOR
                col_p = "#FF9800"; lw_t = 0.9; lw_p = 0.8
            xs = np.concatenate([[cp[n, 0]], target[:n_pred, n, 0]])
            ys = np.concatenate([[cp[n, 1]], target[:n_pred, n, 1]])
            ax.plot(xs, ys, "-", color=col_t, lw=lw_t, alpha=0.55, zorder=2)
            xp = np.concatenate([[cp[n, 0]], pred[:n_pred, n, 0]])
            yp = np.concatenate([[cp[n, 1]], pred[:n_pred, n, 1]])
            ax.plot(xp, yp, "--", color=col_p, lw=lw_p, alpha=0.8, zorder=3)

        # Final state (cercles buits)
        for n in range(N_PLAYERS):
            col = HOME_COLOR if team[n] < 0.5 else AWAY_COLOR
            ax.plot(target[n_pred-1, n, 0], target[n_pred-1, n, 1],
                    "s", color=col, ms=5, alpha=0.4, zorder=3)
            ax.plot(pred[n_pred-1, n, 0], pred[n_pred-1, n, 1],
                    "x", color="#FF9800", ms=6, mew=1.5, zorder=3)

        dt_real = data["delta_proper"][idx]
        dt_pred = float(_mixture_mean_np(
            data["time_mix_logits"][idx:idx+1],
            data["time_mu"][idx:idx+1],
            data["time_log_sigma"][idx:idx+1],
        )[0])
        ev_real = PHASE_TYPES[int(data["phase_target"][idx])]
        ev_pred = PHASE_TYPES[int(data["event_logits"][idx].argmax())]
        # Error mig sobre els steps vàlids
        err = np.sqrt(((pred[:n_pred] - target[:n_pred])**2).sum(-1)).mean()
        ax.set_title(
            f"idx={idx}  ·  Δt: real {dt_real:.1f}s / pred {dt_pred:.1f}s\n"
            f"event: real '{ev_real}' / pred '{ev_pred}'  ·  err: {err:.2f} m",
            color=FG, fontsize=9,
        )

    # Llegenda comuna
    legend_handles = [
        mpatches.Patch(color=HOME_COLOR,   label="home"),
        mpatches.Patch(color=AWAY_COLOR,   label="away"),
        mpatches.Patch(color=BALL_COLOR,   label="ball (target)"),
        mpatches.Patch(color="#FF9800",    label="predicció"),
        plt.Line2D([], [], color="white", lw=1.5, label="target"),
        plt.Line2D([], [], color="#FF9800", lw=1.5, ls="--", label="predit"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               facecolor="#1a1a1a", labelcolor=FG, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Trajectòries: predicció (--) vs realitat (—)",
                 color=FG, fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
    print(f"[OK] {out_path}")


# ── 5. Error per step ───────────────────────────────────────────────────────

def plot_err_by_step(data, out_path: Path):
    valid = data["is_long_pause"] == 0
    pred = data["traj_pred"][valid]              # [N_val, K, N, 2]
    target = data["target_traj"][valid]
    mask = data["target_mask"][valid].astype(np.float32)   # [N_val, K]

    err = np.sqrt(((pred - target)**2).sum(-1)).mean(-1)   # [N_val, K]
    mask_sum = mask.sum(axis=0)                            # [K], #mostres valides per step
    err_per_step = (err * mask).sum(axis=0) / np.maximum(mask_sum, 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG); _style(ax)
    t_axis = np.arange(1, T_PRED_MAX + 1) * (STRIDE / FPS)   # segons
    ax.plot(t_axis, err_per_step, color="#42a5f5", lw=2)
    ax.fill_between(t_axis, 0, err_per_step, color="#42a5f5", alpha=0.2)
    # Eix superior amb % mostres vàlides
    ax2 = ax.twinx()
    ax2.plot(t_axis, mask_sum / mask_sum.max(), color="#ef5350", lw=1, ls=":")
    ax2.set_ylabel("fracció mostres vàlides", color="#ef5350")
    ax2.tick_params(colors="#ef5350", labelsize=8)
    ax2.set_ylim(0, 1.05)

    ax.set_xlabel("temps des de t (s)", color=FG)
    ax.set_ylabel("error euclidià mitjà (m)", color=FG)
    ax.set_title("Error per horitzó temporal — trajectory head",
                 color=FG, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
    print(f"[OK] {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    type=str, default=DATA_DIR)
    parser.add_argument("--out-dir",     type=str, default="multi_head_output")
    parser.add_argument("--split",       type=str, default="test",
                        choices=["test", "val"])
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--device",      type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[i] Device: {device}")

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Splits idèntics a train.py
    train_dirs, val_dirs, test_dirs = split_match_dirs(args.data_dir, seed=args.seed)
    eval_dirs = test_dirs if args.split == "test" else val_dirs
    print(f"[i] Split: {args.split} ({[d.name for d in eval_dirs]})")

    ds = PhaseDataset(eval_dirs, random_t=False, val_t_fractions=VAL_T_FRACTIONS)
    ds.warm_cache()
    print(f"[i] N mostres = {len(ds)} (fractions={VAL_T_FRACTIONS})")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")

    # Model + best weights
    model = MultiHeadModel().to(device)
    best_path = out_dir / "best_model.pt"
    print(f"[i] Carregant pesos de {best_path}")
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    print(f"[i] Acumulant prediccions sobre {len(ds)} mostres...")
    data = collect_predictions(model, loader, device)
    print(f"[i] Fet. Generant figures a {fig_dir}/")

    plot_confusion_event(data,           fig_dir / "confusion_matrix_event.png")
    plot_time_calibration(data,          fig_dir / "time_calibration.png")
    plot_pause_distribution(data,        fig_dir / "pause_distribution.png")
    plot_possession_distribution(data,   fig_dir / "possession_distribution.png")
    plot_trajectories(data,              fig_dir / "trajectory_examples.png")
    plot_err_by_step(data,               fig_dir / "trajectory_err_by_step.png")


if __name__ == "__main__":
    main()
