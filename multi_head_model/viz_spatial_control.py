"""
Visualització del control d'espai sobre frames reals del dataset.

Per a una mostra del conjunt, agafa el frame de l'instant de predicció i
dibuixa, costat a costat, la tessel·lació de Voronoi (nivell 0) i la regió
dominant amb velocitat (nivell 1), sobre un camp de mplsoccer. Serveix per
validar visualment que el control calculat per `spatial_control.py` té sentit.

Ús:
    python -m multi_head_model.viz_spatial_control --split val --sample 0
    python -m multi_head_model.viz_spatial_control --split val --sample 5 --grid-res 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mplsoccer import Pitch

from .constants import DATA_DIR, VAL_T_FRACTIONS
from .dataset import PhaseDataset
from .train import split_match_dirs
from . import spatial_control as sc


HOME_COLOR = "#1f77b4"
AWAY_COLOR = "#d62728"


def _last_valid_frame(node_numeric: np.ndarray, frame_mask: np.ndarray) -> int:
    """Índex de l'últim frame vàlid (= instant de predicció t)."""
    valid = np.where(frame_mask)[0]
    return int(valid[-1]) if len(valid) > 0 else 0


def _draw_control(ax, pitch, ctrl, pos, vel, teams, ball_xy, title,
                  draw_arrows=False):
    """Dibuixa una tessel·lació de control sobre un eix amb el camp."""
    pitch.draw(ax=ax)

    ny, nx = ctrl["grid_shape"]
    team_of_cell = teams[ctrl["assignment"]].reshape(ny, nx).astype(float)

    # Mapa de color: local (blau) / visitant (vermell), semitransparent
    cmap = ListedColormap([HOME_COLOR, AWAY_COLOR])
    ax.imshow(
        team_of_cell,
        extent=[-sc.HALF_L, sc.HALF_L, -sc.HALF_W, sc.HALF_W],
        origin="lower", cmap=cmap, alpha=0.35, vmin=0, vmax=1,
        aspect="auto", zorder=0.5,
    )

    # Jugadors
    for t, color in ((0, HOME_COLOR), (1, AWAY_COLOR)):
        m = teams == t
        ax.scatter(pos[m, 0], pos[m, 1], s=90, color=color,
                   edgecolors="white", linewidths=1.2, zorder=4)

    # Vectors de velocitat (nivell 1)
    if draw_arrows:
        for i in range(len(pos)):
            ax.annotate(
                "", xy=(pos[i, 0] + vel[i, 0] * sc.DEFAULT_REACT_TIME,
                        pos[i, 1] + vel[i, 1] * sc.DEFAULT_REACT_TIME),
                xytext=(pos[i, 0], pos[i, 1]),
                arrowprops=dict(arrowstyle="->", color="black",
                                lw=1.0, alpha=0.7), zorder=5,
            )

    # Pilota
    if ball_xy is not None:
        ax.scatter(ball_xy[0], ball_xy[1], s=70, color="white",
                   edgecolors="black", linewidths=1.2, marker="o", zorder=6)

    ax.set_title(title, fontsize=12, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--sample", type=int, default=0,
                        help="Índex de la mostra dins del split.")
    parser.add_argument("--grid-res", type=float, default=2.0,
                        help="Cel·les per metre de la graella de control.")
    parser.add_argument("--t-react", type=float, default=sc.DEFAULT_REACT_TIME,
                        help="Temps de reacció del nivell 1 (s).")
    parser.add_argument("--out", type=str, default=None,
                        help="Ruta del PNG de sortida.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dirs, val_dirs, test_dirs = split_match_dirs(
        args.data_dir, seed=args.seed,
    )
    split_dirs = {"train": train_dirs, "val": val_dirs,
                  "test": test_dirs}[args.split]

    ds = PhaseDataset(split_dirs, random_t=False,
                      val_t_fractions=VAL_T_FRACTIONS)
    ds.warm_cache()
    print(f"[i] {args.split}: {len(ds)} mostres")

    idx = args.sample % len(ds)
    item = ds[idx]
    node_numeric = item["node_numeric"].numpy()
    frame_mask = item["frame_mask"].numpy()

    t = _last_valid_frame(node_numeric, frame_mask)
    pos, vel, teams, ball_xy = sc.extract_frame_players(node_numeric[t])
    print(f"[i] Mostra {idx}, frame de predicció t={t}: "
          f"{len(pos)} jugadors detectats")

    pitch = Pitch(pitch_type="skillcorner", pitch_length=sc.PITCH_LENGTH,
                  pitch_width=sc.PITCH_WIDTH, line_color="black",
                  pitch_color="white")

    fig, axs = plt.subplots(1, 2, figsize=(20, 7))

    ctrl_v = sc.compute_control(pos, teams, vel, mode="voronoi",
                                grid_res=args.grid_res)
    _draw_control(axs[0], pitch, ctrl_v, pos, vel, teams, ball_xy,
                  "Nivell 0 — Voronoi euclidià")

    ctrl_d = sc.compute_control(pos, teams, vel, mode="dominant",
                                grid_res=args.grid_res, t_react=args.t_react)
    _draw_control(axs[1], pitch, ctrl_d, pos, vel, teams, ball_xy,
                  f"Nivell 1 — Regió dominant (t_react={args.t_react}s)",
                  draw_arrows=True)

    # Resum de control d'equip al peu
    feat_v = sc.control_features(pos, teams, ball_xy, vel, mode="voronoi",
                                 grid_res=args.grid_res)
    feat_d = sc.control_features(pos, teams, ball_xy, vel, mode="dominant",
                                 grid_res=args.grid_res, t_react=args.t_react)
    fig.suptitle(
        f"Control d'espai — mostra {idx} ({args.split})    "
        f"| Voronoi local/visitant: "
        f"{feat_v['team_control'][0]:.2f}/{feat_v['team_control'][1]:.2f}    "
        f"| Dominant: "
        f"{feat_d['team_control'][0]:.2f}/{feat_d['team_control'][1]:.2f}",
        fontsize=13, y=1.02,
    )

    out = args.out or f"multi_head_model/multi_head_output/control_sample{idx}_{args.split}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
