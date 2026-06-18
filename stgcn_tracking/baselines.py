"""
Baselines senzills per a la prediccio STGCN amb horitzo fix.

Exemples:
    python -m stgcn_tracking.baselines --split test
    python -m stgcn_tracking.baselines --split val --velocity-window 5
    python -m stgcn_tracking.baselines --split all --out-csv stgcn_data/baseline_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from stgcn_tracking import constants as C


SPLIT_TO_FILE = {
    "train": "seq_train.npy",
    "val": "seq_val.npy",
    "test": "seq_test.npy",
}


def _load_split(data_dir: str, split: str) -> np.ndarray:
    path = os.path.join(data_dir, SPLIT_TO_FILE[split])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No s'ha trobat {path}. Genera primer les sequencies STGCN."
        )
    return np.load(path)


def _load_mean_std(data_dir: str) -> np.ndarray:
    path = os.path.join(data_dir, "mean_std.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No s'ha trobat {path}. Cal el fitxer mean_std.npy del preprocessament."
        )
    return np.load(path)


def last_position_baseline(seq: np.ndarray) -> np.ndarray:
    """Repeteix l'ultima posicio observada durant tot l'horitzo futur."""
    last_pos = seq[:, C.N_HIS - 1, :, :]                         # [B, N, 2]
    return np.repeat(last_pos[:, None, :, :], C.N_PRED, axis=1)  # [B, K, N, 2]


def constant_velocity_baseline(seq: np.ndarray, velocity_window: int = 5) -> np.ndarray:
    """
    Extrapola la velocitat mitjana recent.

    velocity_window indica quantes diferencies consecutives finals s'utilitzen
    per estimar la velocitat. Amb dades a 10 fps, velocity_window=5 equival a
    mig segon d'historial recent.
    """
    if velocity_window < 1:
        raise ValueError("velocity_window ha de ser com a minim 1.")

    history = seq[:, :C.N_HIS, :, :]                             # [B, H, N, 2]
    n_diffs = min(velocity_window, C.N_HIS - 1)
    diffs = np.diff(history[:, -(n_diffs + 1):, :, :], axis=1)    # [B, n_diffs, N, 2]
    velocity = diffs.mean(axis=1)                                # [B, N, 2]
    last_pos = history[:, -1, :, :]                              # [B, N, 2]
    steps = np.arange(1, C.N_PRED + 1, dtype=seq.dtype).reshape(1, C.N_PRED, 1, 1)
    return last_pos[:, None, :, :] + steps * velocity[:, None, :, :]


def _denorm(arr: np.ndarray, mean_std: np.ndarray) -> np.ndarray:
    mean = mean_std[0].reshape(1, 1, 1, C.N_FEAT)
    std = mean_std[1].reshape(1, 1, 1, C.N_FEAT)
    return arr * std + mean


def compute_metrics(pred_norm: np.ndarray, target_norm: np.ndarray, mean_std: np.ndarray) -> Dict[str, float]:
    """
    Retorna metriques comparables amb l'entrenament STGCN.

    mae_coord_m coincideix amb el MAE desnormalitzat que el codi d'entrenament
    anomena sovint ADE. ade_l2_m es proporciona com a ADE euclidia estandard.
    """
    diff_norm = pred_norm - target_norm
    mse_norm = float(np.mean(diff_norm ** 2))

    pred_m = _denorm(pred_norm, mean_std)
    target_m = _denorm(target_norm, mean_std)
    diff_m = pred_m - target_m

    mae_coord_m = float(np.mean(np.abs(diff_m)))
    ade_l2_m = float(np.mean(np.linalg.norm(diff_m, axis=-1)))
    final_err_m = float(np.mean(np.linalg.norm(diff_m[:, -1, :, :], axis=-1)))

    return {
        "loss_mse_norm": mse_norm,
        "mae_coord_m": mae_coord_m,
        "ade_l2_m": ade_l2_m,
        "final_err_m": final_err_m,
    }


def evaluate_split(data_dir: str, split: str, velocity_window: int) -> List[Dict[str, object]]:
    seq = _load_split(data_dir, split)
    mean_std = _load_mean_std(data_dir)
    target = seq[:, C.N_HIS:C.N_HIS + C.N_PRED, :, :]

    baselines = {
        "last_position": last_position_baseline(seq),
        f"constant_velocity_w{velocity_window}": constant_velocity_baseline(seq, velocity_window),
    }

    rows: List[Dict[str, object]] = []
    for name, pred in baselines.items():
        metrics = compute_metrics(pred, target, mean_std)
        rows.append({
            "split": split,
            "baseline": name,
            "n_sequences": int(seq.shape[0]),
            **metrics,
        })
    return rows


def _format_row(row: Dict[str, object]) -> str:
    return (
        f"{row['split']:>5s}  {row['baseline']:<22s}  "
        f"n={row['n_sequences']:>6d}  "
        f"loss={row['loss_mse_norm']:.5f}  "
        f"MAE_coord={row['mae_coord_m']:.3f}m  "
        f"ADE_L2={row['ade_l2_m']:.3f}m  "
        f"FDE={row['final_err_m']:.3f}m"
    )


def _write_csv(rows: Iterable[Dict[str, object]], out_csv: str) -> None:
    rows = list(rows)
    if not rows:
        return
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=C.OUTPUT_DIR)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="test",
        help="Conjunt sobre el qual s'avaluen els baselines.",
    )
    parser.add_argument(
        "--velocity-window",
        type=int,
        default=5,
        help="Nombre de diferencies finals usades per estimar la velocitat.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Opcionalment desa els resultats en CSV.",
    )
    args = parser.parse_args()

    splits = list(SPLIT_TO_FILE) if args.split == "all" else [args.split]
    rows: List[Dict[str, object]] = []
    for split in splits:
        rows.extend(evaluate_split(args.data_dir, split, args.velocity_window))

    for row in rows:
        print(_format_row(row))

    if args.out_csv:
        _write_csv(rows, args.out_csv)
        print(f"\n[OK] Resultats desats a {args.out_csv}")


if __name__ == "__main__":
    main()
