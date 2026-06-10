"""
Precòmput de les features de control d'espai per a tots els partits.

Per a cada partit i cada frame amb tracking vàlid, calcula:
  - àrea de control (fracció del camp) de cada jugador, indexada per player_id
  - control del local a cada terç (3 valors, per al vector de context)
  - control del local a la zona de la pilota (1 valor auxiliar)

La velocitat es defineix de manera canònica i independent de la finestra:
    v(f) = (p(f) - p(f - STRIDE)) / (STRIDE / FPS),
de manera que el control és una funció pura de (partit, frame). El control
es calcula sobre tots els jugadors detectats al frame amb equip conegut.

L'artefacte de sortida és un .pkl per partit i per mode, amb l'estructura:
    {
      "match_id":  str,
      "mode":      "voronoi" | "dominant",
      "grid_res":  float,
      "t_react":   float,
      "frames": {
          frame:int -> {
              "area_frac":      {player_id:int -> float},
              "third_home":     np.ndarray [3]  (control local left/mid/right),
              "ball_zone_home": float (o NaN si no hi ha pilota),
          }
      }
    }

Ús:
    python -m multi_head_model.precompute_spatial_control            # tots, ambdós modes
    python -m multi_head_model.precompute_spatial_control --mode voronoi
    python -m multi_head_model.precompute_spatial_control --max-matches 1   # prova
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .constants import DATA_DIR, OUTPUT_DIR, STRIDE, FPS
from .dataset import _MatchData
from . import spatial_control as sc


DT_STEP_S = STRIDE / FPS                       # 0.3 s
OUT_SUBDIR = "spatial_control"


# ── Còmput d'un partit ──────────────────────────────────────────────────────

def _frame_players(
    match: _MatchData,
    frame: int,
    prev_players: Optional[Dict],
):
    """
    Construeix (pos, vel, teams, pid_list, ball_xy) per a un frame, amb la
    velocitat canònica respecte de `prev_players` (posicions a frame-STRIDE).
    Inclou tots els jugadors detectats amb equip conegut.
    """
    data = match.builder.get_frame_positions(frame)
    if data is None:
        return None
    players = data["players"]                  # {pid: (x, y, team_or_None)}

    pos_list: List[List[float]] = []
    vel_list: List[List[float]] = []
    team_list: List[int] = []
    pid_list: List[int] = []

    for pid, (x, y, _t) in players.items():
        pid = int(pid)
        team = match.player_team.get(pid)
        if team is None:
            continue
        if not np.all(np.isfinite([x, y])):
            continue
        team_idx = 0 if team == match.home_team_id else 1

        if prev_players is not None and pid in prev_players:
            px, py, _ = prev_players[pid]
            if np.all(np.isfinite([px, py])):
                vx, vy = (x - px) / DT_STEP_S, (y - py) / DT_STEP_S
            else:
                vx, vy = 0.0, 0.0
        else:
            vx, vy = 0.0, 0.0

        pos_list.append([float(x), float(y)])
        vel_list.append([float(vx), float(vy)])
        team_list.append(team_idx)
        pid_list.append(pid)

    if len(pos_list) < 2:
        return None

    ball = data.get("ball")
    ball_xy = (
        np.array([float(ball[0]), float(ball[1])])
        if ball is not None and np.all(np.isfinite(ball[:2]))
        else None
    )

    return (
        np.asarray(pos_list, dtype=np.float64),
        np.asarray(vel_list, dtype=np.float64),
        np.asarray(team_list, dtype=np.int64),
        pid_list,
        ball_xy,
    )


def precompute_match(
    match: _MatchData,
    modes: List[str],
    grid_res: float,
    t_react: float,
) -> Dict[str, Dict[int, dict]]:
    """
    Calcula les features de control per a tots els frames vàlids del partit,
    per a cadascun dels modes demanats. Construeix posicions i velocitats una
    sola vegada per frame i reaprofita-les per a tots els modes.
    """
    frames = sorted(match.builder.tracking_frames.keys())
    tracking = match.builder.tracking_frames
    out: Dict[str, Dict[int, dict]] = {m: {} for m in modes}

    for f in frames:
        period_f = tracking[f].period
        # Frame previ canònic: f - STRIDE, només si és del mateix període
        prev_f = f - STRIDE
        prev_players = None
        prev_ft = tracking.get(prev_f)
        if prev_ft is not None and prev_ft.period == period_f:
            prev_data = match.builder.get_frame_positions(prev_f)
            prev_players = prev_data["players"] if prev_data is not None else None

        built = _frame_players(match, f, prev_players)
        if built is None:
            continue
        pos, vel, teams, pid_list, ball_xy = built

        for mode in modes:
            feat = sc.control_features(
                pos, teams, ball_xy, vel,
                mode=mode, grid_res=grid_res, t_react=t_react,
            )
            out[mode][f] = {
                "area_frac": {
                    pid: float(a)
                    for pid, a in zip(pid_list, feat["player_area_frac"])
                },
                "third_home": feat["third_control"][0].astype(np.float32),
                "ball_zone_home": (
                    float(feat["ball_zone_control"][0])
                    if ball_xy is not None else float("nan")
                ),
            }

    return out


# ── Validacions ──────────────────────────────────────────────────────────────

def validate_artifact(frames_dict: Dict[int, dict], match_id: str) -> None:
    """Comprovacions de sanitat sobre l'artefacte d'un partit."""
    if not frames_dict:
        print(f"  [!] {match_id}: cap frame vàlid")
        return

    area_sums = []
    all_areas = []
    third_vals = []
    n_nan_third = 0
    n_nan_ballzone = 0

    for rec in frames_dict.values():
        areas = np.array(list(rec["area_frac"].values()))
        area_sums.append(areas.sum())
        all_areas.extend(areas.tolist())
        third_vals.append(rec["third_home"])
        if not np.all(np.isfinite(rec["third_home"])):
            n_nan_third += 1
        if not np.isfinite(rec["ball_zone_home"]):
            n_nan_ballzone += 1

    area_sums = np.array(area_sums)
    all_areas = np.array(all_areas)
    third_vals = np.stack(third_vals)

    print(f"  [validació {match_id}]  frames={len(frames_dict)}")
    print(f"    conservació àrea (suma area_frac per frame): "
          f"min={area_sums.min():.4f} max={area_sums.max():.4f} "
          f"mitjana={area_sums.mean():.4f}  (esperat ≈ 1.0)")
    print(f"    àrea per jugador (m²): mitjana={all_areas.mean()*sc.PITCH_AREA:.1f} "
          f"min={all_areas.min()*sc.PITCH_AREA:.1f} "
          f"max={all_areas.max()*sc.PITCH_AREA:.1f}")
    print(f"    third_home dins [0,1]: "
          f"{'OK' if (third_vals.min() >= -1e-6 and third_vals.max() <= 1+1e-6) else 'FORA DE RANG'}"
          f"  (min={third_vals.min():.3f} max={third_vals.max():.3f})")
    if not np.all(np.isfinite(all_areas)):
        raise ValueError(f"{match_id}: area_frac conté NaN o inf.")
    if n_nan_third:
        raise ValueError(
            f"{match_id}: {n_nan_third} frames amb third_home no finit."
        )
    if n_nan_ballzone:
        print(f"    [i] {n_nan_ballzone} frames sense pilota (ball_zone=NaN)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--out-dir", type=str,
                        default=str(Path(OUTPUT_DIR) / OUT_SUBDIR))
    parser.add_argument("--mode", type=str, default="both",
                        choices=["voronoi", "dominant", "both"])
    parser.add_argument("--grid-res", type=float, default=1.0,
                        help="Cel·les per metre (1.0 → precisió ~1 m²).")
    parser.add_argument("--t-react", type=float, default=sc.DEFAULT_REACT_TIME)
    parser.add_argument("--max-matches", type=int, default=None,
                        help="Limita el nombre de partits (per a proves).")
    args = parser.parse_args()

    modes = ["voronoi", "dominant"] if args.mode == "both" else [args.mode]

    match_dirs = sorted(p for p in Path(args.data_dir).iterdir() if p.is_dir())
    if args.max_matches is not None:
        match_dirs = match_dirs[:args.max_matches]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] {len(match_dirs)} partits, modes={modes}, grid_res={args.grid_res}")
    print(f"[i] Sortida: {out_dir}")

    for i, md in enumerate(match_dirs, 1):
        match_id = md.name
        print(f"\n[{i}/{len(match_dirs)}] {match_id}: carregant...")
        t0 = time.perf_counter()
        match = _MatchData(md)
        n_frames = len(match.builder.tracking_frames)
        print(f"    {n_frames} frames de tracking. Calculant...")

        tm = time.perf_counter()
        per_mode = precompute_match(match, modes, args.grid_res, args.t_react)
        dt = time.perf_counter() - tm
        print(f"    còmput ({'+'.join(modes)}) en {dt:.1f}s")

        for mode in modes:
            frames_dict = per_mode[mode]
            artifact = {
                "match_id": match_id,
                "mode": mode,
                "grid_res": args.grid_res,
                "t_react": args.t_react,
                "frames": frames_dict,
            }
            out_path = out_dir / f"{match_id}_{mode}.pkl"
            with out_path.open("wb") as fh:
                pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"    [{mode}] {len(frames_dict)} frames → {out_path.name}")
            validate_artifact(frames_dict, match_id)

        print(f"    (partit fet en {time.perf_counter() - t0:.1f}s)")

    print(f"\n[OK] Precòmput finalitzat a {out_dir}")


if __name__ == "__main__":
    main()
