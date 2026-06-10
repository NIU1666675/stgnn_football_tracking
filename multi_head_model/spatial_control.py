"""
Control d'espai (pitch control) com a feature engineering geomètric per al
model multi-head.

Implementa dos nivells modulars i activables:

  - 'voronoi'  (nivell 0): tessel·lació de Voronoi euclidiana. Cada punt del
                camp pertany al jugador més proper en distància. L'àrea de la
                cel·la de cada jugador és l'espai que controla.

  - 'dominant' (nivell 1): regió dominant amb velocitat (Taki & Hasegawa).
                Cada punt s'assigna al jugador que hi pot arribar primer. Amb
                un model de reacció simple ---el jugador es desplaça amb la
                velocitat actual durant un temps de reacció t_react i després
                esprinta a velocitat màxima--- i assumint velocitat màxima
                uniforme, l'assignació "qui arriba primer" és equivalent a un
                Voronoi sobre les posicions desplaçades pel moment
                x_eff = x + v · t_react. La velocitat estira la regió de cada
                jugador en la direcció del seu desplaçament.

El càlcul es fa discretitzant el camp en una graella: cada cel·la s'assigna
al generador (jugador o posició desplaçada) més proper. Aquest enfocament
resol el retallat als límits del camp de manera natural i unifica els dos
nivells en una sola operació. Treballa en metres amb la convenció de
coordenades de SkillCorner (origen al centre, x al llarg, y a l'ample).

La pilota NO és un generador de la tessel·lació: és un punt de referència per
a les features de control de la seva zona.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial.distance import cdist


# ── Dimensions del camp (SkillCorner, metres, origen al centre) ─────────────

PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0
HALF_L = PITCH_LENGTH / 2.0          # 52.5
HALF_W = PITCH_WIDTH / 2.0           # 34.0
PITCH_AREA = PITCH_LENGTH * PITCH_WIDTH

# ── Model de moviment del nivell 1 ──────────────────────────────────────────
# Temps de reacció (s) durant el qual el jugador es desplaça amb la velocitat
# actual abans d'esprintar. 0.7 s és un valor estàndard a la literatura de
# pitch control (Fernández & Bornn, 2018).
DEFAULT_REACT_TIME = 0.7

CONTROL_MODES = ("voronoi", "dominant")


# ── Generadors ──────────────────────────────────────────────────────────────

def momentum_shifted_positions(
    pos: np.ndarray,
    vel: np.ndarray,
    t_react: float = DEFAULT_REACT_TIME,
) -> np.ndarray:
    """
    Posicions efectives del nivell 1: x_eff = x + v · t_react.
    pos, vel: [M, 2] en metres i m/s. Retorna [M, 2].
    """
    return pos + vel * t_react


def _generators(
    pos: np.ndarray,
    vel: Optional[np.ndarray],
    mode: str,
    t_react: float,
) -> np.ndarray:
    """Tria els generadors de la tessel·lació segons el mode."""
    if mode == "voronoi":
        return pos
    if mode == "dominant":
        if vel is None:
            raise ValueError("mode='dominant' requereix velocitats (vel).")
        return momentum_shifted_positions(pos, vel, t_react)
    raise ValueError(f"mode='{mode}' desconegut; opcions: {CONTROL_MODES}")


# ── Graella del camp ────────────────────────────────────────────────────────

# Cache de la graella i de l'assignació de terços, indexat per grid_res. La
# graella i el terç de cada cel·la no depenen del frame, així que es construeixen
# una sola vegada per resolució i es reaprofiten a tots els frames.
_GRID_CACHE: Dict[float, tuple] = {}


def build_grid(grid_res: float = 1.0):
    """
    Construeix (o recupera del cache) una graella regular de punts sobre el
    camp. grid_res = cel·les per metre (1.0 → cel·les d'1 m²).
    Retorna (grid [G, 2], (ny, nx), cell_area, third_id [G]).
    third_id ∈ {0,1,2} indica el terç (left/mid/right) de cada cel·la.
    """
    cached = _GRID_CACHE.get(grid_res)
    if cached is not None:
        return cached

    nx = int(round(PITCH_LENGTH * grid_res))
    ny = int(round(PITCH_WIDTH * grid_res))
    # Centres de cel·la: evitem els extrems exactes per no esbiaixar les vores.
    xs = (np.arange(nx) + 0.5) / nx * PITCH_LENGTH - HALF_L
    ys = (np.arange(ny) + 0.5) / ny * PITCH_WIDTH - HALF_W
    gx, gy = np.meshgrid(xs, ys)                       # [ny, nx]
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)  # [G, 2]
    cell_area = (PITCH_LENGTH / nx) * (PITCH_WIDTH / ny)
    third_id = np.digitize(grid[:, 0], [-HALF_L / 3.0, HALF_L / 3.0])  # [G]

    result = (grid, (ny, nx), float(cell_area), third_id)
    _GRID_CACHE[grid_res] = result
    return result


# ── Assignació de control ───────────────────────────────────────────────────

def compute_control(
    pos: np.ndarray,
    teams: np.ndarray,
    vel: Optional[np.ndarray] = None,
    mode: str = "voronoi",
    grid_res: float = 1.0,
    t_react: float = DEFAULT_REACT_TIME,
) -> Dict[str, np.ndarray]:
    """
    Assigna cada cel·la de la graella al generador més proper i retorna
    l'assignació, els generadors i les àrees per jugador.

    Args:
      pos:   [M, 2] posicions dels M jugadors (en metres). SENSE la pilota.
      teams: [M]    índex d'equip de cada jugador (0=local, 1=visitant).
      vel:   [M, 2] velocitats (m/s); requerit només per mode='dominant'.
      mode:  'voronoi' o 'dominant'.

    Retorna un dict amb:
      'assignment' [G]   índex del jugador que controla cada cel·la
      'grid'       [G,2] coordenades de les cel·les
      'grid_shape' (ny, nx)
      'cell_area'  escalar (m²)
      'generators' [M,2] posicions generadores (desplaçades si dominant)
      'areas'      [M]   àrea controlada per cada jugador (m²)
    """
    pos = np.asarray(pos, dtype=np.float64)
    teams = np.asarray(teams, dtype=np.int64)
    if pos.ndim != 2 or pos.shape[1] != 2 or len(pos) < 2:
        raise ValueError(f"pos ha de tenir forma [M,2] amb M>=2; rebut {pos.shape}.")
    if teams.shape != (len(pos),):
        raise ValueError(
            f"teams ha de tenir forma ({len(pos)},); rebut {teams.shape}."
        )
    if not np.all(np.isfinite(pos)):
        raise ValueError("Les posicions del control d'espai contenen NaN o inf.")
    if vel is not None:
        vel = np.asarray(vel, dtype=np.float64)
        if vel.shape != pos.shape:
            raise ValueError(f"vel ha de tenir forma {pos.shape}; rebut {vel.shape}.")
        if not np.all(np.isfinite(vel)):
            raise ValueError("Les velocitats del control d'espai contenen NaN o inf.")

    gen = _generators(pos, vel, mode, t_react)
    if not np.all(np.isfinite(gen)):
        raise ValueError("Els generadors del control d'espai contenen NaN o inf.")
    grid, shape, cell_area, _third = build_grid(grid_res)

    dist = cdist(grid, gen)                            # [G, M]
    assignment = np.argmin(dist, axis=1)               # [G]

    counts = np.bincount(assignment, minlength=len(pos))
    areas = counts.astype(np.float64) * cell_area      # [M]

    return {
        "assignment": assignment,
        "grid": grid,
        "grid_shape": shape,
        "cell_area": cell_area,
        "generators": gen,
        "areas": areas,
    }


# ── Features per a la xarxa ─────────────────────────────────────────────────

def control_features(
    pos: np.ndarray,
    teams: np.ndarray,
    ball_xy: Optional[np.ndarray] = None,
    vel: Optional[np.ndarray] = None,
    mode: str = "voronoi",
    grid_res: float = 1.0,
    t_react: float = DEFAULT_REACT_TIME,
    ball_radius: float = 15.0,
) -> Dict[str, np.ndarray]:
    """
    Calcula les features de control d'espai d'un frame.

    Retorna un dict amb:
      Per-node:
        'player_area'        [M]   àrea controlada per jugador (m²)
        'player_area_frac'   [M]   fracció del camp controlada per jugador
      Globals (per al vector de context):
        'team_control'       [2]   fracció del camp controlada per cada equip
        'third_control'      [2,3] control de cada equip per terç (left/mid/right)
        'ball_zone_control'  [2]   control de cada equip dins el radi de la pilota
    """
    ctrl = compute_control(pos, teams, vel, mode, grid_res, t_react)
    assignment = ctrl["assignment"]
    grid = ctrl["grid"]
    cell_area = ctrl["cell_area"]
    areas = ctrl["areas"]

    team_of_cell = teams[assignment]                   # [G]

    # Control global per equip (fracció del camp)
    team_control = np.array([
        (team_of_cell == 0).sum() * cell_area / PITCH_AREA,
        (team_of_cell == 1).sum() * cell_area / PITCH_AREA,
    ])

    # Control per terç geomètric al llarg de x (left / middle / right).
    # third_id ve cachejat amb la graella (invariant entre frames).
    _, _, _, third_id = build_grid(grid_res)
    third_area = PITCH_AREA / 3.0
    third_control = np.zeros((2, 3))
    for t in (0, 1):
        for th in (0, 1, 2):
            m = (team_of_cell == t) & (third_id == th)
            third_control[t, th] = m.sum() * cell_area / third_area

    # Control de la zona al voltant de la pilota
    if ball_xy is not None:
        dist_ball = np.linalg.norm(grid - ball_xy[None, :], axis=1)
        inzone = dist_ball <= ball_radius
        zone_cells = max(int(inzone.sum()), 1)
        ball_zone_control = np.array([
            ((team_of_cell == 0) & inzone).sum() / zone_cells,
            ((team_of_cell == 1) & inzone).sum() / zone_cells,
        ])
    else:
        ball_zone_control = np.array([np.nan, np.nan])

    return {
        "player_area": areas,
        "player_area_frac": areas / PITCH_AREA,
        "team_control": team_control,
        "third_control": third_control,
        "ball_zone_control": ball_zone_control,
    }


# ── Extracció des d'un frame de node_numeric ────────────────────────────────

def extract_frame_players(node_numeric_frame: np.ndarray):
    """
    A partir d'un frame de node_numeric [N_NODES, 8] amb el layout
    [x, y, vx, vy, dx_ball, dy_ball, is_ball, team_idx], separa els jugadors
    vàlids de la pilota.

    Un slot es considera vàlid si no és tot zeros (un jugador absent al frame
    queda com a slot nul).

    Retorna (pos [M,2], vel [M,2], teams [M], ball_xy [2] o None).
    """
    is_ball = node_numeric_frame[:, 6] > 0.5
    # Slots actius: algun valor no nul a les coordenades o velocitats
    nonzero = np.abs(node_numeric_frame[:, :4]).sum(axis=1) > 0.0

    ball_rows = np.where(is_ball & nonzero)[0]
    ball_xy = (
        node_numeric_frame[ball_rows[0], :2].astype(np.float64)
        if len(ball_rows) > 0 else None
    )

    player_mask = (~is_ball) & nonzero
    pos   = node_numeric_frame[player_mask, :2].astype(np.float64)
    vel   = node_numeric_frame[player_mask, 2:4].astype(np.float64)
    teams = node_numeric_frame[player_mask, 7].astype(np.int64)

    return pos, vel, teams, ball_xy
