"""
Dataset multi-head per a la predicció de fases tàctiques.

Cada element correspon a una fase tàctica `curr` que té una fase posterior
`next` dins el mateix període. L'instant de predicció `t` cau dins la fase
actual; per defecte (`random_t=True`) es mostreja uniformement a cada crida
(`__getitem__`), de manera que cada època veu una `t` diferent per la mateixa
fase. Amb `random_t=False`, `t` queda deterministe a `val_t_fraction` de la
durada de la fase (per a validació/test).

L'input cobreix la fase prèvia (si existeix dins el mateix període) + la fase
actual fins a t, mostrejat amb `STRIDE`.

Targets:
  - phase_target  : tipus de la fase posterior (o STOPPAGE si Δt_proper > cap)
  - delta_restant : segons des de t fins a curr.frame_end (~0)
  - delta_proper  : segons des de t fins a next.frame_start, capat a 30 s
  - is_long_pause : 1 si Δt_proper crua > cap, 0 altrament
  - state_final   : posicions [N, 2] al frame next.frame_start

Dimensions de retorn (T_MAX = 200, N = 23, R = 4):
  node_numeric     [T, N, 8]
  position_idx     [N]
  context          [T, 15]
  adj_per_relation [R, T, N, N]
  frame_mask       [T]
  boundary_idx     scalar
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import (
    FPS,
    N_PLAYERS,
    N_NODES,
    N_NODE_NUMERIC_FEAT,
    N_CONTEXT_FEAT,
    N_EVENT_TYPES,
    SPATIAL_PROXIMITY_IDX,
    SPATIAL_SIGMA,
    PHASE_TYPE_TO_IDX,
    EVENT_TYPE_TO_IDX,
    POSITION_TO_IDX,
    BALL_POS_IDX,
    UNK_POS_IDX,
    STOPPAGE_IDX,
    STRIDE,
    T_MAX,
    T_PRED_MAX,
    DELTA_PROPER_CAP,
    MATCH_TIME_MAX_S,
)
from stgcn_tracking.generate_graph import MatchGraphBuilder


# ── Constants internes ──────────────────────────────────────────────────────

BALL_SLOT      = N_PLAYERS              # 22
N_PER_TEAM     = N_PLAYERS // 2         # 11
SCORE_DIFF_CAP = 5.0
PHASE_TIME_REF = 30.0                   # segons usats per normalitzar `frames_since_phase_start`

# Mapeig de noms llargs de role a acrònims del nostre vocabulari.
_POSITION_NAME_MAP = {
    "Goalkeeper":               "GK",
    "Left Center Back":         "LCB",
    "Right Center Back":        "RCB",
    "Center Back":              "CB",
    "Left Back":                "LB",
    "Right Back":               "RB",
    "Left Wing Back":           "LWB",
    "Right Wing Back":          "RWB",
    "Left Defensive Midfield":  "LDM",
    "Right Defensive Midfield": "RDM",
    "Defensive Midfield":       "DM",
    "Left Midfield":            "LM",
    "Right Midfield":           "RM",
    "Attacking Midfield":       "AM",
    "Left Wing":                "LW",
    "Right Wing":               "RW",
    "Left Winger":              "LW",
    "Right Winger":             "RW",
    "Left Forward":             "LF",
    "Right Forward":            "RF",
    "Center Forward":           "CF",
}


def _resolve_position(role_name: Optional[str], acronym: Optional[str]) -> str:
    if acronym and acronym in POSITION_TO_IDX and acronym not in ("BALL", "UNK"):
        return acronym
    if role_name and role_name in _POSITION_NAME_MAP:
        return _POSITION_NAME_MAP[role_name]
    return "UNK"


def _parse_timestamp(ts) -> float:
    """Converteix 'MM:SS.s' o 'HH:MM:SS.s' a segons. Tolerant a None i tipus numèrics."""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    if s in ("", "None", "nan"):
        return 0.0
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return float(h) * 3600 + float(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return float(m) * 60 + float(sec)
        return float(s)
    except ValueError:
        return 0.0


# ── Dades carregades d'un partit ────────────────────────────────────────────

class _MatchData:
    """Carrega tracking + events + phases + match.json d'un partit."""

    def __init__(self, match_dir: Path) -> None:
        match_dir = Path(match_dir)
        match_id = match_dir.name

        self.builder = MatchGraphBuilder(
            tracking_path       = match_dir / f"{match_id}_tracking_extrapolated.jsonl",
            dynamic_events_path = match_dir / f"{match_id}_dynamic_events.csv",
            phases_path         = match_dir / f"{match_id}_phases_of_play.csv",
        )
        self.builder.load_data()

        with (match_dir / f"{match_id}_match.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        self.home_team_id = int(meta["home_team"]["id"])
        self.away_team_id = int(meta["away_team"]["id"])

        self.player_team: Dict[int, int] = {}
        self.player_pos:  Dict[int, str] = {}
        for pl in meta.get("players", []):
            pid = int(pl["id"])
            self.player_team[pid] = int(pl["team_id"])
            role = pl.get("player_role") or {}
            self.player_pos[pid] = _resolve_position(role.get("name"), role.get("acronym"))

        self.phases = (
            self.builder.phases.sort_values(["period", "frame_start"]).reset_index(drop=True)
        )


# Cache mínim: només manté l'últim partit accedit.
class _MatchCache:
    def __init__(self) -> None:
        self._key:  Optional[str]        = None
        self._data: Optional[_MatchData] = None

    def get(self, match_dir: Path) -> _MatchData:
        key = str(match_dir)
        if key != self._key:
            self._data = _MatchData(match_dir)
            self._key  = key
        return self._data


# ── Dataset ─────────────────────────────────────────────────────────────────

class PhaseDataset(Dataset):
    """
    Una mostra per fase tàctica vàlida (que tingui fase posterior dins el mateix període).
    Punt de predicció: t = curr.frame_end - 1.
    """

    def __init__(
        self,
        match_dirs: List[Path],
        t_max: int = T_MAX,
        stride: int = STRIDE,
        delta_proper_cap_s: float = DELTA_PROPER_CAP,
        random_t: bool = True,
        val_t_fraction: float = 0.75,
        min_input_frames: int = 2,
    ) -> None:
        self.match_dirs         = [Path(d) for d in match_dirs]
        self.t_max              = int(t_max)
        self.stride             = int(stride)
        self.delta_proper_cap_s = float(delta_proper_cap_s)
        self.cap_frames         = int(self.delta_proper_cap_s * FPS)
        self.dt_step_s          = self.stride / FPS
        self.random_t           = bool(random_t)
        self.val_t_fraction     = float(np.clip(val_t_fraction, 0.0, 1.0))
        self.min_input_frames   = max(1, int(min_input_frames))

        self._cache = _MatchCache()

        # Index de mostres: (match_idx, phase_idx). Llegim només els CSV de phases
        # per construir-lo (els JSONL grans no s'obren al __init__).
        self.samples: List[Tuple[int, int]] = []
        for m_idx, match_dir in enumerate(self.match_dirs):
            self.samples.extend(
                (m_idx, p_idx)
                for p_idx in self._collect_valid_phase_indices(match_dir)
            )

    # ── helpers d'indexació ─────────────────────────────────────────────────

    def _collect_valid_phase_indices(self, match_dir: Path) -> List[int]:
        match_id = match_dir.name
        df = pd.read_csv(match_dir / f"{match_id}_phases_of_play.csv")
        df = df.sort_values(["period", "frame_start"]).reset_index(drop=True)

        valid: List[int] = []
        for i in range(len(df) - 1):
            curr, nxt = df.iloc[i], df.iloc[i + 1]
            if int(curr["period"]) != int(nxt["period"]):
                continue
            if pd.isna(curr.get("team_in_possession_phase_type")):
                continue
            valid.append(i)
        return valid

    # ── API Dataset ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        m_idx, p_idx = self.samples[idx]
        match = self._cache.get(self.match_dirs[m_idx])
        return self._build_sample(match, p_idx)

    # ── construcció d'una mostra ────────────────────────────────────────────

    def _build_sample(self, match: _MatchData, p_idx: int) -> Dict[str, torch.Tensor]:
        phases = match.phases
        curr   = phases.iloc[p_idx]
        nxt    = phases.iloc[p_idx + 1]
        period = int(curr["period"])

        # Instant de predicció: aleatori dins la fase (train) o deterministe (val).
        t_frame = self._sample_t(curr)

        # Fase prèvia (mateix període)
        prev = None
        if p_idx >= 1:
            cand = phases.iloc[p_idx - 1]
            if int(cand["period"]) == period:
                prev = cand

        start_frame = int(prev["frame_start"]) if prev is not None else int(curr["frame_start"])
        all_frames  = list(range(start_frame, t_frame + 1, self.stride))

        # Frontera fase prèvia / actual al sampling
        boundary = -1
        if prev is not None:
            prev_end = int(prev["frame_end"])
            for k, f in enumerate(all_frames):
                if f <= prev_end:
                    boundary = k

        # Truncament des del començament si supera T_MAX
        if len(all_frames) > self.t_max:
            overflow   = len(all_frames) - self.t_max
            all_frames = all_frames[overflow:]
            boundary   = boundary - overflow if boundary >= 0 else -1
            if boundary < 0:
                boundary = -1

        # ── Tensors d'entrada ───────────────────────────────────────────────
        node_numeric = np.zeros((self.t_max, N_NODES, N_NODE_NUMERIC_FEAT), dtype=np.float32)
        position_idx = np.full((N_NODES,), UNK_POS_IDX, dtype=np.int64)
        context      = np.zeros((self.t_max, N_CONTEXT_FEAT), dtype=np.float32)
        adj          = np.zeros((N_EVENT_TYPES, self.t_max, N_NODES, N_NODES), dtype=np.float32)
        frame_mask   = np.zeros((self.t_max,), dtype=bool)

        position_idx[BALL_SLOT] = BALL_POS_IDX

        # Mapeig estable player_id → slot (basat en el primer frame vàlid)
        slot_map = self._build_slot_map(match, all_frames)
        for pid, slot in slot_map.items():
            position_idx[slot] = POSITION_TO_IDX.get(match.player_pos.get(pid, "UNK"), UNK_POS_IDX)

        prev_xy: Dict[int, Optional[Tuple[float, float]]] = {s: None for s in range(N_NODES)}
        two_sigma_sq = 2.0 * (SPATIAL_SIGMA ** 2)

        for k, f in enumerate(all_frames):
            data = match.builder.get_frame_positions(f)
            if data is None or data.get("ball") is None:
                continue
            bx, by = float(data["ball"][0]), float(data["ball"][1])

            # Trackeja quins slots tenen posició real en aquest frame
            valid_slots = np.zeros(N_NODES, dtype=bool)

            # Jugadors
            for pid, (px, py, _tid) in data["players"].items():
                slot = slot_map.get(int(pid))
                if slot is None:
                    continue
                team_idx = 0.0 if match.player_team.get(int(pid)) == match.home_team_id else 1.0
                pxy = prev_xy[slot]
                vx  = (px - pxy[0]) / self.dt_step_s if pxy is not None else 0.0
                vy  = (py - pxy[1]) / self.dt_step_s if pxy is not None else 0.0
                node_numeric[k, slot] = [
                    float(px), float(py), float(vx), float(vy),
                    float(px) - bx, float(py) - by,
                    0.0, team_idx,
                ]
                prev_xy[slot] = (float(px), float(py))
                valid_slots[slot] = True

            # Pilota
            pxy_b = prev_xy[BALL_SLOT]
            vbx   = (bx - pxy_b[0]) / self.dt_step_s if pxy_b is not None else 0.0
            vby   = (by - pxy_b[1]) / self.dt_step_s if pxy_b is not None else 0.0
            node_numeric[k, BALL_SLOT] = [bx, by, float(vbx), float(vby), 0.0, 0.0, 1.0, 0.0]
            prev_xy[BALL_SLOT] = (bx, by)
            valid_slots[BALL_SLOT] = True

            # Context (coherent amb la fase real del frame: curr/prev/gap)
            context[k] = self._build_context(match, prev, curr, period, f)

            # Graf per relació (events del CSV)
            for ed in match.builder.build_frame_graph(f, period):
                ev = ed["event_type"]
                r  = EVENT_TYPE_TO_IDX.get(ev)
                if r is None:
                    continue
                src = ed["src"]
                dst = ed["dst"]
                ss  = BALL_SLOT if src == "ball" else slot_map.get(int(src))
                ds  = BALL_SLOT if dst == "ball" else slot_map.get(int(dst))
                if ss is None or ds is None:
                    continue
                adj[r, k, ss, ds] = 1.0
                if ev in ("on_ball_engagement", "player_possession"):
                    adj[r, k, ds, ss] = 1.0

            # Spatial proximity: gaussiana de la distància entre nodes vàlids
            xy = node_numeric[k, :, :2]                           # [N, 2]
            diff = xy[:, None, :] - xy[None, :, :]                # [N, N, 2]
            sq_dist = (diff * diff).sum(axis=-1)                  # [N, N]
            gauss = np.exp(-sq_dist / two_sigma_sq).astype(np.float32)
            np.fill_diagonal(gauss, 0.0)                          # sense self-loop
            mask2d = valid_slots[:, None] & valid_slots[None, :]  # [N, N]
            adj[SPATIAL_PROXIMITY_IDX, k] = gauss * mask2d

            frame_mask[k] = True

        # ── Targets ─────────────────────────────────────────────────────────
        delta_proper_frames = int(nxt["frame_start"]) - t_frame
        delta_proper_raw_s  = delta_proper_frames / FPS
        is_long_pause       = float(delta_proper_raw_s > self.delta_proper_cap_s)

        if is_long_pause:
            phase_target = STOPPAGE_IDX
        else:
            nxt_type = nxt.get("team_in_possession_phase_type")
            phase_target = (
                PHASE_TYPE_TO_IDX[nxt_type]
                if isinstance(nxt_type, str) and nxt_type in PHASE_TYPE_TO_IDX
                else STOPPAGE_IDX
            )

        delta_restant = max(0.0, (int(curr["frame_end"]) - t_frame) / FPS)
        delta_proper  = float(min(max(delta_proper_raw_s, 0.0), self.delta_proper_cap_s))

        # Trajectòria target: per cada step k ∈ {1..T_PRED_MAX} llegim el frame
        # t + k·stride del tracking. Si k·stride > delta_proper_frames, queda
        # més enllà de next.frame_start → es marca com a invàlid a target_mask.
        target_traj = np.zeros((T_PRED_MAX, N_NODES, 2), dtype=np.float32)
        target_mask = np.zeros((T_PRED_MAX,), dtype=bool)
        if not is_long_pause:
            n_pred = min(T_PRED_MAX, max(0, round(delta_proper_frames / self.stride)))
            for k in range(n_pred):
                frame_k = t_frame + (k + 1) * self.stride
                target_traj[k] = self._compute_state_final(match, slot_map, frame_k)
                target_mask[k] = True

        return {
            "node_numeric":     torch.from_numpy(node_numeric),
            "position_idx":     torch.from_numpy(position_idx),
            "context":          torch.from_numpy(context),
            "adj_per_relation": torch.from_numpy(adj),
            "phase_target":     torch.tensor(phase_target,  dtype=torch.long),
            "delta_restant":    torch.tensor(delta_restant, dtype=torch.float32),
            "delta_proper":     torch.tensor(delta_proper,  dtype=torch.float32),
            "is_long_pause":    torch.tensor(is_long_pause, dtype=torch.float32),
            "target_traj":      torch.from_numpy(target_traj),
            "target_mask":      torch.from_numpy(target_mask),
            "frame_mask":       torch.from_numpy(frame_mask),
            "boundary_idx":     torch.tensor(boundary, dtype=torch.long),
        }

    # ── helpers de construcció ──────────────────────────────────────────────

    def _sample_t(self, curr: pd.Series) -> int:
        """
        Mostreja l'instant de predicció `t` (en frames originals) dins la fase
        actual. Garanteix com a mínim `min_input_frames` mostrejats de la fase
        actual: t ≥ frame_start + (min_input_frames - 1) * stride.

        En mode `random_t`, uniforme a [t_min, t_max].
        Altrament, deterministe a `val_t_fraction` del rang.

        Si la fase és massa curta per arribar al mínim, retorna el màxim possible.
        """
        fs = int(curr["frame_start"])
        fe = int(curr["frame_end"])
        min_offset = (self.min_input_frames - 1) * self.stride
        t_min = fs + min_offset
        t_max = fe - 1

        if t_max < t_min:
            return max(fs, t_max)
        if t_max == t_min:
            return t_min
        if self.random_t:
            return int(np.random.default_rng().integers(t_min, t_max + 1))
        return t_min + int(self.val_t_fraction * (t_max - t_min))

    def _build_slot_map(self, match: _MatchData, frames: List[int]) -> Dict[int, int]:
        """11 slots home (0–10) + 11 away (11–21), assignats al primer frame amb dades."""
        target = None
        for f in frames:
            data = match.builder.get_frame_positions(f)
            if data is not None and data.get("players"):
                target = data
                break
        if target is None:
            return {}

        home_pids: List[int] = []
        away_pids: List[int] = []
        for pid in target["players"].keys():
            tid = match.player_team.get(int(pid))
            if   tid == match.home_team_id: home_pids.append(int(pid))
            elif tid == match.away_team_id: away_pids.append(int(pid))
        home_pids.sort()
        away_pids.sort()

        slot_map: Dict[int, int] = {}
        for i, pid in enumerate(home_pids[:N_PER_TEAM]):
            slot_map[pid] = i
        for i, pid in enumerate(away_pids[:N_PER_TEAM]):
            slot_map[pid] = N_PER_TEAM + i
        return slot_map

    def _build_context(
        self,
        match: _MatchData,
        prev: Optional[pd.Series],
        curr: pd.Series,
        period: int,
        frame: int,
    ) -> np.ndarray:
        """
        Construeix el vector de context d'un frame, **coherent amb la fase a
        la qual pertany realment el frame**:
          - frames dins de `curr` → atributs de curr,   is_current_phase = 1
          - frames dins de `prev` → atributs de prev,   is_current_phase = 0
          - frames al gap entre prev i curr (stoppage real) →
                phase_type = STOPPAGE,                  is_current_phase = 0
                attacking/possession heretats de prev (per coherència)

        Layout del vector (N_CONTEXT_FEAT = 16):
          [0]    match_time_norm
          [1]    period (0 o 1)
          [2]    score_diff_norm  (0 per ara: no disponible per frame)
          [3]    attacking_LtR
          [4]    team_a_in_poss
          [5]    frames_since_phase_start_norm (relatiu a la fase pròpia)
          [6]    is_current_phase    ← nou
          [7:16] one-hot del phase_type (9 classes, incloent stoppage)
        """
        ctx = np.zeros((N_CONTEXT_FEAT,), dtype=np.float32)

        # ── temps absolut i període (no depenen de la fase) ─────────────────
        ft = match.builder.tracking_frames.get(frame)
        match_time_s = _parse_timestamp(ft.timestamp) if ft is not None else 0.0
        ctx[0] = float(np.clip(match_time_s / MATCH_TIME_MAX_S, 0.0, 1.0))
        ctx[1] = float(period - 1)
        ctx[2] = 0.0   # score_diff (no disponible per frame)

        # ── fase a la qual pertany el frame ────────────────────────────────
        in_curr = (int(curr["frame_start"]) <= frame <= int(curr["frame_end"]))
        in_prev = (
            prev is not None
            and int(prev["frame_start"]) <= frame <= int(prev["frame_end"])
        )

        if in_curr:
            ref = curr
            phase_type   = ref.get("team_in_possession_phase_type")
            phase_start  = int(ref["frame_start"])
            ctx[6] = 1.0  # is_current_phase
        elif in_prev:
            ref = prev
            phase_type   = ref.get("team_in_possession_phase_type")
            phase_start  = int(ref["frame_start"])
            ctx[6] = 0.0
        else:
            # Frame al gap entre prev i curr: stoppage real.
            ref = prev if prev is not None else curr   # per heretar atacant/poss.
            phase_type   = "stoppage"
            phase_start  = int(ref["frame_end"])       # rellotge des del final de prev
            ctx[6] = 0.0

        # ── atributs derivats de la fase de referència ─────────────────────
        attacking_side = ref.get("attacking_side", "left_to_right")
        ctx[3] = 1.0 if attacking_side == "left_to_right" else 0.0

        team_in_poss_id = ref.get("team_in_possession_id")
        if pd.isna(team_in_poss_id):
            ctx[4] = 0.0
        else:
            ctx[4] = 1.0 if int(team_in_poss_id) == match.home_team_id else 0.0

        ctx[5] = float(np.clip((frame - phase_start) / FPS / PHASE_TIME_REF, 0.0, 1.0))

        # ── one-hot del phase_type (índex 7..15) ───────────────────────────
        if isinstance(phase_type, str) and phase_type in PHASE_TYPE_TO_IDX:
            ctx[7 + PHASE_TYPE_TO_IDX[phase_type]] = 1.0
        else:
            ctx[7 + STOPPAGE_IDX] = 1.0

        return ctx

    def _compute_state_final(
        self,
        match: _MatchData,
        slot_map: Dict[int, int],
        target_frame: int,
    ) -> np.ndarray:
        state = np.zeros((N_NODES, 2), dtype=np.float32)
        data = match.builder.get_frame_positions(target_frame)
        if data is None:
            return state
        for pid, (x, y, _tid) in data["players"].items():
            slot = slot_map.get(int(pid))
            if slot is not None:
                state[slot] = [float(x), float(y)]
        ball = data.get("ball")
        if ball is not None:
            state[BALL_SLOT] = [float(ball[0]), float(ball[1])]
        return state
