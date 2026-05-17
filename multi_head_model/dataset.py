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
    N_PHASE_CLASSES,
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


# Cache LRU. Per defecte, mida 1 (només l'últim partit accedit). En entrenament
# amb shuffle=True és essencial pujar la mida a `len(match_dirs)` per evitar
# recarregar contínuament el tracking JSONL (cada partit ≈ 200 MB de RAM,
# però només es carrega un cop per època).
from collections import OrderedDict


class _MatchCache:
    def __init__(self, max_size: int = 1) -> None:
        self.max_size = max(1, int(max_size))
        self._store: "OrderedDict[str, _MatchData]" = OrderedDict()

    def get(self, match_dir: Path) -> _MatchData:
        key = str(match_dir)
        if key in self._store:
            self._store.move_to_end(key)              # marca com a recent
            return self._store[key]
        import time, sys
        t0 = time.time()
        print(f"  [cache] carregant partit {match_dir.name}...",
              end="", flush=True, file=sys.stderr)
        data = _MatchData(match_dir)
        print(f" {time.time() - t0:.1f}s "
              f"(cache: {len(self._store) + 1}/{self.max_size})",
              file=sys.stderr, flush=True)
        self._store[key] = data
        if len(self._store) > self.max_size:
            evicted = self._store.popitem(last=False)
            print(f"  [cache] evict {Path(evicted[0]).name}",
                  file=sys.stderr, flush=True)
        return data


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
        val_t_fractions: Optional[List[float]] = None,
        min_input_frames: int = 2,
        cache_size: Optional[int] = None,
        samples_per_phase: int = 1,
    ) -> None:
        """
        random_t / val_t_fraction / val_t_fractions controlen com es tria
        l'instant de predicció `t` dins de la fase actual:

          - random_t=True  → uniforme aleatori a cada __getitem__
                             (val_t_fraction(s) s'ignoren). Per entrenament.

          - random_t=False
              · si val_t_fractions és None: `t = val_t_fraction · durada`
                (1 mostra per fase, comportament original).
              · si val_t_fractions és una llista: el dataset multiplica cada
                fase per len(val_t_fractions), una mostra per cada fracció.
                Aquesta és la configuració "multipoint" recomanada per a val
                i test, per cobrir uniformement diversos horitzons.
        """
        self.match_dirs         = [Path(d) for d in match_dirs]
        self.t_max              = int(t_max)
        self.stride             = int(stride)
        self.delta_proper_cap_s = float(delta_proper_cap_s)
        self.cap_frames         = int(self.delta_proper_cap_s * FPS)
        self.dt_step_s          = self.stride / FPS
        self.random_t           = bool(random_t)
        self.val_t_fraction     = float(np.clip(val_t_fraction, 0.0, 1.0))
        self.val_t_fractions    = (
            [float(np.clip(f, 0.0, 1.0)) for f in val_t_fractions]
            if val_t_fractions is not None else None
        )
        self.min_input_frames   = max(1, int(min_input_frames))
        self.samples_per_phase  = max(1, int(samples_per_phase))

        # Mida del cache LRU. Per defecte = len(match_dirs) → tots els partits
        # en memòria després del primer accés. Per cada partit ≈ 200 MB de RAM.
        eff_cache = cache_size if cache_size is not None else len(self.match_dirs)
        self._cache = _MatchCache(max_size=eff_cache)

        # Index de mostres: (match_idx, phase_idx, frac_idx). El `frac_idx`
        # només és rellevant quan random_t=False i val_t_fractions està
        # definit; en els altres casos val sempre 0.
        #
        # Quan random_t=True (train), cada fase es replica `samples_per_phase`
        # vegades. Cada còpia rebrà un `t` aleatori independent al
        # __getitem__, fet que multiplica el nombre de mostres efectives per
        # època sense duplicar informació (és augmentació estocàstica
        # explícita). Amb shuffle=True, les còpies queden distribuïdes
        # uniformement al llarg de l'època.
        if self.random_t:
            n_replicas = self.samples_per_phase
        else:
            n_replicas = (
                len(self.val_t_fractions)
                if self.val_t_fractions is not None else 1
            )
        self.samples: List[Tuple[int, int, int]] = []
        for m_idx, match_dir in enumerate(self.match_dirs):
            for p_idx in self._collect_valid_phase_indices(match_dir):
                for frac_idx in range(n_replicas):
                    self.samples.append((m_idx, p_idx, frac_idx))

    def compute_class_stats(self, verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Itera totes les mostres del dataset i agrega comptatges per a:
          - event_counts  [N_PHASE_CLASSES]  → recompte de cada phase_target
          - pause_counts  [2]                → 0=no_pause, 1=long_pause
          - poss_counts   [2]                → 0=manteniment, 1=canvi
                                              (ignora els sentinels -1)

        Implementació *lleugera*: replica només la lògica del càlcul de
        targets de `_build_sample` (sense generar el graf, tracking ni
        context) per ser ràpida. Útil per a pesos de classes a la loss.
        """
        event_counts = np.zeros(N_PHASE_CLASSES, dtype=np.int64)
        pause_counts = np.zeros(2, dtype=np.int64)
        poss_counts  = np.zeros(2, dtype=np.int64)

        if verbose:
            print(f"  [class_stats] iterant {len(self.samples)} mostres...")

        for m_idx, p_idx, frac_idx in self.samples:
            match = self._cache.get(self.match_dirs[m_idx])
            phases = match.phases
            curr   = phases.iloc[p_idx]
            nxt    = phases.iloc[p_idx + 1]

            t_frame = self._sample_t(curr, frac_idx)
            delta_proper_frames = int(nxt["frame_start"]) - t_frame
            delta_proper_raw_s  = delta_proper_frames / FPS
            is_long_pause       = delta_proper_raw_s > self.delta_proper_cap_s

            # phase_target
            if is_long_pause:
                pt = STOPPAGE_IDX
            else:
                nxt_type = nxt.get("team_in_possession_phase_type")
                pt = (
                    PHASE_TYPE_TO_IDX[nxt_type]
                    if isinstance(nxt_type, str) and nxt_type in PHASE_TYPE_TO_IDX
                    else STOPPAGE_IDX
                )
            event_counts[pt] += 1
            pause_counts[int(is_long_pause)] += 1

            # possession_change (ignorem -1 sentinels)
            curr_team = curr.get("team_in_possession_id")
            nxt_team  = nxt.get("team_in_possession_id")
            if not (pd.isna(curr_team) or pd.isna(nxt_team) or is_long_pause):
                poss_counts[int(int(curr_team) != int(nxt_team))] += 1

        if verbose:
            print(f"  [class_stats] event = {event_counts.tolist()}")
            print(f"  [class_stats] pause = {pause_counts.tolist()}  "
                  f"(pos_weight ≈ {pause_counts[0] / max(pause_counts[1], 1):.2f})")
            print(f"  [class_stats] poss  = {poss_counts.tolist()}  "
                  f"(pos_weight ≈ {poss_counts[0] / max(poss_counts[1], 1):.2f})")

        return {
            "event": event_counts,
            "pause": pause_counts,
            "poss":  poss_counts,
        }

    def warm_cache(self, verbose: bool = True) -> None:
        """
        Pre-carrega tots els partits al cache LRU. Útil al començament d'un
        entrenament per evitar que el primer batch trigui minuts a causa de
        càrregues sincrones de JSONL gegants.

        Només té sentit cridar-ho si `cache_size >= len(match_dirs)`; en cas
        contrari, els primers partits carregats seran evictats abans
        d'arribar a entrenar amb ells.
        """
        if self._cache.max_size < len(self.match_dirs) and verbose:
            print(
                f"  [warm_cache] AVÍS: cache_size={self._cache.max_size} < "
                f"#partits={len(self.match_dirs)}; alguns partits es "
                f"recarregaran durant l'entrenament."
            )
        for i, match_dir in enumerate(self.match_dirs, 1):
            if verbose:
                print(f"  [warm_cache] partit {i}/{len(self.match_dirs)}: "
                      f"{match_dir.name}")
            self._cache.get(match_dir)

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
        m_idx, p_idx, frac_idx = self.samples[idx]
        match = self._cache.get(self.match_dirs[m_idx])
        return self._build_sample(match, p_idx, frac_idx)

    # ── construcció d'una mostra ────────────────────────────────────────────

    def _build_sample(
        self,
        match: _MatchData,
        p_idx: int,
        frac_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        phases = match.phases
        curr   = phases.iloc[p_idx]
        nxt    = phases.iloc[p_idx + 1]
        period = int(curr["period"])

        # Instant de predicció: aleatori dins la fase (train) o deterministe (val).
        t_frame = self._sample_t(curr, frac_idx)

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

        # Canvi de possessió entre la fase actual i la propera.
        # 1 = canvi de possessió, 0 = manteniment.
        # En long_pause el target es marca com -1 (mask) per indicar irrelevant.
        curr_team = curr.get("team_in_possession_id")
        nxt_team  = nxt.get("team_in_possession_id")
        if (pd.isna(curr_team) or pd.isna(nxt_team)) or is_long_pause:
            possession_change = -1.0      # sentinel d'ignore
        else:
            possession_change = float(int(curr_team) != int(nxt_team))

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
            "node_numeric":      torch.from_numpy(node_numeric),
            "position_idx":      torch.from_numpy(position_idx),
            "context":           torch.from_numpy(context),
            "adj_per_relation":  torch.from_numpy(adj),
            "phase_target":      torch.tensor(phase_target,      dtype=torch.long),
            "delta_restant":     torch.tensor(delta_restant,     dtype=torch.float32),
            "delta_proper":      torch.tensor(delta_proper,      dtype=torch.float32),
            "is_long_pause":     torch.tensor(is_long_pause,     dtype=torch.float32),
            "possession_change": torch.tensor(possession_change, dtype=torch.float32),
            "target_traj":       torch.from_numpy(target_traj),
            "target_mask":       torch.from_numpy(target_mask),
            "frame_mask":        torch.from_numpy(frame_mask),
            "boundary_idx":      torch.tensor(boundary, dtype=torch.long),
        }

    # ── helpers de construcció ──────────────────────────────────────────────

    def _sample_t(self, curr: pd.Series, frac_idx: int = 0) -> int:
        """
        Mostreja l'instant de predicció `t` (en frames originals) dins la fase
        actual. Garanteix com a mínim `min_input_frames` mostrejats de la fase
        actual: t ≥ frame_start + (min_input_frames - 1) * stride.

        Modes:
          - random_t=True:
              · samples_per_phase==1 → uniforme a [t_min, t_max].
              · samples_per_phase>1  → **mostreig estratificat**:
                el rang es divideix en N=samples_per_phase segments iguals i
                la còpia `frac_idx` mostreja dins del seu segment. Així:
                  · cap còpia pot caure al mateix `t` (sense col·lisions)
                  · les còpies cobreixen uniformement el rang
                  · cada còpia manté aleatorietat *dins* del seu segment
          - random_t=False:
              · si val_t_fractions està definit → fa servir
                `val_t_fractions[frac_idx]` (mode multipoint).
              · si no, fa servir `val_t_fraction` (un sol punt deterministe).

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
            N = self.samples_per_phase
            if N <= 1:
                return int(np.random.default_rng().integers(t_min, t_max + 1))
            # Mostreig estratificat: segment frac_idx-èsim dels N
            seg_size = (t_max - t_min + 1) / N
            seg_lo = int(round(t_min + frac_idx * seg_size))
            seg_hi = int(round(t_min + (frac_idx + 1) * seg_size))
            seg_hi = max(seg_lo + 1, seg_hi)         # garanteix rang no buit
            seg_hi = min(seg_hi, t_max + 1)          # no sortim del rang
            return int(np.random.default_rng().integers(seg_lo, seg_hi))

        frac = (
            self.val_t_fractions[frac_idx]
            if self.val_t_fractions is not None
            else self.val_t_fraction
        )
        return t_min + int(frac * (t_max - t_min))

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

        Layout del vector (N_CONTEXT_FEAT = 15):
          [0]    match_time_norm
          [1]    period (0 o 1)
          [2]    attacking_LtR
          [3]    team_a_in_poss
          [4]    frames_since_phase_start_norm (relatiu a la fase pròpia)
          [5]    is_current_phase
          [6:15] one-hot del phase_type (9 classes, incloent stoppage)
        """
        ctx = np.zeros((N_CONTEXT_FEAT,), dtype=np.float32)

        # ── temps absolut i període (no depenen de la fase) ─────────────────
        ft = match.builder.tracking_frames.get(frame)
        match_time_s = _parse_timestamp(ft.timestamp) if ft is not None else 0.0
        ctx[0] = float(np.clip(match_time_s / MATCH_TIME_MAX_S, 0.0, 1.0))
        ctx[1] = float(period - 1)

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
            ctx[5] = 1.0  # is_current_phase
        elif in_prev:
            ref = prev
            phase_type   = ref.get("team_in_possession_phase_type")
            phase_start  = int(ref["frame_start"])
            ctx[5] = 0.0
        else:
            # Frame al gap entre prev i curr: stoppage real.
            ref = prev if prev is not None else curr   # per heretar atacant/poss.
            phase_type   = "stoppage"
            phase_start  = int(ref["frame_end"])       # rellotge des del final de prev
            ctx[5] = 0.0

        # ── atributs derivats de la fase de referència ─────────────────────
        attacking_side = ref.get("attacking_side", "left_to_right")
        ctx[2] = 1.0 if attacking_side == "left_to_right" else 0.0

        team_in_poss_id = ref.get("team_in_possession_id")
        if pd.isna(team_in_poss_id):
            ctx[3] = 0.0
        else:
            ctx[3] = 1.0 if int(team_in_poss_id) == match.home_team_id else 0.0

        ctx[4] = float(np.clip((frame - phase_start) / FPS / PHASE_TIME_REF, 0.0, 1.0))

        # ── one-hot del phase_type (índex 6..14) ───────────────────────────
        if isinstance(phase_type, str) and phase_type in PHASE_TYPE_TO_IDX:
            ctx[6 + PHASE_TYPE_TO_IDX[phase_type]] = 1.0
        else:
            ctx[6 + STOPPAGE_IDX] = 1.0

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
