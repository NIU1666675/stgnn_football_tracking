from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import argparse

# orjson és ~3-5× més ràpid que json estàndard. Si no està instal·lat,
# caiem a json (drop-in compatible per a .loads).
try:
    import orjson as _fastjson  # type: ignore
    def _json_loads(line: str | bytes):
        return _fastjson.loads(line)
except ImportError:
    def _json_loads(line: str | bytes):
        return json.loads(line)


@dataclass
class FrameTracking:
    frame: int
    timestamp: float
    period: int
    ball_data: dict
    possession: dict
    player_data: list


class MatchGraphBuilder:
    def __init__(
        self,
        tracking_path: str | Path,
        dynamic_events_path: str | Path,
        phases_path: str | Path,
    ) -> None:
        self.tracking_path = Path(tracking_path)
        self.dynamic_events_path = Path(dynamic_events_path)
        self.phases_path = Path(phases_path)

        self.tracking_frames: Dict[int, FrameTracking] = {}
        self.dynamic_events: pd.DataFrame = pd.DataFrame()
        self.phases: pd.DataFrame = pd.DataFrame()
        self._player_team_map_cache: Optional[Dict[int, int]] = None

    def load_data(self) -> None:
        self._load_tracking()
        self._load_dynamic_events()
        self._load_phases()

    def _load_tracking(self) -> None:
        tracking_frames: Dict[int, FrameTracking] = {}

        with self.tracking_path.open("rb") as f:        # rb per a orjson (3–5× més ràpid)
            for line in f:
                row = _json_loads(line)

                # Frames de warm-up del tracking (abans del kickoff o entre
                # períodes) porten `period: null`. Els saltem silenciosament,
                # ja que no formen part del joc.
                if row.get("period") is None:
                    continue

                try:
                    frame_obj = FrameTracking(
                        frame=int(row["frame"]),
                        timestamp=str(row["timestamp"]),
                        period=int(row["period"]),
                        ball_data=row.get("ball_data", {}),
                        possession=row.get("possession", {}),
                        player_data=row.get("player_data", []),
                    )
                    tracking_frames[frame_obj.frame] = frame_obj
                except Exception as e:
                    print(f"Error al procesar frame {row.get('frame', 'desconocido')}: {e}")

        self.tracking_frames = tracking_frames

    def _load_dynamic_events(self) -> None:
        df = pd.read_csv(self.dynamic_events_path, low_memory=False)

        required_cols = {"frame_start", "frame_end", "period"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en dynamic_events: {missing}")

        df["frame_start"] = df["frame_start"].astype(int)
        df["frame_end"]   = df["frame_end"].astype(int)
        df["period"]      = df["period"].astype(int)

        self.dynamic_events = df

        # Pre-computar arrays NumPy per a un accés O(n_events) vectoritzat
        # a get_active_events_at_frame() i build_frame_graph(). Sense això,
        # pandas filtra el DataFrame sencer per cada frame (~50× més lent).
        n = len(df)
        self._ev_period   = df["period"].to_numpy(dtype=np.int64, copy=True)
        self._ev_fs       = df["frame_start"].to_numpy(dtype=np.int64, copy=True)
        self._ev_fe       = df["frame_end"].to_numpy(dtype=np.int64, copy=True)
        self._ev_type     = df.get(
            "event_type", pd.Series([None] * n)
        ).to_numpy(dtype=object, copy=True)
        # player_id i player_in_possession_id poden tenir NaN → float
        self._ev_pid      = df.get(
            "player_id", pd.Series([np.nan] * n)
        ).to_numpy(dtype=np.float64, copy=True)
        self._ev_poss     = df.get(
            "player_in_possession_id", pd.Series([np.nan] * n)
        ).to_numpy(dtype=np.float64, copy=True)
        self._ev_subtype  = df.get(
            "event_subtype", pd.Series([None] * n)
        ).to_numpy(dtype=object, copy=True)

    def _load_phases(self) -> None:
        df = pd.read_csv(self.phases_path)

        required_cols = {"frame_start", "frame_end", "period"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en phases_of_play: {missing}")

        df["frame_start"] = df["frame_start"].astype(int)
        df["frame_end"] = df["frame_end"].astype(int)
        df["period"] = df["period"].astype(int)

        self.phases = df

    def get_phase_interval(
        self,
        period: int,
        phase_row_idx: Optional[int] = None,
        team_in_possession_phase_type: Optional[str] = None,
        team_out_of_possession_phase_type: Optional[str] = None,
    ) -> pd.Series:
        df = self.phases[self.phases["period"] == period].copy()

        if team_in_possession_phase_type is not None:
            df = df[
                df["team_in_possession_phase_type"] == team_in_possession_phase_type
            ]

        if team_out_of_possession_phase_type is not None:
            df = df[
                df["team_out_of_possession_phase_type"] == team_out_of_possession_phase_type
            ]

        if df.empty:
            raise ValueError("No se ha encontrado ninguna fase con esos filtros.")

        if phase_row_idx is None:
            return df.iloc[0]

        if phase_row_idx < 0 or phase_row_idx >= len(df):
            raise IndexError("phase_row_idx fuera de rango.")

        return df.iloc[phase_row_idx]

    def get_tracking_frames_for_phase(self, phase_row: pd.Series) -> List[FrameTracking]:
        frame_start = int(phase_row["frame_start"])
        frame_end = int(phase_row["frame_end"])
        period = int(phase_row["period"])

        frames: List[FrameTracking] = []
        for frame in range(frame_start, frame_end + 1):
            frame_obj = self.tracking_frames.get(frame)
            if frame_obj is None:
                continue
            if frame_obj.period != period:
                continue
            frames.append(frame_obj)

        return frames

    def get_dynamic_events_for_phase(self, phase_row: pd.Series) -> pd.DataFrame:
        frame_start = int(phase_row["frame_start"])
        frame_end = int(phase_row["frame_end"])
        period = int(phase_row["period"])

        df = self.dynamic_events[
            (self.dynamic_events["period"] == period)
            & (self.dynamic_events["frame_start"] <= frame_end)
            & (self.dynamic_events["frame_end"] >= frame_start)
        ].copy()

        return df

    def get_player_team_map(self) -> Dict[int, int]:
        """
        Construeix un mapping player_id → team_id a partir dels dynamic_events.
        Cada (player_id, team_id) apareix com un parell consistent als esdeveniments.

        El resultat es cacheja al primer ús; les crides següents són O(1).
        Cal així perquè `get_frame_positions` el demana per cada frame i sense
        cache cada `_build_sample` repetia el càlcul ~200 vegades.
        """
        if self._player_team_map_cache is not None:
            return self._player_team_map_cache
        if self.dynamic_events.empty:
            self._player_team_map_cache = {}
            return self._player_team_map_cache
        df = self.dynamic_events[["player_id", "team_id"]].dropna().drop_duplicates()
        self._player_team_map_cache = {
            int(r.player_id): int(r.team_id) for r in df.itertuples()
        }
        return self._player_team_map_cache

    def get_frame_positions(self, frame: int) -> Optional[Dict[str, Any]]:
        """
        Retorna les posicions de tots els jugadors detectats i de la pilota
        per a un frame concret.

        Format:
            {
                "players": {player_id: (x, y, team_id_or_None)},
                "ball": (x, y) or None,
                "possession_player_id": int or None,
            }
        """
        ft = self.tracking_frames.get(frame)
        if ft is None:
            return None

        team_map = self.get_player_team_map()
        players: Dict[int, Any] = {}
        for p in ft.player_data:
            pid = p.get("player_id")
            x, y = p.get("x"), p.get("y")
            if pid is None or x is None or y is None:
                continue
            players[int(pid)] = (float(x), float(y), team_map.get(int(pid)))

        ball = None
        b = ft.ball_data
        if b and b.get("x") is not None and b.get("y") is not None:
            ball = (float(b["x"]), float(b["y"]))

        poss = ft.possession.get("player_id") if ft.possession else None
        poss = int(poss) if poss is not None else None

        return {"players": players, "ball": ball, "possession_player_id": poss}

    def get_active_events_at_frame(self, frame: int, period: int) -> pd.DataFrame:
        """Esdeveniments dinàmics actius en el frame indicat (de el període donat).

        Versió compatible amb codi antic (retorna DataFrame). Per al hot path
        del dataset, vegis _active_indices() que retorna índexs NumPy directament.
        """
        df = self.dynamic_events
        return df[
            (df["period"] == period)
            & (df["frame_start"] <= frame)
            & (df["frame_end"] >= frame)
        ].copy()

    def _active_indices(self, frame: int, period: int) -> np.ndarray:
        """Retorna els índexs d'events actius en (frame, period). O(n_events) vectoritzat."""
        mask = (
            (self._ev_period == period)
            & (self._ev_fs <= frame)
            & (self._ev_fe >= frame)
        )
        return np.flatnonzero(mask)

    def build_frame_graph(self, frame: int, period: int) -> List[Dict[str, Any]]:
        """
        Construeix la llista d'arestes del graf per a un frame concret a partir
        dels dynamic_events actius. Cada aresta té:
            {
                "src": player_id_source,
                "dst": player_id_dst   (o "ball"),
                "event_type": str,
                "event_subtype": str or None,
            }

        Convenció:
          - player_possession : aresta player_id  ↔  ball
          - passing_option    : aresta player_in_possession_id  →  player_id
          - off_ball_run      : aresta player_id  →  player_in_possession_id (o "ball")
          - on_ball_engagement: aresta player_id  ↔  player_in_possession_id (duel)

        Implementació vectoritzada amb arrays NumPy pre-computats: ~50× més
        ràpid que el filtre pandas equivalent.
        """
        idx = self._active_indices(frame, period)
        edges: List[Dict[str, Any]] = []

        ev_type_arr = self._ev_type
        ev_pid_arr  = self._ev_pid
        ev_poss_arr = self._ev_poss
        ev_sub_arr  = self._ev_subtype

        for i in idx:
            ev_type = ev_type_arr[i]
            pid_f   = ev_pid_arr[i]
            if not np.isfinite(pid_f):
                continue
            pid     = int(pid_f)
            poss_f  = ev_poss_arr[i]
            poss    = int(poss_f) if np.isfinite(poss_f) else None
            sub     = ev_sub_arr[i]

            if ev_type == "player_possession":
                edges.append({"src": pid, "dst": "ball", "event_type": ev_type, "event_subtype": sub})
            elif ev_type == "passing_option" and poss is not None:
                edges.append({"src": poss, "dst": pid, "event_type": ev_type, "event_subtype": sub})
            elif ev_type == "off_ball_run":
                dst = poss if poss is not None else "ball"
                edges.append({"src": pid, "dst": dst, "event_type": ev_type, "event_subtype": sub})
            elif ev_type == "on_ball_engagement" and poss is not None:
                edges.append({"src": pid, "dst": poss, "event_type": ev_type, "event_subtype": sub})

        return edges

    def plot_phase_graphs(
        self,
        phase_row: pd.Series,
        n_frames: int = 9,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """
        Visualitza el graf d'esdeveniments per a n_frames mostrejats uniformement
        al llarg de la fase indicada. Els subplots es disposen en una graella
        quadrada (3x3 per defecte) sobre representacions de camp de futbol.

        Cada subplot mostra:
          - Camp amb mides estàndard
          - Jugadors com a cercles colorejats per equip
          - Pilota com a cercle blanc
          - Arestes colorejades per tipus d'esdeveniment
          - Posseïdor destacat amb borde groc
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        frame_start = int(phase_row["frame_start"])
        frame_end = int(phase_row["frame_end"])
        period = int(phase_row["period"])

        # Mostreig uniforme de n_frames dins el rang
        if frame_end - frame_start + 1 <= n_frames:
            frames = list(range(frame_start, frame_end + 1))
        else:
            frames = np.linspace(frame_start, frame_end, n_frames, dtype=int).tolist()

        n = len(frames)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        if n == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = np.array(axes).reshape(nrows, ncols)

        # Config visuals
        team_ids = sorted({tid for tid in self.get_player_team_map().values() if tid is not None})
        team_colors = {team_ids[0]: "#1976D2", team_ids[1]: "#D32F2F"} if len(team_ids) >= 2 else {}

        edge_styles = {
            "passing_option":    {"color": "#FFEB3B", "ls": "--", "lw": 1.2, "alpha": 0.8},  # groc
            "off_ball_run":      {"color": "#4CAF50", "ls": "-",  "lw": 1.5, "alpha": 0.85}, # verd
            "on_ball_engagement":{"color": "#F44336", "ls": "-",  "lw": 2.0, "alpha": 0.9},  # vermell
            "player_possession": {"color": "#FFFFFF", "ls": "-",  "lw": 2.0, "alpha": 1.0},  # blanc
        }

        phase_type = phase_row.get("team_in_possession_phase_type", "?")
        attacking_side = phase_row.get("attacking_side", "?")

        for i, frame in enumerate(frames):
            ax = axes[i // ncols, i % ncols]
            self._draw_pitch(ax)

            data = self.get_frame_positions(int(frame))
            if data is None:
                ax.text(0, 0, f"Frame {frame}\nsense dades", ha="center", color="white")
                continue

            # Dibuixar arestes primer (sota dels nodes)
            edges = self.build_frame_graph(int(frame), period)
            for ed in edges:
                style = edge_styles.get(ed["event_type"], {"color": "gray", "ls": "-", "lw": 1.0, "alpha": 0.5})
                src_pos = data["players"].get(ed["src"]) if isinstance(ed["src"], int) else data["ball"]
                if isinstance(ed["dst"], int):
                    dst_pos = data["players"].get(ed["dst"])
                else:
                    dst_pos = data["ball"]

                if src_pos is None or dst_pos is None:
                    continue

                src_xy = src_pos[:2]
                dst_xy = dst_pos[:2]

                ax.plot([src_xy[0], dst_xy[0]], [src_xy[1], dst_xy[1]],
                        color=style["color"], linestyle=style["ls"],
                        lw=style["lw"], alpha=style["alpha"], zorder=2)

            # Dibuixar jugadors
            for pid, (x, y, tid) in data["players"].items():
                color = team_colors.get(tid, "#9E9E9E")
                edge_color = "#FFEB3B" if pid == data["possession_player_id"] else "black"
                edge_w = 2.2 if pid == data["possession_player_id"] else 0.8
                circ = Circle((x, y), 0.8, facecolor=color, edgecolor=edge_color,
                              linewidth=edge_w, zorder=4)
                ax.add_patch(circ)

            # Dibuixar pilota
            if data["ball"] is not None:
                bx, by = data["ball"]
                ax.add_patch(Circle((bx, by), 0.6, facecolor="white",
                                    edgecolor="black", linewidth=1.0, zorder=5))

            # Títol
            t_relative = (frame - frame_start) / 10.0
            ax.set_title(f"frame {frame}  (t={t_relative:.1f}s)",
                         fontsize=10, color="white", pad=3)

        # Amagar subplots no usats
        for i in range(n, nrows * ncols):
            axes[i // ncols, i % ncols].axis("off")

        # Llegenda d'arestes
        legend_elements = [
            plt.Line2D([0], [0], color=s["color"], ls=s["ls"], lw=s["lw"],
                       label=name.replace("_", " "))
            for name, s in edge_styles.items()
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=4,
                   facecolor="#1a1a1a", labelcolor="white", fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))

        fig.patch.set_facecolor("#1a1a1a")
        fig.suptitle(
            f"Phase  period={period}  frames=[{frame_start}, {frame_end}]  "
            f"durada={(frame_end - frame_start) / 10:.1f}s  "
            f"type={phase_type}  attacking={attacking_side}",
            fontsize=12, color="white", y=0.995,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if save_path is not None:
            plt.savefig(save_path, dpi=140, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Guardat: {save_path}")
        plt.show()

    @staticmethod
    def _draw_pitch(ax, length: float = 105.0, width: float = 68.0) -> None:
        """Dibuixa un camp de futbol estàndard sobre l'eix donat."""
        ax.set_facecolor("#2e8b57")
        ax.set_xlim(-length / 2 - 3, length / 2 + 3)
        ax.set_ylim(-width / 2 - 3, width / 2 + 3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        hl, hw = length / 2, width / 2
        ax.plot([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw], "w-", lw=1.2)
        ax.plot([0, 0], [-hw, hw], "w-", lw=1.2)

        import matplotlib.pyplot as plt
        ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="white", lw=1.2))
        for s in [-1, 1]:
            ax.plot([s * hl, s * (hl - 16.5), s * (hl - 16.5), s * hl],
                    [-20.16, -20.16, 20.16, 20.16], "w-", lw=1.2)
            ax.plot([s * hl, s * (hl - 5.5), s * (hl - 5.5), s * hl],
                    [-9.16, -9.16, 9.16, 9.16], "w-", lw=1.2)

    def sample_phase_targets(
        self,
        n_samples: int = 1000,
        fps: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per a n_samples frames aleatoris del partit, calcula:
          - delta_restant: segons fins al final de la fase actual
          - delta_proper:  segons fins a l'inici de la fase següent

        Frames que cauen en gaps entre fases o després de l'última fase del
        període es descarten.

        Retorna:
            (delta_restant, delta_proper) en segons, com a np.ndarray.
        """
        if self.phases.empty:
            raise RuntimeError("Phases no carregades. Crida load_data() primer.")

        rng = np.random.default_rng(seed)

        phases_by_period: Dict[int, pd.DataFrame] = {
            int(p): self.phases[self.phases["period"] == p]
                        .sort_values("frame_start")
                        .reset_index(drop=True)
            for p in self.phases["period"].unique()
        }

        min_frame = int(self.phases["frame_start"].min())
        max_frame = int(self.phases["frame_end"].max())

        deltas_restant: List[float] = []
        deltas_proper:  List[float] = []

        attempts = 0
        max_attempts = n_samples * 10

        while len(deltas_restant) < n_samples and attempts < max_attempts:
            attempts += 1
            frame = int(rng.integers(min_frame, max_frame + 1))

            for _, df_p in phases_by_period.items():
                hit = df_p[
                    (df_p["frame_start"] <= frame) & (df_p["frame_end"] >= frame)
                ]
                if hit.empty:
                    continue

                current_idx = hit.index[0]
                current     = df_p.iloc[current_idx]

                if current_idx + 1 >= len(df_p):
                    break  # última fase del període

                next_phase = df_p.iloc[current_idx + 1]

                d_restant = (int(current["frame_end"])     - frame) / fps
                d_proper  = (int(next_phase["frame_start"]) - frame) / fps

                deltas_restant.append(d_restant)
                deltas_proper.append(d_proper)
                break

        return np.array(deltas_restant), np.array(deltas_proper)


def plot_delta_distribution(
    deltas: np.ndarray,
    name: str,
    output_path: Optional[str | Path] = None,
) -> None:
    """
    Per a un array de Δt (en segons), genera una figura amb 4 subplots:
      1. Histograma de Δt + PDF log-normal ajustada
      2. Histograma de log(Δt) + PDF Gaussiana ajustada
      3. Q-Q plot de log(Δt) vs Gaussiana (linealitat → bon fit log-normal)
      4. Box plot amb estadístiques bàsiques

    Si log(Δt) sembla Gaussià → la distribució log-normal és apropiada.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    deltas = deltas[deltas > 0]   # log indefinit a 0
    log_deltas = np.log(deltas)

    # Ajustar log-normal als deltas (equivalent a Gaussiana sobre log)
    mu, sigma = log_deltas.mean(), log_deltas.std()

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"Distribució de {name}   "
        f"(n={len(deltas)},  median={np.median(deltas):.2f}s,  "
        f"mean={deltas.mean():.2f}s,  max={deltas.max():.1f}s)",
        fontsize=13, fontweight="bold",
    )

    # 1. Histograma de Δt + PDF log-normal
    ax = axes[0]
    ax.hist(deltas, bins=60, density=True, alpha=0.6, color="#1976D2", edgecolor="white")
    x = np.linspace(deltas.min() + 1e-3, deltas.max(), 500)
    pdf = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    ax.plot(x, pdf, color="#D32F2F", lw=2, label=f"Log-normal (μ={mu:.2f}, σ={sigma:.2f})")
    ax.set_title(f"{name}  —  espai original")
    ax.set_xlabel("Δt (s)")
    ax.set_ylabel("densitat")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Histograma de log(Δt) + PDF Gaussiana
    ax = axes[1]
    ax.hist(log_deltas, bins=60, density=True, alpha=0.6, color="#388E3C", edgecolor="white")
    x = np.linspace(log_deltas.min(), log_deltas.max(), 500)
    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    ax.plot(x, pdf, color="#D32F2F", lw=2, label=f"Gaussiana (μ={mu:.2f}, σ={sigma:.2f})")
    ax.set_title(f"log({name})  —  ha de semblar Gaussià")
    ax.set_xlabel("log(Δt)")
    ax.set_ylabel("densitat")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Q-Q plot de log(Δt) vs Gaussiana
    ax = axes[2]
    stats.probplot(log_deltas, dist="norm", plot=ax)
    ax.set_title("Q-Q plot  —  linealitat = bon fit log-normal")
    ax.grid(alpha=0.3)

    # 4. Box plot
    ax = axes[3]
    ax.boxplot(deltas, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#FFA726", alpha=0.7))
    ax.set_title("Box plot")
    ax.set_ylabel("Δt (s)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=140, bbox_inches="tight")
        print(f"Guardat: {output_path}")
    plt.show()



def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta_distribution', action='store_true')
    parser.add_argument('--plot_phase_graphs', action='store_true')
    parser.add_argument('--sample_match', type = str)
    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    import os

    base = Path(r"c:\Users\Eloy Mercader\OneDrive - EDISA SISTEMAS DE INFORMACION, SA"
                r"\Escritorio\eloi edisa\projectes\tracking\opendata\data\matches")

    all_restant: List[float] = []
    all_proper:  List[float] = []

    for match_id in os.listdir(base):
        match_dir = base / match_id
        if not match_dir.is_dir():
            continue

        tracking = match_dir / f"{match_id}_tracking_extrapolated.jsonl"
        events   = match_dir / f"{match_id}_dynamic_events.csv"
        phases   = match_dir / f"{match_id}_phases_of_play.csv"

        if not (events.exists() and phases.exists()):
            continue

        builder = MatchGraphBuilder(tracking, events, phases)
        # Per a la distribució de Δt no cal carregar el tracking sencer
        builder._load_dynamic_events()
        builder._load_phases()

        try:
            d_r, d_p = builder.sample_phase_targets(n_samples=2000, seed=42)
            all_restant.extend(d_r.tolist())
            all_proper.extend(d_p.tolist())
            print(f"Match {match_id}: {len(d_r)} mostres")
        except Exception as e:
            print(f"Match {match_id} omès: {e}")

    out_dir = Path(__file__).resolve().parent.parent
    args = define_parser()
    print(f'args delta_distribution = {args.delta_distribution}  plot_phase_graphs = {args.plot_phase_graphs}  sample_match = {args.sample_match}')
    if args.delta_distribution:
        plot_delta_distribution(
            np.array(all_restant),
            "Δt_restant (final fase actual)",
            out_dir / "dist_delta_restant.png",
        )
        plot_delta_distribution(
            np.array(all_proper),
            "Δt_proper (inici fase següent)",
            out_dir / "dist_delta_proper.png",
        )
    

    # ── Visualització del graf d'esdeveniments per a una fase de mostra ──
    if args.plot_phase_graphs:
        sample_match = args.sample_match 
        match_dir = base / sample_match
        builder = MatchGraphBuilder(
            match_dir / f"{sample_match}_tracking_extrapolated.jsonl",
            match_dir / f"{sample_match}_dynamic_events.csv",
            match_dir / f"{sample_match}_phases_of_play.csv",
        )
        builder.load_data()

        # Triem una fase amb durada raonable (entre 4 i 10 segons)
        phases = builder.phases
        candidates = phases[(phases["duration"] >= 4) & (phases["duration"] <= 10)]
        sample_phase = candidates.iloc[3]
        print(f"\nFase mostra: type={sample_phase['team_in_possession_phase_type']}  "
            f"frames=[{sample_phase['frame_start']}, {sample_phase['frame_end']}]  "
            f"durada={sample_phase['duration']}s")

        builder.plot_phase_graphs(
            sample_phase, n_frames=9,
            save_path=out_dir / "phase_graph_sample.png",
        )