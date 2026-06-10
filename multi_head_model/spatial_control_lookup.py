"""
Loader dels artefactes de control d'espai precalculats per
`precompute_spatial_control.py`.

Carrega, sota demanda i amb cache LRU, els fitxers `{match_id}_{mode}.pkl` i
ofereix consulta O(1) per (match_id, frame):
  - àrea de control de cada jugador (fracció del camp), indexada per player_id
  - control del local a cada terç (3 valors per al vector de context)

El mode ('voronoi' o 'dominant') es fixa a la construcció: cada execució
d'entrenament fa servir un sol mode.
"""

from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from .constants import OUTPUT_DIR, N_SPATIAL_CONTEXT_FEAT

ART_SUBDIR = "spatial_control"
_ZERO_THIRDS = np.zeros(N_SPATIAL_CONTEXT_FEAT, dtype=np.float32)


class SpatialControlLookup:
    """
    Consulta dels artefactes de control d'espai per a un mode concret.

    Args:
      mode:       'voronoi' o 'dominant'.
      art_dir:    directori amb els .pkl (per defecte multi_head_data/spatial_control).
      cache_size: nombre de partits a mantenir en memòria (LRU).
    """

    def __init__(
        self,
        mode: str,
        art_dir: Optional[str] = None,
        cache_size: int = 10,
    ) -> None:
        self.mode = mode
        self.art_dir = Path(art_dir) if art_dir is not None else (
            Path(OUTPUT_DIR) / ART_SUBDIR
        )
        self.max_size = max(1, int(cache_size))
        self._cache: "OrderedDict[str, Dict[int, dict]]" = OrderedDict()

    def _validate_frames(
        self,
        frames: Dict[int, dict],
        match_id: str,
        path: Path,
    ) -> None:
        """Valida que les features carregades siguin finites i dins del rang."""
        for frame, rec in frames.items():
            if "area_frac" not in rec or "third_home" not in rec:
                raise ValueError(
                    f"Artefacte incomplet a {path}: frame {frame} sense "
                    "'area_frac' o 'third_home'. Regenera el precòmput."
                )

            areas = np.asarray(list(rec["area_frac"].values()), dtype=np.float64)
            thirds = np.asarray(rec["third_home"], dtype=np.float64)

            if areas.size and not np.all(np.isfinite(areas)):
                raise ValueError(
                    f"Control d'espai no finit a {path}: match={match_id}, "
                    f"frame={frame}, camp=area_frac. Regenera l'artefacte."
                )
            if areas.size and (areas.min() < -1e-6 or areas.max() > 1.0 + 1e-6):
                raise ValueError(
                    f"Control d'espai fora de [0,1] a {path}: "
                    f"match={match_id}, frame={frame}, camp=area_frac, "
                    f"rang=[{areas.min():.6g}, {areas.max():.6g}]."
                )
            if thirds.shape != (N_SPATIAL_CONTEXT_FEAT,):
                raise ValueError(
                    f"Forma incorrecta a {path}: match={match_id}, frame={frame}, "
                    f"third_home té forma {thirds.shape}, esperada "
                    f"({N_SPATIAL_CONTEXT_FEAT},)."
                )
            if not np.all(np.isfinite(thirds)):
                raise ValueError(
                    f"Control d'espai no finit a {path}: match={match_id}, "
                    f"frame={frame}, camp=third_home. Regenera l'artefacte."
                )
            if thirds.min() < -1e-6 or thirds.max() > 1.0 + 1e-6:
                raise ValueError(
                    f"Control d'espai fora de [0,1] a {path}: "
                    f"match={match_id}, frame={frame}, camp=third_home, "
                    f"rang=[{thirds.min():.6g}, {thirds.max():.6g}]."
                )

    def _get_match(self, match_id: str) -> Dict[int, dict]:
        """Carrega (o recupera del cache) el dict de frames d'un partit."""
        if match_id in self._cache:
            self._cache.move_to_end(match_id)
            return self._cache[match_id]

        path = self.art_dir / f"{match_id}_{self.mode}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Artefacte de control d'espai no trobat: {path}. "
                f"Executa `python -m multi_head_model.precompute_spatial_control "
                f"--mode {self.mode}` primer."
            )
        with path.open("rb") as fh:
            artifact = pickle.load(fh)
        if artifact.get("match_id") != match_id:
            raise ValueError(
                f"Artefacte inconsistent a {path}: match_id="
                f"{artifact.get('match_id')!r}, esperat {match_id!r}."
            )
        if artifact.get("mode") != self.mode:
            raise ValueError(
                f"Artefacte inconsistent a {path}: mode="
                f"{artifact.get('mode')!r}, esperat {self.mode!r}."
            )
        frames = artifact["frames"]
        self._validate_frames(frames, match_id, path)

        self._cache[match_id] = frames
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return frames

    def frame_record(self, match_id: str, frame: int) -> Optional[dict]:
        """Registre de control d'un frame, o None si no existeix."""
        return self._get_match(match_id).get(int(frame))

    def warm_cache(self, match_ids: Iterable[str]) -> None:
        """Carrega i valida els artefactes dels partits indicats."""
        for match_id in match_ids:
            self._get_match(str(match_id))

    def areas(self, match_id: str, frame: int) -> Dict[int, float]:
        """
        Àrees de control (fracció del camp) per player_id en un frame.
        Diccionari buit si el frame no és a l'artefacte.
        """
        rec = self.frame_record(match_id, frame)
        return rec["area_frac"] if rec is not None else {}

    def thirds(self, match_id: str, frame: int) -> np.ndarray:
        """
        Control del local als tres terços en un frame, [3].
        Vector de zeros si el frame no és a l'artefacte.
        """
        rec = self.frame_record(match_id, frame)
        if rec is None:
            return _ZERO_THIRDS
        return np.asarray(rec["third_home"], dtype=np.float32)
