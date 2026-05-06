"""
Smoke test ràpid per a multi_head_model/dataset.py.
Carrega 1 partit, obté algunes mostres i imprimeix shapes + estadístiques bàsiques.
Executar des de l'arrel del projecte:  python -m multi_head_model._smoke_test
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from multi_head_model.dataset import PhaseDataset
from multi_head_model.constants import (
    T_MAX, N_NODES, N_NODE_NUMERIC_FEAT, N_CONTEXT_FEAT, N_EVENT_TYPES,
    PHASE_TYPES, IDX_TO_PHASE_TYPE, IDX_TO_POSITION,
)

# ── 1. Localitzar un partit ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
MATCHES_DIR = ROOT / "opendata" / "data" / "matches"
match_dirs = sorted(p for p in MATCHES_DIR.iterdir() if p.is_dir())[:1]
print(f"[i] Partit usat: {match_dirs[0].name}")

# ── 2. Construir dataset ────────────────────────────────────────────────────
t0 = time.time()
ds = PhaseDataset(match_dirs, random_t=True)
print(f"[i] Dataset construït en {time.time() - t0:.2f}s — N={len(ds)} mostres")

# Versió deterministe per comprovar reproductibilitat
ds_val = PhaseDataset(match_dirs, random_t=False, val_t_fraction=0.75)

# ── 3. Carregar la primera mostra ───────────────────────────────────────────
t0 = time.time()
sample = ds[0]
print(f"[i] ds[0] obtingut en {time.time() - t0:.2f}s (1ª crida; carrega tracking)")

t0 = time.time()
sample2 = ds[0]
print(f"[i] ds[0] segon cop:  {time.time() - t0:.2f}s (cache calent)")

# ── 4. Comprovar shapes i tipus ─────────────────────────────────────────────
expected = {
    "node_numeric":     (T_MAX, N_NODES, N_NODE_NUMERIC_FEAT),
    "position_idx":     (N_NODES,),
    "context":          (T_MAX, N_CONTEXT_FEAT),
    "adj_per_relation": (N_EVENT_TYPES, T_MAX, N_NODES, N_NODES),
    "state_final":      (N_NODES, 2),
    "frame_mask":       (T_MAX,),
}
print("\n[Shapes]")
for k, exp in expected.items():
    got = tuple(sample[k].shape)
    ok  = "OK " if got == exp else "FAIL"
    print(f"  {ok}  {k:18s} {got}  (esperat {exp})")

print("\n[Escalars i tipus]")
for k in ("phase_target", "delta_restant", "delta_proper", "is_long_pause", "boundary_idx"):
    v = sample[k]
    print(f"  {k:18s} dtype={str(v.dtype):15s} value={v.item():.4f}")

# ── 5. Validacions semàntiques ──────────────────────────────────────────────
print("\n[Validacions semàntiques]")
mask = sample["frame_mask"]
T_real = int(mask.sum().item())
print(f"  frames vàlids:     {T_real} / {T_MAX}")
print(f"  últim frame vàlid: posició {T_real - 1}")
print(f"  boundary_idx:      {sample['boundary_idx'].item()}")

pt = int(sample["phase_target"].item())
print(f"  phase_target:      {pt} → '{IDX_TO_PHASE_TYPE[pt]}'")

# Position one-hot per slot 22 (pilota)
ball_pos = int(sample["position_idx"][22].item())
print(f"  position_idx[BALL]: {ball_pos} → '{IDX_TO_POSITION[ball_pos]}'")

# Tipus de fase prevista per cada slot de jugadors
slot_positions = [IDX_TO_POSITION[int(sample['position_idx'][i].item())] for i in range(22)]
print(f"  posicions home (0-10):  {slot_positions[:11]}")
print(f"  posicions away (11-21): {slot_positions[11:]}")

# Stats de coordenades reals dels frames vàlids
nn = sample["node_numeric"][:T_real]            # [T, N, 8]
xs = nn[..., 0]
ys = nn[..., 1]
vx = nn[..., 2]
vy = nn[..., 3]
print(f"  x range (m):    [{xs.min().item():+.2f}, {xs.max().item():+.2f}]")
print(f"  y range (m):    [{ys.min().item():+.2f}, {ys.max().item():+.2f}]")
print(f"  |v| max  (m/s): {torch.sqrt(vx**2 + vy**2).max().item():.2f}")

# Adjacency: nº arestes per relació
adj = sample["adj_per_relation"][:, :T_real]    # [R, T, N, N]
EVENT_NAMES = ["passing", "off_ball", "engagement", "possession", "spatial"]
for r in range(N_EVENT_TYPES):
    a = adj[r]                                  # [T, N, N]
    if r < 4:
        n_edges = int(a.sum().item())
        print(f"  adj[r={r} {EVENT_NAMES[r]:11s}] arestes binàries totals: {n_edges}")
    else:
        nz = (a > 0).float().mean().item()
        rel = (a > 0.05).float().mean().item()
        print(f"  adj[r={r} {EVENT_NAMES[r]:11s}] mitjana={a.mean().item():.3f}, "
              f"max={a.max().item():.3f}, "
              f"%>0={nz*100:.1f}, %>0.05={rel*100:.1f}")
        # Comprovar simetria a un frame concret
        a0 = a[0]
        sym_err = (a0 - a0.T).abs().max().item()
        diag = a0.diag().abs().max().item()
        print(f"             frame[0]: simetria_err={sym_err:.2e}, diagonal_max={diag:.2e}")

# is_ball / team_idx per slot
ib = nn[0, :, 6]
ti = nn[0, :, 7]
print(f"  is_ball (slot 22): {ib[22].item():.0f} (esperat 1)")
print(f"  team_idx home(0):  set={sorted(set(ti[:11].tolist()))}  (esperat {{0.0}})")
print(f"  team_idx away(1):  set={sorted(set(ti[11:22].tolist()))} (esperat {{1.0}})")

# ── 6. Reproductibilitat val vs aleatorietat train ─────────────────────────
v1 = ds_val[0]
v2 = ds_val[0]
print("\n[Reproductibilitat val (random_t=False)]")
mt1 = int(v1["frame_mask"].sum().item())
mt2 = int(v2["frame_mask"].sum().item())
print(f"  T_real cop 1 i cop 2: {mt1}, {mt2}  → idèntic? {mt1 == mt2}")
print(f"  delta_restant:        {v1['delta_restant'].item():.3f}, {v2['delta_restant'].item():.3f}")

print("\n[Aleatorietat train (random_t=True)]")
ts = []
for _ in range(5):
    s = ds[0]
    ts.append(int(s["frame_mask"].sum().item()))
print(f"  T_real en 5 crides:  {ts}  → varia? {len(set(ts)) > 1}")

print("\n[OK] Smoke test del dataset acabat sense errors.")

# ── 7. Encoder sobre un batch real ──────────────────────────────────────────
from multi_head_model.encoder import SpatioTemporalEncoder
from multi_head_model.constants import D_MODEL

print("\n[Encoder amb dades reals]")
enc = SpatioTemporalEncoder()
print(f"  paràmetres entrenables: {enc.count_parameters():,}")

# Construïm un batch de mida 2 amb dues mostres del dataset
batch_keys = ["node_numeric", "position_idx", "context", "adj_per_relation", "frame_mask"]
items = [ds_val[i] for i in range(2)]   # val per estabilitat (random_t=False)
batch = {k: torch.stack([it[k] for it in items], dim=0) for k in batch_keys}
print(f"  batch shapes: " + ", ".join(f"{k}={tuple(batch[k].shape)}" for k in batch_keys))

t0 = time.time()
with torch.no_grad():
    h = enc(batch)
dt = time.time() - t0
print(f"  forward time:  {dt*1000:.1f} ms")
print(f"  h shape:       {tuple(h.shape)}  (esperat (2, {D_MODEL}))")
print(f"  h finite?      {torch.isfinite(h).all().item()}")
print(f"  h norms:       {[round(h[i].norm().item(), 3) for i in range(2)]}")
print(f"  h mean per d:  mitjana={h.mean().item():+.3f}, std={h.std().item():.3f}")

print("\n[OK] Smoke test complet (dataset + encoder).")
