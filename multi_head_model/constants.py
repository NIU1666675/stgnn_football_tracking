"""
Constants i hiperparàmetres per al model multi-head de predicció de fases.

Hi ha quatre seccions:
  1. Dades  : nodes, fps, paths
  2. Mostreig : finestra d'entrada, stride, sostres
  3. Model  : dimensions ocultes, capçaleres, profunditat
  4. Targets : tipus de fase, capping de Δt, tipus d'event del graf
"""

import os

# ── 1. DADES ────────────────────────────────────────────────────────────────

FPS         = 10
N_PLAYERS   = 22                  # 11 per equip
N_NODES     = N_PLAYERS + 1       # + pilota = 23
N_FEAT_POS  = 2                   # x, y per node (coordenades base)

# Paths principals
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, "opendata", "data", "matches")
OUTPUT_DIR  = os.path.join(ROOT_DIR, "multi_head_data")


# ── 2. MOSTREIG ─────────────────────────────────────────────────────────────

# Stride sobre frames originals (10 fps). Stride=3 → ~3.3 fps efectius.
# S'aplica tant a la fase prèvia com a la fase actual.
STRIDE          = 3

# Sostre de longitud temporal d'entrada (en frames mostrejats després de stride).
# Si la combinació prèvia + actual el supera, es trunca des del començament
# de la fase prèvia. La fase actual mai es trunca.
T_MAX           = 200

# Capping per a Δt_proper (segons). Mostres amb Δt_proper > 30s
# es marquen com a "long pause" i el cap log-normal s'ignora.
DELTA_PROPER_CAP = 30.0

# Longitud màxima de la trajectòria predita pel TrajectoryHead.
# Cadència = stride/FPS = 0.3 s. T_PRED_MAX = 100 → fins a 30 s de futur,
# alineat amb DELTA_PROPER_CAP. Frames més enllà de Δt_proper queden
# emmascarats per `target_mask` i no contribueixen a la pèrdua.
T_PRED_MAX = 100

# Partició de partits (per partit, no per seqüència, per evitar fuites).
N_TRAIN_MATCHES = 8
N_VAL_MATCHES   = 1
N_TEST_MATCHES  = 1


# ── 3. MODEL ────────────────────────────────────────────────────────────────

D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 3
DROPOUT    = 0.1
ACTIVATION = "gelu"
NORM       = "layernorm_pre"


# ── 4. TARGETS ──────────────────────────────────────────────────────────────

# 8 fases tàctiques + stoppage (per als gaps entre fases).
# L'ordre fixa l'index de classe del cap d'event.
PHASE_TYPES = [
    "build_up",
    "create",
    "finish",
    "direct",
    "chaotic",
    "transition",
    "quick_break",
    "set_play",
    "stoppage",
]
N_PHASE_CLASSES = len(PHASE_TYPES)        # 9

PHASE_TYPE_TO_IDX = {p: i for i, p in enumerate(PHASE_TYPES)}
IDX_TO_PHASE_TYPE = {i: p for i, p in enumerate(PHASE_TYPES)}
STOPPAGE_IDX      = PHASE_TYPE_TO_IDX["stoppage"]


# Tipus de relacions del graf (R relacions per a la R-GCN).
# Les 4 primeres provenen del CSV dynamic_events. La 5a, "spatial_proximity",
# és purament geomètrica (bidireccional, ponderada amb gaussiana de la
# distància entre nodes), calculada per frame al dataset.
EVENT_TYPES = [
    "passing_option",       # cooperatiu, futur potencial
    "off_ball_run",         # cooperatiu, cinètic
    "on_ball_engagement",   # antagònic, actual
    "player_possession",    # estructural (jugador ↔ pilota)
    "spatial_proximity",    # geomètric, bidireccional, ponderat
]
N_EVENT_TYPES = len(EVENT_TYPES)

EVENT_TYPE_TO_IDX = {e: i for i, e in enumerate(EVENT_TYPES)}
SPATIAL_PROXIMITY_IDX = EVENT_TYPE_TO_IDX["spatial_proximity"]

# Sigma (metres) per al kernel gaussià de la relació spatial_proximity.
# σ ≈ 10m: capta interaccions tàctiques curtes (suport, pressing, marcatge);
# parelles a >25m queden pràcticament desconnectades (pes < 0.05).
SPATIAL_SIGMA = 10.0


# ── 5. FEATURES ─────────────────────────────────────────────────────────────

# Part numèrica de les features per node (al input):
#   [x, y, vx, vy, dx_ball, dy_ball, is_ball, team_idx]
# is_ball:  1 si el node és la pilota
# team_idx: 0 (home), 1 (away). Per la pilota, valor irrellevant (is_ball=1)
#
# La possessió ja està codificada implícitament com a aresta player_possession
# del graf — no cal duplicar-ho com a feature de node.
N_NODE_NUMERIC_FEAT = 8

# Part categòrica per node: índex de posició tàctica del jugador.
# Vocabulari de posicions detectades al dataset complet (19) + BALL + UNK.
PLAYER_POSITIONS = [
    "GK",                                     # porter
    "LCB", "RCB", "CB",                       # centrals
    "LB",  "RB",  "LWB", "RWB",              # laterals
    "LDM", "RDM", "DM",                       # migcampistes defensius
    "LM",  "RM",  "AM",                       # migcampistes
    "LW",  "RW",                              # extrems
    "LF",  "RF",  "CF",                       # davanters
    "BALL",                                   # pilota (categoria especial)
    "UNK",                                    # segur per posicions noves a test
]
POSITION_VOCAB_SIZE  = len(PLAYER_POSITIONS)  # 21
POSITION_EMBED_DIM   = 8                      # dimensió de l'embedding aprés

POSITION_TO_IDX = {p: i for i, p in enumerate(PLAYER_POSITIONS)}
IDX_TO_POSITION = {i: p for i, p in enumerate(PLAYER_POSITIONS)}
BALL_POS_IDX    = POSITION_TO_IDX["BALL"]
UNK_POS_IDX     = POSITION_TO_IDX["UNK"]

# Features contextuals per frame (broadcast a tots els nodes):
#   [match_time_norm, period, score_diff_norm, attacking_LtR,
#    team_a_in_poss, frames_since_phase_start_norm, is_current_phase,
#    one_hot_phase_type (9 classes)]
# El camp `is_current_phase` (binari) diu si el frame pertany a curr (1) o no (0).
# El one-hot `phase_type` indica el tipus de fase a la qual pertany el frame
# (la prèvia si està a prev; la actual si està a curr; stoppage si està al gap).
N_CONTEXT_FEAT = 7 + N_PHASE_CLASSES         # 16

# Total de canals d'entrada per node DESPRÉS de l'embedding de posició
# = N_NODE_NUMERIC_FEAT + POSITION_EMBED_DIM + N_CONTEXT_FEAT
N_FEAT_INPUT_AFTER_EMB = N_NODE_NUMERIC_FEAT + POSITION_EMBED_DIM + N_CONTEXT_FEAT  # 32


# Normalització per match_time. La part més llarga d'un partit és ~5400 s.
MATCH_TIME_MAX_S = 7200.0


# ── 6. ENTRENAMENT ──────────────────────────────────────────────────────────

BATCH_SIZE     = 32
EPOCHS         = 50
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 5.0
PATIENCE_ES    = 10
PATIENCE_LR    = 5
LR_FACTOR      = 0.5

# Pesos relatius de les pèrdues. Inicialment fixos; es poden ajustar
# o substituir per 'uncertainty weighting' (Kendall et al. 2018).
LAMBDA_EVENT   = 1.0
LAMBDA_TIME    = 1.0
LAMBDA_PAUSE   = 0.5
LAMBDA_STATE   = 1.0
