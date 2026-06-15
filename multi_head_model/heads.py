"""
Heads del model multi-head.

A partir del vector global h ∈ R^D produït per l'encoder, la cascada és:

    h ──→ EventHead ──→ logits [B, 9]
                          │
                          ↓ softmax
                    EventEmbedding ──→ emb_event [B, E]
                                          ↓
              h_cond = concat(h, emb_event)  ∈ [B, D + E]
                                          ↓
       ┌──── TimeHead   ──→ (μ, logσ) per log-normal sobre Δt_proper
       ├──── PauseHead  ──→ logit binari (is_long_pause)
       └──── StateHead  ──→ desplaçament Δpos per node, sumat a current_pos
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    D_MODEL,
    DROPOUT,
    FPS,
    N_NODES,
    N_PHASE_CLASSES,
    N_TIME_MIXTURE,
    STRIDE,
    T_PRED_MAX,
    TRAJ_LOG_SIGMA_MIN,
    TRAJ_LOG_SIGMA_MAX,
    TRAJ_LOG_SIGMA_INIT,
)


# ── Configuració ────────────────────────────────────────────────────────────

EVENT_EMB_DIM   = 16          # dimensió de l'embedding soft del tipus d'event
LOG_SIGMA_MIN   = -3.0        # σ ≈ 0.05  (1.6 % de variació)
LOG_SIGMA_MAX   =  3.0        # σ ≈ 20    (saturació superior)


# ── 1. EventHead ────────────────────────────────────────────────────────────

class EventHead(nn.Module):
    """Classificació del tipus de fase posterior (9 classes amb stoppage)."""

    def __init__(self, d_in: int = D_MODEL, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.mlp  = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, N_PHASE_CLASSES),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, D]  →  logits: [B, N_PHASE_CLASSES]"""
        return self.mlp(self.norm(h))


# ── 2. EventEmbedding ───────────────────────────────────────────────────────

class EventEmbedding(nn.Module):
    """
    Embedding soft del tipus d'event a partir dels logits. Utilitza les
    probabilitats softmax (no argmax) perquè el gradient flueixi durant
    l'entrenament (i.e. condicionament suau).

        probs = softmax(logits)        [B, N_classes]
        emb   = probs @ E              [B, emb_dim]
    """

    def __init__(self, emb_dim: int = EVENT_EMB_DIM) -> None:
        super().__init__()
        self.emb = nn.Parameter(torch.empty(N_PHASE_CLASSES, emb_dim))
        nn.init.xavier_uniform_(self.emb)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)   # [B, N_classes]
        return probs @ self.emb             # [B, emb_dim]


# ── 3. TimeHead ─────────────────────────────────────────────────────────────
# TIMEHEAD_DISABLED (Opció C): el TimeHead s'ha eliminat de la cascada.
# La TrajectoryHead ja no rep `delta_t` com a input (depèn només de `dt_k`
# local de cada step). Per recuperar-lo, descomentar aquest bloc i les
# referències marcades amb `TIMEHEAD_DISABLED` a `model.py`, `losses.py` i
# `train.py`.
#
# class TimeHead(nn.Module):
#     """
#     Prediu una *mixture* de K log-normals sobre Δt_proper:
#
#         p(y) = Σ_k  π_k · LogN(y; μ_k, σ_k)
#
#     Output (per cada mostra del batch):
#       - mix_logits : [B, K]  (es passa per softmax per obtenir π_k)
#       - μ          : [B, K]
#       - log σ      : [B, K]  (clipped per estabilitat)
#
#     Inicialització intel·ligent: el biaix de la capa final es prepara perquè
#     el component 0 tendeixi a Δt curt (~2 s) i el component 1 a Δt llarg
#     (~10 s), evitant així el col·lapse a una sola distribució a l'inici.
#
#     L'aprenentatge fa servir la NLL de la mixture (vegis `losses.py`).
#     """
#
#     def __init__(
#         self,
#         d_in: int,
#         dropout: float = DROPOUT,
#         n_components: int = N_TIME_MIXTURE,
#     ) -> None:
#         super().__init__()
#         self.K = n_components
#         self.norm = nn.LayerNorm(d_in)
#         # Sortida: 3 valors per component (mix_logit, μ, log σ)
#         self.mlp = nn.Sequential(
#             nn.Linear(d_in, d_in),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_in, 3 * n_components),
#         )
#         self._init_biases()
#
#     def _init_biases(self) -> None:
#         """
#         Inicialitza el biaix de la darrera Linear perquè els K components
#         partim ja diferenciats:
#           μ_k     = log(linspace(2 s, 10 s, K))
#           log σ_k = log(0.7)   (σ ≈ 0.7, cua moderada)
#           mix_logit_k = 0      (π_k = 1/K uniforme)
#         """
#         K = self.K
#         mu_init = torch.log(torch.linspace(2.0, 10.0, K))
#         log_sigma_init = torch.full((K,), float(torch.log(torch.tensor(0.7))))
#         mix_init = torch.zeros(K)
#
#         # Layout del biaix: [3·K]  ordre intern (mix, μ, σ) per component
#         # Al forward fem .view(*, 3, K) → bias matrix [3, K]:
#         last = self.mlp[-1]   # darrer Linear
#         with torch.no_grad():
#             last.bias.zero_()
#             bias_view = last.bias.view(3, K)
#             bias_view[0] = mix_init
#             bias_view[1] = mu_init
#             bias_view[2] = log_sigma_init
#
#     def forward(
#         self,
#         h_cond: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         out = self.mlp(self.norm(h_cond))            # [B, 3·K]
#         out = out.view(*out.shape[:-1], 3, self.K)   # [B, 3, K]
#         mix_logits = out[..., 0, :]                  # [B, K]
#         mu         = out[..., 1, :]                  # [B, K]
#         log_sigma  = out[..., 2, :].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
#         return mix_logits, mu, log_sigma


# ── 4. PauseHead ────────────────────────────────────────────────────────────

class PauseHead(nn.Module):
    """Prediu si Δt_proper > cap (long pause / stoppage). Output: logit per BCE."""

    def __init__(self, d_in: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.mlp  = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 2, 1),
        )

    def forward(self, h_cond: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(h_cond)).squeeze(-1)   # [B]


# ── 4b. PossessionChangeHead ────────────────────────────────────────────────

class PossessionChangeHead(nn.Module):
    """
    Prediu si entre la fase actual i la propera fase tàctica hi ha canvi
    de possessió (target binari).

    Output: logit per BCE.
    Es masquera amb el sample_mask del long_pause (mateix tractament que la
    TimeHead): quan és long_pause, l'equip de la propera fase és incert i la
    mostra s'ignora a la pèrdua.
    """

    def __init__(self, d_in: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.mlp  = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 2, 1),
        )

    def forward(self, h_cond: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(h_cond)).squeeze(-1)   # [B]


# ── 5. TrajectoryHead ───────────────────────────────────────────────────────

# Pas temporal entre frames predits = stride/FPS (s'agafa de constants.py).
PRED_STEP_S = STRIDE / FPS                # 0.3 s a stride=3 i FPS=10


class TrajectoryHead(nn.Module):
    """
    Prediu, per cada node v i cada step k ∈ {1, ..., T_PRED_MAX}, la posició
    (x, y) corresponent al moment t + k · PRED_STEP_S segons.

    Pipeline intern:

        feat = [h_cond, current_pos, dt_k]
                  ↓
                 MLP                      step-a-step, independent
                  ↓
               Δpos_pre  [B, K, N, 2]
                  ↓
        Conv1d temporal sobre K           barreja steps veïns
                  ↓
               Δpos_post [B, K, N, 2]
                  ↓
        traj_pred = current_pos + Δpos_post

    La MLP és compartida entre tots els steps i nodes (predicció
    independent per step). La Conv1d temporal actua com a refinador que
    força coherència entre steps consecutius, esmorteint dents de serra
    físicament inviables. S'inicialitza com a identitat perquè al
    començament de l'entrenament `Δpos_post ≈ Δpos_pre` i la conv aprengui
    gradualment quan suavitzar.

    TIMEHEAD_DISABLED: la TrajectoryHead no rep `delta_t` global del
    TimeHead; el dt_k local de cada step ja conté tota la informació
    temporal necessària.

    Forma de la sortida: [B, T_PRED_MAX, N, 2]   (posicions absolutes).
    Els steps fora de l'horitzó vàlid són emmascarats per target_mask del
    dataset i no contribueixen a la pèrdua.
    """

    def __init__(
        self,
        d_in: int,
        hidden: int = 128,
        dropout: float = DROPOUT,
        temporal_kernel: int = 5,
        probabilistic: bool = False,
    ) -> None:
        super().__init__()
        self.probabilistic = bool(probabilistic)
        # Sortida de la MLP: 2 canals de mitjana (Δx, Δy) i, si és
        # probabilística, 2 més de log-σ (log σ_x, log σ_y).
        out_dim = 4 if self.probabilistic else 2
        self.norm = nn.LayerNorm(d_in)
        self.mlp  = nn.Sequential(
            nn.Linear(d_in + 2 + 1, hidden),           # h_cond + pos + dt_k
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_dim),
        )
        # Conv1d temporal: suavitza Δpos sobre l'eix dels steps. Cada canal
        # (x o y) es processa independent però amb mateix kernel. Padding
        # "same" perquè la longitud K es conservi. Només actua sobre la
        # mitjana; el log-σ no se suavitza temporalment.
        assert temporal_kernel % 2 == 1, "temporal_kernel ha de ser senar"
        self.temporal_conv = nn.Conv1d(
            in_channels=2, out_channels=2,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
        )
        self._init_temporal_conv_as_identity()

        # Inicialitza el biaix del log-σ a un valor sensat (σ ≈ 5 m) perquè
        # la NLL parteixi d'una variància raonable i no col·lapsi.
        if self.probabilistic:
            with torch.no_grad():
                self.mlp[-1].bias[2:].fill_(TRAJ_LOG_SIGMA_INIT)

        # dt_k[k] = (k+1) * PRED_STEP_S, k ∈ {0, ..., T_PRED_MAX-1}
        dt_k = torch.arange(1, T_PRED_MAX + 1, dtype=torch.float32) * PRED_STEP_S
        self.register_buffer("dt_k", dt_k)              # [T_PRED_MAX]

    def _init_temporal_conv_as_identity(self) -> None:
        """
        Inicialitza la conv1d perquè actuï com a identitat: el step central
        del kernel té pes 1 (en el canal corresponent) i la resta 0. Així
        a l'inici de l'entrenament Δpos_post ≈ Δpos_pre i la xarxa parteix
        del mateix punt que sense conv. Durant l'entrenament la conv pot
        aprendre a desviar-se de la identitat si necessita suavitzar.
        """
        K = self.temporal_conv.kernel_size[0]
        center = K // 2
        with torch.no_grad():
            self.temporal_conv.weight.zero_()
            for c in range(self.temporal_conv.out_channels):
                # weight[out_c, in_c, t]: identitat = δ_{out_c, in_c} · δ_{t, center}
                self.temporal_conv.weight[c, c, center] = 1.0
            self.temporal_conv.bias.zero_()

    def forward(
        self,
        h_cond: torch.Tensor,        # [B, d_in]
        current_pos: torch.Tensor,   # [B, N, 2]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retorna (mean, log_sigma):
          mean       [B, K, N, 2]   posicions absolutes predites (mitjana)
          log_sigma  [B, K, N, 2]   log-desviació per coordenada (o None si
                                    el cap no és probabilístic)
        """
        B, N, _ = current_pos.shape
        K = T_PRED_MAX
        d_in = h_cond.shape[-1]

        h = self.norm(h_cond)                                                 # [B, d_in]

        # Broadcast a [B, K, N, *]
        h_exp        = h.view(B, 1, 1, d_in).expand(B, K, N, d_in)
        pos_exp      = current_pos.view(B, 1, N, 2).expand(B, K, N, 2)
        dt_k_exp     = self.dt_k.view(1, K, 1, 1).expand(B, K, N, 1)
        # TIMEHEAD_DISABLED:
        # delta_t_exp  = delta_t.view(B, 1, 1, 1).expand(B, K, N, 1)
        # feat = torch.cat([h_exp, pos_exp, dt_k_exp, delta_t_exp], dim=-1)

        feat = torch.cat([h_exp, pos_exp, dt_k_exp], dim=-1)                 # [B,K,N,d_in+3]
        out  = self.mlp(feat)                                                 # [B,K,N,out_dim]
        delta_pos = out[..., :2]                                             # [B, K, N, 2]

        # ── Conv1d temporal sobre l'eix K (només la mitjana) ────────────────
        # Reorganitzem a [B*N, 2, K] perquè Conv1d treballa amb format
        # (batch, channels, length): aquí "batch" agrupa cada (mostra, node)
        # i "length" és l'eix temporal dels K steps. Així cada node és
        # processat independent però la conv barreja steps veïns.
        d_in_conv  = delta_pos.permute(0, 2, 3, 1).reshape(B * N, 2, K)       # [B*N, 2, K]
        d_out_conv = self.temporal_conv(d_in_conv)                            # [B*N, 2, K]
        delta_pos  = d_out_conv.view(B, N, 2, K).permute(0, 3, 1, 2)          # [B, K, N, 2]

        mean = pos_exp + delta_pos                                            # [B, K, N, 2]

        if not self.probabilistic:
            return mean, None

        log_sigma = out[..., 2:].clamp(TRAJ_LOG_SIGMA_MIN, TRAJ_LOG_SIGMA_MAX)
        return mean, log_sigma                                               # [B, K, N, 2] ×2


# ── 6. Smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    B = 4
    h = torch.randn(B, D_MODEL)

    event_head = EventHead()
    event_emb  = EventEmbedding()
    logits     = event_head(h)
    emb        = event_emb(logits)
    h_cond     = torch.cat([h, emb], dim=-1)
    d_cond     = D_MODEL + EVENT_EMB_DIM

    # TIMEHEAD_DISABLED:
    # time_head  = TimeHead(d_cond)
    pause_head = PauseHead(d_cond)
    traj_head  = TrajectoryHead(d_cond)

    # TIMEHEAD_DISABLED:
    # mu, log_sigma = time_head(h_cond)
    pause_logit = pause_head(h_cond)

    current_pos = torch.randn(B, N_NODES, 2) * 30.0
    # TIMEHEAD_DISABLED: la TrajectoryHead ja no rep `delta_t`.
    traj_pred, traj_log_sigma = traj_head(h_cond, current_pos)

    prob_head = TrajectoryHead(d_cond, probabilistic=True)
    traj_mean_p, traj_ls_p = prob_head(h_cond, current_pos)

    print(f"  logits      shape={tuple(logits.shape)}")
    print(f"  emb_event   shape={tuple(emb.shape)}")
    print(f"  h_cond      shape={tuple(h_cond.shape)}")
    print(f"  pause_logit shape={tuple(pause_logit.shape)}")
    print(f"  traj_pred   shape={tuple(traj_pred.shape)}    (esperat ({B}, {T_PRED_MAX}, {N_NODES}, 2))")
    print(f"  traj_log_sigma (determinista) = {traj_log_sigma}")
    print(f"  traj_mean_p shape={tuple(traj_mean_p.shape)}  "
          f"traj_log_sigma_p shape={tuple(traj_ls_p.shape)}")

    n_params = sum(
        p.numel() for h in [event_head, event_emb, pause_head, traj_head]
                   for p in h.parameters() if p.requires_grad
    )
    print(f"\n  paràmetres totals heads: {n_params:,}")
    print("[OK] Heads smoke test.")



