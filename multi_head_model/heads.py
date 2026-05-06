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

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    D_MODEL,
    DROPOUT,
    N_NODES,
    N_PHASE_CLASSES,
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

class TimeHead(nn.Module):
    """
    Prediu (μ, log σ) d'una distribució log-normal sobre Δt_proper.
    L'aprenentatge es fa amb NLL log-normal; vegis losses.py.
    """

    def __init__(self, d_in: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.mlp  = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, 2),    # (μ, log σ)
        )

    def forward(self, h_cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(self.norm(h_cond))            # [B, 2]
        mu        = out[..., 0]                      # [B]
        log_sigma = out[..., 1].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return mu, log_sigma


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


# ── 5. StateHead ────────────────────────────────────────────────────────────

class StateHead(nn.Module):
    """
    Prediu, per cada node, el desplaçament (Δx, Δy) des de la seva posició al
    frame de predicció fins a la posició al frame de l'event. La posició final
    és:
        state_final[b, n] = current_pos[b, n] + Δpos[b, n]

    Per cada node concatenem:
        - h_cond (global)        [d_in]
        - posició actual (x, y)  [2]
        - Δt (segons)            [1]
    """

    def __init__(
        self,
        d_in: int,
        hidden: int = 128,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.mlp  = nn.Sequential(
            nn.Linear(d_in + 2 + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(
        self,
        h_cond: torch.Tensor,         # [B, d_in]
        current_pos: torch.Tensor,    # [B, N, 2]
        delta_t: torch.Tensor,        # [B]   (segons)
    ) -> torch.Tensor:
        B, N, _ = current_pos.shape
        h = self.norm(h_cond)                                      # [B, d_in]
        h_exp  = h.unsqueeze(1).expand(B, N, h.shape[-1])          # [B, N, d_in]
        dt_exp = delta_t.view(B, 1, 1).expand(B, N, 1)             # [B, N, 1]
        feat = torch.cat([h_exp, current_pos, dt_exp], dim=-1)     # [B, N, d_in+3]
        delta_pos = self.mlp(feat)                                 # [B, N, 2]
        return current_pos + delta_pos                             # [B, N, 2]


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

    time_head  = TimeHead(d_cond)
    pause_head = PauseHead(d_cond)
    state_head = StateHead(d_cond)

    mu, log_sigma = time_head(h_cond)
    pause_logit   = pause_head(h_cond)

    current_pos = torch.randn(B, N_NODES, 2) * 30.0
    delta_t     = torch.rand(B) * 10.0
    state_final = state_head(h_cond, current_pos, delta_t)

    print(f"  logits      shape={tuple(logits.shape)}      (esperat ({B}, {N_PHASE_CLASSES}))")
    print(f"  emb_event   shape={tuple(emb.shape)}      (esperat ({B}, {EVENT_EMB_DIM}))")
    print(f"  h_cond      shape={tuple(h_cond.shape)}     (esperat ({B}, {d_cond}))")
    print(f"  μ           shape={tuple(mu.shape)}, range=[{mu.min().item():+.2f}, {mu.max().item():+.2f}]")
    print(f"  log σ       shape={tuple(log_sigma.shape)}, range=[{log_sigma.min().item():+.2f}, {log_sigma.max().item():+.2f}]")
    print(f"  pause_logit shape={tuple(pause_logit.shape)}")
    print(f"  state_final shape={tuple(state_final.shape)}     (esperat ({B}, {N_NODES}, 2))")

    n_params = sum(p.numel() for h in [event_head, event_emb, time_head, pause_head, state_head]
                              for p in h.parameters() if p.requires_grad)
    print(f"\n  paràmetres totals de les heads: {n_params:,}")
    print("[OK] Heads forward pass correcte.")
