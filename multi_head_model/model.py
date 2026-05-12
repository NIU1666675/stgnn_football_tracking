"""
Model multi-head complet: encoder + cascada de heads.

Flux d'execució (forward):

    batch  ──┐
             ↓
       SpatioTemporalEncoder ──→ h ∈ [B, D]
                                    │
                          ┌─────────┴─────────┐
                          ↓                   ↓
                     EventHead             (h al complet també
                          │                  s'envia a heads finals)
                          ↓ logits
                     EventEmbedding ──→ emb_event
                                    │
                            h_cond = concat(h, emb_event)
                                    │
                ┌───────────────────┼───────────────────┐
                ↓                   ↓                   ↓
            TimeHead            PauseHead            StateHead
          (μ, log σ)          pause_logit          state_pred [B, N, 2]
                                                  └ usa current_pos i Δt

Δt usat per la StateHead:
  - Entrenament (teacher forcing): batch["delta_proper"] (target real)
  - Inferència:                    exp(μ) (mediana log-normal predita)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .constants import D_MODEL, SPATIAL_SIGMA
from .encoder import SpatioTemporalEncoder
from .heads   import (
    EventHead,
    EventEmbedding,
    TimeHead,
    PauseHead,
    TrajectoryHead,
    EVENT_EMB_DIM,
)


# ── Helper: extreure features del frame de predicció ───────────────────────

def _gather_at_pred_frame(x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """
    x:          [B, T, ...]   tensor amb una dim temporal a la posició 1
    frame_mask: [B, T] bool

    Retorna:    [B, ...]   els features de l'últim frame vàlid de cada mostra.
    """
    B, T = frame_mask.shape
    t_range = torch.arange(T, device=x.device)
    idx_t = (frame_mask.long() * t_range).argmax(dim=1)            # [B]
    return x[torch.arange(B, device=x.device), idx_t]


# ── Model complet ──────────────────────────────────────────────────────────

class MultiHeadModel(nn.Module):
    """
    Encoder + 4 heads en cascada. Comparteixen `h_cond = concat(h, emb_event)`.
    """

    def __init__(
        self,
        ball_pool_sigma: float = SPATIAL_SIGMA,
    ) -> None:
        super().__init__()
        self.encoder = SpatioTemporalEncoder(ball_pool_sigma=ball_pool_sigma)

        d_cond = D_MODEL + EVENT_EMB_DIM

        self.event_head = EventHead(D_MODEL)
        self.event_emb  = EventEmbedding(EVENT_EMB_DIM)
        self.time_head  = TimeHead(d_cond)
        self.pause_head = PauseHead(d_cond)
        self.traj_head  = TrajectoryHead(d_cond)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward end-to-end coherent entre entrenament i inferència:
          - Cap teacher forcing: el TrajectoryHead consumeix sempre
            time_mu.exp().detach() (predicció del TimeHead, sense propagar
            gradient cap a aquest head).
        """
        # 1. Encoder → h global
        h = self.encoder(batch)                                    # [B, D]

        # 2. EventHead → logits
        event_logits = self.event_head(h)                          # [B, 9]

        # 3. EventEmbedding soft → h_cond
        emb_event = self.event_emb(event_logits)                   # [B, E]
        h_cond    = torch.cat([h, emb_event], dim=-1)              # [B, D+E]

        # 4. Time / Pause heads
        time_mu, time_log_sigma = self.time_head(h_cond)           # [B], [B]
        pause_logit             = self.pause_head(h_cond)          # [B]

        # 5. TrajectoryHead: current_pos al frame t + Δt predit (sense leakage)
        node_numeric = batch["node_numeric"]                       # [B, T, N, 8]
        frame_mask   = batch["frame_mask"]                         # [B, T] bool
        current_pos  = _gather_at_pred_frame(
            node_numeric[..., :2], frame_mask,                     # canals 0,1 = x, y
        )                                                          # [B, N, 2]

        # .detach() impedeix que la trajectory_loss propagi gradient al TimeHead
        delta_t = time_mu.exp().detach()                           # [B]

        traj_pred = self.traj_head(h_cond, current_pos, delta_t)   # [B, T_PRED_MAX, N, 2]

        return {
            "event_logits":   event_logits,
            "time_mu":        time_mu,
            "time_log_sigma": time_log_sigma,
            "pause_logit":    pause_logit,
            "traj_pred":      traj_pred,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


