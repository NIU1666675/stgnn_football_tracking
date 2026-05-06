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
    StateHead,
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
        self.state_head = StateHead(d_cond)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        use_predicted_dt: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        batch: dict tal com retorna PhaseDataset (vegis dataset.py).
        use_predicted_dt:
            - False  (entrenament, teacher forcing): la StateHead rep el Δt
                     veritable de batch["delta_proper"].
            - True   (inferència): la StateHead rep exp(μ) (mediana log-normal).

        Retorna un dict amb les 5 sortides necessàries per la loss.
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

        # 5. State head: necessita current_pos del frame de predicció + Δt
        node_numeric = batch["node_numeric"]                       # [B, T, N, 8]
        frame_mask   = batch["frame_mask"]                         # [B, T] bool
        current_pos  = _gather_at_pred_frame(
            node_numeric[..., :2], frame_mask,                     # canals 0,1 = x, y
        )                                                          # [B, N, 2]

        if use_predicted_dt:
            delta_t_for_state = time_mu.exp()                      # [B] mediana
        else:
            delta_t_for_state = batch["delta_proper"]              # [B] teacher forcing

        state_pred = self.state_head(h_cond, current_pos, delta_t_for_state)  # [B, N, 2]

        return {
            "event_logits":   event_logits,
            "time_mu":        time_mu,
            "time_log_sigma": time_log_sigma,
            "pause_logit":    pause_logit,
            "state_pred":     state_pred,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .constants import (
        T_MAX, N_NODES, N_NODE_NUMERIC_FEAT, N_CONTEXT_FEAT,
        N_EVENT_TYPES, POSITION_VOCAB_SIZE, N_PHASE_CLASSES,
    )
    from .losses import MultiHeadLoss, compute_metrics

    torch.manual_seed(0)
    B = 4

    model = MultiHeadModel()
    print(f"[i] Paràmetres entrenables: {model.count_parameters():,}")

    # Batch fictici amb estructura idèntica al dataset
    batch = {
        "node_numeric":     torch.randn(B, T_MAX, N_NODES, N_NODE_NUMERIC_FEAT) * 10,
        "position_idx":     torch.randint(0, POSITION_VOCAB_SIZE, (B, N_NODES)),
        "context":          torch.randn(B, T_MAX, N_CONTEXT_FEAT),
        "adj_per_relation": torch.rand(B, N_EVENT_TYPES, T_MAX, N_NODES, N_NODES),
        "frame_mask":       torch.zeros(B, T_MAX, dtype=torch.bool),
        "phase_target":     torch.randint(0, N_PHASE_CLASSES, (B,)),
        "delta_proper":     torch.rand(B) * 10.0 + 0.1,
        "is_long_pause":    torch.tensor([0., 1., 0., 0.]),
        "state_final":      torch.randn(B, N_NODES, 2) * 30.0,
    }
    # Frames vàlids variats per mostra
    for i, n_valid in enumerate([15, 30, 50, 100]):
        batch["frame_mask"][i, :n_valid] = True

    # Forward (mode entrenament)
    preds = model(batch, use_predicted_dt=False)

    print("\n[Output shapes]")
    for k, v in preds.items():
        print(f"  {k:18s} {tuple(v.shape)}")

    # Loss + backward
    criterion = MultiHeadLoss()
    losses = criterion(preds, batch)
    metrics = compute_metrics(preds, batch)

    print("\n[Loss]")
    for k, v in losses.items():
        print(f"  {k:7s} = {v.item():+.4f}")
    print("\n[Metrics]")
    for k, v in metrics.items():
        print(f"  {k:13s} = {v:.4f}")

    losses["total"].backward()
    print(f"\n[Backprop] OK")

    # Mode inferència (Δt predit)
    with torch.no_grad():
        preds_inf = model(batch, use_predicted_dt=True)
    print(f"[Inferència] state_pred shape={tuple(preds_inf['state_pred'].shape)}  ✓")

    print("\n[OK] Model complet validat.")
