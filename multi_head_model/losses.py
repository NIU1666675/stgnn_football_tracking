"""
Pèrdues i mètriques del model multi-head.

Pèrdues:
  - Event:  cross-entropy sobre 9 classes (incloent stoppage)
  - Time:   NLL log-normal sobre Δt_proper (només mostres no long-pause)
  - Pause:  BCE sobre is_long_pause
  - State:  MSE sobre les posicions finals (només mostres no long-pause)

Mètriques (per monitoring, no entren al gradient):
  - Event accuracy / top-1
  - Pause accuracy
  - Time MAE en segons (sobre la mediana de la log-normal: exp(μ))
  - State error: distància euclidiana mitjana en metres
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    LAMBDA_EVENT,
    LAMBDA_TIME,
    LAMBDA_PAUSE,
    LAMBDA_STATE,
)


# ── 1. Pèrdues individuals ──────────────────────────────────────────────────

def event_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """logits [B, K], target [B] long → escalar."""
    return F.cross_entropy(logits, target)


_LOG_2PI_HALF = 0.5 * math.log(2.0 * math.pi)


def time_lognormal_nll(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    NLL log-normal: y > 0.
        f(y) = 1 / (y · σ · √(2π)) · exp(-(log y - μ)² / (2σ²))
        NLL  = log y + log σ + ½ log(2π) + (log y - μ)² / (2σ²)

    mu, log_sigma: [B]
    y:    [B]   target en segons (Δt_proper)
    mask: [B]   {0,1}, 1 = mostra vàlida (no long pause)
    """
    log_y  = torch.log(y.clamp(min=eps))
    sigma  = log_sigma.exp()
    nll = log_y + log_sigma + _LOG_2PI_HALF + (log_y - mu) ** 2 / (2.0 * sigma ** 2)
    denom = mask.sum().clamp(min=1.0)
    return (nll * mask).sum() / denom


def pause_bce_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """logit [B], target [B] float {0,1} → escalar."""
    return F.binary_cross_entropy_with_logits(logit, target)


# Factor d'escala per normalitzar coordenades a un rang aproximat [-1, 1].
# El camp fa 105×68 m i les coordenades estan centrades a (0, 0), per tant
# els valors absoluts màxims són ~(52, 34). Usem 50 com a mig de compromís
# (deixa la y una mica més comprimida que la x; resultat: gradients estables).
STATE_NORM = 50.0


def state_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    norm: float = STATE_NORM,
) -> torch.Tensor:
    """
    MSE sobre coordenades x, y, normalitzades per `norm` per fer la pèrdua
    comparable en escala a les altres heads.

        pred, target: [B, N, 2]   (en metres)
        mask: [B] {0,1}           (opcional; si es dóna, ignora long-pauses)
    """
    diff = (pred - target) / norm                     # ~[-1, 1]
    sq = diff ** 2                                    # [B, N, 2]
    per_sample = sq.mean(dim=(1, 2))                  # [B]
    if mask is None:
        return per_sample.mean()
    denom = mask.sum().clamp(min=1.0)
    return (per_sample * mask).sum() / denom


# ── 2. Combinador ───────────────────────────────────────────────────────────

class MultiHeadLoss(nn.Module):
    """
    Suma ponderada de les 4 pèrdues. Els pesos λ vénen de constants.py
    però es poden sobreescriure al constructor.
    """

    def __init__(
        self,
        lambda_event: float = LAMBDA_EVENT,
        lambda_time:  float = LAMBDA_TIME,
        lambda_pause: float = LAMBDA_PAUSE,
        lambda_state: float = LAMBDA_STATE,
    ) -> None:
        super().__init__()
        self.le = lambda_event
        self.lt = lambda_time
        self.lp = lambda_pause
        self.ls = lambda_state

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        batch:       Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        predictions ha de contenir:
            event_logits     [B, 9]
            time_mu          [B]
            time_log_sigma   [B]
            pause_logit      [B]
            state_pred       [B, N, 2]
        batch ha de contenir:
            phase_target     [B] long
            delta_proper     [B] float
            is_long_pause    [B] float {0,1}
            state_final      [B, N, 2]
        """
        # Mascara: 1 si la mostra és vàlida per a time/state (no long pause).
        valid_time = 1.0 - batch["is_long_pause"]    # [B]

        l_event = event_ce_loss(predictions["event_logits"], batch["phase_target"])
        l_time  = time_lognormal_nll(
            predictions["time_mu"], predictions["time_log_sigma"],
            batch["delta_proper"], valid_time,
        )
        l_pause = pause_bce_loss(predictions["pause_logit"], batch["is_long_pause"])
        l_state = state_mse_loss(predictions["state_pred"], batch["state_final"], valid_time)

        total = (
            self.le * l_event
          + self.lt * l_time
          + self.lp * l_pause
          + self.ls * l_state
        )
        return {
            "total": total,
            "event": l_event.detach(),
            "time":  l_time.detach(),
            "pause": l_pause.detach(),
            "state": l_state.detach(),
        }


# ── 3. Mètriques (no entren al gradient) ────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    batch:       Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Calcula mètriques interpretables:
      - event_acc   : accuracy top-1 de la classificació de fase
      - pause_acc   : accuracy de la binary head
      - time_mae_s  : MAE de la mediana log-normal exp(μ) vs Δt_proper, sobre
                      mostres no long-pause
      - state_err_m : distància euclidiana mitjana en metres, sobre mostres
                      no long-pause
    """
    out: Dict[str, float] = {}
    valid_time = 1.0 - batch["is_long_pause"]              # [B]
    n_valid = valid_time.sum().clamp(min=1.0)

    # Event accuracy
    pred_cls = predictions["event_logits"].argmax(dim=-1)
    out["event_acc"] = (pred_cls == batch["phase_target"]).float().mean().item()

    # Pause accuracy
    pred_pause = (predictions["pause_logit"].sigmoid() > 0.5).float()
    out["pause_acc"] = (pred_pause == batch["is_long_pause"]).float().mean().item()

    # Time MAE (en segons, usant mediana log-normal = exp(μ))
    time_pred = predictions["time_mu"].exp()                # [B]
    time_abs_err = (time_pred - batch["delta_proper"]).abs()
    out["time_mae_s"] = ((time_abs_err * valid_time).sum() / n_valid).item()

    # State error (distància euclidiana mitjana per node, en metres)
    diff = predictions["state_pred"] - batch["state_final"]  # [B, N, 2]
    eucl = diff.pow(2).sum(dim=-1).sqrt().mean(dim=1)        # [B]
    out["state_err_m"] = ((eucl * valid_time).sum() / n_valid).item()

    return out


# ── 4. Smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .constants import N_NODES, N_PHASE_CLASSES
    torch.manual_seed(0)

    B = 4
    # Prediccions sintètiques (tensors leaf amb requires_grad per provar backward)
    def leaf(t: torch.Tensor) -> torch.Tensor:
        return t.detach().requires_grad_(True)

    predictions = {
        "event_logits":   leaf(torch.randn(B, N_PHASE_CLASSES)),
        "time_mu":        leaf(torch.randn(B) * 0.5 + 1.0),
        "time_log_sigma": leaf(torch.randn(B) * 0.5),
        "pause_logit":    leaf(torch.randn(B)),
        "state_pred":     leaf(torch.randn(B, N_NODES, 2) * 30.0),
    }
    # Batch sintètic (mateixa forma que retorna el dataset)
    batch = {
        "phase_target":  torch.randint(0, N_PHASE_CLASSES, (B,)),
        "delta_proper":  torch.rand(B) * 10.0 + 0.1,         # entre 0.1 i 10.1 s
        "is_long_pause": torch.tensor([0., 1., 0., 0.]),     # 1 mostra és long pause
        "state_final":   torch.randn(B, N_NODES, 2) * 30.0,
    }

    criterion = MultiHeadLoss()
    losses = criterion(predictions, batch)
    metrics = compute_metrics(predictions, batch)

    print("[Loss components]")
    for k, v in losses.items():
        print(f"  {k:7s} = {v.item():+.4f}")
    print(f"\n[Metrics]")
    for k, v in metrics.items():
        print(f"  {k:13s} = {v:.4f}")

    # Comprovació de backward
    losses["total"].backward()
    grad_status = {k: (v.grad is not None) for k, v in predictions.items()}
    print(f"\n[Backprop] tensors amb gradient calculat:")
    for k, has in grad_status.items():
        print(f"  {k:16s} {'OK' if has else 'NO GRAD'}")

    print("\n[OK] Losses smoke test correcte.")
