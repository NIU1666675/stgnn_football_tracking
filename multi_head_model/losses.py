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
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    LAMBDA_EVENT,
    LAMBDA_TIME,
    LAMBDA_PAUSE,
    LAMBDA_POSS,
    LAMBDA_STATE,
)


# ── 1. Pèrdues individuals ──────────────────────────────────────────────────

def event_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """logits [B, K], target [B] long → escalar.
    class_weights [K], opcional, per a equilibrar classes desbalancejades.
    """
    return F.cross_entropy(logits, target, weight=class_weights)


_LOG_2PI_HALF = 0.5 * math.log(2.0 * math.pi)


def time_mixture_lognormal_nll(
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    NLL d'una *mixture* de log-normals:

        p(y) = Σ_k π_k · LogN(y; μ_k, σ_k)
        NLL  = -log p(y) = -logsumexp_k [log π_k + log p_k(y)]

    on log p_k(y) = -log y - log σ_k - ½ log(2π) - (log y - μ_k)² / (2 σ_k²)

    Forma dels tensors:
      mix_logits : [B, K]
      mu, log σ  : [B, K]
      y          : [B]
      mask       : [B]  (1 si mostra vàlida, 0 si long_pause)
    """
    log_y = torch.log(y.clamp(min=eps))                     # [B]
    sigma = log_sigma.exp()                                  # [B, K]

    # log p_k(y) per cada component
    log_pk = (
        -log_y.unsqueeze(-1)                                 # [B, 1]
        - log_sigma                                          # [B, K]
        - _LOG_2PI_HALF
        - (log_y.unsqueeze(-1) - mu) ** 2 / (2.0 * sigma ** 2)
    )                                                        # [B, K]

    log_pi = F.log_softmax(mix_logits, dim=-1)               # [B, K]
    log_px = torch.logsumexp(log_pi + log_pk, dim=-1)        # [B]

    nll = -log_px                                            # [B]
    denom = mask.sum().clamp(min=1.0)
    return (nll * mask).sum() / denom


@torch.no_grad()
def mixture_lognormal_mean(
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Mitjana esperada E[y] = Σ_k π_k · exp(μ_k + σ_k²/2).
    Útil com a predicció puntual per a la TrajectoryHead i mètriques.
    Retorna [B].
    """
    pi = F.softmax(mix_logits, dim=-1)                       # [B, K]
    sigma2 = (2.0 * log_sigma).exp()                         # [B, K]
    mean_per_k = torch.exp(mu + 0.5 * sigma2)                # [B, K]
    return (pi * mean_per_k).sum(dim=-1)                     # [B]


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


def pause_bce_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """logit [B], target [B] float {0,1} → escalar.
    pos_weight escalar tensor (scalar 1-d) per equilibrar la classe positiva.
    """
    return F.binary_cross_entropy_with_logits(
        logit, target, pos_weight=pos_weight,
    )


def possession_change_bce_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    BCE per al canvi de possessió, emmascarada per long_pause:
        logit:      [B]
        target:     [B] float {0,1}
        mask:       [B] {0,1}   (1 = mostra vàlida, no long_pause)
        pos_weight: tensor escalar opcional per a la classe positiva
    """
    per_sample = F.binary_cross_entropy_with_logits(
        logit, target, pos_weight=pos_weight, reduction="none",
    )
    denom = mask.sum().clamp(min=1.0)
    return (per_sample * mask).sum() / denom


# Factor d'escala per normalitzar coordenades a un rang aproximat [-1, 1].
# El camp fa 105×68 m i les coordenades estan centrades a (0, 0), per tant
# els valors absoluts màxims són ~(52, 34). Usem 50 com a mig de compromís
# (deixa la y una mica més comprimida que la x; resultat: gradients estables).
STATE_NORM = 50.0


def trajectory_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    norm: float = STATE_NORM,
) -> torch.Tensor:
    """
    MSE sobre la trajectòria predita, amb dues màscares jeràrquiques:

      pred, target: [B, K, N, 2]   K = T_PRED_MAX
      target_mask:  [B, K] bool    True si el step k té target real (k < n_pred)
      sample_mask:  [B] {0,1}      1 si la mostra no és long_pause

    Procediment:
      1. Calcula MSE per (b, k, n, coord), normalitzat per `norm`.
      2. Mitjana sobre (N, 2) → MSE per (b, k).
      3. Aplica target_mask: només steps vàlids contribueixen.
      4. Mitjana sobre els steps vàlids de cada mostra → MSE per (b).
      5. Aplica sample_mask: només mostres no long_pause contribueixen.
      6. Mitjana sobre les mostres vàlides del batch.
    """
    diff = (pred - target) / norm                            # [B, K, N, 2]
    sq = diff ** 2
    per_step = sq.mean(dim=(2, 3))                           # [B, K]

    mask_f = target_mask.float()
    n_per_sample = mask_f.sum(dim=1).clamp(min=1.0)          # [B]
    per_sample = (per_step * mask_f).sum(dim=1) / n_per_sample  # [B]

    denom = sample_mask.sum().clamp(min=1.0)
    return (per_sample * sample_mask).sum() / denom


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
        lambda_poss:  float = LAMBDA_POSS,
        lambda_state: float = LAMBDA_STATE,
        event_class_weights: Optional[torch.Tensor] = None,
        pause_pos_weight: Optional[float] = None,
        poss_pos_weight:  Optional[float] = None,
    ) -> None:
        super().__init__()
        self.le  = lambda_event
        self.lt  = lambda_time
        self.lp  = lambda_pause
        self.lpo = lambda_poss
        self.ls  = lambda_state

        # Pesos de classe (registrats com a buffers perquè es moguin amb el
        # model en .to(device) i es guardin amb el state_dict).
        if event_class_weights is not None:
            self.register_buffer(
                "event_class_weights",
                event_class_weights.detach().float(),
            )
        else:
            self.event_class_weights = None

        if pause_pos_weight is not None:
            self.register_buffer(
                "pause_pos_weight",
                torch.tensor([float(pause_pos_weight)]),
            )
        else:
            self.pause_pos_weight = None

        if poss_pos_weight is not None:
            self.register_buffer(
                "poss_pos_weight",
                torch.tensor([float(poss_pos_weight)]),
            )
        else:
            self.poss_pos_weight = None

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
            traj_pred        [B, K, N, 2]
        batch ha de contenir:
            phase_target     [B] long
            delta_proper     [B] float
            is_long_pause    [B] float {0,1}
            target_traj      [B, K, N, 2]
            target_mask      [B, K] bool
        """
        # Sample mask: 1 si la mostra és vàlida per a time/traj (no long pause).
        valid_sample = 1.0 - batch["is_long_pause"]    # [B]

        l_event = event_ce_loss(
            predictions["event_logits"], batch["phase_target"],
            class_weights=self.event_class_weights,
        )
        l_time  = time_mixture_lognormal_nll(
            predictions["time_mix_logits"],
            predictions["time_mu"],
            predictions["time_log_sigma"],
            batch["delta_proper"], valid_sample,
        )
        l_pause = pause_bce_loss(
            predictions["pause_logit"], batch["is_long_pause"],
            pos_weight=self.pause_pos_weight,
        )
        l_poss  = possession_change_bce_loss(
            predictions["poss_logit"],
            batch["possession_change"].clamp(min=0.0),    # sentinel −1 → 0 (mascarat)
            valid_sample,
            pos_weight=self.poss_pos_weight,
        )
        l_traj  = trajectory_mse_loss(
            predictions["traj_pred"], batch["target_traj"],
            batch["target_mask"], valid_sample,
        )

        total = (
            self.le  * l_event
          + self.lt  * l_time
          + self.lp  * l_pause
          + self.lpo * l_poss
          + self.ls  * l_traj
        )
        return {
            "total": total,
            "event": l_event.detach(),
            "time":  l_time.detach(),
            "pause": l_pause.detach(),
            "poss":  l_poss.detach(),
            "traj":  l_traj.detach(),
        }


# ── 3. Mètriques (no entren al gradient) ────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    batch:       Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Calcula mètriques interpretables:
      - event_acc    : accuracy top-1 de la classificació de fase
      - pause_acc    : accuracy de la binary head
      - time_mae_s   : MAE de la mediana log-normal exp(μ) vs Δt_proper,
                       sobre mostres no long-pause
      - traj_err_m   : distància euclidiana mitjana en metres, mitjana sobre
                       tots els steps vàlids i tots els nodes
      - final_err_m  : distància euclidiana mitjana al darrer step vàlid
                       (≈ error al frame previ a next.frame_start)
    """
    out: Dict[str, float] = {}
    valid_sample = 1.0 - batch["is_long_pause"]              # [B]
    n_valid_b = valid_sample.sum().clamp(min=1.0)

    # Event accuracy
    pred_cls = predictions["event_logits"].argmax(dim=-1)
    out["event_acc"] = (pred_cls == batch["phase_target"]).float().mean().item()

    # Pause accuracy
    pred_pause = (predictions["pause_logit"].sigmoid() > 0.5).float()
    out["pause_acc"] = (pred_pause == batch["is_long_pause"]).float().mean().item()

    # Possession change accuracy (sobre mostres no long_pause)
    pred_poss = (predictions["poss_logit"].sigmoid() > 0.5).float()
    poss_target = batch["possession_change"].clamp(min=0.0)
    out["poss_acc"] = (
        ((pred_poss == poss_target).float() * valid_sample).sum() / n_valid_b
    ).item()

    # Time MAE (en segons, usant la mitjana de la mixture de log-normals)
    time_pred = mixture_lognormal_mean(
        predictions["time_mix_logits"],
        predictions["time_mu"],
        predictions["time_log_sigma"],
    )                                                          # [B]
    time_abs_err = (time_pred - batch["delta_proper"]).abs()
    out["time_mae_s"] = ((time_abs_err * valid_sample).sum() / n_valid_b).item()

    # Trajectory error (distància euclidiana mitjana, en metres)
    diff = predictions["traj_pred"] - batch["target_traj"]   # [B, K, N, 2]
    eucl_per_step = diff.pow(2).sum(dim=-1).sqrt().mean(dim=2)  # [B, K]

    target_mask_f = batch["target_mask"].float()              # [B, K]
    n_steps_per_sample = target_mask_f.sum(dim=1).clamp(min=1.0)   # [B]
    eucl_per_sample = (
        (eucl_per_step * target_mask_f).sum(dim=1) / n_steps_per_sample
    )                                                         # [B]
    out["traj_err_m"] = (
        (eucl_per_sample * valid_sample).sum() / n_valid_b
    ).item()

    # Error al darrer step vàlid de cada mostra
    last_step_idx = (
        target_mask_f * torch.arange(target_mask_f.shape[1], device=target_mask_f.device)
    ).argmax(dim=1)                                           # [B]
    has_any = (target_mask_f.sum(dim=1) > 0).float()          # [B]
    final_err = eucl_per_step.gather(1, last_step_idx.unsqueeze(1)).squeeze(1)  # [B]
    final_mask = valid_sample * has_any
    out["final_err_m"] = (
        (final_err * final_mask).sum() / final_mask.sum().clamp(min=1.0)
    ).item()

    return out


# ── 4. Smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .constants import N_NODES, N_PHASE_CLASSES, T_PRED_MAX
    torch.manual_seed(0)

    B = 4
    K = T_PRED_MAX

    def leaf(t: torch.Tensor) -> torch.Tensor:
        return t.detach().requires_grad_(True)

    from .constants import N_TIME_MIXTURE
    Km = N_TIME_MIXTURE
    predictions = {
        "event_logits":     leaf(torch.randn(B, N_PHASE_CLASSES)),
        "time_mix_logits":  leaf(torch.zeros(B, Km)),
        "time_mu":          leaf(torch.linspace(0.5, 2.0, Km).unsqueeze(0).expand(B, Km).clone()),
        "time_log_sigma":   leaf(torch.full((B, Km), -0.3)),
        "pause_logit":      leaf(torch.randn(B)),
        "poss_logit":       leaf(torch.randn(B)),
        "traj_pred":        leaf(torch.randn(B, K, N_NODES, 2) * 30.0),
    }
    # Cada mostra té un nombre diferent de steps vàlids
    target_mask = torch.zeros(B, K, dtype=torch.bool)
    for b, n in enumerate([5, 17, 33, 0]):     # 0 = totalment long_pause
        target_mask[b, :n] = True

    batch = {
        "phase_target":      torch.randint(0, N_PHASE_CLASSES, (B,)),
        "delta_proper":      torch.rand(B) * 10.0 + 0.1,
        "is_long_pause":     torch.tensor([0., 0., 0., 1.]),       # b=3 long_pause
        "possession_change": torch.tensor([0., 1., 0., -1.]),      # b=3 mascarat
        "target_traj":       torch.randn(B, K, N_NODES, 2) * 30.0,
        "target_mask":       target_mask,
    }

    criterion = MultiHeadLoss()
    losses = criterion(predictions, batch)
    metrics = compute_metrics(predictions, batch)

    print("[Loss components]")
    for k, v in losses.items():
        print(f"  {k:7s} = {v.item():+.4f}")
    print("\n[Metrics]")
    for k, v in metrics.items():
        print(f"  {k:13s} = {v:.4f}")

    losses["total"].backward()
    grad_status = {k: (v.grad is not None) for k, v in predictions.items()}
    print(f"\n[Backprop] tensors amb gradient calculat:")
    for k, has in grad_status.items():
        print(f"  {k:16s} {'OK' if has else 'NO GRAD'}")

    print("\n[OK] Losses smoke test correcte.")

