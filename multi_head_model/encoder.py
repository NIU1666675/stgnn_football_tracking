"""
Encoder espai-temporal per al model multi-head.

Pipeline:
    [B, T, N, 8]  node_numeric   ─┐
    [B, N]        position_idx ───┤  →  Embedding + concat amb context
    [B, T, 15]    context ────────┘     →  Linear → [B, T, N, D]
                                              ↓
                          ┌── per cada layer (×N_LAYERS) ──┐
                          │   R-GCN espacial (per frame)   │
                          │   Transformer temporal (per node) │
                          └────────────────────────────────┘
                                              ↓
                          BallWeightedPool (frame de predicció +
                                              mitjana ponderada per
                                              distància a la pilota)
                                              ↓
                                     h_global  ∈ [B, D]
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    N_NODE_NUMERIC_FEAT,
    N_CONTEXT_FEAT,
    N_EVENT_TYPES,
    POSITION_VOCAB_SIZE,
    POSITION_EMBED_DIM,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    DROPOUT,
    SPATIAL_SIGMA,
)


# ── 1. Projecció d'entrada ─────────────────────────────────────────────────

class InputProjection(nn.Module):
    """
    Combina les tres branques d'input en un únic vector per (frame, node).

      node_numeric  [B, T, N, 8]
      position_idx  [B, N]              → Embedding(21,8) → [B, N, 8]
      context       [B, T, 15]          → broadcast a tots els nodes

    Concat (8 + 8 + 15 = 31) → Linear(31, D=128) → [B, T, N, D]
    """

    def __init__(self) -> None:
        super().__init__()
        self.position_emb = nn.Embedding(POSITION_VOCAB_SIZE, POSITION_EMBED_DIM)
        in_dim = N_NODE_NUMERIC_FEAT + POSITION_EMBED_DIM + N_CONTEXT_FEAT
        self.proj = nn.Linear(in_dim, D_MODEL)

    def forward(
        self,
        node_numeric: torch.Tensor,
        position_idx: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, _ = node_numeric.shape
        pos_emb = self.position_emb(position_idx)                # [B, N, E_pos]
        pos_emb = pos_emb.unsqueeze(1).expand(B, T, N, POSITION_EMBED_DIM)
        ctx     = context.unsqueeze(2).expand(B, T, N, N_CONTEXT_FEAT)
        x = torch.cat([node_numeric, pos_emb, ctx], dim=-1)       # [B, T, N, in_dim]
        return self.proj(x)                                       # [B, T, N, D]


# ── 2. R-GCN espacial ──────────────────────────────────────────────────────

def normalize_adj(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalització simètrica D^{-1/2} A D^{-1/2}, amb el grau a cada relació
    calculat com la suma de la fila (suporta pesos contínues i binaris).
    Forma: [..., N, N] → [..., N, N].
    """
    deg = adj.sum(dim=-1).clamp(min=eps)             # [..., N]
    deg_inv_sqrt = deg.pow(-0.5)                     # [..., N]
    return adj * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)


class RGCNLayer(nn.Module):
    """
    Capa R-GCN multi-relacional sense bases. Per a cada relació r aprèn una
    matriu W_r (D_in × D_out). El missatge propi s'aplica via W_self separat.

        h_v^{l+1} = σ( W_self · h_v + Σ_r Σ_u  Â_r[u,v] · W_r · h_u )
    """

    def __init__(self, dim_in: int, dim_out: int, n_relations: int = N_EVENT_TYPES) -> None:
        super().__init__()
        self.W_self = nn.Linear(dim_in, dim_out)
        self.W_r    = nn.Parameter(torch.empty(n_relations, dim_in, dim_out))
        nn.init.kaiming_uniform_(self.W_r, a=5 ** 0.5)

    def forward(self, x: torch.Tensor, adj_per_relation: torch.Tensor) -> torch.Tensor:
        """
        x:                [B, T, N, D_in]
        adj_per_relation: [B, R, T, N, N]
        return:           [B, T, N, D_out]
        """
        adj_norm = normalize_adj(adj_per_relation)                       # [B, R, T, N, N]
        # Agregació dels veïns per cada relació
        agg = torch.einsum('brtij,btjd->brtid', adj_norm, x)             # [B, R, T, N, D_in]
        # Projecció amb W_r per relació i suma
        out = torch.einsum('brtid,rde->brtie', agg, self.W_r)            # [B, R, T, N, D_out]
        out = out.sum(dim=1)                                             # [B, T, N, D_out]
        return out + self.W_self(x)                                      # [B, T, N, D_out]


# ── 3. Bloc Encoder = R-GCN + Transformer temporal ─────────────────────────

class EncoderBlock(nn.Module):
    """
    Un bloc combina:
      - R-GCN espacial (per frame), amb pre-LayerNorm + residual
      - Transformer temporal (per node), amb pre-LayerNorm intern (norm_first=True)
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.norm_rgcn = nn.LayerNorm(d_model)
        self.rgcn      = RGCNLayer(d_model, d_model)
        self.dropout   = nn.Dropout(dropout)

        self.transformer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = 4 * d_model,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )

    def forward(
        self,
        x: torch.Tensor,                # [B, T, N, D]
        adj: torch.Tensor,              # [B, R, T, N, N]
        frame_mask: torch.Tensor,       # [B, T] bool, True = vàlid
    ) -> torch.Tensor:
        # Espacial: R-GCN amb residual
        h = self.rgcn(self.norm_rgcn(x), adj)
        h = F.gelu(h)
        x = x + self.dropout(h)

        # Temporal: Transformer per node
        B, T, N, D = x.shape
        x_t = x.permute(0, 2, 1, 3).reshape(B * N, T, D)        # [B*N, T, D]
        # key_padding_mask del Transformer: True = posició a IGNORAR
        kpm = (~frame_mask).unsqueeze(1).expand(B, N, T).reshape(B * N, T)
        x_t = self.transformer(x_t, src_key_padding_mask=kpm)
        return x_t.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()


# ── 4. Pooling ponderat per distància a la pilota ──────────────────────────

class BallWeightedPool(nn.Module):
    """
    Pooling jeràrquic en dues fases:
      1) Selecciona el frame de predicció (últim frame vàlid de cada mostra).
      2) Mitjana sobre N nodes ponderada per la gaussiana de la distància
         a la pilota: w_n = exp(-d_n² / (2σ²)).

    Inputs:
      x:            [B, T, N, D]
      node_numeric: [B, T, N, 8]   (canals 4,5 = dx_ball, dy_ball)
      frame_mask:   [B, T] bool

    Output:
      h_global:     [B, D]
    """

    def __init__(self, sigma: float = SPATIAL_SIGMA) -> None:
        super().__init__()
        self.two_sigma_sq = 2.0 * float(sigma) ** 2

    def forward(
        self,
        x: torch.Tensor,
        node_numeric: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, D = x.shape
        device = x.device

        # Últim frame vàlid per mostra: argmax(mask * arange(T))
        t_range = torch.arange(T, device=device)
        idx_t = (frame_mask.long() * t_range).argmax(dim=1)                  # [B]

        idx_h    = idx_t.view(B, 1, 1, 1).expand(B, 1, N, D)
        idx_node = idx_t.view(B, 1, 1, 1).expand(B, 1, N, node_numeric.shape[-1])
        h_t  = torch.gather(x, 1, idx_h).squeeze(1)                          # [B, N, D]
        nn_t = torch.gather(node_numeric, 1, idx_node).squeeze(1)            # [B, N, 8]

        dx, dy = nn_t[..., 4], nn_t[..., 5]
        sq_dist = dx * dx + dy * dy                                          # [B, N]

        w = torch.exp(-sq_dist / self.two_sigma_sq)                          # [B, N]
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)                   # [B, N]

        return (h_t * w.unsqueeze(-1)).sum(dim=1)                            # [B, D]


# ── 5. Encoder complet ─────────────────────────────────────────────────────

class SpatioTemporalEncoder(nn.Module):
    """
    Encoder que retorna un únic vector h ∈ R^D per mostra.
    """

    def __init__(
        self,
        n_layers: int  = N_LAYERS,
        d_model:  int  = D_MODEL,
        n_heads:  int  = N_HEADS,
        dropout:  float = DROPOUT,
        ball_pool_sigma: float = SPATIAL_SIGMA,
    ) -> None:
        super().__init__()
        self.input_proj = InputProjection()
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.pool     = BallWeightedPool(sigma=ball_pool_sigma)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_numeric = batch["node_numeric"]
        position_idx = batch["position_idx"]
        context      = batch["context"]
        adj          = batch["adj_per_relation"]
        frame_mask   = batch["frame_mask"]

        x = self.input_proj(node_numeric, position_idx, context)             # [B, T, N, D]
        for block in self.blocks:
            x = block(x, adj, frame_mask)                                    # [B, T, N, D]
        x = self.norm_out(x)
        return self.pool(x, node_numeric, frame_mask)                        # [B, D]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

