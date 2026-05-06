"""
STGCN layers adapted for football tracking.

Input shape convention throughout: [B, T, N, C]
  B = batch size
  T = time steps
  N = 23 nodes  (22 players + ball)
  C = channels  (2 at input: x,y  → grows through the network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """
    Spectral graph convolution via Chebyshev polynomial approximation.

    For each time step independently:
      y_j = Σ_i  Σ_k  θ_{i,j,k} · T_k(L̃) · x_i

    Input:  [B, T, N, C_in]
    Output: [B, T, N, C_out]

    Args:
        Lk     : torch.Tensor [N, Ks*N] — precomputed Chebyshev basis (fixed, not trained)
        Ks     : int  — polynomial order (number of hop neighbourhoods)
        c_in   : int  — input channels
        c_out  : int  — output channels
    """

    def __init__(self, Lk: torch.Tensor, Ks: int, c_in: int, c_out: int):
        super().__init__()
        self.register_buffer('Lk', Lk)   # [N, Ks*N]  — not a parameter
        self.Ks   = Ks
        self.c_in  = c_in
        self.c_out = c_out
        # Learnable kernel: maps Ks*c_in features → c_out
        self.theta = nn.Linear(Ks * c_in, c_out, bias=False)
        nn.init.xavier_uniform_(self.theta.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C_in]
        B, T, N, C_in = x.shape
        Ks = self.Ks

        # Flatten batch and time → [B*T, N, C_in]
        x = x.reshape(B * T, N, C_in)

        # x_tmp: [B*T*C_in, N]
        x_tmp = x.permute(0, 2, 1).reshape(B * T * C_in, N)

        # Graph multiplication: [B*T*C_in, N] @ [N, Ks*N] → [B*T*C_in, Ks*N]
        x_mul = x_tmp @ self.Lk                              # [B*T*C_in, Ks*N]
        x_mul = x_mul.reshape(B * T, C_in, Ks, N)           # [B*T, C_in, Ks, N]

        # Rearrange to [B*T*N, Ks*C_in] for the linear layer
        x_ker = x_mul.permute(0, 3, 1, 2).reshape(B * T * N, Ks * C_in)

        # Apply learnable kernel → [B*T*N, C_out]
        out = self.theta(x_ker)

        # Restore shape → [B, T, N, C_out]
        return out.reshape(B, T, N, self.c_out)


class TemporalConv(nn.Module):
    """
    Causal 1-D temporal convolution with Gated Linear Unit (GLU).

    Operates along the time axis independently for each node:
      - Conv kernel: (Kt, 1) — width Kt along time, width 1 along nodes
      - GLU: output split in half → P ⊙ σ(Q)
      - Residual connection when c_in == c_out

    Input:  [B, T, N, C_in]
    Output: [B, T - Kt + 1, N, C_out]

    Args:
        Kt    : int  — temporal kernel size (number of frames)
        c_in  : int  — input channels
        c_out : int  — output channels
    """

    def __init__(self, Kt: int, c_in: int, c_out: int):
        super().__init__()
        self.Kt    = Kt
        self.c_in  = c_in
        self.c_out = c_out

        # Conv2d over (T, N) treating C as the "in_channels" dimension
        # Kernel (Kt, 1): slides along T, fixed along N
        # Output channels = 2*c_out to split into P and Q for GLU
        self.conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=2 * c_out,
            kernel_size=(Kt, 1),
            padding=0        # causal: no future padding
        )

        # Residual projection if channel sizes differ
        self.residual = (
            nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
            if c_in != c_out else None
        )

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C_in]
        # Conv2d expects [B, C, H, W] → treat as [B, C_in, T, N]
        x_in = x.permute(0, 3, 1, 2)   # [B, C_in, T, N]

        # Apply convolution → [B, 2*C_out, T-Kt+1, N]
        out = self.conv(x_in)

        # GLU: split into P and Q, gate with sigmoid
        P, Q = out.chunk(2, dim=1)      # each [B, C_out, T-Kt+1, N]
        out  = P * torch.sigmoid(Q)     # [B, C_out, T-Kt+1, N]

        # Residual: trim the input to match output time length
        T_out = out.shape[2]
        res = x_in[:, :, -T_out:, :]   # align to last T_out frames
        if self.residual is not None:
            res = self.residual(res)

        out = out + res
        # Back to [B, T_out, N, C_out]
        return out.permute(0, 2, 3, 1)


class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block: the core building block of STGCN.

    "Sandwich" structure:
      Temporal Gated Conv  →  Spatial Graph Conv  →  Temporal Gated Conv
      + Layer Norm + Dropout

    Each temporal conv reduces T by (Kt - 1), so a block reduces T by 2*(Kt-1).

    Args:
        Lk       : Chebyshev basis [N, Ks*N]
        Ks       : spatial kernel order
        Kt       : temporal kernel size
        channels : (c_si, c_t, c_oo)
                    c_si → input channels
                    c_t  → bottleneck channels (graph conv)
                    c_oo → output channels
        dropout  : float
    """

    def __init__(self, Lk, Ks, Kt, channels, dropout=0.1):
        super().__init__()
        c_si, c_t, c_oo = channels

        self.tconv1  = TemporalConv(Kt, c_si, c_t)
        self.gconv   = GraphConv(Lk, Ks, c_t, c_t)
        self.relu    = nn.ReLU()
        self.tconv2  = TemporalConv(Kt, c_t, c_oo)
        self.norm    = nn.LayerNorm(c_oo)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, N, c_si]
        x = self.tconv1(x)              # [B, T-Kt+1, N, c_t]
        x = self.relu(self.gconv(x))    # [B, T-Kt+1, N, c_t]
        x = self.tconv2(x)              # [B, T-2(Kt-1), N, c_oo]
        x = self.norm(x)
        x = self.dropout(x)
        return x


class OutputLayer(nn.Module):
    """
    Output head: maps the last ST-Conv block output to N_PRED predictions.

    Temporal conv collapses remaining time steps → 1,
    then a linear layer maps channels → N_PRED * N_FEAT.

    Input:  [B, T_remaining, N, C]
    Output: [B, N_PRED, N, N_FEAT]

    Args:
        c_in   : input channels
        T_in   : remaining time steps entering this layer
        n_pred : number of future frames to predict
        n_feat : output features per node (2 for x,y)
    """

    def __init__(self, c_in: int, T_in: int, n_pred: int, n_feat: int = 2):
        super().__init__()
        self.n_pred = n_pred
        self.n_feat = n_feat

        # Temporal conv to collapse all remaining time steps
        self.tconv = TemporalConv(Kt=T_in, c_in=c_in, c_out=c_in)
        # FC: map channels → n_pred * n_feat
        self.fc    = nn.Linear(c_in, n_pred * n_feat)

    def forward(self, x):
        # x: [B, T_in, N, C]
        x = self.tconv(x)              # [B, 1, N, C]
        x = x.squeeze(1)               # [B, N, C]
        x = self.fc(x)                 # [B, N, n_pred*n_feat]
        B, N, _ = x.shape
        x = x.reshape(B, N, self.n_pred, self.n_feat)
        x = x.permute(0, 2, 1, 3)     # [B, n_pred, N, n_feat]
        return x


# ── Classes dinàmiques (graf calculat per seqüència al forward) ──────────────

class GraphConvDynamic(nn.Module):
    """
    Igual que GraphConv però rep Lk al forward() en lloc de tenir-lo com a buffer.
    Això permet un Lk diferent per cada mostra del batch (graf dinàmic).

    Input:  x:  [B, T, N, C_in]
            Lk: [B, N, Ks*N]    — base de Chebyshev dinàmica per mostra
    Output: [B, T, N, C_out]
    """

    def __init__(self, Ks: int, c_in: int, c_out: int):
        super().__init__()
        self.Ks    = Ks
        self.c_in  = c_in
        self.c_out = c_out
        self.theta = nn.Linear(Ks * c_in, c_out, bias=False)
        nn.init.xavier_uniform_(self.theta.weight)

    def forward(self, x: torch.Tensor, Lk: torch.Tensor) -> torch.Tensor:
        # x:  [B, T, N, C_in]
        # Lk: [B, N, Ks*N]
        B, T, N, C_in = x.shape
        Ks = self.Ks

        # Expandir Lk a tots els passos temporals → [B*T, N, Ks*N]
        Lk_exp = Lk.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, Ks * N)

        # [B*T, C_in, N] × [B*T, N, Ks*N] → [B*T, C_in, Ks*N]
        x_bt  = x.reshape(B * T, N, C_in).permute(0, 2, 1)  # [B*T, C_in, N]
        x_mul = torch.bmm(x_bt, Lk_exp)                      # [B*T, C_in, Ks*N]
        x_mul = x_mul.reshape(B * T, C_in, Ks, N)            # [B*T, C_in, Ks, N]

        x_ker = x_mul.permute(0, 3, 1, 2).reshape(B * T * N, Ks * C_in)
        out   = self.theta(x_ker)                             # [B*T*N, C_out]
        return out.reshape(B, T, N, self.c_out)


class STConvBlockDynamic(nn.Module):
    """
    Igual que STConvBlock però usa GraphConvDynamic:
    el forward rep Lk com a argument en lloc de tenir-lo fix.

    Args:
        Ks       : spatial kernel order
        Kt       : temporal kernel size
        channels : (c_si, c_t, c_oo)
        dropout  : float
    """

    def __init__(self, Ks: int, Kt: int, channels: tuple, dropout: float = 0.1):
        super().__init__()
        c_si, c_t, c_oo = channels

        self.tconv1  = TemporalConv(Kt, c_si, c_t)
        self.gconv   = GraphConvDynamic(Ks, c_t, c_t)
        self.relu    = nn.ReLU()
        self.tconv2  = TemporalConv(Kt, c_t, c_oo)
        self.norm    = nn.LayerNorm(c_oo)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, Lk: torch.Tensor) -> torch.Tensor:
        # x:  [B, T, N, c_si]
        # Lk: [B, N, Ks*N]
        x = self.tconv1(x)                    # [B, T-Kt+1, N, c_t]
        x = self.relu(self.gconv(x, Lk))      # [B, T-Kt+1, N, c_t]
        x = self.tconv2(x)                    # [B, T-2(Kt-1), N, c_oo]
        x = self.norm(x)
        x = self.dropout(x)
        return x
