"""
STGCN model for football tracking position prediction.

Input:  [B, N_HIS, N_NODES, N_FEAT]
Output: [B, N_PRED, N_NODES, N_FEAT]
"""

import os
import numpy as np
import torch
import torch.nn as nn
try:
    from stgcn_tracking.layers import STConvBlock, STConvBlockDynamic, OutputLayer
    from stgcn_tracking import constants as C
except:
    from layers import STConvBlock, STConvBlockDynamic, OutputLayer
    import constants as C


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for football tracking.

    Architecture (2 ST-Conv blocks + output layer):
      Input [B, N_HIS, N, 2]
        → STConvBlock(channels=(2, 32, 64),   Kt=3)   T: 50 → 46
        → STConvBlock(channels=(64, 64, 128),  Kt=3)   T: 46 → 42
        → OutputLayer(c_in=128, T_in=42)               T: 42 → 1 → [B, N_PRED, N, 2]

    Args:
        Lk      : Chebyshev basis tensor [N, Ks*N], loaded from disk
        Ks      : Chebyshev polynomial order (default 3)
        Kt      : temporal kernel size     (default 3)
        dropout : dropout rate             (default 0.1)
    """

    # T reduction per STConvBlock = 2 * (Kt - 1)
    # With 2 blocks and Kt=3: 50 - 2*4 = 42

    def __init__(
        self,
        Lk: torch.Tensor,
        Ks: int = 3,
        Kt: int = 3,
        dropout: float = 0.1,
        c_in: int = C.N_FEAT,
    ):
        super().__init__()

        T_after_block1 = C.N_HIS - 2 * (Kt - 1)       # 46
        T_after_block2 = T_after_block1 - 2 * (Kt - 1) # 42

        self.block1 = STConvBlock(
            Lk=Lk, Ks=Ks, Kt=Kt,
            channels=(c_in, 32, 64),
            dropout=dropout,
        )
        self.block2 = STConvBlock(
            Lk=Lk, Ks=Ks, Kt=Kt,
            channels=(64, 64, 128),
            dropout=dropout,
        )
        self.output = OutputLayer(
            c_in=128,
            T_in=T_after_block2,
            n_pred=C.N_PRED,
            n_feat=C.N_FEAT,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_HIS, N_NODES, N_FEAT]
        x = self.block1(x)   # [B, 46, N_NODES, 64]
        x = self.block2(x)   # [B, 42, N_NODES, 128]
        x = self.output(x)   # [B, N_PRED, N_NODES, N_FEAT]
        return x


N_FEAT_V2 = 6  # x, y, vx, vy, dx_ball, dy_ball

# ── Graf signat + dinàmic ────────────────────────────────────────────────────

# Signe entre nodes: +1 si mateix equip, -1 si rival, +1 si implica la pilota
def _build_sign_matrix(device: torch.device) -> torch.Tensor:
    """
    Retorna [N, N] amb +1 (companys/pilota) i -1 (rivals).
    Team A: nodes 0-10  |  Team B: nodes 11-21  |  Pilota: node 22
    """
    N = C.N_NODES
    team = torch.zeros(N, device=device)
    team[11:22] = 1   # Team B
    team[22]    = -1  # pilota (neutral)

    ti = team.unsqueeze(1).expand(N, N)
    tj = team.unsqueeze(0).expand(N, N)

    ball_involved = (ti == -1) | (tj == -1)   # qualsevol aresta amb la pilota → +1
    rivals        = (ti != tj) & ~ball_involved

    sign = torch.ones(N, N, device=device)
    sign[rivals] = -1.0
    return sign                               # [N, N]


def _compute_Lk_batch(
    pos: torch.Tensor,           # [B, N, 2]  posicions en metres
    sign: torch.Tensor,          # [N, N]     matriu de signes (estàtica)
    Ks: int,
) -> torch.Tensor:
    """
    Calcula la base de Chebyshev dinàmica per a cada mostra del batch.

    Retorna Lk: [B, N, Ks*N]

    Passos:
      1. W[b,i,j] = sign[i,j] · exp(-dist(i,j)² / σ²)
         on σ = distància mitjana off-diagonal de la mostra
      2. Laplacià signat normalitzat: L = I - D^{-½} W D^{-½}
         amb D[i,i] = Σ_j |W[i,j]|  (valor absolut per a W signat)
      3. Escalat: L̃ = 2L/λ_max - I   (λ_max via eigvalsh, exacte i vectoritzat)
      4. Polinomis de Chebyshev: T_0=I, T_1=L̃, T_k=2·L̃·T_{k-1}-T_{k-2}
    """
    B, N, _ = pos.shape
    device  = pos.device

    # 1. Distàncies i pes gaussià
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)   # [B, N, N, 2]
    dist = diff.norm(dim=-1)                      # [B, N, N]

    mask_off = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    sigma2   = dist[mask_off.expand(B, -1, -1)].reshape(B, N * (N - 1)).mean(dim=1) ** 2
    sigma2   = sigma2.view(B, 1, 1).clamp(min=1e-6)

    W = torch.exp(-dist ** 2 / sigma2)           # [B, N, N]  valors positius
    W = W * sign.unsqueeze(0)                    # [B, N, N]  signat per equips
    W.diagonal(dim1=-2, dim2=-1).zero_()         # sense auto-connexions

    # 2. Laplacià signat normalitzat
    deg        = W.abs().sum(dim=-1)             # [B, N]
    d_inv_sqrt = (deg + 1e-8).pow(-0.5)
    D_inv_sqrt = torch.diag_embed(d_inv_sqrt)    # [B, N, N]

    I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
    L = I - D_inv_sqrt @ W @ D_inv_sqrt          # [B, N, N]

    # 3. Escalar al rang [-1, 1]
    lambda_max = torch.linalg.eigvalsh(L)[:, -1].clamp(min=1e-6)  # [B]
    L_tilde    = 2.0 * L / lambda_max.view(B, 1, 1) - I           # [B, N, N]

    # 4. Polinomis de Chebyshev → [B, N, Ks*N]
    polys = [I]
    if Ks > 1:
        polys.append(L_tilde)
    for _ in range(2, Ks):
        polys.append(2 * L_tilde @ polys[-1] - polys[-2])
    return torch.cat(polys, dim=-1)              # [B, N, Ks*N]


class STGCNDynamic(nn.Module):
    """
    STGCN amb graf dinàmic i signat per equips.

    Diferències respecte STGCN:
      - No rep Lk a __init__; el calcula al forward() a partir de les posicions
        (x,y) de la finestra d'entrada, una matriu per seqüència del batch.
      - W[i,j] és positiu entre companys i negatiu entre rivals.
      - Compatible amb TrackingDatasetV2 (c_in=6) i TrackingDataset (c_in=2).

    Architecture:
      Input [B, N_HIS, N, c_in]
        → STConvBlockDynamic(channels=(c_in, 32, 64),  Kt=3)   T: 50 → 46
        → STConvBlockDynamic(channels=(64, 64, 128),   Kt=3)   T: 46 → 42
        → OutputLayer(c_in=128, T_in=42)                        T: 42 → 1 → [B, N_PRED, N, 2]
    """

    def __init__(
        self,
        Ks: int = 3,
        Kt: int = 3,
        dropout: float = 0.1,
        c_in: int = C.N_FEAT,
    ):
        super().__init__()
        self.Ks = Ks

        T_after_block1 = C.N_HIS - 2 * (Kt - 1)        # 46
        T_after_block2 = T_after_block1 - 2 * (Kt - 1) # 42

        self.block1 = STConvBlockDynamic(Ks=Ks, Kt=Kt, channels=(c_in, 32, 64),   dropout=dropout)
        self.block2 = STConvBlockDynamic(Ks=Ks, Kt=Kt, channels=(64,   64, 128),  dropout=dropout)
        self.output = OutputLayer(c_in=128, T_in=T_after_block2, n_pred=C.N_PRED, n_feat=C.N_FEAT)

        # Matriu de signes: estàtica (no entrena), es registra com a buffer
        self.register_buffer('sign', _build_sign_matrix(torch.device('cpu')))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_HIS, N, c_in]  — primers 2 canals sempre són x,y
        pos = x[:, :, :, :2].mean(dim=1)          # [B, N, 2] mitjana sobre N_HIS
        Lk  = _compute_Lk_batch(pos, self.sign, self.Ks)  # [B, N, Ks*N]

        x = self.block1(x, Lk)   # [B, 46, N, 64]
        x = self.block2(x, Lk)   # [B, 42, N, 128]
        x = self.output(x)        # [B, N_PRED, N, 2]
        return x


def build_model_dynamic(device: torch.device, c_in: int = C.N_FEAT) -> STGCNDynamic:
    """
    Instancia STGCNDynamic. No necessita cap fitxer .npy de graf.
      c_in=2  → TrackingDataset  (x, y)
      c_in=6  → TrackingDatasetV2 (x, y, vx, vy, dx_ball, dy_ball)
    """
    model = STGCNDynamic(c_in=c_in).to(device)
    return model


def build_model(device: torch.device) -> STGCN:
    """
    Carrega Lk des de disc i instancia el model, tot movent-lo al device indicat.
    """
    Lk_np = np.load(os.path.join(C.OUTPUT_DIR, 'Lk_chebyshev.npy'))  # [N, Ks*N]
    Lk = torch.tensor(Lk_np, dtype=torch.float32, device=device)
    model = STGCN(Lk=Lk).to(device)
    return model


def build_model_v2(device: torch.device) -> STGCN:
    """
    Igual que build_model però amb c_in=6 per acceptar les features ampliades
    de TrackingDatasetV2: [x, y, vx, vy, dx_ball, dy_ball].
    """
    Lk_np = np.load(os.path.join(C.OUTPUT_DIR, 'Lk_chebyshev.npy'))
    Lk = torch.tensor(Lk_np, dtype=torch.float32, device=device)
    model = STGCN(Lk=Lk, c_in=N_FEAT_V2).to(device)
    return model
