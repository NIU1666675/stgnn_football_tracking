import os
import numpy as np
import torch
from torch.utils.data import Dataset
try: 
    from stgcn_tracking import constants as C
except:
    import constants as C

class TrackingDataset(Dataset):
    """
    Carrega un .npy de seqüències [N_seq, N_FRAME, N_NODES, N_FEAT] i
    el divideix en (x, y):
      x : [N_HIS,  N_NODES, N_FEAT]  — finestra d'entrada
      y : [N_PRED, N_NODES, N_FEAT]  — frames a predir
    """

    def __init__(self, filename: str):
        """
        Args:
            filename: nom del fitxer dins C.OUTPUT_DIR
                      (p.ex. 'seq_train.npy', 'seq_val.npy', 'seq_test.npy')
        """
        super().__init__()
        path = os.path.join(C.OUTPUT_DIR, filename)
        self.data = np.load(path)  # [N_seq, N_FRAME, N_NODES, N_FEAT]

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        sequence = self.data[idx]                                    # [N_FRAME, N_NODES, N_FEAT]
        x = sequence[:C.N_HIS]                                      # [N_HIS,  N_NODES, N_FEAT]
        y = sequence[C.N_HIS:C.N_HIS + C.N_PRED]                   # [N_PRED, N_NODES, N_FEAT]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TrackingDatasetV2(TrackingDataset):
    """
    Estén TrackingDataset afegint 4 features addicionals per node:
      - velocitat (vx, vy): pos[t] - pos[t-1], negativa si el jugador s'allunya
                            el primer frame replica la velocitat del frame 1
      - posició relativa a la pilota (dx_ball, dy_ball): pos[node] - pos[ball]

    Features d'entrada per node: [x, y, vx, vy, dx_ball, dy_ball]  → N_FEAT_V2 = 6
    La sortida y segueix sent [N_PRED, N_NODES, 2] (x, y en coordenades normals).
    """

    BALL_IDX = C.N_NODES - 1   # node 22 és la pilota

    def __getitem__(self, idx: int):
        sequence = self.data[idx]                        # [N_FRAME, N_NODES, 2]
        x = sequence[:C.N_HIS]                          # [N_HIS, N_NODES, 2]
        y = sequence[C.N_HIS:C.N_HIS + C.N_PRED]       # [N_PRED, N_NODES, 2]

        # --- Velocitat: pos[t] - pos[t-1], pot ser negativa ---
        vel = np.diff(x, axis=0)                        # [N_HIS-1, N_NODES, 2]
        vel = np.concatenate([vel[:1], vel], axis=0)    # [N_HIS,   N_NODES, 2]  (replica frame 0)

        # --- Posició relativa a la pilota ---
        ball = x[:, self.BALL_IDX:self.BALL_IDX + 1, :]  # [N_HIS, 1, 2]
        rel_ball = x - ball                               # [N_HIS, N_NODES, 2]

        x_feat = np.concatenate([x, vel, rel_ball], axis=-1)  # [N_HIS, N_NODES, 6]

        return (
            torch.tensor(x_feat, dtype=torch.float32),
            torch.tensor(y,      dtype=torch.float32),
        )


class TrackingDatasetV3(TrackingDatasetV2):
    """
    Estén TrackingDatasetV2 afegint 2 features més: la velocitat de cada
    node *relativa* a la velocitat de la pilota.

      - velocitat relativa a la pilota (dvx_ball, dvy_ball):
            vel[node] - vel[ball]

    Junt amb la posició relativa, completa l'estat cinemàtic de primer
    ordre de cada jugador en el sistema de referència de la pilota.

    Features d'entrada per node:
        [x, y, vx, vy, dx_ball, dy_ball, dvx_ball, dvy_ball]
        → N_FEAT_V3 = 8
    La sortida y segueix sent [N_PRED, N_NODES, 2] (x, y en coordenades normals).
    """

    def __getitem__(self, idx: int):
        sequence = self.data[idx]                        # [N_FRAME, N_NODES, 2]
        x = sequence[:C.N_HIS]                          # [N_HIS, N_NODES, 2]
        y = sequence[C.N_HIS:C.N_HIS + C.N_PRED]       # [N_PRED, N_NODES, 2]

        # --- Velocitat: pos[t] - pos[t-1] ---
        vel = np.diff(x, axis=0)                        # [N_HIS-1, N_NODES, 2]
        vel = np.concatenate([vel[:1], vel], axis=0)    # [N_HIS,   N_NODES, 2]

        # --- Posició relativa a la pilota ---
        ball     = x[:, self.BALL_IDX:self.BALL_IDX + 1, :]    # [N_HIS, 1, 2]
        rel_ball = x - ball                                      # [N_HIS, N_NODES, 2]

        # --- Velocitat relativa a la pilota ---
        vel_ball     = vel[:, self.BALL_IDX:self.BALL_IDX + 1, :]   # [N_HIS, 1, 2]
        rel_vel_ball = vel - vel_ball                                # [N_HIS, N_NODES, 2]

        x_feat = np.concatenate(
            [x, vel, rel_ball, rel_vel_ball], axis=-1,
        )                                                            # [N_HIS, N_NODES, 8]

        return (
            torch.tensor(x_feat, dtype=torch.float32),
            torch.tensor(y,      dtype=torch.float32),
        )