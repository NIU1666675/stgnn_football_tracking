"""
Visualització de les prediccions del model STGCN sobre dades de test.

Funcions principals:
  - predict_sequence(idx)       → retorna (history, gt, pred) en metres
  - plot_snapshot(idx, t_pred)  → camp amb trajectòries + posicions en un instant
  - animate_sequence(idx)       → animació completa de la predicció
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
try:
    from IPython.display import HTML
except ImportError:
    HTML = None

import constants as C
from model import build_model, build_model_dynamic, N_FEAT_V2
from Dataset import TrackingDatasetV2


# ── Constants de camp estàndard (mides típiques) ────────────────────────────
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0

# Colors dels nodes
COLOR_BALL    = 'white'
COLOR_TEAM_A  = '#1f77b4'   # blau
COLOR_TEAM_B  = '#d62728'   # vermell


# ── Càrrega del model i dades ────────────────────────────────────────────────

def _load_resources():
    """Carrega model, dades de test i mean_std. Retorna (model, seq_test, mean_std)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(device)
    ckpt_path = os.path.join(C.OUTPUT_DIR, 'best_model.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    seq_test = np.load(os.path.join(C.OUTPUT_DIR, 'seq_test.npy'), allow_pickle=True)
    mean_std = np.load(os.path.join(C.OUTPUT_DIR, 'mean_std.npy'))
    mean_xy, std_xy = mean_std[0], mean_std[1]

    return model, seq_test, mean_xy, std_xy, device


_cache = {}

def _resources():
    if not _cache:
        _cache['model'], _cache['seq_test'], _cache['mean'], _cache['std'], _cache['device'] = _load_resources()
    return _cache['model'], _cache['seq_test'], _cache['mean'], _cache['std'], _cache['device']


def _load_resources_dynamic():
    """Carrega el model dinàmic (best_model_dynamic.pt) i les dades de test."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model_dynamic(device, c_in=N_FEAT_V2)
    ckpt_path = os.path.join(C.OUTPUT_DIR, 'best_model_dynamic.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    seq_test = np.load(os.path.join(C.OUTPUT_DIR, 'seq_test.npy'), allow_pickle=True)
    mean_std = np.load(os.path.join(C.OUTPUT_DIR, 'mean_std.npy'))

    return model, seq_test, mean_std[0], mean_std[1], device


_cache_dynamic = {}

def _resources_dynamic():
    if not _cache_dynamic:
        (_cache_dynamic['model'], _cache_dynamic['seq_test'],
         _cache_dynamic['mean'],  _cache_dynamic['std'],
         _cache_dynamic['device']) = _load_resources_dynamic()
    return (_cache_dynamic['model'], _cache_dynamic['seq_test'],
            _cache_dynamic['mean'],  _cache_dynamic['std'],
            _cache_dynamic['device'])


def _make_v2_input(seq_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """
    A partir d'una seqüència normalitzada [N_FRAME, N_NODES, 2] construeix
    el tensor d'entrada [1, N_HIS, N_NODES, 6] per al model dinàmic.
    """
    x = seq_norm[:C.N_HIS]                              # [N_HIS, N_NODES, 2]
    vel      = np.diff(x, axis=0)
    vel      = np.concatenate([vel[:1], vel], axis=0)   # [N_HIS, N_NODES, 2]
    ball     = x[:, C.N_NODES - 1:C.N_NODES, :]
    rel_ball = x - ball                                 # [N_HIS, N_NODES, 2]
    x_feat   = np.concatenate([x, vel, rel_ball], axis=-1)  # [N_HIS, N_NODES, 6]
    return torch.tensor(x_feat, dtype=torch.float32).unsqueeze(0)


def _denorm(arr, mean, std):
    """arr: [..., 2] normalitzat → metres reals."""
    return arr * std + mean


# ── Inferència ───────────────────────────────────────────────────────────────

def predict_both(idx: int):
    """
    Executa els dos models sobre la mateixa seqüència idx.

    Retorna:
        history  : [N_HIS,  N_NODES, 2]
        gt       : [N_PRED, N_NODES, 2]
        pred_sta : [N_PRED, N_NODES, 2]  — STGCN estàtic
        pred_dyn : [N_PRED, N_NODES, 2]  — STGCNDynamic
    """
    # Estàtic
    model_s, seq_test, mean, std, device = _resources()
    seq  = seq_test[idx]
    x_s  = torch.tensor(seq[:C.N_HIS], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out_s = model_s(x_s).squeeze(0).cpu().numpy()

    # Dinàmic
    model_d, _, _, _, device_d = _resources_dynamic()
    x_d = _make_v2_input(seq, mean, std).to(device_d)
    with torch.no_grad():
        out_d = model_d(x_d).squeeze(0).cpu().numpy()

    history  = _denorm(seq[:C.N_HIS],                    mean, std)
    gt       = _denorm(seq[C.N_HIS:C.N_HIS + C.N_PRED],  mean, std)
    pred_sta = _denorm(out_s,                             mean, std)
    pred_dyn = _denorm(out_d,                             mean, std)

    return history, gt, pred_sta, pred_dyn


def predict_sequence(idx: int):
    """
    Executa el model sobre la seqüència idx del test set.

    Retorna:
        history : [N_HIS,  N_NODES, 2]  — posicions d'entrada (metres)
        gt      : [N_PRED, N_NODES, 2]  — futur real (metres)
        pred    : [N_PRED, N_NODES, 2]  — predicció del model (metres)
    """
    model, seq_test, mean, std, device = _resources()

    seq = seq_test[idx]                                        # [N_FRAME, N_NODES, 2]
    x   = torch.tensor(seq[:C.N_HIS], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)                                         # [1, N_PRED, N_NODES, 2]

    history = _denorm(seq[:C.N_HIS],                          mean, std)
    gt      = _denorm(seq[C.N_HIS:C.N_HIS + C.N_PRED],       mean, std)
    pred    = _denorm(out.squeeze(0).cpu().numpy(),            mean, std)

    return history, gt, pred


# ── Dibuix del camp ──────────────────────────────────────────────────────────

def _draw_pitch(ax, length=PITCH_LENGTH, width=PITCH_WIDTH):
    ax.set_facecolor('#2e8b57')
    ax.set_xlim(-length / 2 - 3, length / 2 + 3)
    ax.set_ylim(-width  / 2 - 3, width  / 2 + 3)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    hl, hw = length / 2, width / 2
    # Perímetre i línia central
    ax.plot([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw], 'w-', lw=1.5)
    ax.plot([0, 0], [-hw, hw], 'w-', lw=1.5)
    # Cercle central
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color='white', lw=1.5))
    # Àrees
    for s in [-1, 1]:
        ax.plot([s*hl, s*(hl-16.5), s*(hl-16.5), s*hl],
                [-20.16, -20.16, 20.16, 20.16], 'w-', lw=1.5)
        ax.plot([s*hl, s*(hl-5.5), s*(hl-5.5), s*hl],
                [-9.16, -9.16, 9.16, 9.16], 'w-', lw=1.5)


def _node_colors():
    """Retorna llista de colors: primers 11 equip A, 11-21 equip B, 22 balón."""
    return [COLOR_TEAM_A] * 11 + [COLOR_TEAM_B] * 11 + [COLOR_BALL]


# ── Visualització estàtica ───────────────────────────────────────────────────

def plot_snapshot(idx: int, t_pred: int = 0):
    """
    Dibuixa l'últim frame de la historia + les trajectòries fins a t_pred.

    Args:
        idx    : índex de la seqüència de test
        t_pred : frame de predicció a mostrar (0 = primer frame futur,
                 C.N_PRED-1 = últim)
    """
    history, gt, pred = predict_sequence(idx)
    colors = _node_colors()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    titles = ['Ground truth', 'Predicció STGCN']
    futures = [gt, pred]

    for ax, title, future in zip(axes, titles, futures):
        _draw_pitch(ax)
        ax.set_title(title, fontsize=13, fontweight='bold', color='white', pad=8)

        for node in range(C.N_NODES):
            c = colors[node]
            ms = 6 if node < C.N_PLAYERS else 9   # balón más gran

            # Trajectòria de la historia (últims 10 frames)
            hx = history[-10:, node, 0]
            hy = history[-10:, node, 1]
            ax.plot(hx, hy, '-', color=c, alpha=0.3, lw=1)

            # Posició inicial (últim frame historia)
            ax.plot(history[-1, node, 0], history[-1, node, 1],
                    'o', color=c, ms=ms, markeredgecolor='black', markeredgewidth=1, zorder=5)

            # Trajectòria futura fins a t_pred
            fx = future[:t_pred+1, node, 0]
            fy = future[:t_pred+1, node, 1]
            ax.plot(fx, fy, '--', color=c, alpha=0.7, lw=1.5)

            # Posició en t_pred
            ax.plot(future[t_pred, node, 0], future[t_pred, node, 1],
                    's' if node < C.N_PLAYERS else '*',
                    color=c, ms=ms, markeredgecolor='white', markeredgewidth=0.8, zorder=6)

    # Llegenda
    legend_elements = [
        mpatches.Patch(color=COLOR_TEAM_A, label='Equip A'),
        mpatches.Patch(color=COLOR_TEAM_B, label='Equip B'),
        mpatches.Patch(color='gray',       label='Balón'),
        plt.Line2D([0], [0], color='white', ls='-',  lw=1, alpha=0.5, label='Historia'),
        plt.Line2D([0], [0], color='white', ls='--', lw=1.5,          label='Futur'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9,
                   facecolor='#1a1a1a', labelcolor='white')

    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(
        f'Test seq {idx}  |  t_pred = {t_pred+1}/{C.N_PRED} frames'
        f' ({(t_pred+1)/10:.1f}s en el futur)',
        fontsize=13, color='white', y=1.01
    )
    plt.tight_layout()
    plt.show()


# ── Animació ─────────────────────────────────────────────────────────────────

def animate_sequence(idx: int, interval: int = 100):
    """
    Anima les N_PRED frames futures de la seqüència idx.
    Mostra GT (esquerra) vs predicció (dreta) frame a frame.

    Args:
        idx      : índex de la seqüència de test
        interval : mil·lisegons entre frames

    Retorna un objecte HTML per mostrar a Jupyter/Colab.
    """
    history, gt, pred = predict_sequence(idx)
    colors = _node_colors()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#1a1a1a')
    for ax in axes:
        _draw_pitch(ax)

    # Dibuixar historia (estàtica durant tota l'animació)
    for ax in axes:
        for node in range(C.N_NODES):
            c = colors[node]
            ax.plot(history[-10:, node, 0], history[-10:, node, 1],
                    '-', color=c, alpha=0.25, lw=1)
            ax.plot(history[-1, node, 0], history[-1, node, 1],
                    'o', color=c, ms=6 if node < C.N_PLAYERS else 9,
                    markeredgecolor='black', markeredgewidth=1, alpha=0.6, zorder=4)

    # Elements mòbils (es recreen a cada frame)
    scat_gt   = [axes[0].plot([], [], 'o', color=colors[n],
                              ms=6 if n < C.N_PLAYERS else 9,
                              markeredgecolor='white', markeredgewidth=0.8, zorder=6)[0]
                 for n in range(C.N_NODES)]
    scat_pred = [axes[1].plot([], [], 'o', color=colors[n],
                              ms=6 if n < C.N_PLAYERS else 9,
                              markeredgecolor='white', markeredgewidth=0.8, zorder=6)[0]
                 for n in range(C.N_NODES)]
    trail_gt   = [axes[0].plot([], [], '--', color=colors[n], alpha=0.5, lw=1)[0]
                  for n in range(C.N_NODES)]
    trail_pred = [axes[1].plot([], [], '--', color=colors[n], alpha=0.5, lw=1)[0]
                  for n in range(C.N_NODES)]

    title_gt   = axes[0].set_title('Ground truth',      fontsize=13, fontweight='bold', color='white')
    title_pred = axes[1].set_title('Predicció STGCN',   fontsize=13, fontweight='bold', color='white')
    fig_title  = fig.suptitle('', fontsize=12, color='white')

    TRAIL = 5  # frames de traça visible

    def update(t):
        for n in range(C.N_NODES):
            # Posicions
            scat_gt[n].set_data([gt[t, n, 0]],   [gt[t, n, 1]])
            scat_pred[n].set_data([pred[t, n, 0]], [pred[t, n, 1]])
            # Traça
            t0 = max(0, t - TRAIL)
            trail_gt[n].set_data(gt[t0:t+1, n, 0],   gt[t0:t+1, n, 1])
            trail_pred[n].set_data(pred[t0:t+1, n, 0], pred[t0:t+1, n, 1])

        mae_t = np.abs(gt[t] - pred[t]).mean()
        fig_title.set_text(
            f'Test seq {idx}  |  frame futur {t+1}/{C.N_PRED} ({(t+1)/10:.1f}s)'
            f'  |  MAE instant = {mae_t:.2f} m'
        )
        return scat_gt + scat_pred + trail_gt + trail_pred + [fig_title]

    ani = FuncAnimation(fig, update, frames=C.N_PRED, interval=interval, blit=True)
    plt.tight_layout()
    plt.close(fig)
    return HTML(ani.to_jshtml())

# ── Comparació dels dos models ───────────────────────────────────────────────

def plot_comparison(indices, t_pred: int = 29):
    """
    Per a cada idx de la llista, mostra una fila amb 3 camps:
      GT  |  STGCN estàtic  |  STGCN dinàmic

    Args:
        indices : llista d'índexs de test, p.ex. [42, 100, 500]
        t_pred  : frame futur a visualitzar (0-29)
    """
    n_rows = len(indices)

    fig, axes = plt.subplots(n_rows, 3, figsize=(22, 7 * n_rows))
    if n_rows == 1:
        axes = [axes]   # garantir sempre llista de files

    col_titles = ['Ground truth', 'STGCN estàtic', 'STGCN dinàmic']

    for row, idx in enumerate(indices):
        history, gt, pred_sta, pred_dyn = predict_both(idx)
        colors = _node_colors()
        mae_sta = np.abs(gt - pred_sta).mean()
        mae_dyn = np.abs(gt - pred_dyn).mean()
        futures = [gt, pred_sta, pred_dyn]

        for col, (ax, future) in enumerate(zip(axes[row], futures)):
            _draw_pitch(ax)

            for node in range(C.N_NODES):
                c  = colors[node]
                ms = 6 if node < C.N_PLAYERS else 9

                ax.plot(history[-10:, node, 0], history[-10:, node, 1],
                        '-', color=c, alpha=0.25, lw=1)
                ax.plot(history[-1, node, 0], history[-1, node, 1],
                        'o', color=c, ms=ms, markeredgecolor='black', markeredgewidth=1, zorder=5)
                ax.plot(future[:t_pred+1, node, 0], future[:t_pred+1, node, 1],
                        '--', color=c, alpha=0.7, lw=1.5)
                ax.plot(future[t_pred, node, 0], future[t_pred, node, 1],
                        's' if node < C.N_PLAYERS else '*',
                        color=c, ms=ms, markeredgecolor='white', markeredgewidth=0.8, zorder=6)

            subtitle = col_titles[col]
            if col == 1:
                subtitle += f'\nMAE={mae_sta:.2f} m'
            elif col == 2:
                subtitle += f'\nMAE={mae_dyn:.2f} m'
            ax.set_title(subtitle, fontsize=11, fontweight='bold', color='white', pad=6)

        axes[row][0].set_ylabel(f'seq {idx}', color='white', fontsize=10)

    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(
        f't_pred = {t_pred+1}/{C.N_PRED} frames ({(t_pred+1)/10:.1f}s)',
        fontsize=13, color='white'
    )
    plt.tight_layout()
    plt.show()


def evaluate_both(n_samples: int = None):
    """
    Evalua els dos models sobre tot el test set (o n_samples mostres aleatòries).

    Imprimeix i retorna un dict amb:
      - MAE global
      - MAE per horitzó temporal (30 valors)
      - ADE  (Average Displacement Error = MAE global sobre tots els nodes i frames)
      - FDE  (Final Displacement Error   = error euclidià en l'últim frame)
    """
    _, seq_test, _, _, _ = _resources()
    n_total = len(seq_test)
    idxs    = (np.random.choice(n_total, n_samples, replace=False)
               if n_samples else np.arange(n_total))

    mae_per_step_s = np.zeros(C.N_PRED)
    mae_per_step_d = np.zeros(C.N_PRED)
    fde_s_list, fde_d_list = [], []

    for idx in idxs:
        _, gt, pred_sta, pred_dyn = predict_both(int(idx))
        # MAE per step: mitjana sobre nodes i feat
        mae_per_step_s += np.abs(gt - pred_sta).mean(axis=(1, 2))
        mae_per_step_d += np.abs(gt - pred_dyn).mean(axis=(1, 2))
        # FDE: error euclidià en l'últim frame, per node, després mitjana
        fde_s_list.append(np.linalg.norm(gt[-1] - pred_sta[-1], axis=-1).mean())
        fde_d_list.append(np.linalg.norm(gt[-1] - pred_dyn[-1], axis=-1).mean())

    n = len(idxs)
    mae_per_step_s /= n
    mae_per_step_d /= n

    results = {
        'ADE_static':   mae_per_step_s.mean(),
        'ADE_dynamic':  mae_per_step_d.mean(),
        'FDE_static':   np.mean(fde_s_list),
        'FDE_dynamic':  np.mean(fde_d_list),
        'mae_per_step_static':  mae_per_step_s,
        'mae_per_step_dynamic': mae_per_step_d,
    }

    # ── Taula resum ──
    print(f"{'Mètrica':<22} {'Estàtic':>12} {'Dinàmic':>12}  {'Millora':>10}")
    print("-" * 60)
    for key in ('ADE', 'FDE'):
        s = results[f'{key}_static']
        d = results[f'{key}_dynamic']
        print(f"{key:<22} {s:>11.3f}m {d:>11.3f}m  {(s-d)/s*100:>+9.1f}%")

    # ── Gràfic MAE per horitzó ──
    t_axis = np.arange(1, C.N_PRED + 1) / 10
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_axis, mae_per_step_s, label='Estàtic',  color='#1f77b4', lw=2)
    ax.plot(t_axis, mae_per_step_d, label='Dinàmic',  color='#ff7f0e', lw=2)
    ax.fill_between(t_axis, mae_per_step_s, mae_per_step_d,
                    alpha=0.15, color='green' if results['ADE_dynamic'] < results['ADE_static'] else 'red')
    ax.set_xlabel('Segon futur')
    ax.set_ylabel('MAE (m)')
    ax.set_title('Error per horitzó temporal — Estàtic vs Dinàmic')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    return results


if __name__ == '__main__':
    plot_comparison([350, 500, 1000], t_pred=29)
    evaluate_both(n_samples=500)