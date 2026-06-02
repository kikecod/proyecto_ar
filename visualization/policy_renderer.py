"""
visualization/policy_renderer.py — Heatmap de la política aprendida

Genera una tabla 2D con:
    - Eje X: carta visible del dealer (1–10)
    - Eje Y: suma del jugador (12–21)
    - Color: VERDE = PEDIR (Hit), ROJO = PLANTARSE (Stand)

Muestra dos subtablas: sin As utilizable / con As utilizable.
Permite comparar visualmente con la Basic Strategy de casino.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import config


# ─── Basic Strategy de referencia (fuente: Wizard of Odds) ───────────────────
# Filas: suma del jugador 12–21 (índice 0=12, ..., 9=21)
# Columnas: carta del dealer 1–10 (As=1, 2, ..., 10)
# 1 = PEDIR (Hit), 0 = PLANTARSE (Stand)
BASIC_STRATEGY_HARD = np.array([
    # As  2  3  4  5  6  7  8  9 10
    [1,  1, 1, 0, 0, 0, 1, 1, 1, 1],  # 12
    [0,  0, 0, 0, 0, 0, 1, 1, 1, 1],  # 13
    [0,  0, 0, 0, 0, 0, 1, 1, 1, 1],  # 14
    [0,  0, 0, 0, 0, 0, 1, 1, 1, 1],  # 15
    [0,  0, 0, 0, 0, 0, 1, 1, 1, 1],  # 16
    [0,  0, 0, 0, 0, 0, 0, 0, 0, 0],  # 17
    [0,  0, 0, 0, 0, 0, 0, 0, 0, 0],  # 18
    [0,  0, 0, 0, 0, 0, 0, 0, 0, 0],  # 19
    [0,  0, 0, 0, 0, 0, 0, 0, 0, 0],  # 20
    [0,  0, 0, 0, 0, 0, 0, 0, 0, 0],  # 21
])

BASIC_STRATEGY_SOFT = np.array([
    # As  2  3  4  5  6  7  8  9 10
    [1,  1, 1, 1, 1, 1, 1, 1, 1, 1],  # 12 (A+A)
    [1,  1, 1, 1, 1, 1, 1, 1, 1, 1],  # 13 (A+2)
    [1,  1, 1, 1, 1, 1, 1, 1, 1, 1],  # 14 (A+3)
    [1,  1, 1, 1, 0, 0, 1, 1, 1, 1],  # 15 (A+4)
    [1,  1, 1, 0, 0, 0, 1, 1, 1, 1],  # 16 (A+5)
    [1,  0, 0, 0, 0, 0, 1, 1, 1, 1],  # 17 (A+6)
    [0,  0, 0, 0, 0, 0, 0, 0, 1, 1],  # 18 (A+7)
    [0,  0, 0, 0, 0, 0, 0, 0, 0, 0],  # 19 (A+8)
    [0,  0, 0, 0, 0, 0, 0, 0, 0, 0],  # 20 (A+9)
    [0,  0, 0, 0, 0, 0, 0, 0, 0, 0],  # 21 (A+10)
])


def _extract_policy_grid(policy: dict, usable_ace: bool) -> np.ndarray:
    """
    Construye la grilla 10×10 (suma 12–21 × dealer 1–10) desde la política.

    Valores:
        1.0  = PEDIR   (Hit)
        0.0  = PLANTARSE (Stand)
        0.5  = estado no visitado (indefinido)
    """
    player_sums  = list(range(12, 22))   # 12..21
    dealer_cards = list(range(1, 11))    # 1..10 (As=1)
    grid = np.full((len(player_sums), len(dealer_cards)), 0.5)

    for i, ps in enumerate(player_sums):
        for j, dc in enumerate(dealer_cards):
            state = (ps, dc, usable_ace)
            if state in policy:
                grid[i, j] = float(policy[state])

    return grid


def _plot_single_heatmap(ax, grid: np.ndarray, ref_grid: np.ndarray,
                         title: str, show_cbar: bool = False):
    """Dibuja un heatmap individual con anotaciones y comparación."""

    # Colormap personalizado: rojo (Stand) → verde (Hit), gris = no visitado
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'bj_policy',
        [(0.0, '#C0392B'), (0.49, '#C0392B'),
         (0.50, '#808080'),
         (0.51, '#27AE60'), (1.0, '#27AE60')],
    )

    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1,
                   aspect='auto', interpolation='nearest')

    player_sums  = list(range(12, 22))
    dealer_cards = ['As', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(dealer_cards, fontsize=9, color='white')
    ax.set_yticklabels(player_sums, fontsize=9, color='white')
    ax.set_xlabel('Carta Visible del Dealer', fontsize=10, color='white', labelpad=6)
    ax.set_ylabel('Suma del Jugador', fontsize=10, color='white', labelpad=6)
    ax.set_title(title, fontsize=11, fontweight='bold', color='white', pad=8)

    # Anotaciones en cada celda
    for i in range(10):
        for j in range(10):
            val = grid[i, j]
            ref = ref_grid[i, j]

            if val == 0.5:
                txt, color = '?', '#AAAAAA'
            elif val == 1.0:
                txt = 'H'  # Hit
                color = 'white'
            else:
                txt = 'S'  # Stand
                color = 'white'

            # Indicador de diferencia con Basic Strategy
            match_sym = '' if val == 0.5 else ('✓' if val == ref else '✗')
            match_clr = '#7DFFB3' if match_sym == '✓' else ('#FF6B6B' if match_sym == '✗' else '#888888')

            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)
            ax.text(j + 0.35, i - 0.35, match_sym, ha='center', va='center',
                    fontsize=7, color=match_clr)

    # Borde de celdas
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which='minor', color='#333355', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)


def plot_policy_heatmap(policy: dict, agent_name: str, output_path: str):
    """
    Genera la Gráfica 4 — Heatmap de la política aprendida.

    Layout: 2 subtablas (sin As / con As) + leyenda + comparación BS.
    """
    BG_COLOR    = '#1A1A2E'
    PANEL_COLOR = '#16213E'

    plt.rcParams.update({
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': PANEL_COLOR,
        'text.color': 'white',
        'font.size': 10,
    })

    grid_hard = _extract_policy_grid(policy, usable_ace=False)
    grid_soft = _extract_policy_grid(policy, usable_ace=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG_COLOR)
    fig.suptitle(
        f'Política Aprendida — {agent_name}\n'
        f'H = Pedir (Hit) · S = Plantarse (Stand) · ✓ = Coincide con Basic Strategy · ✗ = Difiere',
        fontsize=13, fontweight='bold', color='white', y=1.03
    )

    _plot_single_heatmap(
        axes[0], grid_hard, BASIC_STRATEGY_HARD,
        'Sin As Utilizable (Hard Hand)', show_cbar=False
    )
    _plot_single_heatmap(
        axes[1], grid_soft, BASIC_STRATEGY_SOFT,
        'Con As Utilizable (Soft Hand)', show_cbar=True
    )

    # Leyenda global
    legend_elements = [
        Patch(facecolor='#27AE60', label='H — PEDIR (Hit)'),
        Patch(facecolor='#C0392B', label='S — PLANTARSE (Stand)'),
        Patch(facecolor='#808080', label='? — Estado no visitado'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center', ncol=3,
        fontsize=10, framealpha=0.3,
        facecolor='#222244', edgecolor='#444466',
        bbox_to_anchor=(0.5, -0.05)
    )

    # Calcular coincidencia con Basic Strategy
    def pct_match(grid, ref):
        visited = grid != 0.5
        if visited.sum() == 0:
            return 0.0
        return (grid[visited] == ref[visited]).mean() * 100

    match_hard = pct_match(grid_hard, BASIC_STRATEGY_HARD)
    match_soft = pct_match(grid_soft, BASIC_STRATEGY_SOFT)

    fig.text(
        0.5, -0.02,
        f'Coincidencia con Basic Strategy →  '
        f'Hard: {match_hard:.1f}%  |  Soft: {match_soft:.1f}%  |  '
        f'Promedio: {(match_hard + match_soft) / 2:.1f}%',
        ha='center', fontsize=11, color='#F0E68C', fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f"  ✔ Gráfica guardada: {output_path}")

    return match_hard, match_soft
