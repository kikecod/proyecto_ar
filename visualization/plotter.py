"""
visualization/plotter.py — Gráficas de aprendizaje con matplotlib

Genera:
    - Curva de aprendizaje individual (tasa móvil + histograma)
    - Comparación Q-Learning vs SARSA (curvas juntas + boxplot + tabla)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin pantalla (headless)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

import config


# ─── Paleta de colores ────────────────────────────────────────────────────────
COLOR_QL    = '#4C9BE8'   # Azul para Q-Learning
COLOR_SARSA = '#F5A623'   # Naranja para SARSA
COLOR_WIN   = '#2ECC71'   # Verde victorias
COLOR_LOSE  = '#E74C3C'   # Rojo derrotas
COLOR_DRAW  = '#95A5A6'   # Gris empates
BG_COLOR    = '#1A1A2E'   # Fondo oscuro
PANEL_COLOR = '#16213E'   # Paneles
GRID_COLOR  = '#2A2A4A'   # Cuadrícula


def _setup_dark_style():
    """Configura el estilo oscuro global de matplotlib."""
    plt.rcParams.update({
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': PANEL_COLOR,
        'axes.edgecolor': '#444477',
        'axes.labelcolor': '#E0E0F0',
        'axes.titlecolor': '#FFFFFF',
        'text.color': '#E0E0F0',
        'xtick.color': '#B0B0D0',
        'ytick.color': '#B0B0D0',
        'grid.color': GRID_COLOR,
        'grid.alpha': 0.5,
        'legend.facecolor': '#16213E',
        'legend.edgecolor': '#444477',
        'legend.labelcolor': '#E0E0F0',
        'font.family': 'DejaVu Sans',
        'font.size': 10,
    })


def _moving_average(data, window):
    """Calcula la media móvil de una lista de valores."""
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_training_curve(rewards: list, agent_name: str, output_path: str):
    """
    Gráfica 1/2 — Curva de aprendizaje individual.

    Paneles:
        - Izquierda: tasa de victorias móvil (ventana 500)
        - Derecha: histograma de resultados (win/lose/draw)
    """
    _setup_dark_style()

    color = COLOR_QL if 'Q-Learning' in agent_name else COLOR_SARSA

    # Calcular métricas
    wins   = [1 if r == 1.0 else 0 for r in rewards]
    losses = [1 if r == -1.0 else 0 for r in rewards]
    draws  = [1 if r == 0.0 else 0 for r in rewards]

    win_rate_full  = _moving_average(wins, config.WINDOW_SIZE)
    episodes_x     = np.arange(config.WINDOW_SIZE, len(wins) + 1)

    total_wins   = sum(wins)
    total_losses = sum(losses)
    total_draws  = sum(draws)
    total_ep     = len(rewards)

    fig = plt.figure(figsize=(14, 6), facecolor=BG_COLOR)
    fig.suptitle(
        f'Curva de Aprendizaje — {agent_name}',
        fontsize=16, fontweight='bold', color='white', y=1.02
    )

    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.3)

    # ── Panel izquierdo: curva de victorias ──
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(episodes_x, win_rate_full, alpha=0.15, color=color)
    ax1.plot(episodes_x, win_rate_full, color=color, linewidth=1.5,
             label=f'Win Rate (ventana {config.WINDOW_SIZE})')

    # Línea de referencia: Basic Strategy ≈ 42%
    ax1.axhline(0.42, color='#F0E68C', linewidth=1, linestyle='--', alpha=0.7,
                label='Basic Strategy ≈ 42%')

    # Resaltar últimos 50k
    last_n = config.EVAL_LAST_N
    if len(episodes_x) > last_n:
        ax1.axvspan(total_ep - last_n, total_ep, alpha=0.08, color='white',
                    label=f'Últimos {last_n//1000}k ep')

    ax1.set_xlabel('Episodio', fontsize=11)
    ax1.set_ylabel('Tasa de Victorias', fontsize=11)
    ax1.set_title(f'Tasa de Victorias Móvil — {agent_name}', fontsize=12)
    ax1.set_ylim(0, 0.70)
    ax1.set_xlim(config.WINDOW_SIZE, total_ep)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.grid(True, alpha=0.4)
    ax1.legend(loc='upper left', fontsize=9)

    # Anotación del win rate final
    final_wr = float(np.mean(win_rate_full[-min(1000, len(win_rate_full)):]))
    ax1.annotate(
        f'Final: {final_wr:.1%}',
        xy=(total_ep, final_wr),
        xytext=(-80, 20), textcoords='offset points',
        fontsize=10, color='white', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='white', lw=1.2),
    )

    # ── Panel derecho: histograma ──
    ax2 = fig.add_subplot(gs[1])
    categories = ['Victorias', 'Derrotas', 'Empates']
    counts     = [total_wins, total_losses, total_draws]
    colors_bar = [COLOR_WIN, COLOR_LOSE, COLOR_DRAW]

    bars = ax2.bar(categories, counts, color=colors_bar, width=0.6,
                   edgecolor='#333355', linewidth=0.8)

    for bar, count in zip(bars, counts):
        pct = count / total_ep * 100
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total_ep * 0.005,
            f'{pct:.1f}%', ha='center', va='bottom',
            fontsize=9, color='white', fontweight='bold'
        )

    ax2.set_title('Distribución\nde Resultados', fontsize=11)
    ax2.set_ylabel('Cantidad', fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/1000:.0f}k'))
    ax2.grid(True, axis='y', alpha=0.4)
    ax2.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f"  ✔ Gráfica guardada: {output_path}")


def plot_comparison(ql_rewards: list, sarsa_rewards: list, output_path: str):
    """
    Gráfica 3 — Comparación Q-Learning vs SARSA.

    Paneles:
        - Superior izquierda : curvas de win rate juntas
        - Superior derecha   : boxplot de últimos 2000 episodios
        - Inferior           : tabla de métricas clave
    """
    _setup_dark_style()

    # Win rates móviles
    ql_wins    = [1 if r == 1.0 else 0 for r in ql_rewards]
    sarsa_wins = [1 if r == 1.0 else 0 for r in sarsa_rewards]

    ql_curve    = _moving_average(ql_wins, config.WINDOW_SIZE)
    sarsa_curve = _moving_average(sarsa_wins, config.WINDOW_SIZE)
    x_ql        = np.arange(config.WINDOW_SIZE, len(ql_wins) + 1)
    x_sarsa     = np.arange(config.WINDOW_SIZE, len(sarsa_wins) + 1)

    # Datos para boxplot (últimos 2000 episodios, ventanas de 100)
    def windowed_rates(wins_list, n=2000, w=100):
        tail = wins_list[-n:]
        return [np.mean(tail[i:i+w]) for i in range(0, len(tail), w)]

    ql_box    = windowed_rates(ql_wins)
    sarsa_box = windowed_rates(sarsa_wins)

    # Métricas generales
    def metrics(rewards_list):
        wins_  = sum(1 for r in rewards_list if r == 1.0)
        losses = sum(1 for r in rewards_list if r == -1.0)
        draws  = sum(1 for r in rewards_list if r == 0.0)
        n      = len(rewards_list)
        # Últimos 50k
        last   = rewards_list[-config.EVAL_LAST_N:]
        last_wr = sum(1 for r in last if r == 1.0) / len(last)
        return {
            'win_rate_total': wins_ / n,
            'loss_rate_total': losses / n,
            'draw_rate_total': draws / n,
            'win_rate_last50k': last_wr,
            'avg_reward': np.mean(rewards_list),
        }

    ql_m    = metrics(ql_rewards)
    sarsa_m = metrics(sarsa_rewards)

    # ── Layout ──
    fig = plt.figure(figsize=(16, 10), facecolor=BG_COLOR)
    fig.suptitle(
        'Comparación: Q-Learning vs SARSA — Blackjack RL',
        fontsize=17, fontweight='bold', color='white', y=1.01
    )

    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.4, wspace=0.3)

    # ── Curvas juntas ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(x_ql, ql_curve, alpha=0.12, color=COLOR_QL)
    ax1.fill_between(x_sarsa, sarsa_curve, alpha=0.12, color=COLOR_SARSA)
    ax1.plot(x_ql, ql_curve, color=COLOR_QL, linewidth=1.5, label='Q-Learning')
    ax1.plot(x_sarsa, sarsa_curve, color=COLOR_SARSA, linewidth=1.5, label='SARSA')
    ax1.axhline(0.42, color='#F0E68C', linewidth=1, linestyle='--',
                alpha=0.7, label='Basic Strategy ≈ 42%')
    ax1.set_xlabel('Episodio', fontsize=11)
    ax1.set_ylabel('Tasa de Victorias', fontsize=11)
    ax1.set_title('Curvas de Aprendizaje Comparadas', fontsize=12)
    ax1.set_ylim(0, 0.70)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.grid(True, alpha=0.4)
    ax1.legend(loc='upper left', fontsize=9)

    # ── Boxplot ──
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot(
        [ql_box, sarsa_box],
        labels=['Q-Learning', 'SARSA'],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color='#8888AA'),
        capprops=dict(color='#8888AA'),
        flierprops=dict(marker='o', markersize=4, markerfacecolor='#8888AA',
                        linestyle='none'),
    )
    bp['boxes'][0].set_facecolor(COLOR_QL + '88')
    bp['boxes'][1].set_facecolor(COLOR_SARSA + '88')
    ax2.set_title(f'Distribución Win Rate\n(últimos 2 000 episodios)', fontsize=11)
    ax2.set_ylabel('Tasa de Victorias', fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(True, axis='y', alpha=0.4)

    # ── Tabla de métricas ──
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    table_data = [
        ['Métrica', 'Q-Learning', 'SARSA'],
        ['Win Rate Total',
         f"{ql_m['win_rate_total']:.2%}", f"{sarsa_m['win_rate_total']:.2%}"],
        ['Loss Rate Total',
         f"{ql_m['loss_rate_total']:.2%}", f"{sarsa_m['loss_rate_total']:.2%}"],
        ['Draw Rate Total',
         f"{ql_m['draw_rate_total']:.2%}", f"{sarsa_m['draw_rate_total']:.2%}"],
        [f'Win Rate Últimos {config.EVAL_LAST_N//1000}k ep',
         f"{ql_m['win_rate_last50k']:.2%}", f"{sarsa_m['win_rate_last50k']:.2%}"],
        ['Recompensa Promedio',
         f"{ql_m['avg_reward']:.4f}", f"{sarsa_m['avg_reward']:.4f}"],
        ['Episodios Totales',
         f"{len(ql_rewards):,}", f"{len(sarsa_rewards):,}"],
    ]

    tbl = ax3.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    # Estilo de la tabla
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2A2A5A')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 1:
            cell.set_facecolor(COLOR_QL + '33')
            cell.set_text_props(color='#C8E0FF')
        elif col == 2:
            cell.set_facecolor(COLOR_SARSA + '33')
            cell.set_text_props(color='#FFE0A0')
        else:
            cell.set_facecolor(PANEL_COLOR)
            cell.set_text_props(color='#D0D0E8')
        cell.set_edgecolor('#333355')

    ax3.set_title('Tabla Resumen de Métricas', fontsize=12, pad=15)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f"  ✔ Gráfica guardada: {output_path}")
