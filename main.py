"""
main.py — Punto de entrada del proyecto Blackjack RL

Uso:
    python3 main.py                        # Entrenar ambos agentes (500k ep)
    python3 main.py --episodes 200000      # Cambiar número de episodios
    python3 main.py --agent qlearning      # Solo Q-Learning
    python3 main.py --agent sarsa          # Solo SARSA
"""

import argparse
import time
import os
import sys

import config
from environment.blackjack_env import BlackjackEnv
from agent.q_learning import QLearningAgent
from agent.sarsa import SARSAAgent
from visualization.plotter import plot_training_curve, plot_comparison
from visualization.policy_renderer import plot_policy_heatmap
from utils.saver import save_scores, save_win_rates_comparison

os.makedirs(config.DATA_DIR, exist_ok=True)


# ─── Utilidades de terminal ───────────────────────────────────────────────────

def _header(text: str):
    width = 68
    print('\n' + '═' * width)
    print(f'  {text}')
    print('═' * width)


def _section(text: str):
    print(f'\n  ── {text}')


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f'{h}h {m}m {s}s'
    return f'{m}m {s}s' if m else f'{s}s'


# ─── Bucle de entrenamiento — Q-Learning ─────────────────────────────────────

def train_q_learning(episodes: int):
    """
    Entrena el agente Q-Learning durante `episodes` episodios.
    
    Retorna:
        rewards     : lista de recompensas por episodio
        records     : lista de dicts para el CSV de scores
        win_rates   : lista de (episodio, win_rate) cada WIN_RATE_INTERVAL
        agent       : agente entrenado (con Q-table)
    """
    _header('ENTRENAMIENTO — Q-Learning (off-policy)')

    env   = BlackjackEnv()
    agent = QLearningAgent()

    rewards   = []
    records   = []
    win_rates = []

    # Ventana deslizante para logging
    recent_wins    = []
    recent_rewards = []

    t_start = time.time()

    for ep in range(1, episodes + 1):
        state = env.reset()
        done  = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward = reward  # En Blackjack el reward relevante es el terminal

        agent.decay_epsilon()

        # Resultado del episodio
        if ep_reward == 1.0:
            result = 'win'
        elif ep_reward == -1.0:
            result = 'lose'
        else:
            result = 'draw'

        rewards.append(ep_reward)
        records.append({'episodio': ep, 'resultado': result, 'recompensa': ep_reward})
        recent_wins.append(1 if ep_reward == 1.0 else 0)
        recent_rewards.append(ep_reward)

        # Win rate cada WIN_RATE_INTERVAL
        if ep % config.WIN_RATE_INTERVAL == 0:
            wr = sum(recent_wins[-config.WIN_RATE_INTERVAL:]) / config.WIN_RATE_INTERVAL
            win_rates.append((ep, wr))

        # Log cada LOG_INTERVAL episodios
        if ep % config.LOG_INTERVAL == 0:
            window = min(config.LOG_INTERVAL, len(recent_wins))
            wr_log = sum(recent_wins[-window:]) / window
            avg_r  = sum(recent_rewards[-window:]) / window
            elapsed = time.time() - t_start
            eta = elapsed / ep * (episodes - ep)
            print(
                f'[Q-Learning] Ep {ep:>7,}/{episodes:,} | '
                f'Win rate: {wr_log:.1%} | '
                f'Avg reward: {avg_r:+.3f} | '
                f'ε: {agent.epsilon:.4f} | '
                f'Elapsed: {_fmt_time(elapsed)} | ETA: {_fmt_time(eta)}'
            )

    elapsed_total = time.time() - t_start

    # ── Resumen final ──
    last_n   = min(config.EVAL_LAST_N, episodes)
    last_wr  = sum(1 for r in rewards[-last_n:] if r == 1.0) / last_n
    last_avg = sum(rewards[-last_n:]) / last_n
    total_wr = sum(1 for r in rewards if r == 1.0) / episodes

    print(f'\n  {"─"*60}')
    print(f'  [Q-Learning] RESUMEN FINAL ({episodes:,} episodios)')
    print(f'  {"─"*60}')
    print(f'  Win Rate Total         : {total_wr:.2%}')
    print(f'  Win Rate Últimos {last_n//1000}k   : {last_wr:.2%}')
    print(f'  Recompensa Promedio    : {last_avg:+.4f}')
    print(f'  Epsilon Final          : {agent.epsilon:.4f}')
    print(f'  Tiempo total           : {_fmt_time(elapsed_total)}')
    print(f'  {"─"*60}')

    return rewards, records, win_rates, agent


# ─── Bucle de entrenamiento — SARSA ──────────────────────────────────────────

def train_sarsa(episodes: int):
    """
    Entrena el agente SARSA durante `episodes` episodios.

    SARSA es on-policy: la actualización usa la acción real a' seleccionada
    por la política ε-greedy (no el máximo).
    """
    _header('ENTRENAMIENTO — SARSA (on-policy)')

    env   = BlackjackEnv()
    agent = SARSAAgent()

    rewards   = []
    records   = []
    win_rates = []

    recent_wins    = []
    recent_rewards = []

    t_start = time.time()

    for ep in range(1, episodes + 1):
        state  = env.reset()
        action = agent.select_action(state)
        done   = False
        ep_reward = 0.0

        while not done:
            next_state, reward, done = env.step(action)

            if done:
                next_action = 0  # Acción arbitraria (no se usa en terminal)
            else:
                next_action = agent.select_action(next_state)

            agent.update(state, action, reward, next_state, next_action, done)

            state  = next_state
            action = next_action
            ep_reward = reward

        agent.decay_epsilon()

        if ep_reward == 1.0:
            result = 'win'
        elif ep_reward == -1.0:
            result = 'lose'
        else:
            result = 'draw'

        rewards.append(ep_reward)
        records.append({'episodio': ep, 'resultado': result, 'recompensa': ep_reward})
        recent_wins.append(1 if ep_reward == 1.0 else 0)
        recent_rewards.append(ep_reward)

        if ep % config.WIN_RATE_INTERVAL == 0:
            wr = sum(recent_wins[-config.WIN_RATE_INTERVAL:]) / config.WIN_RATE_INTERVAL
            win_rates.append((ep, wr))

        if ep % config.LOG_INTERVAL == 0:
            window = min(config.LOG_INTERVAL, len(recent_wins))
            wr_log = sum(recent_wins[-window:]) / window
            avg_r  = sum(recent_rewards[-window:]) / window
            elapsed = time.time() - t_start
            eta = elapsed / ep * (episodes - ep)
            print(
                f'[SARSA]      Ep {ep:>7,}/{episodes:,} | '
                f'Win rate: {wr_log:.1%} | '
                f'Avg reward: {avg_r:+.3f} | '
                f'ε: {agent.epsilon:.4f} | '
                f'Elapsed: {_fmt_time(elapsed)} | ETA: {_fmt_time(eta)}'
            )

    elapsed_total = time.time() - t_start

    last_n   = min(config.EVAL_LAST_N, episodes)
    last_wr  = sum(1 for r in rewards[-last_n:] if r == 1.0) / last_n
    last_avg = sum(rewards[-last_n:]) / last_n
    total_wr = sum(1 for r in rewards if r == 1.0) / episodes

    print(f'\n  {"─"*60}')
    print(f'  [SARSA] RESUMEN FINAL ({episodes:,} episodios)')
    print(f'  {"─"*60}')
    print(f'  Win Rate Total         : {total_wr:.2%}')
    print(f'  Win Rate Últimos {last_n//1000}k   : {last_wr:.2%}')
    print(f'  Recompensa Promedio    : {last_avg:+.4f}')
    print(f'  Epsilon Final          : {agent.epsilon:.4f}')
    print(f'  Tiempo total           : {_fmt_time(elapsed_total)}')
    print(f'  {"─"*60}')

    return rewards, records, win_rates, agent


# ─── Orquestación principal ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Blackjack RL — Entrenamiento de agentes Q-Learning y SARSA'
    )
    parser.add_argument(
        '--episodes', type=int, default=config.EPISODES,
        help=f'Número de episodios de entrenamiento (default: {config.EPISODES:,})'
    )
    parser.add_argument(
        '--agent', type=str, default='both',
        choices=['qlearning', 'sarsa', 'both'],
        help='Agente a entrenar: qlearning | sarsa | both (default: both)'
    )
    args = parser.parse_args()

    episodes = args.episodes
    run_ql   = args.agent in ('qlearning', 'both')
    run_sarsa = args.agent in ('sarsa', 'both')

    print('\n' + '█' * 68)
    print('  🃏  BLACKJACK — APRENDIZAJE POR REFUERZO')
    print(f'  Agente(s): {args.agent.upper()}  |  Episodios: {episodes:,}')
    print('  Implementación desde cero (sin OpenAI Gym)')
    print('█' * 68)
    print(f'\n  Hiperparámetros:')
    print(f'    α (learning rate)  = {config.ALPHA}')
    print(f'    γ (discount)       = {config.GAMMA}')
    print(f'    ε inicial          = {config.EPSILON_INIT}')
    print(f'    ε decay            = {config.EPSILON_DECAY}')
    print(f'    ε mínimo           = {config.EPSILON_MIN}')

    ql_rewards = ql_records = ql_rates = ql_agent = None
    sarsa_rewards = sarsa_records = sarsa_rates = sarsa_agent = None

    # ── Entrenar Q-Learning ──
    if run_ql:
        ql_rewards, ql_records, ql_rates, ql_agent = train_q_learning(episodes)

    # ── Entrenar SARSA ──
    if run_sarsa:
        sarsa_rewards, sarsa_records, sarsa_rates, sarsa_agent = train_sarsa(episodes)

    # ── Guardar CSVs ──
    _header('GUARDANDO DATOS')

    if run_ql:
        save_scores(config.CSV_Q_LEARNING, ql_records)

    if run_sarsa:
        save_scores(config.CSV_SARSA, sarsa_records)

    if run_ql and run_sarsa:
        save_win_rates_comparison(ql_rates, sarsa_rates)

    # ── Generar gráficas ──
    _header('GENERANDO GRÁFICAS')

    if run_ql:
        _section('Gráfica 1: Curva de aprendizaje Q-Learning')
        plot_training_curve(ql_rewards, 'Q-Learning', config.PNG_TRAIN_QL)

    if run_sarsa:
        _section('Gráfica 2: Curva de aprendizaje SARSA')
        plot_training_curve(sarsa_rewards, 'SARSA', config.PNG_TRAIN_SARSA)

    if run_ql and run_sarsa:
        _section('Gráfica 3: Comparación Q-Learning vs SARSA')
        plot_comparison(ql_rewards, sarsa_rewards, config.PNG_COMPARISON)

    if run_ql:
        _section('Gráfica 4: Heatmap de política aprendida (Q-Learning)')
        policy = ql_agent.get_policy()
        match_hard, match_soft = plot_policy_heatmap(
            policy, 'Q-Learning', config.PNG_HEATMAP
        )
        print(f'    Coincidencia con Basic Strategy → Hard: {match_hard:.1f}% | '
              f'Soft: {match_soft:.1f}%')

    # ── Resumen final ──
    _header('✅  PROYECTO COMPLETADO')
    print(f'  Archivos generados en: ./{config.DATA_DIR}/')
    print()
    if run_ql:
        last_n  = min(config.EVAL_LAST_N, episodes)
        ql_last_wr = sum(1 for r in ql_rewards[-last_n:] if r == 1.0) / last_n
        print(f'  Q-Learning  Win Rate (últimos {last_n//1000}k): {ql_last_wr:.2%}')
    if run_sarsa:
        last_n = min(config.EVAL_LAST_N, episodes)
        sarsa_last_wr = sum(1 for r in sarsa_rewards[-last_n:] if r == 1.0) / last_n
        print(f'  SARSA       Win Rate (últimos {last_n//1000}k): {sarsa_last_wr:.2%}')
    print(f'  Basic Strategy referencia              : ≈ 42.00%')
    print()


if __name__ == '__main__':
    main()
