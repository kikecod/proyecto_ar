"""
agent/sarsa.py — Agente SARSA (on-policy)

Fórmula de actualización:
    Q(s,a) ← Q(s,a) + α · [r + γ · Q(s',a') − Q(s,a)]

El agente es ON-POLICY: actualiza Q usando la acción a' que realmente se tomará
(muestreada desde la misma política ε-greedy).
"""

import random
from collections import defaultdict

import config


class SARSAAgent:
    """
    Agente SARSA tabular para Blackjack.

    La diferencia clave con Q-Learning: la actualización usa Q(s', a') donde
    a' es la acción real seleccionada por la política (no el máximo teórico).
    Esto hace a SARSA más conservador en la exploración.
    """

    def __init__(
        self,
        alpha: float = config.ALPHA,
        gamma: float = config.GAMMA,
        epsilon: float = config.EPSILON_INIT,
        epsilon_decay: float = config.EPSILON_DECAY,
        epsilon_min: float = config.EPSILON_MIN,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = defaultdict(lambda: [0.0, 0.0])

    # ─── Selección de acción ──────────────────────────────────────────────────

    def select_action(self, state) -> int:
        """
        Política ε-greedy (idéntica a Q-Learning en selección,
        diferente en la actualización).
        """
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return self._greedy_action(state)

    def _greedy_action(self, state) -> int:
        """Devuelve la acción con mayor Q-valor para el estado dado."""
        q_values = self.q_table[state]
        return int(q_values[1] > q_values[0])

    # ─── Actualización SARSA ──────────────────────────────────────────────────

    def update(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        next_action: int,
        done: bool,
    ):
        """
        Actualización on-policy (usa Q(s', a') con a' real, no el máximo).

        Q(s,a) ← Q(s,a) + α · [r + γ · Q(s',a') − Q(s,a)]
        """
        q_current = self.q_table[state][action]

        if done:
            td_target = reward
        else:
            q_next = self.q_table[next_state][next_action]
            td_target = reward + self.gamma * q_next

        td_error = td_target - q_current
        self.q_table[state][action] = q_current + self.alpha * td_error

    # ─── Decaimiento de epsilon ───────────────────────────────────────────────

    def decay_epsilon(self):
        """Aplica el decaimiento exponencial de epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ─── Política greedy (para evaluación) ───────────────────────────────────

    def get_policy(self) -> dict:
        """Extrae la política greedy aprendida."""
        return {state: self._greedy_action(state) for state in self.q_table}

    def get_q_table(self) -> dict:
        """Retorna la Q-table como dict estándar."""
        return dict(self.q_table)
