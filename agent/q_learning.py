"""
agent/q_learning.py — Agente Q-Learning (off-policy)

Fórmula de actualización:
    Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

El agente es OFF-POLICY: aprende la política greedy óptima mientras explora
con una política ε-greedy.
"""

import random
from collections import defaultdict

import config


class QLearningAgent:
    """
    Agente Q-Learning tabular para Blackjack.

    Atributos:
        q_table  : dict {estado: [Q(s,0), Q(s,1)]}
        epsilon  : nivel actual de exploración
        alpha    : tasa de aprendizaje
        gamma    : factor de descuento
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

        # Q-table: valor por defecto 0.0 para todos los pares (estado, acción)
        self.q_table = defaultdict(lambda: [0.0, 0.0])

    # ─── Selección de acción ──────────────────────────────────────────────────

    def select_action(self, state) -> int:
        """
        Política ε-greedy:
        - Con probabilidad ε → exploración aleatoria
        - Con probabilidad (1-ε) → explotación (acción con mayor Q)
        """
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return self._greedy_action(state)

    def _greedy_action(self, state) -> int:
        """Devuelve la acción con mayor Q-valor para el estado dado."""
        q_values = self.q_table[state]
        return int(q_values[1] > q_values[0])  # 1 si Q(s,Hit) > Q(s,Stand)

    # ─── Actualización Q-Learning ─────────────────────────────────────────────

    def update(self, state, action: int, reward: float, next_state, done: bool):
        """
        Actualización off-policy (bootstrap con max_a' Q(s',a')).

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
        """
        q_current = self.q_table[state][action]

        if done:
            td_target = reward
        else:
            q_next_max = max(self.q_table[next_state])
            td_target = reward + self.gamma * q_next_max

        td_error = td_target - q_current
        self.q_table[state][action] = q_current + self.alpha * td_error

    # ─── Decaimiento de epsilon ───────────────────────────────────────────────

    def decay_epsilon(self):
        """Aplica el decaimiento exponencial de epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ─── Política greedy (para evaluación) ───────────────────────────────────

    def get_policy(self) -> dict:
        """
        Extrae la política greedy aprendida.

        Retorna:
            dict {estado: acción_greedy}
        """
        return {state: self._greedy_action(state) for state in self.q_table}

    def get_q_table(self) -> dict:
        """Retorna la Q-table como dict estándar."""
        return dict(self.q_table)
