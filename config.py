"""
config.py — Hiperparámetros centrales del proyecto Blackjack RL
"""

# ─── Hiperparámetros de entrenamiento ────────────────────────────────────────
EPISODES = 500_000          # Número total de episodios de entrenamiento

ALPHA = 0.1                 # Tasa de aprendizaje (learning rate)
GAMMA = 1.0                 # Factor de descuento (sin descuento para episodios cortos)
EPSILON_INIT = 1.0          # Epsilon inicial (exploración total)
EPSILON_DECAY = 0.9999      # Factor de decaimiento de epsilon por episodio
EPSILON_MIN = 0.01          # Epsilon mínimo (1% de exploración permanente)

# ─── Logging y reportes ──────────────────────────────────────────────────────
LOG_INTERVAL = 10_000       # Cada cuántos episodios mostrar progreso en terminal
WINDOW_SIZE = 500           # Ventana móvil para calcular tasa de victorias
EVAL_LAST_N = 50_000        # Últimos N episodios para resumen final
WIN_RATE_INTERVAL = 1_000   # Cada cuántos episodios registrar win_rate en CSV

# ─── Rutas de salida ─────────────────────────────────────────────────────────
DATA_DIR = "data"

CSV_Q_LEARNING  = f"{DATA_DIR}/scores_q_learning.csv"
CSV_SARSA       = f"{DATA_DIR}/scores_sarsa.csv"
CSV_COMPARISON  = f"{DATA_DIR}/win_rates_comparison.csv"

PNG_TRAIN_QL    = f"{DATA_DIR}/training_q_learning.png"
PNG_TRAIN_SARSA = f"{DATA_DIR}/training_sarsa.png"
PNG_COMPARISON  = f"{DATA_DIR}/comparison_ql_vs_sarsa.png"
PNG_HEATMAP     = f"{DATA_DIR}/policy_heatmap_q_learning.png"
