# 🃏 Blackjack RL — Aprendizaje por Refuerzo

Proyecto de la materia **Aprendizaje Automático (Machine Learning) INF 323**  
Universidad Mayor de San Andrés — Carrera de Informática

---

## Descripción

Implementación desde cero de dos agentes de **Aprendizaje por Refuerzo** aplicados al juego de Blackjack, sin usar OpenAI Gym ni librerías de RL:

| Agente | Tipo | Win Rate final |
|--------|------|---------------|
| **Q-Learning** | Off-policy | 42.38% |
| **SARSA** | On-policy | 42.03% |
| Basic Strategy (referencia) | Óptima | ~42.00% |

La política aprendida alcanza **82.5% de coincidencia** con la Basic Strategy de casino.

---

## Estructura

```
proyecto_ar/
├── main.py                  # Punto de entrada
├── config.py                # Hiperparámetros
├── requirements.txt
├── agent/
│   ├── q_learning.py        # Q-Learning (off-policy)
│   └── sarsa.py             # SARSA (on-policy)
├── environment/
│   └── blackjack_env.py     # MDP implementado desde cero
├── visualization/
│   ├── plotter.py           # Curvas de aprendizaje
│   └── policy_renderer.py  # Heatmap vs Basic Strategy
├── utils/
│   └── saver.py             # Guardar/cargar CSV
├── demo.html                # Demo animado en navegador
└── data/                    # Generado al correr main.py
    ├── scores_q_learning.csv
    ├── scores_sarsa.csv
    ├── win_rates_comparison.csv
    ├── training_q_learning.png
    ├── training_sarsa.png
    ├── comparison_ql_vs_sarsa.png
    └── policy_heatmap_q_learning.png
```

---

## Instalación y uso

```bash
pip install numpy matplotlib
```

```bash
# Entrenar ambos agentes (500k episodios)
python3 main.py

# Opciones
python3 main.py --episodes 200000   # Menos episodios
python3 main.py --agent qlearning   # Solo Q-Learning
python3 main.py --agent sarsa       # Solo SARSA
```

El log muestra progreso cada 10 000 episodios:
```
[Q-Learning] Ep  50,000/500,000 | Win rate: 41.8% | Avg reward: -0.071 | ε: 0.0100
```

---

## Demo animado

Levantá un servidor local y abrí el demo en el navegador:

```bash
python3 -m http.server 8765
# Abrir: http://localhost:8765/demo.html
```

El demo carga la Q-table entrenada y muestra al agente jugando en tiempo real con estadísticas en vivo.

---

## MDP — Definición del problema

- **Estados:** (suma jugador 4–21) × (carta dealer 1–10) × (as utilizable) = **360 estados**
- **Acciones:** `0` = Plantarse (Stand) · `1` = Pedir (Hit)
- **Recompensas:** +1 ganar · -1 perder · 0 empate

## Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| α (learning rate) | 0.1 |
| γ (discount) | 1.0 |
| ε inicial | 1.0 |
| ε decay | 0.9999 |
| ε mínimo | 0.01 |
| Episodios | 500 000 |

---

## Autor

**Fernández Chiri, Enrique Rafael**  
C.I.: 10900348 · INF 323 · FCPN — UMSA  
La Paz, Bolivia · 2026
