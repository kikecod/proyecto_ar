"""
utils/saver.py — Guardar y cargar resultados en CSV

Formatos:
    scores_*.csv           → episodio, resultado, recompensa
    win_rates_comparison   → episodio, win_rate_ql, win_rate_sarsa
"""

import os
import csv

import config


def ensure_data_dir():
    """Crea el directorio data/ si no existe."""
    os.makedirs(config.DATA_DIR, exist_ok=True)


def save_scores(filename: str, records: list):
    """
    Guarda los resultados episodio a episodio.

    Args:
        filename: ruta del CSV de salida
        records: lista de dicts con keys 'episodio', 'resultado', 'recompensa'
    """
    ensure_data_dir()
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['episodio', 'resultado', 'recompensa'])
        writer.writeheader()
        writer.writerows(records)
    print(f"  ✔ CSV guardado: {filename}  ({len(records)} registros)")


def save_win_rates_comparison(ql_rates: list, sarsa_rates: list):
    """
    Guarda la tabla comparativa de win rates cada WIN_RATE_INTERVAL episodios.

    Args:
        ql_rates   : lista de (episodio, win_rate) para Q-Learning
        sarsa_rates: lista de (episodio, win_rate) para SARSA
    """
    ensure_data_dir()

    # Alinear por episodio (ambas listas deben tener la misma longitud)
    rows = []
    for (ep_ql, wr_ql), (ep_sarsa, wr_sarsa) in zip(ql_rates, sarsa_rates):
        rows.append({
            'episodio': ep_ql,
            'win_rate_q_learning': round(wr_ql, 4),
            'win_rate_sarsa': round(wr_sarsa, 4),
        })

    with open(config.CSV_COMPARISON, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['episodio', 'win_rate_q_learning', 'win_rate_sarsa']
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✔ CSV guardado: {config.CSV_COMPARISON}  ({len(rows)} registros)")


def load_scores(filename: str) -> list:
    """
    Carga un CSV de scores.

    Retorna:
        Lista de dicts con 'episodio', 'resultado', 'recompensa'
    """
    records = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'episodio': int(row['episodio']),
                'resultado': row['resultado'],
                'recompensa': float(row['recompensa']),
            })
    return records
