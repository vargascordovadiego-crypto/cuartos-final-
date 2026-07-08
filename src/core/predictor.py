"""
Prediccion de los partidos de Octavos de Final aun no jugados (sec. 4 y 5
del manual): truco del espejo (sede neutral), afilado por temperatura, y
resolucion de cada cruce con una simulacion de Monte Carlo (10.000 mundiales
en miniatura, con penaltis si el resultado simulado es empate).
"""
import math

import numpy as np
import pandas as pd

from src.core.config import N_SIMULACIONES_MONTECARLO, RANDOM_STATE

LABEL_ORDEN = ["home", "draw", "away"]


def _construir_meta(X: pd.DataFrame, modelo_l, modelo_v) -> pd.DataFrame:
    meta = X.copy()
    meta["pred_goles_l"] = modelo_l.predict(X)
    meta["pred_goles_v"] = modelo_v.predict(X)
    return meta


def _invertir_features(fila: pd.Series, columnas: list) -> pd.Series:
    """El 'truco del espejo': al intercambiar Local <-> Visitante, toda
    variable diff_* cambia de signo, y prob_implicita_elo se invierte
    (1 - p), porque pasa a medir la probabilidad del equipo contrario."""
    invertida = fila.copy()
    for c in columnas:
        if c.startswith("diff_"):
            invertida[c] = -fila[c]
    invertida["prob_implicita_elo"] = 1 - fila["prob_implicita_elo"]
    return invertida


def _poisson_pmf(k: int, lam: float) -> float:
    lam = max(lam, 1e-6)
    return math.exp(-lam) * lam ** k / math.factorial(k)


def marcador_mas_probable(lam_l: float, lam_v: float, resultado: str) -> tuple:
    mejor = (0, 0)
    mejor_p = -1.0
    for i in range(9):
        for j in range(9):
            if resultado == "home" and i <= j:
                continue
            if resultado == "away" and i >= j:
                continue
            if resultado == "draw" and i != j:
                continue
            p = _poisson_pmf(i, lam_l) * _poisson_pmf(j, lam_v)
            if p > mejor_p:
                mejor_p, mejor = p, (i, j)
    return mejor


def _simular_cruce(prob_local: float, prob_empate: float, prob_visitante: float,
                    n_sim: int, rng: np.random.Generator) -> float:
    """Devuelve la fraccion de simulaciones ganadas por el Local, resolviendo
    los empates por penaltis (sec. 5.2)."""
    resultados = rng.choice([0, 1, 2], size=n_sim, p=[prob_local, prob_empate, prob_visitante])
    empates = resultados == 1
    if empates.any():
        p1_pen = prob_local / (prob_local + prob_visitante)
        resultados[empates] = rng.choice([0, 2], size=empates.sum(), p=[p1_pen, 1 - p1_pen])
    return float((resultados == 0).mean())


def predecir_partidos_pendientes(df_pendientes: pd.DataFrame, modelo: dict) -> pd.DataFrame:
    columnas = modelo["columnas"]
    T = modelo["temperatura"]
    clasificador = modelo["clasificador"]
    modelo_l, modelo_v = modelo["modelo_l"], modelo["modelo_v"]
    rng = np.random.default_rng(RANDOM_STATE)

    filas_out = []
    for _, fila in df_pendientes.iterrows():
        X_normal = fila[columnas].to_frame().T.astype(float)
        fila_inv = _invertir_features(fila, columnas)
        X_inv = fila_inv[columnas].to_frame().T.astype(float)

        meta_normal = _construir_meta(X_normal, modelo_l, modelo_v)
        meta_inv = _construir_meta(X_inv, modelo_l, modelo_v)

        probs_normal = clasificador.predict_proba(meta_normal)[0]
        probs_inv = clasificador.predict_proba(meta_inv)[0]

        prob_local = (probs_normal[0] + probs_inv[2]) / 2
        prob_empate = (probs_normal[1] + probs_inv[1]) / 2
        prob_visitante = (probs_normal[2] + probs_inv[0]) / 2

        probs = np.array([prob_local, prob_empate, prob_visitante])
        probs_afiladas = probs ** (1 / T)
        probs_afiladas = probs_afiladas / probs_afiladas.sum()
        prob_local, prob_empate, prob_visitante = probs_afiladas

        pct_local = _simular_cruce(prob_local, prob_empate, prob_visitante,
                                    N_SIMULACIONES_MONTECARLO, rng)
        ganador = fila["home_team"] if pct_local >= 0.5 else fila["away_team"]
        confianza = pct_local if pct_local >= 0.5 else 1 - pct_local

        lam_l = (meta_normal["pred_goles_l"].iloc[0] + meta_inv["pred_goles_v"].iloc[0]) / 2
        lam_v = (meta_normal["pred_goles_v"].iloc[0] + meta_inv["pred_goles_l"].iloc[0]) / 2
        resultado_1x2 = "home" if pct_local >= 0.5 else "away"
        marcador = marcador_mas_probable(max(lam_l, 0.1), max(lam_v, 0.1), resultado_1x2)

        filas_out.append({
            "match_id": fila["match_id"], "fecha": fila["date"],
            "equipo_local": fila["home_team"], "equipo_visitante": fila["away_team"],
            "prob_local": prob_local, "prob_empate": prob_empate, "prob_visitante": prob_visitante,
            "pct_montecarlo_local": pct_local,
            "ganador_predicho": ganador, "confianza": confianza,
            "marcador_probable": f"{marcador[0]}-{marcador[1]}",
            "ya_jugado": False,
        })

    return pd.DataFrame(filas_out)


def partidos_ya_decididos(df_decididos: pd.DataFrame) -> pd.DataFrame:
    filas_out = []
    for _, fila in df_decididos.iterrows():
        gano_local = fila["ganador_final"] == "home"
        filas_out.append({
            "match_id": fila["match_id"], "fecha": fila["date"],
            "equipo_local": fila["home_team"], "equipo_visitante": fila["away_team"],
            "prob_local": 1.0 if gano_local else 0.0, "prob_empate": 0.0,
            "prob_visitante": 0.0 if gano_local else 1.0,
            "pct_montecarlo_local": 1.0 if gano_local else 0.0,
            "ganador_predicho": fila["home_team"] if gano_local else fila["away_team"],
            "confianza": 1.0,
            "marcador_probable": f"{int(fila['home_score'])}-{int(fila['away_score'])}",
            "ya_jugado": True,
        })
    return pd.DataFrame(filas_out)
