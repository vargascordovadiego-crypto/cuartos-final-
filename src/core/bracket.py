"""
Construccion del cruce de Cuartos de Final a partir del resultado (real o
predicho) de cada Octavo de Final.

El nucleo del modelo predictivo (csv_sources, features, cascade_model,
predictor) no se toca aqui. Este modulo solo arma las 4 filas de partido
nuevas (con sus equipos, fecha y sede) para que esa misma cascada, ya
entrenada, las evalue como una etapa mas.

OJO: si el cruce ya venia incluido en 'results.csv' de antemano (caso de
Francia-Marruecos, conocido desde que ambos ganaron su Octavo real, ver
'data/results.csv'), NO se duplica esa fila -- se reutiliza la que ya trae
el historico y solo se agrega la fila de 'matches_2026' (para asignarle
stage_id=4 y que participe del mismo flujo que el resto de partidos
pendientes).

Los pares de Octavos que se enfrentan en Cuartos siguen el bracket oficial:
se confirmo con el propio dataset (`results.csv` ya trae la fila
"Francia vs Marruecos", 2026-07-09, salida del par de partidos 89/90) que el
ganador del partido de match_id MAS ALTO de cada pareja juega de Local en
Cuartos y el de match_id mas bajo de Visitante. Esa misma convencion se usa
para las otras 3 llaves. Nota: al ser sede neutral, el "truco del espejo" en
core/predictor.py promedia ambas orientaciones, asi que esta eleccion de
Local/Visitante no sesga la prediccion.
"""
import numpy as np
import pandas as pd

PARES_OCTAVOS = [(89, 90), (91, 92), (93, 94), (95, 96)]

FECHA_CUARTOS = {
    (89, 90): "2026-07-09",
    (91, 92): "2026-07-11",
    (93, 94): "2026-07-10",
    (95, 96): "2026-07-11",
}

SEDE_CUARTOS = {
    (89, 90): ("Foxborough", "United States"),
    (91, 92): ("Miami Gardens", "United States"),
    (93, 94): ("Inglewood", "United States"),
    (95, 96): ("Kansas City", "United States"),
}


def construir_cruces_cuartos(resultados_octavos: pd.DataFrame) -> list:
    """Devuelve los 4 cruces de Cuartos [{home_team, away_team, date, city,
    country}] a partir de la tabla de resultados (reales + predichos) de
    Octavos de Final."""
    ganador_por_match = {
        int(mid): ganador
        for mid, ganador in zip(resultados_octavos["match_id"], resultados_octavos["ganador_predicho"])
    }
    cruces = []
    for par in PARES_OCTAVOS:
        m_bajo, m_alto = par
        home = ganador_por_match[m_alto]
        away = ganador_por_match[m_bajo]
        fecha = FECHA_CUARTOS[par]
        ciudad, pais = SEDE_CUARTOS[par]
        cruces.append({"home_team": home, "away_team": away, "date": fecha, "city": ciudad, "country": pais})
    return cruces


def agregar_fixture_cuartos(historico: pd.DataFrame, matches_2026: pd.DataFrame,
                             teams: pd.DataFrame, cruces: list) -> tuple:
    """Agrega los 4 partidos de Cuartos como filas nuevas de 'historico' (al
    estilo results.csv, sin marcador) y de 'matches_2026' (al estilo
    matches.csv, stage_id=4, status Scheduled), para que
    feature_engineering.construir_dataset_modelo los procese exactamente
    igual que cualquier otro partido pendiente."""
    id_por_nombre = teams.set_index("team_name_hist")["team_id"]
    next_match_id = int(matches_2026["match_id"].max()) + 1

    filas_hist, filas_matches = [], []
    for cruce in cruces:
        fecha = pd.Timestamp(cruce["date"])

        ya_existe = (
            (historico["home_team"] == cruce["home_team"])
            & (historico["away_team"] == cruce["away_team"])
            & (historico["date"] >= "2026-07-08")
            & historico["home_score"].isna()
        ).any()
        if not ya_existe:
            filas_hist.append({
                "date": fecha, "home_team": cruce["home_team"], "away_team": cruce["away_team"],
                "home_score": np.nan, "away_score": np.nan, "tournament": "FIFA World Cup",
                "city": cruce["city"], "country": cruce["country"], "neutral": True,
            })
        filas_matches.append({
            "match_id": next_match_id, "date": fecha, "stage_id": 4,
            "home_team_id": id_por_nombre.get(cruce["home_team"]),
            "away_team_id": id_por_nombre.get(cruce["away_team"]),
            "home_score": np.nan, "away_score": np.nan, "status": "Scheduled",
            "home_xg": np.nan, "away_xg": np.nan,
            "home_team": cruce["home_team"], "away_team": cruce["away_team"],
        })
        next_match_id += 1

    historico_aug = pd.concat([historico, pd.DataFrame(filas_hist)], ignore_index=True)
    matches_aug = pd.concat([matches_2026, pd.DataFrame(filas_matches)], ignore_index=True)
    return historico_aug, matches_aug
