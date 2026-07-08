"""
Carga y unificacion de fuentes crudas (equipos, historial internacional,
shootouts y partidos del Mundial 2026).

Hallazgo clave de la inspeccion de datos: 'results.csv' NO es solo historia
antigua -- ya incluye los partidos del Mundial 2026 jugados hasta hoy, y hasta
los partidos de rondas eliminatorias AUN NO JUGADOS (con marcador NaN). Es
decir, 'results.csv' es exactamente el log historico + actual, sin fuga de
futuro (los resultados que no se conocen todavia aparecen en NaN). Esto nos
permite usarlo como la unica fuente de "goles" para el entrenamiento
(1872-2026), mientras que 'matches.csv' + 'match_team_stats.csv' aportan
estadisticas mas ricas (xG, posesion, tiros, corners, paradas) SOLO
disponibles para los partidos de este Mundial 2026.
"""
import pandas as pd

from src.core.config import MAPEO_NOMBRES
from src.infra.paths import (
    MATCHES_CSV, TEAMS_CSV, MATCH_TEAM_STATS_CSV, RESULTS_CSV,
    SHOOTOUTS_CSV, SQUADS_CSV,
)


def cargar_equipos() -> pd.DataFrame:
    teams = pd.read_csv(TEAMS_CSV)
    teams["team_name_hist"] = teams["team_name"].replace(MAPEO_NOMBRES)
    return teams


def cargar_valor_mercado() -> pd.DataFrame:
    """Suma el valor de mercado de la plantilla por equipo."""
    sq = pd.read_csv(SQUADS_CSV)
    valor = sq.groupby("team_id")["market_value_eur"].sum().rename("valor_mercado_eur")
    return valor.reset_index()


def cargar_historico_resultados() -> pd.DataFrame:
    """results.csv: historico completo 1872-2026, incluye el Mundial 2026 en
    curso (con NaN en los partidos aun no jugados)."""
    r = pd.read_csv(RESULTS_CSV)
    r["date"] = pd.to_datetime(r["date"])
    r["neutral"] = r["neutral"].astype(bool)
    return r


def cargar_shootouts() -> pd.DataFrame:
    s = pd.read_csv(SHOOTOUTS_CSV)
    s["date"] = pd.to_datetime(s["date"])
    return s


def resolver_ganador_penaltis(historico: pd.DataFrame, shootouts: pd.DataFrame) -> pd.DataFrame:
    """Cuando un partido eliminatorio termina empatado y se decide por
    penaltis, 'results.csv' registra el marcador de los 90/120 min (empate),
    pero el ganador real es el de la tanda. Aqui anotamos ese ganador
    (columna 'ganador_penaltis') para etiquetar correctamente el 1X2."""
    df = historico.merge(
        shootouts[["date", "home_team", "away_team", "winner"]],
        on=["date", "home_team", "away_team"], how="left",
    )
    df = df.rename(columns={"winner": "ganador_penaltis"})
    return df


def cargar_matches_2026() -> pd.DataFrame:
    """matches.csv del Mundial 2026: trae stage_id, status y home_xg/away_xg
    reales por partido."""
    m = pd.read_csv(MATCHES_CSV)
    m["date"] = pd.to_datetime(m["date"])
    return m


def cargar_match_team_stats() -> pd.DataFrame:
    return pd.read_csv(MATCH_TEAM_STATS_CSV)


def construir_matches_2026_enriquecido(teams: pd.DataFrame) -> pd.DataFrame:
    """Arma una tabla ancha (1 fila = 1 partido del Mundial 2026) con nombres
    normalizados de equipo y las estadisticas ricas (xG, posesion, tiros,
    corners, paradas) de cada lado, usando match_id/team_id como llave."""
    matches = cargar_matches_2026()
    stats = cargar_match_team_stats()

    id2name = teams.set_index("team_id")["team_name_hist"]

    m = matches.copy()
    m["home_team"] = m["home_team_id"].map(id2name)
    m["away_team"] = m["away_team_id"].map(id2name)

    stats_cols = ["possession_pct", "total_shots", "shots_on_target", "corners", "fouls", "saves"]
    home_stats = stats.rename(columns={c: f"home_{c}" for c in stats_cols})
    home_stats = home_stats.rename(columns={"team_id": "home_team_id"})[["match_id", "home_team_id"] + [f"home_{c}" for c in stats_cols]]
    away_stats = stats.rename(columns={c: f"away_{c}" for c in stats_cols})
    away_stats = away_stats.rename(columns={"team_id": "away_team_id"})[["match_id", "away_team_id"] + [f"away_{c}" for c in stats_cols]]

    m = m.merge(home_stats, on=["match_id", "home_team_id"], how="left")
    m = m.merge(away_stats, on=["match_id", "away_team_id"], how="left")
    return m
