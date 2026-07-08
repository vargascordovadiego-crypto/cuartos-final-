"""
Ingenieria de variables (sec. 2 de 'Cerebro_Del_Modelo.md'), adaptada a los
datos reales disponibles:

- El "estado de forma" (avg_*, trend_xG_5, forma_vs_historia,
  clean_sheet_rate_5) se calcula sobre el HISTORICO COMPLETO de cada
  seleccion (1872-2026, cualquier rival), usando goles reales como proxy de
  xG historico (no existe xG antes del Mundial 2026).
- El Head-to-Head (H2H) se calcula igual, por pareja de equipos, con
  shift() para no filtrar el resultado del propio partido (sec. 2.6).
- Elo/FIFA (`elo_rating`), Tier, Peso de confederacion y Valor de Mercado
  (sec. 2.5) solo existen para las 48 selecciones del Mundial 2026, asi que
  el dataset de modelado final se restringe a los partidos historicos en los
  que AMBOS equipos son de ese universo (7.617 partidos con resultado
  conocido, 1872-2026) -- sigue siendo un historico grande y legitimo.
- Estadisticas ricas del Mundial 2026 (xG, posesion, tiros, corners,
  paradas) solo existen desde el inicio del torneo: en vez de imputarlas con
  IterativeImputer (pensado para huecos esporadicos, no para una frontera
  estructural del 99% de los datos), se dejan en 0 fuera del torneo y se
  agrega una bandera `es_mundial_2026` para que XGBoost aprenda a confiar en
  ellas solo cuando corresponde.
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.core.config import VENTANAS, PESOS_CONFEDERACION

HOY = pd.Timestamp("2026-07-05")

RICH_STATS = ["xg", "possession_pct", "total_shots", "shots_on_target", "corners", "saves"]


def _pendiente(serie: np.ndarray) -> float:
    validos = serie[~np.isnan(serie)]
    if len(validos) < 2:
        return 0.0
    return linregress(np.arange(len(validos)), validos).slope


def _construir_pair_key(home: pd.Series, away: pd.Series) -> pd.Series:
    a = home.to_numpy()
    b = away.to_numpy()
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return pd.Series(list(zip(lo, hi)), index=home.index)


def preparar_historico(historico: pd.DataFrame, matches_2026: pd.DataFrame) -> pd.DataFrame:
    """Une results.csv con las estadisticas ricas del Mundial 2026 y calcula
    el ganador final (goles, o penaltis si hubo empate eliminatorio)."""
    df = historico.reset_index(drop=True).copy()
    df["orig_index"] = df.index

    keep = ["date", "home_team", "away_team", "stage_id", "match_id", "status",
            "home_xg", "away_xg", "home_possession_pct", "away_possession_pct",
            "home_total_shots", "away_total_shots", "home_shots_on_target", "away_shots_on_target",
            "home_corners", "away_corners", "home_saves", "away_saves"]
    rich = matches_2026[keep].copy().drop(columns=["date"])

    # OJO: 'matches.csv' y 'results.csv' a veces difieren en un dia en la
    # fecha de partidos recientes (posible corte de huso horario en el
    # kickoff), asi que no se puede unir por fecha exacta. Se une solo por
    # el par de equipos, pero SOLO dentro de la ventana del Mundial 2026
    # (verificado: 0 pares local/visitante duplicados ahi) para no
    # contaminar enfrentamientos historicos con el mismo local/visitante
    # de decadas atras.
    ventana_mundial = df["date"] >= "2026-06-01"
    df_mundial = df.loc[ventana_mundial].merge(rich, on=["home_team", "away_team"], how="left", validate="many_to_one")
    df_resto = df.loc[~ventana_mundial].copy()
    for c in rich.columns:
        if c not in ("home_team", "away_team"):
            df_resto[c] = np.nan
    df = pd.concat([df_resto, df_mundial], ignore_index=True).sort_values("orig_index").reset_index(drop=True)
    df["es_mundial_2026"] = df["match_id"].notna()

    es_empate = df["home_score"] == df["away_score"]
    gano_local_penal = df["ganador_penaltis"] == df["home_team"]
    gano_visita_penal = df["ganador_penaltis"] == df["away_team"]

    condiciones = [
        df["home_score"] > df["away_score"],
        df["home_score"] < df["away_score"],
        es_empate & gano_local_penal,
        es_empate & gano_visita_penal,
    ]
    valores = ["home", "away", "home", "away"]
    df["ganador_final"] = np.select(condiciones, valores, default="draw")
    # Partidos sin marcador (aun no jugados) no tienen ganador
    df.loc[df["home_score"].isna(), "ganador_final"] = np.nan

    df["pair_key"] = _construir_pair_key(df["home_team"], df["away_team"])
    return df


def _tabla_larga(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana el dataset (1 fila = 1 partido) a formato largo (1 fila =
    1 equipo en 1 partido), sec. 2.1."""
    comunes = ["orig_index", "date", "pair_key", "es_mundial_2026"]

    home = df[comunes + ["home_team", "away_team", "home_score", "away_score", "ganador_final"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent",
                                 "home_score": "goals_for", "away_score": "goals_against"})
    home["is_home"] = True
    home["gano"] = (home["ganador_final"] == "home").astype(float)

    away = df[comunes + ["home_team", "away_team", "home_score", "away_score", "ganador_final"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent",
                                 "away_score": "goals_for", "home_score": "goals_against"})
    away["is_home"] = False
    away["gano"] = (away["ganador_final"] == "away").astype(float)

    for lado, out in [("home", home), ("away", away)]:
        for stat in RICH_STATS:
            src_col = f"{lado}_{stat}"
            if src_col in df.columns:
                out[stat] = df[src_col].values

    largo = pd.concat([home, away], ignore_index=True)
    largo = largo.sort_values(["team", "date", "orig_index"]).reset_index(drop=True)
    return largo


def _rolling_shift(grupo: pd.Series, ventana):
    despl = grupo.shift()
    if ventana == "total":
        return despl.expanding().mean()
    return despl.rolling(ventana, min_periods=1).mean()


def calcular_features_forma(largo: pd.DataFrame) -> pd.DataFrame:
    """Medias moviles con shift() por equipo (sec. 2.2), trend_xG_5,
    forma_vs_historia y clean_sheet_rate_5 (sec. 2.3), tanto para goles
    reales (disponibles siempre) como para xG/posesion/tiros/corners/paradas
    (solo Mundial 2026)."""
    g = largo.groupby("team", group_keys=False)

    for ventana in VENTANAS + ["total"]:
        etiqueta = ventana if ventana != "total" else "total"
        largo[f"avg_goals_{etiqueta}"] = g["goals_for"].transform(lambda s, v=ventana: _rolling_shift(s, v))
        largo[f"avg_conceded_{etiqueta}"] = g["goals_against"].transform(lambda s, v=ventana: _rolling_shift(s, v))
        for stat in RICH_STATS:
            if stat in largo.columns:
                largo[f"avg_{stat}_{etiqueta}"] = g[stat].transform(lambda s, v=ventana: _rolling_shift(s, v))

    largo["trend_goals_5"] = g["goals_for"].transform(
        lambda s: s.shift().rolling(5, min_periods=2).apply(lambda w: _pendiente(w.to_numpy()), raw=False)
    )
    largo["clean_sheet_rate_5"] = g["goals_against"].transform(
        lambda s: s.shift().rolling(5, min_periods=1).apply(lambda w: np.mean(w.to_numpy() < 0.5), raw=False)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        largo["forma_vs_historia"] = np.where(
            largo["avg_goals_total"] > 0, largo["avg_goals_2"] / largo["avg_goals_total"], 1.0
        )
    return largo


def calcular_h2h(largo: pd.DataFrame) -> pd.DataFrame:
    """Historial directo (H2H) con shift(), sec. 2.6."""
    largo = largo.sort_values(["pair_key", "team", "date", "orig_index"]).reset_index(drop=True)
    gp = largo.groupby(["pair_key", "team"], group_keys=False)
    largo["h2h_win_rate"] = gp["gano"].transform(lambda s: s.shift().expanding().mean())
    largo["h2h_goals_avg"] = gp["goals_for"].transform(lambda s: s.shift().expanding().mean())
    largo["h2h_win_rate"] = largo["h2h_win_rate"].fillna(1 / 3)
    media_global_goles = largo["goals_for"].mean()
    largo["h2h_goals_avg"] = largo["h2h_goals_avg"].fillna(media_global_goles)
    return largo


def calcular_h2h_empate(df: pd.DataFrame) -> pd.Series:
    tmp = df.sort_values(["pair_key", "date", "orig_index"]).copy()
    tmp["is_draw"] = (tmp["ganador_final"] == "draw").astype(float)
    tmp["h2h_draw_rate"] = tmp.groupby("pair_key")["is_draw"].transform(lambda s: s.shift().expanding().mean())
    tmp["h2h_draw_rate"] = tmp["h2h_draw_rate"].fillna(1 / 3)
    return tmp.set_index("orig_index")["h2h_draw_rate"]


def _columnas_avg(largo: pd.DataFrame) -> list:
    cols = []
    for ventana in list(VENTANAS) + ["total"]:
        cols.append(f"avg_goals_{ventana}")
        cols.append(f"avg_conceded_{ventana}")
        for stat in RICH_STATS:
            c = f"avg_{stat}_{ventana}"
            if c in largo.columns:
                cols.append(c)
    cols += ["trend_goals_5", "clean_sheet_rate_5", "forma_vs_historia", "h2h_win_rate", "h2h_goals_avg"]
    return cols


def volver_a_ancho(df: pd.DataFrame, largo: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye el formato ancho (1 fila = 1 partido) uniendo las
    variables de Local y Visitante (sec. 4 del manual paso a paso)."""
    avg_cols = _columnas_avg(largo)

    home_feats = largo[largo["is_home"]].set_index("orig_index")[avg_cols]
    home_feats.columns = [f"{c}_home" for c in avg_cols]
    away_feats = largo[~largo["is_home"]].set_index("orig_index")[avg_cols]
    away_feats.columns = [f"{c}_away" for c in avg_cols]

    df = df.set_index("orig_index")
    df = df.join(home_feats).join(away_feats)
    df["h2h_draw_rate"] = calcular_h2h_empate(df.reset_index())
    return df.reset_index()


def agregar_diferencias(df: pd.DataFrame, largo: pd.DataFrame) -> pd.DataFrame:
    """Variables diff_* = Local - Visitante (sec. 2.4, la mas importante)."""
    avg_cols = _columnas_avg(largo)
    for c in avg_cols:
        if c in ("h2h_win_rate", "h2h_goals_avg"):
            continue
        home_c, away_c = f"{c}_home", f"{c}_away"
        if home_c in df.columns and away_c in df.columns:
            df[f"diff_{c}"] = df[home_c].fillna(0) - df[away_c].fillna(0)
    df["diff_h2h_win_rate"] = df["h2h_win_rate_home"] - df["h2h_win_rate_away"]
    df["diff_h2h_goals_avg"] = df["h2h_goals_avg_home"] - df["h2h_goals_avg_away"]

    # Estadisticas ricas de Mundial 2026: solo tienen sentido dentro del torneo
    for stat in RICH_STATS:
        for ventana in list(VENTANAS) + ["total"]:
            c = f"diff_avg_{stat}_{ventana}"
            if c in df.columns:
                df[c] = df[c].fillna(0)
                df.loc[~df["es_mundial_2026"].astype(bool), c] = 0
    return df


def agregar_elo_tier_confederacion(df: pd.DataFrame, teams: pd.DataFrame, valor_mercado: pd.DataFrame) -> pd.DataFrame:
    """Elo/FIFA, Tier, Peso de confederacion y Valor de Mercado (sec. 2.5)."""
    equipos = teams.merge(valor_mercado, on="team_id", how="left")
    equipos["tier"] = np.select(
        [equipos["elo_rating"] >= 1700, equipos["elo_rating"] >= 1600, equipos["elo_rating"] >= 1500],
        [1, 2, 3], default=4,
    )
    equipos["peso_confederacion"] = equipos["confederation"].map(PESOS_CONFEDERACION).fillna(0.7)

    cols = ["team_name_hist", "team_id", "elo_rating", "tier", "confederation", "peso_confederacion", "valor_mercado_eur"]
    home_e = equipos[cols].rename(columns={c: f"{c}_home" for c in cols if c != "team_name_hist"})
    home_e = home_e.rename(columns={"team_name_hist": "home_team"})
    away_e = equipos[cols].rename(columns={c: f"{c}_away" for c in cols if c != "team_name_hist"})
    away_e = away_e.rename(columns={"team_name_hist": "away_team"})

    df = df.merge(home_e, on="home_team", how="left").merge(away_e, on="away_team", how="left")

    df["diff_elo"] = df["elo_rating_home"] - df["elo_rating_away"]
    df["prob_implicita_elo"] = 1 / (1 + 10 ** (-df["diff_elo"] / 400))
    df["diff_tier"] = df["tier_home"] - df["tier_away"]
    df["diff_valor_mercado"] = (df["valor_mercado_eur_home"].fillna(0) - df["valor_mercado_eur_away"].fillna(0)) / 1_000_000
    return df


def construir_dataset_modelo(historico: pd.DataFrame, matches_2026: pd.DataFrame,
                              teams: pd.DataFrame, valor_mercado: pd.DataFrame) -> pd.DataFrame:
    """Orquesta toda la ingenieria de variables y devuelve el dataset ancho
    final, restringido a partidos donde ambos equipos pertenecen al universo
    de 48 selecciones del Mundial 2026 (para que Elo/Tier/Valor de Mercado
    esten siempre definidos)."""
    df = preparar_historico(historico, matches_2026)
    largo = _tabla_larga(df)
    largo = calcular_features_forma(largo)
    largo = calcular_h2h(largo)
    df = volver_a_ancho(df, largo)
    df = agregar_diferencias(df, largo)
    df = agregar_elo_tier_confederacion(df, teams, valor_mercado)

    ambos_conocidos = df["team_id_home"].notna() & df["team_id_away"].notna()
    df = df[ambos_conocidos].reset_index(drop=True)
    return df
