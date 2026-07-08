"""
Caso de uso 1: cargar las fuentes crudas, calcular la ingenieria de
variables completa y entrenar la cascada XGBoost una sola vez.

Devuelve un contexto (dict) con todo lo que necesitan los casos de uso
siguientes (resolve_round_of_16 y predict_quarterfinals), para no volver a
tocar disco ni reentrenar nada.
"""
import joblib

from src.core import cascade_model as cascade
from src.core import features
from src.infra import csv_sources as sources
from src.infra.paths import MODELS_DIR


def ejecutar() -> dict:
    print("\n[1/3] Cargando fuentes crudas (teams, results, matches, squads)...")
    teams = sources.cargar_equipos()
    valor_mercado = sources.cargar_valor_mercado()
    historico = sources.cargar_historico_resultados()
    shootouts = sources.cargar_shootouts()
    historico = sources.resolver_ganador_penaltis(historico, shootouts)
    matches_2026 = sources.construir_matches_2026_enriquecido(teams)
    print(f"      {len(historico):,} partidos historicos (1872-2026) | {len(teams)} selecciones Mundial 2026")

    print("      Ingenieria de variables (medias moviles, H2H, Elo, diff_*)...")
    df_modelo = features.construir_dataset_modelo(historico, matches_2026, teams, valor_mercado)
    print(f"      Dataset final: {len(df_modelo):,} partidos entre selecciones del Mundial 2026")

    df_train = df_modelo[df_modelo["home_score"].notna()].reset_index(drop=True)
    print(f"      Entrenamiento: {len(df_train):,} partidos")

    print("      Entrenando cascada XGBoost (Piso 1: goles Tweedie | Piso 2: 1X2 calibrado)...")
    modelo = cascade.entrenar_pipeline(df_train)
    joblib.dump(modelo, MODELS_DIR / "modelo_cascada_mundial2026.pkl")
    print(f"      Modelo guardado en {MODELS_DIR}")

    return {
        "teams": teams,
        "valor_mercado": valor_mercado,
        "historico": historico,
        "matches_2026": matches_2026,
        "df_modelo": df_modelo,
        "modelo": modelo,
    }
