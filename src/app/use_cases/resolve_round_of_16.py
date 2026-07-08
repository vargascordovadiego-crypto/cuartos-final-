"""
Caso de uso 2: a partir del dataset ya calculado por train_cascade, confirma
los resultados reales de Octavos de Final y predice con el modelo cualquiera
que aun quedara pendiente. Devuelve la tabla completa (real + predicho) que
necesita el bracket de Cuartos para saber quien avanza.
"""
import pandas as pd

from src.core import predictor


def ejecutar(ctx: dict) -> pd.DataFrame:
    print("\n[2/3] Confirmando los resultados reales de Octavos de Final (y prediciendo con "
          "el modelo cualquiera que aun quedara pendiente)...")
    df_modelo = ctx["df_modelo"]
    modelo = ctx["modelo"]

    pendientes_mask = df_modelo["home_score"].isna() & (df_modelo["stage_id"] == 3)
    decididos_mask = df_modelo["home_score"].notna() & (df_modelo["stage_id"] == 3)
    df_pendientes = df_modelo[pendientes_mask].reset_index(drop=True)
    df_decididos = df_modelo[decididos_mask].reset_index(drop=True)
    print(f"      Octavos ya jugados: {len(df_decididos)} | Octavos por predecir: {len(df_pendientes)}")

    resultados = predictor.predecir_partidos_pendientes(df_pendientes, modelo)
    resultados = pd.concat(
        [resultados, predictor.partidos_ya_decididos(df_decididos)], ignore_index=True
    )

    for _, r in resultados.sort_values("match_id").iterrows():
        estado = "(real)" if r["ya_jugado"] else f"({r['confianza']*100:.0f}% conf.)"
        print(f"      {r['equipo_local']:<24} vs {r['equipo_visitante']:<24} -> "
              f"avanza {r['ganador_predicho']:<20} {estado}")

    return resultados
