"""
Caso de uso 3: con los 8 clasificados de Octavos (reales + predichos), arma
el cruce de Cuartos de Final, lo predice con la misma cascada ya entrenada,
guarda la tabla de probabilidades y genera la imagen final de clasificados.
"""
import pandas as pd

from src.core import bracket
from src.core import features
from src.core import predictor
from src.infra import image_report
from src.infra.paths import OUTPUT_DIR


def ejecutar(ctx: dict, resultados_octavos: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/3] Armando el cruce de Cuartos de Final y prediciendolo con el mismo modelo...")
    cruces = bracket.construir_cruces_cuartos(resultados_octavos)
    historico_aug, matches_aug = bracket.agregar_fixture_cuartos(
        ctx["historico"], ctx["matches_2026"], ctx["teams"], cruces
    )
    df_modelo_cuartos = features.construir_dataset_modelo(
        historico_aug, matches_aug, ctx["teams"], ctx["valor_mercado"]
    )

    pendientes_mask = df_modelo_cuartos["home_score"].isna() & (df_modelo_cuartos["stage_id"] == 4)
    df_pendientes = df_modelo_cuartos[pendientes_mask].reset_index(drop=True)
    resultados = predictor.predecir_partidos_pendientes(df_pendientes, ctx["modelo"])

    for _, r in resultados.sort_values("match_id").iterrows():
        print(f"      {r['equipo_local']:<24} vs {r['equipo_visitante']:<24} -> "
              f"avanza {r['ganador_predicho']:<20} ({r['confianza']*100:.0f}% conf.) "
              f"marcador probable {r['marcador_probable']}")

    resultados.to_csv(OUTPUT_DIR / "predicciones_cuartos.csv", index=False)

    print("      Generando 'clasificados_cuartos.png'...")
    ruta_img = OUTPUT_DIR / "clasificados_cuartos.png"
    image_report.generar_imagen(resultados, ruta_img)
    print(f"      Imagen guardada en {ruta_img}")

    return resultados
