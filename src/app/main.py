"""
Punto de entrada unico del proyecto. Orquesta 3 casos de uso encadenados,
todos resueltos por la MISMA cascada XGBoost entrenada una sola vez:

  1. train_cascade        -- carga datos, calcula variables, entrena la cascada.
  2. resolve_round_of_16   -- confirma quien llega a Cuartos (real + predicho).
  3. predict_quarterfinals -- arma el cruce de Cuartos y lo predice.

Uso (desde la raiz del proyecto):
    python -m src.app.main
"""
from src.app.use_cases import predict_quarterfinals, resolve_round_of_16, train_cascade


def main():
    print("=" * 70)
    print("PROYECTO PREDICTIVO MUNDIAL 2026 -- Cuartos de Final -> Semifinales")
    print("=" * 70)

    ctx = train_cascade.ejecutar()
    resultados_octavos = resolve_round_of_16.ejecutar(ctx)
    resultados_cuartos = predict_quarterfinals.ejecutar(ctx, resultados_octavos)

    print("\n" + "=" * 70)
    print("LOS 4 EQUIPOS QUE CLASIFICAN A SEMIFINALES:")
    for _, r in resultados_cuartos.sort_values("match_id").iterrows():
        rival = r["equipo_visitante"] if r["ganador_predicho"] == r["equipo_local"] else r["equipo_local"]
        print(f"   -> {r['ganador_predicho']:<20} (vence a {rival}, {r['marcador_probable']})")
    print("=" * 70)


if __name__ == "__main__":
    main()
