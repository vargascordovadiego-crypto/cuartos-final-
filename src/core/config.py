"""
Constantes y parametros del modelo (cascada de goles + clasificador 1X2).

Esta capa de dominio no conoce el sistema de archivos: las rutas de datos y
de salida viven en `infra/paths.py`.
"""

# Ventanas de medias moviles ("forma reciente")
VENTANAS = [2, 5]

# Umbral de goleada para sobreponderar muestras de entrenamiento
UMBRAL_GOLEADA = 3
PESO_GOLEADA = 1.5

# Decaimiento exponencial de recencia (nuestro historico llega a 150 anios,
# no unos pocos, de ahi la reescala frente a un decay tipico)
DECAY_RECENCIA = 0.0006

# Peso por confederacion
PESOS_CONFEDERACION = {
    "UEFA": 1.00,      # Europa
    "CONMEBOL": 0.95,  # Sudamerica
    "CONCACAF": 0.75,  # Norteamerica
    "CAF": 0.60,       # Africa
    "AFC": 0.70,       # Asia
    "OFC": 0.50,       # Oceania
}

# Normalizacion de nombres: teams.csv (Mundial 2026) -> results.csv / shootouts.csv (historico)
MAPEO_NOMBRES = {
    "Czechia": "Czech Republic",
    "USA": "United States",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
}

TWEEDIE_VARIANCE_POWER = 1.5
RANDOM_STATE = 42
N_SIMULACIONES_MONTECARLO = 10_000
