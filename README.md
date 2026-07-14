# Proyecto Predictivo Mundial 2026 — Octavos de Final

Modelo de Machine Learning (cascada de XGBoost) que predice qué selecciones
avanzan de **Octavos de Final** a **Cuartos de Final** en el Mundial 2026,
entrenado con el historial real de fútbol internacional (1872-2026).

La arquitectura combina dos regresores XGBoost (objetivo Tweedie) que
predicen goles esperados, apilados (*stacking*) con un clasificador
XGBoost 1X2 calibrado isotónicamente, más el "truco del espejo" (sede
neutral), ajuste de temperatura y una simulación de Monte Carlo con
penaltis para resolver cada cruce eliminatorio.

## Resultado

El resultado final es una imagen: **`output/clasificados_octavos.png`**,
con los 8 equipos que clasifican a Cuartos de Final (2 ya decididos en la
cancha + 6 predichos por el modelo, cada uno con su % de confianza).
## Resultado

### Tabla de predicciones (Octavos de Final)

| Fecha | Local | Visitante | Prob. Local | Prob. Empate | Prob. Visitante | Ganador predicho | Confianza | Marcador probable |
|---|---|---|---|---|---|---|---|---|
| 2026-07-04 | Canada | Morocco | 33.8% | 23.7% | 42.5% | **Morocco** | 56.1% | 0-2 |
| 2026-07-04 | Paraguay | France | 15.6% | 26.1% | 58.3% | **France** | 78.1% | 0-1 |
| 2026-07-05 | Brazil | Norway | 61.3% | 20.1% | 18.7% | **Brazil** | 77.0% | 1-0 |
| 2026-07-05 | Mexico | England | 27.5% | 27.4% | 45.0% | **England** | 62.5% | 0-1 |
| 2026-07-06 | Portugal | Spain | 28.9% | 25.6% | 45.5% | **Spain** | 61.1% | 0-1 |
| 2026-07-06 | United States | Belgium | 28.9% | 25.0% | 46.2% | **Belgium** | 61.9% | 0-1 |
| 2026-07-06 | Argentina | Egypt | 68.4% | 22.0% | 9.6% | **Argentina** | 87.9% | 2-0 |
| 2026-07-06 | Switzerland | Colombia | 35.9% | 19.0% | 45.0% | **Colombia** | 56.0% | 0-1 |

## Estructura del proyecto

```
Proyecto_Predictivo_Futbol/
├── data/                        # CSV de entrada (equipos, partidos, historial, plantillas)
├── src/
│   ├── config.py                # Rutas y constantes del modelo
│   ├── data_loader.py           # Carga y normalización de las fuentes crudas
│   ├── feature_engineering.py   # Medias móviles, H2H, Elo, diff_*, etc.
│   ├── modeling.py               # Entrenamiento de la cascada XGBoost + calibración
│   ├── predict.py                # Predicción de Octavos (espejo + Monte Carlo)
│   ├── visualize.py              # Genera clasificados_octavos.png
│   └── main.py                   # Orquesta todo el pipeline de punta a punta
├── output/
│   ├── clasificados_octavos.png     # <- AQUÍ SE VEN LOS EQUIPOS QUE CLASIFICAN
│   ├── predicciones_octavos.csv     # Tabla con las probabilidades de cada partido
│   └── models/modelo_cascada_octavos.pkl  # Modelos entrenados (joblib)
├── requirements.txt
└── README.md
```

## Requisitos
- **Python 3.10 o superior** (probado con 3.13)
- Las librerías listadas en `requirements.txt`: `pandas`, `numpy`, `scipy`,
  `scikit-learn`, `xgboost`, `joblib`, `matplotlib`

## Instalación

```bash
# 1. Clonar el repositorio y entrar a la carpeta del proyecto
cd Proyecto_Predictivo_Futbol

# 2. (Recomendado) crear un entorno virtual
python -m venv venv
source venv/bin/activate      # en Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Cómo correrlo

Todo el pipeline (carga de datos → ingeniería de variables → entrenamiento
del modelo → predicción → imagen final) se ejecuta con un solo comando desde
la raíz del proyecto:

```bash
python src/main.py
```

Esto va a:
1. Cargar y unir los CSV de `data/` (equipos, partidos del Mundial 2026,
   estadísticas por partido, historial internacional desde 1872, plantillas).
2. Calcular todas las variables del modelo (forma reciente, head-to-head,
   Elo, tier, peso por confederación, valor de mercado, etc.).
3. Entrenar los 3 modelos de la cascada (2 regresores de goles + 1
   clasificador 1X2 calibrado) con ~7.600 partidos históricos entre las 48
   selecciones del Mundial 2026, y guardarlos en `output/models/`.
4. Predecir los partidos de Octavos de Final que aún no se han jugado.
5. Generar `output/clasificados_octavos.png`.

El entrenamiento completo tarda entre 1 y 3 minutos en una laptop normal (no
requiere GPU).

## ¿Dónde veo los equipos que pasan a Cuartos de Final?

Abre la imagen generada en:

```
output/clasificados_octavos.png
```

Ahí aparecen las 8 selecciones clasificadas, cada una marcada con ✓, junto
al rival al que vence, el marcador (real o el más probable según el
modelo), y una etiqueta que distingue **"RESULTADO REAL"** (partidos ya
jugados) de **"PREDICCIÓN IA"** (partidos que el modelo predijo, con su %
de confianza calculado por simulación de Monte Carlo).

También puedes revisar el detalle numérico de cada partido (probabilidades
Local/Empate/Visitante, % de Monte Carlo, marcador probable) en:

```
output/predicciones_octavos.csv
```

## Notas sobre los datos

- `data/teams.csv`, `data/matches.csv`, `data/match_team_stats.csv`: datos
  oficiales del Mundial 2026 (equipos, partidos, estadísticas por partido).
- `data/results.csv`, `data/shootouts.csv`: historial de fútbol
  internacional 1872-2026 (incluye el propio Mundial 2026 en curso), usado
  para entrenar el modelo con más de 7.000 partidos reales.
- `data/squads_and_players.csv`: valor de mercado de las plantillas.

No se necesita ninguna llave de API ni conexión a internet: todo corre en
local a partir de estos CSV.
