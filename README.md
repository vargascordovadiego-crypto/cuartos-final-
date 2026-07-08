# Proyecto Predictivo Mundial 2026 — Cuartos de Final

Modelo de Machine Learning (cascada de XGBoost) que predice qué selecciones
avanzan de **Cuartos de Final** a **Semifinales** en el Mundial 2026,
entrenado con el historial real de fútbol internacional (1872-2026).

La arquitectura combina dos regresores XGBoost (objetivo Tweedie) que
predicen goles esperados, apilados (*stacking*) con un clasificador
XGBoost 1X2 calibrado isotónicamente, más el "truco del espejo" (sede
neutral), ajuste de temperatura y una simulación de Monte Carlo con
penaltis para resolver cada cruce eliminatorio.

El modelo se entrena **una sola vez** y se usa en dos pasadas encadenadas:
primero resuelve los Octavos de Final (los 8 resultados ya son reales,
verificados e incorporados a `data/` tras jugarse cada partido), y con esos
8 clasificados arma automáticamente el cruce de Cuartos de Final (ver
`src/core/bracket.py`) para predecirlo con exactamente el mismo modelo. Los 6
resultados de Octavos que faltaban (Brasil-Noruega, México-Inglaterra,
Portugal-España, EE.UU.-Bélgica, Argentina-Egipto, Suiza-Colombia) se
sumaron también como partidos reales de entrenamiento.

## Resultado

El resultado final es una imagen: **`output/clasificados_cuartos.png`**,
con los 4 equipos que clasifican a Semifinales, cada uno con su marcador
probable y su % de confianza (simulación de Monte Carlo).

## Estructura del proyecto

Arquitectura por capas: `core` (dominio del modelo, no conoce el disco),
`infra` (acceso a CSV/imagenes) y `app` (casos de uso que orquestan todo).

```
MODELO CUARTOS/
├── data/                        # CSV de entrada (equipos, partidos, historial, plantillas)
├── src/
│   ├── core/                    # Dominio: el modelo en si, sin IO
│   │   ├── config.py             # Constantes e hiperparametros del modelo
│   │   ├── features.py           # Medias móviles, H2H, Elo, diff_*, etc.
│   │   ├── cascade_model.py      # Entrenamiento de la cascada XGBoost + calibración
│   │   ├── predictor.py          # Predicción de partidos (espejo + Monte Carlo)
│   │   └── bracket.py            # Arma el cruce de Cuartos a partir de los clasificados de Octavos
│   ├── infra/                    # Infraestructura: todo lo que toca el sistema de archivos
│   │   ├── paths.py               # Rutas de datos y salidas
│   │   ├── csv_sources.py         # Carga y normalización de las fuentes crudas
│   │   └── image_report.py        # Genera clasificados_cuartos.png
│   └── app/                      # Aplicación: casos de uso + entrypoint
│       ├── use_cases/
│       │   ├── train_cascade.py         # Carga datos + entrena la cascada
│       │   ├── resolve_round_of_16.py   # Confirma quien llega a Cuartos
│       │   └── predict_quarterfinals.py # Arma y predice el cruce de Cuartos
│       └── main.py                # Orquesta los 3 casos de uso de punta a punta
├── output/
│   ├── clasificados_cuartos.png            # <- AQUÍ SE VEN LOS EQUIPOS QUE CLASIFICAN
│   ├── predicciones_cuartos.csv            # Tabla con las probabilidades de cada cruce de Cuartos
│   └── models/modelo_cascada_mundial2026.pkl  # Modelo entrenado (joblib)
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
cd "MODELO CUARTOS"

# 2. (Recomendado) crear un entorno virtual
python -m venv venv
source venv/bin/activate      # en Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Cómo correrlo

Todo el pipeline (carga de datos → ingeniería de variables → entrenamiento
del modelo → resolución de Octavos → predicción de Cuartos → imagen final)
se ejecuta con un solo comando desde la raíz del proyecto:

```bash
python -m src.app.main
```

Esto va a:
1. Cargar y unir los CSV de `data/` (equipos, partidos del Mundial 2026,
   estadísticas por partido, historial internacional desde 1872, plantillas).
2. Calcular todas las variables del modelo (forma reciente, head-to-head,
   Elo, tier, peso por confederación, valor de mercado, etc.).
3. Entrenar los 3 modelos de la cascada (2 regresores de goles + 1
   clasificador 1X2 calibrado) con ~7.600 partidos históricos entre las 48
   selecciones del Mundial 2026, y guardarlos en `output/models/`.
4. Confirmar los 8 resultados reales de Octavos de Final ya cargados en
   `data/`, para saber qué 8 selecciones llegan a Cuartos.
5. Armar el cruce de Cuartos de Final a partir de esos 8 clasificados y
   predecirlo con el mismo modelo (mismo espejo + Monte Carlo).
6. Generar `output/clasificados_cuartos.png`.

El entrenamiento completo tarda entre 1 y 3 minutos en una laptop normal (no
requiere GPU).

## ¿Dónde veo los equipos que pasan a Semifinales?

Abre la imagen generada en:

```
output/clasificados_cuartos.png
```

Ahí aparecen las 4 selecciones clasificadas a Semifinales, cada una marcada
con ✓, junto al rival al que vence, el marcador más probable según el
modelo, y su % de confianza calculado por simulación de Monte Carlo.

También puedes revisar el detalle numérico de cada partido (probabilidades
Local/Empate/Visitante, % de Monte Carlo, marcador probable) en:

```
output/predicciones_cuartos.csv
```

## ¿Cómo se arma el cruce de Cuartos si todavía no está jugado?

El dataset (`data/matches.csv`) solo traía los partidos hasta Octavos de
Final (stage_id 1, 2 y 3); los cruces de Cuartos dependen de quién gane cada
Octavo, así que no existían todavía como filas fijas. `src/core/bracket.py`
resuelve esto sin tocar el núcleo del modelo:

1. Toma los 8 resultados reales de Octavos (Marruecos, Francia, Noruega,
   Inglaterra, España, Bélgica, Argentina y Suiza, ya cargados en `data/`).
2. Con esos 8 clasificados arma las 4 llaves de Cuartos siguiendo el bracket
   oficial (confirmado con el propio dataset: `data/results.csv` ya traía de
   antemano el cruce "Francia vs Marruecos" del 2026-07-09, que sale de los
   partidos 89 y 90 de Octavos) y genera esas 4 filas nuevas para que el
   mismo modelo las prediga exactamente igual que cualquier otro partido.

## Notas sobre los datos

- `data/teams.csv`, `data/matches.csv`, `data/match_team_stats.csv`: datos
  oficiales del Mundial 2026 (equipos, partidos, estadísticas por partido).
- `data/results.csv`, `data/shootouts.csv`: historial de fútbol
  internacional 1872-2026 (incluye el propio Mundial 2026 en curso), usado
  para entrenar el modelo con más de 7.000 partidos reales.
- `data/squads_and_players.csv`: valor de mercado de las plantillas.

No se necesita ninguna llave de API ni conexión a internet: todo corre en
local a partir de estos CSV.
