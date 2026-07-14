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

## Resultado

### Detalle numérico de cada cruce

| Partido | Prob. Local | Prob. Empate | Prob. Visitante | Ganador predicho | Confianza (Monte Carlo) | Marcador probable |
|---|---|---|---|---|---|---|
| Francia 🆚 Marruecos | 58.2% | 21.6% | 20.2% | **Francia** | 74.8% | 2-0 |
| Inglaterra 🆚 Noruega | 58.7% | 25.0% | 16.3% | **Inglaterra** | 78.0% | 1-0 |
| Bélgica 🆚 España | 27.3% | 24.4% | 48.3% | **España** | 63.9% | 0-1 |
| Suiza 🆚 Argentina | 19.0% | 24.9% | 56.1% | **Argentina** | 74.6% | 0-1 |

*(Fuente: `output/predicciones_cuartos.csv`, corte de datos 2026-07-07. "Prob. Local/Empate/Visitante" son las probabilidades 1X2 del clasificador calibrado; "Confianza" es el % del ganador predicho tras la simulación de Monte Carlo con penaltis.)*

**Semifinalistas predichos por el modelo: Francia, Inglaterra, España y Argentina.**

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
