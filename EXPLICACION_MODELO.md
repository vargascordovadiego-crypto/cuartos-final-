# ¿Qué modelo se usó, por qué, y cómo se mide si funciona?

Este documento explica en lenguaje simple las decisiones detrás del modelo
predictivo, qué variables usa y qué tan bien predice. Los números de este
documento corresponden a una ejecución real del pipeline (`python -m src.app.main`)
sobre los datos actuales del proyecto.

El modelo se entrena **una sola vez** sobre el historial 1872-2026, y esa
misma cascada se reutiliza en dos pasadas: primero resuelve los Octavos de
Final (los 8 con resultado real, incorporado a `data/results.csv` y
`data/matches.csv` a partir de fuentes públicas tras jugarse cada partido),
y con esos 8 clasificados arma y predice el cruce de Cuartos de Final
(`src/core/bracket.py`). El núcleo del modelo -- arquitectura, variables y
calibración -- es idéntico en ambas pasadas; lo único que cambia son los
partidos que se le dan de entrada. Los 6 resultados de Octavos que faltaban
(Brasil-Noruega, México-Inglaterra, Portugal-España, EE.UU.-Bélgica,
Argentina-Egipto, Suiza-Colombia) también se sumaron como **partidos de
entrenamiento reales**, así el modelo tiene 6 filas más de historial genuino
para calcular forma reciente, H2H y validación temporal.

---

## 1. ¿Qué modelo se usó?

Se usó **XGBoost** (Extreme Gradient Boosting), organizado en una
**arquitectura en cascada de 2 pisos** más una capa de calibración y una
simulación final. No es "un modelo", son tres modelos que se pasan
información entre sí:

```
Piso 1 (goles esperados)          Piso 2 (resultado 1X2)              Ajuste final
┌─────────────────────┐           ┌──────────────────────┐           ┌───────────────────┐
│ Regresor goles Local │──┐        │                      │           │                    │
│ (XGBoost, Tweedie)   │  ├──────▶ │ Clasificador 1X2     │──────────▶│ Calibración        │
│                      │  │        │ (XGBoost, softprob)  │           │ isotónica +        │
│ Regresor goles       │──┘        │ Local / Empate /     │           │ temperatura        │
│ Visitante (Tweedie)  │           │ Visitante            │           │                    │
└─────────────────────┘           └──────────────────────┘           └───────────────────┘
                                                                               │
                                                                               ▼
                                                                  Simulación Monte Carlo
                                                                  (10.000 mundiales en
                                                                  miniatura, con penaltis
                                                                  si hay empate)
```

**Piso 1 — ¿cuántos goles se esperan?**
Dos modelos (uno para el equipo Local, otro para el Visitante) intentan
adivinar cuántos goles va a meter cada selección. Se entrenan con objetivo
`Tweedie`, que es el adecuado para datos como "goles por partido": números
enteros, casi siempre bajos (0, 1, 2...) y nunca negativos.

**Piso 2 — ¿quién gana?**
Un tercer modelo recibe dos tipos de información: (a) las variables propias
del partido (Elo, forma reciente, historial entre los equipos, etc.) y (b)
las predicciones de goles del Piso 1, como pistas adicionales. Con eso decide
tres probabilidades: gana el Local, empatan, o gana el Visitante.

**Calibración — que el % de confianza sea honesto.**
Un modelo de Machine Learning puede decir "80% de probabilidad" sin que eso
signifique que ese equipo gane 8 de cada 10 veces en la realidad. Por eso se
aplica `CalibratedClassifierCV` (calibración isotónica): ajusta las
probabilidades de salida para que efectivamente reflejen la frecuencia real
observada en los datos históricos.

**Temperatura — afinar la seguridad del modelo.**
Después de calibrar, se prueba un parámetro extra (`T`) que "afila" o
"suaviza" las probabilidades finales, buscando el valor que minimiza el error
en un conjunto de validación reservado (ver sección 3). En la última
ejecución, el valor óptimo fue **T = 0.90**.

**Simulación de Monte Carlo — resolver el partido.**
Con las probabilidades finales (Local/Empate/Visitante), se "juegan" 10.000
partidos simulados al azar. Si en una simulación da empate, se resuelve por
penaltis (como en un Mundial real). El % de veces que gana cada equipo en
esas 10.000 simulaciones es el "% de confianza" que se muestra en los
resultados.

### ¿Por qué XGBoost y no otro modelo?

- Los datos son **tabulares** (una fila = un partido, columnas = estadísticas),
  y XGBoost es el estándar de facto para ese tipo de datos — normalmente
  supera a redes neuronales cuando no hay millones de filas.
- Aprende relaciones **no lineales** por sí solo (por ejemplo, que una
  diferencia de Elo de 200 puntos no vale lo mismo si ambos equipos son
  débiles que si ambos son top-5).
- Tolera bien variables con muchos ceros o datos ausentes (por ejemplo, las
  estadísticas de xG/posesión/tiros solo existen para partidos del Mundial
  2026, y son 0 para todo el historial anterior).
- Es rápido de entrenar (el pipeline completo tarda 1–3 minutos sin GPU),
  lo cual permitió usar validación temporal y calibración sin que el proyecto
  se vuelva pesado de ejecutar.
- **No se hizo búsqueda automática de hiperparámetros** (`RandomizedSearchCV`)
  para mantener el tiempo de ejecución bajo; se usaron valores razonables
  fijos (300 árboles para los regresores de goles, 250 para el clasificador,
  profundidad máxima 4, tasa de aprendizaje 0.05).

---

## 2. Las variables (features) que ve el modelo

Todas las variables que finalmente usa el modelo son **diferencias entre el
equipo Local y el Visitante** (`diff_x = valor_local − valor_visitante`). Esto
tiene una ventaja importante: el modelo no aprende "a quién le gusta más el
equipo A", aprende "qué tan mejor es un equipo que otro en cada aspecto",
lo cual generaliza mucho mejor a partidos nuevos.

Antes de llegar a las 19 variables finales, el pipeline calcula muchas más
(promedios, historial, etc.) y luego descarta las redundantes (ver sección
2.2). Estas son las **19 variables que el modelo realmente usa** en la
última ejecución:

| Variable | Qué significa en palabras simples |
|---|---|
| `diff_elo` | Diferencia de rating Elo/FIFA. La variable más fuerte: resume la "calidad" histórica de cada selección. |
| `prob_implicita_elo` | La probabilidad de victoria que el Elo por sí solo sugeriría (fórmula estándar de Elo), como referencia. |
| `diff_tier` | Diferencia de "categoría" (1 a 4) asignada según el Elo de cada equipo. |
| `diff_valor_mercado` | Diferencia en el valor de mercado (millones de €) de las plantillas. |
| `diff_avg_goals_2` / `_5` / `_total` | Diferencia en el promedio de goles anotados en los últimos 2, 5, o todos los partidos históricos de cada equipo. |
| `diff_avg_conceded_2` / `_5` / `_total` | Igual, pero goles **recibidos** (qué tan sólida es la defensa). |
| `diff_trend_goals_5` | Diferencia en la "tendencia" de goles de los últimos 5 partidos (¿el equipo está mejorando o empeorando su ataque?). |
| `diff_clean_sheet_rate_5` | Diferencia en la frecuencia con la que cada equipo dejó su arco en cero en sus últimos 5 partidos. |
| `diff_h2h_win_rate` | Diferencia en el historial directo entre ambos equipos (quién le ha ganado más veces a quién). |
| `diff_h2h_goals_avg` | Diferencia en el promedio de goles metidos específicamente en los enfrentamientos entre esos dos equipos. |
| `diff_avg_xg_2` | Diferencia en goles esperados (xG) de los últimos 2 partidos — solo tiene valor distinto de 0 dentro del Mundial 2026 (antes de eso no existe esta estadística). |
| `diff_avg_possession_pct_5` | Diferencia en % de posesión de balón (últimos 5 partidos, solo Mundial 2026). |
| `diff_avg_total_shots_2` | Diferencia en tiros totales (últimos 2 partidos, solo Mundial 2026). |
| `diff_avg_shots_on_target_2` | Diferencia en tiros al arco (últimos 2 partidos, solo Mundial 2026). |
| `diff_avg_corners_2` | Diferencia en tiros de esquina (últimos 2 partidos, solo Mundial 2026). |
| `diff_avg_saves_2` | Diferencia en atajadas del arquero (últimos 2 partidos, solo Mundial 2026). |

Además, internamente el clasificador del Piso 2 recibe dos columnas extra que
no son "datos crudos" sino predicciones del Piso 1: `pred_goles_l` y
`pred_goles_v` (los goles esperados de Local y Visitante). Esto es lo que
técnicamente se llama *stacking*: un modelo usa la salida de otro modelo como
insumo.

### 2.1 ¿De dónde salen estas variables?

- **Forma reciente** (`avg_goals`, `avg_conceded`, etc.): se calculan sobre
  **todo el historial de cada selección desde 1872**, no solo el Mundial
  2026, usando goles reales como aproximación de "rendimiento" (no existía
  xG antes del torneo).
- **Head-to-Head (H2H)**: se calcula por cada pareja de equipos específica
  (ej. Argentina vs Brasil), mirando solo los partidos anteriores a la fecha
  del partido que se está prediciendo (nunca se usa información del futuro).
- **Elo, Tier, peso por confederación y valor de mercado**: solo existen para
  las 48 selecciones del Mundial 2026, así que el dataset de entrenamiento se
  restringe a partidos históricos donde **ambos** equipos pertenecen a ese
  grupo de 48 (esto da 7.617 partidos con resultado conocido, entre 1872 y
  2026).
- **Estadísticas ricas del propio Mundial 2026** (xG, posesión, tiros,
  corners, atajadas): solo existen desde que arrancó el torneo. Fuera de esa
  ventana se dejan en 0, y por eso su impacto en el modelo es naturalmente
  bajo comparado con Elo o la forma histórica — son "pistas extra" que solo
  se activan cuando hay datos reales del Mundial.

### 2.2 ¿Por qué 19 y no más?

El pipeline calcula muchas más variables candidatas, pero antes de entrenar
descarta las que están **muy correlacionadas entre sí** (correlación > 0.9),
quedándose con la de mayor varianza de cada par redundante. Por ejemplo, si
"goles promedio en los últimos 2 partidos" y "goles promedio en los últimos 5"
dijeran casi lo mismo, se eliminaría una. En la última ejecución, este filtro
dejó **19 variables finales** de un conjunto más grande.

---

## 3. Las métricas: ¿qué tan bueno es el modelo?

### 3.1 La métrica principal: log-loss

Para un problema de "Local / Empate / Visitante" con probabilidades (no solo
una respuesta binaria), la métrica estándar no es el % de aciertos
(*accuracy*), sino el **log-loss** (pérdida logarítmica). En palabras
simples:

- El log-loss **castiga fuerte** cuando el modelo está muy seguro y se
  equivoca (ej. decir "90% gana el Local" y que gane el Visitante).
- **Premia** cuando el modelo está seguro y acierta.
- Es más bajo = mejor. Un modelo que simplemente dijera "33% / 33% / 33%"
  siempre (sin aprender nada) tendría un log-loss de referencia de
  **ln(3) ≈ 1.0986**.

**Resultado real de la última ejecución:**

| Métrica | Valor |
|---|---|
| Log-loss con temperatura neutra (T=1) | **1.0058** |
| Log-loss con la mejor temperatura encontrada (T=0.95) | **1.0056** |
| Referencia de "no saber nada" (33/33/33 siempre) | 1.0986 |

Es decir, el modelo mejora sobre "tirar una moneda de 3 caras" (1.0986), pero
la mejora es modesta (~8%). Esto es esperable y honesto: predecir resultados
de fútbol es un problema con muchísima aleatoriedad inherente (un gol de
media cancha, un penal dudoso, una tarjeta roja temprana), así que ningún
modelo — ni las casas de apuestas profesionales — logra un log-loss
dramáticamente mejor que este orden de magnitud.

### 3.2 Cómo se calculó esta métrica (para que sea confiable)

1. **Validación temporal, no aleatoria.** Se ordenan los 7.623 partidos de
   entrenamiento por fecha, y se separa el **15% más reciente** (1.144
   partidos) como conjunto de validación. El modelo se entrena solo con el
   85% más antiguo (6.479 partidos) y se evalúa contra ese 15% que nunca vio.
   Esto simula la situación real: predecir partidos futuros con datos del
   pasado, evitando que el modelo "haga trampa" mirando el futuro.
2. **Sin fuga de información (*no leakage*) en el stacking.** Las
   predicciones de goles que alimentan al clasificador del Piso 2 se generan
   con `cross_val_predict` (5 particiones, `KFold` sin mezclar por fecha), de
   modo que ningún partido es predicho por un modelo que fue entrenado
   viendo ese mismo partido.
3. **Una vez medida la calidad, se reentrena con el 100% de los datos** (los
   7.623 partidos completos, ya con los 8 resultados reales de Octavos
   incluidos) para el modelo que efectivamente se usa a la hora de predecir
   los 4 cruces de Cuartos de Final. La métrica de validación (1.0056) es la
   mejor estimación honesta de qué tan bien generalizará ese modelo final.

### 3.3 Otros números de contexto de esta ejecución

| Dato | Valor |
|---|---|
| Partidos históricos totales cargados (1872-2026) | 49.502 |
| Selecciones del Mundial 2026 | 48 |
| Partidos usados para entrenar (ambos equipos del Mundial 2026, resultado conocido) | 7.623 |
| Partidos de Octavos de Final ya jugados (resultado real, no se predicen) | 8 (los 8) |
| Cruces de Cuartos de Final armados a partir de esos 8 clasificados | 4 |
| Variables finales usadas por el modelo | 19 |
| Simulaciones de Monte Carlo por partido | 10.000 |
| Temperatura óptima encontrada | 0.95 |

---

## 4. De Octavos a Cuartos: el mismo modelo, un cruce nuevo

`data/matches.csv` solo traía partidos hasta Octavos de Final -- los cruces
de Cuartos no existen todavía como filas fijas porque dependen de quién gane
cada Octavo. Los 8 resultados de Octavos ya se conocen en la realidad (se
verificaron uno por uno y se cargaron en `data/results.csv` /
`data/matches.csv` / `data/shootouts.csv`), así que ya no hace falta que el
modelo los prediga: Marruecos, Francia, Noruega, Inglaterra, España,
Bélgica, Argentina y Suiza son los 8 clasificados reales.

`src/core/bracket.py` toma esos 8 ganadores y arma las 4 llaves de Cuartos
siguiendo el bracket oficial (confirmado con el propio dataset, que ya traía
de antemano el cruce "Francia vs Marruecos" del 2026-07-09, salida de los
partidos 89 y 90 de Octavos, y con el calendario público de Cuartos:
Francia-Marruecos 9/7, España-Bélgica 10/7, Noruega-Inglaterra y
Argentina-Suiza 11/7). Esas 4 filas nuevas pasan por la misma ingeniería de
variables y la misma predicción (espejo + Monte Carlo) que cualquier otro
partido -- el modelo no distingue si el partido ya se jugó o es futuro, solo
ve las mismas 19 variables de siempre.

## 5. Resumen en una frase

Se usó una **cascada de XGBoost** (goles esperados → resultado 1X2)
calibrada y afinada por temperatura, alimentada con **19 variables**
(la mayoría diferencias Local-Visitante de Elo, forma reciente, historial
directo y valor de mercado), y se validó de forma temporal (entrenar con el
pasado, medir contra el futuro más reciente) obteniendo un **log-loss de
1.0056**, una mejora modesta pero real sobre el "no saber nada" (1.0986) —
coherente con lo impredecible que es el fútbol por naturaleza.
