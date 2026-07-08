"""
Arquitectura en cascada (stacking) con XGBoost (sec. 3 del manual):

Piso 1 -- dos XGBRegressor con objetivo Tweedie que predicen los goles de
          Local y Visitante (proxy de "goles esperados", ya que no hay xG
          historico anterior al Mundial 2026).
Piso 2 -- un XGBClassifier 1X2 (multi:softprob) que recibe las variables
          diff_*/Elo/H2H MAS las predicciones de goles del Piso 1 como
          meta-variables (stacking), generadas sin fuga de informacion via
          cross_val_predict(KFold(shuffle=False)).
Calibracion -- CalibratedClassifierCV(method='isotonic') para que las
          probabilidades de salida sean estadisticamente honestas.
Temperatura -- se barre T en un set de validacion temporal para elegir el
          valor que minimiza el log-loss (sec. 3.5).

No se hizo busqueda de hiperparametros (RandomizedSearchCV) para mantener el
tiempo de ejecucion razonable; se usan hiperparametros razonables fijos.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_predict

from src.core.config import DECAY_RECENCIA, PESO_GOLEADA, RANDOM_STATE, UMBRAL_GOLEADA

HOY = pd.Timestamp("2026-07-05")

LABEL_MAP = {"home": 0, "draw": 1, "away": 2}


def obtener_columnas_features(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if c.startswith("diff_")]
    cols += ["prob_implicita_elo"]
    return sorted(set(cols))


def eliminar_multicolinealidad(df: pd.DataFrame, columnas: list, umbral: float = 0.9) -> list:
    """Sec. 2.7: descarta una de cada par de variables con correlacion
    absoluta > umbral (se conserva la de mayor varianza)."""
    corr = df[columnas].corr().abs()
    descartadas = set()
    for i, c1 in enumerate(columnas):
        if c1 in descartadas:
            continue
        for c2 in columnas[i + 1:]:
            if c2 in descartadas:
                continue
            if corr.loc[c1, c2] > umbral:
                if df[c1].var() >= df[c2].var():
                    descartadas.add(c2)
                else:
                    descartadas.add(c1)
    return [c for c in columnas if c not in descartadas]


def _pesos_muestra(df: pd.DataFrame, goles: pd.Series) -> np.ndarray:
    peso_goleada = np.where(goles >= UMBRAL_GOLEADA, PESO_GOLEADA, 1.0)
    dias = (HOY - df["date"]).dt.days.clip(lower=0)
    peso_recencia = np.exp(-DECAY_RECENCIA * dias)
    return peso_goleada * peso_recencia


def _nuevo_regresor_goles() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.5,
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
    )


def _nuevo_clasificador() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
    )


def _oof_meta_goles(X: pd.DataFrame, y_l: pd.Series, y_v: pd.Series) -> tuple:
    kf = KFold(n_splits=5, shuffle=False)
    pred_l = cross_val_predict(_nuevo_regresor_goles(), X, y_l, cv=kf)
    pred_v = cross_val_predict(_nuevo_regresor_goles(), X, y_v, cv=kf)
    return pred_l, pred_v


def _validar_temperatura(df_train: pd.DataFrame, columnas: list) -> float:
    """Separa el 15% mas reciente como validacion temporal, entrena la
    cascada solo con el 85% restante, y barre T para minimizar el log-loss
    en la validacion (sec. 3.5)."""
    df_sorted = df_train.sort_values("date")
    corte = int(len(df_sorted) * 0.85)
    tr, val = df_sorted.iloc[:corte], df_sorted.iloc[corte:]

    X_tr, X_val = tr[columnas], val[columnas]
    y_tr_l, y_tr_v = tr["home_score"], tr["away_score"]
    y_val_label = val["ganador_final"].map(LABEL_MAP).to_numpy()

    pesos_l = _pesos_muestra(tr, y_tr_l)
    pesos_v = _pesos_muestra(tr, y_tr_v)

    modelo_l = _nuevo_regresor_goles().fit(X_tr, y_tr_l, sample_weight=pesos_l)
    modelo_v = _nuevo_regresor_goles().fit(X_tr, y_tr_v, sample_weight=pesos_v)

    oof_l, oof_v = _oof_meta_goles(X_tr, y_tr_l, y_tr_v)
    X_tr_meta = X_tr.copy()
    X_tr_meta["pred_goles_l"] = oof_l
    X_tr_meta["pred_goles_v"] = oof_v

    X_val_meta = X_val.copy()
    X_val_meta["pred_goles_l"] = modelo_l.predict(X_val)
    X_val_meta["pred_goles_v"] = modelo_v.predict(X_val)

    y_tr_label = tr["ganador_final"].map(LABEL_MAP).to_numpy()
    tscv = TimeSeriesSplit(n_splits=5)
    clasificador = CalibratedClassifierCV(estimator=_nuevo_clasificador(), method="isotonic", cv=tscv)
    clasificador.fit(X_tr_meta, y_tr_label)

    probs_val = clasificador.predict_proba(X_val_meta)

    mejor_t, mejor_loss = 1.0, log_loss(y_val_label, probs_val, labels=[0, 1, 2])
    for t in np.arange(0.3, 2.55, 0.05):
        p_t = probs_val ** (1 / t)
        p_t = p_t / p_t.sum(axis=1, keepdims=True)
        loss = log_loss(y_val_label, p_t, labels=[0, 1, 2])
        if loss < mejor_loss:
            mejor_loss, mejor_t = loss, t

    print(f"  [validacion temporal] {len(tr)} train / {len(val)} val | "
          f"log-loss calibrado T=1: {log_loss(y_val_label, probs_val, labels=[0,1,2]):.4f} | "
          f"mejor T={mejor_t:.2f} (log-loss={mejor_loss:.4f})")
    return mejor_t


def entrenar_pipeline(df_train: pd.DataFrame) -> dict:
    columnas = obtener_columnas_features(df_train)
    columnas = eliminar_multicolinealidad(df_train, columnas, umbral=0.9)
    print(f"  Variables finales tras eliminar multicolinealidad: {len(columnas)}")

    temperatura = _validar_temperatura(df_train, columnas)

    # --- Reentrenamiento final con el 100% de los datos (sec. 3.4) ---
    X = df_train[columnas]
    y_l, y_v = df_train["home_score"], df_train["away_score"]
    y_label = df_train["ganador_final"].map(LABEL_MAP).to_numpy()

    pesos_l = _pesos_muestra(df_train, y_l)
    pesos_v = _pesos_muestra(df_train, y_v)

    modelo_l = _nuevo_regresor_goles().fit(X, y_l, sample_weight=pesos_l)
    modelo_v = _nuevo_regresor_goles().fit(X, y_v, sample_weight=pesos_v)

    oof_l, oof_v = _oof_meta_goles(X, y_l, y_v)
    X_meta = X.copy()
    X_meta["pred_goles_l"] = oof_l
    X_meta["pred_goles_v"] = oof_v

    tscv = TimeSeriesSplit(n_splits=5)
    clasificador = CalibratedClassifierCV(estimator=_nuevo_clasificador(), method="isotonic", cv=tscv)
    clasificador.fit(X_meta, y_label)

    return {
        "modelo_l": modelo_l,
        "modelo_v": modelo_v,
        "clasificador": clasificador,
        "columnas": columnas,
        "temperatura": temperatura,
    }
