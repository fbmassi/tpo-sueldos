"""
entrenar_modelo.py
==================

Entrena el modelo predictivo de sueldos del TPO y guarda el artefacto que
consume la app de Streamlit.

Diseño (decisiones clave, ver evaluación del TPO):
  - TARGET único: salario_bruto_ars (en escala log1p, porque el sueldo está
    muy sesgado a la derecha).
  - FEATURES = SOLO el perfil del profesional. NO se usan columnas derivadas
    (sueldo_real, canastas, big_macs, ratio_vs_ripte, usd...) porque son el
    target multiplicado por una constante -> data leakage.
  - Las otras dimensiones (USD MEP, canastas, big macs, ratio vs RIPTE) se
    DERIVAN de la predicción usando los valores macro de referencia, no se
    predicen ni se usan como entrada.
  - Rango de predicción REAL: 3 modelos Gradient Boosting con pérdida por
    cuantiles (P10 / P50 / P90). El "rango" del predictor sale del modelo.

Produce:
  data/processed/modelo_sueldo.pkl       (artefacto para la app)
  data/processed/modelo_metricas.txt     (evaluación)

Ejecutar:
  python data/entrenar_modelo.py
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
PROC_DIR = DATA_DIR / "processed"
DATASET = PROC_DIR / "dataset_final_mercado_laboral.parquet"
MODELO_OUT = PROC_DIR / "modelo_sueldo.pkl"
METRICAS_OUT = PROC_DIR / "modelo_metricas.txt"

TARGET = "salario_bruto_ars"
IPC_BASE = 4744.45            # IPC enero 2024 (igual que el ETL)
ROLES_TOP_N = 20              # roles más frecuentes; el resto -> "Otro"
TECHS_TOP_N = 15             # tecnologías con flag binario propio
TEST_SIZE = 0.2
RANDOM_STATE = 42
CUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

CAT_COLS = ["rol", "seniority", "provincia", "modalidad"]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("modelo")


# ----------------------------------------------------------------------------
# Feature engineering (compartido con la app)
# ----------------------------------------------------------------------------
def _set_techs(texto: str) -> set[str]:
    if not isinstance(texto, str):
        return set()
    return {t.strip().lower() for t in texto.split(",")
            if t.strip() and t.strip().lower() != "no especifica"}


def construir_features(df: pd.DataFrame, tech_list: list[str],
                       roles_top: list[str]) -> pd.DataFrame:
    """
    Convierte filas crudas (perfil) en la matriz de features del modelo.

    Esta función la usan TANTO el entrenamiento COMO la app, para garantizar
    que un perfil ingresado a mano se transforme exactamente igual.
    """
    X = pd.DataFrame(index=df.index)
    rol = df["rol"].astype("object").fillna("Otro")
    X["rol"] = rol.where(rol.isin(roles_top), "Otro")
    X["seniority"] = df["seniority"].astype("object").fillna("semi-senior")
    X["provincia"] = df["provincia"].astype("object").fillna("Otro")
    X["modalidad"] = df["modalidad"].astype("object").fillna("100% remoto")
    X["anos_experiencia_total"] = pd.to_numeric(
        df["anos_experiencia_total"], errors="coerce")

    sets = df["tecnologias"].apply(_set_techs)
    X["lenguajes_count"] = sets.apply(len)
    for tech in tech_list:
        X[f"tech_{tech}"] = sets.apply(lambda s, t=tech: int(t in s))
    return X


def _num_cols(tech_list: list[str]) -> list[str]:
    return (["anos_experiencia_total", "lenguajes_count"] +
            [f"tech_{t}" for t in tech_list])


# ----------------------------------------------------------------------------
# Predicción (la usa la app)
# ----------------------------------------------------------------------------
def predecir(artefacto: dict, perfil: dict) -> dict:
    """
    Dado el artefacto cargado y un perfil (dict con rol, seniority,
    anos_experiencia_total, tecnologias, provincia, modalidad), devuelve el
    sueldo estimado en sus tres dimensiones + rango P10/P90.
    """
    fila = pd.DataFrame([{
        "rol": perfil.get("rol"),
        "seniority": perfil.get("seniority"),
        "provincia": perfil.get("provincia"),
        "modalidad": perfil.get("modalidad"),
        "anos_experiencia_total": perfil.get("anos_experiencia_total", 0),
        "tecnologias": perfil.get("tecnologias", ""),
    }])
    X = construir_features(fila, artefacto["tech_list"], artefacto["roles_top"])

    pred = {}
    for nombre, modelo in artefacto["modelos"].items():
        pred[nombre] = float(np.expm1(modelo.predict(X)[0]))

    macro = artefacto["macro_ref"]

    def _dims(sal_nominal: float) -> dict:
        real = sal_nominal / (macro["ipc"] / macro["ipc_base"])
        return {
            "salario_bruto_ars": sal_nominal,
            "sueldo_real_ars": real,
            "sueldo_usd_mep": sal_nominal / macro["dolar_mep"],
            "canastas_basicas_cubiertas": sal_nominal / macro["cbt"],
            "big_macs_mensuales": sal_nominal / macro["precio_bigmac_ars"],
            # nominal vs nominal (mismo período): comparar el bruto con RIPTE,
            # no el sueldo deflactado, para no mezclar bases temporales.
            "ratio_vs_ripte": sal_nominal / macro["ripte"],
        }

    return {
        "p10": _dims(pred["p10"]),
        "p50": _dims(pred["p50"]),
        "p90": _dims(pred["p90"]),
        "macro_ref": macro,
    }


# ----------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------
def main() -> None:
    if not DATASET.exists():
        log.error("No existe %s. Corré primero limpiar_y_unificar_datos.py", DATASET)
        return

    df = pd.read_parquet(DATASET)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()

    # Multi-edición: el modelo predice el sueldo ACTUAL (nominal), así que
    # entrena SÓLO con la última edición. Mezclar ediciones con 10x de inflación
    # entre medio haría que el modelo aprenda "época", no "perfil". Las ediciones
    # anteriores quedan en el dataset para el análisis temporal, no para el modelo.
    if "fecha_edicion" in df.columns and df["fecha_edicion"].nunique() > 1:
        ult = df["fecha_edicion"].max()
        df = df[df["fecha_edicion"] == ult].copy()
        log.info("Multi-edición detectada: entreno sólo con %s", str(ult)[:10])
    log.info("Dataset de entrenamiento: %d filas", len(df))

    # --- definir top roles y top techs desde los datos ---
    roles_top = df["rol"].value_counts().head(ROLES_TOP_N).index.tolist()
    cont = Counter()
    for s in df["tecnologias"].apply(_set_techs):
        cont.update(s)
    tech_list = [t for t, _ in cont.most_common(TECHS_TOP_N)]
    log.info("Roles top: %d | Techs flag: %s", len(roles_top), tech_list)

    # --- features / target ---
    X = construir_features(df, tech_list, roles_top)
    y = np.log1p(df[TARGET].values)

    num_cols = _num_cols(tech_list)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # --- entrenar 3 modelos de cuantiles ---
    modelos = {}
    for nombre, alpha in CUANTILES.items():
        pipe = Pipeline([
            ("pre", pre),
            ("gbr", GradientBoostingRegressor(
                loss="quantile", alpha=alpha,
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.9, random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_tr, y_tr)
        modelos[nombre] = pipe
        log.info("Modelo %s (cuantil %.2f) entrenado", nombre, alpha)

    # --- evaluación (modelo mediana P50) ---
    y_te_real = np.expm1(y_te)
    pred50 = np.expm1(modelos["p50"].predict(X_te))
    pred10 = np.expm1(modelos["p10"].predict(X_te))
    pred90 = np.expm1(modelos["p90"].predict(X_te))

    r2 = r2_score(y_te_real, pred50)
    mae = mean_absolute_error(y_te_real, pred50)
    rmse = float(np.sqrt(mean_squared_error(y_te_real, pred50)))
    mape = float(np.mean(np.abs((y_te_real - pred50) / y_te_real)) * 100)
    cobertura = float(np.mean((y_te_real >= pred10) & (y_te_real <= pred90)) * 100)

    # baseline: mediana por (rol, seniority)
    base_map = df.groupby(["rol", "seniority"])[TARGET].median()
    global_med = df[TARGET].median()
    df_te = df.loc[X_te.index]
    base_pred = [base_map.get((r, s), global_med)
                 for r, s in zip(df_te["rol"], df_te["seniority"])]
    mae_base = mean_absolute_error(y_te_real, base_pred)

    # --- feature importance (sobre el modelo mediana) ---
    perm = permutation_importance(
        modelos["p50"], X_te, y_te, n_repeats=5,
        random_state=RANDOM_STATE, scoring="r2")
    imp = (pd.Series(perm.importances_mean, index=X.columns)
           .sort_values(ascending=False).head(15))

    # --- valores macro de referencia (para derivar dimensiones en la app) ---
    # Desde el ETL el macro vive en contexto_edicion.parquet (normalizado).
    # Como respaldo, si alguna columna siguiera en el dataset, se usa esa.
    contexto = None
    ctx_path = PROC_DIR / "contexto_edicion.parquet"
    if ctx_path.exists():
        ctx_df = pd.read_parquet(ctx_path)
        if "fecha_edicion" in ctx_df.columns:
            ctx_df = ctx_df.sort_values("fecha_edicion")
        contexto = ctx_df.iloc[-1]   # última edición (la que entrena el modelo)

    def _ref(col):
        if contexto is not None and col in contexto.index and pd.notna(contexto[col]):
            return float(contexto[col])
        if col in df.columns:
            s = df[col].dropna()
            return float(s.iloc[0]) if len(s) else np.nan
        return np.nan

    macro_ref = {
        "ipc": _ref("ipc"),
        "ipc_base": IPC_BASE,
        "dolar_mep": _ref("dolar_mep"),
        "cbt": _ref("cbt"),
        "precio_bigmac_ars": _ref("precio_bigmac_ars"),
        "ripte": _ref("ripte"),
        "fecha_edicion": str(df["fecha_edicion"].dropna().iloc[0])[:10],
    }

    opciones = {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_COLS}
    opciones["rol"] = roles_top + ["Otro"]

    metricas = {
        "n_train": len(X_tr), "n_test": len(X_te),
        "r2": r2, "mae": mae, "rmse": rmse, "mape_pct": mape,
        "cobertura_intervalo_p10_p90_pct": cobertura,
        "mae_baseline_mediana": mae_base,
        "mejora_vs_baseline_pct": (1 - mae / mae_base) * 100,
    }

    # --- guardar artefacto ---
    artefacto = {
        "modelos": modelos,
        "tech_list": tech_list,
        "roles_top": roles_top,
        "cat_cols": CAT_COLS,
        "num_cols": num_cols,
        "opciones": opciones,
        "macro_ref": macro_ref,
        "metricas": metricas,
        "feature_importance": imp.to_dict(),
        "target": TARGET,
        "entrenado": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(artefacto, MODELO_OUT)
    log.info("Artefacto guardado -> %s", MODELO_OUT)

    # --- reporte ---
    lineas = [
        "REPORTE DE MODELO — Predictor de sueldos tech AR",
        f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 60,
        f"Filas train/test: {len(X_tr)} / {len(X_te)}",
        f"Target: {TARGET} (escala log1p)",
        f"Features categóricas: {CAT_COLS}",
        f"Features numéricas: {len(num_cols)} (exp + count + {TECHS_TOP_N} techs)",
        "",
        "-- MÉTRICAS (modelo mediana P50, escala pesos) --",
        f"R²:    {r2:.3f}",
        f"MAE:   $ {mae:,.0f}",
        f"RMSE:  $ {rmse:,.0f}",
        f"MAPE:  {mape:.1f}%",
        f"Cobertura intervalo P10–P90: {cobertura:.1f}%  (ideal ~80%)",
        "",
        f"Baseline (mediana por rol+seniority) MAE: $ {mae_base:,.0f}",
        f"Mejora del modelo vs baseline: {metricas['mejora_vs_baseline_pct']:.1f}%",
        "",
        "-- TOP 15 FEATURES (permutation importance) --",
    ]
    for nombre, val in imp.items():
        lineas.append(f"  {nombre:30s} {val:.4f}")
    lineas += ["", "-- VALORES MACRO DE REFERENCIA --"]
    for k, v in macro_ref.items():
        lineas.append(f"  {k:20s} {v}")
    METRICAS_OUT.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # --- resumen consola ---
    print("\n" + "=" * 60)
    print("MODELO ENTRENADO")
    print("=" * 60)
    print(f"  R²={r2:.3f} | MAE=${mae:,.0f} | MAPE={mape:.1f}%")
    print(f"  Cobertura intervalo P10–P90: {cobertura:.1f}% (ideal ~80%)")
    print(f"  Mejora vs baseline: {metricas['mejora_vs_baseline_pct']:.1f}%")
    print(f"  -> {MODELO_OUT}")
    print(f"  -> {METRICAS_OUT}")
    print("\n  Top 5 features:")
    for nombre, val in list(imp.items())[:5]:
        print(f"    {nombre:28s} {val:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
