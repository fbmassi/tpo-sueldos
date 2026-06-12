"""
evaluacion_modelos.py
=====================

Entrena y compara modelos de regresión para predecir el salario real de un
profesional tech usando SOLO variables de su perfil (no macro ni derivadas).

TARGET:  salario_real_ars (pesos constantes de may-2026, ya deflactado)
DATASET: data/processed/dataset_final_mercado_laboral.parquet

Modelos comparados (técnicas de la materia):
  1. Regresión lineal simple (solo años de experiencia)  — baseline conceptual
  2. Regresión lineal múltiple                            — baseline real
  3. Regresión polinómica (grado 2 en numéricas)
  4. Árbol de regresión (max_depth=5)
  5. Random Forest (n_estimators=100)

Split principal: aleatorio 80/20 ESTRATIFICADO por fecha_edicion.
  Justificación: el target ya está deflactado (pesos constantes), por lo que el
  efecto inflacionario está neutralizado. El objetivo es estimación TRANSVERSAL
  (dado un perfil, estimar su sueldo de mercado), no forecasting temporal. El
  split aleatorio estratificado garantiza la misma proporción de cada edición
  en train y test, consistente con la metodología estándar.

Validación secundaria: split TEMPORAL (train: 2022-07..2025-01 / test: 2025-07)
  sólo para el mejor modelo, para detectar cambios estructurales de nivel
  salarial entre períodos.

Ejecutar:  python data/evaluacion_modelos.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

sns.set_style("whitegrid")

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = PROC_DIR / "modelos"

DATASET = PROC_DIR / "dataset_final_mercado_laboral.parquet"
TARGET = "salario_real_ars"
RANDOM_STATE = 42
TEST_SIZE = 0.20

TOP_ROLES = 15        # roles más frecuentes; el resto -> "Otro"
TOP_TECHS = 20        # tecnologías con columna binaria propia
MIN_PROVINCIA = 100   # provincias con menos registros -> "Otra"
DPI = 300

COLS_NUM = ["edad", "anos_experiencia_total", "anos_empresa_actual"]
COLS_CAT = ["provincia", "genero", "seniority", "modalidad",
            "tamano_empresa", "rol", "cobra_en_dolares"]

ARCHIVOS: list[str] = []


def guardar(fig, nombre: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / nombre, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    ARCHIVOS.append(nombre)
    print(f"  ✓ {nombre}")


# ----------------------------------------------------------------------------
# Preparación de features (solo perfil)
# ----------------------------------------------------------------------------
def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Devuelve (X, y, fechas, cols_tech).

    Se EXCLUYEN explícitamente: salario_real_usd, canastas_basicas,
    cobertura_cbt (derivadas del propio salario -> data leakage) y es_outlier.
    fecha_edicion se guarda aparte SOLO para estratificar y para la validación
    temporal — no entra como feature.
    """
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    y = df[TARGET]
    fechas = df["fecha_edicion"]

    X = pd.DataFrame(index=df.index)

    # --- numéricas ---
    for c in COLS_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")

    # --- provincia: agrupar las de <MIN_PROVINCIA registros en "Otra" ---
    vc = df["provincia"].value_counts()
    chicas = vc[vc < MIN_PROVINCIA].index
    X["provincia"] = df["provincia"].where(~df["provincia"].isin(chicas), "Otra")

    # --- rol: alta cardinalidad -> top 15 + "Otro" ---
    top_roles = df["rol"].value_counts().head(TOP_ROLES).index
    X["rol"] = df["rol"].where(df["rol"].isin(top_roles), "Otro")

    # --- resto de categóricas directas ---
    X["genero"] = df["genero"].fillna("no especifica")
    X["seniority"] = df["seniority"]
    X["modalidad"] = df["modalidad"]
    X["tamano_empresa"] = df["tamano_empresa"].fillna("No especifica")
    X["cobra_en_dolares"] = df["cobra_en_dolares"].astype(str)

    # --- tecnologías: multi-hot del top 20 ---
    listas = (df["tecnologias"].fillna("")
              .replace("No especifica", "")
              .str.split(",")
              .apply(lambda ts: {t.strip() for t in ts if t.strip()}))
    conteo = pd.Series([t for s in listas for t in s]).value_counts()
    conteo = conteo.drop("ninguno de los anteriores", errors="ignore")
    techs = conteo.head(TOP_TECHS).index.tolist()
    cols_tech = []
    for t in techs:
        col = f"usa_{t.replace(' ', '_').replace('.', '').replace('#', 'sharp')}"
        X[col] = listas.apply(lambda s, tt=t: int(tt in s))
        cols_tech.append(col)

    return X, y, fechas, cols_tech


def hacer_preprocesador(cols_tech: list[str], escalar: bool,
                        poly: bool = False) -> ColumnTransformer:
    """
    Preprocesador por tipo de modelo:
      - lineales: StandardScaler en numéricas (y PolynomialFeatures si poly)
      - árboles: passthrough en numéricas (no necesitan escala)
    Las binarias de tecnología pasan tal cual; las categóricas van a one-hot.
    Al ir dentro de un Pipeline, el fit ocurre SOLO con train (sin leakage).
    """
    if poly:
        num = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                        ("scaler", StandardScaler())])
    elif escalar:
        num = Pipeline([("scaler", StandardScaler())])
    else:
        num = "passthrough"
    return ColumnTransformer([
        ("num", num, COLS_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COLS_CAT),
        ("tech", "passthrough", cols_tech),
    ])


# ----------------------------------------------------------------------------
# Evaluación
# ----------------------------------------------------------------------------
def evaluar(modelo, X_tr, X_te, y_tr, y_te) -> dict:
    """Entrena y devuelve métricas en train y test + ratio de overfitting."""
    modelo.fit(X_tr, y_tr)
    p_tr, p_te = modelo.predict(X_tr), modelo.predict(X_te)
    rmse_tr = float(np.sqrt(mean_squared_error(y_tr, p_tr)))
    rmse_te = float(np.sqrt(mean_squared_error(y_te, p_te)))
    return {
        "RMSE_train": rmse_tr,
        "RMSE_test": rmse_te,
        "MAE_train": mean_absolute_error(y_tr, p_tr),
        "MAE_test": mean_absolute_error(y_te, p_te),
        "R2_train": r2_score(y_tr, p_tr),
        "R2_test": r2_score(y_te, p_te),
        "Overfit_ratio": rmse_te / rmse_tr,
    }


def nivel_overfit(ratio: float) -> str:
    if ratio < 1.1:
        return "sin overfitting"
    if ratio < 1.3:
        return "leve"
    if ratio < 1.5:
        return "moderado"
    return "SEVERO"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    if not DATASET.exists():
        print(f"✗ No existe {DATASET}. Corré primero limpiar_y_unificar_datos.py")
        return

    df = pd.read_parquet(DATASET)
    print(f"Dataset: {len(df):,} filas")
    X, y, fechas, cols_tech = preparar_features(df)
    print(f"Features: {X.shape[1]} columnas "
          f"({len(COLS_NUM)} numéricas, {len(COLS_CAT)} categóricas, "
          f"{len(cols_tech)} tecnologías multi-hot)")

    # ------------------------------------------------------------------
    # SPLIT PRINCIPAL: aleatorio 80/20 estratificado por edición.
    # El target está deflactado -> el split aleatorio es el correcto para
    # estimación transversal (no es un problema de forecasting).
    # ------------------------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=fechas)
    print(f"Split aleatorio estratificado por edición: "
          f"train={len(X_tr):,} / test={len(X_te):,}\n")

    # ------------------------------------------------------------------
    # Definición de los 5 modelos (cada uno con su preprocesador adecuado)
    # ------------------------------------------------------------------
    modelos: dict[str, Pipeline] = {
        # 1. baseline conceptual: SOLO años de experiencia
        "1. Lineal simple (solo exp.)": Pipeline([
            ("pre", ColumnTransformer(
                [("num", StandardScaler(), ["anos_experiencia_total"])])),
            ("reg", LinearRegression()),
        ]),
        # 2. baseline real: todas las features, lineal
        "2. Lineal múltiple": Pipeline([
            ("pre", hacer_preprocesador(cols_tech, escalar=True)),
            ("reg", LinearRegression()),
        ]),
        # 3. no linealidad de la experiencia: polinomio g2 SOLO en numéricas
        "3. Polinómica (g=2)": Pipeline([
            ("pre", hacer_preprocesador(cols_tech, escalar=True, poly=True)),
            ("reg", LinearRegression()),
        ]),
        # 4. árbol podado (los árboles no necesitan escalado)
        "4. Árbol (max_depth=5)": Pipeline([
            ("pre", hacer_preprocesador(cols_tech, escalar=False)),
            ("reg", DecisionTreeRegressor(max_depth=5,
                                          random_state=RANDOM_STATE)),
        ]),
        # 5. ensamble
        "5. Random Forest (100)": Pipeline([
            ("pre", hacer_preprocesador(cols_tech, escalar=False)),
            ("reg", RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth=12, min_samples_leaf=5,
                                          random_state=RANDOM_STATE)),
        ]),
    }

    # ------------------------------------------------------------------
    # Entrenar y evaluar
    # ------------------------------------------------------------------
    resultados: dict[str, dict] = {}
    for nombre, pipe in modelos.items():
        try:
            resultados[nombre] = evaluar(pipe, X_tr, X_te, y_tr, y_te)
            r = resultados[nombre]
            print(f"{nombre:32s} R²test={r['R2_test']:.3f}  "
                  f"RMSEtest=${r['RMSE_test']/1e6:.2f}M  "
                  f"overfit={r['Overfit_ratio']:.2f} ({nivel_overfit(r['Overfit_ratio'])})")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {nombre} FALLÓ: {exc}")

    tabla = pd.DataFrame(resultados).T
    tabla.index.name = "Modelo"

    # mejor modelo = mayor R² en test
    mejor_nombre = tabla["R2_test"].idxmax()
    mejor = modelos[mejor_nombre]
    print(f"\n🏆 MEJOR MODELO: {mejor_nombre} "
          f"(R²test={tabla.loc[mejor_nombre,'R2_test']:.3f})")

    # ------------------------------------------------------------------
    # VALIDACIÓN SECUNDARIA: split temporal con el mejor modelo
    # train: 2022-07..2025-01  /  test: 2025-07 (última edición)
    # ------------------------------------------------------------------
    corte = fechas.max()
    m_tr, m_te = fechas < corte, fechas == corte
    pipe_temporal = clone(mejor)
    res_temp = evaluar(pipe_temporal, X[m_tr], X[m_te], y[m_tr], y[m_te])
    print(f"\nValidación temporal (train<{str(corte)[:10]} / test={str(corte)[:10]}): "
          f"R²={res_temp['R2_test']:.3f}  RMSE=${res_temp['RMSE_test']/1e6:.2f}M")

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------
    print("\nGenerando gráficos…")
    nombres = list(tabla.index)
    xpos = np.arange(len(nombres))

    # 1 — RMSE train vs test
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(xpos - 0.2, tabla["RMSE_train"] / 1e6, 0.4, label="Train",
           color="steelblue")
    ax.bar(xpos + 0.2, tabla["RMSE_test"] / 1e6, 0.4, label="Test",
           color="darkorange")
    ax.set_xticks(xpos)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in nombres], fontsize=8)
    ax.set_ylabel("RMSE (millones de $ de may-2026)")
    ax.set_title("Comparación de modelos — RMSE train vs test\n"
                 "(barras muy distintas = overfitting)")
    ax.legend(); ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_comparacion_rmse.png")

    # 2 — R² test
    fig, ax = plt.subplots(figsize=(12, 6))
    colores = ["seagreen" if n == mejor_nombre else "steelblue" for n in nombres]
    ax.bar(xpos, tabla["R2_test"], color=colores)
    ax.axhline(tabla["R2_test"].max(), color="red", ls="--", lw=1,
               label=f"Mejor: {tabla['R2_test'].max():.3f}")
    ax.set_xticks(xpos)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in nombres], fontsize=8)
    ax.set_ylabel("R² en test")
    ax.set_title("Comparación de modelos — R² en test")
    ax.legend(); ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_comparacion_r2.png")

    # 3 — predicho vs real (mejor modelo)
    pred_te = mejor.predict(X_te)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_te / 1e6, pred_te / 1e6, alpha=0.25, s=12, color="steelblue",
               edgecolors="none")
    lim = [0, np.percentile(y_te, 99.5) / 1e6]
    ax.plot(lim, lim, color="red", lw=2, label="Predicción perfecta (y=x)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Salario real (millones $)"); ax.set_ylabel("Salario predicho (millones $)")
    ax.set_title(f"Predicho vs real — {mejor_nombre} (test)")
    ax.legend(); ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_predicho_vs_real.png")

    # 4 — residuos vs predicho
    residuos = y_te - pred_te
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pred_te / 1e6, residuos / 1e6, alpha=0.25, s=12,
               color="mediumpurple", edgecolors="none")
    ax.axhline(0, color="red", lw=2)
    ax.set_xlabel("Salario predicho (millones $)")
    ax.set_ylabel("Residuo = real − predicho (millones $)")
    ax.set_title(f"Análisis de residuos — {mejor_nombre} (test)")
    ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_residuos.png")

    # 5 — importancia de features (Random Forest)
    rf_pipe = modelos["5. Random Forest (100)"]
    rf = rf_pipe.named_steps["reg"]
    nombres_feat = rf_pipe.named_steps["pre"].get_feature_names_out()
    imp = (pd.Series(rf.feature_importances_, index=nombres_feat)
           .sort_values().tail(20))
    imp.index = [i.split("__")[-1] for i in imp.index]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp.index, imp.values, color="seagreen")
    ax.set_title("Top 20 features más importantes — Random Forest")
    ax.set_xlabel("Importancia (reducción de impureza)")
    ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_importancia_features.png")

    # 6 — distribución de errores (mejor modelo)
    fig, ax = plt.subplots(figsize=(10, 6))
    res_clip = residuos[residuos.between(residuos.quantile(0.005),
                                         residuos.quantile(0.995))]
    ax.hist(res_clip / 1e6, bins=60, color="indianred", edgecolor="white")
    ax.axvline(0, color="black", ls="--", lw=1.5)
    ax.axvline(residuos.median() / 1e6, color="blue", ls=":",
               label=f"Mediana: ${residuos.median()/1e6:.2f}M")
    ax.set_xlabel("Residuo (millones $)"); ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de errores — {mejor_nombre} (test)")
    ax.legend(); ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_distribucion_errores.png")

    # 7 — split aleatorio vs temporal (mejor modelo)
    res_rand = resultados[mejor_nombre]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].bar(["Aleatorio\nestratificado", "Temporal\n(test=2025.2)"],
               [res_rand["RMSE_test"] / 1e6, res_temp["RMSE_test"] / 1e6],
               color=["steelblue", "darkorange"])
    axs[0].set_ylabel("RMSE test (millones $)"); axs[0].set_title("RMSE")
    axs[0].grid(True, alpha=0.3)
    axs[1].bar(["Aleatorio\nestratificado", "Temporal\n(test=2025.2)"],
               [res_rand["R2_test"], res_temp["R2_test"]],
               color=["steelblue", "darkorange"])
    axs[1].set_ylabel("R² test"); axs[1].set_title("R²")
    axs[1].grid(True, alpha=0.3)
    fig.suptitle(f"Split aleatorio vs temporal — {mejor_nombre}\n"
                 "(peor en temporal = cambio estructural de nivel salarial entre períodos)")
    guardar(fig, "modelos_aleatorio_vs_temporal.png")

    # ------------------------------------------------------------------
    # Salida final
    # ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("TABLA COMPARATIVA (split aleatorio 80/20 estratificado por edición)")
    print("=" * 96)
    tt = tabla.copy()
    for c in ["RMSE_train", "RMSE_test", "MAE_train", "MAE_test"]:
        tt[c] = (tt[c] / 1e6).map(lambda v: f"${v:.2f}M")
    for c in ["R2_train", "R2_test"]:
        tt[c] = tt[c].map(lambda v: f"{v:.3f}")
    tt["Overfit_ratio"] = tabla["Overfit_ratio"].map(
        lambda v: f"{v:.2f} ({nivel_overfit(v)})")
    print(tt.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla.round(4).to_csv(PROC_DIR / "resultados_modelos.csv")
    print(f"\nTabla guardada en {PROC_DIR / 'resultados_modelos.csv'}")

    print(f"\n🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   R² test = {tabla.loc[mejor_nombre, 'R2_test']:.3f} | "
          f"MAE test = ${tabla.loc[mejor_nombre, 'MAE_test']/1e6:.2f}M | "
          f"overfitting: {nivel_overfit(tabla.loc[mejor_nombre, 'Overfit_ratio'])}")

    print("\nTOP 10 FEATURES (Random Forest):")
    for nom, val in imp.tail(10)[::-1].items():
        print(f"   {nom:35s} {val:.3f}")

    print("\nALEATORIO vs TEMPORAL (mejor modelo):")
    print(f"   Aleatorio: R²={res_rand['R2_test']:.3f}  RMSE=${res_rand['RMSE_test']/1e6:.2f}M")
    print(f"   Temporal : R²={res_temp['R2_test']:.3f}  RMSE=${res_temp['RMSE_test']/1e6:.2f}M")
    delta = res_rand["R2_test"] - res_temp["R2_test"]
    if delta > 0.03:
        print("   ⚠ El modelo rinde PEOR al predecir la edición 2025.2 sin haberla visto:")
        print("     hay un cambio estructural del nivel salarial real entre períodos")
        print("     (el 'precio de mercado' de un mismo perfil se movió).")
    else:
        print("   ✓ Rendimiento similar: el nivel salarial real se mantuvo estable;")
        print("     un modelo entrenado con el pasado generaliza bien al presente.")

    print(f"\nGráficos generados en {OUT_DIR}/:")
    for a in ARCHIVOS:
        print(f"   - {a}")


if __name__ == "__main__":
    main()
