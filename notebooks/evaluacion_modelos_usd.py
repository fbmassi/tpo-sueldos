"""
evaluacion_modelos_usd.py
=========================

Igual que evaluacion_modelos.py pero con el TARGET en DÓLARES REALES
(salario_real_usd, USD constantes de may-2026) en lugar de pesos.

NOTA METODOLÓGICA: salario_real_usd = salario_real_ars / MEP_base (una constante
≈ 1408,57). Como el target sólo cambia de escala por una constante, el R², las
importancias de features y los ratios de overfitting son IDÉNTICOS a la versión
en pesos; lo único que cambia es la UNIDAD del error (RMSE/MAE en USD). Sirve
para reportar el desempeño en dólares, más interpretable internacionalmente.

TARGET:  salario_real_usd (USD constantes de may-2026)
DATASET: data/processed/dataset_final_mercado_laboral.parquet

Split principal: TEMPORAL (train: ediciones previas / test: última encuesta 2025.2).
Validación secundaria: aleatorio 80/20 estratificado (contraste).

Ejecutar:  python notebooks/evaluacion_modelos_usd.py
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
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = PROC_DIR / "modelos_usd"          # carpeta separada de la de pesos

DATASET = PROC_DIR / "dataset_final_mercado_laboral.parquet"
TARGET = "salario_real_usd"                  # <-- dólares reales
RANDOM_STATE = 42
TEST_SIZE = 0.20

TOP_ROLES = 15
TOP_TECHS = 20
MIN_PROVINCIA = 100
DPI = 300

# Se excluye 'seniority' (redundante con anos_experiencia_total).
COLS_NUM = ["edad", "anos_experiencia_total", "anos_empresa_actual"]
COLS_CAT = ["provincia", "genero", "modalidad",
            "tamano_empresa", "rol", "cobra_en_dolares"]

ARCHIVOS: list[str] = []


def guardar(fig, nombre: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / nombre, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    ARCHIVOS.append(nombre)
    print(f"  ✓ {nombre}")


def usd(v: float) -> str:
    return f"USD {v:,.0f}"


# ----------------------------------------------------------------------------
# Preparación de features (solo perfil)
# ----------------------------------------------------------------------------
def preparar_features(df: pd.DataFrame):
    """
    Devuelve (X, y, fechas, cols_tech). Se excluyen del set de features las
    columnas derivadas del salario (salario_real_ars, canastas_basicas,
    cobertura_cbt) y es_outlier. fecha_edicion no es feature: sólo se usa para
    el split temporal y la estratificación.
    """
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    y = df[TARGET]
    fechas = df["fecha_edicion"]

    X = pd.DataFrame(index=df.index)
    for c in COLS_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")

    vc = df["provincia"].value_counts()
    chicas = vc[vc < MIN_PROVINCIA].index
    X["provincia"] = df["provincia"].where(~df["provincia"].isin(chicas), "Otra")

    top_roles = df["rol"].value_counts().head(TOP_ROLES).index
    X["rol"] = df["rol"].where(df["rol"].isin(top_roles), "Otro")

    X["genero"] = df["genero"].fillna("otro / no especifica")
    X["modalidad"] = df["modalidad"]
    X["tamano_empresa"] = df["tamano_empresa"].fillna("No especifica")
    X["cobra_en_dolares"] = df["cobra_en_dolares"].astype(str)

    # Tecnologías EXCLUIDAS del modelo (ruido/extrapolación; premium = proxy del
    # contexto). El análisis descriptivo se conserva en el EDA.
    cols_tech: list[str] = []
    return X, y, fechas, cols_tech


def hacer_preprocesador(cols_tech, escalar: bool, poly: bool = False):
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
    print(f"Dataset: {len(df):,} filas | TARGET = {TARGET} (dólares reales)")
    X, y, fechas, cols_tech = preparar_features(df)
    print(f"Features: {X.shape[1]} columnas "
          f"({len(COLS_NUM)} numéricas, {len(COLS_CAT)} categóricas, "
          f"{len(cols_tech)} tecnologías multi-hot)")

    # SPLIT PRINCIPAL: TEMPORAL (train: previas / test: 2025.2)
    corte = fechas.max()
    m_tr, m_te = fechas < corte, fechas == corte
    X_tr, X_te = X[m_tr], X[m_te]
    y_tr, y_te = y[m_tr], y[m_te]
    print(f"Split TEMPORAL: train (<{str(corte)[:10]})={len(X_tr):,} / "
          f"test (={str(corte)[:10]})={len(X_te):,}\n")

    modelos: dict[str, Pipeline] = {
        "1. Lineal simple (solo exp.)": Pipeline([
            ("pre", ColumnTransformer(
                [("num", StandardScaler(), ["anos_experiencia_total"])])),
            ("reg", LinearRegression()),
        ]),
        "2. Lineal múltiple": Pipeline([
            ("pre", hacer_preprocesador(cols_tech, escalar=True)),
            ("reg", LinearRegression()),
        ]),
        "3. Polinómica (g=2)": Pipeline([
            ("pre", hacer_preprocesador(cols_tech, escalar=True, poly=True)),
            ("reg", LinearRegression()),
        ]),
        "4. Árbol (max_depth=5)": Pipeline([
            ("pre", hacer_preprocesador(cols_tech, escalar=False)),
            ("reg", DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
        ]),
        "5. Random Forest (100)": Pipeline([
            ("pre", hacer_preprocesador(cols_tech, escalar=False)),
            ("reg", RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth=12,
                                          min_samples_leaf=5, random_state=RANDOM_STATE)),
        ]),
    }

    resultados: dict[str, dict] = {}
    for nombre, pipe in modelos.items():
        try:
            resultados[nombre] = evaluar(pipe, X_tr, X_te, y_tr, y_te)
            r = resultados[nombre]
            print(f"{nombre:32s} R²test={r['R2_test']:.3f}  "
                  f"RMSEtest={usd(r['RMSE_test'])}  "
                  f"overfit={r['Overfit_ratio']:.2f} ({nivel_overfit(r['Overfit_ratio'])})")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {nombre} FALLÓ: {exc}")

    tabla = pd.DataFrame(resultados).T
    tabla.index.name = "Modelo"
    mejor_nombre = tabla["R2_test"].idxmax()
    mejor = modelos[mejor_nombre]
    print(f"\n🏆 MEJOR MODELO: {mejor_nombre} (R²test={tabla.loc[mejor_nombre,'R2_test']:.3f})")

    # VALIDACIÓN SECUNDARIA: aleatorio estratificado (contraste)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=fechas)
    res_rand = evaluar(clone(mejor), X_tr_r, X_te_r, y_tr_r, y_te_r)
    print(f"\nValidación aleatoria estratificada (contraste): "
          f"R²={res_rand['R2_test']:.3f}  RMSE={usd(res_rand['RMSE_test'])}")

    # ------------------------------------------------------------------
    # Gráficos (todo en USD)
    # ------------------------------------------------------------------
    print("\nGenerando gráficos…")
    nombres = list(tabla.index)
    xpos = np.arange(len(nombres))

    # 1 — RMSE train vs test
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(xpos - 0.2, tabla["RMSE_train"], 0.4, label="Train", color="steelblue")
    ax.bar(xpos + 0.2, tabla["RMSE_test"], 0.4, label="Test", color="darkorange")
    ax.set_xticks(xpos)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in nombres], fontsize=8)
    ax.set_ylabel("RMSE (USD reales de may-2026)")
    ax.set_title("Comparación de modelos — RMSE train vs test (USD)\n"
                 "(barras muy distintas = overfitting)")
    ax.legend(); ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_usd_comparacion_rmse.png")

    # 2 — R² test
    fig, ax = plt.subplots(figsize=(12, 6))
    colores = ["seagreen" if n == mejor_nombre else "steelblue" for n in nombres]
    ax.bar(xpos, tabla["R2_test"], color=colores)
    ax.axhline(tabla["R2_test"].max(), color="red", ls="--", lw=1,
               label=f"Mejor: {tabla['R2_test'].max():.3f}")
    ax.set_xticks(xpos)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in nombres], fontsize=8)
    ax.set_ylabel("R² en test")
    ax.set_title("Comparación de modelos — R² en test (target en USD)")
    ax.legend(); ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_usd_comparacion_r2.png")

    # 3 — predicho vs real
    pred_te = mejor.predict(X_te)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_te, pred_te, alpha=0.25, s=12, color="steelblue", edgecolors="none")
    lim = [0, np.percentile(y_te, 99.5)]
    ax.plot(lim, lim, color="red", lw=2, label="Predicción perfecta (y=x)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Salario real (USD)"); ax.set_ylabel("Salario predicho (USD)")
    ax.set_title(f"Predicho vs real — {mejor_nombre} (test, USD)")
    ax.legend(); ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_usd_predicho_vs_real.png")

    # 4 — residuos vs predicho
    residuos = y_te - pred_te
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pred_te, residuos, alpha=0.25, s=12, color="mediumpurple", edgecolors="none")
    ax.axhline(0, color="red", lw=2)
    ax.set_xlabel("Salario predicho (USD)")
    ax.set_ylabel("Residuo = real − predicho (USD)")
    ax.set_title(f"Análisis de residuos — {mejor_nombre} (test, USD)")
    ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_usd_residuos.png")

    # 5 — importancia de features (Random Forest)
    rf_pipe = modelos["5. Random Forest (100)"]
    rf = rf_pipe.named_steps["reg"]
    nombres_feat = rf_pipe.named_steps["pre"].get_feature_names_out()
    imp = (pd.Series(rf.feature_importances_, index=nombres_feat)
           .sort_values().tail(20))
    imp.index = [i.split("__")[-1] for i in imp.index]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp.index, imp.values, color="seagreen")
    ax.set_title("Top 20 features más importantes — Random Forest (target USD)")
    ax.set_xlabel("Importancia (reducción de impureza)")
    ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_usd_importancia_features.png")

    # 6 — distribución de errores
    fig, ax = plt.subplots(figsize=(10, 6))
    res_clip = residuos[residuos.between(residuos.quantile(0.005), residuos.quantile(0.995))]
    ax.hist(res_clip, bins=60, color="indianred", edgecolor="white")
    ax.axvline(0, color="black", ls="--", lw=1.5)
    ax.axvline(residuos.median(), color="blue", ls=":",
               label=f"Mediana: {usd(residuos.median())}")
    ax.set_xlabel("Residuo (USD)"); ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de errores — {mejor_nombre} (test, USD)")
    ax.legend(); ax.grid(True, alpha=0.3)
    guardar(fig, "modelos_usd_distribucion_errores.png")

    # 7 — split temporal (principal) vs aleatorio (contraste)
    res_temp = resultados[mejor_nombre]
    etiquetas = ["Temporal\n(principal, test=2025.2)", "Aleatorio\n(contraste)"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].bar(etiquetas, [res_temp["RMSE_test"], res_rand["RMSE_test"]],
               color=["darkorange", "steelblue"])
    axs[0].set_ylabel("RMSE test (USD)"); axs[0].set_title("RMSE")
    axs[0].grid(True, alpha=0.3)
    axs[1].bar(etiquetas, [res_temp["R2_test"], res_rand["R2_test"]],
               color=["darkorange", "steelblue"])
    axs[1].set_ylabel("R² test"); axs[1].set_title("R²")
    axs[1].grid(True, alpha=0.3)
    fig.suptitle(f"Split temporal (principal) vs aleatorio — {mejor_nombre} (USD)\n"
                 "(peor en temporal = cambio estructural de nivel salarial entre períodos)")
    guardar(fig, "modelos_usd_aleatorio_vs_temporal.png")

    # ------------------------------------------------------------------
    # Salida final
    # ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("TABLA COMPARATIVA (target USD — split TEMPORAL: train previas / test 2025.2)")
    print("=" * 96)
    tt = tabla.copy()
    for c in ["RMSE_train", "RMSE_test", "MAE_train", "MAE_test"]:
        tt[c] = tt[c].map(usd)
    for c in ["R2_train", "R2_test"]:
        tt[c] = tt[c].map(lambda v: f"{v:.3f}")
    tt["Overfit_ratio"] = tabla["Overfit_ratio"].map(
        lambda v: f"{v:.2f} ({nivel_overfit(v)})")
    print(tt.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla.round(4).to_csv(PROC_DIR / "resultados_modelos_usd.csv")
    print(f"\nTabla guardada en {PROC_DIR / 'resultados_modelos_usd.csv'}")

    print(f"\n🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   R² test = {tabla.loc[mejor_nombre, 'R2_test']:.3f} | "
          f"MAE test = {usd(tabla.loc[mejor_nombre, 'MAE_test'])} | "
          f"overfitting: {nivel_overfit(tabla.loc[mejor_nombre, 'Overfit_ratio'])}")

    print("\nTOP 10 FEATURES (Random Forest):")
    for nom, val in imp.tail(10)[::-1].items():
        print(f"   {nom:35s} {val:.3f}")

    print("\nTEMPORAL (principal) vs ALEATORIO (contraste) — mejor modelo:")
    print(f"   Temporal : R²={res_temp['R2_test']:.3f}  RMSE={usd(res_temp['RMSE_test'])}")
    print(f"   Aleatorio: R²={res_rand['R2_test']:.3f}  RMSE={usd(res_rand['RMSE_test'])}")

    print(f"\nGráficos generados en {OUT_DIR}/:")
    for a in ARCHIVOS:
        print(f"   - {a}")


if __name__ == "__main__":
    main()
