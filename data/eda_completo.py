"""
eda_completo.py
===============

Análisis Exploratorio de Datos (AED) completo del dataset final del TPO.

Lee:
    data/processed/dataset_final_mercado_laboral.parquet  (31k filas, por persona)
    data/processed/contexto_macroeconomico.parquet        (macro por edición, FK fecha_edicion)
    data/raw/ipc_indec.csv, us_cpi.csv, dolar_mep.parquet (para los valores base)

Genera 11 PNG (uno por sección) en data/processed/eda/.

NOTA sobre el esquema: el dataset final guarda los salarios YA ajustados
(salario_real_ars / salario_real_usd, base mayo-2026) y NO el nominal. Para las
secciones que necesitan el nominal (evolución nominal vs real) o métricas
derivadas (ratio vs RIPTE, big macs) este script:
  - reconstruye el nominal por fila invirtiendo el ajuste según la moneda
    nativa del sueldo (cobra_en_dolares), y
  - deriva ratio_vs_ripte = nominal / RIPTE_mes  y  big_macs = nominal / BigMac_mes
usando el contexto macroeconómico de cada edición.

Ejecutar:  python data/eda_completo.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin display: solo exportar PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
PROC_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = PROC_DIR / "eda"

DATASET = PROC_DIR / "dataset_final_mercado_laboral.parquet"
CONTEXTO = PROC_DIR / "contexto_macroeconomico.parquet"

FECHA_BASE = pd.Timestamp("2026-05-01")  # base de los salarios reales (= ETL)
DPI = 300

ARCHIVOS_GENERADOS: list[str] = []


def guardar(fig, nombre: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destino = OUT_DIR / nombre
    fig.tight_layout()
    fig.savefig(destino, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    ARCHIVOS_GENERADOS.append(nombre)
    print(f"  ✓ {nombre}")


def fmt_m(x, _pos=None) -> str:
    """Formatea pesos en millones."""
    return f"${x/1e6:.1f}M"


# ----------------------------------------------------------------------------
# Carga y preparación
# ----------------------------------------------------------------------------
def _valor_base(path: Path, col_fecha_hint: str, col_valor_hint: str) -> float:
    """Valor de una serie cruda en el mes más cercano a FECHA_BASE."""
    s = pd.read_csv(path)
    fcol = next((c for c in s.columns
                 if col_fecha_hint in c.lower() or "date" in c.lower()
                 or "tiempo" in c.lower()), s.columns[0])
    vcol = next(c for c in s.columns if col_valor_hint in c.lower())
    s["_f"] = pd.to_datetime(s[fcol], errors="coerce")
    s["_v"] = pd.to_numeric(s[vcol], errors="coerce")
    s = s.dropna(subset=["_f", "_v"])
    return float(s.loc[(s["_f"] - FECHA_BASE).abs().idxmin(), "_v"])


def cargar_datos() -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    ctx = pd.read_parquet(CONTEXTO)
    df = df.merge(ctx, on="fecha_edicion", how="left")
    print(f"Dataset: {len(df)} filas | contexto: {len(ctx)} ediciones")

    # ---- valores base (mayo-2026) para reconstruir el nominal ----
    ipc_base = _valor_base(RAW_DIR / "ipc_indec.csv", "tiempo", "ipc")
    uscpi_base = _valor_base(RAW_DIR / "us_cpi.csv", "fecha", "cpi")
    p = pd.read_parquet(RAW_DIR / "dolar_mep.parquet")
    p["date"] = pd.to_datetime(p["date"])
    blue = p[p["source"] == "Blue"].copy()
    blue["v"] = blue[["value_sell", "value_buy"]].mean(axis=1)
    blue["mes"] = blue["date"].dt.to_period("M").dt.to_timestamp()
    mens = blue.groupby("mes")["v"].mean()
    mep_base = float(mens.iloc[(mens.index - FECHA_BASE).map(abs).argmin()])
    print(f"Bases (mayo-2026): IPC={ipc_base:,.1f}  MEP={mep_base:,.1f}  US_CPI={uscpi_base:.1f}")

    # ---- reconstruir NOMINAL por fila (inverso del ajuste del ETL) ----
    cobra = df["cobra_en_dolares"].fillna(False).astype(bool)
    nominal_pesos = df["salario_real_ars"] * (df["ipc"] / ipc_base)
    nominal_dolar = (df["salario_real_usd"] * (df["us_cpi"] / uscpi_base)
                     * df["dolar_mep"])
    df["salario_nominal_ars"] = nominal_pesos.where(~cobra, nominal_dolar)

    # ---- métricas derivadas con el macro de cada edición ----
    df["ratio_vs_ripte"] = df["salario_nominal_ars"] / df["ripte"]
    df["big_macs_mensuales"] = df["salario_nominal_ars"] / df["precio_bigmac_ars"]
    return df


# ----------------------------------------------------------------------------
# SECCIÓN 1 — Composición y calidad
# ----------------------------------------------------------------------------
def seccion_01(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    comp = (df.notna().mean() * 100).sort_values()
    axs[0, 0].barh(comp.index, comp.values, color="steelblue")
    axs[0, 0].set_xlim(0, 105)
    axs[0, 0].set_title("Completitud por columna (%)")
    axs[0, 0].set_xlabel("% de datos presentes")
    axs[0, 0].tick_params(axis="y", labelsize=7)
    axs[0, 0].grid(True, alpha=0.3)

    prov = df["provincia"].value_counts().head(10).sort_values()
    axs[0, 1].barh(prov.index, prov.values, color="darkorange")
    axs[0, 1].set_title("Registros por provincia (top 10)")
    axs[0, 1].set_xlabel("Cantidad de registros")
    axs[0, 1].grid(True, alpha=0.3)

    sen = df["seniority"].value_counts()
    axs[1, 0].pie(sen.values, labels=sen.index, autopct="%1.1f%%",
                  colors=["#4c72b0", "#dd8452", "#55a868"], startangle=90)
    axs[1, 0].set_title("Distribución por seniority")

    axs[1, 1].hist(df["anos_experiencia_total"].dropna(), bins=40,
                   color="seagreen", edgecolor="white")
    axs[1, 1].set_title("Distribución de años de experiencia total")
    axs[1, 1].set_xlabel("Años de experiencia")
    axs[1, 1].set_ylabel("Frecuencia")
    axs[1, 1].grid(True, alpha=0.3)

    fig.suptitle("SECCIÓN 1 — Composición y calidad del dataset "
                 f"({len(df):,} registros, 6 ediciones)", fontsize=14, y=1.0)
    guardar(fig, "eda_01_composicion.png")


# ----------------------------------------------------------------------------
# SECCIÓN 2 — H1: Seniority
# ----------------------------------------------------------------------------
ORDEN_SEN = ["junior", "semi-senior", "senior"]


def seccion_02(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    sub = df[df["salario_real_ars"] < df["salario_real_ars"].quantile(0.99)]

    sns.boxplot(data=sub, x="seniority", y="salario_real_ars",
                order=ORDEN_SEN, ax=axs[0, 0], palette="Blues")
    axs[0, 0].set_title("Salario real ARS vs seniority")
    axs[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(fmt_m))
    axs[0, 0].grid(True, alpha=0.3)

    sns.boxplot(data=sub, x="seniority", y="salario_real_usd",
                order=ORDEN_SEN, ax=axs[0, 1], palette="Greens")
    axs[0, 1].set_title("Salario real USD vs seniority")
    axs[0, 1].grid(True, alpha=0.3)

    sns.violinplot(data=sub, x="seniority", y="salario_real_ars",
                   order=ORDEN_SEN, ax=axs[1, 0], palette="Oranges")
    axs[1, 0].set_title("Forma de la distribución (violin)")
    axs[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(fmt_m))
    axs[1, 0].grid(True, alpha=0.3)

    g = df.groupby("seniority").agg(
        Count=("salario_real_ars", "size"),
        Media=("salario_real_ars", "mean"),
        Mediana=("salario_real_ars", "median"),
        Std=("salario_real_ars", "std"),
        Ratio_RIPTE=("ratio_vs_ripte", "median"),
    ).reindex(ORDEN_SEN)
    filas = [[s, f"{int(r.Count):,}", f"${r.Media/1e6:.2f}M", f"${r.Mediana/1e6:.2f}M",
              f"${r.Std/1e6:.2f}M", f"{r.Ratio_RIPTE:.1f}x"]
             for s, r in g.iterrows()]
    axs[1, 1].axis("off")
    t = axs[1, 1].table(cellText=filas,
                        colLabels=["Seniority", "Count", "Media", "Mediana",
                                   "Std Dev", "Ratio RIPTE"],
                        loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.8)
    axs[1, 1].set_title("Estadísticas por seniority (pesos de may-2026)")

    fig.suptitle("SECCIÓN 2 — H1: Seniority como factor predictivo", fontsize=14, y=1.0)
    guardar(fig, "eda_02_seniority.png")


# ----------------------------------------------------------------------------
# SECCIÓN 3 — H2: Experiencia
# ----------------------------------------------------------------------------
def seccion_03(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    sub = df[(df["anos_experiencia_total"] <= 40) &
             (df["salario_real_ars"] < df["salario_real_ars"].quantile(0.99))].dropna(
                 subset=["anos_experiencia_total", "salario_real_ars"])

    axs[0, 0].scatter(sub["anos_experiencia_total"], sub["salario_real_ars"],
                      alpha=0.5, s=20, color="steelblue", edgecolors="none")
    coef = np.polyfit(sub["anos_experiencia_total"], sub["salario_real_ars"], 2)
    xs = np.linspace(0, sub["anos_experiencia_total"].max(), 200)
    axs[0, 0].plot(xs, np.polyval(coef, xs), color="red", lw=2,
                   label="Regresión polinómica (g=2)")
    axs[0, 0].set_title("Experiencia vs salario real ARS")
    axs[0, 0].set_xlabel("Años de experiencia"); axs[0, 0].set_ylabel("Salario real")
    axs[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(fmt_m))
    axs[0, 0].legend(); axs[0, 0].grid(True, alpha=0.3)

    rangos = pd.cut(df["anos_experiencia_total"], bins=[-0.1, 2, 5, 10, 60],
                    labels=["0-2", "3-5", "6-10", "10+"])
    ct = pd.crosstab(rangos, df["seniority"])[ORDEN_SEN]
    ct.plot(kind="bar", ax=axs[0, 1], color=["#4c72b0", "#dd8452", "#55a868"])
    axs[0, 1].set_title("Experiencia (rangos) vs seniority")
    axs[0, 1].set_xlabel("Rango de experiencia"); axs[0, 1].set_ylabel("Registros")
    axs[0, 1].tick_params(axis="x", rotation=0); axs[0, 1].grid(True, alpha=0.3)

    prom = df.groupby(rangos)["salario_real_ars"].mean()
    axs[1, 0].bar(prom.index.astype(str), prom.values, color="mediumpurple")
    axs[1, 0].set_title("Salario real promedio por rango de experiencia")
    axs[1, 0].set_xlabel("Rango"); axs[1, 0].yaxis.set_major_formatter(
        plt.FuncFormatter(fmt_m))
    axs[1, 0].grid(True, alpha=0.3)

    e = df["anos_experiencia_total"].dropna()
    pear = df[["anos_experiencia_total", "salario_real_ars"]].dropna()
    r, p = stats.pearsonr(pear["anos_experiencia_total"], pear["salario_real_ars"])
    filas = [["Media años", f"{e.mean():.1f}"], ["Mediana", f"{e.median():.1f}"],
             ["Std Dev", f"{e.std():.1f}"], ["Mín", f"{e.min():.0f}"],
             ["Máx", f"{e.max():.0f}"],
             ["Pearson r (vs salario real)", f"{r:.3f}  (p={p:.1e})"]]
    axs[1, 1].axis("off")
    t = axs[1, 1].table(cellText=filas, colLabels=["Métrica", "Valor"],
                        loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.8)
    axs[1, 1].set_title("Estadísticas de experiencia y correlación")

    fig.suptitle("SECCIÓN 3 — H2: La experiencia agrega valor", fontsize=14, y=1.0)
    guardar(fig, "eda_03_experiencia.png")


# ----------------------------------------------------------------------------
# SECCIÓN 4 — H3: Tecnologías
# ----------------------------------------------------------------------------
def _explotar_techs(df: pd.DataFrame) -> pd.DataFrame:
    t = df[["tecnologias", "salario_real_ars"]].dropna()
    t = t[t["tecnologias"] != "No especifica"]
    t = t.assign(tech=t["tecnologias"].str.split(",")).explode("tech")
    t["tech"] = t["tech"].str.strip()
    return t[(t["tech"] != "") & (t["tech"] != "ninguno de los anteriores")]


def seccion_04(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    t = _explotar_techs(df)

    top15 = t["tech"].value_counts().head(15).sort_values()
    axs[0, 0].barh(top15.index, top15.values, color="steelblue")
    axs[0, 0].set_title("Top 15 tecnologías mencionadas")
    axs[0, 0].set_xlabel("Menciones"); axs[0, 0].grid(True, alpha=0.3)

    freq = t["tech"].value_counts()
    elegibles = freq[freq >= 200].index  # mínimo de menciones para promediar
    sal = (t[t["tech"].isin(elegibles)].groupby("tech")["salario_real_ars"]
           .mean().sort_values().tail(10))
    axs[0, 1].barh(sal.index, sal.values, color="darkorange")
    axs[0, 1].set_title("Salario real promedio por tecnología (top 10, ≥200 menciones)")
    axs[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(fmt_m))
    axs[0, 1].grid(True, alpha=0.3)

    mediana_gral = df["salario_real_ars"].median()
    premium = (t[t["tech"].isin(elegibles)].groupby("tech")["salario_real_ars"]
               .median() / mediana_gral * 100 - 100).sort_values().tail(10)
    colores = ["seagreen" if v > 0 else "indianred" for v in premium.values]
    axs[1, 0].barh(premium.index, premium.values, color=colores)
    axs[1, 0].axvline(0, color="black", lw=0.8)
    axs[1, 0].set_title("Premium salarial vs mediana general (%)")
    axs[1, 0].set_xlabel("% sobre la mediana general"); axs[1, 0].grid(True, alpha=0.3)

    stats_t = (t.groupby("tech")["salario_real_ars"]
               .agg(Menciones="size", Promedio="mean", Mediana="median")
               .sort_values("Menciones", ascending=False).head(8))
    filas = [[tech, f"{r.Menciones:,.0f}", f"${r.Promedio/1e6:.2f}M",
              f"${r.Mediana/1e6:.2f}M"] for tech, r in stats_t.iterrows()]
    axs[1, 1].axis("off")
    tb = axs[1, 1].table(cellText=filas,
                         colLabels=["Tecnología", "Menciones", "Promedio", "Mediana"],
                         loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.6)
    axs[1, 1].set_title("Top tecnologías (pesos de may-2026)")

    fig.suptitle("SECCIÓN 4 — H3: Tecnologías premium", fontsize=14, y=1.0)
    guardar(fig, "eda_04_tecnologias.png")


# ----------------------------------------------------------------------------
# SECCIÓN 5 — H4: Geografía
# ----------------------------------------------------------------------------
def seccion_05(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    top10 = df["provincia"].value_counts().head(10).index.tolist()
    sub = df[df["provincia"].isin(top10)]

    prom = sub.groupby("provincia")["salario_real_ars"].mean().sort_values()
    axs[0, 0].barh(prom.index, prom.values, color="steelblue")
    axs[0, 0].set_title("Salario real promedio por provincia (top 10 por registros)")
    axs[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(fmt_m))
    axs[0, 0].grid(True, alpha=0.3)

    s99 = sub[sub["salario_real_ars"] < sub["salario_real_ars"].quantile(0.99)]
    orden = prom.index.tolist()
    sns.boxplot(data=s99, y="provincia", x="salario_real_ars", order=orden,
                ax=axs[0, 1], palette="crest")
    axs[0, 1].set_title("Distribución de salarios por provincia")
    axs[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(fmt_m))
    axs[0, 1].grid(True, alpha=0.3)

    moda = pd.crosstab(sub["provincia"], sub["modalidad"], normalize="index") * 100
    moda = moda.loc[orden]
    moda.plot(kind="barh", stacked=True, ax=axs[1, 0],
              color=["#4c72b0", "#dd8452", "#55a868"][:moda.shape[1]])
    axs[1, 0].set_title("Modalidad de trabajo por provincia (%)")
    axs[1, 0].set_xlabel("%"); axs[1, 0].legend(fontsize=7, loc="lower right")
    axs[1, 0].grid(True, alpha=0.3)

    top5 = df["provincia"].value_counts().head(5).index
    filas = []
    for pv in top5:
        s = df[df["provincia"] == pv]
        rem = (s["modalidad"] == "100% remoto").mean() * 100
        filas.append([pv, f"{len(s):,}", f"${s['salario_real_ars'].mean()/1e6:.2f}M",
                      f"{s['ratio_vs_ripte'].median():.1f}x", f"{rem:.0f}%"])
    axs[1, 1].axis("off")
    t = axs[1, 1].table(cellText=filas,
                        colLabels=["Provincia", "Registros", "Salario prom.",
                                   "Ratio RIPTE", "% Remoto"],
                        loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.8)
    axs[1, 1].set_title("Top 5 provincias")

    fig.suptitle("SECCIÓN 5 — H4: Geografía", fontsize=14, y=1.0)
    guardar(fig, "eda_05_geografia.png")


# ----------------------------------------------------------------------------
# SECCIÓN 6 — H5: Nominal vs Real
# ----------------------------------------------------------------------------
def seccion_06(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    g = df.groupby(df["fecha_edicion"].dt.date).agg(
        nominal=("salario_nominal_ars", "median"),
        real=("salario_real_ars", "median"))
    x = pd.to_datetime(g.index)
    axs[0].plot(x, g["nominal"], "o-", color="royalblue", label="Nominal (del momento)")
    axs[0].plot(x, g["real"], "o-", color="darkorange",
                label="Real (pesos de may-2026)")
    axs[0].fill_between(x, g["nominal"], g["real"], alpha=0.15, color="gray")
    axs[0].set_title("Evolución: salario nominal vs real (mediana por edición)")
    axs[0].set_xlabel("Edición"); axs[0].yaxis.set_major_formatter(
        plt.FuncFormatter(fmt_m))
    axs[0].legend(); axs[0].grid(True, alpha=0.3)

    brecha = ((df["salario_nominal_ars"] - df["salario_real_ars"])
              / df["salario_real_ars"] * 100).dropna()
    brecha = brecha[brecha.between(brecha.quantile(0.005), brecha.quantile(0.995))]
    axs[1].hist(brecha, bins=60, color="indianred", edgecolor="white")
    axs[1].axvline(brecha.mean(), color="black", ls="--",
                   label=f"Media: {brecha.mean():.0f}%")
    axs[1].set_title("¿Cuánto 'engaña' el número nominal?\n"
                     "(brecha nominal-real por registro, %)")
    axs[1].set_xlabel("Brecha (%)  —  negativa: el nominal viejo subestima el valor real")
    axs[1].set_ylabel("Frecuencia"); axs[1].legend(); axs[1].grid(True, alpha=0.3)

    fig.suptitle("SECCIÓN 6 — H5: Poder adquisitivo real vs nominal", fontsize=14, y=1.03)
    guardar(fig, "eda_06_poder_adquisitivo.png")


# ----------------------------------------------------------------------------
# SECCIÓN 7 — H6: Dolarización
# ----------------------------------------------------------------------------
def seccion_07(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    c = df["cobra_en_dolares"].value_counts()
    axs[0, 0].pie([c.get(False, 0), c.get(True, 0)],
                  labels=["Cobra en pesos", "Cobra en dólares"],
                  autopct="%1.1f%%", colors=["#4c72b0", "#55a868"], startangle=90)
    axs[0, 0].set_title("¿Quién cobra en dólares?")

    ev = df.groupby(df["fecha_edicion"].dt.date)["cobra_en_dolares"].mean() * 100
    axs[0, 1].plot(pd.to_datetime(ev.index), ev.values, "o-", color="seagreen")
    axs[0, 1].set_title("Evolución del % que cobra en dólares")
    axs[0, 1].set_ylabel("%"); axs[0, 1].grid(True, alpha=0.3)

    sub = df[df["salario_real_ars"] < df["salario_real_ars"].quantile(0.99)].copy()
    sub["grupo"] = np.where(sub["cobra_en_dolares"], "En dólares", "En pesos")
    sns.boxplot(data=sub, x="grupo", y="salario_real_ars", ax=axs[1, 0],
                palette=["#4c72b0", "#55a868"])
    axs[1, 0].set_title("Salario real según moneda de cobro")
    axs[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(fmt_m))
    axs[1, 0].grid(True, alpha=0.3)

    ars = df[~df["cobra_en_dolares"]]["salario_real_ars"]
    usd = df[df["cobra_en_dolares"]]["salario_real_ars"]
    filas = [
        ["Cobran en pesos", f"{len(ars):,} ({len(ars)/len(df)*100:.0f}%)"],
        ["Cobran en dólares", f"{len(usd):,} ({len(usd)/len(df)*100:.0f}%)"],
        ["Salario prom. (pesos)", f"${ars.mean()/1e6:.2f}M"],
        ["Salario prom. (dólares)", f"${usd.mean()/1e6:.2f}M"],
        ["Premium dolarizado", f"+{(usd.median()/ars.median()-1)*100:.0f}% (mediana)"],
        ["Volatilidad (CV) pesos", f"{ars.std()/ars.mean():.2f}"],
        ["Volatilidad (CV) dólares", f"{usd.std()/usd.mean():.2f}"],
    ]
    axs[1, 1].axis("off")
    t = axs[1, 1].table(cellText=filas, colLabels=["Métrica", "Valor"],
                        loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.7)
    axs[1, 1].set_title("Estadísticas de dolarización")

    fig.suptitle("SECCIÓN 7 — H6: Dolarización", fontsize=14, y=1.0)
    guardar(fig, "eda_07_dolarizacion.png")


# ----------------------------------------------------------------------------
# SECCIÓN 8 — H7: ITCRM
# ----------------------------------------------------------------------------
def seccion_08(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    g = df.groupby(df["fecha_edicion"].dt.date).agg(
        itcrm=("itcrm", "first"), usd=("salario_real_usd", "median"))
    x = pd.to_datetime(g.index)

    ax1 = axs[0]
    ax1.plot(x, g["itcrm"], "o-", color="royalblue", label="ITCRM")
    ax1.set_ylabel("ITCRM", color="royalblue")
    ax1.tick_params(axis="y", labelcolor="royalblue")
    ax2 = ax1.twinx()
    ax2.plot(x, g["usd"], "s--", color="firebrick", label="Salario real USD (mediana)")
    ax2.set_ylabel("Salario real USD", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax1.set_title("¿Se mueven juntos ITCRM y salario en USD?")
    ax1.grid(True, alpha=0.3)

    r, p = stats.pearsonr(g["itcrm"], g["usd"])
    axs[1].scatter(g["itcrm"], g["usd"], s=80, color="steelblue", zorder=3)
    for fecha, fila in g.iterrows():
        axs[1].annotate(str(fecha)[:7], (fila["itcrm"], fila["usd"]),
                        fontsize=7, xytext=(5, 5), textcoords="offset points")
    coef = np.polyfit(g["itcrm"], g["usd"], 1)
    xs = np.linspace(g["itcrm"].min(), g["itcrm"].max(), 50)
    axs[1].plot(xs, np.polyval(coef, xs), color="red", lw=1.5)
    axs[1].set_title(f"Correlación ITCRM vs salario real USD\n"
                     f"Pearson r = {r:.2f} (p={p:.2f}, n=6 ediciones)")
    axs[1].set_xlabel("ITCRM (peso real-alto = dev 'caro')")
    axs[1].set_ylabel("Salario real USD (mediana)")
    axs[1].grid(True, alpha=0.3)

    fig.suptitle("SECCIÓN 8 — H7: Contexto macro (ITCRM)", fontsize=14, y=1.03)
    guardar(fig, "eda_08_itcrm.png")


# ----------------------------------------------------------------------------
# SECCIÓN 9 — Poder adquisitivo concreto
# ----------------------------------------------------------------------------
def seccion_09(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    can = df["canastas_basicas"].dropna()
    can = can[can < can.quantile(0.99)]
    axs[0, 0].hist(can, bins=50, color="steelblue", edgecolor="white")
    axs[0, 0].axvline(1, color="red", ls="--", lw=2, label="Línea de pobreza (1 CBT)")
    axs[0, 0].axvline(2, color="darkorange", ls="--", lw=2, label="Comodidad (2 CBT)")
    axs[0, 0].set_title("Canastas básicas cubiertas por el salario")
    axs[0, 0].set_xlabel("Canastas básicas"); axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    bm = df["big_macs_mensuales"].dropna()
    bm = bm[bm < bm.quantile(0.99)]
    axs[0, 1].hist(bm, bins=50, color="goldenrod", edgecolor="white")
    axs[0, 1].set_title("Big Macs mensuales que compra el salario")
    axs[0, 1].set_xlabel("Big Macs / mes"); axs[0, 1].grid(True, alpha=0.3)

    rr = df["ratio_vs_ripte"].dropna()
    rr = rr[rr < rr.quantile(0.99)]
    axs[1, 0].hist(rr, bins=50, color="mediumpurple", edgecolor="white")
    axs[1, 0].axvline(1, color="red", ls="--", lw=2, label="= salario formal promedio")
    axs[1, 0].set_title("Ratio vs RIPTE (salario formal promedio AR)")
    axs[1, 0].set_xlabel("Veces el RIPTE"); axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    percs = [10, 25, 50, 75, 90]
    metricas = {"Salario real ARS": df["salario_real_ars"],
                "Canastas básicas": df["canastas_basicas"],
                "Big Macs/mes": df["big_macs_mensuales"],
                "Ratio vs RIPTE": df["ratio_vs_ripte"]}
    filas = []
    for nom, serie in metricas.items():
        vals = np.percentile(serie.dropna(), percs)
        if nom == "Salario real ARS":
            filas.append([nom] + [f"${v/1e6:.1f}M" for v in vals])
        else:
            filas.append([nom] + [f"{v:.1f}" for v in vals])
    axs[1, 1].axis("off")
    t = axs[1, 1].table(cellText=filas,
                        colLabels=["Métrica", "P10", "P25", "P50", "P75", "P90"],
                        loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.8)
    axs[1, 1].set_title("Percentiles de poder adquisitivo")

    fig.suptitle("SECCIÓN 9 — Poder adquisitivo en métricas concretas", fontsize=14, y=1.0)
    guardar(fig, "eda_09_poder_adquisitivo_concreto.png")


# ----------------------------------------------------------------------------
# SECCIÓN 10 — Correlaciones
# ----------------------------------------------------------------------------
def seccion_10(df: pd.DataFrame) -> None:
    # Sólo campos PROPIOS del dataset (perfil + salario). Las macro y las
    # derivadas de poder adquisitivo se excluyen: se correlacionan entre sí por
    # construcción y ensucian la matriz. Las categóricas relevantes se
    # codifican para poder incluirlas en el Pearson.
    m = pd.DataFrame()
    m["edad"] = df["edad"]
    m["años_experiencia"] = df["anos_experiencia_total"]
    m["años_empresa_actual"] = df["anos_empresa_actual"]
    m["seniority (0=jr,1=ssr,2=sr)"] = df["seniority"].map(
        {"junior": 0, "semi-senior": 1, "senior": 2})
    m["cobra_en_dolares (0/1)"] = df["cobra_en_dolares"].astype(int)
    m["es_remoto (0/1)"] = (df["modalidad"] == "100% remoto").astype(int)
    m["genero_fem (0/1)"] = (df["genero"] == "femenino").astype(int)
    m["cant_tecnologias"] = (df["tecnologias"]
                             .replace("No especifica", np.nan)
                             .str.split(",").str.len())
    m["salario_real_ars"] = df["salario_real_ars"]
    m["salario_real_usd"] = df["salario_real_usd"]
    corr = m.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
    ax.set_title("SECCIÓN 10 — Matriz de correlaciones (Pearson) — "
                 "campos del dataset (perfil + salario)", fontsize=13)
    guardar(fig, "eda_10_correlaciones.png")


# ----------------------------------------------------------------------------
# SECCIÓN 12 — Correlación de categóricas con el salario (punto-biserial)
# ----------------------------------------------------------------------------
def _r_dummies(df: pd.DataFrame, dummies: pd.DataFrame,
               objetivo: pd.Series, min_n: int = 100) -> pd.Series:
    """Correlación punto-biserial (Pearson con dummy 0/1) de cada categoría."""
    rs = {}
    for col in dummies.columns:
        d = dummies[col]
        if d.sum() < min_n or d.sum() > len(d) - min_n:
            continue
        rs[col] = objetivo.corr(d.astype(float))
    return pd.Series(rs).sort_values()


def seccion_12(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    sal = df["salario_real_ars"]

    def _barh(ax, serie, titulo, top=12):
        serie = pd.concat([serie.head(top // 2), serie.tail(top // 2)])
        colores = ["indianred" if v < 0 else "seagreen" for v in serie.values]
        ax.barh(serie.index, serie.values, color=colores)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(titulo)
        ax.set_xlabel("correlación punto-biserial con salario real")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.3)

    # 1) tecnologías (dummy: ¿menciona esta tech?)
    techs = df["tecnologias"].fillna("").str.get_dummies(sep=", ")
    techs = techs.drop(columns=[c for c in ("No especifica",
                                            "ninguno de los anteriores")
                                if c in techs.columns], errors="ignore")
    _barh(axs[0, 0], _r_dummies(df, techs, sal, 150),
          "Tecnologías: ¿cuáles empujan el salario?")

    # 2) provincias
    provs = pd.get_dummies(df["provincia"])
    _barh(axs[0, 1], _r_dummies(df, provs, sal, 100),
          "Provincia de residencia")

    # 3) roles
    roles = pd.get_dummies(df["rol"])
    _barh(axs[1, 0], _r_dummies(df, roles, sal, 150),
          "Rol / puesto")

    # 4) género, modalidad, dólar, tamaño de empresa
    otras = pd.concat([
        pd.get_dummies(df["genero"]).add_prefix("género: "),
        pd.get_dummies(df["modalidad"]).add_prefix("modalidad: "),
        df["cobra_en_dolares"].astype(int).rename("cobra en dólares"),
        pd.get_dummies(df["tamano_empresa"]).add_prefix("empresa: "),
    ], axis=1)
    _barh(axs[1, 1], _r_dummies(df, otras, sal, 100),
          "Género, modalidad, dólar y tamaño de empresa", top=14)

    fig.suptitle("SECCIÓN 12 — ¿Qué categorías se asocian con mejor salario?\n"
                 "(rojo: asociación negativa · verde: positiva)", fontsize=14, y=1.0)
    guardar(fig, "eda_12_correlaciones_categoricas.png")


# ----------------------------------------------------------------------------
# SECCIÓN 11 — Outliers
# ----------------------------------------------------------------------------
def seccion_11(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")

    cols_show = ["rol", "seniority", "anos_experiencia_total", "provincia",
                 "salario_real_ars"]
    def _filas(sub):
        return [[str(r["rol"])[:28], r["seniority"],
                 f"{r['anos_experiencia_total']:.0f}", str(r["provincia"])[:18],
                 f"${r['salario_real_ars']/1e6:.1f}M"]
                for _, r in sub[cols_show].iterrows()]

    top5 = df.nlargest(5, "salario_real_ars")
    low5 = df.nsmallest(5, "salario_real_ars")

    s = df["salario_real_ars"]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((s < lo) | (s > hi)).sum())

    y = 0.97
    ax.text(0.5, y, "SECCIÓN 11 — Outliers y anomalías (salario real, may-2026)",
            ha="center", fontsize=14, weight="bold", transform=ax.transAxes)

    ax.text(0.02, 0.88, "TOP 5 SALARIOS MÁS ALTOS", fontsize=11, weight="bold",
            transform=ax.transAxes, color="darkgreen")
    t1 = ax.table(cellText=_filas(top5),
                  colLabels=["Rol", "Seniority", "Exp.", "Provincia", "Salario real"],
                  bbox=[0.02, 0.62, 0.96, 0.24], cellLoc="center")
    t1.auto_set_font_size(False); t1.set_fontsize(9)

    ax.text(0.02, 0.55, "TOP 5 SALARIOS MÁS BAJOS", fontsize=11, weight="bold",
            transform=ax.transAxes, color="darkred")
    t2 = ax.table(cellText=_filas(low5),
                  colLabels=["Rol", "Seniority", "Exp.", "Provincia", "Salario real"],
                  bbox=[0.02, 0.29, 0.96, 0.24], cellLoc="center")
    t2.auto_set_font_size(False); t2.set_fontsize(9)

    texto = (f"DETECCIÓN DE OUTLIERS POR IQR (sobre salario_real_ars)\n"
             f"  Q1 = ${q1/1e6:.2f}M    Q3 = ${q3/1e6:.2f}M    IQR = ${iqr/1e6:.2f}M\n"
             f"  Límite inferior = ${lo/1e6:.2f}M    Límite superior = ${hi/1e6:.2f}M\n"
             f"  Outliers detectados: {n_out:,} ({n_out/len(df)*100:.1f}% del dataset)\n"
             f"  Política del proyecto: NO se eliminan; quedan marcados (es_outlier).")
    ax.text(0.02, 0.05, texto, fontsize=10, family="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    guardar(fig, "eda_11_outliers.png")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    if not DATASET.exists():
        print(f"✗ No existe {DATASET}. Corré primero limpiar_y_unificar_datos.py")
        return
    print("Cargando datos…")
    df = cargar_datos()
    print(f"Generando AED en {OUT_DIR}/\n")

    secciones = [seccion_01, seccion_02, seccion_03, seccion_04, seccion_05,
                 seccion_06, seccion_07, seccion_08, seccion_09, seccion_10,
                 seccion_11, seccion_12]
    for fn in secciones:
        try:
            fn(df)
        except Exception as exc:  # noqa: BLE001 — seguir con el resto
            print(f"  ✗ {fn.__name__} FALLÓ: {exc}")

    print("\n" + "=" * 50)
    print("✅ AED COMPLETADO")
    print(f"Archivos generados en {OUT_DIR}:")
    for a in ARCHIVOS_GENERADOS:
        print(f"  - {a}")
    print("=" * 50)


if __name__ == "__main__":
    main()
