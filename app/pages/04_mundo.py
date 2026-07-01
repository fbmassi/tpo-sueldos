"""
app/pages/04_mundo.py
=====================

Comparación del salario de developers de Argentina contra el resto del mundo,
usando la encuesta global de Stack Overflow. Descriptivo e interactivo
(selector de países). El salario está en USD nominales mensuales (no ajustado
por poder adquisitivo entre países).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
SO_CSV = ROOT / "data" / "raw" / "datosInternacionales.csv"
DATASET = ROOT / "data" / "processed" / "dataset_final_mercado_laboral.parquet"

AMBAR, GRAY, BLACK, RED = "#E6A100", "#3A3A3A", "#0A0A0A", "#C0392B"

# nombres largos de Stack Overflow -> nombre corto en español
PAISES = {
    "United States of America": "EE.UU.", "Germany": "Alemania", "Canada": "Canadá",
    "United Kingdom of Great Britain and Northern Ireland": "Reino Unido",
    "Netherlands": "Países Bajos", "France": "Francia", "Spain": "España",
    "Poland": "Polonia", "Brazil": "Brasil", "India": "India", "Ukraine": "Ucrania",
    "Australia": "Australia", "Italy": "Italia", "Mexico": "México", "Argentina": "Argentina",
    "Chile": "Chile", "Uruguay": "Uruguay", "Colombia": "Colombia", "Portugal": "Portugal",
}

st.set_page_config(page_title="Argentina vs el mundo", page_icon="🌍", layout="wide")


@st.cache_data(show_spinner="Cargando encuesta internacional…")
def cargar():
    so = pd.read_csv(SO_CSV, usecols=["Country", "ConvertedCompYearly"], low_memory=False)
    so = so[so["ConvertedCompYearly"].notna() & (so["ConvertedCompYearly"] > 0)].copy()
    so["mensual"] = so["ConvertedCompYearly"] / 12
    sys_usd = pd.read_parquet(DATASET, columns=["salario_real_usd"])["salario_real_usd"].median()
    return so, float(sys_usd)


try:
    so, sys_ar = cargar()
except FileNotFoundError:
    st.error("No encuentro el dataset de Stack Overflow en data/raw/datosInternacionales.csv")
    st.stop()

st.title("🌍 Argentina vs el mundo")
st.caption("Salario mediano de developers, en USD/mes. Fuente: encuesta global de "
           "Stack Overflow. Valores nominales (sin ajustar por costo de vida).")

# medianas por país disponibles (con muestra mínima)
conteo = so["Country"].value_counts()
disponibles = [p for p in PAISES if p in conteo.index and conteo[p] >= 30]
default = ["United States of America", "Germany", "Canada", "Spain", "Poland",
           "Brazil", "Argentina", "India", "Ukraine"]
default = [p for p in default if p in disponibles]

sel = st.multiselect(
    "Países a comparar (Argentina siempre incluida):",
    options=[PAISES[p] for p in disponibles],
    default=[PAISES[p] for p in default])
inv = {v: k for k, v in PAISES.items()}
paises_sel = {inv[s] for s in sel} | {"Argentina"}

med = {PAISES[p]: so[so["Country"] == p]["mensual"].median() for p in paises_sel}
med = pd.Series(med).sort_values()

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(9, max(3, len(med) * 0.5)))
    colors = [AMBAR if p == "Argentina" else GRAY for p in med.index]
    ax.barh(med.index, med.values, color=colors, edgecolor=BLACK, linewidth=1)
    for i, (p, v) in enumerate(med.items()):
        ax.text(v + med.max() * 0.01, i, f"USD {v:,.0f}", va="center",
                fontsize=9, fontweight="bold")
    ax.axvline(sys_ar, color=RED, ls="--", lw=2)
    ax.text(sys_ar, len(med) - 0.4, f"  Sysarmy AR\n  USD {sys_ar:,.0f}",
            color=RED, fontsize=8.5, fontweight="bold", va="top")
    ax.set_xlabel("Salario mediano (USD / mes)")
    ax.set_xlim(0, med.max() * 1.25)
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig); plt.close(fig)

with col2:
    ar = med.get("Argentina", float("nan"))
    st.metric("Argentina (Stack Overflow)", f"USD {ar:,.0f}")
    st.metric("Argentina (Sysarmy)", f"USD {sys_ar:,.0f}")
    if len(med) > 1:
        top = med.idxmax()
        st.metric(f"Techo ({top})", f"USD {med.max():,.0f}",
                  delta=f"{med.max()/ar:.1f}× Argentina")

st.info(
    "**Lectura:** el developer argentino se ubica por encima de otros emergentes "
    "(Brasil, India, Ucrania) pero muy por debajo de EE.UU. y Europa. Sysarmy mide "
    "más bajo que Stack Overflow porque la encuesta global capta perfiles más "
    "internacionales y senior."
)
st.warning(
    "**Limitación:** son USD nominales, no ajustados por poder adquisitivo — USD 40k "
    "rinden distinto en India que en Suiza. Replicar el modelo predictivo a nivel "
    "internacional es una línea futura (otro esquema de variables, el país domina el "
    "salario, y Argentina queda diluida en la muestra global)."
)
