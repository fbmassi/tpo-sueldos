"""
app/pages/01_eda.py
===================

Página de Análisis Exploratorio de Datos (EDA) interactivo del dataset de
salarios tech. Permite explorar la distribución de sueldos por distintos
factores del perfil, las tecnologías mejor pagas y la evolución temporal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = ROOT / "data" / "processed" / "dataset_final_mercado_laboral.parquet"
CONTEXTO = ROOT / "data" / "processed" / "contexto_macroeconomico.parquet"

@st.cache_data(show_spinner="Cargando datos…")
def cargar() -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    ctx = pd.read_parquet(CONTEXTO)
    return df.merge(ctx, on="fecha_edicion", how="left")


def fmt_m(x, _p=None) -> str:
    return f"${x/1e6:.1f}M"


try:
    df = cargar()
except FileNotFoundError:
    st.error("No encuentro el dataset. Corré antes "
             "`python notebooks/limpiar_y_unificar_datos.py`")
    st.stop()

st.title("📊 Análisis exploratorio")
st.caption("Salarios en pesos reales de mayo 2026. Base: Sysarmy 2022–2025.")

# ---- KPIs ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Ediciones", df["fecha_edicion"].nunique())
k3.metric("Salario mediano", f"${df['salario_real_ars'].median()/1e6:.2f}M")
k4.metric("Cobran en USD", f"{df['cobra_en_dolares'].mean()*100:.0f}%")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Distribución", "Salario por factor", "Tecnologías", "Evolución temporal",
     "Poder de compra en el tiempo"])

# ---- TAB 1: distribución general ----
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribución de salarios")
        s = df["salario_real_ars"]
        s = s[s < s.quantile(0.99)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s / 1e6, bins=50, color="steelblue", edgecolor="white")
        ax.axvline(s.median() / 1e6, color="red", ls="--",
                   label=f"Mediana ${s.median()/1e6:.2f}M")
        ax.set_xlabel("Salario real (millones $)"); ax.set_ylabel("Frecuencia")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)
    with c2:
        st.subheader("Composición por seniority")
        sen = df["seniority"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(sen.values, labels=sen.index, autopct="%1.0f%%",
               colors=["#4c72b0", "#dd8452", "#55a868"], startangle=90)
        st.pyplot(fig); plt.close(fig)

# ---- TAB 2: salario por factor (interactivo) ----
with tab2:
    factor = st.selectbox(
        "Ver salario según…",
        ["rol", "provincia", "seniority", "modalidad", "tamano_empresa",
         "genero", "cobra_en_dolares"])
    sub = df[df["salario_real_ars"] < df["salario_real_ars"].quantile(0.99)]
    # ordenar categorías por mediana, top 12
    orden = (sub.groupby(factor)["salario_real_ars"].median()
             .sort_values(ascending=False).head(12).index.tolist())
    datos = [sub[sub[factor] == cat]["salario_real_ars"].values / 1e6
             for cat in orden]
    fig, ax = plt.subplots(figsize=(10, max(3, len(orden) * 0.45)))
    ax.boxplot(datos, vert=False, tick_labels=orden, showfliers=False)
    ax.set_xlabel("Salario real (millones $)")
    ax.invert_yaxis(); ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close(fig)

    tabla = (sub.groupby(factor)["salario_real_ars"]
             .agg(n="size", mediana="median").sort_values("mediana", ascending=False))
    tabla["mediana"] = (tabla["mediana"] / 1e6).round(2)
    st.dataframe(tabla.head(12), width="stretch")

# ---- TAB 3: tecnologías ----
with tab3:
    techs = (df.assign(t=df["tecnologias"].fillna("").str.split(","))
             .explode("t"))
    techs["t"] = techs["t"].str.strip()
    techs = techs[(techs["t"] != "") & (techs["t"] != "No especifica") &
                  (techs["t"] != "ninguno de los anteriores")]
    freq = techs["t"].value_counts()
    elegibles = freq[freq >= 200].index
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Más mencionadas")
        top = freq.head(12).sort_values()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top.index, top.values, color="steelblue")
        ax.set_xlabel("Menciones"); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)
    with c2:
        st.subheader("Mejor pagas (≥200 menciones)")
        pago = (techs[techs["t"].isin(elegibles)]
                .groupby("t")["salario_real_ars"].median()
                .sort_values().tail(12) / 1e6)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(pago.index, pago.values, color="seagreen")
        ax.set_xlabel("Salario mediano (millones $)"); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)

# ---- TAB 4: evolución en el tiempo (pesos reales vs USD del momento) ----
with tab4:
    st.subheader("Evolución del salario en el tiempo")
    m = df.copy()
    m["nominal"] = m["canastas_basicas"] * m["cbt"]        # salario nominal en pesos del mes
    m["usd_momento"] = m["nominal"] / m["dolar_mep"]       # USD al dólar de CADA edición
    m["ratio_ripte"] = m["nominal"] / m["ripte"]
    g = (m.groupby(m["fecha_edicion"].dt.date)
         .agg(pesos_real=("salario_real_ars", "median"),
              usd_momento=("usd_momento", "median"),
              canastas=("canastas_basicas", "median"),
              ripte=("ratio_ripte", "median")))

    c1, c2 = st.columns(2)
    with c1:
        st.caption("En **pesos reales** (deflactado por IPC, base may-2026)")
        st.line_chart(g["pesos_real"] / 1e6, y_label="Millones $")
    with c2:
        st.caption("En **dólares del momento** (salario ÷ dólar de cada edición)")
        st.line_chart(g["usd_momento"], y_label="USD del mes")
    st.warning(
        "**No son lo mismo:** en pesos reales el salario quedó casi estancado, pero en "
        "dólares del momento **casi se triplicó** (USD 945 → 2.257). La diferencia es el "
        "dólar: se atrasó frente a los salarios, sobre todo desde 2024, así que un mismo "
        "sueldo compra cada vez más dólares.")

    st.subheader("Poder de compra según cada vara (base 2022.2 = 100)")
    idx = (g / g.iloc[0] * 100).rename(columns={
        "pesos_real": "Pesos reales (IPC)",
        "usd_momento": "USD del momento",
        "canastas": "Canastas básicas (CBT)",
        "ripte": "vs mercado formal (RIPTE)"})
    st.line_chart(idx, y_label="Índice (2022.2 = 100)")
    st.caption("El Big Mac se omite: su dato anual desactualizado distorsiona la serie.")

# ---- Análisis completo (PNGs generados por eda_completo.py) ----
eda_dir = ROOT / "data" / "processed" / "eda"
if eda_dir.exists():
    with st.expander("📑 Ver el análisis completo (12 láminas generadas por el script)"):
        for png in sorted(eda_dir.glob("eda_*.png")):
            st.image(str(png), width="stretch")
