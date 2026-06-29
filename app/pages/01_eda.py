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

st.set_page_config(page_title="EDA — Salarios Tech", page_icon="📊",
                   layout="wide")


@st.cache_data(show_spinner="Cargando datos…")
def cargar() -> pd.DataFrame:
    return pd.read_parquet(DATASET)


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

tab1, tab2, tab3, tab4 = st.tabs(
    ["Distribución", "Salario por factor", "Tecnologías", "Evolución temporal"])

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

# ---- TAB 4: evolución temporal ----
with tab4:
    st.subheader("Evolución del salario real (mediana por edición)")
    g = (df.groupby(df["fecha_edicion"].dt.date)
         .agg(ARS=("salario_real_ars", "median"),
              USD=("salario_real_usd", "median")))
    c1, c2 = st.columns(2)
    with c1:
        st.caption("En pesos reales de mayo 2026")
        st.line_chart(g["ARS"] / 1e6, y_label="Millones $")
    with c2:
        st.caption("En dólares reales de mayo 2026")
        st.line_chart(g["USD"], y_label="USD")
    st.info("En pesos reales el poder de compra se mantuvo ~estable, pero en "
            "dólares reales casi se duplicó entre 2022 y 2025.")

# ---- Análisis completo (PNGs generados por eda_completo.py) ----
eda_dir = ROOT / "data" / "processed" / "eda"
if eda_dir.exists():
    with st.expander("📑 Ver el análisis completo (12 láminas generadas por el script)"):
        for png in sorted(eda_dir.glob("eda_*.png")):
            st.image(str(png), width="stretch")
