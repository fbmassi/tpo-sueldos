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
def cargar(ver: float) -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    ctx = pd.read_parquet(CONTEXTO)
    return df.merge(ctx, on="fecha_edicion", how="left")


def fmt_m(x, _p=None) -> str:
    return f"${x/1e6:.1f}M"


try:
    df = cargar(DATASET.stat().st_mtime)
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

tab1, tab2, tab3 = st.tabs(
    ["Distribución", "Salario por factor", "Tecnologías"])

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
    min_n = st.slider("Mínimo de respuestas por categoría", 10, 300, 50, step=10,
                      help="Evita mostrar categorías con muy pocos casos, no representativas.")
    sub = df[df["salario_real_ars"] < df["salario_real_ars"].quantile(0.99)]

    # sólo categorías con suficientes casos, luego ordenadas por mediana (top 12)
    conteo = sub[factor].value_counts()
    validas = conteo[conteo >= min_n].index
    orden = (sub[sub[factor].isin(validas)].groupby(factor)["salario_real_ars"]
             .median().sort_values(ascending=False).head(12).index.tolist())

    if not orden:
        st.info("No hay categorías con esa cantidad mínima de casos. Bajá el umbral.")
    else:
        datos = [sub[sub[factor] == cat]["salario_real_ars"].values / 1e6 for cat in orden]
        fig, ax = plt.subplots(figsize=(10, max(3, len(orden) * 0.45)))
        ax.boxplot(datos, vert=False, tick_labels=orden, showfliers=False)
        ax.set_xlabel("Salario real (millones $)")
        ax.invert_yaxis(); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)
        st.caption(f"Sólo categorías con ≥ {min_n} respuestas (las de 1–2 casos no son "
                   "representativas).")

        tabla = (sub[sub[factor].isin(validas)].groupby(factor)["salario_real_ars"]
                 .agg(Casos="size", Mediana="median").sort_values("Mediana", ascending=False))
        tabla["Mediana"] = (tabla["Mediana"] / 1e6).round(2)
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

    st.subheader("¿Qué stack usa cada seniority?")
    st.caption("% de profesionales de cada nivel que usan cada tecnología (top 12).")
    orden_sen = [s for s in ["junior", "semi-senior", "senior"]
                 if s in df["seniority"].unique()]
    top_st = freq.head(12).index
    tot = df["seniority"].value_counts()
    uso = (techs[techs["t"].isin(top_st)]
           .groupby(["seniority", "t"]).size().unstack(fill_value=0))
    uso = (uso.div(tot, axis=0) * 100).reindex(orden_sen)[top_st]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(uso.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(top_st))); ax.set_xticklabels(top_st, rotation=40, ha="right")
    ax.set_yticks(range(len(orden_sen))); ax.set_yticklabels(orden_sen)
    for i in range(uso.shape[0]):
        for j in range(uso.shape[1]):
            v = uso.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > uso.values.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, label="% de uso", fraction=0.025)
    st.pyplot(fig); plt.close(fig)
    st.caption("Comparando junior vs senior se ve qué tecnologías se concentran en cada nivel "
               "(p. ej. si los seniors migran más a backend/infra).")

# ---- Análisis completo (PNGs generados por eda_completo.py) ----
eda_dir = ROOT / "data" / "processed" / "eda"
if eda_dir.exists():
    with st.expander("📑 Ver el análisis completo (12 láminas generadas por el script)"):
        for png in sorted(eda_dir.glob("eda_*.png")):
            st.image(str(png), width="stretch")
