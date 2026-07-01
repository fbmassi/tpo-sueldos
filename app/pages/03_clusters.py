"""
app/pages/03_clusters.py
========================

Página de segmentación (clustering no supervisado) de profesionales tech.

Agrupa perfiles SIN usar el salario (K-Means sobre edad, experiencia,
antigüedad, modalidad, cobra-en-dólares y tamaño de empresa) y recién después
muestra cuánto gana cada segmento. Permite además ubicar un perfil propio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = ROOT / "data" / "processed" / "dataset_final_mercado_laboral.parquet"

# Features de PERFIL para agrupar (NO incluye el salario: es no supervisado).
NUM = ["edad", "anos_experiencia_total", "anos_empresa_actual"]
CAT = ["modalidad", "cobra_en_dolares", "tamano_empresa"]


@st.cache_data(show_spinner="Cargando datos…")
def cargar() -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    df["cobra_en_dolares"] = df["cobra_en_dolares"].astype(str)
    return df


@st.cache_resource(show_spinner="Calculando segmentos…")
def entrenar(k: int):
    df = cargar()
    pre = ColumnTransformer([("n", StandardScaler(), NUM),
                             ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    Xp = pre.fit_transform(df[NUM + CAT])
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xp)
    coords = PCA(n_components=2, random_state=42).fit_transform(
        Xp.toarray() if hasattr(Xp, "toarray") else Xp)
    return pre, km, coords


try:
    df = cargar()
except FileNotFoundError:
    st.error("No encuentro el dataset. Corré antes "
             "`python notebooks/limpiar_y_unificar_datos.py`")
    st.stop()

st.title("🧩 Segmentos del mercado (clustering)")
st.markdown(
    "Agrupamos perfiles parecidos **sin mirar el salario** (K-Means). "
    "Recién después vemos cuánto gana cada grupo: así descubrimos los "
    "**arquetipos** del mercado tech argentino."
)

k = st.slider("Cantidad de segmentos (k)", 2, 8, 4)
pre, km, coords = entrenar(k)
df = df.copy()
df["cluster"] = km.labels_

# ---- Scatter PCA ----
c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Mapa de segmentos (proyección 2D)")
    fig, ax = plt.subplots(figsize=(7, 5))
    samp = np.random.RandomState(0).choice(len(df), min(6000, len(df)), replace=False)
    sc = ax.scatter(coords[samp, 0], coords[samp, 1],
                    c=df["cluster"].values[samp], cmap="tab10", s=8, alpha=0.5)
    ax.set_xlabel("Componente 1"); ax.set_ylabel("Componente 2")
    ax.set_title("Cada punto es un profesional; el color, su segmento")
    legend = ax.legend(*sc.legend_elements(), title="Segmento", loc="best", fontsize=8)
    ax.add_artist(legend); ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close(fig)
with c2:
    st.subheader("Salario por segmento")
    salc = (df.groupby("cluster")["salario_real_ars"].median()
            .sort_values(ascending=False) / 1e6)
    st.bar_chart(salc, y_label="Salario mediano (M$)", horizontal=True)

# ---- Caracterización ----
st.subheader("¿Quién es quién en cada segmento?")
filas = []
for c in range(k):
    g = df[df["cluster"] == c]
    filas.append({
        "Segmento": c,
        "N": len(g),
        "%": f"{len(g)/len(df)*100:.0f}%",
        "Edad": f"{g['edad'].median():.0f}",
        "Exp.": f"{g['anos_experiencia_total'].median():.0f}a",
        "% USD": f"{g['cobra_en_dolares'].eq('True').mean()*100:.0f}%",
        "% Remoto": f"{g['modalidad'].eq('100% remoto').mean()*100:.0f}%",
        "Rol típico": g["rol"].mode().iloc[0] if len(g) else "—",
        "Empresa típica": g["tamano_empresa"].mode().iloc[0] if len(g) else "—",
        "Salario mediano": f"${g['salario_real_ars'].median()/1e6:.2f}M",
    })
st.dataframe(pd.DataFrame(filas).set_index("Segmento"),
             width="stretch")

# ---- ¿A qué segmento pertenezco? ----
st.divider()
st.subheader("🔎 ¿A qué segmento pertenecés?")
with st.form("perfil_cluster"):
    a, b, c = st.columns(3)
    edad = a.number_input("Edad", 18, 75, 30)
    exp = b.number_input("Años de experiencia", 0, 50, 5)
    emp = c.number_input("Años en la empresa", 0, 50, 2)
    d, e, f = st.columns(3)
    modalidad = d.selectbox("Modalidad", sorted(df["modalidad"].unique()))
    cobra = e.radio("¿Cobra en USD?", ["No", "Sí"], horizontal=True) == "Sí"
    tam = f.selectbox("Tamaño de empresa", sorted(df["tamano_empresa"].unique()))
    ok = st.form_submit_button("Ver mi segmento", type="primary")

if ok:
    fila = pd.DataFrame([{
        "edad": edad, "anos_experiencia_total": exp, "anos_empresa_actual": emp,
        "modalidad": modalidad, "cobra_en_dolares": str(cobra),
        "tamano_empresa": tam,
    }])
    cl = int(km.predict(pre.transform(fila))[0])
    g = df[df["cluster"] == cl]
    st.success(f"Pertenecés al **Segmento {cl}** "
               f"({len(g)/len(df)*100:.0f}% del mercado).")
    m1, m2, m3 = st.columns(3)
    m1.metric("Salario mediano del segmento", f"${g['salario_real_ars'].median()/1e6:.2f}M")
    m2.metric("Rol típico", g["rol"].mode().iloc[0])
    m3.metric("Experiencia típica", f"{g['anos_experiencia_total'].median():.0f} años")

st.caption("Clustering = aprendizaje **no supervisado**: complementa al modelo "
           "predictivo (Random Forest) describiendo la estructura del mercado.")
