"""
app/pages/03_clusters.py
========================

Segmentación (clustering no supervisado) de profesionales tech. Agrupa perfiles
SIN usar el salario (K-Means) y luego muestra cuánto gana cada segmento.
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

NUM = ["edad", "anos_experiencia_total", "anos_empresa_actual"]
CAT = ["modalidad", "cobra_en_dolares", "tamano_empresa"]
PALETA = ["#2E86C1", "#E6A100", "#8C8C8C", "#0A0A0A", "#C0392B", "#27AE60",
          "#8E44AD", "#16A085"]


@st.cache_data(show_spinner="Cargando datos…")
def cargar(ver: float) -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    df["cobra_en_dolares"] = df["cobra_en_dolares"].astype(str)
    return df


@st.cache_resource(show_spinner="Calculando segmentos…")
def entrenar(k: int, ver: float):
    df = cargar(ver)
    pre = ColumnTransformer([("n", StandardScaler(), NUM),
                             ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    Xp = pre.fit_transform(df[NUM + CAT])
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xp)
    coords = PCA(n_components=2, random_state=42).fit_transform(
        Xp.toarray() if hasattr(Xp, "toarray") else Xp)
    return pre, km, coords


def nombrar(df: pd.DataFrame) -> dict:
    """
    Nombre descriptivo de cada cluster combinando NIVEL (por experiencia) y un
    RASGO distintivo (dolarizado / corporativo / empresa grande / local). Así
    hay nombres variados y únicos aunque haya muchos segmentos.
    """
    p = df.groupby("cluster").agg(
        exp=("anos_experiencia_total", "median"),
        usd=("cobra_en_dolares", lambda s: (s == "True").mean()),
        ant=("anos_empresa_actual", "median"),
        hib=("modalidad", lambda s: (s == "híbrido").mean()),
        pres=("modalidad", lambda s: (s == "100% presencial").mean()))
    nombres, usados = {}, set()
    for c in p.index:
        e = p.loc[c, "exp"]
        nivel = ("🌱 Junior" if e < 4 else "🧑‍💻 Semi-senior" if e < 7
                 else "👨‍💻 Senior" if e <= 15 else "🎖️ Senior +15")
        if p.loc[c, "usd"] > 0.6:
            rasgo = "dolarizado"
        elif p.loc[c, "ant"] >= 10:
            rasgo = "corporativo estable"
        elif p.loc[c, "hib"] > 0.5:
            rasgo = "híbrido"
        elif p.loc[c, "pres"] > 0.4:
            rasgo = "presencial"
        else:
            rasgo = "remoto"
        n = f"{nivel} {rasgo}"
        if n in usados:               # desambiguar colisiones
            n = f"{n} #{c}"
        usados.add(n)
        nombres[c] = n
    return nombres


try:
    df = cargar(DATASET.stat().st_mtime)
except FileNotFoundError:
    st.error("No encuentro el dataset. Corré antes "
             "`python notebooks/limpiar_y_unificar_datos.py`")
    st.stop()

st.title("🧩 Perfiles del mercado (clustering)")
st.markdown(
    "Agrupamos perfiles parecidos **sin mirar el salario** (K-Means). Recién "
    "después vemos cuánto gana cada grupo: así descubrimos los **arquetipos** "
    "del mercado tech argentino."
)

k = st.slider("Cantidad de segmentos (k)", 2, 8, 4,
              help="Cuántos arquetipos separar. Los nombres se recalculan al cambiar k.")
pre, km, coords = entrenar(k, DATASET.stat().st_mtime)
df = df.copy()
df["cluster"] = km.labels_
nombres = nombrar(df)

# ---- Scatter PCA (grande, ejes legibles, con zoom) ----
st.subheader("Mapa de segmentos")
fig, ax = plt.subplots(figsize=(12, 6.2))
samp = np.random.RandomState(0).choice(len(df), min(8000, len(df)), replace=False)
for c in range(k):
    m = df["cluster"].values[samp] == c
    ax.scatter(coords[samp][m, 0], coords[samp][m, 1], c=PALETA[c % len(PALETA)],
               s=18, alpha=0.6, edgecolors="none", label=nombres[c])
x1, x2 = np.percentile(coords[:, 0], [0.5, 99])
y1, y2 = np.percentile(coords[:, 1], [1, 98.5])
ax.set_xlim(x1 - 0.4, x2 + 0.4); ax.set_ylim(y1 - 0.4, y2 + 0.6)
ax.set_xlabel("Componente 1  →  más años de carrera (junior a senior)", fontsize=13)
ax.set_ylabel("Componente 2  →  más antigüedad / rotación", fontsize=13)
ax.tick_params(labelsize=11)
ax.legend(loc="upper right", framealpha=1, fontsize=12, markerscale=2, title="Segmento")
ax.grid(alpha=0.25)
st.pyplot(fig); plt.close(fig)
st.caption("Es una proyección 2D (PCA) que muestra el 62% de la información; en el "
           "espacio completo los grupos están mejor separados.")

# ---- Salario por segmento ----
st.subheader("Salario por segmento")
salc = (df.groupby("cluster")["salario_real_ars"].median() / 1e6)
salc.index = [nombres[c] for c in salc.index]
st.bar_chart(salc.sort_values(), y_label="Salario mediano (M$)", horizontal=True)

# ---- Caracterización (con nombres) ----
st.subheader("¿Quién es quién en cada segmento?")
filas = []
for c in range(k):
    g = df[df["cluster"] == c]
    filas.append({
        "Segmento": nombres[c],
        "Del mercado": f"{len(g)/len(df)*100:.0f}%",
        "Edad": f"{g['edad'].median():.0f}",
        "Experiencia": f"{g['anos_experiencia_total'].median():.0f} años",
        "Antigüedad": f"{g['anos_empresa_actual'].median():.0f} años",
        "Cobra en USD": f"{g['cobra_en_dolares'].eq('True').mean()*100:.0f}%",
        "Remoto": f"{g['modalidad'].eq('100% remoto').mean()*100:.0f}%",
        "Híbrido": f"{g['modalidad'].eq('híbrido').mean()*100:.0f}%",
        "Salario mediano": f"${g['salario_real_ars'].median()/1e6:.2f}M",
    })
tabla = pd.DataFrame(filas).set_index("Segmento")
st.dataframe(tabla.sort_values("Salario mediano", ascending=False), width="stretch")

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
        "modalidad": modalidad, "cobra_en_dolares": str(cobra), "tamano_empresa": tam,
    }])
    cl = int(km.predict(pre.transform(fila))[0])
    g = df[df["cluster"] == cl]
    st.success(f"Pertenecés al segmento **{nombres[cl]}** "
               f"({len(g)/len(df)*100:.0f}% del mercado).")
    m1, m2, m3 = st.columns(3)
    m1.metric("Salario mediano del segmento", f"${g['salario_real_ars'].median()/1e6:.2f}M")
    m2.metric("Modalidad típica", g["modalidad"].mode().iloc[0])
    m3.metric("Experiencia típica", f"{g['anos_experiencia_total'].median():.0f} años")

st.caption("Clustering = aprendizaje **no supervisado**: complementa al modelo "
           "predictivo (Random Forest) describiendo la estructura del mercado.")
