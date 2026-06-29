"""
app/main.py
===========

Portada de la app web del TPO: estima salarios tech y explora el mercado
laboral argentino. Es la página de inicio de una app multipágina de Streamlit
(las secciones viven en app/pages/).

Levantar:
    .venv/bin/streamlit run app/main.py
    # abre http://localhost:8501
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "processed" / "dataset_final_mercado_laboral.parquet"

st.set_page_config(page_title="Salarios Tech AR", page_icon="💰",
                   layout="centered")


@st.cache_data(show_spinner=False)
def cargar() -> pd.DataFrame | None:
    if not DATASET.exists():
        return None
    return pd.read_parquet(DATASET)


st.title("💰 Mercado Laboral Tech 🇦🇷")
st.markdown(
    "Análisis y predicción de salarios del sector tecnológico argentino, "
    "a partir de **6 ediciones de la encuesta de Sysarmy (2022–2025)**, "
    "ajustados por inflación a **pesos reales de mayo 2026**."
)

df = cargar()
if df is None:
    st.warning("Falta el dataset. Corré `python notebooks/limpiar_y_unificar_datos.py`.")
else:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Profesionales", f"{len(df):,}")
    k2.metric("Salario mediano", f"${df['salario_real_ars'].median()/1e6:.2f}M")
    k3.metric("Cobran en USD", f"{df['cobra_en_dolares'].mean()*100:.0f}%")
    k4.metric("Provincias", df["provincia"].nunique())

st.divider()
st.subheader("¿Qué podés hacer acá?")

st.page_link("pages/01_eda.py", label="**📊 Análisis exploratorio** — "
             "explorá salarios por rol, provincia, tecnología y en el tiempo")
st.page_link("pages/02_predict.py", label="**💰 Estimador de salario** — "
             "ingresá un perfil y predecí su sueldo con Random Forest")
st.page_link("pages/03_clusters.py", label="**🧩 Segmentos del mercado** — "
             "descubrí los arquetipos de profesionales (clustering)")

st.caption("Usá el menú de la izquierda o los enlaces de arriba para navegar.")

with st.sidebar:
    st.header("ℹ️ Sobre el proyecto")
    st.markdown(
        "**TPO — Mercado Laboral Tech**\n\n"
        "- **Datos:** Sysarmy 2022–2025 + macro (IPC, dólar MEP, US CPI…)\n"
        "- **Salarios:** brutos, reales (pesos de mayo 2026)\n"
        "- **Modelo:** Random Forest (supervisado)\n"
        "- **Segmentación:** K-Means (no supervisado)\n"
    )
