"""
app/main.py
===========

Portada y navegación de la app del TPO. Usa st.navigation para que las
secciones tengan nombres propios y atractivos en el menú lateral.

Levantar:
    .venv/bin/streamlit run app/main.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "processed" / "dataset_final_mercado_laboral.parquet"

st.set_page_config(page_title="Salarios Tech AR", page_icon="💰", layout="wide")


@st.cache_data(show_spinner=False)
def cargar(ver: float) -> pd.DataFrame | None:
    return pd.read_parquet(DATASET) if DATASET.exists() else None


def inicio() -> None:
    """Página de portada."""
    st.title("💰 Mercado Laboral Tech 🇦🇷")
    st.markdown(
        "Análisis y predicción de salarios del sector tecnológico argentino, a "
        "partir de **6 ediciones de la encuesta de Sysarmy (2022–2025)**, "
        "ajustados por inflación a **valores reales de mayo 2026**."
    )
    df = cargar(DATASET.stat().st_mtime if DATASET.exists() else 0.0)
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
    st.markdown(
        "- **📊 Análisis del mercado** — explorá salarios por rol, provincia, "
        "tecnología y su evolución en el tiempo.\n"
        "- **💰 Estimador de sueldo** — ingresá un perfil y predecí su salario con "
        "Random Forest, comparándolo con perfiles similares.\n"
        "- **🧩 Perfiles del mercado** — descubrí los arquetipos de profesionales "
        "(segmentación K-Means).\n"
        "- **🌍 Argentina vs el mundo** — cómo se compara el salario local con el "
        "internacional."
    )
    st.caption("Usá el menú de la izquierda para navegar.")

    with st.sidebar:
        st.caption("TPO — Minería de Datos · Sysarmy 2022–2025 · valores reales de mayo 2026")


# ---- Navegación con nombres propios ----
paginas = [
    st.Page(inicio, title="Inicio", icon="🏠", default=True),
    st.Page("pages/01_eda.py", title="Análisis del mercado", icon="📊"),
    st.Page("pages/02_predict.py", title="Estimador de sueldo", icon="💰"),
    st.Page("pages/03_clusters.py", title="Perfiles del mercado", icon="🧩"),
    st.Page("pages/04_mundo.py", title="Argentina vs el mundo", icon="🌍"),
]
st.navigation(paginas).run()
