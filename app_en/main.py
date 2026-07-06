"""
app_en/main.py
==============

English / USD version of the TPO app — same structure as the Spanish app
(home + market analysis + estimator + market profiles + world comparison),
with every salary expressed in real US dollars (May-2026 constant).

Run:
    .venv/bin/streamlit run app_en/main.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "processed" / "dataset_final_mercado_laboral.parquet"

st.set_page_config(page_title="Argentine Tech Salaries", page_icon="💵", layout="wide")


@st.cache_data(show_spinner=False)
def load(ver: float) -> pd.DataFrame | None:
    return pd.read_parquet(DATASET) if DATASET.exists() else None


def home() -> None:
    """Landing page."""
    st.title("💵 Argentine Tech Labor Market 🇦🇷")
    st.markdown(
        "Analysis and prediction of tech-sector salaries in Argentina, based on "
        "**6 editions of the Sysarmy salary survey (2022–2025)**, inflation-adjusted "
        "to **real May-2026 US dollars**."
    )
    df = load(DATASET.stat().st_mtime if DATASET.exists() else 0.0)
    if df is None:
        st.warning("Dataset not found. Run `python notebooks/limpiar_y_unificar_datos.py`.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Professionals", f"{len(df):,}")
        k2.metric("Median salary", f"USD {df['salario_real_usd'].median():,.0f}")
        k3.metric("Paid in USD", f"{df['cobra_en_dolares'].astype(str).eq('True').mean()*100:.0f}%")
        k4.metric("Provinces", df["provincia"].nunique())

    st.divider()
    st.subheader("What can you do here?")
    st.markdown(
        "- **📊 Market analysis** — explore salaries by role, province and "
        "technology.\n"
        "- **💵 Salary estimator** — enter a profile and predict its salary with "
        "Random Forest, compared against similar professionals.\n"
        "- **🧩 Market profiles** — discover the professional archetypes "
        "(K-Means segmentation).\n"
        "- **🌍 Argentina vs the world** — how the local salary compares "
        "internationally."
    )
    st.caption("Use the left-side menu to navigate.")

    with st.sidebar:
        st.caption("Data Mining TPO · Sysarmy 2022–2025 · real May-2026 values")


pages = [
    st.Page(home, title="Home", icon="🏠", default=True),
    st.Page("pages/01_eda.py", title="Market analysis", icon="📊"),
    st.Page("pages/02_predict.py", title="Salary estimator", icon="💵"),
    st.Page("pages/03_clusters.py", title="Market profiles", icon="🧩"),
    st.Page("pages/04_world.py", title="Argentina vs the world", icon="🌍"),
]
st.navigation(pages).run()
