"""
app/pages/04_mundo.py
=====================

Comparación del salario de developers de Argentina contra el resto del mundo.

Para Argentina se usa NUESTRO dato (Sysarmy, salario real en USD), que es más
representativo del mercado local; para el resto de los países, la encuesta
global de Stack Overflow. Incluye comparación general por país y por rol.
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

PAISES = {
    "United States of America": "EE.UU.", "Germany": "Alemania", "Canada": "Canadá",
    "United Kingdom of Great Britain and Northern Ireland": "Reino Unido",
    "Netherlands": "Países Bajos", "France": "Francia", "Spain": "España",
    "Poland": "Polonia", "Brazil": "Brasil", "India": "India", "Ukraine": "Ucrania",
    "Australia": "Australia", "Italy": "Italia", "Mexico": "México",
    "Chile": "Chile", "Uruguay": "Uruguay", "Colombia": "Colombia", "Portugal": "Portugal",
}

# mapeo de NUESTROS roles -> DevType de Stack Overflow (por subcadena)
ROL_MAP = {
    "Developer": ["developer, full-stack", "developer, back-end", "developer, front-end",
                  "developer, desktop", "developer, mobile", "developer, embedded"],
    "Data Engineer": ["data engineer"],
    "Data Scientist": ["data scientist or machine learning", "academic researcher"],
    "Data Analyst": ["data or business analyst"],
    "Architect": ["architect, software or solutions"],
    "QA": ["developer, qa or test"],
    "Infosec": ["security professional"],
    "UX/UI Designer": ["designer"],
    "Project Manager": ["project manager"],
    "Manager / Director": ["engineering manager", "senior executive"],
    "DevOps": ["devops specialist", "system administrator"],
}


@st.cache_data(show_spinner="Cargando datos…")
def cargar():
    so = pd.read_csv(SO_CSV, usecols=["Country", "DevType", "ConvertedCompYearly"],
                     low_memory=False)
    so = so[so["ConvertedCompYearly"].notna() & (so["ConvertedCompYearly"] > 0)].copy()
    so["mensual"] = so["ConvertedCompYearly"] / 12
    so["DevType"] = so["DevType"].fillna("").str.lower()
    df = pd.read_parquet(DATASET, columns=["salario_real_usd", "rol"])
    return so, df


try:
    so, sysd = cargar()
except FileNotFoundError:
    st.error("No encuentro data/raw/datosInternacionales.csv")
    st.stop()

sys_ar = float(sysd["salario_real_usd"].median())

st.title("🌍 Argentina vs el mundo")
st.caption("Salario mediano de developers (USD/mes). **Argentina = nuestro dato (Sysarmy)**; "
           "resto de países = encuesta global de Stack Overflow. Valores nominales, sin "
           "ajustar por costo de vida.")

# ============================ 1) POR PAÍS ============================
st.subheader("Comparación general por país")
conteo = so["Country"].value_counts()
disp = [p for p in PAISES if p in conteo.index and conteo[p] >= 30]
default = ["United States of America", "Germany", "Canada", "Spain", "Poland",
           "Brazil", "India", "Ukraine"]
sel = st.multiselect("Países a comparar (Argentina siempre incluida):",
                     [PAISES[p] for p in disp],
                     default=[PAISES[p] for p in default if p in disp])
inv = {v: k for k, v in PAISES.items()}
med = {PAISES[p]: so[so["Country"] == p]["mensual"].median() for p in {inv[s] for s in sel}}
med["Argentina"] = sys_ar                                    # <-- nuestro dato
med = pd.Series(med).sort_values()

c1, c2 = st.columns([3, 1])
with c1:
    fig, ax = plt.subplots(figsize=(9, max(3, len(med) * 0.5)))
    ax.barh(med.index, med.values,
            color=[AMBAR if p == "Argentina" else GRAY for p in med.index],
            edgecolor=BLACK, linewidth=1)
    for i, (p, v) in enumerate(med.items()):
        ax.text(v + med.max() * 0.01, i, f"USD {v:,.0f}", va="center",
                fontsize=9, fontweight="bold")
    ax.set_xlabel("Salario mediano (USD / mes)"); ax.set_xlim(0, med.max() * 1.25)
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig); plt.close(fig)
with c2:
    st.metric("Argentina (Sysarmy)", f"USD {sys_ar:,.0f}")
    if len(med) > 1:
        st.metric(f"Techo ({med.idxmax()})", f"USD {med.max():,.0f}",
                  delta=f"{med.max()/sys_ar:.1f}× Argentina")

# ============================ 2) POR ROL ============================
st.divider()
st.subheader("Comparación por rol")
rol = st.selectbox("Rol", list(ROL_MAP.keys()))
patrones = ROL_MAP[rol]
mask_rol = so["DevType"].apply(lambda d: any(p in d for p in patrones))
ar_rol = sysd[sysd["rol"] == rol]["salario_real_usd"].median()
mundo_rol = so[mask_rol]["mensual"].median()
n_ar = int((sysd["rol"] == rol).sum()); n_mundo = int(mask_rol.sum())

if pd.isna(ar_rol) or n_ar < 10:
    st.info(f"Pocos casos de «{rol}» en Sysarmy para comparar.")
else:
    d1, d2, d3 = st.columns(3)
    d1.metric(f"{rol} — Argentina (Sysarmy)", f"USD {ar_rol:,.0f}", help=f"{n_ar} casos")
    d2.metric(f"{rol} — Mundo (Stack Overflow)",
              f"USD {mundo_rol:,.0f}" if not pd.isna(mundo_rol) else "—",
              help=f"{n_mundo} casos")
    if not pd.isna(mundo_rol):
        d3.metric("Brecha", f"{mundo_rol/ar_rol:.1f}×",
                  delta=f"{(mundo_rol-ar_rol):,.0f} USD", delta_color="inverse")
    fig, ax = plt.subplots(figsize=(8, 2.2))
    barras = {"Argentina\n(Sysarmy)": ar_rol, "Mundo\n(Stack Overflow)": mundo_rol}
    ax.barh(list(barras.keys()), list(barras.values()),
            color=[AMBAR, GRAY], edgecolor=BLACK)
    for i, v in enumerate(barras.values()):
        if not pd.isna(v):
            ax.text(v * 1.01, i, f"USD {v:,.0f}", va="center", fontweight="bold")
    ax.set_xlabel("Salario mediano (USD / mes)"); ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig); plt.close(fig)

st.warning(
    "**Limitaciones:** (1) para Argentina usamos Sysarmy y para el resto Stack Overflow — "
    "distintas encuestas y metodologías; (2) son USD nominales, no ajustados por costo de "
    "vida; (3) el mapeo de roles entre ambas encuestas es aproximado. Replicar el modelo "
    "predictivo a nivel internacional es una línea de trabajo futura."
)
