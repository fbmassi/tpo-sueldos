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
# Agregado liviano de Stack Overflow (18 países) precomputado desde el CSV crudo
# de 140 MB, que NO se sube a GitHub. Ver notebooks/preparar_mundo.py.
SO_MUNDO = ROOT / "data" / "processed" / "stackoverflow_mundo.parquet"
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
def cargar(ver: float):
    so = pd.read_parquet(SO_MUNDO)
    so["mensual"] = so["ConvertedCompYearly"] / 12
    so["DevType"] = so["DevType"].fillna("").str.lower()
    df = pd.read_parquet(DATASET, columns=["salario_real_usd", "rol"])
    return so, df


try:
    so, sysd = cargar(SO_MUNDO.stat().st_mtime)
except FileNotFoundError:
    st.error("No encuentro data/processed/stackoverflow_mundo.parquet")
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
paises_sel = {inv[s] for s in sel}                           # países elegidos (form SO)
med = {PAISES[p]: so[so["Country"] == p]["mensual"].median() for p in paises_sel}
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
st.subheader("Comparación por rol (contra los países seleccionados)")
rol = st.selectbox("Rol", list(ROL_MAP.keys()))
patrones = ROL_MAP[rol]
mask_rol = so["DevType"].apply(lambda d: any(p in d for p in patrones))

# salario de ese rol en cada país SELECCIONADO (mín. 5 casos) + Argentina (nuestro dato)
med_rol = {}
for p in paises_sel:
    sub = so[(so["Country"] == p) & mask_rol]
    if len(sub) >= 5:
        med_rol[PAISES[p]] = sub["mensual"].median()
ar_rol = sysd[sysd["rol"] == rol]["salario_real_usd"].median()
n_ar = int((sysd["rol"] == rol).sum())

if pd.isna(ar_rol) or n_ar < 10:
    st.info(f"Pocos casos de «{rol}» en Sysarmy para comparar.")
else:
    med_rol["Argentina"] = ar_rol
    s = pd.Series(med_rol).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(2.2, len(s) * 0.5)))
    ax.barh(s.index, s.values,
            color=[AMBAR if p == "Argentina" else GRAY for p in s.index],
            edgecolor=BLACK, linewidth=1)
    for i, (p, v) in enumerate(s.items()):
        ax.text(v * 1.01, i, f"USD {v:,.0f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Salario mediano (USD / mes)"); ax.set_xlim(0, s.max() * 1.25)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"{rol}: Argentina vs países seleccionados", fontsize=12, fontweight="bold")
    st.pyplot(fig); plt.close(fig)
    if len(s) > 1 and s.get("Argentina", 0) > 0:
        st.caption(f"El techo para «{rol}» es {s.idxmax()} (USD {s.max():,.0f}), "
                   f"{s.max()/ar_rol:.1f}× lo de Argentina (USD {ar_rol:,.0f}).")

st.warning(
    "**Limitaciones:** (1) para Argentina usamos Sysarmy y para el resto Stack Overflow — "
    "distintas encuestas y metodologías; (2) son USD nominales, no ajustados por costo de "
    "vida; (3) el mapeo de roles entre ambas encuestas es aproximado. Replicar el modelo "
    "predictivo a nivel internacional es una línea de trabajo futura."
)
