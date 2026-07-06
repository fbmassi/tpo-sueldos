"""
app_en/pages/04_world.py
========================

Argentina vs the world: median developer salaries compared against other
countries. Argentina uses OUR data (Sysarmy, real USD); other countries use
the global Stack Overflow survey. Mirrors app/pages/04_mundo.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
# Slim Stack Overflow aggregate (18 countries) precomputed from the raw 140 MB
# CSV, which is not pushed to GitHub. See notebooks/preparar_mundo.py.
SO_MUNDO = ROOT / "data" / "processed" / "stackoverflow_mundo.parquet"
DATASET = ROOT / "data" / "processed" / "dataset_final_mercado_laboral.parquet"

AMBER, GRAY, BLACK = "#E6A100", "#3A3A3A", "#0A0A0A"

COUNTRIES = {
    "United States of America": "United States", "Germany": "Germany",
    "Canada": "Canada",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Netherlands": "Netherlands", "France": "France", "Spain": "Spain",
    "Poland": "Poland", "Brazil": "Brazil", "India": "India", "Ukraine": "Ukraine",
    "Australia": "Australia", "Italy": "Italy", "Mexico": "Mexico",
    "Chile": "Chile", "Uruguay": "Uruguay", "Colombia": "Colombia",
    "Portugal": "Portugal",
}

# our roles -> Stack Overflow DevType (substring match)
ROLE_MAP = {
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


@st.cache_data(show_spinner="Loading data…")
def load(ver: float):
    so = pd.read_parquet(SO_MUNDO)
    so["monthly"] = so["ConvertedCompYearly"] / 12
    so["DevType"] = so["DevType"].fillna("").str.lower()
    df = pd.read_parquet(DATASET, columns=["salario_real_usd", "rol"])
    return so, df


try:
    so, sysd = load(SO_MUNDO.stat().st_mtime)
except FileNotFoundError:
    st.error("Missing data/processed/stackoverflow_mundo.parquet")
    st.stop()

sys_ar = float(sysd["salario_real_usd"].median())

st.title("🌍 Argentina vs the world")
st.caption("Median developer salary (USD/month). **Argentina = our data (Sysarmy)**; "
           "other countries = the global Stack Overflow survey. Nominal values, not "
           "adjusted for cost of living.")

# ============================ 1) BY COUNTRY ============================
st.subheader("Overall comparison by country")
counts = so["Country"].value_counts()
avail = [p for p in COUNTRIES if p in counts.index and counts[p] >= 30]
default = ["United States of America", "Germany", "Canada", "Spain", "Poland",
           "Brazil", "India", "Ukraine"]
sel = st.multiselect("Countries to compare (Argentina always included):",
                     [COUNTRIES[p] for p in avail],
                     default=[COUNTRIES[p] for p in default if p in avail])
inv = {v: k for k, v in COUNTRIES.items()}
sel_countries = {inv[s] for s in sel}
med = {COUNTRIES[p]: so[so["Country"] == p]["monthly"].median() for p in sel_countries}
med["Argentina"] = sys_ar
med = pd.Series(med).sort_values()

c1, c2 = st.columns([3, 1])
with c1:
    fig, ax = plt.subplots(figsize=(9, max(3, len(med) * 0.5)))
    ax.barh(med.index, med.values,
            color=[AMBER if p == "Argentina" else GRAY for p in med.index],
            edgecolor=BLACK, linewidth=1)
    for i, (p, v) in enumerate(med.items()):
        ax.text(v + med.max() * 0.01, i, f"USD {v:,.0f}", va="center",
                fontsize=9, fontweight="bold")
    ax.set_xlabel("Median salary (USD / month)"); ax.set_xlim(0, med.max() * 1.25)
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig); plt.close(fig)
with c2:
    st.metric("Argentina (Sysarmy)", f"USD {sys_ar:,.0f}")
    if len(med) > 1:
        st.metric(f"Ceiling ({med.idxmax()})", f"USD {med.max():,.0f}",
                  delta=f"{med.max()/sys_ar:.1f}× Argentina")

# ============================ 2) BY ROLE ============================
st.divider()
st.subheader("Comparison by role (against the selected countries)")
role = st.selectbox("Role", list(ROLE_MAP.keys()))
patterns = ROLE_MAP[role]
mask_role = so["DevType"].apply(lambda d: any(p in d for p in patterns))

med_role = {}
for p in sel_countries:
    sub = so[(so["Country"] == p) & mask_role]
    if len(sub) >= 5:
        med_role[COUNTRIES[p]] = sub["monthly"].median()
ar_role = sysd[sysd["rol"] == role]["salario_real_usd"].median()
n_ar = int((sysd["rol"] == role).sum())

if pd.isna(ar_role) or n_ar < 10:
    st.info(f"Too few “{role}” cases in Sysarmy to compare.")
else:
    med_role["Argentina"] = ar_role
    s = pd.Series(med_role).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(2.2, len(s) * 0.5)))
    ax.barh(s.index, s.values,
            color=[AMBER if p == "Argentina" else GRAY for p in s.index],
            edgecolor=BLACK, linewidth=1)
    for i, (p, v) in enumerate(s.items()):
        ax.text(v * 1.01, i, f"USD {v:,.0f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Median salary (USD / month)"); ax.set_xlim(0, s.max() * 1.25)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"{role}: Argentina vs selected countries", fontsize=12, fontweight="bold")
    st.pyplot(fig); plt.close(fig)
    if len(s) > 1 and s.get("Argentina", 0) > 0:
        st.caption(f"The ceiling for “{role}” is {s.idxmax()} (USD {s.max():,.0f}), "
                   f"{s.max()/ar_role:.1f}× Argentina (USD {ar_role:,.0f}).")

st.warning(
    "**Limitations:** (1) Argentina uses Sysarmy while other countries use Stack "
    "Overflow — different surveys and methodologies; (2) nominal USD, not adjusted "
    "for cost of living; (3) the role mapping between both surveys is approximate. "
    "Replicating the predictive model internationally is future work."
)
