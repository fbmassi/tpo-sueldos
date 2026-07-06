"""
app_en/pages/02_predict.py
==========================

Salary estimator (Random Forest) in English / real USD. Reuses the logic in
notebooks/predictor_cli.py — same features as the official model, single
source of truth. Mirrors app/pages/02_predict.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "app_en"))

import predictor_cli as pc  # noqa: E402
from i18n import GENDER_EN, MODALITY_EN, size_en  # noqa: E402


@st.cache_resource(show_spinner="Training the model (first run only)…")
def load_model(ver: float):
    # `ver` = dataset mtime: invalidates the cache when the dataset is rebuilt.
    return pc.preparar_datos(target="salario_real_usd")


try:
    X, y, cols_tech, options, pre_model = load_model(pc.DATASET.stat().st_mtime)
except FileNotFoundError:
    st.error("Dataset not found. Run "
             "`python notebooks/limpiar_y_unificar_datos.py` first.")
    st.stop()

SO_MUNDO = ROOT / "data" / "processed" / "stackoverflow_mundo.parquet"
ROL_MAP = {
    "Developer": ["developer, full-stack", "developer, back-end", "developer, front-end",
                  "developer, desktop", "developer, mobile", "developer, embedded"],
    "Data Engineer": ["data engineer"], "Data Scientist": ["data scientist or machine learning"],
    "Data Analyst": ["data or business analyst"], "Architect": ["architect, software or solutions"],
    "QA": ["developer, qa or test"], "Infosec": ["security professional"],
    "UX/UI Designer": ["designer"], "Project Manager": ["project manager"],
    "Manager / Director": ["engineering manager", "senior executive"],
}


@st.cache_data(show_spinner=False)
def load_world():
    try:
        so = pd.read_parquet(SO_MUNDO, columns=["DevType", "ConvertedCompYearly"])
    except (FileNotFoundError, ValueError):
        return None
    so = so[so["ConvertedCompYearly"].notna() & (so["ConvertedCompYearly"] > 0)].copy()
    so["monthly"] = so["ConvertedCompYearly"] / 12
    so["DevType"] = so["DevType"].fillna("").str.lower()
    return so


so_world = load_world()

st.title("💵 Salary estimator")
st.caption(
    f"Random Forest trained on {len(X):,} Sysarmy survey responses (2022–2025). "
    "Estimates the **gross monthly salary in real US dollars (May 2026)**."
)

with st.form("profile"):
    st.subheader("Professional profile")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 18, 75, 30)
    exp = c2.number_input("Years of experience", 0, 50, 5)
    tenure = c3.number_input("Years at current company", 0, 50, 2)

    c4, c5 = st.columns(2)
    role = c4.selectbox("Role", options["rol"])
    province = c5.selectbox("Province", options["provincia"])

    c6, c7 = st.columns(2)
    modality = c6.selectbox("Work mode", options["modalidad"],
                            format_func=lambda v: MODALITY_EN.get(v, v))
    company = c7.selectbox("Company size", options["tamano_empresa"],
                           format_func=size_en)

    c8, c9 = st.columns([2, 1])
    gender = c8.selectbox("Gender", options["genero"],
                          format_func=lambda v: GENDER_EN.get(v, v))
    paid_usd = c9.radio("Paid in USD?", ["No", "Yes"], horizontal=True) == "Yes"

    submitted = st.form_submit_button("Estimate salary", type="primary",
                                      width="stretch")

if submitted:
    row = pd.DataFrame([{
        "edad": age, "anos_experiencia_total": exp, "anos_empresa_actual": tenure,
        "provincia": province, "genero": gender, "modalidad": modality,
        "tamano_empresa": company, "rol": role, "cobra_en_dolares": str(paid_usd),
    }])

    pre, model = pre_model
    pred = float(model.predict(pre.transform(row))[0])

    st.divider()
    st.metric("Estimated gross salary", f"USD {pred:,.0f} / month",
              help="Real US dollars (May 2026)")

    p25, p50, p75 = y.quantile([0.25, 0.50, 0.75])
    pos = ("🔵 Below the market's lower quartile (P25)" if pred < p25
           else "🟢 Above the market's upper quartile (P75)" if pred > p75
           else "🟡 Within the market's mid-range")
    st.write(f"**Market position:** {pos}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Market P25", f"USD {p25:,.0f}")
    d2.metric("Median", f"USD {p50:,.0f}")
    d3.metric("Market P75", f"USD {p75:,.0f}")

    mask = (X["rol"] == role) & (X["provincia"] == province)
    sim = y[mask]
    if len(sim) >= 5:
        st.info(f"👥 **{len(sim):,} professionals** with the same role in "
                f"{province} earn a median of **USD {sim.median():,.0f}**.")

    # ---- Same role around the world (Stack Overflow) ----
    patterns = ROL_MAP.get(role, [])
    if so_world is not None and patterns:
        mask_r = so_world["DevType"].apply(lambda d: any(p in d for p in patterns))
        world_usd = so_world[mask_r]["monthly"].median()
        if not pd.isna(world_usd):
            st.divider()
            st.markdown(f"**🌍 “{role}” around the world** (Stack Overflow, USD):")
            w1, w2, w3 = st.columns(3)
            w1.metric("Your profile (Argentina)", f"USD {pred:,.0f}")
            w2.metric("Worldwide", f"USD {world_usd:,.0f}")
            w3.metric("Gap", f"{world_usd/pred:.1f}×",
                      delta=f"{world_usd-pred:,.0f} USD", delta_color="off")
            st.caption("The world usually pays more: different survey, and not "
                       "adjusted for cost of living.")

    st.caption("⚠️ Indicative estimate (R²≈0.27): the rest depends on the "
               "company, negotiation and factors not captured by the survey.")

with st.sidebar:
    st.header("ℹ️ About the model")
    st.markdown(
        "- **Algorithm:** Random Forest (100 trees, regularized)\n"
        "- **Data:** 6 Sysarmy editions (2022.2 → 2025.2)\n"
        "- **Target:** gross salary in **real USD** (May 2026)\n"
        "- **Features:** age, experience, tenure, role, province, work mode, "
        "company size, gender, paid-in-USD\n"
        "- *No* seniority (redundant with experience) and *no* technologies "
        "(they were a proxy of context and added noise)"
    )
