"""
app_en/main.py
==============

English / USD version of the salary estimator web app.

Same Random Forest model as the Spanish app, but the target is
`salario_real_usd` (real US dollars, May-2026 constant) and the whole UI is in
English. Lives in its own folder so it does not inherit the Spanish pages/.

Run:
    .venv/bin/streamlit run app_en/main.py
    # opens http://localhost:8501
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import predictor_cli as pc  # noqa: E402

st.set_page_config(page_title="Argentine Tech Salary Estimator",
                   page_icon="💵", layout="centered")


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

# Display labels in English for a few coded fields (data values stay as-is)
MODALITY_EN = {"100% remoto": "100% remote", "100% presencial": "100% on-site",
               "híbrido": "Hybrid"}
GENDER_EN = {"masculino": "male", "femenino": "female",
             "otro / no especifica": "other / N.A."}

st.title("💵 Argentine Tech Salary Estimator")
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
    company = c7.selectbox("Company size", options["tamano_empresa"])

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
        "(they were a proxy of context and added noise)\n\n"
        "🇦🇷 Spanish / ARS version: `streamlit run app/main.py`"
    )
