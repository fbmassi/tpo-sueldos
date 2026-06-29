"""
app/pages/02_predict.py
=======================

Estimador de salario (Random Forest). Reutiliza la lógica de
notebooks/predictor_cli.py: mismas features que el modelo oficial (SIN
seniority, RF regularizado), una sola fuente de verdad.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import predictor_cli as pc  # noqa: E402

st.set_page_config(page_title="Estimador — Salarios Tech", page_icon="💰",
                   layout="centered")


@st.cache_resource(show_spinner="Entrenando el modelo (sólo la primera vez)…")
def cargar_modelo():
    return pc.preparar_datos()  # (X, y, cols_tech, opciones, (pre, model))


try:
    X, y, cols_tech, opciones, pre_model = cargar_modelo()
except FileNotFoundError:
    st.error("No encuentro el dataset. Corré antes "
             "`python notebooks/limpiar_y_unificar_datos.py`")
    st.stop()

tech_map = dict(zip(opciones["tecnologias"], cols_tech))

st.title("💰 Estimador de salario")
st.caption(
    f"Random Forest entrenado con {len(X):,} respuestas de Sysarmy (2022–2025). "
    "Estima el salario **bruto** en **pesos reales de mayo 2026**."
)

with st.form("perfil"):
    st.subheader("Perfil del profesional")
    c1, c2, c3 = st.columns(3)
    edad = c1.number_input("Edad", 18, 75, 30)
    exp = c2.number_input("Años de experiencia", 0, 50, 5)
    emp = c3.number_input("Años en la empresa", 0, 50, 2)

    c4, c5 = st.columns(2)
    rol = c4.selectbox("Rol / puesto", opciones["rol"])
    prov = c5.selectbox("Provincia", opciones["provincia"])

    c6, c7 = st.columns(2)
    modalidad = c6.selectbox("Modalidad", opciones["modalidad"])
    tam = c7.selectbox("Tamaño de empresa", opciones["tamano_empresa"])

    c8, c9 = st.columns([2, 1])
    genero = c8.selectbox("Género", opciones["genero"])
    cobra = c9.radio("¿Cobra en USD?", ["No", "Sí"], horizontal=True) == "Sí"

    techs = st.multiselect("Tecnologías que usás", opciones["tecnologias"])
    enviado = st.form_submit_button("Estimar salario", type="primary",
                                    width="stretch")

if enviado:
    fila = pd.DataFrame([{
        "edad": edad, "anos_experiencia_total": exp, "anos_empresa_actual": emp,
        "provincia": prov, "genero": genero, "modalidad": modalidad,
        "tamano_empresa": tam, "rol": rol, "cobra_en_dolares": str(cobra),
    }])
    for nombre, col in tech_map.items():
        fila[col] = int(nombre in techs)

    pre, model = pre_model
    pred = float(model.predict(pre.transform(fila))[0])

    st.divider()
    st.metric("Salario bruto estimado", f"$ {pred/1e6:.2f} M / mes",
              help="Pesos reales de mayo 2026")
    st.caption(f"≈ USD {pred/1408.57:,.0f} / mes (al dólar de mayo 2026)")

    p25, p50, p75 = y.quantile([0.25, 0.50, 0.75])
    pos = ("🔵 Por debajo del P25" if pred < p25
           else "🟢 Por encima del P75" if pred > p75
           else "🟡 En el rango medio del mercado")
    st.write(f"**Posición de mercado:** {pos}")
    d1, d2, d3 = st.columns(3)
    d1.metric("P25 mercado", f"${p25/1e6:.2f}M")
    d2.metric("Mediana", f"${p50/1e6:.2f}M")
    d3.metric("P75 mercado", f"${p75/1e6:.2f}M")

    mask = (X["rol"] == rol) & (X["provincia"] == prov)
    sim = y[mask]
    if len(sim) >= 5:
        st.info(f"👥 **{len(sim):,} profesionales** con el mismo rol en {prov} "
                f"ganan, en mediana, **${sim.median()/1e6:.2f}M**.")
    st.caption("⚠️ Estimación orientativa (R²≈0.30): el resto depende de la "
               "empresa, la negociación y factores no capturados.")
