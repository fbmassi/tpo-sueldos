"""
app/main.py
===========

App web (Streamlit) para estimar el salario de un profesional tech a partir de
su perfil, usando el modelo Random Forest entrenado sobre el dataset de Sysarmy.

Reutiliza la lógica de notebooks/predictor_cli.py (mismas features que el modelo
oficial: SIN seniority, RF regularizado), así que hay una sola fuente de verdad.

Levantar:
    .venv/bin/streamlit run app/main.py
    # abre http://localhost:8501
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permitir importar los ejecutables que viven en notebooks/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import predictor_cli as pc  # noqa: E402  (entrenamiento + features compartidas)

# ----------------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Estimador de Salario Tech AR",
                   page_icon="💰", layout="centered")


@st.cache_resource(show_spinner="Entrenando el modelo (sólo la primera vez)…")
def cargar_modelo():
    """Entrena el Random Forest una sola vez y lo cachea entre interacciones."""
    return pc.preparar_datos()  # (X, y, cols_tech, opciones, (pre, model))


# ----------------------------------------------------------------------------
# Carga del modelo
# ----------------------------------------------------------------------------
try:
    X, y, cols_tech, opciones, pre_model = cargar_modelo()
except FileNotFoundError:
    st.error("No encuentro el dataset. Corré antes:\n\n"
             "`python notebooks/limpiar_y_unificar_datos.py`")
    st.stop()

# nombre de tecnología -> columna binaria (mismo orden que en preparar_datos)
tech_map = dict(zip(opciones["tecnologias"], cols_tech))

# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
st.title("💰 Estimador de Salario Tech 🇦🇷")
st.caption(
    f"Random Forest entrenado con {len(X):,} respuestas de Sysarmy (2022–2025). "
    "Estima el salario **bruto** en **pesos reales de mayo 2026**."
)

# ----------------------------------------------------------------------------
# Formulario de perfil
# ----------------------------------------------------------------------------
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
                                    use_container_width=True)

# ----------------------------------------------------------------------------
# Predicción y resultado
# ----------------------------------------------------------------------------
if enviado:
    # construir la fila con el mismo esquema con que se entrenó
    fila = pd.DataFrame([{
        "edad": edad,
        "anos_experiencia_total": exp,
        "anos_empresa_actual": emp,
        "provincia": prov,
        "genero": genero,
        "modalidad": modalidad,
        "tamano_empresa": tam,
        "rol": rol,
        "cobra_en_dolares": str(cobra),
    }])
    for nombre, col in tech_map.items():
        fila[col] = int(nombre in techs)

    pre, model = pre_model
    pred = float(model.predict(pre.transform(fila))[0])

    st.divider()
    st.metric("Salario bruto estimado", f"$ {pred/1e6:.2f} M / mes",
              help="Pesos reales de mayo 2026 (ajustado por inflación)")

    # USD al MEP base (mayo 2026 ≈ 1.409) para dar referencia en dólares
    usd = pred / 1408.57
    st.caption(f"≈ USD {usd:,.0f} / mes (al dólar de mayo 2026)")

    # posición en la distribución del mercado
    p25, p50, p75 = y.quantile([0.25, 0.50, 0.75])
    if pred < p25:
        pos = "🔵 Por debajo del cuartil bajo del mercado (P25)"
    elif pred > p75:
        pos = "🟢 Por encima del cuartil alto del mercado (P75)"
    else:
        pos = "🟡 En el rango medio del mercado"
    st.write(f"**Posición de mercado:** {pos}")

    d1, d2, d3 = st.columns(3)
    d1.metric("P25 mercado", f"${p25/1e6:.2f}M")
    d2.metric("Mediana", f"${p50/1e6:.2f}M")
    d3.metric("P75 mercado", f"${p75/1e6:.2f}M")

    # comparación con profesionales similares (mismo rol + provincia)
    mask = (X["rol"] == rol) & (X["provincia"] == prov)
    sim = y[mask]
    if len(sim) >= 5:
        st.info(
            f"👥 **{len(sim):,} profesionales** con el mismo rol en {prov} ganan, "
            f"en mediana, **${sim.median()/1e6:.2f}M** "
            f"(rango intercuartil ${sim.quantile(.25)/1e6:.2f}M – "
            f"${sim.quantile(.75)/1e6:.2f}M)."
        )
    else:
        st.caption("Pocos casos con ese rol y provincia para comparar.")

    st.caption(
        "⚠️ Estimación orientativa. El modelo explica ~30% de la variación "
        "salarial (R²≈0.30); el resto depende de la empresa, la negociación y "
        "factores no capturados en la encuesta."
    )

# ----------------------------------------------------------------------------
# Barra lateral informativa
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Sobre el modelo")
    st.markdown(
        "- **Algoritmo:** Random Forest (100 árboles, regularizado)\n"
        "- **Datos:** 6 ediciones de Sysarmy (2022.2 → 2025.2)\n"
        "- **Target:** salario bruto real (pesos de mayo 2026)\n"
        "- **Features:** edad, experiencia, antigüedad, rol, provincia, "
        "modalidad, tamaño de empresa, género, cobra en USD y 20 tecnologías\n"
        "- *Sin* seniority (es redundante con los años de experiencia)"
    )
