import streamlit as st

st.set_page_config(
    page_title="Mercado Laboral Tech Argentina",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Mercado Laboral Tech en Argentina")
st.subheader("Análisis de salarios, inflación y comparativa regional — 2025")

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Profesionales analizados", "3.676")
with col2:
    st.metric("Salario mediano", "USD 2.161/mes")
with col3:
    st.metric("Inflación acumulada 2020-2025", "3.072%")
with col4:
    st.metric("Brecha vs LATAM (QA)", "-56%")

st.divider()

st.markdown("""
## 🎯 Hipótesis
> ¿Qué factores determinan el salario de un profesional tech en Argentina 
> y cómo evolucionó el poder adquisitivo real frente a la inflación?

## 📦 Fuentes de datos
| Dataset | Origen | Registros |
|---|---|---|
| Encuesta de sueldos | Sysarmy 2025.2 | 3.676 respuestas |
| Índice de precios | INDEC IPC 2020-2026 | 76 períodos |
| Comparativa global | Stack Overflow Survey 2025 | 49.191 respuestas |

## 👥 Equipo
| Rol | Responsabilidad |
|---|---|
| Product Owner | Hipótesis y conclusiones |
| Data Engineer | Pipeline y ETL |
| Data Quality | Limpieza de datos |
| Data Analyst | Análisis exploratorio |
| Data Scientist | Modelos de minería |
| Visualización | App y dashboards |
""")