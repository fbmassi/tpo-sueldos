import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Inflación", page_icon="📈", layout="wide")

st.title("📈 Inflación y Poder Adquisitivo")
st.caption("Evolución del IPC Nacional 2020-2026 y su impacto en los salarios tech")

@st.cache_data
def cargar_datos():
    df = pd.read_csv('data/processed/dataset_final.csv')
    indec = pd.read_csv('data/processed/indec_limpio.csv', dtype={'periodo': str})
    return df, indec

df, indec = cargar_datos()

# Calcular métricas
ipc_2020 = indec[indec['periodo'] == '202001']['indice_ipc'].values[0]
ipc_2025 = indec[indec['periodo'] == '202508']['indice_ipc'].values[0]
inflacion_acumulada = ((ipc_2025 / ipc_2020) - 1) * 100
poder_adquisitivo = (ipc_2020 / ipc_2025) * 100

st.divider()

# --- MÉTRICAS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("IPC Enero 2020", f"{ipc_2020:,.2f}")
with col2:
    st.metric("IPC Agosto 2025", f"{ipc_2025:,.2f}")
with col3:
    st.metric("Inflación acumulada 2020-2025", f"{inflacion_acumulada:,.1f}%", delta=None)

st.divider()

# --- GRÁFICO 1: EVOLUCIÓN IPC ---
st.subheader("Evolución del IPC Nacional 2020-2026")

tickvals = indec[indec['periodo'].str[4:] == '01']['periodo'].tolist()
ticktext = indec[indec['periodo'].str[4:] == '01']['periodo'].str[:4].tolist()

fig1 = px.line(
    indec, x='periodo', y='indice_ipc',
    labels={'periodo': 'Año', 'indice_ipc': 'Índice IPC'},
    color_discrete_sequence=['#dc2626']
)
fig1.update_layout(xaxis=dict(tickvals=tickvals, ticktext=ticktext))
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# --- GRÁFICO 2: VARIACIÓN MENSUAL ---
st.subheader("Variación Mensual del IPC (%)")
fig2 = px.bar(
    indec, x='periodo', y='var_mensual',
    labels={'periodo': 'Período', 'var_mensual': 'Variación mensual (%)'},
    color='var_mensual',
    color_continuous_scale='Reds'
)
fig2.update_layout(
    xaxis=dict(tickvals=tickvals, ticktext=ticktext),
    coloraxis_showscale=False
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- IMPACTO EN SALARIO REAL ---
st.subheader("¿Cuánto vale hoy un sueldo de 2020?")

salario_2020 = st.slider(
    "Seleccioná un salario mensual en pesos de enero 2020:",
    min_value=50000,
    max_value=500000,
    value=100000,
    step=10000,
    format="$%d"
)

salario_equivalente_hoy = salario_2020 * (ipc_2025 / ipc_2020)
poder_hoy = (salario_2020 / salario_equivalente_hoy) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Salario en enero 2020", f"${salario_2020:,.0f}")
with col2:
    st.metric("Equivalente en agosto 2025", f"${salario_equivalente_hoy:,.0f}")
with col3:
    st.metric("Poder adquisitivo real hoy", f"{poder_hoy:.1f}%")

st.info(f"💡 Un sueldo de **${salario_2020:,.0f}** en enero 2020 equivale a **${salario_equivalente_hoy:,.0f}** en agosto 2025 para mantener el mismo poder de compra. Si tu sueldo no llegó a ese número, perdiste poder adquisitivo.")