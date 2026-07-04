import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.title("📊 Análisis Exploratorio de Datos")
st.caption("Exploración del mercado laboral tech argentino — Sysarmy 2025.2")

# Carga de datos
@st.cache_data
def cargar_datos():
    return pd.read_csv('data/processed/dataset_final.csv')

df = cargar_datos()

st.divider()

# --- MÉTRICAS RÁPIDAS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total encuestados", f"{len(df):,}")
with col2:
    st.metric("Salario mediano USD", f"USD {df['salario_usd_mensual'].median():,.0f}")
with col3:
    st.metric("Salario promedio USD", f"USD {df['salario_usd_mensual'].mean():,.0f}")
with col4:
    st.metric("Provincias representadas", f"{df['provincia'].nunique()}")

st.divider()

# --- GRÁFICO 1: DISTRIBUCIÓN ---
st.subheader("Distribución de Salarios (USD/mes)")
fig1 = px.histogram(
    df, x='salario_usd_mensual', nbins=50,
    labels={'salario_usd_mensual': 'Salario USD/mes'},
    color_discrete_sequence=['#2563eb']
)
fig1.add_vline(x=df['salario_usd_mensual'].median(), line_dash='dash',
               line_color='red',
               annotation_text=f"Mediana: USD {df['salario_usd_mensual'].median():,.0f}")
fig1.update_layout(showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

# --- GRÁFICO 2 Y 3 EN COLUMNAS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Salario por Seniority")
    orden = ['junior', 'semi-senior', 'senior']
    fig2 = px.box(
        df, x='seniority', y='salario_usd_mensual',
        category_orders={'seniority': orden},
        labels={'seniority': 'Seniority', 'salario_usd_mensual': 'Salario USD/mes'},
        color='seniority',
        color_discrete_sequence=['#93c5fd', '#2563eb', '#1e3a8a']
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Salario por Modalidad")
    mediana_modalidad = df.groupby('modalidad')['salario_usd_mensual'].median().reset_index()
    mediana_modalidad = mediana_modalidad.sort_values('salario_usd_mensual')
    fig3 = px.bar(
        mediana_modalidad, x='salario_usd_mensual', y='modalidad',
        orientation='h',
        labels={'salario_usd_mensual': 'Salario USD/mes', 'modalidad': ''},
        color='salario_usd_mensual',
        color_continuous_scale='Blues'
    )
    fig3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

# --- GRÁFICO 4: TECNOLOGÍAS ---
st.subheader("Top 15 Tecnologías más Usadas")
excluir = ['ninguno de los anteriores', 'no utilizo', 'ninguna', 'otros', 'other']
techs = df['tecnologias'].dropna().str.lower().str.split(',').explode().str.strip()
techs = techs[~techs.isin(excluir)]
top_techs = techs.value_counts().head(15).reset_index()
top_techs.columns = ['tecnologia', 'cantidad']

fig4 = px.bar(
    top_techs, x='cantidad', y='tecnologia', orientation='h',
    labels={'cantidad': 'Cantidad de profesionales', 'tecnologia': ''},
    color='cantidad', color_continuous_scale='Blues'
)
fig4.update_layout(coloraxis_showscale=False, yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig4, use_container_width=True)

# --- GRÁFICO 5: SALARIO POR PROVINCIA ---
st.subheader("Salario Mediano por Provincia (Top 8)")
top_prov = df['provincia'].value_counts().head(8).index
df_prov = df[df['provincia'].isin(top_prov)]
mediana_prov = df_prov.groupby('provincia')['salario_usd_mensual'].median().reset_index()
mediana_prov = mediana_prov.sort_values('salario_usd_mensual')

fig5 = px.bar(
    mediana_prov, x='salario_usd_mensual', y='provincia', orientation='h',
    labels={'salario_usd_mensual': 'Salario USD/mes', 'provincia': ''},
    color='salario_usd_mensual', color_continuous_scale='Blues'
)
fig5.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig5, use_container_width=True)