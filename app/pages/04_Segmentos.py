import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Segmentos", page_icon="🤖", layout="wide")

st.title("🤖 Segmentos del Mercado Tech")
st.caption("Clustering de profesionales tech argentinos — KMeans con 4 segmentos")

@st.cache_data
def cargar_datos():
    return pd.read_csv('data/processed/dataset_final.csv')

df = cargar_datos()

nombres_cluster = {
    0: 'Junior Remoto',
    1: 'Senior Remoto',
    2: 'Senior Manager Híbrido',
    3: 'Senior Data Híbrido'
}
df['cluster_nombre'] = df['cluster'].map(nombres_cluster)

st.divider()

# --- MÉTRICAS POR CLUSTER ---
st.subheader("Perfil de cada segmento")
cols = st.columns(4)
colores = ['#93c5fd', '#2563eb', '#1e3a8a', '#60a5fa']

for i, (cluster_id, nombre) in enumerate(nombres_cluster.items()):
    subset = df[df['cluster'] == cluster_id]
    with cols[i]:
        st.markdown(f"### {nombre}")
        st.metric("Profesionales", f"{len(subset):,}")
        st.metric("Salario mediano", f"USD {subset['salario_usd_mensual'].median():,.0f}/mes")
        st.metric("Experiencia mediana", f"{subset['anos_experiencia'].median():.0f} años")
        st.metric("Seniority típico", subset['seniority'].mode()[0])
        st.metric("Modalidad típica", subset['modalidad'].mode()[0])

st.divider()

# --- SCATTER PLOT ---
st.subheader("Distribución de Segmentos — Experiencia vs Salario")
fig1 = px.scatter(
    df.dropna(subset=['anos_experiencia', 'salario_usd_mensual', 'cluster_nombre']),
    x='anos_experiencia',
    y='salario_usd_mensual',
    color='cluster_nombre',
    title='Segmentos de Profesionales Tech en Argentina',
    labels={
        'anos_experiencia': 'Años de experiencia',
        'salario_usd_mensual': 'Salario USD/mes',
        'cluster_nombre': 'Segmento'
    },
    color_discrete_sequence=colores,
    opacity=0.6
)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# --- COMPARATIVA DE SALARIOS POR SEGMENTO ---
st.subheader("Salario por Segmento")
mediana_cluster = df.groupby('cluster_nombre')['salario_usd_mensual'].median().reset_index()
mediana_cluster = mediana_cluster.sort_values('salario_usd_mensual')

fig2 = px.bar(
    mediana_cluster,
    x='salario_usd_mensual', y='cluster_nombre', orientation='h',
    labels={'salario_usd_mensual': 'Salario Mediano USD/mes', 'cluster_nombre': ''},
    color='salario_usd_mensual',
    color_continuous_scale='Blues'
)
fig2.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- DISTRIBUCIÓN DE ROLES POR SEGMENTO ---
st.subheader("Explorar un segmento")
segmento_sel = st.selectbox("Seleccioná un segmento:", list(nombres_cluster.values()))

subset = df[df['cluster_nombre'] == segmento_sel]

col1, col2 = st.columns(2)
with col1:
    roles = subset['rol_estandar'].value_counts().reset_index()
    roles.columns = ['rol', 'cantidad']
    fig3 = px.pie(
        roles.head(6), values='cantidad', names='rol',
        title='Distribución de Roles',
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    modalidades = subset['modalidad'].value_counts().reset_index()
    modalidades.columns = ['modalidad', 'cantidad']
    fig4 = px.pie(
        modalidades, values='cantidad', names='modalidad',
        title='Distribución de Modalidad',
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    st.plotly_chart(fig4, use_container_width=True)