import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Comparativa LATAM", page_icon="🌎", layout="wide")

st.title("🌎 Comparativa Argentina vs LATAM")
st.caption("Brecha salarial entre profesionales tech argentinos y el resto de Latinoamérica")

@st.cache_data
def cargar_datos():
    df = pd.read_csv('data/processed/dataset_final.csv')
    so_latam = pd.read_csv('data/processed/stackoverflow_latam.csv')
    return df, so_latam

df, so_latam = cargar_datos()

st.divider()

# --- MÉTRICAS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Salario mediano Argentina", f"USD {df['salario_usd_mensual'].median():,.0f}/mes")
with col2:
    st.metric("Salario mediano LATAM", f"USD {so_latam['salario_anual_usd'].median()/12:,.0f}/mes")
with col3:
    brecha_general = ((df['salario_usd_mensual'].median() * 12) - so_latam['salario_anual_usd'].median()) / so_latam['salario_anual_usd'].median() * 100
    st.metric("Brecha general", f"{brecha_general:.1f}%")

st.divider()

# --- GRÁFICO 1: BRECHA POR ROL ---
st.subheader("Salario Anual USD — Argentina vs LATAM por Rol")

brecha = df.groupby('rol_estandar').agg(
    argentina=('salario_usd_anual', 'median'),
    latam=('salario_usd_anual_latam', 'first')
).dropna().reset_index()

brecha_melted = brecha.melt(
    id_vars='rol_estandar',
    value_vars=['argentina', 'latam'],
    var_name='mercado',
    value_name='salario_usd_anual'
)

fig1 = px.bar(
    brecha_melted,
    x='salario_usd_anual', y='rol_estandar',
    color='mercado', barmode='group', orientation='h',
    labels={'salario_usd_anual': 'Salario USD/año', 'rol_estandar': '', 'mercado': 'Mercado'},
    color_discrete_map={'argentina': '#2563eb', 'latam': '#93c5fd'}
)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# --- GRÁFICO 2: BRECHA % POR ROL ---
st.subheader("Brecha Porcentual por Rol (Argentina vs LATAM)")

brecha['brecha_pct'] = ((brecha['argentina'] - brecha['latam']) / brecha['latam'] * 100).round(1)
brecha = brecha.sort_values('brecha_pct')

fig2 = px.bar(
    brecha,
    x='brecha_pct', y='rol_estandar', orientation='h',
    labels={'brecha_pct': 'Brecha (%)', 'rol_estandar': ''},
    color='brecha_pct',
    color_continuous_scale='RdBu',
    color_continuous_midpoint=0
)
fig2.add_vline(x=0, line_dash='dash', line_color='black')
fig2.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- TABLA RESUMEN ---
st.subheader("Resumen por Rol")
brecha['argentina_usd'] = brecha['argentina'].apply(lambda x: f"USD {x:,.0f}")
brecha['latam_usd'] = brecha['latam'].apply(lambda x: f"USD {x:,.0f}")
brecha['brecha'] = brecha['brecha_pct'].apply(lambda x: f"{x:.1f}%")

st.dataframe(
    brecha[['rol_estandar', 'argentina_usd', 'latam_usd', 'brecha']].rename(columns={
        'rol_estandar': 'Rol',
        'argentina_usd': 'Argentina (anual)',
        'latam_usd': 'LATAM (anual)',
        'brecha': 'Brecha'
    }),
    use_container_width=True,
    hide_index=True
)