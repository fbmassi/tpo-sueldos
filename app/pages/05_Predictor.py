import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

st.set_page_config(page_title="Predictor", page_icon="💰", layout="wide")

st.title("💰 ¿Cuánto debería ganar?")
st.caption("Estimá el salario de un profesional tech en Argentina según su perfil")

# --- CARGA DE MODELOS ---
@st.cache_resource
def cargar_modelos():
    with open('data/processed/modelo_salario.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open('data/processed/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('data/processed/features.json', 'r') as f:
        features = json.load(f)
    return modelo, encoders, features

@st.cache_data
def cargar_datos():
    return pd.read_csv('data/processed/dataset_final.csv')

modelo, encoders, features = cargar_modelos()
df = cargar_datos()

st.divider()

# --- FORMULARIO ---
st.subheader("Ingresá tu perfil")

col1, col2 = st.columns(2)

with col1:
    rol = st.selectbox("Rol", sorted(df['rol_estandar'].dropna().unique()))
    seniority = st.selectbox("Seniority", ['junior', 'semi-senior', 'senior'])
    modalidad = st.selectbox("Modalidad", [
        '100% remoto', 
        'híbrido (presencial y remoto)', 
        '100% presencial'
    ])

with col2:
    anos_experiencia = st.slider("Años de experiencia", 0, 30, 3)
    provincia = st.selectbox("Provincia", sorted(df['provincia'].dropna().unique()))
    sueldo_dolarizado = st.selectbox("¿Tu sueldo está dolarizado?", 
                                      sorted(df['sueldo_dolarizado'].dropna().unique()))
    tamano_empresa = st.selectbox("Tamaño de empresa",
                                   sorted(df['tamano_empresa'].dropna().unique()))

st.divider()

# --- PREDICCIÓN ---
if st.button("💡 Estimar salario", use_container_width=True):

    try:
        input_data = {
            'anos_experiencia': anos_experiencia,
            'seniority': seniority,
            'modalidad': modalidad,
            'rol_estandar': rol,
            'provincia': provincia,
            'sueldo_dolarizado': sueldo_dolarizado,
            'tamano_empresa': tamano_empresa
        }

        input_df = pd.DataFrame([input_data])

        # Encoding
        cols_cat = ['seniority', 'modalidad', 'rol_estandar', 
                    'provincia', 'sueldo_dolarizado', 'tamano_empresa']
        
        for col in cols_cat:
            le = encoders[col]
            val = input_df[col].astype(str).values[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = 0

        prediccion = modelo.predict(input_df[features])[0]

        # Contexto del mercado para ese rol y seniority
        subset = df[
            (df['rol_estandar'] == rol) & 
            (df['seniority'] == seniority)
        ]['salario_usd_mensual']

        p25 = subset.quantile(0.25) if len(subset) > 0 else prediccion * 0.8
        p75 = subset.quantile(0.75) if len(subset) > 0 else prediccion * 1.2
        mediana = subset.median() if len(subset) > 0 else prediccion

        st.divider()
        st.subheader("📊 Resultado")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Salario estimado", f"USD {prediccion:,.0f}/mes")
        with col2:
            st.metric("Mediana del mercado", f"USD {mediana:,.0f}/mes")
        with col3:
            st.metric("Percentil 25", f"USD {p25:,.0f}/mes")
        with col4:
            st.metric("Percentil 75", f"USD {p75:,.0f}/mes")

        # Posición relativa
        if prediccion >= p75:
            st.success(f"✅ Tu perfil está en el **top 25%** del mercado para {rol} {seniority}")
        elif prediccion >= mediana:
            st.info(f"📊 Tu perfil está **por encima de la mediana** del mercado para {rol} {seniority}")
        elif prediccion >= p25:
            st.warning(f"⚠️ Tu perfil está **por debajo de la mediana** del mercado para {rol} {seniority}")
        else:
            st.error(f"🔴 Tu perfil está en el **25% inferior** del mercado para {rol} {seniority}")

        # Equivalente en ARS
        TC = 1342
        IPC_REF = 9193.24
        IPC_2020 = 289.83
        salario_ars = prediccion * TC
        salario_real_2020 = salario_ars / (IPC_REF / IPC_2020)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Equivalente en ARS (agosto 2025)", f"${salario_ars:,.0f}/mes")
        with col2:
            st.metric("Poder adquisitivo real (pesos 2020)", f"${salario_real_2020:,.0f}/mes")

    except Exception as e:
        st.error(f"Error al calcular: {e}")