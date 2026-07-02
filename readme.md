# Mercado Laboral Tech Argentino 🇦🇷

**Análisis y predicción de salarios del sector IT — Trabajo Práctico Obligatorio de Minería de Datos**
Ciencia de Datos — UADE 2026 | Prof. Santiago Martín

Integramos seis ediciones de la encuesta salarial de **Sysarmy (2022–2025)** con ocho series
macroeconómicas para construir un dataset único y comparable de **31.088 profesionales** tech.
Sobre esa base entrenamos un modelo predictivo de salario (Random Forest), una segmentación de
mercado (K-Means) y dos aplicaciones web interactivas.

## Hipótesis

**Hipótesis central:** el sueldo real de un profesional tech está determinado por su perfil
laboral (rol, experiencia) y su contexto (provincia, modalidad, tamaño de empresa), una vez
expresado en moneda constante para que las ediciones sean comparables entre sí.

| # | Hipótesis desagregada | Resultado |
|---|----------------------|-----------|
| H1 | El seniority/experiencia es la variable más predictiva | ✅ Confirmada |
| H2 | La experiencia suma con rendimientos decrecientes | ✅ Confirmada (0→3 años: +85%; 10→13: +3%) |
| H3 | Ciertas tecnologías tienen premium salarial | 🟡 Parcial (lo lideran Go/Rust, no solo Python) |
| H4 | CABA paga más que el interior, pero menos de lo que se cree | ✅ Confirmada (+24%) |
| H5 | Quienes cobran en dólares tienen poder de compra más estable | ❌ Refutada (más volatilidad real) |
| H6 | Cuando el ITCRM sube, suben los sueldos en USD | 🟡 Indicio (corr. 0,70; n=6) |

## Fuentes de datos (9)

| Fuente | Aporte |
|--------|--------|
| [Sysarmy](https://sysarmy.com) (6 ediciones) | Salarios y perfiles — 31.088 respuestas |
| [IPC INDEC](https://www.indec.gob.ar) | Deflactor de pesos |
| Dólar MEP/Blue (bluelytics) | Tipo de cambio histórico |
| US CPI (FRED) | Deflactor de dólares |
| CBT INDEC | Canasta básica total |
| RIPTE | Salario formal promedio (benchmark) |
| Big Mac Index | Paridad de poder adquisitivo |
| ITCRM (BCRA) | Tipo de cambio real multilateral |
| [Stack Overflow Survey](https://survey.stackoverflow.co) | Benchmark internacional |

> **Reproducibilidad:** los CSV crudos no se versionan (el de Stack Overflow pesa 140 MB, por
> encima del límite de GitHub), pero cada fuente tiene un **gemelo `.parquet` liviano en
> `data/raw/`** que sí viaja con el repo (140 MB → 11 MB). El pipeline los usa automáticamente
> si el CSV no está: **clonar el repo alcanza para reproducir todo el trabajo**.

## Metodología

1. **Limpieza y unificación** (`notebooks/limpiar_y_unificar_datos.py`): armonización de esquemas
   entre ediciones (43–56 columnas con nombres distintos), parseo y tipado, imputación de
   faltantes, normalización de categorías (género, modalidad, provincias) y marcado de outliers
   sin eliminarlos. De 32.309 filas crudas quedan 31.088 (solo 3,8% descartado).
2. **Variable target — salario real:** cada sueldo se deflacta según su **moneda nativa**
   (pesos por IPC, dólares por US CPI) a base **mayo 2026**. Sin esto, el modelo aprendería el
   paso del tiempo (la nominalidad creció 10×), no el perfil.
3. **Feature engineering:** roles agrupados en top-15 + «Otro» (cubren el 90% de los casos),
   provincias con <100 registros → «Otra», one-hot de categóricas y exclusiones deliberadas:
   *seniority* (redundante con experiencia), variables derivadas del salario (fuga de
   información) y **tecnologías** (su premium es un proxy del contexto — backend/dolarización —
   y generaban ruido como entrada; se analizan de forma descriptiva en el EDA).
4. **Validación temporal:** entrenamiento con las ediciones 2022.2 → 2025.1 y test con 2025.2
   (la última), como pide un escenario productivo real.

## Modelos

| Modelo | R² (test temporal) |
|--------|--------------------|
| Regresión lineal simple | 0,13 |
| Regresión lineal múltiple | 0,22 |
| Árbol de decisión | 0,20 |
| Regresión polinómica | 0,28 |
| **Random Forest (elegido)** | **0,27** |

Se elige **Random Forest** (100 árboles, regularizado): empata en desempeño con la polinómica
pero es más robusto ante outliers, no extrapola valores absurdos y expone la importancia de
cada variable. Un R² ≈ 0,27 es honesto para salarios humanos: el resto depende de la empresa,
la negociación y factores que la encuesta no captura.

**Segmentación (K-Means, k=4):** arquetipos del mercado según experiencia, antigüedad,
modalidad y dolarización — de «🌱 Junior» a «🎖️ Senior +15».

## Aplicaciones

```bash
# app en español (pesos reales)
.venv/bin/streamlit run app/main.py

# app en inglés (dólares reales)
.venv/bin/streamlit run app_en/main.py
```

Páginas: **Inicio · Análisis del mercado (EDA) · Estimador de sueldo (RF) · Perfiles del
mercado (K-Means) · Argentina vs el mundo**.

Un estimador de consola también está disponible: `python notebooks/predictor_cli.py`.

## Cómo reproducir todo

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python notebooks/limpiar_y_unificar_datos.py   # dataset final
.venv/bin/python notebooks/eda_completo.py               # gráficos del EDA
.venv/bin/python notebooks/evaluacion_modelos.py         # comparación de modelos (ARS)
.venv/bin/python notebooks/evaluacion_modelos_usd.py     # ídem en USD
.venv/bin/python notebooks/preparar_mundo.py             # agregado Stack Overflow p/ la app
```

Si se agregan CSV crudos nuevos: `python notebooks/fuentes_raw.py` regenera los gemelos parquet.

## Estructura del proyecto

```
data/
  raw/            ← fuentes originales (gemelos .parquet versionados)
  processed/      ← dataset final, EDA, gráficos de modelos
notebooks/        ← pipeline, EDA, modelos, utilidades (ejecutables)
app/              ← app Streamlit en español (ARS)
app_en/           ← app Streamlit en inglés (USD)
presentacion/     ← defensa oral (PPTX + DOCX)
```

## Equipo

| Rol | Integrante |
|-----|-----------|
| Product Owner | XX |
| Data Engineer | XX |
| Data Quality | XX |
| Data Analyst | XX |
| Data Scientist | XX |
| Visualización / Dev | XX |
