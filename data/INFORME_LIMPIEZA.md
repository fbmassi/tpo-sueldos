# Informe de Limpieza y Transformación de Datos
**TPO — Mercado Laboral Tech Argentina**

Documento que detalla el procesamiento aplicado a cada fuente de datos del
proyecto, desde los archivos crudos (`data/raw/`) hasta el dataset final
unificado (`data/processed/dataset_final_mercado_laboral.parquet`).

Generado por el pipeline `data/limpiar_y_unificar_datos.py`.

---

## 1. Resumen

| Concepto | Valor |
|---|---|
| Fuentes integradas | 9 (Sysarmy, IPC, Dólar, CBT, RIPTE, Big Mac, ITCRM, US CPI, Stack Overflow) |
| Ediciones de Sysarmy | 6 (2022.2 → 2025.2) |
| Filas Sysarmy crudas | 32.309 |
| Filas en el dataset final | 31.088 (se descartó **3,8%**) |
| Columnas del dataset final | 15 |
| Base de ajuste (pesos y USD reales) | Mayo 2026 (fecha más reciente del dólar) |

**Decisión central (consigna del profesor):** el salario se reduce a **una sola
medida real**, expresada en dos monedas:
- `salario_real_ars` — pesos ajustados por inflación argentina (IPC).
- `salario_real_usd` — dólares MEP ajustados por inflación de EE.UU. (US CPI).

Se descartan el salario nominal y las métricas de poder adquisitivo.

**Honestidad metodológica:** los campos de Sysarmy son en su mayoría *dropdowns
controlados* (no texto libre) y 3 de las 6 ediciones venían pre-limpias por
OpenQube. Por eso la limpieza "pesada" (normalización fuzzy, deduplicación)
aporta poco. El valor real del pipeline está en la **armonización de las 6
ediciones, el parseo de campos puntuales, la integración con las series macro y
el ajuste por inflación (argentina y de EE.UU.) a una base común.**

---

## 2. Fuente principal — SYSARMY (6 ediciones)

### 2.1 Lectura
- Los CSV traen filas de preámbulo antes del encabezado real. El lector
  (`leer_sysarmy_crudo`) prueba `skiprows` de 0 a 13 y elige el que reconoce más
  columnas críticas. Detectó automáticamente `skiprows` distinto por edición
  (7, 8 o 9).
- Cada archivo (`sysarmy_YYYY_S.csv`) recibe su **fecha de referencia** según el
  nombre: semestre 1 → enero, semestre 2 → julio.

### 2.2 Armonización de esquemas (lo más valioso)
Las ediciones tienen entre **43 y 56 columnas** y los nombres cambian año a año.
Ejemplo del mismo dato (salario):

| Edición | Nombre de la columna |
|---|---|
| 2022.2 | `Último salario mensual  o retiro BRUTO (en tu moneda local)` |
| 2025.2 | `ultimo_salario_mensual_o_retiro_bruto_en_pesos_argentinos` |

`resolver_columna` los mapea al mismo campo destino mediante coincidencia por
tokens completos (no por subcadena, para evitar falsos positivos).

### 2.3 Selección de columnas
Se conservan 12 columnas: provincia, edad, género, rol, seniority, años de
experiencia, antigüedad, tecnologías, salario bruto, cobra en dólares,
modalidad, tamaño de empresa.

### 2.4 Parseo de tipos
- **Salario**: texto → número (`a_numero`), tolerando separadores de miles y
  símbolos. *Ej.: `'300000'` → `300000.0`.* (En estos archivos el campo ya venía
  como entero limpio; el parser es robusto pero hizo poco trabajo.)
- Edad fuera de \[18, 75\] → nulo.

### 2.5 Normalización de texto
- Provincia, rol, género, modalidad: `.lower().strip()` + mapa de variantes +
  fuzzy (rapidfuzz). **Nota honesta:** los valores ya venían estandarizados
  (`'Ciudad Autónoma de Buenos Aires'`, `'Senior'`, `'100% remoto'`), así que la
  parte fuzzy casi no se activa.
- **Tecnologías** (texto libre multi-valor): `.lower()` + split. *Ej.:
  `'Javascript, Python'` → `'javascript, python'`.*

### 2.6 Inferencia de seniority (sólido)
Las ediciones 2022–2023 **no traen** la columna `seniority`. Se infiere de los
años de experiencia: < 2 → junior, 2–5 → semi-senior, > 5 → senior.

### 2.7 Fix de `cobra_en_dolares` (bug real corregido)
El campo `pagos_en_dolares` es **texto**, no sí/no:

| Valor crudo | → |
|---|---|
| `Cobro todo el salario en dólares` | **True** |
| `Cobro parte del salario en dólares` | **True** |
| `Mi sueldo está dolarizado (pero cobro en moneda local)` | False |
| (vacío) | False |

### 2.8 Valores faltantes
| Columna | Tratamiento |
|---|---|
| provincia | **Eliminación** de la fila (crítica) |
| salario | **Eliminación** de la fila (crítica) |
| edad | Imputación por **mediana** (si faltan ≤5%), sino eliminación |
| género / seniority / modalidad | Imputación por **moda** |
| tecnologías | `'No especifica'` |

### 2.9 Errores de carga evidentes
Se elimina un salario sólo si está fuera de `[mediana/50, mediana×50]` (error de
carga). **Los outliers normales se conservan** (marcados con `es_outlier`). El
umbral se evalúa **por edición**.

### 2.10 Duplicados
Datos anónimos → sólo se eliminan filas **100% idénticas**. Elimina casi nada.

**Resultado:** las 6 ediciones limpias se apilan → `sysarmy_limpio.parquet`
(31.088 filas).

---

## 3. Series macro (IPC, Dólar, RIPTE, CBT, Big Mac, ITCRM, US CPI)

Diagnóstico de los archivos crudos: **0 nulos, 0 fechas duplicadas, 0 negativos**
en todas. Son datos oficiales / API, vienen limpios. El procesamiento es
**transformación liviana** (no scrubbing) y se hace **en memoria** — no se
guardan parquets intermedios.

| Fuente | Origen | Transformación | Uso |
|---|---|---|---|
| IPC | datos.gob.ar | fecha→mes, float, deriva `inflacion_mensual_pct` | **Pesos reales** |
| Dólar | bluelytics (parquet) | fuente "Blue" ≈ MEP, promedio mensual (2011-2026) | **USD** |
| US CPI | FRED (`CPIAUCSL`) | fecha→mes, float | **USD reales** |
| RIPTE | datos.gob.ar | fecha→mes, promedio mensual | Contexto |
| CBT | datos.gob.ar (GBA) | fecha→mes | Contexto |
| Big Mac | The Economist | fecha→datetime | Contexto |
| ITCRM | BCRA (.xlsx) | detección de hoja mensual + encabezado | Contexto |

Único caso con estructura no trivial: **ITCRM** (Excel con preámbulo y varias
hojas) — resuelto detectando la hoja "prom. mens." y la fila de encabezado.

---

## 4. Stack Overflow (benchmark internacional)

`datosInternacionales.csv`: 49.191 filas × 172 columnas. **No se mergea** con
Sysarmy (poblaciones distintas). Filtrado: `Country == 'Argentina'` (222) y con
salario (160). Se guarda separado: `stackoverflow_argentina_limpio.parquet`.

---

## 5. Integración y medidas finales de salario

1. **Merge** de cada serie macro sobre `fecha_edicion` con `merge_asof` (mes más
   cercano) → cada edición toma el macro de **su** momento.
2. **Contexto macroeconómico:** las variables macro (iguales para toda la
   edición) se separan a `contexto_macroeconomico.parquet` (6 filas, una por
   edición) y se unen por `fecha_edicion` (FK).
3. **Las dos medidas de salario** (base = mayo 2026, la fecha más reciente del
   dólar; IPC y US CPI toman su último dato, abril 2026). El ajuste depende de la
   **moneda nativa** del sueldo (`cobra_en_dolares`), porque un sueldo dolarizado
   NO pierde valor con la inflación argentina:

   | Caso | salario_real_ars | salario_real_usd |
   |---|---|---|
   | **Cobra en PESOS** | `nominal × (IPC_base / IPC_mes)` (inflación AR) | `real_ars / MEP_base` |
   | **Cobra en DÓLARES** | `real_usd × MEP_base` | `(nominal / MEP_mes) × (US_CPI_base / US_CPI_mes)` (inflación US) |

   En **ambos** casos `salario_real_ars / salario_real_usd = MEP_base`, así que el
   cociente es un tipo de cambio coherente y las dos columnas son consistentes.
   Cada edición queda en moneda de la misma fecha → **comparable en el tiempo**.

---

## 6. Salidas del pipeline

| Archivo | Contenido |
|---|---|
| `dataset_final_mercado_laboral.parquet` | 31.088 filas × 15 cols: perfil + `fecha_edicion` (FK) + `salario_real_ars` + `salario_real_usd` + `es_outlier` |
| `contexto_macroeconomico.parquet` | Macro por edición (6 filas, FK `fecha_edicion`) |
| `sysarmy_limpio.parquet` | Sysarmy limpio (paso intermedio, con nominal) |
| `stackoverflow_argentina_limpio.parquet` | Benchmark, separado |
| `data_quality_report.txt` | Log detallado de cada paso |

---

## 7. Conclusión honesta

- **Lo que hizo poco** (porque la data ya venía ordenada): normalización fuzzy de
  provincias/roles, deduplicación.
- **Lo que aportó de verdad:** armonizar 6 esquemas distintos, parsear salario,
  inferir seniority en ediciones viejas, el fix de `cobra_en_dolares`, y descartar
  filas sin datos críticos.
- **El músculo del proyecto** no es el scrubbing (los datos estaban limpios) sino
  la **integración multi-fuente y el ajuste a una medida real comparable** (pesos
  por IPC argentino, dólares por US CPI), que es lo que convierte datos crudos
  dispersos en un dataset analítico y comparable en el tiempo.
