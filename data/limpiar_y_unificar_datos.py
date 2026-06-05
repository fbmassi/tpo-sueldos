"""
limpiar_y_unificar_datos.py
===========================

Pipeline ETL completo del TPO "Mercado Laboral Tech Argentina".

Limpia TODAS las fuentes de datos de data/raw/, las guarda limpias en
data/processed/ y genera el dataset final unificado:

    data/processed/dataset_final_mercado_laboral.parquet

Además produce:
    data/processed/data_quality_report.txt   (reporte de limpieza)

Los sueldos outlier NO se eliminan ni se exportan aparte: quedan dentro del
dataset final marcados con la columna `es_outlier`. Sólo se descartan los
errores de carga muy evidentes (valores varios órdenes de magnitud fuera de
la mediana).

------------------------------------------------------------------------------
ARCHIVOS QUE DEBEN ESTAR EN data/raw/ ANTES DE CORRER (descarga manual):
    - sysarmy_2025_2.csv          (sysarmy.com/blog, edición más reciente)
    - datosInternacionales.csv    (survey.stackoverflow.co/2024)
Y los que descarga `descargar_datos.py`:
    - ipc_indec.csv, dolar_mep.csv, bigmac_index_argentina.csv,
      ripte.csv, cbt_indec.csv

Si alguna fuente falta, se loguea el problema y el pipeline continúa con el
resto (no aborta).

Ejecutar:
    python data/limpiar_y_unificar_datos.py
------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process

    _RAPIDFUZZ = True
except Exception:  # pragma: no cover - rapidfuzz es requisito, pero degradamos
    _RAPIDFUZZ = False

# ----------------------------------------------------------------------------
# Configuración de rutas
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

# Nombres de archivo de entrada (según enunciado)
F_SYSARMY = RAW_DIR / "sysarmy_2025_2.csv"
F_IPC = RAW_DIR / "ipc_indec.csv"
F_DOLAR = RAW_DIR / "dolar_mep.csv"
F_BIGMAC = RAW_DIR / "bigmac_index_argentina.csv"
F_RIPTE = RAW_DIR / "ripte.csv"
# CBT real de datos.gob.ar (Gran Buenos Aires), valores realistas.
# (Se descartó cbt_indec.csv porque sus valores estaban ~6x inflados.)
F_CBT = RAW_DIR / "cbt.csv"
F_STACKOVERFLOW = RAW_DIR / "datosInternacionales.csv"
# Dólar con historia (bluelytics 2011-2026); el CSV de MEP sólo cubre 2026.
F_DOLAR_PARQUET = RAW_DIR / "dolar_mep.parquet"
# ITCRM (BCRA): índice de tipo de cambio real multilateral.
F_ITCRM = RAW_DIR / "ITCRMSerie.xlsx"
# US CPI (FRED, CPIAUCSL): inflación de EE.UU. -> ajusta USD a dólares reales.
F_USCPI = RAW_DIR / "us_cpi.csv"

# Constantes de negocio
IPC_BASE = 4744.45  # fallback (IPC ene-2024) si no se puede leer la base real
FUZZ_THRESHOLD = 88  # umbral de similitud para normalización fuzzy

# Fecha a la que se ajustan TODOS los sueldos: pesos reales (por IPC argentino)
# y dólares reales (por US CPI). Se usa la fecha más reciente disponible (mayo
# 2026). Nota: IPC y US CPI son mensuales con rezago, su último dato es abril
# 2026; _valor_en_fecha toma el mes más cercano para cada serie.
FECHA_BASE_REAL = pd.Timestamp("2026-05-01")

# Un sueldo fuera de [mediana / FACTOR, mediana * FACTOR] se considera un ERROR
# de carga evidente (p.ej. cargado en miles, en USD, o con ceros de más) y se
# descarta. Los outliers estadísticos "normales" (Q1/Q99) SÍ se conservan.
FACTOR_ERROR_EVIDENTE = 50

# Mes de referencia de la edición de Sysarmy usado para el merge con las series
# macro (IPC, dólar, RIPTE, CBT, Big Mac). La encuesta es la edición 2026.1, así
# que la referencia es un mes de 2026 cubierto por las series macro. Si Sysarmy
# trae una columna de fecha propia, esa tiene prioridad; esto es el fallback.
# IMPORTANTE: debe caer dentro del rango de los datos macro (todos llegan a 2026).
FECHA_EDICION_SYSARMY = pd.Timestamp("2026-03-01")

# CONTEXTO MACROECONÓMICO: todas las variables macro del momento de cada
# edición. Se guardan en contexto_macroeconomico.parquet (una fila por edición)
# y se unen al dataset por 'fecha_edicion' (FK).
COLS_CONTEXTO_EDICION = [
    "fecha_edicion", "ipc", "inflacion_mensual_pct", "dolar_mep",
    "ripte", "cbt", "fecha_bigmac", "precio_bigmac_ars", "itcrm", "us_cpi",
]
# El dataset final guarda dos medidas de salario (pesos reales y USD reales),
# las canastas básicas que cubre, perfil y FK. Se descarta el nominal.
COLS_DESCARTAR_FINAL = ["salario_bruto_ars", "es_outlier_sueldo"]

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("etl")

# ----------------------------------------------------------------------------
# Acumulador del reporte de calidad
# ----------------------------------------------------------------------------
REPORTE: list[str] = []


def rep(linea: str = "") -> None:
    """Agrega una línea al reporte de calidad."""
    REPORTE.append(linea)


def rep_seccion(titulo: str) -> None:
    rep("")
    rep("=" * 70)
    rep(titulo)
    rep("=" * 70)


# ----------------------------------------------------------------------------
# Helpers genéricos
# ----------------------------------------------------------------------------
def quitar_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar_clave(texto: str) -> str:
    """Normaliza un nombre de columna para comparar (sin acentos, snake)."""
    t = quitar_acentos(str(texto)).lower().strip()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    return t.strip("_")


def resolver_columna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    """
    Encuentra en `df` una columna que matchee alguno de los `candidatos`.

    Estrategia: match exacto -> tokens completos -> fuzzy (rapidfuzz).
    Devuelve el nombre ORIGINAL de la columna o None.

    El match por tokens (no por subcadena) evita falsos positivos peligrosos:
    p.ej. el candidato corto 'id' NO debe matchear 'modalidad' por contener
    'id'; sólo matchea si 'id' aparece como palabra entera en la columna.
    """
    cols_norm = {normalizar_clave(c): c for c in df.columns}
    cols_tokens = {cn: set(cn.split("_")) for cn in cols_norm}
    cand_norm = [normalizar_clave(c) for c in candidatos]

    # 1) match exacto normalizado
    for cn in cand_norm:
        if cn in cols_norm:
            return cols_norm[cn]

    # 2) tokens completos: la columna empieza con el candidato, o todos los
    #    tokens del candidato están presentes como palabras en la columna.
    for cn in cand_norm:
        if not cn:
            continue
        cand_tokens = set(cn.split("_"))
        for col_norm, col_orig in cols_norm.items():
            if col_norm.startswith(cn + "_"):
                return col_orig
            if cand_tokens and cand_tokens.issubset(cols_tokens[col_norm]):
                return col_orig

    # 3) fuzzy (sólo para candidatos suficientemente específicos)
    if _RAPIDFUZZ:
        for cn in cand_norm:
            if len(cn) < 4:
                continue
            match = process.extractOne(
                cn, list(cols_norm.keys()), scorer=fuzz.token_sort_ratio
            )
            if match and match[1] >= FUZZ_THRESHOLD:
                return cols_norm[match[0]]
    return None


def a_numero(serie: pd.Series) -> pd.Series:
    """Convierte una serie a float tolerando separadores de miles y símbolos."""

    def _conv(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "":
            return np.nan
        # quitar todo lo que no sea dígito, coma, punto o signo
        s = re.sub(r"[^0-9,.\-]", "", s)
        if s in {"", "-", ".", ","}:
            return np.nan
        # si hay coma y punto, asumir punto = miles, coma = decimal (es-AR)
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            # coma sola: decimal
            s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return np.nan

    return serie.map(_conv)


def a_fecha_mes(serie: pd.Series) -> pd.Series:
    """Convierte a datetime y trunca al primer día del mes (YYYY-MM-01)."""
    fechas = pd.to_datetime(serie, errors="coerce")
    return fechas.dt.to_period("M").dt.to_timestamp()


def pct_faltantes(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {c: 0.0 for c in df.columns}
    return (df.isna().mean() * 100).round(2).to_dict()


def guardar_parquet(df: pd.DataFrame, destino: Path, etiqueta: str) -> None:
    try:
        df.to_parquet(destino, index=False)
        log.info("Guardado %s (%d filas, %d cols) -> %s",
                 etiqueta, len(df), df.shape[1], destino.name)
    except Exception as exc:  # noqa: BLE001
        log.error("No se pudo guardar %s: %s", etiqueta, exc)


# ----------------------------------------------------------------------------
# Normalizadores de dominio (Sysarmy / Stack Overflow)
# ----------------------------------------------------------------------------
MAPA_PROVINCIAS = {
    "caba": "CABA",
    "ciudad autonoma de buenos aires": "CABA",
    "ciudad de buenos aires": "CABA",
    "capital federal": "CABA",
    "buenos aires": "Buenos Aires",
    "gba": "Buenos Aires",
    "provincia de buenos aires": "Buenos Aires",
    "cordoba": "Córdoba",
    "santa fe": "Santa Fe",
    "mendoza": "Mendoza",
    "tucuman": "Tucumán",
    "salta": "Salta",
    "entre rios": "Entre Ríos",
    "chaco": "Chaco",
    "corrientes": "Corrientes",
    "misiones": "Misiones",
    "san juan": "San Juan",
    "san luis": "San Luis",
    "neuquen": "Neuquén",
    "rio negro": "Río Negro",
    "chubut": "Chubut",
    "santa cruz": "Santa Cruz",
    "tierra del fuego": "Tierra del Fuego",
    "la pampa": "La Pampa",
    "la rioja": "La Rioja",
    "catamarca": "Catamarca",
    "santiago del estero": "Santiago del Estero",
    "jujuy": "Jujuy",
    "formosa": "Formosa",
}

PROVINCIAS_CANON = sorted(set(MAPA_PROVINCIAS.values()))

MAPA_SENIORITY = {
    "junior": "junior",
    "jr": "junior",
    "trainee": "junior",
    "semi senior": "semi-senior",
    "semi-senior": "semi-senior",
    "ssr": "semi-senior",
    "semisenior": "semi-senior",
    "senior": "senior",
    "sr": "senior",
}

MAPA_MODALIDAD = {
    "100% remoto": "100% remoto",
    "remoto": "100% remoto",
    "remote": "100% remoto",
    "full remoto": "100% remoto",
    "hibrido": "híbrido",
    "hybrid": "híbrido",
    "mixto": "híbrido",
    "presencial": "100% presencial",
    "100% presencial": "100% presencial",
    "on site": "100% presencial",
    "in person": "100% presencial",
    "oficina": "100% presencial",
}

MAPA_GENERO = {
    "masculino": "masculino",
    "hombre": "masculino",
    "varon": "masculino",
    "male": "masculino",
    "m": "masculino",
    "femenino": "femenino",
    "mujer": "femenino",
    "female": "femenino",
    "f": "femenino",
    "no binario": "otro",
    "no binarie": "otro",
    "otro": "otro",
    "other": "otro",
    "prefiero no decir": "no especifica",
    "prefiero no responder": "no especifica",
}

ROLES_CANON = [
    "Developer", "QA", "DevOps", "DBA", "Data Scientist", "Data Engineer",
    "Data Analyst", "Tech Lead", "Engineering Manager", "Product Manager",
    "Project Manager", "UX/UI Designer", "SysAdmin", "Security",
    "Architect", "Scrum Master", "Support", "BI Analyst", "Mobile Developer",
    "Machine Learning Engineer",
]

MAPA_ROLES_EXTRA = {
    "dev": "Developer",
    "developer": "Developer",
    "desarrollador": "Developer",
    "programador": "Developer",
    "software engineer": "Developer",
    "fullstack": "Developer",
    "full stack": "Developer",
    "frontend": "Developer",
    "backend": "Developer",
    "qa": "QA",
    "tester": "QA",
    "quality assurance": "QA",
    "devops": "DevOps",
    "sre": "DevOps",
    "dba": "DBA",
}

TECHS_CANON = [
    "python", "javascript", "typescript", "java", "c#", "c++", "c", "go",
    "rust", "kotlin", "swift", "php", "ruby", "scala", "r", "sql", "html",
    "css", "bash", "powershell", "react", "angular", "vue", "node.js",
    "django", "flask", "spring", "dotnet", "laravel", "rails", "express",
    "postgresql", "mysql", "mongodb", "redis", "oracle", "sqlserver",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "git",
    "linux", "pandas", "numpy", "spark", "hadoop", "tensorflow", "pytorch",
]


def _map_directo(valor: str, mapa: dict[str, str], default: str) -> str:
    if pd.isna(valor):
        return default
    base = quitar_acentos(str(valor)).lower().strip()
    if base == "" or base in {"nan", "none"}:
        return default
    if base in mapa:
        return mapa[base]
    # match por 'contiene'
    for clave, val in mapa.items():
        if clave in base:
            return val
    return default


def normalizar_provincia(valor: str) -> str:
    res = _map_directo(valor, MAPA_PROVINCIAS, "")
    if res:
        return res
    # fuzzy contra canónicas
    if _RAPIDFUZZ and not pd.isna(valor):
        base = quitar_acentos(str(valor)).lower().strip()
        match = process.extractOne(base, PROVINCIAS_CANON, scorer=fuzz.WRatio)
        if match and match[1] >= FUZZ_THRESHOLD:
            return match[0]
    return "No especifica"


def normalizar_seniority(valor: str, anos_exp: float | None = None) -> str:
    res = _map_directo(valor, MAPA_SENIORITY, "")
    if res:
        return res
    # inferir de años de experiencia
    if anos_exp is not None and not pd.isna(anos_exp):
        if anos_exp < 2:
            return "junior"
        if anos_exp <= 5:
            return "semi-senior"
        return "senior"
    return ""  # se imputa luego por moda


def normalizar_modalidad(valor: str) -> str:
    return _map_directo(valor, MAPA_MODALIDAD, "")


def normalizar_genero(valor: str) -> str:
    return _map_directo(valor, MAPA_GENERO, "")


def normalizar_rol(valor: str) -> str:
    if pd.isna(valor):
        return "No especifica"
    base = quitar_acentos(str(valor)).lower().strip()
    if base == "":
        return "No especifica"
    if base in MAPA_ROLES_EXTRA:
        return MAPA_ROLES_EXTRA[base]
    for clave, val in MAPA_ROLES_EXTRA.items():
        if clave in base:
            return val
    if _RAPIDFUZZ:
        match = process.extractOne(
            base, [r.lower() for r in ROLES_CANON], scorer=fuzz.WRatio
        )
        if match and match[1] >= FUZZ_THRESHOLD:
            idx = [r.lower() for r in ROLES_CANON].index(match[0])
            return ROLES_CANON[idx]
    # title-case del original como fallback
    return str(valor).strip().title()


def normalizar_tecnologias(valor: str) -> str:
    """Normaliza un texto libre de tecnologías usando rapidfuzz."""
    if pd.isna(valor) or str(valor).strip() == "":
        return "No especifica"
    tokens = re.split(r"[,;/|]+", str(valor).lower())
    salida: list[str] = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        if t in TECHS_CANON:
            canon = t
        elif _RAPIDFUZZ:
            match = process.extractOne(t, TECHS_CANON, scorer=fuzz.WRatio)
            canon = match[0] if match and match[1] >= FUZZ_THRESHOLD else t
        else:
            canon = t
        if canon not in salida:
            salida.append(canon)
    return ", ".join(salida) if salida else "No especifica"


# ----------------------------------------------------------------------------
# FUENTE 1: SYSARMY
# ----------------------------------------------------------------------------
def leer_sysarmy_crudo(path: Path) -> pd.DataFrame | None:
    """
    Lee el CSV de sysarmy. El export suele traer filas de preámbulo antes del
    header real; probamos varios `skiprows` y elegimos el que más columnas
    críticas reconoce.
    """
    criticas = ["donde_estas_trabajando", "tengo", "genero", "seniority",
                "ultimo_salario", "trabajo_de"]
    mejor_df, mejor_score, mejor_skip = None, -1, 0
    for skip in range(0, 14):
        try:
            df = pd.read_csv(path, skiprows=skip, low_memory=False)
        except Exception:
            continue
        if df.shape[1] < 3:
            continue
        score = sum(
            1 for c in criticas if resolver_columna(df, [c]) is not None
        )
        if score > mejor_score:
            mejor_df, mejor_score, mejor_skip = df, score, skip
        if score >= len(criticas):
            break
    if mejor_df is not None:
        log.info("Sysarmy leído con skiprows=%d (%d columnas críticas detectadas)",
                 mejor_skip, mejor_score)
    return mejor_df


def _fecha_edicion_desde_nombre(nombre: str) -> pd.Timestamp | None:
    """
    Deriva la fecha de referencia de una edición a partir del nombre de archivo.
    Acepta tanto el formato 'sysarmy_YYYY_S.csv' como el viejo 'YYYY.S - ...'
    (separador '_' o '.'). Convención: semestre 1 -> enero, semestre 2 -> julio
    (los meses en que Sysarmy reporta los sueldos de cada edición).
    """
    m = re.search(r"(\d{4})[._](\d)", nombre)
    if not m:
        return None
    anio, sem = int(m.group(1)), int(m.group(2))
    if sem not in (1, 2):
        return None
    mes = 1 if sem == 1 else 7
    return pd.Timestamp(year=anio, month=mes, day=1)


def _listar_ediciones_sysarmy() -> list[tuple[Path, pd.Timestamp]]:
    """Busca en raw/ los CSV de ediciones ('YYYY.S - Sysarmy ...') con su fecha."""
    ediciones = []
    for p in sorted(RAW_DIR.glob("*.csv")):
        if "sysarmy" not in p.name.lower():
            continue
        fecha = _fecha_edicion_desde_nombre(p.name)
        if fecha is not None:
            ediciones.append((p, fecha))
    return ediciones


def limpiar_sysarmy() -> pd.DataFrame | None:
    """Limpia TODAS las ediciones de Sysarmy halladas y las apila."""
    nombre = "Sysarmy"
    rep_seccion(f"FUENTE 1 — {nombre} (multi-edición)")
    ediciones = _listar_ediciones_sysarmy()
    if not ediciones:
        # compatibilidad: un único archivo legacy
        if F_SYSARMY.exists():
            ediciones = [(F_SYSARMY, FECHA_EDICION_SYSARMY)]
        else:
            log.warning("No se encontraron ediciones de Sysarmy en raw/. Se omite.")
            rep("ARCHIVO AUSENTE: ninguna edición de Sysarmy en raw/.")
            return None

    partes = []
    for path, fecha in ediciones:
        try:
            df_ed = _limpiar_una_edicion(path, fecha)
            if df_ed is not None and len(df_ed):
                partes.append(df_ed)
                rep(f"  · {path.stem} ({fecha.date()}): {len(df_ed)} filas limpias")
                log.info("Edición %s (%s): %d filas", path.stem, fecha.date(),
                         len(df_ed))
        except Exception as exc:  # noqa: BLE001
            log.error("FALLO edición %s: %s", path.name, exc)
            rep(f"  · {path.stem}: ERROR {exc}")

    if not partes:
        rep("Ninguna edición se pudo procesar.")
        return None

    df = pd.concat(partes, ignore_index=True)
    rep(f"TOTAL Sysarmy unificado: {len(df)} filas de {len(partes)} ediciones")
    rep(f"Ediciones: {sorted(str(f.date()) for _, f in ediciones)}")
    guardar_parquet(df, PROC_DIR / "sysarmy_limpio.parquet", nombre)
    return df


def _limpiar_una_edicion(path: Path,
                         fecha_edicion: pd.Timestamp) -> pd.DataFrame | None:
    """Limpia UNA edición de Sysarmy y le asigna su fecha de referencia."""
    crudo = leer_sysarmy_crudo(path)
    if crudo is None or len(crudo) == 0:
        raise ValueError("No se pudo leer contenido válido.")
    if True:
        # --- Mapear columnas de interés ---
        mapa = {
            "provincia": ["donde_estas_trabajando", "donde estas trabajando",
                          "provincia"],
            "edad": ["tengo", "edad", "age"],
            "genero": ["genero", "tengo_el_siguiente_genero", "gender"],
            "rol": ["trabajo_de", "rol", "puesto", "devtype"],
            "seniority": ["seniority", "nivel"],
            "anos_experiencia_total": ["anos_de_experiencia",
                                       "anos_experiencia", "experiencia"],
            "anos_empresa_actual": ["antiguedad_en_la_empresa_actual",
                                    "antiguedad"],
            "tecnologias": ["lenguajes_de_programacion", "tecnologias",
                            "lenguajes", "plataformas"],
            "salario_bruto_ars": [
                "ultimo_salario_mensual_o_retiro_bruto_en_pesos_argentinos",
                "ultimo_salario_mensual_bruto", "salario_bruto",
                "salario_mensual_bruto", "sueldo_bruto"],
            "cobra_en_dolares": ["pagos_en_dolares", "cobra_en_dolares"],
            "modalidad": ["modalidad_de_trabajo", "modalidad"],
            "tamano_empresa": ["cantidad_de_personas_en_tu_organizacion",
                               "tamano_empresa", "tamano_de_la_empresa"],
        }
        df = pd.DataFrame()
        col_origen: dict[str, str] = {}
        for destino, cands in mapa.items():
            col = resolver_columna(crudo, cands)
            if col is not None:
                df[destino] = crudo[col]
                col_origen[destino] = col
            else:
                df[destino] = np.nan
                log.warning("Sysarmy: no se encontró columna para '%s'", destino)

        # fecha_edicion: la fecha de referencia de la edición (misma para todas
        # las filas de este archivo). Se usa para mergear con el macro de su mes.
        df["fecha_edicion"] = fecha_edicion

        # email / id para deduplicar
        col_id = resolver_columna(crudo, ["email", "id", "response_id", "mail"])
        df["_clave_id"] = crudo[col_id].astype(str) if col_id is not None else np.nan
        col_nombre = resolver_columna(crudo, ["nombre", "name"])
        df["_nombre"] = crudo[col_nombre].astype(str) if col_nombre is not None else ""

        # --- Tipos numéricos ---
        df["edad"] = a_numero(df["edad"])
        df["anos_experiencia_total"] = a_numero(df["anos_experiencia_total"])
        df["anos_empresa_actual"] = a_numero(df["anos_empresa_actual"])
        df["salario_bruto_ars"] = a_numero(df["salario_bruto_ars"])

        # edad fuera de [18,75] -> NaN
        df.loc[(df["edad"] < 18) | (df["edad"] > 75), "edad"] = np.nan

        # --- Normalización de texto ---
        df["provincia"] = df["provincia"].map(normalizar_provincia)
        df["rol"] = df["rol"].map(normalizar_rol)
        df["genero"] = df["genero"].map(normalizar_genero)
        df["modalidad"] = df["modalidad"].map(normalizar_modalidad)
        df["tecnologias"] = df["tecnologias"].map(normalizar_tecnologias)
        df["seniority"] = [
            normalizar_seniority(s, e)
            for s, e in zip(df["seniority"], df["anos_experiencia_total"])
        ]
        df["cobra_en_dolares"] = df["cobra_en_dolares"].map(_cobra_en_dolares)

        # --- Valores faltantes ---
        tecnica: list[str] = []

        # provincia: crítica -> eliminar filas "No especifica"/NaN
        antes = len(df)
        df = df[df["provincia"].notna() & (df["provincia"] != "No especifica")]
        tecnica.append(f"provincia: ELIMINACIÓN ({antes - len(df)} filas sin provincia)")

        # salario: crítico -> eliminar nulos
        antes = len(df)
        df = df[df["salario_bruto_ars"].notna() & (df["salario_bruto_ars"] > 0)]
        tecnica.append(f"salario_bruto_ars: ELIMINACIÓN ({antes - len(df)} filas sin sueldo)")

        # salario: errores de carga EVIDENTES (no outliers normales) -> eliminar.
        # Los outliers estadísticos se conservan; sólo se quitan valores
        # imposibles, varios órdenes de magnitud fuera de la mediana.
        if len(df) > 0:
            mediana_sal = df["salario_bruto_ars"].median()
            piso = mediana_sal / FACTOR_ERROR_EVIDENTE
            techo = mediana_sal * FACTOR_ERROR_EVIDENTE
            antes = len(df)
            df = df[df["salario_bruto_ars"].between(piso, techo)]
            n_err = antes - len(df)
            tecnica.append(
                f"salario_bruto_ars: ELIMINACIÓN de {n_err} errores de carga "
                f"evidentes (fuera de [{piso:,.0f}, {techo:,.0f}]; "
                f"mediana={mediana_sal:,.0f}). Outliers normales conservados.")

        # edad: >5% faltante -> eliminar columna? El enunciado dice eliminar filas
        # si la columna supera 5% de faltantes; si no, imputar mediana.
        if len(df) > 0:
            pct_edad = df["edad"].isna().mean() * 100
            if pct_edad > 5:
                antes = len(df)
                df = df[df["edad"].notna()]
                tecnica.append(f"edad: ELIMINACIÓN ({pct_edad:.1f}% > 5%, {antes - len(df)} filas)")
            else:
                med = df["edad"].median()
                df["edad"] = df["edad"].fillna(med)
                tecnica.append(f"edad: IMPUTACIÓN por MEDIANA ({med:.0f})")

        # genero: moda
        df["genero"] = df["genero"].replace("", np.nan)
        moda_gen = _moda(df["genero"], "no especifica")
        df["genero"] = df["genero"].fillna(moda_gen)
        tecnica.append(f"genero: IMPUTACIÓN por MODA ('{moda_gen}')")

        # seniority: moda
        df["seniority"] = df["seniority"].replace("", np.nan)
        moda_sen = _moda(df["seniority"], "semi-senior")
        df["seniority"] = df["seniority"].fillna(moda_sen)
        tecnica.append(f"seniority: IMPUTACIÓN por MODA ('{moda_sen}')")

        # modalidad: moda
        df["modalidad"] = df["modalidad"].replace("", np.nan)
        moda_mod = _moda(df["modalidad"], "100% presencial")
        df["modalidad"] = df["modalidad"].fillna(moda_mod)
        tecnica.append(f"modalidad: IMPUTACIÓN por MODA ('{moda_mod}')")

        # tecnologias: ya quedó 'No especifica' en normalización
        tecnica.append("tecnologias: IMPUTACIÓN con 'No especifica'")

        # --- Duplicados / Golden Record ---
        n_pre_dup = len(df)
        df = _deduplicar_sysarmy(df)
        dups = n_pre_dup - len(df)
        tecnica.append(f"duplicados: {dups} eliminados (clave+timestamp / fuzzy)")

        # --- Outliers (NO se filtran, se documentan) ---
        q1 = df["salario_bruto_ars"].quantile(0.01)
        q99 = df["salario_bruto_ars"].quantile(0.99)
        df["es_outlier_sueldo"] = (df["salario_bruto_ars"] > q99) | (
            df["salario_bruto_ars"] < q1)
        n_out = int(df["es_outlier_sueldo"].sum())

        # limpiar columnas auxiliares
        df = df.drop(columns=["_clave_id", "_nombre"], errors="ignore")
        return df


def _a_bool(v) -> bool:
    if pd.isna(v):
        return False
    s = quitar_acentos(str(v)).lower().strip()
    return s in {"si", "sí", "yes", "true", "1", "x", "y"}


def _cobra_en_dolares(v) -> bool:
    """
    En Sysarmy 'pagos_en_dolares' es texto, no sí/no. Marcamos True sólo cuando
    la persona EFECTIVAMENTE recibe dólares (todo o parte del salario):
      'Cobro todo el salario en dólares'  -> True
      'Cobro parte del salario en dólares'-> True
      'Mi sueldo está dolarizado (pero cobro en moneda local)' -> False (no recibe USD)
      vacío -> False
    """
    if pd.isna(v):
        return False
    s = quitar_acentos(str(v)).lower().strip()
    if s in {"true", "si", "yes", "1"}:        # por si la fuente ya es booleana
        return True
    return "en dolares" in s                    # 'cobro ... en dolares'


def _moda(serie: pd.Series, default: str) -> str:
    s = serie.dropna()
    s = s[s != ""]
    if len(s) == 0:
        return default
    m = s.mode()
    return m.iloc[0] if len(m) else default


def _fecha_desde_nombre(nombre: str) -> pd.Timestamp:
    """Deriva una fecha de edición desde 'sysarmy_2025_2.csv' -> 2025-07-01."""
    m = re.search(r"(20\d{2})[_\-]?([12])", nombre)
    if m:
        anio = int(m.group(1))
        sem = int(m.group(2))
        mes = 1 if sem == 1 else 7
        return pd.Timestamp(year=anio, month=mes, day=1)
    return pd.Timestamp(year=datetime.now().year, month=1, day=1)


def _resumen_faltantes(d: dict[str, float], top: int = 12) -> str:
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return ", ".join(f"{k}={v}%" for k, v in items if v > 0) or "(sin faltantes relevantes)"


def _n_validos_texto(serie: pd.Series) -> int:
    """
    Cuenta cuántos valores son texto 'real' (no nulo, no vacío, no 'nan').

    Se filtran los nulos sobre la columna ORIGINAL con .notna() ANTES de
    convertir a str, porque con strings backed por pyarrow `np.nan.astype(str)`
    genera un 'nan' que `isin(["nan"])` no detecta correctamente.
    """
    s = serie[serie.notna()]
    if len(s) == 0:
        return 0
    s = s.astype(str).str.strip().str.lower()
    return int((~s.isin(["nan", "none", ""])).sum())


def _deduplicar_sysarmy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Golden Record según la identidad disponible:

      1) Si hay email/ID en >50% de las filas -> clave + timestamp (más reciente).
      2) Si hay nombre en >50% -> fuzzy firma (nombre + provincia + rol).
      3) Si NO hay identidad (encuesta anónima, caso Sysarmy) -> sólo se
         eliminan duplicados EXACTOS de fila completa. NO se deduplica por
         (provincia, rol) porque colapsaría miles de respuestas legítimas.
    """
    df = df.copy()
    n = len(df)

    n_id = _n_validos_texto(df["_clave_id"])
    if n_id > 0.5 * n:
        return df.sort_values("fecha_edicion").drop_duplicates(
            subset=["_clave_id"], keep="last")

    n_nombre = _n_validos_texto(df["_nombre"])
    if n_nombre > 0.5 * n:
        nombre_vals = df["_nombre"].astype(str).str.strip().str.lower()
        df["_firma"] = (
            nombre_vals + "|" +
            df["provincia"].fillna("").astype(str).str.lower() + "|" +
            df["rol"].fillna("").astype(str).str.lower()
        )
        df["_completitud"] = df.notna().sum(axis=1)
        df = df.sort_values(["fecha_edicion", "_completitud"]).drop_duplicates(
            subset=["_firma"], keep="last")
        return df.drop(columns=["_firma", "_completitud"], errors="ignore")

    # --- Datos anónimos: sólo duplicados exactos de fila completa ---
    cols_valor = [c for c in df.columns if not c.startswith("_")]
    return df.drop_duplicates(subset=cols_valor, keep="first")


# ----------------------------------------------------------------------------
# Helper común para series temporales (IPC, dólar, ripte, cbt, bigmac)
# ----------------------------------------------------------------------------
def limpiar_serie_temporal(
    path: Path,
    nombre: str,
    col_valor_cands: list[str],
    valor_final: str,
    *,
    fecha_dia: bool = False,
    no_negativos: bool = False,
    redondear: int | None = None,
    ffill: bool = False,
    eliminar_nulos: bool = False,
    imputar_media_si_menor_5: bool = False,
    agregar_mensual: bool = False,
) -> pd.DataFrame | None:
    rep_seccion(f"{nombre}")
    if not path.exists():
        log.warning("No se encontró %s. Se omite.", path.name)
        rep(f"ARCHIVO AUSENTE: {path.name}")
        return None
    try:
        df = pd.read_csv(path)
        n0 = len(df)
        col_fecha = resolver_columna(df, ["fecha", "indice_tiempo", "date",
                                          "periodo"])
        col_valor = resolver_columna(df, col_valor_cands)
        if col_fecha is None or col_valor is None:
            raise ValueError(
                f"No se hallaron columnas (fecha={col_fecha}, valor={col_valor}). "
                f"Columnas: {list(df.columns)}")

        out = pd.DataFrame()
        out["fecha"] = pd.to_datetime(df[col_fecha], errors="coerce")
        out[valor_final] = a_numero(df[col_valor])

        faltantes_antes = pct_faltantes(out)

        # fecha al primer día del mes salvo que se pida nivel día
        if not fecha_dia:
            out["fecha"] = out["fecha"].dt.to_period("M").dt.to_timestamp()

        # nulos en fecha siempre se eliminan
        out = out[out["fecha"].notna()]

        tecnica: list[str] = []

        if no_negativos:
            out = out[out[valor_final] >= 0]
            tecnica.append("filtrado de valores negativos")

        # forward fill
        if ffill:
            out = out.sort_values("fecha")
            out[valor_final] = out[valor_final].ffill()
            tecnica.append("FORWARD FILL de valores faltantes")

        # imputación media si <5%, sino eliminación
        if imputar_media_si_menor_5:
            pct = out[valor_final].isna().mean() * 100
            if pct < 5:
                media = out[valor_final].mean()
                out[valor_final] = out[valor_final].fillna(media)
                tecnica.append(f"IMPUTACIÓN por MEDIA ({media:.2f}) [{pct:.1f}% faltante]")
            else:
                out = out[out[valor_final].notna()]
                tecnica.append(f"ELIMINACIÓN de nulos [{pct:.1f}% > 5%]")

        if eliminar_nulos:
            antes = len(out)
            out = out[out[valor_final].notna()]
            tecnica.append(f"ELIMINACIÓN de nulos ({antes - len(out)} filas)")

        # duplicados por fecha -> mantener último
        antes = len(out)
        out = out.sort_values("fecha").drop_duplicates(subset=["fecha"],
                                                       keep="last")
        dups = antes - len(out)

        # agregación mensual (promedio)
        if agregar_mensual:
            out["fecha"] = out["fecha"].dt.to_period("M").dt.to_timestamp()
            out = (out.groupby("fecha", as_index=False)[valor_final]
                   .mean())
            tecnica.append("AGREGACIÓN mensual (promedio)")

        if redondear is not None:
            out[valor_final] = out[valor_final].round(redondear)

        out = out.sort_values("fecha").reset_index(drop=True)

        rep(f"Registros: {n0} -> {len(out)}")
        rep(f"Columnas usadas: fecha='{col_fecha}', valor='{col_valor}'")
        rep(f"Duplicados por fecha eliminados: {dups}")
        rep("Técnicas: " + ("; ".join(tecnica) if tecnica else "ninguna"))
        if len(out):
            rep(f"Rango de fechas: {out['fecha'].min().date()} -> "
                f"{out['fecha'].max().date()}")
        rep("% faltantes ANTES: " + _resumen_faltantes(faltantes_antes))
        rep("% faltantes DESPUÉS: " + _resumen_faltantes(pct_faltantes(out)))
        return out
    except Exception as exc:  # noqa: BLE001
        log.error("FALLO limpieza %s: %s", nombre, exc)
        rep(f"ERROR: {exc}")
        return None


# ----------------------------------------------------------------------------
# FUENTE 7: STACK OVERFLOW
# ----------------------------------------------------------------------------
def limpiar_stackoverflow() -> pd.DataFrame | None:
    nombre = "Stack Overflow (Argentina)"
    rep_seccion(f"FUENTE 7 — {nombre}")
    if not F_STACKOVERFLOW.exists():
        log.warning("No se encontró %s (descarga manual). Se omite.",
                    F_STACKOVERFLOW.name)
        rep(f"ARCHIVO AUSENTE: {F_STACKOVERFLOW.name} (descarga manual).")
        return None
    try:
        df = pd.read_csv(F_STACKOVERFLOW, low_memory=False)
        n0 = len(df)

        c_country = resolver_columna(df, ["Country", "pais"])
        c_comp = resolver_columna(df, ["ConvertedCompYearly", "salario_anual_usd",
                                       "compensation"])
        if c_country is None or c_comp is None:
            raise ValueError("Faltan columnas Country/ConvertedCompYearly.")

        # filtrado: Argentina + con salario
        df = df[df[c_country].astype(str).str.strip().str.lower() == "argentina"]
        n_arg = len(df)
        df = df[df[c_comp].notna()]
        n_sal = len(df)

        mapa = {
            "id": ["ResponseId", "id"],
            "edad": ["Age", "edad"],
            "nivel_estudios": ["EdLevel", "nivel_estudios"],
            "empleo": ["Employment", "empleo"],
            "anos_codigo": ["YearsCode", "anos_codigo"],
            "rol": ["DevType", "rol"],
            "modalidad": ["RemoteWork", "modalidad"],
            "pais": ["Country", "pais"],
            "salario_anual_usd": ["ConvertedCompYearly", "salario_anual_usd"],
            "tecnologias": ["LanguageHaveWorkedWith", "tecnologias"],
        }
        out = pd.DataFrame()
        for destino, cands in mapa.items():
            col = resolver_columna(df, cands)
            out[destino] = df[col].values if col is not None else np.nan

        # tipos
        out["edad"] = _edad_so(out["edad"])
        out["salario_anual_usd"] = a_numero(out["salario_anual_usd"])
        out["salario_mensual_usd"] = (out["salario_anual_usd"] / 12).round(2)
        out["tecnologias"] = out["tecnologias"].map(normalizar_tecnologias)
        out["modalidad"] = out["modalidad"].map(
            lambda v: normalizar_modalidad(v) or "No especifica")
        out["rol"] = out["rol"].map(normalizar_rol)

        # faltantes: eliminar sin edad o sin salario
        antes = len(out)
        out = out[out["edad"].notna() & out["salario_anual_usd"].notna()]
        elim = antes - len(out)

        # edad 18-75
        out = out[(out["edad"] >= 18) & (out["edad"] <= 75)]

        # imputar resto por moda / 'No especifica'
        for col in ["nivel_estudios", "empleo", "anos_codigo"]:
            if col in out:
                out[col] = out[col].fillna(_moda(out[col], "No especifica"))

        # duplicados por id
        antes = len(out)
        if out["id"].notna().any():
            out = out.drop_duplicates(subset=["id"], keep="first")
        dups = antes - len(out)

        # outliers (NO filtrar)
        if len(out):
            q1 = out["salario_mensual_usd"].quantile(0.01)
            q99 = out["salario_mensual_usd"].quantile(0.99)
            out["es_outlier_sueldo"] = (out["salario_mensual_usd"] > q99) | (
                out["salario_mensual_usd"] < q1)

        rep(f"Registros: {n0} -> Argentina={n_arg} -> con salario={n_sal} -> final={len(out)}")
        rep(f"Eliminados sin edad/salario: {elim}; duplicados por id: {dups}")
        if len(out):
            rep(f"Rango salario mensual USD: {out['salario_mensual_usd'].min():,.0f} "
                f"- {out['salario_mensual_usd'].max():,.0f}")

        guardar_parquet(out, PROC_DIR / "stackoverflow_argentina_limpio.parquet",
                        nombre)
        return out
    except Exception as exc:  # noqa: BLE001
        log.error("FALLO limpieza %s: %s", nombre, exc)
        rep(f"ERROR: {exc}")
        return None


def _edad_so(serie: pd.Series) -> pd.Series:
    """Stack Overflow 'Age' suele venir como rango textual; lo aproximamos."""
    mapa = {
        "under 18 years old": 17,
        "18-24 years old": 21,
        "25-34 years old": 29,
        "35-44 years old": 39,
        "45-54 years old": 49,
        "55-64 years old": 59,
        "65 years or older": 67,
    }

    def _conv(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in mapa:
            return mapa[s]
        try:
            return float(s)
        except ValueError:
            return np.nan

    return serie.map(_conv)


# ----------------------------------------------------------------------------
# PARTE 2: MERGE
# ----------------------------------------------------------------------------
def merge_fuentes(
    sysarmy: pd.DataFrame | None,
    ipc: pd.DataFrame | None,
    dolar: pd.DataFrame | None,
    ripte: pd.DataFrame | None,
    cbt: pd.DataFrame | None,
    bigmac: pd.DataFrame | None,
    itcrm: pd.DataFrame | None = None,
    uscpi: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rep_seccion("PARTE 2 — MERGE Y UNIFICACIÓN")
    if sysarmy is None or len(sysarmy) == 0:
        log.warning("Sin base Sysarmy: el dataset final quedará vacío.")
        rep("Base Sysarmy ausente: no es posible construir el dataset unificado.")
        return pd.DataFrame()

    df = sysarmy.copy()
    df["_mes"] = df["fecha_edicion"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("_mes").reset_index(drop=True)

    def _join(base: pd.DataFrame, otra: pd.DataFrame | None, etiqueta: str,
              guardar_fecha_en: str | None = None):
        """
        Une una serie macro mensual usando merge_asof con dirección 'nearest':
        para cada fila toma el valor del mes MÁS CERCANO disponible. Así un mes
        sin dato exacto (p.ej. el dólar que no cubre todos los meses) ya no
        produce NaN, que era el bug que dejaba la dimensión USD vacía.
        """
        if otra is None or len(otra) == 0:
            rep(f"  - {etiqueta}: omitido (sin datos)")
            return base
        tmp = otra.copy()
        tmp["_mes"] = tmp["fecha"].dt.to_period("M").dt.to_timestamp()
        tmp = tmp.sort_values("_mes")
        col_valor = [c for c in tmp.columns if c not in ("fecha", "_mes")][0]
        if guardar_fecha_en:
            tmp = tmp.rename(columns={"fecha": guardar_fecha_en})
        else:
            tmp = tmp.drop(columns=["fecha"])
        base = pd.merge_asof(base, tmp, on="_mes", direction="nearest")
        cob = base[col_valor].notna().mean() * 100
        rep(f"  - {etiqueta}: merge_asof nearest (cobertura {cob:.1f}%)")
        return base

    rep("Merge de series macro sobre fecha_edicion (mes más cercano):")
    df = _join(df, ipc, "IPC")
    df = _join(df, dolar, "Dólar MEP mensual")
    df = _join(df, ripte, "RIPTE mensual")
    df = _join(df, cbt, "CBT mensual")
    df = _join(df, bigmac, "Big Mac", guardar_fecha_en="fecha_bigmac")
    df = _join(df, itcrm, "ITCRM")
    df = _join(df, uscpi, "US CPI")

    df = df.drop(columns=["_mes"], errors="ignore")
    rep(f"Filas tras merge: {len(df)}; columnas: {df.shape[1]}")
    return df


# ----------------------------------------------------------------------------
# PARTE 3: COLUMNAS DERIVADAS
# ----------------------------------------------------------------------------
def _valor_en_fecha(serie: pd.DataFrame | None, col: str,
                    fecha: pd.Timestamp) -> float | None:
    """Valor de una serie mensual en (o más cercano a) `fecha`."""
    if serie is None or len(serie) == 0 or col not in serie.columns:
        return None
    s = serie.dropna(subset=[col]).sort_values("fecha")
    if len(s) == 0:
        return None
    idx = (s["fecha"] - pd.Timestamp(fecha)).abs().idxmin()
    return float(s.loc[idx, col])


def columnas_derivadas(df: pd.DataFrame,
                       base_ref: dict | None = None) -> pd.DataFrame:
    rep_seccion("PARTE 3 — COLUMNAS DERIVADAS")
    if len(df) == 0:
        rep("Sin filas: no se calculan derivadas.")
        return df

    base_ref = base_ref or {}
    ipc_base = base_ref.get("ipc") or IPC_BASE
    uscpi_base = base_ref.get("us_cpi")
    mep_base = base_ref.get("dolar")
    fbase = base_ref.get("fecha", FECHA_BASE_REAL)
    etiqueta_base = pd.Timestamp(fbase).strftime("%Y-%m")
    rep(f"BASE de ajuste: {etiqueta_base} "
        f"(IPC={ipc_base:,.1f}, MEP={mep_base}, US_CPI={uscpi_base})")

    sal = df.get("salario_bruto_ars")

    # SALARIO REAL según la MONEDA NATIVA del sueldo (cobra_en_dolares):
    #   - Si es en PESOS:  se deflacta por inflación ARGENTINA (IPC) y luego se
    #     convierte a USD al MEP base.
    #   - Si es en DÓLARES: se recupera el USD del mes (nominal/MEP_mes), se
    #     deflacta por inflación de EE.UU. (US CPI) y luego se pasa a pesos al
    #     MEP base.
    # En ambos casos salario_real_ars / salario_real_usd = MEP_base (consistente).
    necesarias = {"ipc", "dolar_mep", "us_cpi"}
    if sal is not None and necesarias <= set(df.columns) and mep_base and uscpi_base:
        cobra = df["cobra_en_dolares"].fillna(False).astype(bool)

        # camino PESOS-nativo -> real en pesos por inflación argentina
        real_ars_pesos = sal * (ipc_base / df["ipc"])
        # camino DÓLAR-nativo -> real en USD por inflación US, luego a pesos
        usd_edicion = sal / df["dolar_mep"]
        real_usd_dolar = usd_edicion * (uscpi_base / df["us_cpi"])
        real_ars_dolar = real_usd_dolar * mep_base

        real_ars = real_ars_pesos.where(~cobra, real_ars_dolar)
        df["salario_real_ars"] = real_ars.round(2)
        df["salario_real_usd"] = (real_ars / mep_base).round(2)
        n_usd = int(cobra.sum())
        rep(f"salario_real_ars / salario_real_usd según moneda nativa "
            f"({n_usd} dolarizados por US CPI, resto por IPC). "
            f"Cociente ARS/USD = MEP_base ({mep_base:,.1f}).")

    # CANASTAS BÁSICAS que cubre el salario en su propio mes (salario / CBT del
    # momento de la encuesta). Es un ratio de valores del mismo período, así que
    # ya es "real" y comparable entre ediciones sin necesidad de base.
    if "cbt" in df and sal is not None:
        df["canastas_basicas"] = (sal / df["cbt"]).round(2)
        df["cobertura_cbt"] = pd.cut(
            df["canastas_basicas"], bins=[-np.inf, 1, 2, np.inf],
            labels=["crítica", "ajustada", "holgada"],
        ).astype("object")
        rep("canastas_basicas = salario_nominal / CBT del mes (+ cobertura_cbt)")

    # es_outlier (sobre el salario real en pesos, ya comparable entre ediciones)
    if "salario_real_ars" in df:
        r = df["salario_real_ars"]
        q1, q99 = r.quantile(0.01), r.quantile(0.99)
        df["es_outlier"] = (r > q99) | (r < q1)
        rep(f"es_outlier: salario_real_ars > Q99({q99:,.0f}) o < Q1({q1:,.0f})")
    return df


# ----------------------------------------------------------------------------
# PARTE 4: OUTPUT
# ----------------------------------------------------------------------------
def escribir_salidas(final: pd.DataFrame, so: pd.DataFrame | None) -> pd.DataFrame:
    rep_seccion("PARTE 4 — OUTPUT FINAL")

    # 1) separar el CONTEXTO MACROECONÓMICO (una fila por edición, FK fecha_edicion)
    ctx_cols = [c for c in COLS_CONTEXTO_EDICION if c in final.columns]
    if "fecha_edicion" in final.columns and len(ctx_cols) > 1:
        contexto = (final[ctx_cols].drop_duplicates()
                    .sort_values("fecha_edicion").reset_index(drop=True))
        guardar_parquet(contexto, PROC_DIR / "contexto_macroeconomico.parquet",
                        "contexto_macroeconomico")
        rep(f"contexto_macroeconomico.parquet: {len(contexto)} fila(s) x "
            f"{contexto.shape[1]} cols (macro por edición, FK fecha_edicion)")
        a_quitar = [c for c in ctx_cols if c != "fecha_edicion"]
        final = final.drop(columns=a_quitar)

    # 2) descartar nominal + métricas de poder adquisitivo: el dataset final
    #    guarda SÓLO salario_real_ars y salario_real_usd como medidas de salario.
    descartar = [c for c in COLS_DESCARTAR_FINAL if c in final.columns]
    if descartar:
        final = final.drop(columns=descartar)
        rep(f"Columnas descartadas del final (nominal + poder adq.): {descartar}")

    destino = PROC_DIR / "dataset_final_mercado_laboral.parquet"
    guardar_parquet(final, destino, "dataset_final")
    rep(f"dataset_final_mercado_laboral.parquet: {len(final)} filas, "
        f"{final.shape[1]} columnas")

    # Los outliers NO se exportan a un CSV aparte: quedan DENTRO del dataset
    # final marcados con la columna `es_outlier` (sólo se descartaron los
    # errores de carga muy evidentes durante la limpieza).
    if "es_outlier" in final.columns:
        n_out = int((final["es_outlier"] == True).sum())  # noqa: E712
        rep(f"Outliers conservados dentro del dataset final (es_outlier=True): {n_out}")

    if so is not None:
        rep(f"stackoverflow_argentina_limpio.parquet: {len(so)} filas "
            "(guardado SEPARADO, solo benchmark)")

    # reporte de calidad
    reporte_path = PROC_DIR / "data_quality_report.txt"
    encabezado = [
        "REPORTE DE CALIDAD DE DATOS — Pipeline ETL Mercado Laboral Tech AR",
        f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}",
    ]
    try:
        reporte_path.write_text("\n".join(encabezado + REPORTE) + "\n",
                                encoding="utf-8")
        log.info("data_quality_report.txt escrito (%d líneas)", len(REPORTE))
    except Exception as exc:  # noqa: BLE001
        log.error("No se pudo escribir el reporte: %s", exc)

    return final


# ----------------------------------------------------------------------------
# FUENTE 3 (histórico) y FUENTE 8 (ITCRM)
# ----------------------------------------------------------------------------
def limpiar_dolar_historico() -> pd.DataFrame | None:
    """
    Dólar mensual con historia completa. Usa el parquet de bluelytics (2011-2026,
    fuente 'Blue' como proxy de MEP, porque el CSV de MEP sólo cubre 2026 y no
    alcanza para las ediciones viejas).
    """
    rep_seccion("FUENTE 3 — Dólar (Blue ≈ MEP, histórico)")
    if F_DOLAR_PARQUET.exists():
        try:
            p = pd.read_parquet(F_DOLAR_PARQUET)
            p["date"] = pd.to_datetime(p["date"], errors="coerce")
            blue = p[p["source"].astype(str).str.lower() == "blue"].copy()
            if len(blue) == 0:
                blue = p.copy()
            blue["valor"] = blue[["value_sell", "value_buy"]].mean(axis=1)
            blue["fecha"] = blue["date"].dt.to_period("M").dt.to_timestamp()
            out = (blue.groupby("fecha", as_index=False)["valor"].mean()
                   .rename(columns={"valor": "dolar_mep"}))
            out["dolar_mep"] = out["dolar_mep"].round(2)
            out = out.sort_values("fecha").reset_index(drop=True)
            rep(f"Fuente: {F_DOLAR_PARQUET.name} (Blue). {len(out)} meses "
                f"({out['fecha'].min().date()} -> {out['fecha'].max().date()})")
            return out
        except Exception as exc:  # noqa: BLE001
            log.error("FALLO dólar histórico: %s", exc)
            rep(f"ERROR: {exc}")
    # fallback: el CSV de sólo-2026
    return limpiar_serie_temporal(
        F_DOLAR, "FUENTE 3 — Dólar MEP (CSV fallback)",
        ["dolar_mep", "valor", "venta", "value"], "dolar_mep",
        fecha_dia=True, ffill=True, redondear=2, agregar_mensual=True)


def limpiar_itcrm() -> pd.DataFrame | None:
    """Lee el ITCRM del BCRA (hoja mensual) y lo deja como serie fecha/itcrm."""
    rep_seccion("FUENTE 8 — ITCRM (competitividad real, BCRA)")
    if not F_ITCRM.exists():
        log.warning("No se encontró %s. Se omite.", F_ITCRM.name)
        rep(f"ARCHIVO AUSENTE: {F_ITCRM.name}")
        return None
    try:
        xl = pd.ExcelFile(F_ITCRM)
        hoja = next((s for s in xl.sheet_names if "mens" in s.lower()),
                    xl.sheet_names[0])
        crudo = pd.read_excel(xl, sheet_name=hoja, header=None)
        # localizar la fila de encabezado (la que tiene 'Período' e 'ITCRM')
        hdr = 0
        for i in range(min(12, len(crudo))):
            fila = crudo.iloc[i].astype(str).str.lower()
            if fila.str.contains("per").any() and fila.str.contains("itcrm").any():
                hdr = i
                break
        df = pd.read_excel(xl, sheet_name=hoja, header=hdr)
        col_f = resolver_columna(df, ["periodo", "fecha", "indice_tiempo"])
        col_v = resolver_columna(df, ["itcrm"])
        if col_f is None or col_v is None:
            raise ValueError(f"No se hallaron columnas (fecha={col_f}, itcrm={col_v})")
        out = pd.DataFrame()
        out["fecha"] = (pd.to_datetime(df[col_f], errors="coerce")
                        .dt.to_period("M").dt.to_timestamp())
        out["itcrm"] = a_numero(df[col_v])
        out = out[out["fecha"].notna() & out["itcrm"].notna()]
        out = (out.drop_duplicates("fecha", keep="last")
               .sort_values("fecha").reset_index(drop=True))
        rep(f"Hoja '{hoja}'. {len(out)} meses "
            f"({out['fecha'].min().date()} -> {out['fecha'].max().date()})")
        return out
    except Exception as exc:  # noqa: BLE001
        log.error("FALLO ITCRM: %s", exc)
        rep(f"ERROR: {exc}")
        return None


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Iniciando pipeline ETL. raw=%s  processed=%s", RAW_DIR, PROC_DIR)
    if not _RAPIDFUZZ:
        log.warning("rapidfuzz no disponible: normalización fuzzy degradada.")

    # PARTE 1 — limpieza por fuente
    sysarmy = limpiar_sysarmy()

    ipc = limpiar_serie_temporal(
        F_IPC, "FUENTE 2 — IPC INDEC",
        ["ipc", "ipc_2016_nivgeneral", "valor", "indice"], "ipc",
        eliminar_nulos=True,
    )
    # Nota: las series macro vienen limpias (datos oficiales/API). Se transforman
    # en memoria para el merge pero NO se guardan parquets intermedios: los
    # valores quedan en contexto_macroeconomico.parquet. (Ahorra disco.)
    if ipc is not None and len(ipc):
        ipc = ipc.sort_values("fecha")
        ipc["inflacion_mensual_pct"] = (ipc["ipc"].pct_change() * 100).round(2)

    dolar = limpiar_dolar_historico()

    bigmac = limpiar_serie_temporal(
        F_BIGMAC, "FUENTE 4 — Big Mac Index",
        ["precio_bigmac_ars", "local_price", "precio", "valor"],
        "precio_bigmac_ars",
        fecha_dia=True, redondear=2, eliminar_nulos=True,
    )

    ripte = limpiar_serie_temporal(
        F_RIPTE, "FUENTE 5 — RIPTE",
        ["ripte", "valor", "value"], "ripte",
        no_negativos=True, imputar_media_si_menor_5=True, agregar_mensual=True,
    )

    cbt = limpiar_serie_temporal(
        F_CBT, "FUENTE 6 — CBT INDEC",
        ["cbt", "gran_buenos_aires", "valor", "value"], "cbt",
        no_negativos=True, eliminar_nulos=True,
    )

    itcrm = limpiar_itcrm()

    uscpi = limpiar_serie_temporal(
        F_USCPI, "FUENTE 9 — US CPI (inflación EE.UU.)",
        ["cpiaucsl", "us_cpi", "valor", "value"], "us_cpi",
        eliminar_nulos=True,
    )

    stackoverflow = limpiar_stackoverflow()

    # PARTE 2 — merge
    final = merge_fuentes(sysarmy, ipc, dolar, ripte, cbt, bigmac, itcrm, uscpi)

    # PARTE 3 — derivadas (todo ajustado a FECHA_BASE_REAL)
    base_ref = {
        "fecha": FECHA_BASE_REAL,
        "ipc": _valor_en_fecha(ipc, "ipc", FECHA_BASE_REAL),
        "dolar": _valor_en_fecha(dolar, "dolar_mep", FECHA_BASE_REAL),
        "itcrm": _valor_en_fecha(itcrm, "itcrm", FECHA_BASE_REAL),
        "us_cpi": _valor_en_fecha(uscpi, "us_cpi", FECHA_BASE_REAL),
    }
    final = columnas_derivadas(final, base_ref)

    # PARTE 4 — output
    final = escribir_salidas(final, stackoverflow)

    # ----- Resumen en consola -----
    print("\n" + "=" * 64)
    print("RESUMEN DEL PIPELINE ETL")
    print("=" * 64)
    def _info(nombre, df):
        if df is None:
            print(f"  ✗ {nombre}: AUSENTE / no procesado")
        else:
            print(f"  ✓ {nombre}: {len(df)} filas, {df.shape[1]} cols")
    _info("Sysarmy", sysarmy)
    _info("IPC", ipc)
    _info("Dólar MEP mensual", dolar)
    _info("Big Mac", bigmac)
    _info("RIPTE", ripte)
    _info("CBT", cbt)
    _info("ITCRM", itcrm)
    _info("US CPI", uscpi)
    _info("Stack Overflow AR", stackoverflow)
    print("-" * 64)
    print(f"  DATASET FINAL: {len(final)} filas, "
          f"{final.shape[1] if len(final.columns) else 0} columnas")
    print(f"  -> {PROC_DIR / 'dataset_final_mercado_laboral.parquet'}")
    print(f"  -> {PROC_DIR / 'data_quality_report.txt'}")
    if "es_outlier" in final.columns:
        print(f"  (outliers conservados dentro del dataset, columna 'es_outlier')")
    print("=" * 64)


if __name__ == "__main__":
    main()
