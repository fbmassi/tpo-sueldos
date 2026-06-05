"""
descargar_datos.py
==================

Descarga automáticamente las fuentes de datos públicas (vía API) necesarias
para el TPO y las guarda en data/raw/.

------------------------------------------------------------------------------
IMPORTANTE — ARCHIVOS QUE DEBEN DESCARGARSE MANUALMENTE ANTES DE CORRER ESTE
SCRIPT (no tienen API pública / requieren registro o aceptar términos):

  1. sysarmy_2025_2.csv      -> sysarmy.com/blog -> edición más reciente
  2. datosInternacionales.csv -> survey.stackoverflow.co/2024
  3. serie_ipc_divisiones.csv -> indec.gob.ar (ya hay referencias en notebooks)
  4. ITCRMSerie.xlsx          -> bcra.gob.ar/publicacionesestadisticas/
  5. usu_individual_T224.txt  -> indec.gob.ar/EPH/microdatos

Guardarlos en data/raw/ y luego correr este script:

    python data/descargar_datos.py
------------------------------------------------------------------------------

Descargas automáticas (este script):
  1. IPC INDEC Nacional      -> data/raw/ipc_indec.csv
  2. Dólar MEP (Bluelytics)  -> data/raw/dolar_mep.parquet
  3. Big Mac Index           -> data/raw/bigmac_index.csv
  4. RIPTE                   -> data/raw/ripte.csv
  5. Canasta Básica Total    -> data/raw/cbt.csv
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
RAW_DIR = Path(__file__).resolve().parent / "raw"
TIMEOUT = 60  # segundos

# La API de Series de Tiempo de datos.gob.ar devuelve 100 filas por defecto;
# pedimos el máximo para traer la serie completa.
API_SERIES_LIMIT = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("descargar_datos")

# Resumen final: nombre -> (ok: bool, detalle: str)
resumen: dict[str, tuple[bool, str]] = {}


def _get(url: str) -> requests.Response:
    """GET con timeout y raise_for_status."""
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp


# ----------------------------------------------------------------------------
# 1. IPC INDEC Nacional
# ----------------------------------------------------------------------------
def descargar_ipc_indec() -> None:
    nombre = "IPC INDEC Nacional"
    destino = RAW_DIR / "ipc_indec.csv"
    url = (
        "https://apis.datos.gob.ar/series/api/series/"
        f"?ids=103.1_I2N_2016_M_19&format=csv&limit={API_SERIES_LIMIT}"
    )
    try:
        resp = _get(url)
        df = pd.read_csv(io.StringIO(resp.text))
        df.to_csv(destino, index=False)
        resumen[nombre] = (True, f"{len(df)} filas -> {destino.name}")
        log.info("OK %s: %d filas guardadas en %s", nombre, len(df), destino)
    except Exception as exc:  # noqa: BLE001 - queremos seguir con el resto
        resumen[nombre] = (False, str(exc))
        log.error("FALLO %s: %s", nombre, exc)


# ----------------------------------------------------------------------------
# 2. Cotizaciones Dólar MEP (Bluelytics)
# ----------------------------------------------------------------------------
def descargar_dolar_mep() -> None:
    nombre = "Dólar MEP (Bluelytics)"
    destino = RAW_DIR / "dolar_mep.parquet"
    url = "https://api.bluelytics.com.ar/v2/evolution.json"
    try:
        resp = _get(url)
        df = pd.DataFrame(resp.json())
        # La API devuelve todas las cotizaciones; nos interesa el MEP.
        if "source" in df.columns:
            mep = df[df["source"].str.contains("MEP", case=False, na=False)]
            if not mep.empty:
                df = mep
        df.to_parquet(destino, index=False)
        resumen[nombre] = (True, f"{len(df)} filas -> {destino.name}")
        log.info("OK %s: %d filas guardadas en %s", nombre, len(df), destino)
    except Exception as exc:  # noqa: BLE001
        resumen[nombre] = (False, str(exc))
        log.error("FALLO %s: %s", nombre, exc)


# ----------------------------------------------------------------------------
# 3. Big Mac Index
# ----------------------------------------------------------------------------
def descargar_bigmac() -> None:
    nombre = "Big Mac Index"
    destino = RAW_DIR / "bigmac_index.csv"
    url = (
        "https://raw.githubusercontent.com/TheEconomist/big-mac-data/"
        "master/output-data/big-mac-full-index.csv"
    )
    try:
        resp = _get(url)
        df = pd.read_csv(io.StringIO(resp.text))
        df.to_csv(destino, index=False)
        resumen[nombre] = (True, f"{len(df)} filas -> {destino.name}")
        log.info("OK %s: %d filas guardadas en %s", nombre, len(df), destino)
    except Exception as exc:  # noqa: BLE001
        resumen[nombre] = (False, str(exc))
        log.error("FALLO %s: %s", nombre, exc)


# ----------------------------------------------------------------------------
# 4. RIPTE (API Series de Tiempo datos.gob.ar)
# ----------------------------------------------------------------------------
def descargar_ripte() -> None:
    nombre = "RIPTE"
    destino = RAW_DIR / "ripte.csv"
    # Remuneración imponible promedio de los trabajadores estables.
    url = (
        "https://apis.datos.gob.ar/series/api/series/"
        f"?ids=158.1_REPTE_0_0_5&format=csv&limit={API_SERIES_LIMIT}"
    )
    try:
        resp = _get(url)
        df = pd.read_csv(io.StringIO(resp.text))
        df.to_csv(destino, index=False)
        resumen[nombre] = (True, f"{len(df)} filas -> {destino.name}")
        log.info("OK %s: %d filas guardadas en %s", nombre, len(df), destino)
    except Exception as exc:  # noqa: BLE001
        resumen[nombre] = (False, str(exc))
        log.error("FALLO %s: %s", nombre, exc)


# ----------------------------------------------------------------------------
# 5. Canasta Básica Total (API Series de Tiempo datos.gob.ar)
# ----------------------------------------------------------------------------
def descargar_cbt() -> None:
    nombre = "Canasta Básica Total"
    destino = RAW_DIR / "cbt.csv"
    # Canasta Básica Total - Gran Buenos Aires.
    url = (
        "https://apis.datos.gob.ar/series/api/series/"
        f"?ids=444.1_CANASTA_batotGBA_0_0_26_47&format=csv&limit={API_SERIES_LIMIT}"
    )
    try:
        resp = _get(url)
        df = pd.read_csv(io.StringIO(resp.text))
        df.to_csv(destino, index=False)
        resumen[nombre] = (True, f"{len(df)} filas -> {destino.name}")
        log.info("OK %s: %d filas guardadas en %s", nombre, len(df), destino)
    except Exception as exc:  # noqa: BLE001
        resumen[nombre] = (False, str(exc))
        log.error("FALLO %s: %s", nombre, exc)


# ----------------------------------------------------------------------------
# 6. US CPI (inflación de EE.UU. — FRED, serie CPIAUCSL)
# ----------------------------------------------------------------------------
def descargar_us_cpi() -> None:
    nombre = "US CPI (inflación USD)"
    destino = RAW_DIR / "us_cpi.csv"
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    try:
        resp = _get(url)
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["fecha", "us_cpi"]
        df.to_csv(destino, index=False)
        resumen[nombre] = (True, f"{len(df)} filas -> {destino.name}")
        log.info("OK %s: %d filas guardadas en %s", nombre, len(df), destino)
    except Exception as exc:  # noqa: BLE001
        resumen[nombre] = (False, str(exc))
        log.error("FALLO %s: %s", nombre, exc)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Directorio de destino: %s", RAW_DIR)

    descargar_ipc_indec()
    descargar_dolar_mep()
    descargar_bigmac()
    descargar_ripte()
    descargar_cbt()
    descargar_us_cpi()

    # ----- Resumen final -----
    print("\n" + "=" * 60)
    print("RESUMEN DE DESCARGAS")
    print("=" * 60)
    ok_count = 0
    for nombre, (ok, detalle) in resumen.items():
        estado = "✓ OK  " if ok else "✗ FALLO"
        ok_count += int(ok)
        print(f"  {estado}  {nombre}: {detalle}")
    print("-" * 60)
    print(f"  {ok_count}/{len(resumen)} descargas exitosas")
    print("=" * 60)


if __name__ == "__main__":
    main()
