"""
notebooks/fuentes_raw.py
========================

Fuentes crudas versionables: espejo .parquet de cada CSV de data/raw/.

Los CSV crudos están en .gitignore (`*.csv`) y algunos superan el límite de
GitHub (datosInternacionales.csv pesa 140 MB), así que no viajan con el repo:
quien clona no puede correr el pipeline. Este módulo resuelve eso guardando un
gemelo .parquet LOSSLESS de cada CSV (todas las celdas como texto, sin
interpretar) que sí se versiona (140 MB → 11 MB).

Uso:
    # convertir todos los CSV de data/raw a sus gemelos .parquet
    python notebooks/fuentes_raw.py

    # en los scripts, en lugar de pd.read_csv(path, ...):
    from fuentes_raw import leer_csv, existe
    df = leer_csv(path, skiprows=3, low_memory=False)   # mismos kwargs

`leer_csv` usa el CSV si está en el disco; si no, reconstruye el contenido
exacto desde el gemelo .parquet y le aplica los mismos argumentos de
`pd.read_csv` (skiprows, usecols, etc.), por lo que el resto del pipeline no
cambia en nada.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _gemelo(path: Path) -> Path:
    return path.with_suffix(".parquet")


def existe(path: Path) -> bool:
    """¿Está la fuente disponible (CSV local o gemelo parquet versionado)?"""
    return path.exists() or _gemelo(path).exists()


def csv_a_parquet(path: Path) -> Path:
    """Convierte un CSV a su gemelo .parquet sin interpretar nada (lossless)."""
    crudo = pd.read_csv(path, header=None, dtype=str,
                        keep_default_na=False, low_memory=False)
    crudo.columns = [str(c) for c in crudo.columns]
    out = _gemelo(path)
    crudo.to_parquet(out, index=False)
    return out


def leer_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Como pd.read_csv, pero cae al gemelo .parquet si el CSV no está."""
    if path.exists():
        return pd.read_csv(path, **kwargs)
    gem = _gemelo(path)
    if not gem.exists():
        raise FileNotFoundError(f"No existe {path.name} ni su gemelo {gem.name}")
    crudo = pd.read_parquet(gem)
    buf = io.StringIO()
    crudo.to_csv(buf, index=False, header=False)
    buf.seek(0)
    return pd.read_csv(buf, **kwargs)


def main() -> None:
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        print("No hay CSV en data/raw/ para convertir.")
        return
    for f in csvs:
        if _gemelo(f).exists():
            # p. ej. dolar_mep.parquet: fuente tipada ya versionada, no se pisa
            print(f"• {f.name:32s} → ya tiene gemelo parquet, no se pisa")
            continue
        out = csv_a_parquet(f)
        print(f"✓ {f.name:32s} → {out.name:36s} "
              f"({f.stat().st_size/1024:8.0f} KB → {out.stat().st_size/1024:7.0f} KB)")


if __name__ == "__main__":
    main()
