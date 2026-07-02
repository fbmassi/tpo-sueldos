"""
notebooks/preparar_mundo.py
===========================

Precomputa un agregado LIVIANO de la encuesta global de Stack Overflow para la
comparación "Argentina vs el mundo".

El CSV crudo (`data/raw/datosInternacionales.csv`, ~140 MB) NO se sube a GitHub
(está en .gitignore y además supera el límite de 100 MB por archivo), por lo que
las apps desplegadas en la nube no lo encuentran. Este script filtra ese CSV a
los países que la app compara y guarda un parquet de pocas KB que SÍ se versiona
y viaja al deploy.

Salida: data/processed/stackoverflow_mundo.parquet
Uso:    python notebooks/preparar_mundo.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SO_CSV = ROOT / "data" / "raw" / "datosInternacionales.csv"
OUT = ROOT / "data" / "processed" / "stackoverflow_mundo.parquet"

# Países que comparan las apps (deben coincidir con PAISES en app/pages/04_mundo.py)
PAISES = [
    "United States of America", "Germany", "Canada",
    "United Kingdom of Great Britain and Northern Ireland", "Netherlands",
    "France", "Spain", "Poland", "Brazil", "India", "Ukraine", "Australia",
    "Italy", "Mexico", "Chile", "Uruguay", "Colombia", "Portugal",
]


def main() -> None:
    df = pd.read_csv(SO_CSV, usecols=["Country", "DevType", "ConvertedCompYearly"],
                     low_memory=False)
    df = df[df["ConvertedCompYearly"].notna() & (df["ConvertedCompYearly"] > 0)]
    df = df[df["Country"].isin(PAISES)].copy()
    df["DevType"] = df["DevType"].fillna("").str.lower()
    df = df[["Country", "DevType", "ConvertedCompYearly"]].reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"✓ {OUT.relative_to(ROOT)}  |  {len(df):,} filas  |  "
          f"{OUT.stat().st_size / 1024:.1f} KB  |  {df['Country'].nunique()} países")


if __name__ == "__main__":
    main()
