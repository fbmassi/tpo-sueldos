    #!/usr/bin/env python3
"""
predictor_cli.py
================

Predictor interactivo por consola: estima el salario de un profesional tech
basado en su perfil (edad, experiencia, rol, provincia, tecnologías, etc.)
usando un modelo Random Forest entrenado en el dataset de Sysarmy.

Uso:
    python data/predictor_cli.py

Ejemplo de sesión:
    > Edad: 28
    > Años de experiencia total: 5
    > Provincia: Buenos Aires
    > Seniority: semi-senior
    > ...
    Salario predicho: $3.2M (pesos reales de mayo 2026)

Las predicciones están en pesos constantes (deflactados a mayo 2026).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# Configuración
DATA_DIR = Path(__file__).resolve().parent
PROC_DIR = DATA_DIR / "processed"
DATASET = PROC_DIR / "dataset_final_mercado_laboral.parquet"
TARGET = "salario_real_ars"
RANDOM_STATE = 42

TOP_ROLES = 15
TOP_TECHS = 20
MIN_PROVINCIA = 100

COLS_NUM = ["edad", "anos_experiencia_total", "anos_empresa_actual"]
COLS_CAT = ["provincia", "genero", "seniority", "modalidad",
            "tamano_empresa", "rol", "cobra_en_dolares"]


def hacer_preprocesador(cols_tech: list[str]) -> ColumnTransformer:
    """Preprocesador para árboles (sin escala)."""
    return ColumnTransformer([
        ("num", "passthrough", COLS_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COLS_CAT),
        ("tech", "passthrough", cols_tech),
    ])


def preparar_datos() -> tuple[pd.DataFrame, pd.Series, list[str],
                               dict, RandomForestRegressor]:
    """Carga dataset, prepara features, entrena Random Forest. Devuelve todo."""
    df = pd.read_parquet(DATASET)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    y = df[TARGET]

    X = pd.DataFrame(index=df.index)

    # Numéricas
    for c in COLS_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")

    # Provincia: agrupar chicas
    vc = df["provincia"].value_counts()
    chicas = vc[vc < MIN_PROVINCIA].index
    X["provincia"] = df["provincia"].where(~df["provincia"].isin(chicas), "Otra")

    # Rol: top 15 + Otro
    top_roles = df["rol"].value_counts().head(TOP_ROLES).index
    X["rol"] = df["rol"].where(df["rol"].isin(top_roles), "Otro")

    # Categóricas
    X["genero"] = df["genero"].fillna("no especifica")
    X["seniority"] = df["seniority"]
    X["modalidad"] = df["modalidad"]
    X["tamano_empresa"] = df["tamano_empresa"].fillna("No especifica")
    X["cobra_en_dolares"] = df["cobra_en_dolares"].astype(str)

    # Tecnologías: top 20 multi-hot
    listas = (df["tecnologias"].fillna("")
              .replace("No especifica", "")
              .str.split(",")
              .apply(lambda ts: {t.strip() for t in ts if t.strip()}))
    conteo = pd.Series([t for s in listas for t in s]).value_counts()
    conteo = conteo.drop("ninguno de los anteriores", errors="ignore")
    techs = conteo.head(TOP_TECHS).index.tolist()

    cols_tech = []
    for t in techs:
        col = f"usa_{t.replace(' ', '_').replace('.', '').replace('#', 'sharp')}"
        X[col] = listas.apply(lambda s, tt=t: int(tt in s))
        cols_tech.append(col)

    # Entrenar Random Forest
    pre = hacer_preprocesador(cols_tech)
    X_pre = pre.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1,
                                   random_state=RANDOM_STATE)
    model.fit(X_pre, y)

    # Devolver todo necesario para predicciones
    opciones = {
        "provincia": sorted(X["provincia"].unique()),
        "genero": sorted(X["genero"].unique()),
        "seniority": sorted(X["seniority"].unique()),
        "modalidad": sorted(X["modalidad"].unique()),
        "tamano_empresa": sorted(X["tamano_empresa"].unique()),
        "rol": sorted(X["rol"].unique()),
        "cobra_en_dolares": ["False", "True"],
        "tecnologias": techs,
    }

    return X, y, cols_tech, opciones, (pre, model)


def pedir_entrada(prompt: str, opciones: list[str] | None = None,
                 tipo: type = str) -> str | float | int | bool:
    """Pide entrada con validación. Si opciones, muestra menú."""
    while True:
        if opciones:
            print(f"\n{prompt}")
            for i, opt in enumerate(opciones, 1):
                print(f"  {i}. {opt}")
            try:
                idx = int(input("Elegir (número): ")) - 1
                if 0 <= idx < len(opciones):
                    return opciones[idx]
                print("❌ Opción inválida.")
            except ValueError:
                print("❌ Ingrese un número válido.")
        else:
            try:
                val = input(f"{prompt} ")
                if tipo == float or tipo == int:
                    return tipo(val)
                return val
            except ValueError:
                print(f"❌ Ingrese un valor válido ({tipo.__name__}).")


def hacer_prediccion(entrada: dict, X: pd.DataFrame, cols_tech: list[str],
                     pre_model: tuple) -> float:
    """Construye un row con la entrada del usuario y predice."""
    pre, model = pre_model

    # Construir row igual a X (con mismas columnas)
    row = pd.DataFrame([{
        "edad": entrada["edad"],
        "anos_experiencia_total": entrada["anos_experiencia_total"],
        "anos_empresa_actual": entrada["anos_empresa_actual"],
        "provincia": entrada["provincia"],
        "genero": entrada["genero"],
        "seniority": entrada["seniority"],
        "modalidad": entrada["modalidad"],
        "tamano_empresa": entrada["tamano_empresa"],
        "rol": entrada["rol"],
        "cobra_en_dolares": entrada["cobra_en_dolares"],
    }])

    # Agregar tecnologías (multi-hot)
    for tech in cols_tech:
        tech_name = tech.replace("usa_", "").replace("_", " ").replace("sharp", "#")
        row[tech] = int(tech_name in entrada.get("tecnologias", []))

    # Preprocesar y predecir
    row_pre = pre.transform(row)
    pred = model.predict(row_pre)[0]
    return pred


def comparar_similares(entrada: dict, X: pd.DataFrame, y: pd.Series) -> None:
    """Busca profesionales similares (mismo rol, seniority, provincia)."""
    mask = ((X["rol"] == entrada["rol"]) &
            (X["seniority"] == entrada["seniority"]) &
            (X["provincia"] == entrada["provincia"]))
    similares = y[mask]
    if len(similares) > 0:
        print(f"\n  👥 Profesionales similares (mismo rol, seniority, provincia): {len(similares):,}")
        print(f"     Mediana: ${similares.median()/1e6:.2f}M")
        print(f"     Rango: ${similares.min()/1e6:.2f}M — ${similares.max()/1e6:.2f}M")
        print(f"     IQR: ${similares.quantile(0.25)/1e6:.2f}M — ${similares.quantile(0.75)/1e6:.2f}M")
    else:
        print(f"\n  👥 No hay profesionales similares en la base de datos.")


def exportar_resultado(entrada: dict, pred: float) -> None:
    """Guarda el resultado en un CSV."""
    import csv
    from datetime import datetime
    path = PROC_DIR / "predicciones_historial.csv"
    existe = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fecha", "edad", "experiencia", "rol",
                                           "seniority", "provincia", "salario_predicho"])
        if not existe:
            w.writeheader()
        w.writerow({
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "edad": entrada["edad"],
            "experiencia": entrada["anos_experiencia_total"],
            "rol": entrada["rol"],
            "seniority": entrada["seniority"],
            "provincia": entrada["provincia"],
            "salario_predicho": f"${pred/1e6:.2f}M",
        })
    print(f"\n  📊 Resultado guardado en {path}")


def main() -> None:
    print("=" * 70)
    print("PREDICTOR DE SALARIO — Profesionales Tech Argentina")
    print("=" * 70)
    print("Base: 31.088 registros de Sysarmy (2022-2025)")
    print("Salarios en: pesos reales de mayo 2026 (ajustados por inflación)\n")

    print("Cargando datos y entrenando modelo...")
    X, y, cols_tech, opciones, pre_model = preparar_datos()
    print(f"✓ Listo. Modelo entrenado en {len(X):,} registros.\n")

    while True:
        print("\n" + "─" * 70)
        print("NUEVA PREDICCIÓN")
        print("─" * 70)

        entrada = {}

        # Entrada numérica
        entrada["edad"] = pedir_entrada("Edad (años):", tipo=int)
        entrada["anos_experiencia_total"] = pedir_entrada(
            "Años de experiencia total:", tipo=int)
        entrada["anos_empresa_actual"] = pedir_entrada(
            "Años en la empresa actual:", tipo=int)

        # Categóricas con menú
        entrada["provincia"] = pedir_entrada(
            "Provincia de trabajo:", opciones["provincia"])
        entrada["genero"] = pedir_entrada(
            "Género:", opciones["genero"])
        entrada["seniority"] = pedir_entrada(
            "Nivel (seniority):", opciones["seniority"])
        entrada["modalidad"] = pedir_entrada(
            "Modalidad de trabajo:", opciones["modalidad"])
        entrada["tamano_empresa"] = pedir_entrada(
            "Tamaño de empresa:", opciones["tamano_empresa"])
        entrada["rol"] = pedir_entrada(
            "Rol / puesto:", opciones["rol"])
        # cobra_en_dolares: pregunta simple sí/no
        print("\n¿Cobra en dólares?")
        print("  1. No")
        print("  2. Sí")
        while True:
            try:
                idx = int(input("Elegir (número): ")) - 1
                if idx == 0:
                    entrada["cobra_en_dolares"] = "False"
                    break
                elif idx == 1:
                    entrada["cobra_en_dolares"] = "True"
                    break
                else:
                    print("❌ Opción inválida.")
            except ValueError:
                print("❌ Ingrese 1 o 2.")

        # Tecnologías (multi-select)
        print("\nTecnologías (escriba los números separados por comas, o 0 para omitir):")
        for i, tech in enumerate(opciones["tecnologias"], 1):
            print(f"  {i:2d}. {tech}")
        techs_input = input("Seleccionar (ej: 1,3,5): ").strip()
        entrada["tecnologias"] = []
        if techs_input and techs_input != "0":
            try:
                indices = [int(x.strip()) - 1 for x in techs_input.split(",")]
                entrada["tecnologias"] = [opciones["tecnologias"][i]
                                          for i in indices if 0 <= i < len(opciones["tecnologias"])]
            except (ValueError, IndexError):
                print("⚠ Entrada inválida. Sin tecnologías seleccionadas.")

        # Predicción
        pred = hacer_prediccion(entrada, X, cols_tech, pre_model)

        print("\n" + "=" * 70)
        print(f"💰 SALARIO PREDICHO: ${pred/1e6:.2f}M (pesos reales de mayo 2026)")
        print("=" * 70)
        print("\n📈 INTERVALO DE CONFIANZA:")
        print(f"  • P25 (cuartil bajo): ${y.quantile(0.25)/1e6:.2f}M")
        print(f"  • P50 (mediana): ${y.quantile(0.50)/1e6:.2f}M")
        print(f"  • P75 (cuartil alto): ${y.quantile(0.75)/1e6:.2f}M")
        print(f"  → Tu predicción: ${pred/1e6:.2f}M", end="")
        if pred < y.quantile(0.25):
            print(" (BAJO P25)")
        elif pred > y.quantile(0.75):
            print(" (ALTO P75)")
        else:
            print(" (RANGO MEDIO)")

        # Comparar con similares
        comparar_similares(entrada, X, y)

        # Menú de opciones
        print("\n" + "─" * 70)
        print("OPCIONES:")
        print("  1. Nueva predicción")
        print("  2. Exportar este resultado a CSV")
        print("  3. Salir")
        opt = input("Elegir (1/2/3): ").strip()
        if opt == "2":
            exportar_resultado(entrada, pred)
            otra = input("\n¿Otra predicción? (s/n): ").strip().lower()
            if otra != "s":
                print("\n¡Hasta luego!")
                break
        elif opt == "3":
            print("\n¡Hasta luego!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n¡Hasta luego!")
    except FileNotFoundError:
        print(f"❌ Dataset no encontrado en {DATASET}")
        print("Asegúrate de haber corrido limpiar_y_unificar_datos.py primero.")
    except Exception as exc:
        print(f"❌ Error: {exc}")
