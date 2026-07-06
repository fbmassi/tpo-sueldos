"""
app_en/pages/01_eda.py
======================

Interactive Exploratory Data Analysis (EDA) of the tech-salary dataset, in
English and real US dollars. Mirrors app/pages/01_eda.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "app_en"))

from i18n import value_en  # noqa: E402

DATASET = ROOT / "data" / "processed" / "dataset_final_mercado_laboral.parquet"
CONTEXTO = ROOT / "data" / "processed" / "contexto_macroeconomico.parquet"

FACTORS = {"rol": "Role", "provincia": "Province", "seniority": "Seniority",
           "modalidad": "Work mode", "tamano_empresa": "Company size",
           "genero": "Gender", "cobra_en_dolares": "Paid in USD"}


@st.cache_data(show_spinner="Loading data…")
def load(ver: float) -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    ctx = pd.read_parquet(CONTEXTO)
    return df.merge(ctx, on="fecha_edicion", how="left")


try:
    df = load(DATASET.stat().st_mtime)
except FileNotFoundError:
    st.error("Dataset not found. Run "
             "`python notebooks/limpiar_y_unificar_datos.py` first.")
    st.stop()

st.title("📊 Market analysis")
st.caption("Salaries in real May-2026 US dollars. Source: Sysarmy 2022–2025.")

# ---- KPIs ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", f"{len(df):,}")
k2.metric("Survey editions", df["fecha_edicion"].nunique())
k3.metric("Median salary", f"USD {df['salario_real_usd'].median():,.0f}")
k4.metric("Paid in USD", f"{df['cobra_en_dolares'].astype(str).eq('True').mean()*100:.0f}%")

st.divider()

tab1, tab2, tab3 = st.tabs(["Distribution", "Salary by factor", "Technologies"])

# ---- TAB 1: overall distribution ----
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Salary distribution")
        s = df["salario_real_usd"]
        s = s[s < s.quantile(0.99)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s, bins=50, color="steelblue", edgecolor="white")
        ax.axvline(s.median(), color="red", ls="--",
                   label=f"Median USD {s.median():,.0f}")
        ax.set_xlabel("Real salary (USD / month)"); ax.set_ylabel("Frequency")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)
    with c2:
        st.subheader("Seniority mix")
        sen = df["seniority"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(sen.values, labels=sen.index, autopct="%1.0f%%",
               colors=["#4c72b0", "#dd8452", "#55a868"], startangle=90)
        st.pyplot(fig); plt.close(fig)

# ---- TAB 2: salary by factor (interactive) ----
with tab2:
    factor = st.selectbox("Salary by…", list(FACTORS),
                          format_func=lambda f: FACTORS[f])
    min_n = st.slider("Minimum answers per category", 10, 300, 50, step=10,
                      help="Hides categories with too few cases (not representative).")
    sub = df[df["salario_real_usd"] < df["salario_real_usd"].quantile(0.99)].copy()
    sub[factor] = sub[factor].astype(str)

    counts = sub[factor].value_counts()
    valid = counts[counts >= min_n].index
    order = (sub[sub[factor].isin(valid)].groupby(factor)["salario_real_usd"]
             .median().sort_values(ascending=False).head(12).index.tolist())

    if not order:
        st.info("No category reaches that minimum. Lower the threshold.")
    else:
        data = [sub[sub[factor] == cat]["salario_real_usd"].values for cat in order]
        labels = [value_en(c) for c in order]
        fig, ax = plt.subplots(figsize=(10, max(3, len(order) * 0.45)))
        ax.boxplot(data, vert=False, tick_labels=labels, showfliers=False)
        ax.set_xlabel("Real salary (USD / month)")
        ax.invert_yaxis(); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)
        st.caption(f"Only categories with ≥ {min_n} answers (1–2 cases are not "
                   "representative).")

        table = (sub[sub[factor].isin(valid)].groupby(factor)["salario_real_usd"]
                 .agg(Cases="size", Median="median")
                 .sort_values("Median", ascending=False))
        table["Median"] = table["Median"].round(0)
        table.index = [value_en(i) for i in table.index]
        st.dataframe(table.head(12), width="stretch")

# ---- TAB 3: technologies ----
with tab3:
    techs = (df.assign(t=df["tecnologias"].fillna("").str.split(","))
             .explode("t"))
    techs["t"] = techs["t"].str.strip()
    techs = techs[(techs["t"] != "") & (techs["t"] != "No especifica") &
                  (techs["t"] != "ninguno de los anteriores")]
    freq = techs["t"].value_counts()
    eligible = freq[freq >= 200].index
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Most mentioned")
        top = freq.head(12).sort_values()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top.index, top.values, color="steelblue")
        ax.set_xlabel("Mentions"); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)
    with c2:
        st.subheader("Best paid (≥200 mentions)")
        pay = (techs[techs["t"].isin(eligible)]
               .groupby("t")["salario_real_usd"].median()
               .sort_values().tail(12))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(pay.index, pay.values, color="seagreen")
        ax.set_xlabel("Median salary (USD / month)"); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    st.subheader("Which stack does each seniority use?")
    st.caption("% of professionals at each level using each technology (top 12).")
    sen_order = [s for s in ["junior", "semi-senior", "senior"]
                 if s in df["seniority"].unique()]
    top_st = freq.head(12).index
    tot = df["seniority"].value_counts()
    use = (techs[techs["t"].isin(top_st)]
           .groupby(["seniority", "t"]).size().unstack(fill_value=0))
    use = (use.div(tot, axis=0) * 100).reindex(sen_order)[top_st]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(use.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(top_st))); ax.set_xticklabels(top_st, rotation=40, ha="right")
    ax.set_yticks(range(len(sen_order))); ax.set_yticklabels(sen_order)
    for i in range(use.shape[0]):
        for j in range(use.shape[1]):
            v = use.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > use.values.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, label="% usage", fraction=0.025)
    st.pyplot(fig); plt.close(fig)
    st.caption("Comparing junior vs senior shows which technologies concentrate at "
               "each level (e.g. seniors shifting towards backend/infra).")
