"""
app_en/pages/03_clusters.py
===========================

Unsupervised segmentation (K-Means) of tech professionals, in English / USD.
Groups profiles WITHOUT looking at salary, then shows what each segment earns.
Mirrors app/pages/03_clusters.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "app_en"))

from i18n import MODALITY_EN, size_en, value_en  # noqa: E402

DATASET = ROOT / "data" / "processed" / "dataset_final_mercado_laboral.parquet"

NUM = ["edad", "anos_experiencia_total", "anos_empresa_actual"]
CAT = ["modalidad", "cobra_en_dolares", "tamano_empresa"]
PALETTE = ["#2E86C1", "#E6A100", "#8C8C8C", "#0A0A0A", "#C0392B", "#27AE60",
           "#8E44AD", "#16A085"]


@st.cache_data(show_spinner="Loading data…")
def load(ver: float) -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    df["cobra_en_dolares"] = df["cobra_en_dolares"].astype(str)
    return df


@st.cache_resource(show_spinner="Computing segments…")
def train(k: int, ver: float):
    df = load(ver)
    pre = ColumnTransformer([("n", StandardScaler(), NUM),
                             ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    Xp = pre.fit_transform(df[NUM + CAT])
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xp)
    coords = PCA(n_components=2, random_state=42).fit_transform(
        Xp.toarray() if hasattr(Xp, "toarray") else Xp)
    return pre, km, coords


def name_segments(df: pd.DataFrame) -> dict:
    """Descriptive name per cluster: LEVEL (by experience) + distinctive trait."""
    p = df.groupby("cluster").agg(
        exp=("anos_experiencia_total", "median"),
        usd=("cobra_en_dolares", lambda s: (s == "True").mean()),
        ten=("anos_empresa_actual", "median"),
        hyb=("modalidad", lambda s: (s == "híbrido").mean()),
        ons=("modalidad", lambda s: (s == "100% presencial").mean()))
    names, used = {}, set()
    for c in p.index:
        e = p.loc[c, "exp"]
        level = ("🌱 Junior" if e < 4 else "🧑‍💻 Mid-level" if e < 7
                 else "👨‍💻 Senior" if e <= 15 else "🎖️ Senior 15+")
        if p.loc[c, "usd"] > 0.6:
            trait = "USD-paid"
        elif p.loc[c, "ten"] >= 10:
            trait = "long-tenure corporate"
        elif p.loc[c, "hyb"] > 0.5:
            trait = "hybrid"
        elif p.loc[c, "ons"] > 0.4:
            trait = "on-site"
        else:
            trait = "remote"
        n = f"{level} {trait}"
        if n in used:
            n = f"{n} #{c}"
        used.add(n)
        names[c] = n
    return names


try:
    df = load(DATASET.stat().st_mtime)
except FileNotFoundError:
    st.error("Dataset not found. Run "
             "`python notebooks/limpiar_y_unificar_datos.py` first.")
    st.stop()

st.title("🧩 Market profiles (clustering)")
st.markdown(
    "We group similar profiles **without looking at the salary** (K-Means), and "
    "only then check what each group earns: that is how we discover the "
    "**archetypes** of the Argentine tech market."
)

k = st.slider("Number of segments (k)", 2, 8, 4,
              help="How many archetypes to split into. Names are recomputed when k changes.")
pre, km, coords = train(k, DATASET.stat().st_mtime)
df = df.copy()
df["cluster"] = km.labels_
names = name_segments(df)

# ---- PCA scatter ----
st.subheader("Segment map")
fig, ax = plt.subplots(figsize=(12, 6.2))
samp = np.random.RandomState(0).choice(len(df), min(8000, len(df)), replace=False)
# matplotlib's font has no emoji glyphs: legend uses emoji-free labels
plain = {c: re.sub(r"[^\w\s+#/&.-]", "", n).strip() for c, n in names.items()}
for c in range(k):
    m = df["cluster"].values[samp] == c
    ax.scatter(coords[samp][m, 0], coords[samp][m, 1], c=PALETTE[c % len(PALETTE)],
               s=18, alpha=0.6, edgecolors="none", label=plain[c])
x1, x2 = np.percentile(coords[:, 0], [0.5, 99])
y1, y2 = np.percentile(coords[:, 1], [1, 98.5])
ax.set_xlim(x1 - 0.4, x2 + 0.4); ax.set_ylim(y1 - 0.4, y2 + 0.6)
ax.set_xlabel("Component 1  →  more career years (junior to senior)", fontsize=13)
ax.set_ylabel("Component 2  →  more tenure / rotation", fontsize=13)
ax.tick_params(labelsize=11)
ax.legend(loc="upper right", framealpha=1, fontsize=12, markerscale=2, title="Segment")
ax.grid(alpha=0.25)
st.pyplot(fig); plt.close(fig)
st.caption("2D projection (PCA) showing 62% of the information; in the full "
           "feature space the groups are better separated.")

# ---- Salary per segment ----
st.subheader("Salary per segment")
salc = df.groupby("cluster")["salario_real_usd"].median()
salc.index = [names[c] for c in salc.index]
st.bar_chart(salc.sort_values(), y_label="Median salary (USD/month)", horizontal=True)

# ---- Characterization ----
st.subheader("Who is who in each segment?")
rows = []
for c in range(k):
    g = df[df["cluster"] == c]
    rows.append({
        "Segment": names[c],
        "Market share": f"{len(g)/len(df)*100:.0f}%",
        "Age": f"{g['edad'].median():.0f}",
        "Experience": f"{g['anos_experiencia_total'].median():.0f} yrs",
        "Tenure": f"{g['anos_empresa_actual'].median():.0f} yrs",
        "Paid in USD": f"{g['cobra_en_dolares'].eq('True').mean()*100:.0f}%",
        "Remote": f"{g['modalidad'].eq('100% remoto').mean()*100:.0f}%",
        "Hybrid": f"{g['modalidad'].eq('híbrido').mean()*100:.0f}%",
        "Median salary": f"USD {g['salario_real_usd'].median():,.0f}",
    })
table = pd.DataFrame(rows).set_index("Segment")
st.dataframe(table.sort_values("Median salary", ascending=False), width="stretch")

# ---- Which segment am I in? ----
st.divider()
st.subheader("🔎 Which segment do you belong to?")
with st.form("profile_cluster"):
    a, b, c = st.columns(3)
    age = a.number_input("Age", 18, 75, 30)
    exp = b.number_input("Years of experience", 0, 50, 5)
    ten = c.number_input("Years at current company", 0, 50, 2)
    d, e, f = st.columns(3)
    modality = d.selectbox("Work mode", sorted(df["modalidad"].unique()),
                           format_func=lambda v: MODALITY_EN.get(v, v))
    paid = e.radio("Paid in USD?", ["No", "Yes"], horizontal=True) == "Yes"
    size = f.selectbox("Company size", sorted(df["tamano_empresa"].unique()),
                       format_func=size_en)
    ok = st.form_submit_button("Find my segment", type="primary")

if ok:
    row = pd.DataFrame([{
        "edad": age, "anos_experiencia_total": exp, "anos_empresa_actual": ten,
        "modalidad": modality, "cobra_en_dolares": str(paid), "tamano_empresa": size,
    }])
    cl = int(km.predict(pre.transform(row))[0])
    g = df[df["cluster"] == cl]
    st.success(f"You belong to the **{names[cl]}** segment "
               f"({len(g)/len(df)*100:.0f}% of the market).")
    m1, m2, m3 = st.columns(3)
    m1.metric("Segment median salary", f"USD {g['salario_real_usd'].median():,.0f}")
    m2.metric("Typical work mode", value_en(g["modalidad"].mode().iloc[0]))
    m3.metric("Typical experience", f"{g['anos_experiencia_total'].median():.0f} yrs")

st.caption("Clustering = **unsupervised** learning: it complements the predictive "
           "model (Random Forest) by describing the market structure.")
