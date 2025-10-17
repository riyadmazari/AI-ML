# app.py — simple dealer app (uses your soft_buy_score, user-chosen clustering axis)
import os, re, json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


# ---------- FIXED CONFIG ----------
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.2
SYSTEM_PROMPT = (
    "You are DealBot, an expert assistant for a car dealership. "
    "Use ONLY the provided data context. Be concise (max 5 short lines)."
)
DATA_PATH = "exports/final_model_dataset.csv"  # your file, unchanged
TOP_N = 20
GOOD_DEAL_THRESHOLD = 0.80  # soft_buy_score ≥ threshold => 'Good deal'
# ----------------------------------

try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None

st.set_page_config(page_title="Dealer Deals", page_icon="🚗", layout="wide")

st.markdown("""
<style>
section.main > div { padding-top: 0.5rem; }
.top-nav button[kind="secondary"] {
  border-radius: 9999px !important;
  padding: 0.6rem 1.1rem !important;
  font-weight: 600 !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
.dataframe th, .dataframe td { font-size: 0.90rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- Data ----------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at **{DATA_PATH}**")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def pick_cols(df):
    f = lambda cands: next((c for c in cands if c in df.columns), None)
    return dict(
        price=f(["selling_price","price"]),
        year=f(["year"]),
        make=f(["make","manufacturer"]),
        model=f(["model"]),
        mileage=f(["mileage","mileage_val"]),
        km=f(["km_driven","kilometers","km"]),
        power=f(["power_bhp","power"]),
        fuel=f(["fuel"]),
        trans=f(["transmission","gearbox"]),
        body=f(["body_type","type"]),
        soft=f(["soft_buy_score","softbuy_score","soft_score"])  # your score
    )

def summarize_table(df, c, k=TOP_N):
    score = c["soft"]
    sort_cols = [score] if score in df.columns else ([c["price"]] if c["price"] in df.columns else [df.columns[0]])
    asc = [False] if score in df.columns else [True]
    tbl = df.sort_values(sort_cols, ascending=asc).head(k)
    show_cols = [x for x in [c["year"], c["make"], c["model"], c["price"], c["mileage"], c["km"], score] if x in tbl.columns]
    return tbl[show_cols] if show_cols else tbl.head(k)

def relevant_rows(df, query, c, limit=25):
    if df.empty or not query or not query.strip():
        return df.head(limit)
    q = query.lower()
    text_cols = [x for x in [c["make"], c["model"], c["fuel"], c["trans"], c["body"]] if x in df.columns]
    text_cols += [x for x in df.columns if df[x].dtype == object]
    text_cols = list(dict.fromkeys(text_cols))[:8]
    temp = df.copy()
    for col in text_cols: temp[col] = temp[col].astype(str).str.lower()
    temp["_blob"] = temp[text_cols].agg(" ".join, axis=1)
    terms = re.findall(r"[a-z0-9]+", q)
    score = np.zeros(len(temp))
    for t in terms:
        score += temp["_blob"].str.contains(fr"\b{re.escape(t)}\b", na=False).astype(int).values
    temp["_rel"] = score
    return df.loc[temp.sort_values("_rel", ascending=False).head(limit).index]

def to_context(df, c, n=15):
    keep = [x for x in [c["year"], c["make"], c["model"], c["price"], c["mileage"], c["km"], c["soft"]] if x in df.columns]
    small = df[keep].head(n).reset_index(drop=True)
    small.insert(0, "id", small.index + 1)
    return small.to_dict(orient="records")

def ask_openai(msg, rows):
    if client is None:
        return "Chat unavailable (OpenAI client not initialized)."
    content = "Dataset sample:\n" + json.dumps(rows, ensure_ascii=False) + "\n\nUser question:\n" + msg
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":content}
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# ---------------- Clustering ----------------
def run_clustering(df, feature_cols, k=4):
    try:
        data = df[feature_cols].dropna().copy()  # keeps original df index
        if data.empty or len(data) < k:
            return None, None

        scaler = StandardScaler()
        X = scaler.fit_transform(data.values)

        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)

        clustered = data.copy()
        clustered["cluster"] = labels  # same index as df

        centers_std = km.cluster_centers_
        centers = pd.DataFrame(
            scaler.inverse_transform(centers_std),
            columns=feature_cols
        )
        centers.index.name = "cluster"
        centers.reset_index(inplace=True)
        return clustered, centers
    except Exception:
        return None, None


def name_clusters(centers):
    if centers is None: return {}
    names = {}
    for i, row in centers.iterrows():
        tags = []
        for col in centers.columns:
            tags.append(f"{col}:{'high' if row[col] >= centers[col].median() else 'low'}")
        names[i] = " / ".join(tags)
    return names

# ---------------- Classification ----------------
@st.cache_resource
def train_classifier(df, c):
    score = c["soft"]
    if score not in df.columns: return None
    y_raw = df[score].copy()
    # If it's already 0/1, use it; else threshold to 1=good deal
    if set(pd.Series(y_raw.dropna().unique()).astype(int)).issubset({0,1}) and y_raw.dropna().isin([0,1]).all():
        y = y_raw.astype(int)
    else:
        y = (y_raw >= GOOD_DEAL_THRESHOLD).astype(int)

    feat_num = [x for x in [c["year"], c["mileage"], c["km"], c["price"]] if x in df.columns]
    feat_cat = [x for x in [c["make"], c["model"], c["fuel"], c["trans"], c["body"]] if x in df.columns]
    if not feat_num and not feat_cat: return None

    pre = ColumnTransformer([
        ("num", StandardScaler(), feat_num) if feat_num else ("num","drop",[]),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02), feat_cat) if feat_cat else ("cat","drop",[]),
    ])
    safe = df.dropna(subset=feat_num + feat_cat).copy()
    if safe.empty: return None
    clf = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000))])
    clf.fit(safe[feat_num + feat_cat], y.loc[safe.index])
    return clf, feat_num + feat_cat

# ==================== UI ====================
st.title("🚗 Dealer Deals")

df = load_data()
if df.empty: st.stop()
st.caption(f"Loaded **{DATA_PATH}** — {df.shape[0]:,} rows")
cols = pick_cols(df)

# Nav
if "section" not in st.session_state:
    st.session_state.section = "Best Deals"

st.write("")
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🏆 Best Deals", type="secondary", use_container_width=True):
        st.session_state.section = "Best Deals"
with c2:
    if st.button("💬 Chat", type="secondary", use_container_width=True):
        st.session_state.section = "Chat"
with c3:
    if st.button("🧩 Clustering", type="secondary", use_container_width=True):
        st.session_state.section = "Clustering"
with c4:
    if st.button("✅ Classification", type="secondary", use_container_width=True):
        st.session_state.section = "Classification"
st.write("")

section = st.session_state.section

# ----------- Best Deals (uses soft_buy_score) -----------
if section == "Best Deals":
    st.subheader("🏆 Best Deals (Top 20)")
    top_tbl = summarize_table(df, cols, k=TOP_N)
    st.dataframe(top_tbl, use_container_width=True, hide_index=True)
    if cols["soft"] in df.columns:
        st.caption("Sorted by your **soft_buy_score** (higher = better).")
    else:
        st.caption("soft_buy_score not found—showing a basic top list.")

# ---------------- Chat ----------------
elif section == "Chat":
    st.subheader("💬 Ask about the inventory")
    if "chat" not in st.session_state: st.session_state.chat=[]
    for r,m in st.session_state.chat: st.chat_message(r).markdown(m)
    msg = st.chat_input("Ask anything (e.g., 'Best SUVs under 15k')")
    if msg:
        st.chat_message("user").markdown(msg)
        subset = relevant_rows(df, msg, cols, limit=25)
        ctx = to_context(subset, cols, n=15)
        ans = ask_openai(msg, ctx)
        st.chat_message("assistant").markdown(ans)
        st.session_state.chat += [("user", msg), ("assistant", ans)]

# ---------------- Clustering ----------------
elif section == "Clustering":
    st.subheader("🧩 Clustering")

    import altair as alt

    # figure out mileage column name from your mapping
    mileage_col = cols["mileage"] if cols["mileage"] in df.columns else cols["km"]

    options = {
        "Year": [c for c in [cols["year"]] if c],
        "Price": [c for c in [cols["price"]] if c],
        "Mileage/KM": [c for c in [mileage_col] if c],
        "Year + Price": [x for x in [cols["year"], cols["price"]] if x],
    }

    choice = st.radio("What to cluster by", list(options.keys()), horizontal=True)
    k = st.slider("Clusters (k)", min_value=3, max_value=10, value=4, step=1)
    view = st.radio("View", ["Table", "Plot"], horizontal=True)

    use_cols = options[choice]
    clustered, centers = run_clustering(df, use_cols, k=k)
    if clustered is None or centers is None or len(centers) == 0:
        st.info("Not enough data for that choice. Try a smaller k or a different axis.")
        st.stop()

    # readable names
    name_map = name_clusters(centers.set_index("cluster").drop(columns=[c for c in ["cluster"] if c in centers.columns], errors="ignore"))
    clustered["cluster_name"] = clustered["cluster"].map(name_map).fillna("Cluster")

    # single overview table: centers + size
    if view == "Table":
        sizes = clustered["cluster"].value_counts().rename_axis("cluster").reset_index(name="count")
        overview = centers.merge(sizes, on="cluster", how="left")
        overview["name"] = overview["cluster"].map(name_map).fillna("Cluster")

        # tidy order
        metric_cols = [c for c in [cols["year"], cols["price"], mileage_col] if c in overview.columns]
        overview = overview[["cluster", "name", "count"] + metric_cols].sort_values("cluster")

        st.markdown("**Cluster Overview**")
        st.dataframe(overview.round(1), use_container_width=True, hide_index=True)

    else:
        # ---- Cluster Graph (network view) ----
        st.markdown("**Cluster Graph**")

        import networkx as nx
        import plotly.graph_objects as go
        from scipy.spatial.distance import pdist, squareform

        G = nx.Graph()

        # pairwise distances between centers (exclude 'cluster' col)
        numeric_centers = centers.drop(columns=["cluster"], errors="ignore")
        if numeric_centers.shape[0] > 1:
            dists = squareform(pdist(numeric_centers.values))
            thr = float(np.median(dists) * 1.5)
            for i in range(len(dists)):
                for j in range(i + 1, len(dists)):
                    w = float(dists[i, j])
                    if w < thr:
                        G.add_edge(int(i), int(j), weight=w)

        # nodes
        for _, row in centers.iterrows():
            idx = int(row["cluster"])
            G.add_node(idx, label=f"Cluster {idx}", size=20)

        pos = nx.spring_layout(G, seed=42, k=0.5)

        # edges
        edge_x, edge_y = [], []
        for a, b in G.edges():
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            hoverinfo="none",
            line=dict(width=1, color="#888")
        )

        # nodes
        node_x, node_y, text = [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            text.append(f"Cluster {n}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=text,
            textposition="top center",
            marker=dict(
                showscale=False,
                color=list(G.nodes()),  # distinct colors per cluster id
                size=22,
                line=dict(width=2, color="DarkSlateGrey")
            ),
            hoverinfo="text"
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text="Cluster Relationships", font=dict(size=18)),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Each node is a cluster center; edges link clusters with similar centers.")






# ---------------- Classification ----------------
elif section == "Classification":
    st.subheader("✅ Good-Deal Classifier (based on soft_buy_score)")
    model = train_classifier(df, cols)
    if model is None:
        st.info("Classifier could not be trained (missing columns or empty rows).")
    else:
        clf, feat_cols = model
        slice_df = df.copy()
        if cols["soft"] in slice_df.columns:
            slice_df = slice_df.sort_values(cols["soft"], ascending=False)
        slice_df = slice_df.head(30).copy()

        X = slice_df[[c for c in feat_cols if c in slice_df.columns]]
        try:
            proba = clf.predict_proba(X)[:,1]
            slice_df["good_deal_prob"] = np.round(proba, 3)
            slice_df["label"] = np.where(slice_df["good_deal_prob"] >= 0.5, "Good deal", "Fair/Poor")
        except Exception:
            if cols["soft"] in slice_df.columns:
                slice_df["label"] = np.where(slice_df[cols["soft"]] >= GOOD_DEAL_THRESHOLD, "Good deal", "Fair/Poor")
                slice_df["good_deal_prob"] = np.nan
            else:
                slice_df["label"] = "N/A"
                slice_df["good_deal_prob"] = np.nan

        show_cols = [x for x in [cols["year"], cols["make"], cols["model"], cols["price"], mileage_col, cols["soft"], "good_deal_prob", "label"] if x in slice_df.columns or x in ["good_deal_prob","label"]]
        st.markdown("**Top Cars with Predicted Quality**")
        st.dataframe(slice_df[show_cols], use_container_width=True, hide_index=True)
        st.caption("Prediction target = (soft_buy_score ≥ threshold). Threshold is fixed in code.")

# --------------- Footer ---------------
st.write("")
st.caption("Simplicity first: single-choice clustering; soft_buy_score drives rankings & labels.")
