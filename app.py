# app.py — ultra-simple dealer app (no filters, clear section buttons, clustering & classification kept simple)
import os, re, json
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# ---------------- CONFIG (no user tweaking) ----------------
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.2
SYSTEM_PROMPT = (
    "You are DealBot, an expert assistant for a car dealership. "
    "Use only the provided data context. Keep answers very concise (max 5 short lines)."
)
DATA_PATH = "exports/final_model_dataset.csv"  # <-- your dataset name (unchanged)
TOP_N = 20  # how many rows to show by default
GOOD_DEAL_THRESHOLD = 0.80  # deal_score ≥ threshold => 'Good deal'
# ----------------------------------------------------------

# OpenAI client (if missing, chat will show an error)
try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None

st.set_page_config(page_title="Dealer Deals", page_icon="🚗", layout="wide")

# --- light styling: big “section” buttons & clean tables ---
st.markdown("""
<style>
/* App-wide polish */
section.main > div { padding-top: 0.5rem; }
/* Big top buttons */
.top-nav button[kind="secondary"] {
  border-radius: 9999px !important;
  padding: 0.6rem 1.1rem !important;
  font-weight: 600 !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
/* Make dataframes compact & neat */
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
    # Try to map common names → your dataset’s columns if present
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
    )

def compute_deal_score(df, c):
    out = df.copy()
    price, year, mileage, km = c["price"], c["year"], c["mileage"], c["km"]

    # Fallback if we only have price
    if not price:
        out["deal_score"] = (-out.iloc[:,0].rank(pct=True))  # arbitrary fallback
        return out

    # Build simple expected-price model using whatever is available
    features, cats = [], []
    for f in [year, mileage, km]:
        if f in out.columns: features.append(f)
    for f in [c["make"], c["model"], c["fuel"], c["trans"], c["body"]]:
        if f in out.columns: cats.append(f)

    if not features and not cats:
        out["deal_score"] = (-out[price]).rank(pct=True)
        return out

    pre = ColumnTransformer([
        ("num", StandardScaler(), [f for f in features if f in out.columns]),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02), [f for f in cats if f in out.columns]),
    ], remainder="drop")

    pipe = Pipeline([("prep", pre), ("reg", LinearRegression())])

    try:
        needed = [price] + [x for x in features+cats if x]
        train = out.dropna(subset=[x for x in needed if x in out.columns])
        X, y = train[[x for x in features+cats if x]], train[price]
        pipe.fit(X, y)
        out["expected_price"] = pipe.predict(out[[x for x in features+cats if x]].fillna(method="ffill").fillna(0))
        out["deal_delta"] = out["expected_price"] - out[price]
        out["deal_score"] = out["deal_delta"].rank(ascending=False, pct=True)
    except Exception:
        out["deal_score"] = (-out[price]).rank(pct=True)

    return out

def summarize_table(df, c, k=TOP_N):
    show_cols = [x for x in [c["year"], c["make"], c["model"], c["price"], c["mileage"], c["km"], "deal_score"] if x in df.columns or x == "deal_score"]
    tbl = df.sort_values(["deal_score", c["price"]] if c["price"] in df.columns else ["deal_score"],
                         ascending=[False, True]).head(k)
    return tbl[ [col for col in show_cols if col in tbl.columns] ]

def relevant_rows(df, query, c, limit=25):
    if df.empty or not query or not query.strip():
        return df.head(limit)
    q = query.lower()
    text_cols = [x for x in [c["make"], c["model"], c["fuel"], c["trans"], c["body"]] if x in df.columns]
    text_cols += [x for x in df.columns if df[x].dtype == object]
    text_cols = list(dict.fromkeys(text_cols))[:8]  # cap
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
    keep = [x for x in [c["year"], c["make"], c["model"], c["price"], c["mileage"], c["km"], "deal_score"] if x in df.columns or x=="deal_score"]
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
def simple_clustering(df, c, k=5):
    num_cols = [x for x in [c["price"], c["year"], c["mileage"], c["km"]] if x in df.columns]
    if not num_cols: 
        return None, None, None
    base = df.dropna(subset=num_cols).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(base[num_cols])
    km = KMeans(n_clusters=min(k, max(2, len(base)//50)), n_init="auto", random_state=42)
    base["cluster"] = km.fit_predict(X)
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=num_cols)
    return base, centers, num_cols

def name_clusters(centers, cols):
    # Heuristic names based on relative values
    names = {}
    if centers is None: return names
    for i, row in centers.iterrows():
        tags = []
        if "year" in cols:
            tags.append("newer" if row["year"] >= centers["year"].median() else "older")
        if any(x in cols for x in ["mileage", "km"]):
            miles_col = "mileage" if "mileage" in cols else ("km" if "km" in cols else None)
            if miles_col:
                tags.append("low-mileage" if row[miles_col] <= centers[miles_col].median() else "high-mileage")
        if "selling_price" in cols or "price" in cols:
            pcol = "selling_price" if "selling_price" in cols else "price"
            tags.append("budget" if row[pcol] <= centers[pcol].median() else "premium")
        names[i] = " / ".join(tags).title()
    return names

# ---------------- Classification ----------------
@st.cache_resource
def train_classifier(df, c):
    if "deal_score" not in df.columns: 
        return None
    y = (df["deal_score"] >= GOOD_DEAL_THRESHOLD).astype(int)  # 1 = good deal
    feat_num = [x for x in [c["year"], c["mileage"], c["km"]] if x in df.columns]
    feat_cat = [x for x in [c["make"], c["model"], c["fuel"], c["trans"], c["body"]] if x in df.columns]
    if not feat_num and not feat_cat:
        return None
    pre = ColumnTransformer([
        ("num", StandardScaler(), feat_num) if feat_num else ("num","drop",[]),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02), feat_cat) if feat_cat else ("cat","drop",[]),
    ])
    clf = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000))])
    safe_df = df.dropna(subset=feat_num + feat_cat).copy()
    if safe_df.empty:
        return None
    clf.fit(safe_df[feat_num + feat_cat], y.loc[safe_df.index])
    return clf, feat_num + feat_cat

# ==================== UI ====================
st.title("🚗 Dealer Deals")

df = load_data()
if df.empty: st.stop()
st.caption(f"Loaded **{DATA_PATH}** — {df.shape[0]:,} rows")

cols = pick_cols(df)
scored = compute_deal_score(df, cols)

# Navigation buttons (big & obvious)
if "section" not in st.session_state:
    st.session_state.section = "Best Deals"

st.write("")  # small spacing
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

st.write("")  # spacing

# ---------- Sections ----------
section = st.session_state.section

if section == "Best Deals":
    st.subheader("🏆 Best Deals (Top 20)")
    top_tbl = summarize_table(scored, cols, k=TOP_N)
    st.dataframe(top_tbl, use_container_width=True, hide_index=True)
    st.caption("“Deal Score” compares each car’s price to a simple expected price for its specs (higher is better).")

elif section == "Chat":
    st.subheader("💬 Ask about the inventory")
    # minimal UX: just the chat input — no advanced toggles
    if "chat" not in st.session_state: st.session_state.chat=[]
    for r,m in st.session_state.chat: st.chat_message(r).markdown(m)
    msg = st.chat_input("Ask anything about the cars (e.g., 'Best SUVs under 15k')")
    if msg:
        st.chat_message("user").markdown(msg)
        # Ground the answer in a small, relevant slice
        subset = relevant_rows(scored, msg, cols, limit=25)
        ctx = to_context(subset, cols, n=15)
        ans = ask_openai(msg, ctx)
        st.chat_message("assistant").markdown(ans)
        st.session_state.chat += [("user", msg), ("assistant", ans)]

elif section == "Clustering":
    st.subheader("🧩 Simple Clusters")
    clustered, centers, num_cols = simple_clustering(scored, cols, k=5)
    if clustered is None:
        st.info("Not enough numeric columns to cluster.")
    else:
        names = name_clusters(centers, num_cols)
        # Show a tiny summary: cluster sizes + center preview
        sizes = clustered["cluster"].value_counts().rename_axis("cluster").reset_index(name="count")
        sizes["name"] = sizes["cluster"].map(names).fillna("Cluster")
        st.markdown("**Cluster Sizes**")
        st.dataframe(sizes[["cluster","name","count"]], use_container_width=True, hide_index=True)

        st.markdown("**Cluster Centers (rough)**")
        st.dataframe(centers.round(1), use_container_width=True, hide_index=True)

        # Show a compact sample per cluster
        st.markdown("**Quick Samples per Cluster**")
        show_cols = [x for x in [cols["year"], cols["make"], cols["model"], cols["price"], cols["mileage"], cols["km"], "deal_score", "cluster"] if (x in clustered.columns or x=="deal_score")]
        sample = clustered.groupby("cluster").head(3)[show_cols].sort_values(["cluster","deal_score"], ascending=[True, False])
        st.dataframe(sample, use_container_width=True, hide_index=True)
        st.caption("Heuristic names summarize each cluster by price level, age, and mileage.")

elif section == "Classification":
    st.subheader("✅ Good-Deal Classifier")
    model = train_classifier(scored, cols)
    if model is None:
        st.info("Classifier could not be trained (not enough suitable columns).")
    else:
        clf, feat_cols = model
        # Score a simple slice (top 30 by deal score) so user isn’t overwhelmed
        slice_df = scored.sort_values("deal_score", ascending=False).head(30).copy()
        X = slice_df[[c for c in feat_cols if c in slice_df.columns]]
        # Predict
        try:
            proba = clf.predict_proba(X)[:,1]
            slice_df["good_deal_prob"] = np.round(proba, 3)
            slice_df["label"] = np.where(slice_df["good_deal_prob"] >= 0.5, "Good deal", "Fair/Poor")
        except Exception:
            slice_df["good_deal_prob"] = np.nan
            slice_df["label"] = np.where(slice_df["deal_score"] >= GOOD_DEAL_THRESHOLD, "Good deal", "Fair/Poor")

        show_cols = [x for x in [cols["year"], cols["make"], cols["model"], cols["price"], cols["mileage"], cols["km"], "deal_score", "good_deal_prob", "label"] if x in slice_df.columns or x in ["deal_score","good_deal_prob","label"]]
        st.markdown("**Top Cars with Predicted Quality**")
        st.dataframe(slice_df[show_cols], use_container_width=True, hide_index=True)
        st.caption("A lightweight logistic model marks likely good deals. No settings to tune — just the results.")

# --------------- Footer ---------------
st.write("")
st.caption("Simplicity first: no filters, minimal knobs. Sections are one-click.")
