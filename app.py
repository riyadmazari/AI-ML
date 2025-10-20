# app.py — dealer app with role gate + split sections (no sidebar)
import os, re, json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import joblib

# ---------- CONFIG ----------
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.2
SYSTEM_PROMPT = (
    "You are DealBot, a car-dealer assistant.\n"
    "You will receive JSON with {schema, rows, question}.\n"
    "- Use ONLY the provided 'rows'. If the answer isn't implied by them, reply exactly: 'Not in this dataset.'\n"
    "- When referring to price, ALWAYS use 'price_eur' (numeric) or 'price_eur_str' (already formatted). "
    "NEVER use 'selling_price' or show '₹'.\n"
    "- For the car label, use the 'car' column EXACTLY as given; do not prepend/duplicate brand/model.\n"
    "- If the question mentions a body style (SUV, sedan, hatchback, pickup, wagon, convertible, MPV), "
    "filter by 'body_norm' or mapped synonyms in schema.synonyms when present.\n"
    "- For 'best'/'top' ranking: sort by soft_buy_score ↓, then price_eur ↑, then year ↓, then km_driven ↑ when present.\n"
    "- If the user asks for N items, return up to N.\n"
    "- Be concise (≤ 5 short lines) and cite car name/year/price from the rows."
)
DATA_PATH = "exports/final_enriched_full.csv"  # <-- merged file with names
# --- Pretrained model paths ---
HEDONIC_MODEL_PATH = "models/hedonic_model.joblib"
CLASSIFIER_MODEL_PATH = "models/classification_model.joblib"  # or .pkl if that's what you saved
TOP_N = 20
GOOD_DEAL_THRESHOLD = 0.80
OWNER_PASSWORD = "123"
FIXED_K = 5    # fixed number of clusters for customers
# ----------------------------

try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None

st.set_page_config(
    page_title="Dealer Deals", layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Styling ----------
st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none; }
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

# ---------------- Data + ML helpers (unchanged from previous version) ----------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at **{DATA_PATH}**")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    # keep original training names — only strip whitespace
    df.columns = [c.strip() for c in df.columns]   # ← remove .replace(" ", "_")
    return df

@st.cache_resource
def load_hedonic_model():
    try:
        return joblib.load(HEDONIC_MODEL_PATH)  # either (pipeline, meta) or just pipeline
    except Exception as e:
        st.warning(f"Could not load hedonic model at {HEDONIC_MODEL_PATH}: {e}")
        return None

@st.cache_resource
def load_classifier_model():
    try:
        return joblib.load(CLASSIFIER_MODEL_PATH)
    except Exception as e:
        st.warning(f"Could not load classifier model at {CLASSIFIER_MODEL_PATH}: {e}")
        return None


def pick_cols(df):
    f = lambda cands: next((c for c in cands if c in df.columns), None)
    return dict(
        price=f(["price_eur", "selling_price", "price"]),
        year=f(["year"]),
        make=f(["make","manufacturer"]), model=f(["model"]),
        mileage=f(["mileage_val", "mileage"]),     # ← swapped order here
        km=f(["km_driven","kilometers","km"]),
        power=f(["power_bhp","power"]), fuel=f(["fuel"]),
        trans=f(["transmission","gearbox"]), body=f(["body_type","type"]),
        soft=f(["soft_buy_score","softbuy_score","soft_score"])
    )


def ensure_car_name(df: pd.DataFrame) -> pd.DataFrame:
    """Create/repair a single 'car' column for display using name/brand/model fallbacks."""
    out = df.copy()

    def _clean(s: pd.Series) -> pd.Series:
        # normalize text and collapse placeholders to empty
        s = s.astype(str).str.strip()
        placeholders = {"nan", "none", "null", "na", "n/a", "-", "—"}
        return s.apply(lambda x: "" if x.lower() in placeholders else x)

    for c in ["car", "name", "brand", "model"]:
        if c in out.columns:
            out[c] = _clean(out[c])

    # start from existing car (cleaned), else empty series
    base = out["car"] if "car" in out.columns else pd.Series("", index=out.index)

    # prefer 'name' where base is empty/placeholder
    if "name" in out.columns:
        empty_mask = base.eq("")
        base = base.where(~empty_mask, out["name"])

    # then brand + model
    if {"brand", "model"}.issubset(out.columns):
        bm = (out["brand"].fillna("") + " " + out["model"].fillna("")).str.strip()
        empty_mask = base.eq("")
        base = base.where(~empty_mask, bm)

    # or brand alone
    if "brand" in out.columns:
        empty_mask = base.eq("")
        base = base.where(~empty_mask, out["brand"])

    out["car"] = base.replace("", "—")
    return out

# ---------- Chat helpers: parse & answer from the dataframe (v2) ----------
_NUMBER_RE = re.compile(r"\d[\d,\.]*")

def _to_num(x):
    try:
        if isinstance(x, str):
            x = x.replace(",", "").replace(" ", "")
        return float(x)
    except Exception:
        return None

def _extract_first_number(txt):
    s = txt.lower()
    # Normalize common units into plain numbers
    s = re.sub(r"(\d+)\s*(lakh|lakhs)\b",   lambda m: str(int(m.group(1)) * 100000), s)
    s = re.sub(r"(\d+)\s*(crore|cr)\b",     lambda m: str(int(m.group(1)) * 10000000), s)
    s = re.sub(r"(\d+)\s*(million|mn|m)\b", lambda m: str(int(m.group(1)) * 1000000), s)
    s = re.sub(r"(\d+)\s*k\b",              lambda m: str(int(m.group(1)) * 1000), s)
    m = _NUMBER_RE.search(s)
    return _to_num(m.group(0)) if m else None

def _extract_top_n(txt, default=None):
    """
    Look for 'top 10', 'show 8', 'list 3 cars', etc.
    Returns an int or None if user didn't ask for a specific N.
    """
    m = re.search(r"\btop\s+(\d+)\b", txt)
    if m: return int(m.group(1))
    m = re.search(r"\bshow\s+(\d+)\b", txt)
    if m: return int(m.group(1))
    m = re.search(r"\blist\s+(\d+)\b", txt)
    if m: return int(m.group(1))
    m = re.search(r"\b(\d+)\s+cars?\b", txt)
    if m: return int(m.group(1))
    return default

def _canon_cols(df, cols):
    return dict(
        price = cols["price"] if cols["price"] in df.columns else None,
        year  = cols["year"] if cols["year"] in df.columns else None,
        km    = cols["km"] if cols["km"] in df.columns else None,
        power = cols["power"] if cols["power"] in df.columns else None,
        soft  = cols["soft"] if cols["soft"] in df.columns else None,
        brand = "brand" if "brand" in df.columns else None,
        model = "model" if "model" in df.columns else None,
        car   = "car" if "car" in df.columns else None,
        fuel  = cols["fuel"] if cols["fuel"] in df.columns else None,
        trans = cols["trans"] if cols["trans"] in df.columns else None,
        name  = "name" if "name" in df.columns else None,
        body  = cols["body"] if cols["body"] in df.columns else None,
        make  = cols["make"] if cols["make"] in df.columns else None,
        mileage = cols["mileage"] if cols["mileage"] in df.columns else None,
    )


STOPWORDS = {
    "what","whats","which","is","are","the","a","an","for","show","list","top","best","good",
    "with","and","or","of","to","please","give","find","me","under","over","between","vs",
    "any","then","buy"
}

def _model_tokens(ql: str):
    toks = re.findall(r"[a-z0-9\-]+", ql)
    return [t for t in toks if len(t) >= 2 and t not in STOPWORDS]

def _apply_text_filter_on_model(df, ql, c, brand_already_applied: bool):
    """
    Try to narrow by tokens against car/name/model, but ONLY if at least one token hits.
    - Do NOT remove brand tokens unless a brand filter was already applied.
    - If no token hits any row, return the original df (no filtering).
    """
    cols = [x for x in [c.get("car"), c.get("name"), c.get("model")] if x]
    if not cols:
        return df, None

    blob = df[cols].astype(str).agg(" ".join, axis=1).str.lower()
    tokens = _model_tokens(ql)

    # If brand wasn't already filtered, keep brand words in tokens
    if brand_already_applied and "brand" in df.columns:
        brands_low = {str(b).lower() for b in df["brand"].dropna().unique()}
        tokens = [t for t in tokens if t not in brands_low]

    if not tokens:
        return df, None

    # Keep only tokens that actually hit at least one row
    tokens_that_hit = [t for t in tokens if blob.str.contains(re.escape(t), na=False).any()]
    if not tokens_that_hit:
        # Nothing in the dataset matches any token -> don't filter; let other logic handle it.
        return df, None

    # Require all hitting tokens to be present in a row
    mask = pd.Series(True, index=df.index)
    for t in tokens_that_hit:
        mask &= blob.str.contains(re.escape(t), na=False)

    filtered = df.loc[mask]
    note = "model/name contains: " + ", ".join(tokens_that_hit[:5]) + (" …" if len(tokens_that_hit) > 5 else "")
    return filtered, note


def summarize_table(df, c, k=TOP_N):
    score = c["soft"]
    sort_cols = [score] if score in df.columns else ([c["price"]] if c["price"] in df.columns else [df.columns[0]])
    asc = [False] if score in df.columns else [True]
    tbl = df.sort_values(sort_cols, ascending=asc).head(k)
    show_cols = [x for x in [c["year"], c.get("make"), c["model"], c["price"], c.get("mileage"), c["km"], score] if x in tbl.columns]
    return tbl[show_cols] if show_cols else tbl.head(k)

def relevant_rows(df, query, c, limit=25):
    if df.empty or not query or not query.strip(): return df.head(limit)
    q = query.lower()
    text_cols = [x for x in [c.get("make"), c["model"], c["fuel"], c["trans"], c["body"]] if x in df.columns]
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

def to_context(df, c, n=30):
    candidate_cols = [
        "id_placeholder",        # will be removed; we insert id later
        "car","brand","model","year",
        "price_eur","price_eur_str",   # <— Euro fields only
        "km_driven","soft_buy_score","power_bhp","engine_cc",
        c.get("mileage") or "mileage_val",
        "fuel","transmission",
        "body_norm"
    ]
    keep = [col for col in candidate_cols if isinstance(col, str) and col in df.columns]
    small = df[keep].head(n).reset_index(drop=True)
    small.insert(0, "id", small.index + 1)
    return small.to_dict(orient="records")


    
def ask_llm_with_schema(schema, rows, question):
    if client is None:
        return "Chat unavailable (OpenAI client not initialized)."
    payload = {"schema": schema, "rows": rows, "question": question}
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)}
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# ========= NEW: flexible clustering helpers =========
def build_variable_catalog(df, cols):
    """Return the variables customers care most about, with UI metadata."""
    cat = {}

    # Numeric (prefer your canonical columns)
    if "year" in df.columns:
        cat["Year"] = {"key":"year", "type":"numeric"}
    if "price_eur" in df.columns:
        cat["Price (EUR)"] = {"key":"price_eur", "type":"numeric"}
    if cols.get("km") in df.columns:
        cat["Mileage (km_driven)"] = {"key": cols["km"], "type":"numeric"}
    if "power_bhp" in df.columns:
        cat["Power (bhp)"] = {"key":"power_bhp", "type":"numeric"}
    if "engine_cc" in df.columns:
        cat["Engine (cc)"] = {"key":"engine_cc", "type":"numeric"}
    if "seats" in df.columns:
        cat["Seats"] = {"key":"seats", "type":"numeric"}

    # Categorical
    if "brand" in df.columns:
        cat["Brand"] = {"key":"brand", "type":"categorical"}
    if "fuel" in df.columns:
        cat["Fuel"] = {"key":"fuel", "type":"categorical"}
    if "transmission" in df.columns:
        cat["Transmission"] = {"key":"transmission", "type":"categorical"}
    # Use normalized body if available
    if "body_norm" in df.columns:
        cat["Body"] = {"key":"body_norm", "type":"categorical"}

    # Optional extras if you have them
    if "owner" in df.columns:
        cat["Owner"] = {"key":"owner", "type":"categorical"}
    if "seller_type" in df.columns:
        cat["Seller Type"] = {"key":"seller_type", "type":"categorical"}

    return cat

def render_filters_ui(df, picked_vars, catalog):
    """
    Build per-variable filters UI and return a dict of {key: filter_spec}.
    For numeric: (min, max). For categorical: list of chosen categories (or all).
    """
    import streamlit as st
    filters = {}
    for label in picked_vars:
        meta = catalog[label]
        key = meta["key"]
        if meta["type"] == "numeric" and key in df.columns:
            s = pd.to_numeric(df[key], errors="coerce")
            s = s.dropna()
            if s.empty: 
                continue
            lo, hi = float(s.min()), float(s.max())
            # Nicer bounds for price
            step = 1.0
            if key == "price_eur":
                step = max(100.0, (hi - lo) / 100.0)
            v1, v2 = st.slider(
                f"{label} range", 
                min_value=float(lo), max_value=float(hi),
                value=(float(lo), float(hi)), step=step
            )
            filters[key] = ("range", (v1, v2))
        elif meta["type"] == "categorical" and key in df.columns:
            choices = sorted([c for c in df[key].dropna().unique().tolist()])
            picked = st.multiselect(f"{label} (choose one or more)", choices, default=choices)
            filters[key] = ("in", picked)
    return filters

def apply_filters(df, filters):
    out = df.copy()
    for key, (ftype, spec) in filters.items():
        if ftype == "range":
            lo, hi = spec
            out = out[(pd.to_numeric(out[key], errors="coerce") >= lo) &
                      (pd.to_numeric(out[key], errors="coerce") <= hi)]
        elif ftype == "in":
            # keep all if user left default (all values)
            if spec:
                out = out[out[key].isin(spec)]
    return out

def run_clustering_flexible(df, numeric_cols, categorical_cols, k=5):
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.cluster import KMeans
    
    if df.empty:
        return None

    use_cols = [c for c in (numeric_cols + categorical_cols) if c in df.columns]
    if not use_cols:
        return None

    X = df[use_cols].copy()
    # coerce numerics
    for c in [c for c in numeric_cols if c in X.columns]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # drop rows missing any numeric col (if we have numeric cols)
    if numeric_cols:
        X = X.dropna(subset=[c for c in numeric_cols if c in X.columns])
    if X.empty:
        return None

    # If too few rows for the requested K, lower K safely
    k_eff = min(k, len(X))
    if k_eff < 2:
        return None

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [c for c in numeric_cols if c in X.columns]) if numeric_cols else ("num","drop",[]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in categorical_cols if c in X.columns]) if categorical_cols else ("cat","drop",[]),
        ]
    )
    pipe = Pipeline([("prep", pre), ("km", KMeans(n_clusters=k_eff, random_state=42, n_init="auto"))])
    labels = pipe.fit_predict(X)
    return pd.Series(labels, index=X.index, name="cluster")


def summarize_clusters(df_run, labels, numeric_cols_selected, categorical_cols_selected):
    """
    df_run: the filtered dataset being clustered (after user filters)
    labels: pd.Series of cluster ids (index aligned with df_run)
    numeric_cols_selected: list of numeric variables the user picked for clustering/labeling
    categorical_cols_selected: list of categorical variables the user picked
    """
    if labels is None or labels.empty:
        return pd.DataFrame()

    base = df_run.loc[labels.index].copy()
    base["cluster"] = labels.values

    # ensure numeric
    for c in numeric_cols_selected:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # compute 5-bucket cutpoints on current working set
    qs = _quantiles_for(base, numeric_cols_selected)

    parts = []
    for cid, g in base.groupby("cluster"):
        row = {"cluster": int(cid), "count": int(len(g))}
        # medians for selected numeric
        numeric_medians = {}
        for c in numeric_cols_selected:
            if c in g.columns:
                med = g[c].median(skipna=True)
                numeric_medians[c] = med
                row[c + "_median"] = med

        # top for selected categoricals
        top_cats = {}
        for c in categorical_cols_selected:
            if c in g.columns and not g[c].dropna().empty:
                tv = g[c].mode(dropna=True).iloc[0]
                top_cats[c] = tv
                row[c + "_top"] = tv
            else:
                row[c + "_top"] = None

        # pretty € median if present
        if "price_eur" in numeric_cols_selected and "price_eur_median" in row and pd.notna(row["price_eur_median"]):
            row["price_eur_median_str"] = _fmt_eur(row["price_eur_median"])

        # human label (ONLY selected vars)
        row["label"] = _fmt_label_parts_selected(numeric_cols_selected, categorical_cols_selected,
                                                 numeric_medians, qs, top_cats)
        parts.append(row)

    summ = pd.DataFrame(parts).sort_values("cluster").reset_index(drop=True)
    return summ


# ========= 5-bucket labeling (only for selected variables) =========

def _quantiles_for(df, cols, q=(0.1, 0.33, 0.66, 0.9)):
    """
    Return cut points {col: (q10, q33, q66, q90)} for numeric columns on the current working set.
    """
    qs = {}
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) >= 10:
                qs[c] = tuple(float(s.quantile(x)) for x in q)
    return qs

def _bucket5(val, q10, q33, q66, q90):
    if val is None or pd.isna(val): return "mid"
    if val <= q10: return "very_low"
    if val <= q33: return "low"
    if val <= q66: return "mid"
    if val <= q90: return "high"
    return "very_high"

def _pretty5(col, b):
    # Customize per variable
    if col == "year":
        M = {"very_low":"Very old","low":"Old","mid":"Mid-age","high":"New","very_high":"Very new"}
        return M[b]
    if col == "price_eur":
        M = {"very_low":"Ultra-budget","low":"Budget","mid":"Mid-price","high":"Premium","very_high":"Ultra-premium"}
        return M[b]
    if col == "km_driven":
        M = {"very_low":"Very low miles","low":"Low miles","mid":"Mid miles","high":"High miles","very_high":"Very high miles"}
        return M[b]
    if col == "power_bhp":
        M = {"very_low":"Very low power","low":"Low power","mid":"Mid power","high":"High power","very_high":"Very high power"}
        return M[b]
    if col == "engine_cc":
        M = {"very_low":"Very small engine","low":"Small engine","mid":"Mid engine","high":"Large engine","very_high":"Very large engine"}
        return M[b]
    if col == "seats":
        M = {"very_low":"Very few seats","low":"Few seats","mid":"Standard seats","high":"Many seats","very_high":"Very many seats"}
        return M[b]
    # fallback
    return f"{col}:{b}"

def _fmt_label_parts_selected(selected_numeric, selected_cats, numeric_medians, qs, top_cats):
    """
    Build label using ONLY variables the user selected.
    - numeric: 5-bucket words
    - categorical: top category (body_norm/fuel/transmission/brand etc.) but only if selected
    """
    parts = []

    # order for readability; include only those that were selected
    ordered_numeric = [c for c in ["year","price_eur","km_driven","power_bhp","engine_cc","seats"]
                       if c in selected_numeric]

    for col in ordered_numeric:
        med = numeric_medians.get(col, None)
        if col in qs and med is not None:
            q10, q33, q66, q90 = qs[col]
            b = _bucket5(med, q10, q33, q66, q90)
            parts.append(_pretty5(col, b))

    # reasonable order for cats; include only selected
    ordered_cats = [c for c in ["body_norm","fuel","transmission","brand"] if c in selected_cats]
    for col in ordered_cats:
        topv = top_cats.get(col, None)
        if topv:
            if col == "body_norm":
                parts.append(str(topv).upper() if str(topv).lower() in {"suv","mpv"} else str(topv).title())
            else:
                parts.append(str(topv))

    # keep concise
    return " • ".join(parts[:4]) if parts else ""



def top_categories_series(df, labels, col, topn=6):
    """Return a small frame of top categories for a given column within a cluster."""
    base = df.loc[labels.index].copy()
    base["cluster"] = labels.values
    frames = {}
    if col not in base.columns:
        return frames
    for cid, g in base.groupby("cluster"):
        vc = (g[col].fillna("—").value_counts().rename_axis(col).reset_index(name="count").head(topn))
        frames[int(cid)] = vc
    return frames



def run_clustering(df, feature_cols, k=4):
    try:
        data = df[feature_cols].dropna().copy()
        if data.empty or len(data) < k: return None, None
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        scaler = StandardScaler()
        X = scaler.fit_transform(data.values)
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        clustered = data.copy(); clustered["cluster"] = labels
        centers_std = km.cluster_centers_
        centers = pd.DataFrame(scaler.inverse_transform(centers_std), columns=feature_cols)
        centers.index.name = "cluster"; centers.reset_index(inplace=True)
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

def cluster_labels_from_centers(centers: pd.DataFrame) -> dict:
    """
    Turn cluster centers into friendly labels using quantiles across available metrics.
    Example output: 'New • Low miles • Budget • Mid power'
    """
    if centers is None or centers.empty:
        return {}

    label_cols = [c for c in ["year", "price_eur_str", "km_driven", "power_bhp"] if c in centers.columns]
    if not label_cols:
        return {int(r["cluster"]): f"Cluster {int(r['cluster'])}" for _, r in centers.iterrows()}

    # Compute global quantiles per feature for bucketing
    qs = {}
    for c in label_cols:
        qs[c] = centers[c].quantile([0.33, 0.66]).values.tolist()  # [low, high]

    def bucket(cname, val):
        low, high = qs[cname]
        if val <= low: return "low"
        if val >= high: return "high"
        return "mid"

    def pretty(cname, b):
        if cname == "year":
            return {"low":"Old", "mid":"Mid-age", "high":"New"}[b]
        if cname == "price_eur_str":
            return {"low":"Budget", "mid":"Mid-price", "high":"Premium"}[b]
        if cname == "km_driven":
            return {"low":"Low miles", "mid":"Mid miles", "high":"High miles"}[b]
        if cname == "power_bhp":
            return {"low":"Low power", "mid":"Mid power", "high":"High power"}[b]
        return f"{cname}:{b}"

    labels = {}
    for _, row in centers.iterrows():
        parts = []
        for c in ["year", "price_eur_str", "km_driven", "power_bhp"]:
            if c in centers.columns:
                parts.append(pretty(c, bucket(c, row[c])))
        labels[int(row["cluster"])] = " • ".join(parts) if parts else f"Cluster {int(row['cluster'])}"
    return labels

def representative_rows(df_with_cluster, clus_col="cluster", n=8):
    """
    For each cluster, pick a few representatives nearest the cluster median on available key fields.
    """
    out = {}
    by = [c for c in ["year", "price_eur", "km_driven", "power_bhp"] if c in df_with_cluster.columns]
    if not by:
        for cid, g in df_with_cluster.groupby(clus_col):
            out[int(cid)] = g.head(min(n, len(g)))
        return out

    for cid, g in df_with_cluster.groupby(clus_col):
        med = g[by].median(numeric_only=True)
        dist = ((g[by] - med).abs()).sum(axis=1)
        out[int(cid)] = g.loc[dist.nsmallest(n).index]
    return out


# ==================== ROLE GATE ====================
st.session_state.setdefault("role", None)
st.session_state.setdefault("authed", False)
st.session_state.setdefault("section", None)
st.session_state.setdefault("chat", [])

# ---------- Top bar ----------
if st.session_state["role"]:
    # Button on top, single-line title underneath
    c1, _ = st.columns([0.3, 0.7])
    with c1:
        if st.button("⬅ Switch role", width="stretch", key="btn_switch_role"):
            st.session_state.update(role=None, authed=False, section=None)
            st.rerun()
        st.markdown(
            "<h1 style='white-space: nowrap; margin-top:0.5rem;'>Dealer Deals</h1>",
            unsafe_allow_html=True
        )
else:
    st.title("Dealer Deals") 

# ---------- Who are you? ----------
if not st.session_state["role"]:
    st.subheader("Who are you?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Customer", width="stretch", key="btn_role_customer"):
            st.session_state.update(role="customer", authed=True, section="Chat")
            st.rerun()
    with col2:
        if st.button("Owner", width="stretch", key="btn_role_owner"):
            st.session_state.update(role="owner", authed=False)
            st.rerun()
    st.stop()

# ---------- Owner login ----------
if st.session_state["role"] == "owner" and not st.session_state["authed"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    center1, center2, center3 = st.columns([0.35, 0.3, 0.35])
    with center2:
        st.markdown("### Owner Login")
        pwd = st.text_input("Password", type="password", label_visibility="collapsed")
        if st.button("Unlock", type="primary"):
            if pwd == OWNER_PASSWORD:
                st.session_state.update(authed=True, section="Best Deals")
                st.success("Access granted.")
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()

# ---------- After auth ----------
df = load_data()
if df.empty: st.stop()
# --- Ensure Euro price exists & is numeric ---

if "price_eur" not in df.columns and "selling_price" in df.columns:
    # your rule: 5,500,000 -> €55,000.00  (divide by 100)
    df["price_eur"] = pd.to_numeric(df["selling_price"], errors="coerce") / 100.0
    
# --- Normalize mileage to a numeric column ---
if "mileage_val" not in df.columns and "mileage" in df.columns:
    # extract the leading number from strings like "23.4 kmpl", "18 km/l", etc.
    m = df["mileage"].astype(str).str.extract(r'([\d\.]+)', expand=False)
    df["mileage_val"] = pd.to_numeric(m, errors="coerce")


# nice formatter: €55 000.00 (space as thousands sep)
def _fmt_eur(x):
    try:
        return "€{:,.2f}".format(float(x)).replace(",", " ")
    except Exception:
        return "€—"

def _best_table(df_like: pd.DataFrame) -> pd.DataFrame:
    # exact order, but fall back safely if any column is missing
    wanted = ["brand", "model", "year", "price_eur_str", "km_driven"]
    return df_like[[c for c in wanted if c in df_like.columns]]


df["price_eur_str"] = df["price_eur"].apply(_fmt_eur) if "price_eur" in df.columns else "€—"

df = ensure_car_name(df)
cols = pick_cols(df)
st.caption(f"Loaded **{DATA_PATH}** — {df.shape[0]:,} rows")


BODY_COLS = ["body_type", "type", "segment", "category"]

BODY_MAP = {
    "suv": "suv", "crossover": "suv", "cross-over": "suv",
    "mpv": "mpv", "muv": "mpv",
    "sedan": "sedan",
    "hatchback": "hatchback", "hatch": "hatchback",
    "wagon": "wagon", "estate": "wagon",
    "coupe": "coupe",
    "convertible": "convertible", "cabrio": "convertible", "cabriolet": "convertible", "roadster": "convertible",
    "pickup": "pickup", "pick-up": "pickup", "truck": "pickup",
    "van": "van", "minivan": "van",
}

def _resolve_body_col(df):
    return next((c for c in BODY_COLS if c in df.columns), None)

def add_body_norm(df):
    col = _resolve_body_col(df)
    if not col:
        return df, None
    s = df[col].astype(str).str.strip().str.lower()
    # normalize by keywords
    def _norm(x):
        for k, v in BODY_MAP.items():
            if k in x:
                return v
        return None
    df["body_norm"] = s.apply(_norm)
    return df, "body_norm"


def _rank_key_exists(col): 
    return col in df.columns

SCHEMA_HINT = {
    "columns": list(df.columns),
    "primary_keys": [c for c in ["car","brand","model","year"] if c in df.columns],
    "categories": {
        "body": sorted(df["body_norm"].dropna().unique().tolist()) if "body_norm" in df.columns else [],
        "brand": sorted(df["brand"].dropna().unique().tolist()) if "brand" in df.columns else [],
    },
    "synonyms": {
        "SUV": ["suv","crossover"],
        "MPV": ["mpv","muv"],
        "Pickup": ["pickup","truck"],
        "Wagon": ["wagon","estate"],
        "Convertible": ["convertible","cabrio","cabriolet","roadster"],
        "Hatchback": ["hatchback","hatch"]
    },
    "ranking": [
        {"column": "soft_buy_score", "direction": "desc"} if "soft_buy_score" in df.columns else None,
        {"column": "price_eur",      "direction": "asc"}  if "price_eur" in df.columns else None,
        {"column": "year",           "direction": "desc"} if "year" in df.columns else None,
        {"column": "km_driven",      "direction": "asc"}  if "km_driven" in df.columns else None,
    ]
}
SCHEMA_HINT["ranking"] = [r for r in SCHEMA_HINT["ranking"] if r]

df, BODY_USED = add_body_norm(df)

# --- Demand index helpers (add near ML helpers) ---
def _minmax(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.max() == s.min() or s.isna().all(): return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def compute_demand_index(df, hedonic_pack):
    if hedonic_pack is None:
        return None

    # Unpack model
    if isinstance(hedonic_pack, tuple) and len(hedonic_pack) == 2:
        reg, meta = hedonic_pack
        expected_num = list(meta.get("num", []))
        expected_cat = list(meta.get("cat", []))
    else:
        reg = hedonic_pack
        expected_num, expected_cat = [], []

    # --- Build feature frame + engineered cols FIRST ---
    X = df.copy()
    X["price_eur"] = pd.to_numeric(X.get("price_eur"), errors="coerce")
    X["km_driven"] = pd.to_numeric(X.get("km_driven"), errors="coerce")
    X["year"] = pd.to_numeric(X.get("year"), errors="coerce")
    X = X.dropna(subset=["price_eur","year"])

    # engineered
    this_year = pd.Timestamp.today().year
    X["age"] = (this_year - X["year"]).clip(lower=0)
    X["log_km"] = np.log1p(X["km_driven"].clip(lower=0)) if "km_driven" in X.columns else 0.0
    X["age2"] = X["age"]**2

    # if pipeline meta wasn't saved, read the columns it expects from the ColumnTransformer
    if (not expected_num and not expected_cat) and hasattr(reg, "named_steps") and "prep" in reg.named_steps:
        pre = reg.named_steps["prep"]
        # prefer fitted transformers_ (post-fit)
        tr_list = getattr(pre, "transformers_", getattr(pre, "transformers", []))
        for name, tr, cols_used in tr_list:
            if cols_used in (None, "drop"): 
                continue
            if name == "num":
                expected_num = list(cols_used)
            elif name == "cat":
                expected_cat = list(cols_used)

    # Ensure all expected columns exist (imputers will handle NaNs)
    for col in (expected_num + expected_cat):
        if col not in X.columns:
            X[col] = np.nan

    # Build the matrix in the exact column order the pipeline expects
    feature_cols = [c for c in (expected_num + expected_cat) if c in X.columns]
    if not feature_cols:
        return None

    Z = X[feature_cols]

    # --- Predict log price and compute residuals ---
    pred_log = reg.predict(Z)
    resid = np.log(X["price_eur"].clip(lower=1.0)) - pred_log
    X["_resid"] = resid

    # Grouping key
    keys = [k for k in ["brand","model","body_norm"] if k in X.columns]
    if not keys: 
        keys = ["brand"] if "brand" in X.columns else None
    if not keys:
        return None

    g = X.groupby(keys, dropna=False)
    out = g.agg(
        n=("price_eur","size"),
        median_price=("price_eur","median"),
        resid_mean=("_resid","mean"),
        soft_mean=("soft_buy_score","mean") if "soft_buy_score" in X.columns else ("_resid","mean")
    ).reset_index()

    # Blend into demand index
    def _minmax(s):
        s = s.replace([np.inf, -np.inf], np.nan)
        if s.max() == s.min() or s.isna().all(): 
            return pd.Series(0.0, index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    out["scarcity"] = 1.0 / out["n"].clip(lower=1)
    out["premium"] = out["resid_mean"]
    out["desirability"] = out["soft_mean"] if "soft_buy_score" in X.columns else out["premium"]

    sc = _minmax(out["scarcity"])
    pr = _minmax(out["premium"])
    ds = _minmax(out["desirability"])
    out["demand_index"] = (0.4*sc + 0.4*pr + 0.2*ds)

    out = out.sort_values("demand_index", ascending=False)
    return out




# ---------- Navigation ----------
role = st.session_state["role"]
if role == "owner":
    if st.session_state["section"] not in ["Best Deals", "Classification", "Demand"]:
        st.session_state["section"] = "Best Deals"

    st.session_state.setdefault("show_check_modal", False)

    c1, c2, c3, c4 = st.columns([0.2, 0.2, 0.2, 0.4])
    with c1:
        if st.button("Best Deals", type="secondary", width="stretch", key="nav_best"):
            st.session_state["section"] = "Best Deals"; st.rerun()
    with c2:
        if st.button("Classification", type="secondary", width="stretch", key="nav_class"):
            st.session_state["section"] = "Classification"; st.rerun()
    with c3:
        if st.button("Demand", type="secondary", width="stretch", key="nav_demand"):
            st.session_state["section"] = "Demand"; st.rerun()
    with c4:
        if st.button("Add & Check Car", type="primary", width="stretch", key="nav_checkcar"):
            st.session_state["show_check_modal"] = True

else:
    if st.session_state["section"] not in ["Chat", "Clustering"]:
        st.session_state["section"] = "Chat"
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💬 Chat", type="secondary", width="stretch", key="nav_chat"):
            st.session_state["section"] = "Chat"; st.rerun()
    with c2:
        if st.button("🧩 Clustering", type="secondary", width="stretch", key="nav_cluster"):
            st.session_state["section"] = "Clustering"; st.rerun()

st.write("")

# ---------- Owner: Add & Check Car modal ----------
def render_check_car_form():
    """Renders the form and returns (checked, sample_df, canceled)."""
    with st.form("check_car_form", clear_on_submit=False):
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            year = st.number_input("year", min_value=1980, max_value=2030, value=2018, step=1)
            seats = st.number_input("seats", min_value=2, max_value=10, value=5, step=1)
            engine_cc = st.number_input("engine_cc", min_value=300, max_value=8000, value=1498, step=10)
        with n2:
            selling_price = st.number_input("selling_price", min_value=0, value=500000, step=5000)
            mileage_val = st.number_input("mileage_val", min_value=0.0, value=18.0, step=0.1, format="%.1f")
            power_bhp = st.number_input("power_bhp", min_value=0.0, value=100.0, step=1.0, format="%.1f")
        with n3:
            km_driven = st.number_input("km_driven", min_value=0, value=40000, step=500)
            torque_nm = st.number_input("torque_nm", min_value=0.0, value=140.0, step=1.0, format="%.1f")
            torque_rpm = st.number_input("torque_rpm", min_value=0, value=4000, step=100)
        with n4:
            fuel_choice   = st.selectbox("fuel", ["CNG","Diesel","LPG","Petrol"], index=3)
            seller_choice = st.selectbox("seller_type", ["Dealer","Individual","Trustmark Dealer"], index=0)
            trans_choice  = st.selectbox("transmission", ["Automatic","Manual"], index=1)
            owner_choice  = st.selectbox("owner", ["First","Second","TestDrive","Third+"], index=0)

        brands = [
            "Ambassador","Ashok","Audi","BMW","Chevrolet","Daewoo","Datsun","Fiat","Force","Ford",
            "Honda","Hyundai","Isuzu","Jaguar","Jeep","Kia","Land Rover","Lexus","MG","Mahindra",
            "Maruti","Mercedes-Benz","Mitsubishi","Nissan","Opel","Peugeot","Renault","Skoda",
            "Tata","Toyota","Volkswagen","Volvo"
        ]
        brand = st.selectbox("brend", brands, index=21 if "Maruti" in brands else 0)

        cta1, cta2, _ = st.columns([0.2, 0.2, 0.6])
        with cta1:
            checked  = st.form_submit_button("Check", type="primary", width="stretch", key="frm_check")
        with cta2:
            canceled = st.form_submit_button("Cancel", width="content", key="frm_cancel")

    if not (checked or canceled):
        return False, None, False
    if canceled:
        return False, None, True

    # build one-hot sample
    sample = {
        "year": int(year),
        "selling_price": float(selling_price),
        "km_driven": int(km_driven),
        "seats": int(seats),
        "mileage_val": float(mileage_val),
        "engine_cc": int(engine_cc),
        "power_bhp": float(power_bhp),
        "torque_nm": float(torque_nm),
        "torque_rpm": int(torque_rpm),
    }

    for col in ["fuel_CNG","fuel_Diesel","fuel_LPG","fuel_Petrol"]:
        sample[col] = 1 if col.lower().endswith(fuel_choice.lower()) else 0

    seller_cols = {
        "Dealer": "seller_type_Dealer",
        "Individual": "seller_type_Individual",
        "Trustmark Dealer": "seller_type_Trustmark Dealer",
    }
    for v in seller_cols.values(): sample[v] = 0
    sample[seller_cols[seller_choice]] = 1

    trans_cols = {"Automatic":"transmission_Automatic","Manual":"transmission_Manual"}
    for v in trans_cols.values(): sample[v] = 0
    sample[trans_cols[trans_choice]] = 1

    owner_cols = {"First":"owner_First","Second":"owner_Second","TestDrive":"owner_TestDrive","Third+":"owner_Third+"}
    for v in owner_cols.values(): sample[v] = 0
    sample[owner_cols[owner_choice]] = 1

    make_cols_all = [
        'make_Ambassador','make_Ashok','make_Audi','make_BMW','make_Chevrolet','make_Daewoo',
        'make_Datsun','make_Fiat','make_Force','make_Ford','make_Honda','make_Hyundai',
        'make_Isuzu','make_Jaguar','make_Jeep','make_Kia','make_Land Rover','make_Lexus',
        'make_MG','make_Mahindra','make_Maruti','make_Mercedes-Benz','make_Mitsubishi',
        'make_Nissan','make_Opel','make_Peugeot','make_Renault','make_Skoda','make_Tata',
        'make_Toyota','make_Volkswagen','make_Volvo'
    ]
    for mc in make_cols_all: sample[mc] = 0
    sample["make_" + brand] = 1

    model_feature_cols = df.drop(columns=['soft_buy_score','good_buy'], errors='ignore').columns
    sample_df = pd.DataFrame([sample]).reindex(columns=model_feature_cols, fill_value=0)
    return True, sample_df, False


def _predict_and_show(sample_df):
    try:
        loaded_model = joblib.load('classification_model.pkl')

        feat_names = getattr(loaded_model, "feature_names_in_", None)
        if feat_names is not None:
            for col in feat_names:
                if col not in sample_df.columns:
                    sample_df[col] = 0
            sample_df = sample_df.reindex(columns=feat_names, fill_value=0)

        sample_df = sample_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        pred = float(loaded_model.predict(sample_df)[0])

        if pred >= 0.7:
            klass, badge = "Class A — Premium", "🟢"
            st.success(f"Predicted soft_buy_score: **{pred:.4f}**  \n{badge} **{klass}**")
        elif pred >= 0.5:
            klass, badge = "Class B — Good", "🟡"
            st.warning(f"Predicted soft_buy_score: **{pred:.4f}**  \n{badge} **{klass}**")
        else:
            klass, badge = "Class C — Fair/Poor", "🟠"
            st.error(f"Predicted soft_buy_score: **{pred:.4f}**  \n{badge} **{klass}**")

    except Exception as e:
        st.error(f"Failed to load or run model: {e}")


# Open a real modal if st.dialog (decorator) exists; else fallback to expander
if role == "owner" and st.session_state.get("show_check_modal", False):
    if hasattr(st, "dialog"):
        # --- decorator style ---
        @st.dialog("Add & Check Car")  # remove width arg for max compatibility
        def open_check_dialog():
            checked, sample_df, canceled = render_check_car_form()
            if canceled:
                st.session_state["show_check_modal"] = False
                st.rerun()
            if checked and sample_df is not None:
                _predict_and_show(sample_df)
            if st.button("Close", width="stretch", key="dlg_close"):
                st.session_state["show_check_modal"] = False
                st.rerun()

        # Trigger the dialog
        open_check_dialog()

    else:
        # Fallback for older Streamlit
        with st.expander("➕ Add & Check Car", expanded=True):
            checked, sample_df, canceled = render_check_car_form()
            if canceled:
                st.session_state["show_check_modal"] = False
                st.rerun()
            if checked and sample_df is not None:
                _predict_and_show(sample_df)
            st.button("Close", width="stretch", key="exp_close", on_click=lambda: st.session_state.update(show_check_modal=False))

section = st.session_state["section"]

# ----------- Best Deals (owners) -----------
if section == "Best Deals" and role == "owner":

    score_col = cols["soft"]
    if score_col not in df.columns:
        st.info("soft_buy_score not found—showing a basic top list.")
        tmp = df.copy()  # already has reliable 'car'
        show_cols = [c for c in ["brand","model","year","price_eur_str","km_driven"] if c in tmp.columns]
        st.dataframe(tmp.head(10)[show_cols], width="stretch", hide_index=True)
        st.stop()


    # --- Class A ---
    class_a = df[df[score_col] >= 0.7].copy()
    st.markdown("### Class A — Premium Deals")
    if class_a.empty:
        st.caption("No vehicles meet this criterion.")
    else:
        top_a = ensure_car_name(class_a.sort_values(score_col, ascending=False).head(10).copy())
        show_cols = [c for c in ["brand","model","year","price_eur_str","km_driven"] if c in top_a.columns]
        st.dataframe(top_a[show_cols], width="stretch", hide_index=True)


    # --- Class B ---
    class_b = df[(df[score_col] >= 0.5) & (df[score_col] < 0.7)].copy()
    st.markdown("### Class B — Good Deals")
    if class_b.empty:
        st.caption("No vehicles meet this criterion.")
    else:
        top_b = class_b.sort_values(score_col, ascending=False).head(10).copy()
        if "car" not in top_b.columns:
            if {"brand","model"}.issubset(top_b.columns):
                top_b["car"] = (top_b["brand"].fillna("") + " " + top_b["model"].fillna("")).str.strip()
            elif "brand" in top_b.columns:
                top_b["car"] = top_b["brand"]
            else:
                top_b["car"] = "—"
        show_cols = [c for c in ["brand","model","year","price_eur_str","km_driven"] if c in top_b.columns]
        st.dataframe(top_b[show_cols], width="stretch", hide_index=True)



# ----------- Classification (owners) -----------
elif section == "Classification" and role == "owner":

    score_col = cols["soft"]
    if score_col not in df.columns:
        st.info("soft_buy_score not found—cannot classify.")
        st.stop()

    # Train model for optional probability (kept, but we won't display it here)
    model = CLASSIFIER_MODEL_PATH
    slice_df = df.copy()
    if model is not None:
        try:
            clf, feat_cols = model
            mask = slice_df[feat_cols].notna().all(axis=1) if feat_cols else pd.Series(False, index=slice_df.index)
            if mask.any():
                probs = clf.predict_proba(slice_df.loc[mask, feat_cols])[:, 1]
                slice_df.loc[mask, "good_deal_prob"] = np.round(probs, 3)
        except Exception:
            pass

    # Classes by soft_buy_score
    class_a = slice_df[slice_df[score_col] >= 0.7].sort_values(score_col, ascending=False).copy()
    class_b = slice_df[(slice_df[score_col] >= 0.5) & (slice_df[score_col] < 0.7)].sort_values(score_col, ascending=False).copy()
    class_c = slice_df[(slice_df[score_col] >= 0.0) & (slice_df[score_col] < 0.5)].sort_values(score_col, ascending=False).copy()

    # We'll show only the requested columns
    def show_block(df_block, title, caption_text):
        st.markdown(title)
        if df_block.empty:
            st.caption("No vehicles meet this criterion.")
            return
        tmp = ensure_car_name(df_block)  # standardize
        cols_to_show = [c for c in ["brand","model","year","price_eur_str","km_driven"] if c in tmp.columns]
        st.dataframe(tmp[cols_to_show], width="stretch", hide_index=True)
        st.caption(caption_text)



    show_block(class_a, "### Class A — Premium Deals", "All vehicles with score ≥ 0.7.")
    st.divider()
    show_block(class_b, "### Class B — Good Deals", "All vehicles within the 0.5–0.7 range.")
    st.divider()
    show_block(class_c, "### Class C — Fair/Poor Deals", "All vehicles below 0.5.")

elif section == "Demand" and role == "owner":
    st.subheader("🔥 Demand Radar")

    hedonic = load_hedonic_model()
    if hedonic is None:
        st.info("Need at least price_eur, year (km/brand/body/fuel/transmission/model improve accuracy).")
        st.stop()

    # --- Brand filter only ---
    brand_pick = st.multiselect(
        "Brands",
        sorted(df["brand"].dropna().unique()) if "brand" in df.columns else [],
    )

    base = df.copy()
    if brand_pick and "brand" in base.columns:
        base = base[base["brand"].isin(brand_pick)]


    res = compute_demand_index(base, hedonic)
    if res is None or res.empty:
        st.info("Couldn’t compute a demand index on the current slice.")
        st.stop()

    # Show top models by demand
    # format & rename first
    # --- pretty table (Demand section only) ---
    res = res.copy()
    res["median_price"] = res["median_price"].apply(_fmt_eur)
    if "demand_index" in res.columns:
        res["demand_index"] = res["demand_index"].round(3)

    # exact columns to show (fall back safely if any are missing)
    display_cols = [c for c in ["brand", "model", "median_price", "demand_index"] if c in res.columns]

    st.dataframe(
        res[display_cols].head(25),
        width="stretch",
        hide_index=True
    )

    st.caption("Demand Index blends scarcity (few listings), price premium vs. peers (positive residual), and soft desirability.")

    
# ----------- Chat (customers) -----------
elif section == "Chat" and role == "customer":
    st.subheader("💬 Chat with DealBot")
    import re

    def _year_from_query(q):
        yrs = re.findall(r"\b(19|20)\d{2}\b", q)
        return [int(y) if isinstance(y, str) else int("".join(y)) for y in yrs] if yrs else []

    def _body_tokens(q):
        ql = q.lower()
        tokens = []
        for k, syns in {**SCHEMA_HINT["synonyms"], "Sedan": ["sedan"]}.items():
            for s in syns:
                if re.search(rf"\b{s}s?\b", ql):
                    tokens.append(s)  # store normalized synonym
        return list(set(tokens))

    def _brand_tokens(q):
        if "brand" not in df.columns: return []
        ql = q.lower()
        brands = [b for b in SCHEMA_HINT.get("categories", {}).get("brand", [])]
        return [b for b in brands if b and b.lower() in ql]

    def build_slice_for_llm(user_msg, df):
        base = df.copy()

        # Body prefilter (only if we have body_norm)
        btoks = _body_tokens(user_msg)
        if "body_norm" in base.columns and btoks:
            # map synonyms back to our normalized labels
            backmap = {s: "suv" for s in ["suv","crossover"]}
            backmap.update({s: "mpv" for s in ["mpv","muv"]})
            backmap.update({s: "pickup" for s in ["pickup","truck"]})
            backmap.update({s: "wagon" for s in ["wagon","estate"]})
            backmap.update({s: "convertible" for s in ["convertible","cabrio","cabriolet","roadster"]})
            backmap.update({"hatchback":"hatchback", "hatch":"hatchback", "sedan":"sedan"})
            wanted = {backmap.get(s, s) for s in btoks}
            base = base[base["body_norm"].isin(wanted)]

        # Year prefilter
        yrs = _year_from_query(user_msg)
        if "year" in base.columns and yrs:
            if len(yrs) == 1:
                base = base[base["year"] == yrs[0]]
            elif len(yrs) >= 2:
                lo, hi = sorted(yrs[:2])
                base = base[(base["year"] >= lo) & (base["year"] <= hi)]

        # Brand prefilter
        bt = _brand_tokens(user_msg)
        if "brand" in base.columns and bt:
            base = base[base["brand"].str.lower().isin([x.lower() for x in bt])]

        # Deduplicate so the model doesn't repeat identical lines
        dedupe_cols = [c for c in ["car","year","price_eur_str"] if c in base.columns]
        if dedupe_cols:
            base = base.drop_duplicates(subset=dedupe_cols)

        return base


    # tiny toolbar
    col_a, col_b = st.columns([0.8, 0.2])
    with col_b:
        if st.button("Reset chat", width="stretch", key="btn_reset_chat"):
            st.session_state["chat"] = []
            st.rerun()

    # show previous turns
    for who, text in st.session_state["chat"]:
        with st.chat_message("assistant" if who == "assistant" else "user"):
            st.markdown(text)

    user_msg = st.chat_input("Ask anything about the cars in this dataset…")
    if user_msg:
        st.session_state["chat"].append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)

        # Always answer from data (brief, <= 5 lines)
        # LLM-first: build a relevant slice, send schema + rows + question
        try:
            # bigger slice for recall, smaller context for the model
            pref = build_slice_for_llm(user_msg, df)
            # if we filtered too hard and got tiny, fall back to global relevance
            source_df = pref if len(pref) >= 5 else df
            rel_df = relevant_rows(source_df, user_msg, cols, limit=80)
            rows = to_context(rel_df, cols, n=30)
            ans_text = ask_llm_with_schema(SCHEMA_HINT, rows, user_msg)
        except Exception as e:
            ans_text = f"Sorry — something went wrong: {e}"

        with st.chat_message("assistant"):
            st.markdown(ans_text)
            with st.expander("Data used for this answer", expanded=False):
                # Show exactly what the model saw
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.session_state["chat"].append(("assistant", ans_text))




# ----------- Chat (customers) -----------
elif section == "Clustering" and role == "customer":
    st.subheader("🧩 Build Your Own Clusters")

    # 2.1 Variable catalog + smart defaults
    var_catalog = build_variable_catalog(df, cols)
    if not var_catalog:
        st.info("Not enough fields to cluster.")
        st.stop()

    # Suggested defaults (good for most shoppers)
    smart_defaults = [v for v in ["Year", "Price (EUR)", "Mileage (km_driven)", "Power (bhp)"] if v in var_catalog][:2]
    picked = st.multiselect(
        "Choose the variables to cluster on (numeric and/or categorical):",
        list(var_catalog.keys()),
        default=smart_defaults
    )

    if not picked:
        st.warning("Pick at least one variable.")
        st.stop()

    # 2.2 Per-variable filters
    st.markdown("### Filters")
    filters = render_filters_ui(df, picked, var_catalog)
    df_f = apply_filters(df, filters)
    st.caption(f"Filtered dataset: **{len(df_f):,}** cars")

    if df_f.empty:
        st.info("No cars match your filters. Adjust and try again.")
        st.stop()

    # 2.3 Map picks to actual column names and types
    numeric_cols = [var_catalog[p]["key"] for p in picked if var_catalog[p]["type"] == "numeric"]
    categorical_cols = [var_catalog[p]["key"] for p in picked if var_catalog[p]["type"] == "categorical"]

    # 2.4 Cluster controls (K and sample size)
    c1, c2 = st.columns(2)
    with c1:
        K = st.slider("How many clusters?", min_value=2, max_value=10, value=FIXED_K, step=1)
    with c2:
        max_rows = st.slider("Max cars to consider (speed vs. accuracy)", 500, 10000, 3000, step=500)

    # For speed: sample if huge
    if len(df_f) > max_rows:
        df_run = df_f.sample(n=max_rows, random_state=42)
    else:
        df_run = df_f

    # 2.5 Run flexible clustering
    labels = run_clustering_flexible(df_run, numeric_cols, categorical_cols, k=K)
    if labels is None:
        st.info("Not enough usable data after filtering to run clustering.")
        st.stop()

    # 2.6 Build summaries
    summ = summarize_clusters(df_run, labels, numeric_cols, categorical_cols)

    # 2.7 Show group summary
    st.markdown("### Group Summary")
    # choose columns to show: always show count, medians for common numeric, and top for categs
    show_cols = ["cluster", "label", "count"]
    for c in ["year","price_eur","km_driven","power_bhp","engine_cc","seats"]:
        cmed = c + "_median"
        if cmed in summ.columns:
            show_cols.append(cmed)
    # pretty Euro if present
    if "price_eur_median_str" in summ.columns:
        # replace numeric price median with string display
        show_cols = [c for c in show_cols if c != "price_eur_median"]
        show_cols.insert( show_cols.index("count")+1, "price_eur_median_str")

    # add top category columns
    for c in categorical_cols:
        col = c + "_top"
        if col in summ.columns:
            show_cols.append(col)

    st.dataframe(summ[show_cols], width="stretch", hide_index=True)

    # 2.8 Let the user pick a cluster
    st.markdown("### Explore a Cluster")
    cluster_labels = {int(r.cluster): (r.label or f"Cluster {int(r.cluster)}") for _, r in summ.iterrows()}
    pick_cluster = st.selectbox(
        "Show cars in cluster",
        options=summ["cluster"].tolist(),
        format_func=lambda cid: f"Cluster {cid} — {cluster_labels.get(int(cid), '')}"
    )

    # Representatives: show cars from chosen cluster
    chosen_idx = labels[labels == int(pick_cluster)].index
    display = df_run.loc[chosen_idx].copy()
    display = ensure_car_name(display)

    # Optional brand snapshot if brand is available
    if "brand" in display.columns and not display["brand"].dropna().empty:
        st.markdown("**Top brands in this cluster**")
        top_brands = (
            display["brand"].fillna("—").value_counts().rename_axis("brand").reset_index(name="count").head(6)
        )
        st.dataframe(top_brands, width="stretch", hide_index=True)

    # 2.9 Show the cars table
    nicer_cols = [c for c in ["car","year","price_eur_str","km_driven","power_bhp","engine_cc","seats","fuel","transmission","body_norm","soft_buy_score"] if c in display.columns]
    if "price_eur" in display.columns:
        display = display.sort_values(["price_eur","year"] if "year" in display.columns else ["price_eur"], na_position="last")
    st.dataframe(display[nicer_cols], width="stretch", hide_index=True)
    st.caption("Tip: tweak variables and filters above to reshape the clusters.")
