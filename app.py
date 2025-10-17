# app.py — dealer app with role gate + split sections (no sidebar)
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
import joblib

# ---------- CONFIG ----------
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.2
SYSTEM_PROMPT = (
    "You are DealBot, an expert assistant for a car dealership. "
    "Use ONLY the provided data context. Be concise (max 5 short lines)."
)
DATA_PATH = "exports/final_model_dataset.csv"
TOP_N = 20
GOOD_DEAL_THRESHOLD = 0.80
OWNER_PASSWORD = "123"
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

def pick_cols(df):
    f = lambda cands: next((c for c in cands if c in df.columns), None)
    return dict(
        price=f(["selling_price","price"]), year=f(["year"]),
        make=f(["make","manufacturer"]), model=f(["model"]),
        mileage=f(["mileage","mileage_val"]), km=f(["km_driven","kilometers","km"]),
        power=f(["power_bhp","power"]), fuel=f(["fuel"]),
        trans=f(["transmission","gearbox"]), body=f(["body_type","type"]),
        soft=f(["soft_buy_score","softbuy_score","soft_score"])
    )

def summarize_table(df, c, k=TOP_N):
    score = c["soft"]
    sort_cols = [score] if score in df.columns else ([c["price"]] if c["price"] in df.columns else [df.columns[0]])
    asc = [False] if score in df.columns else [True]
    tbl = df.sort_values(sort_cols, ascending=asc).head(k)
    show_cols = [x for x in [c["year"], c["make"], c["model"], c["price"], c["mileage"], c["km"], score] if x in tbl.columns]
    return tbl[show_cols] if show_cols else tbl.head(k)

def relevant_rows(df, query, c, limit=25):
    if df.empty or not query or not query.strip(): return df.head(limit)
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
            model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":content}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

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

@st.cache_resource
def train_classifier(df, c):
    score = c["soft"]
    if score not in df.columns: return None
    y_raw = df[score].copy()
    if set(pd.Series(y_raw.dropna().unique()).astype(int)).issubset({0,1}) and y_raw.dropna().isin([0,1]).all():
        y = y_raw.astype(int)
    else:
        y = (y_raw >= GOOD_DEAL_THRESHOLD).astype(int)
    feat_num = [x for x in [c["year"], c["mileage"], c["km"], c["price"]] if x in df.columns]
    feat_cat = [x for x in [c["make"], c["model"], c["fuel"], c["trans"], c["body"]] if x in df.columns]
    if not feat_num and not feat_cat: return None
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    pre = ColumnTransformer([
        ("num", StandardScaler(), feat_num) if feat_num else ("num","drop",[]),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02), feat_cat) if feat_cat else ("cat","drop",[]),
    ])
    safe = df.dropna(subset=feat_num + feat_cat).copy()
    if safe.empty: return None
    clf = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000))])
    clf.fit(safe[feat_num + feat_cat], y.loc[safe.index])
    return clf, feat_num + feat_cat

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
        if st.button("⬅ Switch role", use_container_width=True):
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
        if st.button("Customer", use_container_width=True):
            st.session_state.update(role="customer", authed=True, section="Chat")
            st.rerun()
    with col2:
        if st.button("Owner", use_container_width=True):
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
cols = pick_cols(df)
st.caption(f"Loaded **{DATA_PATH}** — {df.shape[0]:,} rows")

def add_brend_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Create df['brend'] from one-hot make_* columns and ensure the requested columns exist."""
    df2 = df.copy()

    # Pretty brand list (as you'd like it shown)
    brands_pretty = [
        "Ambassador","Ashok","Audi","BMW","Chevrolet","Daewoo","Datsun","Fiat","Force","Ford",
        "Honda","Hyundai","Isuzu","Jaguar","Jeep","Kia","Land Rover","Lexus","MG","Mahindra",
        "Maruti","Mercedes-Benz","Mitsubishi","Nissan","Opel","Peugeot","Renault","Skoda",
        "Tata","Toyota","Volkswagen","Volvo"
    ]
    # How these column names appear in your dataframe (lowercased, spaces -> _, hyphens kept)
    make_cols = []
    col_to_brand = {}
    for b in brands_pretty:
        normalized = b.lower().replace(" ", "_")  # keep hyphens as-is
        col = f"make_{normalized}"
        if col in df2.columns:
            make_cols.append(col)
            col_to_brand[col] = b

    if make_cols:
        vals = df2[make_cols]
        maxvals = vals.max(axis=1)
        idx = vals.idxmax(axis=1)  # returns a column name per row
        brend = idx.map(col_to_brand)
        brend = brend.where(maxvals >= 1, other=np.nan)  # only set if some make_* == 1
        df2["brend"] = brend
    else:
        df2["brend"] = np.nan

    # Normalize/alias the exact column names you want to display
    if "selling_price" not in df2.columns and "price" in df2.columns:
        df2["selling_price"] = df2["price"]

    if "km_driven" not in df2.columns:
        for alt in ["km_driven", "kilometers", "km", "mileage", "mileage_val"]:
            if alt in df2.columns:
                df2["km_driven"] = df2[alt]
                break

    if "engine_cc" not in df2.columns:
        for alt in ["engine_cc", "engine", "cc"]:
            if alt in df2.columns:
                df2["engine_cc"] = df2[alt]
                break

    if "power_bhp" not in df2.columns and "power" in df2.columns:
        df2["power_bhp"] = df2["power"]

    return df2

# ---------- Navigation ----------
role = st.session_state["role"]
if role == "owner":
    if st.session_state["section"] not in ["Best Deals", "Classification"]:
        st.session_state["section"] = "Best Deals"

    # state for modal
    st.session_state.setdefault("show_check_modal", False)

    c1, c2, c3 = st.columns([0.25, 0.25, 0.5])
    with c1:
        if st.button("Best Deals", type="secondary", use_container_width=True):
            st.session_state["section"] = "Best Deals"; st.rerun()
    with c2:
        if st.button("Classification", type="secondary", use_container_width=True):
            st.session_state["section"] = "Classification"; st.rerun()
    with c3:
        if st.button("Add & Check Car", type="primary", use_container_width=True):
            st.session_state["show_check_modal"] = True
            # no st.rerun() here
else:
    if st.session_state["section"] not in ["Chat", "Clustering"]:
        st.session_state["section"] = "Chat"
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💬 Chat", type="secondary", use_container_width=True):
            st.session_state["section"] = "Chat"; st.rerun()
    with c2:
        if st.button("🧩 Clustering", type="secondary", use_container_width=True):
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
            checked  = st.form_submit_button("Check", type="primary", use_container_width=True)
        with cta2:
            canceled = st.form_submit_button("Cancel", use_container_width=True)

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
            if st.button("Close", use_container_width=True):
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
            st.button("Close", use_container_width=True, on_click=lambda: st.session_state.update(show_check_modal=False))

section = st.session_state["section"]

# ----------- Best Deals (owners) -----------
if section == "Best Deals" and role == "owner":

    score_col = cols["soft"]
    if score_col not in df.columns:
        st.info("soft_buy_score not found—showing a basic top list.")
        tmp = add_brend_and_standardize(df)
        show_cols = [c for c in ["brend","year","selling_price","km_driven","engine_cc","power_bhp"] if c in tmp.columns]
        st.dataframe(tmp.sort_values(tmp.columns[0]).head(10)[show_cols], use_container_width=True, hide_index=True)
        st.stop()

    # --- Class A ---
    class_a = df[df[score_col] >= 0.7].copy()
    st.markdown("### Class A — Premium Deals")
    if class_a.empty:
        st.caption("No vehicles meet this criterion.")
    else:
        top_a = class_a.sort_values(score_col, ascending=False).head(10)
        top_a = add_brend_and_standardize(top_a)
        show_cols = [c for c in ["brend","year","selling_price","km_driven","engine_cc","power_bhp"] if c in top_a.columns]
        st.dataframe(top_a[show_cols], use_container_width=True, hide_index=True)
        st.caption("Top 10 high-performing vehicles with score ≥ 0.7.")

    st.divider()

    # --- Class B ---
    class_b = df[(df[score_col] >= 0.5) & (df[score_col] < 0.7)].copy()
    st.markdown("### Class B — Good Deals")
    if class_b.empty:
        st.caption("No vehicles meet this criterion.")
    else:
        top_b = class_b.sort_values(score_col, ascending=False).head(10)
        top_b = add_brend_and_standardize(top_b)
        show_cols = [c for c in ["brend","year","selling_price","km_driven","engine_cc","power_bhp"] if c in top_b.columns]
        st.dataframe(top_b[show_cols], use_container_width=True, hide_index=True)
        st.caption("Top 10 moderately strong offers with score between 0.5 – 0.7.")


# ----------- Classification (owners) -----------
elif section == "Classification" and role == "owner":

    score_col = cols["soft"]
    if score_col not in df.columns:
        st.info("soft_buy_score not found—cannot classify.")
        st.stop()

    # Train model for optional probability (kept, but we won't display it here)
    model = train_classifier(df, cols)
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
        tmp = add_brend_and_standardize(df_block)
        cols_to_show = [c for c in ["brend","year","selling_price","km_driven","engine_cc","power_bhp"] if c in tmp.columns]
        st.dataframe(tmp[cols_to_show], use_container_width=True, hide_index=True)
        st.caption(caption_text)

    show_block(class_a, "### Class A — Premium Deals", "All vehicles with score ≥ 0.7.")
    st.divider()
    show_block(class_b, "### Class B — Good Deals", "All vehicles within the 0.5–0.7 range.")
    st.divider()
    show_block(class_c, "### Class C — Fair/Poor Deals", "All vehicles below 0.5.")

# ----------- Chat (customers) -----------
elif section == "Chat" and role == "customer":
    st.subheader("💬 Ask about the inventory")
    for r,m in st.session_state["chat"]: st.chat_message(r).markdown(m)
    msg = st.chat_input("Ask anything (e.g., 'Best SUVs under 15k')")
    if msg:
        st.chat_message("user").markdown(msg)
        subset = relevant_rows(df, msg, cols, limit=25)
        ctx = to_context(subset, cols, n=15)
        ans = ask_openai(msg, ctx)
        st.chat_message("assistant").markdown(ans)
        st.session_state["chat"] += [("user", msg), ("assistant", ans)]

# ----------- Clustering (customers) -----------
elif section == "Clustering" and role == "customer":
    st.subheader("🧩 Clustering")
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

    name_map = name_clusters(centers.set_index("cluster").drop(columns=[c for c in ["cluster"] if c in centers.columns], errors="ignore"))
    clustered["cluster_name"] = clustered["cluster"].map(name_map).fillna("Cluster")

    if view == "Table":
        sizes = clustered["cluster"].value_counts().rename_axis("cluster").reset_index(name="count")
        overview = centers.merge(sizes, on="cluster", how="left")
        overview["name"] = overview["cluster"].map(name_map).fillna("Cluster")
        metric_cols = [c for c in [cols["year"], cols["price"], mileage_col] if c in overview.columns]
        overview = overview[["cluster", "name", "count"] + metric_cols].sort_values("cluster")
        st.markdown("**Cluster Overview**")
        st.dataframe(overview.round(1), use_container_width=True, hide_index=True)
    else:
        import networkx as nx
        import plotly.graph_objects as go
        from scipy.spatial.distance import pdist, squareform

        st.markdown("**Cluster Graph**")
        G = nx.Graph()
        numeric_centers = centers.drop(columns=["cluster"], errors="ignore")
        if numeric_centers.shape[0] > 1:
            dists = squareform(pdist(numeric_centers.values))
            thr = float(np.median(dists) * 1.5)
            for i in range(len(dists)):
                for j in range(i + 1, len(dists)):
                    w = float(dists[i, j])
                    if w < thr:
                        G.add_edge(int(i), int(j), weight=w)
        for _, row in centers.iterrows():
            idx = int(row["cluster"]); G.add_node(idx, label=f"Cluster {idx}", size=20)
        pos = nx.spring_layout(G, seed=42, k=0.5)
        edge_x, edge_y = [], []
        for a, b in G.edges():
            x0, y0 = pos[a]; x1, y1 = pos[b]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", line=dict(width=1, color="#888"))
        node_x, node_y, text = [], [], []
        for n in G.nodes():
            x, y = pos[n]; node_x.append(x); node_y.append(y); text.append(f"Cluster {n}")
        node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=text, textposition="top center",
                                marker=dict(showscale=False, color=list(G.nodes()), size=22,
                                            line=dict(width=2, color="DarkSlateGrey")), hoverinfo="text")
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title=dict(text="Cluster Relationships", font=dict(size=18)),
                                         showlegend=False, hovermode="closest",
                                         margin=dict(b=0, l=0, r=0, t=40),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        st.plotly_chart(fig, use_container_width=True)
