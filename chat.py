# app_chat.py — Minimal Streamlit chatbot over Movelit dataset (no roles, no other sections)

import os, re, json
import numpy as np
import pandas as pd
import streamlit as st

# ---------- OpenAI client ----------
OPENAI_MODEL = "gpt-5-nano-2025-08-07"   # keep your chosen model id
OPENAI_TEMPERATURE = 0.2
SYSTEM_PROMPT = (
    "You are DealBot, a car-dealer assistant.\n"
    "You will receive JSON with {schema, rows, question}.\n"
    "- Use ONLY the provided 'rows'. If the answer isn't implied by them, reply exactly: 'Not in this dataset.'\n"
    "- When referring to price, ALWAYS use 'price_eur' (numeric) or 'price_eur_str' (already formatted). "
    "NEVER use 'selling_price' or show '₹'.\n"
    "- For the car label, use the 'car' column EXACTLY as given; do not prepend/duplicate brand/model.\n"
    "- For 'best'/'top' ranking: sort by soft_buy_score ↓, then price_eur ↑, then year ↓, then km_driven ↑ when present.\n"
    "- If the user asks for N items, return up to N.\n"
    "- Be concise (≤ 5 short lines) and cite car name/year/price from the rows."
)
DATA_PATH = "exports/final_enriched_full.csv"

try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None

st.set_page_config(page_title="Dealer Deals — Chat", layout="wide")

# ---------- Data helpers ----------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at **{DATA_PATH}**")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df

def pick_cols(df):
    f = lambda cands: next((c for c in cands if c in df.columns), None)
    return dict(
        price=f(["price_eur", "selling_price", "price"]),
        year=f(["year"]),
        make=f(["make","manufacturer"]), model=f(["model"]),
        mileage=f(["mileage_val", "mileage"]),
        km=f(["km_driven","kilometers","km"]),
        power=f(["power_bhp","power"]), fuel=f(["fuel"]),
        trans=f(["transmission","gearbox"]), body=f(["body_type","type"]),
        soft=f(["soft_buy_score","softbuy_score","soft_score"])
    )

def _fmt_eur(x):
    try:
        return "€{:,.2f}".format(float(x)).replace(",", " ")
    except Exception:
        return "€—"

def ensure_car_name(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def _clean(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip()
        placeholders = {"nan", "none", "null", "na", "n/a", "-", "—"}
        return s.apply(lambda x: "" if x.lower() in placeholders else x)
    for c in ["car", "name", "brand", "model"]:
        if c in out.columns:
            out[c] = _clean(out[c])
    base = out["car"] if "car" in out.columns else pd.Series("", index=out.index)
    if "name" in out.columns:
        base = base.where(~base.eq(""), out["name"])
    if {"brand","model"}.issubset(out.columns):
        bm = (out["brand"].fillna("") + " " + out["model"].fillna("")).str.strip()
        base = base.where(~base.eq(""), bm)
    if "brand" in out.columns:
        base = base.where(~base.eq(""), out["brand"])
    out["car"] = base.replace("", "—")
    return out

# ---------- Chat retrieval helpers ----------
STOPWORDS = {
    "what","whats","which","is","are","the","a","an","for","show","list","top","best","good",
    "with","and","or","of","to","please","give","find","me","under","over","between","vs",
    "any","then","buy"
}

def relevant_rows(df, query, cols, limit=25):
    if df.empty or not query or not query.strip(): 
        return df.head(limit)
    q = query.lower()
    text_cols = [x for x in [cols.get("make"), cols["model"], cols["fuel"], cols["trans"], cols["body"]] if x in df.columns]
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

def to_context(df, n=12):
    keep = [c for c in ["car","year","price_eur","price_eur_str","km_driven","soft_buy_score","body_norm"] if c in df.columns]
    small = df[keep].head(n).reset_index(drop=True)
    small.insert(0, "id", small.index + 1)
    alias = {"id":"i","car":"c","year":"y","price_eur":"p","price_eur_str":"ps",
             "km_driven":"km","soft_buy_score":"s","body_norm":"b"}
    return [{alias[k]: v for k,v in row.items()} for row in small.to_dict(orient="records")]

def ask_llm_with_schema(schema, rows, question):
    if client is None:
        return "Chat unavailable (OpenAI client not initialized)."
    payload = {"schema": schema, "rows": rows, "question": question}
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)}
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# ---------- Body normalization (optional but useful) ----------
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
    def _norm(x):
        for k, v in BODY_MAP.items():
            if k in x: return v
        return None
    df["body_norm"] = s.apply(_norm)
    return df, "body_norm"

# ---------- Schema hint for the LLM (compact) ----------
SCHEMA_HINT = {
    "ranking": [{"col":"s","dir":"desc"}, {"col":"p","dir":"asc"}, {"col":"y","dir":"desc"}, {"col":"km","dir":"asc"}],
    "synonyms": {
        "SUV":["suv","crossover"], "MPV":["mpv","muv"], "Pickup":["pickup","truck"],
        "Wagon":["wagon","estate"], "Convertible":["convertible","cabrio","cabriolet","roadster"],
        "Hatchback":["hatchback","hatch"]
    }
}

# ---------- App ----------
st.title("Dealer Deals — Chat")

df = load_data()
if df.empty: st.stop()

# price preparations
if "price_eur" not in df.columns and "selling_price" in df.columns:
    df["price_eur"] = pd.to_numeric(df["selling_price"], errors="coerce") / 100.0
df["price_eur_str"] = df["price_eur"].apply(_fmt_eur) if "price_eur" in df.columns else "€—"

# body normalization + car label
df, _ = add_body_norm(df)
df = ensure_car_name(df)
cols = pick_cols(df)

# session state
st.session_state.setdefault("chat", [])

# toolbar
col_a, col_b = st.columns([0.8, 0.2])
with col_b:
    if st.button("Reset chat", use_container_width=True):
        st.session_state["chat"] = []
        st.rerun()

# render history
for who, text in st.session_state["chat"]:
    with st.chat_message("assistant" if who == "assistant" else "user"):
        st.markdown(text)

# chat input & response
user_msg = st.chat_input("Ask anything about the cars in this dataset…")
if user_msg:
    st.session_state["chat"].append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("DealBot is thinking..."):
            try:
                # relevance slice
                rel_df = relevant_rows(df, user_msg, cols, limit=80)
                # context rows (N controlled by question if present)
                m = re.search(r"\b(top|show|list)\s+(\d+)", user_msg.lower())
                N = min(int(m.group(2)) if m else 10, 15)
                rows = to_context(rel_df, n=N)
                ans_text = ask_llm_with_schema(SCHEMA_HINT, rows, user_msg)
            except Exception as e:
                ans_text = f"Sorry — something went wrong: {e}"

        st.markdown(ans_text)
        with st.expander("Data used for this answer", expanded=False):
            try:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except Exception:
                st.caption("No rows to display.")
    st.session_state["chat"].append(("assistant", ans_text))
