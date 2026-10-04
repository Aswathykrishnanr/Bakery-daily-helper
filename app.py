"""Bakery Daily Helper (Streamlit)
A small shop's daily money tracker.
- Gemma (open-weight, via Ollama) reads the owner's plain-English note and
  turns his questions into queries.
- Python (pandas) does ALL the maths, so numbers are always exact.
Run: streamlit run app.py
"""
import json
import os
from datetime import date, timedelta

import ollama
import pandas as pd
import streamlit as st

MODEL = "gemma3:4b"  
PURCHASES_FILE = "purchases.csv"   # date, category, spent  
DAILY_FILE = "daily.csv"           # date, cash, online, kept_for_tomorrow 
PURCHASE_COLS = ["date", "category", "spent"]
DAILY_COLS = ["date", "cash", "online", "kept_for_tomorrow"]
CATEGORIES = [
    "Snacks", "Bakery items", "Drinks", "Milk", "Ice creams",
    "Samosa", "Fried snacks", "Groceries", "Fruits for juice", "Other spendings",
]
_CANON = {c.lower(): c for c in CATEGORIES}


# ---------- helpers ----------
def num(x):
    try:
        v = float(x or 0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v else v


def canonical(name):
    name = str(name or "").strip()
    if name.lower() in ("", "nan", "none"):
        return ""
    key = name.lower()
    if key in _CANON:
        return _CANON[key]
    for k, v in _CANON.items():
        if key in k or k in key:
            return v
    return "Other spendings"


def _read(path, cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    for c in cols:
        if c not in ("date", "category"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def _write(path, df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def load_purchases():
    return _read(PURCHASES_FILE, PURCHASE_COLS)


def load_daily():
    return _read(DAILY_FILE, DAILY_COLS)


def save_day(day, items, cash, online, kept):
    """Saves one day. Saving the same date again REPLACES it (nothing doubles)."""
    ts = pd.Timestamp(day)
    rows = [{"date": ts, "category": canonical(r["Category"]), "spent": num(r["Amount"])}
            for _, r in items.iterrows()
            if canonical(r["Category"]) and num(r["Amount"]) > 0]
    p = load_purchases()
    p = p[p["date"] != ts]
    _write(PURCHASES_FILE, pd.concat([p, pd.DataFrame(rows, columns=PURCHASE_COLS)], ignore_index=True))
    d = load_daily()
    d = d[d["date"] != ts]
    new_d = pd.DataFrame([{"date": ts, "cash": cash, "online": online, "kept_for_tomorrow": kept}])
    _write(DAILY_FILE, pd.concat([d, new_d], ignore_index=True))
    return len(rows)


# ---------- Gemma job 1: read the daily note ----------
PARSE_PROMPT = """You read a shopkeeper's daily note. Return ONLY JSON in this form:
{{"purchases": [{{"category": "...", "amount": 0}}], "cash": 0, "online": 0, "kept_for_tomorrow": 0}}

purchases = money spent buying stock for the shop,Every category with an amount must be its own entry in purchases. Do not skip or merge any.
category must be chosen ONLY from this list: {categories}
cash = total cash received in hand for the WHOLE day (not per category).
online = total received through online payment for the WHOLE day (not per category).
kept_for_tomorrow = money kept aside for the next day.
If something is not mentioned, use 0. Do not do any calculations.

Example note: snacks 3456, milk 800, drinks 1200, cash 6500, online 2300, kept 2000
Example output: {{"purchases": [{{"category": "Snacks", "amount": 3456}}, {{"category": "Milk", "amount": 800}}, {{"category": "Drinks", "amount": 1200}}], "cash": 6500, "online": 2300, "kept_for_tomorrow": 2000}}

Note: {note}
Output:"""


def understand_note(note):
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PARSE_PROMPT.format(note=note, categories=", ".join(CATEGORIES))}],
        format="json",
    )
    d = json.loads(resp["message"]["content"])
    items = [{"Category": canonical(e.get("category")), "Amount": num(e.get("amount"))}
             for e in d.get("purchases", []) if isinstance(e, dict)]
    return (pd.DataFrame(items, columns=["Category", "Amount"]),
            num(d.get("cash")), num(d.get("online")), num(d.get("kept_for_tomorrow")))


# ---------- Gemma job 2: turn a question into a query; Python gets the number ----------
METRICS = ["spent", "cash", "online", "credited", "difference"]
PERIODS = ["today", "yesterday", "last_7_days", "this_month", "last_month", "all_time"]
OPERATIONS = ["total", "highest_category", "lowest_category"]

QUERY_PROMPT = """Turn the shopkeeper's question into a JSON query. Return ONLY JSON with these keys:
operation: one of {operations}. Use highest_category or lowest_category only for "which category..." questions.
metric: one of {metrics}. (spent = money spent buying stock, cash = cash received, online = online payments received,
credited = cash + online, difference = credited minus spent)
category: one of {categories}, or "all" if the question is not about one category.
period: one of {periods}.

Example question: how much did I spend on milk last week?
Example output: {{"operation": "total", "metric": "spent", "category": "Milk", "period": "last_7_days"}}

Question: {q}
Output:"""


def parse_question(q):
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": QUERY_PROMPT.format(
            operations=", ".join(OPERATIONS), metrics=", ".join(METRICS),
            categories=", ".join(CATEGORIES), periods=", ".join(PERIODS), q=q)}],
        format="json",
    )
    d = json.loads(resp["message"]["content"])
    cat = str(d.get("category", "all"))
    return {
        "operation": d.get("operation") if d.get("operation") in OPERATIONS else "total",
        "metric": d.get("metric") if d.get("metric") in METRICS else "spent",
        "category": "all" if cat.lower() == "all" else (canonical(cat) or "all"),
        "period": d.get("period") if d.get("period") in PERIODS else "this_month",
    }


def period_bounds(period):
    t = date.today()
    if period == "today":
        return t, t
    if period == "yesterday":
        y = t - timedelta(days=1)
        return y, y
    if period == "last_7_days":
        return t - timedelta(days=6), t
    if period == "this_month":
        return t.replace(day=1), t
    if period == "last_month":
        end = t.replace(day=1) - timedelta(days=1)
        return end.replace(day=1), end
    return None, None


def in_range(df, start, end):
    if not start:
        return df
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]


def run_query(p, d, q):
    """Exact maths in pandas. Returns a sentence, or None if there is no data."""
    start, end = period_bounds(q["period"])
    p, d = in_range(p, start, end), in_range(d, start, end)
    metric, period = q["metric"], q["period"].replace("_", " ")
    if metric == "spent":
        if q["category"] != "all":
            p = p[p["category"] == q["category"]]
        if p.empty:
            return None
        if q["operation"] == "total":
            scope = q["category"] if q["category"] != "all" else "all categories"
            return f"Total spent ({scope}, {period}): ₹{p['spent'].sum():,.0f}"
        g = p.groupby("category")["spent"].sum()
        if q["operation"] == "highest_category":
            return f"Highest spending category ({period}): {g.idxmax()} (₹{g.max():,.0f})"
        return f"Lowest spending category ({period}): {g.idxmin()} (₹{g.min():,.0f})"
    # cash / online / credited / difference are whole-day totals
    if q["category"] != "all":
        return "Cash and online are recorded for the whole day, not per category. Try the question without a category."
    if q["operation"] != "total":
        return "I can compare categories only for spending (what was bought)."
    if d.empty:
        return None
    cash, online, spent = d["cash"].sum(), d["online"].sum(), p["spent"].sum()
    value = {"cash": cash, "online": online, "credited": cash + online, "difference": cash + online - spent}[metric]
    return f"Total {metric} ({period}): ₹{value:,.0f}"


# ---------- page setup ----------
st.set_page_config(page_title="Bakery Daily Helper", page_icon="🥐", layout="wide")
st.markdown("""
<style>
[data-testid="stMetric"] {background: rgba(128,128,128,0.10); padding: 16px 20px;
  border-radius: 14px; border: 1px solid rgba(128,128,128,0.25);}
</style>""", unsafe_allow_html=True)

st.sidebar.title("🥐 Bakery Helper")
page = st.sidebar.radio("Go to", ["➕ Add today's entry", "📊 Month report", "💬 Ask your notebook", "🛒 What to load next"])
st.sidebar.caption(f"Model: {MODEL}\n\nRuns offline")

# ---------- Add entry ----------
if page.startswith("➕"):
    st.title("Daily Book")
    EMPTY = lambda: pd.DataFrame({"Category": pd.Series(dtype=str), "Amount": pd.Series(dtype=float)})
    if st.session_state.pop("do_reset", False):
        st.session_state["rows"] = EMPTY()
        for k in ("cash_in", "online_in", "kept_in"):
            st.session_state[k] = 0.0
        st.session_state.pop("editor", None)
    if (msg := st.session_state.pop("saved_msg", None)):
        st.success(msg)
    st.session_state.setdefault("rows", EMPTY())
    for k in ("cash_in", "online_in", "kept_in"):
        st.session_state.setdefault(k, 0.0)

    st.write("Just like in your notebook.")
    note = st.text_area(
        "Today's note", height=100,
        placeholder="snacks 3456, drinks 1200, milk 800, cash 9000, online 3000, kept 2000",
    )
    day = st.date_input("Date", value=date.today())

    if st.button("✨ Let Gemma read it", type="primary"):
        if not note.strip():
            st.warning("Type a note first.")
        else:
            with st.spinner("Gemma is reading your note..."):
                try:
                    items, cash, online, kept = understand_note(note)
                    st.session_state["rows"] = items
                    st.session_state["cash_in"], st.session_state["online_in"], st.session_state["kept_in"] = cash, online, kept
                    st.session_state.pop("editor", None)
                except Exception as e:
                    st.error(f"Could not understand the note: {e}\n\nIs Ollama running and is '{MODEL}' installed?")

    st.subheader("Bought for the shop today")
    edited = st.data_editor(
        st.session_state["rows"], key="editor", num_rows="dynamic", use_container_width=True,
        column_config={
            "Category": st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True),
            "Amount": st.column_config.NumberColumn("Amount spent ₹", min_value=0),
        },
    )
    st.subheader("Money for the whole day")
    c1, c2, c3 = st.columns(3)
    cash = c1.number_input("Total cash in hand ₹", min_value=0.0, step=100.0, key="cash_in")
    online = c2.number_input("Total online received ₹", min_value=0.0, step=100.0, key="online_in")
    kept = c3.number_input("Kept for tomorrow ₹", min_value=0.0, step=100.0, key="kept_in")

    if (pd.Timestamp(day) == load_daily()["date"]).any():
        st.info("This date already has an entry. Saving will replace it.")
    if st.button("💾 Save this day"):
        n = save_day(day, edited, cash, online, kept)
        st.session_state["saved_msg"] = f"Saved {day}: {n} purchase row(s), cash ₹{cash:,.0f}, online ₹{online:,.0f}."
        st.session_state["do_reset"] = True
        st.rerun()

# ---------- Month report ----------
elif page.startswith("📊"):
    st.title("Month report")
    p, d = load_purchases(), load_daily()
    if p.empty and d.empty:
        st.info("No entries yet. Add one first.")
    else:
        all_dates = pd.concat([p["date"], d["date"]])
        months = sorted(all_dates.dt.strftime("%Y-%m").unique(), reverse=True)
        month = st.selectbox("Month", months)
        pm = p[p["date"].dt.strftime("%Y-%m") == month]
        dm = d[d["date"].dt.strftime("%Y-%m") == month]
        credited = (dm["cash"] + dm["online"]).sum()
        debited = pm["spent"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Credited (cash + online)", f"₹{credited:,.0f}")
        c2.metric("Debited (spent on stock)", f"₹{debited:,.0f}")
        c3.metric("Difference", f"₹{credited - debited:,.0f}", delta=f"{credited - debited:,.0f}")
        c4, c5 = st.columns(2)
        c4.metric("Received in cash", f"₹{dm['cash'].sum():,.0f}")
        c5.metric("Received online", f"₹{dm['online'].sum():,.0f}")

        st.subheader("Spent by category")
        by_cat = pm.groupby("category")["spent"].sum().sort_values(ascending=False)
        st.bar_chart(by_cat)
        st.subheader("Day by day: credited vs spent")
        per_day = pd.DataFrame({
            "credited": (dm.set_index("date")["cash"] + dm.set_index("date")["online"]),
            "spent": pm.groupby("date")["spent"].sum(),
        }).fillna(0).sort_index()
        st.bar_chart(per_day)
        with st.expander("All days this month"):
            st.dataframe(dm.sort_values("date"), use_container_width=True)
            st.dataframe(pm.sort_values("date"), use_container_width=True)

# ---------- Ask your notebook ----------
elif page.startswith("💬"):
    st.title("Ask your notebook")
    st.caption("Try: How much did I spend on milk last week?")
    history = st.session_state.setdefault("chat", [])
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask about your shop's money...")
    if q:
        history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        p, d = load_purchases(), load_daily()
        query = None
        with st.chat_message("assistant"):
            if p.empty and d.empty:
                reply = "No entries saved yet. Add some first."
            else:
                with st.spinner("Thinking..."):
                    try:
                        query = parse_question(q)
                        result = run_query(p, d, query)
                        if result is None:
                            reply = "I found no entries for that."
                        else:
                            reply = f"**{result}**"
                            try:
                                resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": (
                                    f"Question: {q}\nCalculated answer: {result}\n"
                                    "Say this as one short friendly sentence for a shop owner. "
                                    "Do not change any number.")}])
                                reply += "\n\n" + resp["message"]["content"]
                            except Exception:
                                pass
                    except Exception as e:
                        reply = f"Sorry, I could not understand that: {e}"
            st.markdown(reply)
            if query:
                with st.expander("How I understood your question"):
                    st.json(query)
        history.append({"role": "assistant", "content": reply})

# ---------- What to load next ----------
else:
    st.title("What to load next")
    p, d = load_purchases(), load_daily()
    since = pd.Timestamp(date.today() - timedelta(days=7))
    pr, dr = p[p["date"] >= since], d[d["date"] >= since]
    if pr.empty and dr.empty:
        st.info("No entries in the last 7 days. Add some first.")
    else:
        by_cat = pr.groupby("category")["spent"].sum().sort_values(ascending=False)
        credited = (dr["cash"] + dr["online"]).sum()
        kept = dr.sort_values("date")["kept_for_tomorrow"].iloc[-1] if not dr.empty else 0
        st.subheader("Last 7 days")
        c1, c2, c3 = st.columns(3)
        c1.metric("Credited", f"₹{credited:,.0f}")
        c2.metric("Spent on stock", f"₹{pr['spent'].sum():,.0f}")
        c3.metric("Kept for tomorrow (latest)", f"₹{kept:,.0f}")
        st.bar_chart(by_cat)
        if st.button("🛒 Ask Gemma for suggestions", type="primary"):
            summary = "\n".join(f"- {c}: spent ₹{v:,.0f}" for c, v in by_cat.items())
            prompt = (
                "You advise the owner of a small local shop. These are the last 7 days' totals. "
                "The numbers are already calculated, do not recalculate them.\n\n"
                f"Spending per category:\n{summary}\n\n"
                f"Total credited (cash + online): ₹{credited:,.0f}\n"
                f"Total spent on stock: ₹{pr['spent'].sum():,.0f}\n"
                f"Money kept for tomorrow in the latest entry: ₹{kept:,.0f}\n\n"
                "Give 3 short, simple suggestions on which categories to load more or less of next, "
                "and whether the money kept looks enough. Use only these numbers."
            )
            with st.spinner("Gemma is thinking..."):
                try:
                    resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
                    st.markdown(resp["message"]["content"])
                except Exception as e:
                    st.error(f"Could not reach the model: {e}")