import os
from datetime import date, timedelta
import calendar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

from database import (
    initialize_database,
    create_user,
    authenticate_user,
    assign_legacy_data_to_user,
    add_transaction,
    get_transactions,
    delete_transaction,
    get_budgets,
    save_budget,
    save_budgets
)
from versapay_llm import (
    ask_versapay,
    check_ollama_status,
    get_financial_context,
)

st.set_page_config(
    page_title="VersaPay AI | Personal Finance",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_database()

CATEGORIES = [
    "Food", "Transport", "Shopping", "Bills", "Entertainment",
    "Education", "Health", "Travel", "Home", "Subscriptions", "Other"
]
DEFAULT_BUDGETS = {
    "Food": 5000, "Transport": 2500, "Shopping": 4000, "Bills": 3000,
    "Entertainment": 2000, "Education": 2500, "Health": 1500,
    "Travel": 3000, "Home": 3000, "Subscriptions": 1000, "Other": 2000
}
INCOME_CATEGORIES = ["Salary", "Allowance", "Freelance", "Investment", "Gift", "Other"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#0b1020; --panel:#121a2d; --line:rgba(148,163,184,.16); --muted:#9ba8c4; --text:#f4f7ff; --purple:#8b7cff; --cyan:#59d7e8; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(circle at 82% 0%, rgba(93,72,190,.16), transparent 30%), #0b1020; color:var(--text); }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.6rem; padding-bottom:2.5rem; max-width:1500px; }
section[data-testid="stSidebar"] { background:#0e1527; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] * { color:#eef2ff; }
.brand { font-family:'Space Grotesk',sans-serif; font-size:1.65rem; font-weight:700; letter-spacing:-.8px; }
.brand span { color:#9c8cff; }
.kicker { color:#a5b4fc; text-transform:uppercase; letter-spacing:2px; font-size:.72rem; font-weight:700; }
.hero { background:linear-gradient(120deg,rgba(91,74,184,.35),rgba(23,42,73,.84)), #151d32; border:1px solid rgba(159,145,255,.24); padding:30px 32px; border-radius:24px; margin-bottom:20px; position:relative; overflow:hidden; }
.hero h1 { font-family:'Space Grotesk',sans-serif; font-size:2.35rem; letter-spacing:-1.5px; margin:.3rem 0 .6rem; color:#fff; }
.hero p { color:#c3cce1; max-width:700px; margin:0; line-height:1.65; }
.hero-chip { display:inline-block; padding:7px 12px; background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.12); border-radius:999px; font-size:.75rem; color:#e6e4ff; }
.metric-card { background:linear-gradient(145deg,#151e33,#11192b); border:1px solid var(--line); border-radius:18px; padding:19px 20px; min-height:126px; }
.metric-label { color:#9ba8c4; font-size:.83rem; margin-bottom:12px; }
.metric-value { font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; color:#f8faff; letter-spacing:-.7px; }
.metric-foot { color:#9ba8c4; font-size:.75rem; margin-top:8px; }
.section-title { font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700; margin:1.2rem 0 .25rem; color:#f5f7ff; }
.section-subtitle { color:#9ba8c4; font-size:.86rem; margin-bottom:1rem; }
.insight { background:#121c31; border:1px solid var(--line); border-left:3px solid #8b7cff; border-radius:13px; padding:15px 17px; margin:.55rem 0; }
.insight.warning { border-left-color:#f6b85d; }
.insight.good { border-left-color:#45d6a0; }
.insight b { color:#f8faff; }
.insight p { color:#aebbd3; margin:.35rem 0 0; font-size:.88rem; line-height:1.55; }
.small-pill { display:inline-block; background:rgba(89,215,232,.10); border:1px solid rgba(89,215,232,.2); color:#83e7f0; border-radius:999px; padding:4px 9px; font-size:.7rem; }
div[data-testid="stMetric"] { background:#131c30; border:1px solid var(--line); padding:15px 17px; border-radius:16px; }
div[data-testid="stMetricLabel"] { color:#aab6d0; }
div[data-testid="stMetricValue"] { color:#fff; }
.stButton button, .stDownloadButton button { border-radius:11px; font-weight:700; min-height:2.6rem; }
.stTextInput input, .stNumberInput input, .stDateInput input, .stSelectbox div[data-baseweb="select"] { border-radius:10px; }
div[data-testid="stTabs"] button { font-weight:600; }
hr { border-color:var(--line); }
footer { visibility:hidden; }
/* Fix chatbot input visibility */
.stChatInput textarea {
    color: #111827 !important;
    background-color: #ffffff !important;
    caret-color: #7c3aed !important;
    -webkit-text-fill-color: #111827 !important;
    font-size: 16px !important;
}

.stChatInput textarea::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
}

div[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border: 2px solid #8b7cff !important;
    border-radius: 14px !important;
}

div[data-testid="stChatInput"] textarea {
    color: #111827 !important;
    caret-color: #7c3aed !important;
    -webkit-text-fill-color: #111827 !important;
}

div[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(139, 124, 255, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)


def rupees(value):
    return f"₹{float(value):,.0f}"


def load_data(user_id):
    df = get_transactions(user_id)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["category"] = df["category"].fillna("Other")
    df["description"] = df["description"].fillna("")
    return df.dropna(subset=["date"])


def finances(df):
    if df.empty:
        return 0.0, 0.0, 0.0
    income = df.loc[df.transaction_type == "Income", "amount"].sum()
    expense = df.loc[df.transaction_type == "Expense", "amount"].sum()
    return float(income), float(expense), float(income - expense)


def budget_map(user_id):
    stored = get_budgets(user_id)
    result = dict(DEFAULT_BUDGETS)
    if not stored.empty:
        for row in stored.itertuples(index=False):
            result[row.category] = float(row.monthly_limit)
    return result


def month_expenses(df, month_start, month_end):
    if df.empty:
        return df.copy()
    return df[
        (df["transaction_type"] == "Expense")
        & (df["date"].dt.date >= month_start)
        & (df["date"].dt.date <= month_end)
    ]


def month_bounds(year, month):
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    return first, last


def get_forecast(df):
    """Forecast next month's spending from monthly totals; requires at least 3 months."""
    if df.empty:
        return None, "Add transactions to start building a forecast."
    exp = df[df.transaction_type == "Expense"].copy()
    if exp.empty:
        return None, "No expense transactions have been recorded yet."
    exp["month"] = exp["date"].dt.to_period("M").dt.to_timestamp()
    monthly = exp.groupby("month")["amount"].sum().sort_index()
    # Include zero-spend months between the first and latest recorded month.
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_index, fill_value=0)
    if len(monthly) < 3:
        return None, "Record expenses across at least 3 different months to unlock the forecast."
    x = list(range(len(monthly)))
    model = LinearRegression()
    model.fit(pd.DataFrame({"month_index": x}), monthly.values)
    prediction = max(0.0, float(model.predict(pd.DataFrame({"month_index": [len(monthly)]}))[0]))
    return {"prediction": prediction, "history": monthly, "model": model}, None


def personalized_insights(df, budgets):
    insights = []
    if df.empty:
        return [("Start here", "Add a few income and expense transactions. VersaPay will use your own history to identify patterns.", "info")]
    expenses = df[df.transaction_type == "Expense"].copy()
    if expenses.empty:
        return [("Your spending profile is empty", "Add expense transactions to get category-level recommendations and forecasts.", "info")]

    today = date.today()
    start, end = month_bounds(today.year, today.month)
    current = month_expenses(df, start, end)
    total_current = float(current.amount.sum()) if not current.empty else 0.0

    if not current.empty:
        by_cat = current.groupby("category")["amount"].sum().sort_values(ascending=False)
        top_cat, top_amt = by_cat.index[0], float(by_cat.iloc[0])
        share = top_amt / total_current * 100 if total_current else 0
        insights.append(("Largest spending category",
                         f"{top_cat} is your biggest expense this month at {rupees(top_amt)} ({share:.0f}% of recorded monthly spending). Review repeat purchases in this category first.",
                         "warning" if share >= 35 else "info"))
        for cat, spent in by_cat.items():
            limit = budgets.get(cat, DEFAULT_BUDGETS.get(cat, 2000))
            if limit > 0 and spent >= limit:
                insights.append((f"{cat} budget exceeded",
                                 f"You have recorded {rupees(spent)} against a {rupees(limit)} monthly budget. You are {rupees(spent-limit)} over the limit; consider pausing non-essential purchases in this category.",
                                 "warning"))
            elif limit > 0 and spent >= limit * .8:
                insights.append((f"{cat} budget nearly used",
                                 f"{rupees(spent)} of your {rupees(limit)} {cat} budget is used ({spent/limit:.0%}). About {rupees(limit-spent)} remains for this month.",
                                 "warning"))

    # Compare current-month daily run rate with last month's actual spend.
    prev_month_last = start - timedelta(days=1)
    prev_start = date(prev_month_last.year, prev_month_last.month, 1)
    previous = month_expenses(df, prev_start, prev_month_last)
    if not current.empty and not previous.empty:
        days_elapsed = max(1, min(today.day, end.day))
        run_rate = total_current / days_elapsed * end.day
        previous_total = float(previous.amount.sum())
        if run_rate > previous_total * 1.1:
            insights.append(("Spending pace is rising",
                             f"At the current daily pace, this month could reach about {rupees(run_rate)}, versus {rupees(previous_total)} last month. This is a pace estimate, not a guarantee.",
                             "warning"))
        elif run_rate < previous_total * .9:
            insights.append(("Spending pace is lower",
                             f"Your current pace projects about {rupees(run_rate)} this month, compared with {rupees(previous_total)} last month. Keep monitoring through month-end.",
                             "good"))

    if len(expenses) >= 6:
        recent = expenses.sort_values("date").tail(10)
        if len(recent) >= 6:
            frequent = recent["category"].value_counts()
            if not frequent.empty:
                cat = frequent.index[0]
                insights.append(("Repeated category pattern",
                                 f"{cat} appears most often among your latest {len(recent)} expense records. Small repeated purchases can add up, so check whether each one was planned.",
                                 "info"))

    if not insights:
        insights.append(("Good start", "No major budget warnings were detected from the transactions recorded so far. Keep logging expenses for more useful insights.", "good"))
    return insights[:8]


def answer_local_question(question, df, budgets):
    """Deterministic keyword-based fallback used only when Ollama is completely unavailable."""
    income, expense, balance = finances(df)
    today = date.today()
    start, end = month_bounds(today.year, today.month)
    current = month_expenses(df, start, end)
    current_total = float(current.amount.sum()) if not current.empty else 0.0
    q = question.lower()

    if df.empty:
        return "I need some transaction history first. Add a few income and expense records, then ask me about your spending."
    if any(w in q for w in ["highest", "biggest", "most spending", "top category"]):
        if current.empty:
            return "There are no expense records for the current month yet."
        totals = current.groupby("category").amount.sum().sort_values(ascending=False)
        return f"Your highest-spending category this month is {totals.index[0]} at {rupees(totals.iloc[0])}. Total recorded monthly expenses are {rupees(current_total)}."
    if "budget" in q or "limit" in q:
        warnings = []
        if not current.empty:
            for cat, amount in current.groupby("category").amount.sum().items():
                limit = budgets.get(cat, DEFAULT_BUDGETS.get(cat, 2000))
                if limit and amount >= .8 * limit:
                    warnings.append(f"{cat}: {rupees(amount)} spent of {rupees(limit)}")
        return "Budget check: " + ("; ".join(warnings) if warnings else "no categories have reached 80% of their current budget based on recorded transactions.")
    if any(w in q for w in ["save", "reduce", "cut", "advice", "suggest"]):
        if current.empty:
            return "Start by recording this month's expenses. Then I can identify the categories where a reduction may have the largest impact."
        totals = current.groupby("category").amount.sum().sort_values(ascending=False)
        top = totals.index[0]
        return f"Start by reviewing {top}, your largest category at {rupees(totals.iloc[0])} this month. Set a realistic cap, separate needs from wants, and compare spending weekly. Avoid cutting essential costs just to meet a target."
    if any(w in q for w in ["forecast", "next month", "predict"]):
        result, error = get_forecast(df)
        if error:
            return error
        return f"The simple trend model estimates next month's expenses at about {rupees(result['prediction'])}. It is based on your recorded monthly totals and may be unreliable with limited or unusual data."
    if any(w in q for w in ["balance", "net", "left"]):
        return f"Based on recorded transactions, income is {rupees(income)}, expenses are {rupees(expense)}, and net balance is {rupees(balance)}. This is only as complete as the transactions you've entered."
    if any(w in q for w in ["expense", "spent", "spending", "this month"]):
        return f"Your recorded expenses this month total {rupees(current_total)}. All-time recorded expenses are {rupees(expense)}."
    return f"I can help with your recorded balance ({rupees(balance)}), monthly spending ({rupees(current_total)}), category budgets, saving suggestions, and spending forecasts. Try asking: 'Where am I spending the most?' or 'How can I save money?'"


@st.cache_data(ttl=30, show_spinner=False)
def cached_ollama_status() -> tuple[bool, bool, str]:
    """Cache the Ollama health-check result for 30 seconds to avoid hammering localhost."""
    return check_ollama_status()


def metric_card(label, value, foot, icon):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{icon} &nbsp; {label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True
    )

# =========================
# VERSAPAY AI AUTHENTICATION
# =========================

if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None

if st.session_state.authenticated_user is None:
    st.markdown("""
    <div style="max-width:650px;margin:60px auto 20px auto;padding:20px;text-align:center;">
        <div style="font-size:50px;">✦</div>
        <h1 style="font-size:42px;">VersaPay <span style="color:#8b7cff;">AI</span></h1>
        <p style="color:#9ba8c4;font-size:17px;">Your personal finance copilot</p>
        <p style="color:#9ba8c4;">Sign in or create an account to continue.</p>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.5, 1])
    with center:
        login_tab, register_tab = st.tabs(["Sign In", "Create Account"])

        with login_tab:
            with st.form("login_form"):
                login_identifier = st.text_input("Username or email", placeholder="Enter username or email")
                login_password = st.text_input("Password", type="password", placeholder="Enter your password")
                login_button = st.form_submit_button("Sign In", use_container_width=True, type="primary")
            if login_button:
                user = authenticate_user(login_identifier, login_password)
                if user:
                    st.session_state.authenticated_user = user
                    st.session_state.chat_history = []
                    st.rerun()
                else:
                    st.error("Incorrect username/email or password.")

        with register_tab:
            with st.form("register_form", clear_on_submit=True):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_email = st.text_input("Email", placeholder="you@example.com")
                new_password = st.text_input("Password", type="password", placeholder="At least 8 characters")
                confirm_password = st.text_input("Confirm password", type="password")
                register_button = st.form_submit_button("Create Account", use_container_width=True, type="primary")
            if register_button:
                if new_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    success, message = create_user(new_username, new_email, new_password)
                    if success:
                        st.success("Account created. You can now sign in.")
                    else:
                        st.error(message)
    st.stop()

current_user = st.session_state.authenticated_user
user_id = int(current_user["id"])
df = load_data(user_id)
income, expense, balance = finances(df)
budgets = budget_map(user_id)

with st.sidebar:
    st.markdown('<div class="brand">VersaPay<span> AI</span> ✦</div>', unsafe_allow_html=True)
    st.caption("YOUR PERSONAL FINANCE COPILOT")
    st.markdown(f"Signed in as **{current_user.get('username', 'User')}**")
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Overview", "Transactions", "Budget Studio", "AI Insights", "AI Chat"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**Recorded net balance**")
    st.markdown(f"<div style='font-size:1.8rem;font-weight:700;color:#9d91ff'>{rupees(balance)}</div>", unsafe_allow_html=True)
    st.caption(f"{len(df)} transactions in your account" if not df.empty else "No transactions yet")
    st.markdown("---")
    # ── Ollama AI Status indicator ──
    _svc_ok, _mdl_ok, _status_msg = cached_ollama_status()
    if _svc_ok and _mdl_ok:
        st.markdown("<span style='color:#45d6a0;font-size:.82rem;font-weight:600'>🟢 AI ENGINE ONLINE</span>", unsafe_allow_html=True)
    elif _svc_ok and not _mdl_ok:
        st.markdown("<span style='color:#f6b85d;font-size:.82rem;font-weight:600'>🟡 AI — MODEL MISSING</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#f87171;font-size:.82rem;font-weight:600'>🔴 AI ENGINE OFFLINE</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<span class="small-pill">● USER-SPECIFIC DATA</span>', unsafe_allow_html=True)
    st.caption("Your transactions and budgets are separated by account.")
    st.markdown("---")
    if st.button("Log Out", use_container_width=True):
        st.session_state.authenticated_user = None
        st.session_state.chat_history = []
        st.rerun()

if page == "Overview":
    st.markdown(f"""
    <div class="hero">
      <span class="hero-chip">✦ SMART MONEY, CLEARER DECISIONS</span>
      <h1>Your money, in better focus.</h1>
      <p>Track your cash flow, understand spending patterns, and turn your budget into a plan you can actually follow.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("TOTAL INCOME", rupees(income), "All recorded income", "↗")
    with c2: metric_card("TOTAL EXPENSES", rupees(expense), "All recorded expenses", "↘")
    with c3: metric_card("NET BALANCE", rupees(balance), "Income minus expenses", "◈")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="section-title">Where your money goes</div><div class="section-subtitle">Expense split by category, all recorded dates</div>', unsafe_allow_html=True)
        exp_df = df[df.transaction_type == "Expense"] if not df.empty else pd.DataFrame()
        if not exp_df.empty:
            cat_df = exp_df.groupby("category", as_index=False).amount.sum().sort_values("amount", ascending=False)
            fig = px.pie(cat_df, values="amount", names="category", hole=.68)
            fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="#0b1020", width=3)))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e9edff",
                              legend=dict(font=dict(color="#c4cce0"), orientation="h", y=-.14),
                              margin=dict(t=10,b=30,l=5,r=5), height=330)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Your spending chart will appear here after you add expense transactions.")
    with right:
        st.markdown('<div class="section-title">Cash flow over time</div><div class="section-subtitle">Monthly income vs expenses</div>', unsafe_allow_html=True)
        if not df.empty:
            chart_df = df.copy()
            chart_df["month"] = chart_df["date"].dt.to_period("M").astype(str)
            monthly = chart_df.groupby(["month", "transaction_type"], as_index=False).amount.sum()
            fig = px.bar(monthly, x="month", y="amount", color="transaction_type", barmode="group",
                         color_discrete_map={"Income":"#45d6a0","Expense":"#8b7cff"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#c4cce0", xaxis_title="", yaxis_title="Amount (₹)",
                              legend_title="", margin=dict(t=10,b=10,l=0,r=0), height=330)
            fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
            fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add transactions to see monthly cash flow.")

    st.markdown('<div class="section-title">Your money check-in</div><div class="section-subtitle">A quick summary from your saved records</div>', unsafe_allow_html=True)
    today = date.today()
    start, end = month_bounds(today.year, today.month)
    current = month_expenses(df, start, end)
    current_total = float(current.amount.sum()) if not current.empty else 0
    budget_total = sum(budgets.values())
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f'<div class="insight"><b>This month</b><p>{rupees(current_total)} in recorded expenses so far.</p></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f'<div class="insight"><b>Configured monthly budgets</b><p>Combined category limits: {rupees(budget_total)}. These are planning limits, not money held in an account.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recent transactions</div>', unsafe_allow_html=True)
    if not df.empty:
        recent = df.head(6).copy()
        recent["date"] = recent["date"].dt.strftime("%d %b %Y")
        recent["amount"] = recent.apply(lambda r: ("+" if r.transaction_type == "Income" else "−") + rupees(r.amount), axis=1)
        st.dataframe(recent[["date","transaction_type","category","description","amount"]], use_container_width=True, hide_index=True)
    else:
        st.info("No transactions recorded yet. Open Transactions to add your first entry.")


elif page == "Transactions":
    st.markdown('<div class="kicker">YOUR ACTIVITY</div><h1 style="font-family:Space Grotesk">Transactions</h1><p style="color:#9ba8c4">Record money in and money out. Changes are saved to SQLite.</p>', unsafe_allow_html=True)
    add_tab, history_tab = st.tabs(["＋ Add transaction", "▤ Transaction history"])
    with add_tab:
        with st.form("add_transaction_form", clear_on_submit=True):
            a, b = st.columns(2)
            with a:
                transaction_type = st.selectbox("Transaction type", ["Expense", "Income"])
                txn_date = st.date_input("Date", value=date.today())
                amount = st.number_input("Amount (₹)", min_value=1.0, value=100.0, step=50.0)
            with b:
                category = st.selectbox("Category", CATEGORIES if transaction_type == "Expense" else INCOME_CATEGORIES)
                description = st.text_input("Description", placeholder="e.g. lunch, monthly allowance")
            submitted = st.form_submit_button("Save transaction", use_container_width=True, type="primary")
            if submitted:
                try:
                    add_transaction(user_id, txn_date, transaction_type, category, description.strip(), amount)
                    st.success("Transaction saved successfully.")
                    st.rerun()
                except Exception:
                    st.error("Could not save the transaction. Check the values and try again.")
    with history_tab:
        if df.empty:
            st.info("Your transaction history is empty.")
        else:
            filt1, filt2 = st.columns(2)
            with filt1:
                type_filter = st.selectbox("Filter type", ["All", "Income", "Expense"])
            with filt2:
                category_filter = st.selectbox("Filter category", ["All"] + sorted(df.category.unique().tolist()))
            filtered = df.copy()
            if type_filter != "All": filtered = filtered[filtered.transaction_type == type_filter]
            if category_filter != "All": filtered = filtered[filtered.category == category_filter]
            show = filtered.copy()
            show["date"] = show["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(show[["id","date","transaction_type","category","description","amount"]], use_container_width=True, hide_index=True)
            st.download_button("Export filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "versapay_transactions.csv", "text/csv")
            ids = filtered["id"].astype(int).tolist()
            if ids:
                selected_id = st.selectbox("Select transaction ID to delete", ids)
                confirm = st.checkbox("I confirm I want to permanently delete this transaction")
                if st.button("Delete selected transaction", disabled=not confirm):
                    delete_transaction(selected_id, user_id)
                    st.success(f"Transaction #{selected_id} deleted.")
                    st.rerun()


elif page == "Budget Studio":
    st.markdown('<div class="kicker">PLAN WITH PURPOSE</div><h1 style="font-family:Space Grotesk">Budget Studio</h1><p style="color:#9ba8c4">Set monthly limits by category. Your limits are saved between app restarts.</p>', unsafe_allow_html=True)
    today = date.today()
    start, end = month_bounds(today.year, today.month)
    current = month_expenses(df, start, end)
    spent_map = current.groupby("category").amount.sum().to_dict() if not current.empty else {}
    st.markdown('<div class="section-title">Monthly category limits</div>', unsafe_allow_html=True)
    updated = {}
    cols = st.columns(2)
    for i, cat in enumerate(CATEGORIES):
        with cols[i % 2]:
            updated[cat] = st.number_input(
                f"{cat} budget (₹)", min_value=0.0, value=float(budgets.get(cat, DEFAULT_BUDGETS.get(cat, 2000))),
                step=250.0, key=f"budget_{cat}"
            )
            spent = float(spent_map.get(cat, 0))
            limit = updated[cat]
            if limit > 0:
                progress = min(spent / limit, 1.0)
                st.progress(progress, text=f"{rupees(spent)} spent / {rupees(limit)} ({spent/limit:.0%})")
                if spent > limit:
                    st.caption(f"Over budget by {rupees(spent-limit)}")
            else:
                st.caption(f"{rupees(spent)} spent · no limit configured")
    if st.button("Save all budgets", type="primary", use_container_width=True):
        save_budgets(user_id, updated)
        st.success("Your budgets are saved.")
        st.rerun()
    st.caption("Budget limits are stored locally in versapay.db. They do not move or reserve real money.")


elif page == "AI Insights":
    st.markdown('<div class="kicker">PERSONALIZED ANALYTICS</div><h1 style="font-family:Space Grotesk">AI Insights</h1><p style="color:#9ba8c4">Pattern detection, budget alerts and a trend-based spending forecast from your recorded data.</p>', unsafe_allow_html=True)
    if df.empty:
        st.info("Add transactions first. Insights become more useful as you build a history.")
    for title, body, kind in personalized_insights(df, budgets):
        css_kind = "warning" if kind == "warning" else ("good" if kind == "good" else "")
        st.markdown(f'<div class="insight {css_kind}"><b>✦ {title}</b><p>{body}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Spending forecast</div><div class="section-subtitle">A simple linear-regression trend estimate, not a guaranteed prediction</div>', unsafe_allow_html=True)
    forecast, error = get_forecast(df)
    if error:
        st.info(error)
    else:
        fc1, fc2 = st.columns([1, 1.5])
        with fc1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">ESTIMATED NEXT-MONTH EXPENSES</div><div class="metric-value">{rupees(forecast["prediction"])}</div><div class="metric-foot">Linear regression on monthly totals</div></div>', unsafe_allow_html=True)
        with fc2:
            hist = forecast["history"]
            fdf = hist.rename_axis("month").reset_index(name="expenses")
            fdf["month"] = fdf["month"].dt.strftime("%b %Y")
            fig = px.line(fdf, x="month", y="expenses", markers=True)
            fig.add_trace(go.Scatter(x=[fdf["month"].iloc[-1], "Next month"], y=[float(hist.iloc[-1]), forecast["prediction"]], mode="lines+markers", name="Forecast", line=dict(dash="dash", color="#8b7cff")))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c4cce0",
                              showlegend=False, xaxis_title="", yaxis_title="Expenses (₹)", margin=dict(t=10,b=10,l=0,r=0), height=260)
            fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
            fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Method note: the forecast uses monthly expense totals and linear regression. Three or more months are required. Seasonal changes, missing entries and one-off purchases can make it inaccurate.")


elif page == "AI Chat":
    # ── Header ──
    st.markdown(
        '<div class="kicker">ASK YOUR MONEY COPILOT</div>'
        '<h1 style="font-family:Space Grotesk">VersaPay Chat</h1>'
        '<p style="color:#9ba8c4">Your personal finance assistant powered by Ollama running locally on your device. '
        'All data stays on your machine — nothing is sent to external servers.</p>',
        unsafe_allow_html=True
    )

    # ── Ollama status banner ──
    ollama_svc, ollama_mdl, ollama_msg = cached_ollama_status()
    if ollama_svc and ollama_mdl:
        st.success(ollama_msg, icon="🤖")
    elif ollama_svc and not ollama_mdl:
        st.warning(ollama_msg)
    else:
        st.error(
            "🔴 **Ollama is not running.** The AI chatbot will use the local keyword assistant as a fallback.\n\n"
            "To start Ollama, open a new PowerShell window and run: `ollama serve`\n\n"
            + ollama_msg,
        )

    # ── Session state initialisation ──
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_suggestions" not in st.session_state:
        st.session_state.show_suggestions = True

    # ── Suggested starter questions (shown until the user sends their first message) ──
    SUGGESTED_QUESTIONS = [
        "📊 Analyse my spending this month",
        "💡 How can I save more money?",
        "⚠️ Am I exceeding any of my budgets?",
        "🛍️ Can I afford a purchase of ₹5,000?",
        "📅 Help me plan next month's budget",
    ]

    if st.session_state.show_suggestions and not st.session_state.chat_history:
        st.markdown(
            "<div style='margin:8px 0 4px;color:#a5b4fc;font-size:.82rem;font-weight:600;letter-spacing:.06em'>'"
            "SUGGESTED QUESTIONS</div>",
            unsafe_allow_html=True
        )
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
            with cols[i]:
                if st.button(
                    suggestion,
                    key=f"suggest_{i}",
                    use_container_width=True,
                ):
                    st.session_state._pending_suggestion = suggestion
                    st.session_state.show_suggestions = False
                    st.rerun()

    # ── Render existing chat history ──
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("source"):
                st.caption(message["source"])

    # ── Handle a suggestion that was clicked (inject as user message) ──
    if "_pending_suggestion" in st.session_state:
        injected = st.session_state.pop("_pending_suggestion")
        # Strip leading emoji for cleaner display
        clean_suggestion = injected.split(" ", 1)[-1] if injected[0] in "📊💡⚠️🛍️📅" else injected
        st.session_state.chat_history.append({"role": "user", "content": clean_suggestion})
        with st.chat_message("user"):
            st.markdown(clean_suggestion)
        # Generate response immediately
        fin_ctx = get_financial_context(df, budgets)
        with st.chat_message("assistant"):
            with st.spinner("VersaPay AI is thinking..."):
                reply, source_label = ask_versapay(
                    clean_suggestion,
                    st.session_state.chat_history[:-1],   # history before this message
                    financial_context=fin_ctx,
                )
            if reply is None:
                # Ollama error — use local keyword fallback with a clear notice
                reply = (
                    answer_local_question(clean_suggestion, df, budgets)
                    + "\n\n---\n*⚠️ Ollama was unavailable — this answer was generated by the local keyword assistant, not the AI model.*"
                )
                source_label = "🔴 Local keyword assistant (Ollama unavailable)"
            st.markdown(reply)
            st.caption(source_label)
        st.session_state.chat_history.append({"role": "assistant", "content": reply, "source": source_label})
        st.rerun()

    # ── Chat input ──
    prompt = st.chat_input("Ask anything about your finances…")
    if prompt and prompt.strip():
        prompt = prompt.strip()
        st.session_state.show_suggestions = False
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build financial context from Python-verified calculations
        fin_ctx = get_financial_context(df, budgets)

        with st.chat_message("assistant"):
            with st.spinner("VersaPay AI is thinking…"):
                reply, source_label = ask_versapay(
                    prompt,
                    st.session_state.chat_history[:-1],   # exclude the message we just appended
                    financial_context=fin_ctx,
                )
            if reply is None:
                # source_label contains the error string prefixed with 'ollama_error:'
                error_detail = source_label.replace("ollama_error:", "")
                reply = (
                    answer_local_question(prompt, df, budgets)
                    + "\n\n---\n*⚠️ Ollama was unavailable — this answer was generated by the local keyword assistant, not the AI model.*"
                )
                source_label = "🔴 Local keyword assistant (Ollama unavailable)"
            st.markdown(reply)
            st.caption(source_label)

        st.session_state.chat_history.append({"role": "assistant", "content": reply, "source": source_label})

    # ── Footer controls ──
    st.markdown("")
    ctrl_col1, ctrl_col2 = st.columns([1, 5])
    with ctrl_col1:
        if st.button("🗑️ New conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.show_suggestions = True
            st.rerun()
    with ctrl_col2:
        st.caption(
            "💬 Conversation history is kept only for this browser session and is never saved to the database. "
            "All AI processing runs locally on your device via Ollama."
        )


st.markdown("<hr><div style='text-align:center;color:#697895;font-size:.75rem;padding-top:8px'>VERSAPAY AI · PERSONAL FINANCE PROJECT · DEMONSTRATION ONLY</div>", unsafe_allow_html=True)
