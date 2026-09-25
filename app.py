import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI Stock Analysis System", page_icon="📈", layout="wide")

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.08);
        border: 1px solid rgba(150, 150, 150, 0.2);
        padding: 12px 16px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    .badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
    }
    .badge-up { background-color: #1a7f37; color: white; }
    .badge-down { background-color: #cf222e; color: white; }
    .badge-hold { background-color: #eac54f; color: #222; }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Stock Analysis System")
st.caption("Group G4 | Risk analysis + dashboard")

# Samar ki asli file aane ke baad ise True kar dena
PREDICTIONS_ARE_REAL = False

stocks = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
}
MODELS = ["LogReg", "RandomForest", "XGBoost", "LSTM"]
LABELS = ["UP", "DOWN", "HOLD"]
BADGE_CLASS = {"UP": "badge-up", "DOWN": "badge-down", "HOLD": "badge-hold"}


def badge(label):
    cls = BADGE_CLASS.get(label, "badge-hold")
    return f'<span class="badge {cls}">{label}</span>'


st.sidebar.header("Controls")
choice = st.sidebar.selectbox("Select stock", list(stocks.keys()))
years = st.sidebar.select_slider("History (years)", options=[1, 2, 3, 5], value=5)
risk_free_pct = st.sidebar.slider("Risk-free rate (%)", 0.0, 10.0, 6.0, 0.5)
conf = st.sidebar.select_slider("VaR confidence (%)", options=[90, 95, 99], value=95)
chart_type = st.sidebar.radio("Chart type", ["Line", "Candlestick"])
show_sma = st.sidebar.checkbox("Show SMA 20", value=True)
show_ema = st.sidebar.checkbox("Show EMA 20", value=False)


@st.cache_data(ttl=3600)
def load_data(ticker):
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=5)
    data = yf.download(ticker, start=start, end=end,
                       progress=False, auto_adjust=True)
    data.columns = data.columns.get_level_values(0)
    return data


def last_years(data, n):
    cut = data.index.max() - pd.DateOffset(years=n)
    return data[data.index >= cut].copy()


def risk_summary(data, risk_free=0.06, conf=95):
    ret = data["Close"].pct_change().dropna()
    annual_vol = ret.std() * np.sqrt(252)
    cum = (1 + ret).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    var = np.percentile(ret, 100 - conf)
    annual_ret = ret.mean() * 252
    sharpe = (annual_ret - risk_free) / annual_vol

    if annual_vol < 0.20:
        level = "Low"
    elif annual_vol < 0.30:
        level = "Medium"
    else:
        level = "High"

    m = {
        "vol": round(annual_vol * 100, 2),
        "dd": round(drawdown.min() * 100, 2),
        "var": round(var * 100, 2),
        "sharpe": round(sharpe, 2),
        "level": level,
    }
    return m, ret, drawdown


@st.cache_data
def load_predictions():
    return pd.read_csv("predictions.csv")


def evaluate(preds, model):
    y_true, y_pred = preds["Actual"], preds[model]
    p_list, r_list, f_list = [], [], []
    for lbl in LABELS:
        tp = ((y_true == lbl) & (y_pred == lbl)).sum()
        fp = ((y_true != lbl) & (y_pred == lbl)).sum()
        fn = ((y_true == lbl) & (y_pred != lbl)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        p_list.append(p); r_list.append(r); f_list.append(f1)
    accuracy = (y_true == y_pred).mean()
    return accuracy, np.mean(p_list), np.mean(r_list), np.mean(f_list)


df = last_years(load_data(stocks[choice]), years)
df["SMA20"] = df["Close"].rolling(20).mean()
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
m, ret, drawdown = risk_summary(df, risk_free_pct / 100, conf)

last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
t1, t2, t3 = st.columns(3)
t1.metric(f"{choice} latest close", f"₹{last:,.2f}", f"{(last / prev - 1) * 100:.2f}%")
t2.metric("Period high", f"₹{df['High'].max():,.2f}")
t3.metric("Period low", f"₹{df['Low'].min():,.2f}")

pct_from_peak = drawdown.iloc[-1] * 100
st.caption(f"📌 {choice} abhi apne {years}-year peak se **{abs(pct_from_peak):.1f}%** neeche hai.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Price", "⚠️ Risk", "🤖 Predictions", "🆚 Compare stocks"])

with tab1:
    fig = go.Figure()
    if chart_type == "Line":
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(width=2)))
    else:
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"], name="Price"))
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA 20"))
    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20"))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_rangeslider_visible=False, hovermode="x unified")
    st.plotly_chart(fig)
    st.caption("Volume (kitne shares bike)")
    st.bar_chart(df["Volume"], height=150)

    with st.expander("📄 Show raw price data"):
        st.dataframe(df[["Open", "High", "Low", "Close", "Volume"]].tail(100), height=250)
        st.download_button("Download this data (CSV)",
                           df.to_csv().encode("utf-8"),
                           f"{choice}_price_data.csv", "text/csv")

with tab2:
    icons = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Volatility (annual)", f"{m['vol']}%")
    r2.metric("Max drawdown", f"{m['dd']}%")
    r3.metric(f"VaR {conf}% (1 day)", f"{m['var']}%")
    r4.metric("Sharpe ratio", f"{m['sharpe']}")
    r5.metric("Risk level", icons[m["level"]])

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.subheader("Drawdown over time")
        dd_fig = px.area(x=drawdown.index, y=drawdown * 100,
                         labels={"x": "Date", "y": "Drawdown (%)"})
        dd_fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(dd_fig)
    with c2:
        st.subheader("Daily returns spread")
        h = px.histogram(x=ret * 100, nbins=50, labels={"x": "Daily return (%)"})
        h.add_vline(x=m["var"], line_dash="dash", line_color="red",
                    annotation_text=f"VaR {conf}%")
        h.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(h)
    with c3:
        st.subheader("Volatility gauge")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=m["vol"],
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 60]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 20], "color": "#d4f4dd"},
                    {"range": [20, 30], "color": "#fff3cd"},
                    {"range": [30, 60], "color": "#f8d7da"},
                ],
            },
        ))
        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(gauge)

    with st.expander("What do these risk numbers mean?"):
        st.markdown("""
- **Volatility:** price kitna upar-neeche hilta hai. Zyada ho to risk zyada.
- **Max drawdown:** peak se sabse badi girawat.
- **VaR:** is confidence pe, ek din ka loss is number se zyada bura nahi tha.
- **Sharpe ratio:** risk ke hisaab se kitna fayda mila. 1 se upar achha hai.
- **Risk level:** volatility ke hisaab se Low / Medium / High (hamara simple rule).
""")

with tab3:
    try:
        preds = load_predictions()
        has_preds = True
    except FileNotFoundError:
        has_preds = False

    if not has_preds:
        st.info("predictions.csv abhi repo me nahi hai. Samar ka data aane tak dummy values.")
        p1, p2 = st.columns(2)
        with p1:
            st.write("**ML prediction**")
            st.markdown(badge("UP"), unsafe_allow_html=True)
        with p2:
            st.write("**LSTM prediction**")
            st.markdown(badge("UP"), unsafe_allow_html=True)
        eval_df = pd.DataFrame({
            "Model": MODELS, "Accuracy": ["-"] * 4, "Precision": ["-"] * 4,
            "Recall": ["-"] * 4, "F1": ["-"] * 4,
        })
        st.dataframe(eval_df, hide_index=True)
    else:
        if not PREDICTIONS_ARE_REAL:
            st.warning("Ye FAKE sample data hai, sirf testing ke liye. Report/PPT me use mat karna.")

        latest = preds.iloc[-1]
        st.write(f"**Latest prediction ({latest['Date']})**")
        cols = st.columns(len(MODELS))
        for c, model in zip(cols, MODELS):
            with c:
                st.write(f"**{model}**")
                st.markdown(badge(latest[model]), unsafe_allow_html=True)

        st.subheader("Model evaluation")
        rows = []
        for model in MODELS:
            acc, prec, rec, f1 = evaluate(preds, model)
            rows.append({"Model": model, "Accuracy": round(acc, 3), "Precision": round(prec, 3),
                        "Recall": round(rec, 3), "F1": round(f1, 3)})
        eval_df = pd.DataFrame(rows)
        st.dataframe(eval_df, hide_index=True)

        best = eval_df.loc[eval_df["F1"].idxmax()]
        st.success(f"🏆 Best model here: **{best['Model']}** (F1 = {best['F1']})")

        st.download_button("Download evaluation table (CSV)",
                           eval_df.to_csv(index=False).encode("utf-8"),
                           "evaluation.csv", "text/csv")

        st.subheader("Confusion matrix")
        cm_model = st.selectbox("Model", MODELS, key="cm_model")
        cm = pd.crosstab(preds["Actual"], preds[cm_model]).reindex(index=LABELS, columns=LABELS, fill_value=0)
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predicted", y="Actual", color="Count"))
        cm_fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(cm_fig)

with tab4:
    rows = []
    for name, tk in stocks.items():
        d = last_years(load_data(tk), years)
        s, _, _ = risk_summary(d, risk_free_pct / 100, conf)
        rows.append({"Stock": name, "Volatility %": s["vol"], "Max drawdown %": s["dd"],
                     f"VaR {conf}% (1 day) %": s["var"], "Sharpe": s["sharpe"],
                     "Risk level": s["level"]})
    comp = pd.DataFrame(rows)
    st.dataframe(comp, hide_index=True)

    safest = comp.loc[comp["Volatility %"].idxmin()]
    riskiest = comp.loc[comp["Max drawdown %"].idxmin()]
    ic1, ic2 = st.columns(2)
    ic1.info(f"🟢 Safest (lowest volatility): **{safest['Stock']}** ({safest['Volatility %']}%)")
    ic2.warning(f"🔴 Riskiest (biggest drawdown): **{riskiest['Stock']}** ({riskiest['Max drawdown %']}%)")

    bar = px.bar(comp, x="Stock", y="Max drawdown %", title="Max drawdown by stock",
                color="Max drawdown %", color_continuous_scale="Reds_r")
    bar.update_layout(height=350)
    st.plotly_chart(bar)

    st.download_button("Download comparison table (CSV)",
                       comp.to_csv(index=False).encode("utf-8"),
                       "stock_comparison.csv", "text/csv")
