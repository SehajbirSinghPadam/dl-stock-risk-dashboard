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

# ---------- NSE stock universe (~170 major stocks, A to Z, multi-sector) ----------
NSE_STOCKS = {
    "3M India": "3MINDIA.NS", "ABB India": "ABB.NS", "Abbott India": "ABBOTINDIA.NS",
    "ACC": "ACC.NS", "Adani Enterprises": "ADANIENT.NS", "Adani Green Energy": "ADANIGREEN.NS",
    "Adani Ports": "ADANIPORTS.NS", "Adani Power": "ADANIPOWER.NS",
    "Aditya Birla Capital": "ABCAPITAL.NS", "Ajanta Pharma": "AJANTPHARM.NS",
    "Alkem Laboratories": "ALKEM.NS", "Ambuja Cements": "AMBUJACEM.NS",
    "Angel One": "ANGELONE.NS", "APL Apollo Tubes": "APLAPOLLO.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS", "Apollo Tyres": "APOLLOTYRE.NS",
    "Ashok Leyland": "ASHOKLEY.NS", "Asian Paints": "ASIANPAINT.NS", "Astral": "ASTRAL.NS",
    "AU Small Finance Bank": "AUBANK.NS", "Aurobindo Pharma": "AUROPHARMA.NS",
    "Avenue Supermarts (DMart)": "DMART.NS", "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS", "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS", "Balkrishna Industries": "BALKRISIND.NS",
    "Bandhan Bank": "BANDHANBNK.NS", "Bank of Baroda": "BANKBARODA.NS",
    "Bharat Electronics": "BEL.NS", "Bharat Forge": "BHARATFORG.NS",
    "Bharat Petroleum": "BPCL.NS", "Bharti Airtel": "BHARTIARTL.NS", "Biocon": "BIOCON.NS",
    "Bosch": "BOSCHLTD.NS", "Britannia Industries": "BRITANNIA.NS", "BSE Ltd": "BSE.NS",
    "Canara Bank": "CANBK.NS", "Castrol India": "CASTROLIND.NS", "CDSL": "CDSL.NS",
    "CG Power": "CGPOWER.NS", "Cholamandalam Investment": "CHOLAFIN.NS", "Cipla": "CIPLA.NS",
    "City Union Bank": "CUB.NS", "Coal India": "COALINDIA.NS", "Coforge": "COFORGE.NS",
    "Colgate-Palmolive India": "COLPAL.NS", "Container Corp of India": "CONCOR.NS",
    "Crompton Greaves Consumer": "CROMPTON.NS", "CRISIL": "CRISIL.NS",
    "Cummins India": "CUMMINSIND.NS", "Dabur India": "DABUR.NS", "Delhivery": "DELHIVERY.NS",
    "Divi's Laboratories": "DIVISLAB.NS", "Dixon Technologies": "DIXON.NS", "DLF": "DLF.NS",
    "Dr Reddy's Labs": "DRREDDY.NS", "Eicher Motors": "EICHERMOT.NS", "Emami": "EMAMILTD.NS",
    "Escorts Kubota": "ESCORTS.NS", "Exide Industries": "EXIDEIND.NS",
    "Federal Bank": "FEDERALBNK.NS", "Fine Organic Industries": "FINEORG.NS",
    "GAIL India": "GAIL.NS", "Gillette India": "GILLETTE.NS", "Glenmark Pharma": "GLENMARK.NS",
    "Godrej Consumer Products": "GODREJCP.NS", "Godrej Properties": "GODREJPROP.NS",
    "Granules India": "GRANULES.NS", "Grasim Industries": "GRASIM.NS",
    "Havells India": "HAVELLS.NS", "HCL Technologies": "HCLTECH.NS", "HDFC AMC": "HDFCAMC.NS",
    "HDFC Bank": "HDFCBANK.NS", "HDFC Life": "HDFCLIFE.NS", "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco Industries": "HINDALCO.NS", "Hindustan Aeronautics (HAL)": "HAL.NS",
    "Hindustan Petroleum": "HINDPETRO.NS", "Hindustan Unilever": "HINDUNILVR.NS",
    "Hindustan Zinc": "HINDZINC.NS", "ICICI Bank": "ICICIBANK.NS", "ICICI Lombard": "ICICIGI.NS",
    "ICICI Prudential Life": "ICICIPRULI.NS", "IDFC First Bank": "IDFCFIRSTB.NS",
    "Indian Hotels (Taj)": "INDHOTEL.NS", "Indian Oil Corporation": "IOC.NS",
    "Indian Railway Finance Corp": "IRFC.NS", "IndiGo (InterGlobe Aviation)": "INDIGO.NS",
    "Indraprastha Gas": "IGL.NS", "IndusInd Bank": "INDUSINDBK.NS",
    "Indus Towers": "INDUSTOWER.NS", "Info Edge (Naukri)": "NAUKRI.NS", "Infosys": "INFY.NS",
    "IRCTC": "IRCTC.NS", "ITC": "ITC.NS", "J K Cement": "JKCEMENT.NS",
    "Jindal Steel": "JINDALSTEL.NS", "JSW Steel": "JSWSTEEL.NS",
    "Jubilant FoodWorks": "JUBLFOOD.NS", "Kajaria Ceramics": "KAJARIACER.NS",
    "KEI Industries": "KEI.NS", "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "KPIT Technologies": "KPITTECH.NS", "L&T (Larsen & Toubro)": "LT.NS",
    "L&T Technology Services": "LTTS.NS", "Laurus Labs": "LAURUSLABS.NS",
    "LIC (Life Insurance Corp)": "LICI.NS", "LIC Housing Finance": "LICHSGFIN.NS",
    "Lupin": "LUPIN.NS", "Mahindra & Mahindra": "M&M.NS", "Manappuram Finance": "MANAPPURAM.NS",
    "Marico": "MARICO.NS", "Maruti Suzuki": "MARUTI.NS", "Max Financial Services": "MFSL.NS",
    "Mphasis": "MPHASIS.NS", "Muthoot Finance": "MUTHOOTFIN.NS", "NATCO Pharma": "NATCOPHARM.NS",
    "National Aluminium": "NATIONALUM.NS", "NBCC India": "NBCC.NS", "Nestle India": "NESTLEIND.NS",
    "NMDC": "NMDC.NS", "NTPC": "NTPC.NS", "Oil India": "OIL.NS", "ONGC": "ONGC.NS",
    "Persistent Systems": "PERSISTENT.NS", "Petronet LNG": "PETRONET.NS",
    "Pidilite Industries": "PIDILITIND.NS", "PI Industries": "PIIND.NS",
    "Polycab India": "POLYCAB.NS", "Power Finance Corp": "PFC.NS",
    "Power Grid Corp": "POWERGRID.NS", "Punjab National Bank": "PNB.NS",
    "Ratnamani Metals": "RATNAMANI.NS", "REC Limited": "RECLTD.NS", "Reliance": "RELIANCE.NS",
    "SAIL": "SAIL.NS", "SBI (State Bank of India)": "SBIN.NS", "SBI Cards": "SBICARD.NS",
    "SBI Life Insurance": "SBILIFE.NS", "Shree Cement": "SHREECEM.NS",
    "Shriram Finance": "SHRIRAMFIN.NS", "Siemens": "SIEMENS.NS", "SRF Ltd": "SRF.NS",
    "Star Health Insurance": "STARHEALTH.NS", "Sun Pharma": "SUNPHARMA.NS",
    "Sun TV Network": "SUNTV.NS", "Swiggy": "SWIGGY.NS", "Tata Communications": "TATACOMM.NS",
    "Tata Consumer Products": "TATACONSUM.NS", "Tata Elxsi": "TATAELXSI.NS",
    "Tata Power": "TATAPOWER.NS", "Tata Steel": "TATASTEEL.NS", "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS", "Titan Company": "TITAN.NS", "Torrent Pharma": "TORNTPHARM.NS",
    "Torrent Power": "TORNTPOWER.NS", "Trent Ltd (Westside/Zudio)": "TRENT.NS",
    "TVS Motor": "TVSMOTOR.NS", "UltraTech Cement": "ULTRACEMCO.NS",
    "Union Bank of India": "UNIONBANK.NS", "United Breweries": "UBL.NS",
    "United Spirits": "UNITDSPR.NS", "UPL Ltd": "UPL.NS", "Vedanta": "VEDL.NS",
    "Voltas": "VOLTAS.NS", "Wipro": "WIPRO.NS", "Yes Bank": "YESBANK.NS",
    "Zydus Lifesciences": "ZYDUSLIFE.NS",
}
DEFAULT_WATCHLIST = ["Reliance", "TCS", "HDFC Bank", "Infosys"]

MODELS = ["LogReg", "RandomForest", "XGBoost", "LSTM"]
LABELS = ["UP", "DOWN", "HOLD"]
BADGE_CLASS = {"UP": "badge-up", "DOWN": "badge-down", "HOLD": "badge-hold"}


def badge(label):
    cls = BADGE_CLASS.get(label, "badge-hold")
    return f'<span class="badge {cls}">{label}</span>'


with st.expander("📚 Naye ho stock market me? Yahan se shuru karo"):
    st.markdown("""
- **Stock/Share**: ek company ka chhota hissa jo tum khareed sakte ho.
- **Open/Close**: din shuru/khatam hote waqt ka price.
- **High/Low**: din ka sabse upar/neeche price.
- **Volume**: us din kitne shares trade hue.
- **SMA/EMA**: pichle N dino ka average price, trend samajhne ke liye.
- **UP/DOWN/HOLD prediction**: model ka andaza hai, guarantee nahi.
- Yeh dashboard ek **learning project** hai, real investment advice nahi hai.
""")

st.sidebar.header("Controls")
st.sidebar.caption("🔍 Box par click karke company ka naam type karo (search)")
choice = st.sidebar.selectbox("Select stock", sorted(NSE_STOCKS.keys()),
                              index=sorted(NSE_STOCKS.keys()).index("Reliance"))
watchlist = st.sidebar.multiselect("⭐ Pin your watchlist (max 5)",
                                   options=sorted(NSE_STOCKS.keys()),
                                   default=DEFAULT_WATCHLIST, max_selections=5)
if not watchlist:
    watchlist = DEFAULT_WATCHLIST

st.sidebar.markdown("---")
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


@st.cache_data(ttl=3600)
def get_company_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        mcap = info.get("marketCap")
        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector", "N/A"),
            "market_cap_cr": round(mcap / 1e7) if mcap else None,
        }
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_news(ticker):
    try:
        items = yf.Ticker(ticker).news or []
        out = []
        for it in items[:4]:
            title = it.get("title") or it.get("content", {}).get("title")
            link = it.get("link") or it.get("content", {}).get("canonicalUrl", {}).get("url")
            if title:
                out.append((title, link))
        return out
    except Exception:
        return []


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


ticker = NSE_STOCKS[choice]
df = last_years(load_data(ticker), years)
df["SMA20"] = df["Close"].rolling(20).mean()
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
m, ret, drawdown = risk_summary(df, risk_free_pct / 100, conf)

last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
t1, t2, t3 = st.columns(3)
t1.metric(f"{choice} latest close", f"₹{last:,.2f}", f"{(last / prev - 1) * 100:.2f}%")
t2.metric("Period high", f"₹{df['High'].max():,.2f}")
t3.metric("Period low", f"₹{df['Low'].min():,.2f}")

info = get_company_info(ticker)
if info:
    mc = f"₹{info['market_cap_cr']:,.0f} Cr" if info["market_cap_cr"] else "N/A"
    st.caption(f"🏢 {info['name']} · Sector: {info['sector']} · Market cap: {mc}")

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

    with st.expander("📰 Recent news"):
        news_items = get_news(ticker)
        if not news_items:
            st.caption("News abhi available nahi hai.")
        else:
            for title, link in news_items:
                st.markdown(f"- [{title}]({link})" if link else f"- {title}")

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
    st.caption("Yeh tumhare pinned watchlist ke stocks hain (sidebar se ⭐ Pin your watchlist badal sakte ho).")
    rows = []
    for name in watchlist:
        tk = NSE_STOCKS[name]
        d = last_years(load_data(tk), years)
        s, _, _ = risk_summary(d, risk_free_pct / 100, conf)
        last_p, prev_p = d["Close"].iloc[-1], d["Close"].iloc[-2]
        rows.append({
            "Stock": name, "Price ₹": round(last_p, 2),
            "Day change %": round((last_p / prev_p - 1) * 100, 2),
            "Volatility %": s["vol"], "Max drawdown %": s["dd"],
            f"VaR {conf}% (1 day) %": s["var"], "Sharpe": s["sharpe"],
            "Risk level": s["level"],
        })
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
