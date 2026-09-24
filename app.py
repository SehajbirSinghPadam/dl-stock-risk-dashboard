import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI Stock Analysis System", layout="wide")
st.title("AI Stock Analysis System")

stocks = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
}
choice = st.selectbox("Selected stock:", list(stocks.keys()))


@st.cache_data
def load_data(ticker):
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=5)
    data = yf.download(ticker, start=start, end=end,
                       progress=False, auto_adjust=True)
    data.columns = data.columns.get_level_values(0)
    return data


def risk_summary(df, risk_free=0.06):
    ret = df["Close"].pct_change().dropna()
    annual_vol = ret.std() * np.sqrt(252)
    cum = (1 + ret).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    var_95 = np.percentile(ret, 5)
    annual_ret = ret.mean() * 252
    sharpe = (annual_ret - risk_free) / annual_vol

    if annual_vol < 0.20:
        level = "Low"
    elif annual_vol < 0.30:
        level = "Medium"
    else:
        level = "High"

    return annual_vol * 100, max_dd * 100, var_95 * 100, sharpe, level


df = load_data(stocks[choice])

st.subheader("Historical price chart")
st.line_chart(df["Close"])

st.subheader("Predictions (dummy for now)")
c1, c2 = st.columns(2)
c1.metric("ML prediction", "UP")
c2.metric("LSTM prediction", "UP")

st.subheader("Risk analysis")
vol, dd, var, sharpe, level = risk_summary(df)
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Volatility (annual)", f"{vol:.2f}%")
r2.metric("Max drawdown", f"{dd:.2f}%")
r3.metric("VaR 95% (1 day)", f"{var:.2f}%")
r4.metric("Sharpe ratio", f"{sharpe:.2f}")
r5.metric("Risk level", level)
