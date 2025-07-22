import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ========== Page Config ==========
st.set_page_config(page_title="ML Trading Bot", layout="wide")

# ========== Sidebar ==========
st.sidebar.title("Input Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)

# ========== Date Calculation ==========
today = datetime.today()
if period == "6mo":
    start_date = today - relativedelta(months=6)
elif period == "1y":
    start_date = today - relativedelta(years=1)
elif period == "2y":
    start_date = today - relativedelta(years=2)
end_date = today

# ========== Get Data ==========
@st.cache_data(ttl=3600)
def get_data(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval)
    df = df.dropna(subset=["Close"])
    df["SMA_10"] = ta.sma(df["Close"], length=10).shift(1)
    df["RSI"] = ta.rsi(df["Close"], length=14).shift(1)
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"].shift(1)
    df["Return"] = df["Close"].pct_change()
    df["Lag1"] = df["Return"].shift(1)
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
    return df.dropna()

def compute_metrics(returns):
    cumulative = (1 + returns).prod() - 1
    annualized = (1 + cumulative) ** (252 / len(returns)) - 1
    vol = returns.std() * (252 ** 0.5)
    sharpe = annualized / vol if vol != 0 else 0
    dd = (1 + returns).cumprod()
    max_dd = ((dd.cummax() - dd) / dd.cummax()).max()
    return {
        "Cumulative Return": cumulative,
        "Annualized Return": annualized,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

# ========== Main Run ==========
df = get_data(ticker, start_date, end_date, interval)

features = ["SMA_10", "RSI", "MACD", "Lag1"]
X = df[features]
y = df["Target"]
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

df = df.loc[X_test.index].copy()
df["Predicted"] = y_pred
df["Trade"] = df["Predicted"].shift(1).fillna(0)
df["Strategy"] = df["Trade"] * df["Return"] - 0.001 * df["Trade"].diff().abs().fillna(0)

# ========== Charts ==========
st.title(f"📈 Results for {ticker}")
st.line_chart((df[["Return", "Strategy"]] + 1).cumprod())

# ========== Classification Report ==========
st.subheader("📋 Classification Report")
st.json(classification_report(y_test, y_pred, output_dict=True))

# ========== Metrics ==========
ml_metrics = compute_metrics(df["Strategy"].dropna())
bh_metrics = compute_metrics(df["Return"].dropna())

st.subheader("📊 Performance Comparison")
col1, col2, col3 = st.columns(3)
col1.metric("Sharpe Ratio", f"{ml_metrics['Sharpe Ratio']:.2f}")
col2.metric("Max Drawdown", f"{ml_metrics['Max Drawdown']:.2%}")
col3.metric("Annualized Return", f"{ml_metrics['Annualized Return']:.2%}")

with st.expander("🔎 Full Metrics"):
    st.write("**ML Strategy:**")
    for k, v in ml_metrics.items():
        st.write(f"{k}: {v:.2%}" if 'Return' in k or 'Drawdown' in k else f"{k}: {v:.2f}")
    st.write("**Buy & Hold:**")
    for k, v in bh_metrics.items():
        st.write(f"{k}: {v:.2%}" if 'Return' in k or 'Drawdown' in k else f"{k}: {v:.2f}")

# ========== Screener ==========
st.header("🧠 Screener: Top Stocks Where ML Beats Buy & Hold")

tickers = [  # preloaded
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JPM",
    "JNJ", "UNH", "V", "PG", "MA", "HD", "DIS", "BAC", "PFE", "KO"
]

if st.button("Run Screener"):
    results = []
    with st.spinner("Scanning..."):
        for t in tickers:
            try:
                data = get_data(t, start_date, end_date, interval)
                if data.empty:
                    continue
                X = data[features]
                y = data["Target"]
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                data = data.loc[X_test.index].copy()
                data["Predicted"] = y_pred
                data["Trade"] = data["Predicted"].shift(1).fillna(0)
                data["Strategy"] = data["Trade"] * data["Return"] - 0.001 * data["Trade"].diff().abs().fillna(0)

                ml = compute_metrics(data["Strategy"].dropna())
                bh = compute_metrics(data["Return"].dropna())

                if ml["Sharpe Ratio"] > bh["Sharpe Ratio"] and ml["Annualized Return"] > bh["Annualized Return"]:
                    results.append({
                        "Ticker": t,
                        "ML Sharpe": ml["Sharpe Ratio"],
                        "BH Sharpe": bh["Sharpe Ratio"],
                        "Sharpe Diff": ml["Sharpe Ratio"] - bh["Sharpe Ratio"],
                        "ML Return": ml["Annualized Return"],
                        "BH Return": bh["Annualized Return"]
                    })
            except:
                continue

    if results:
        df_results = pd.DataFrame(results).sort_values(by="Sharpe Diff", ascending=False).reset_index(drop=True)
        st.success(f"Found {len(df_results)} outperforming stocks.")
        st.dataframe(df_results.style.format({
            "ML Sharpe": "{:.2f}",
            "BH Sharpe": "{:.2f}",
            "Sharpe Diff": "{:.2f}",
            "ML Return": "{:.2%}",
            "BH Return": "{:.2%}"
        }))
    else:
        st.warning("No outperforming stocks found.")