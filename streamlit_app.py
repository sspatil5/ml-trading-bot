import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import streamlit as st
from stock_screener import screen_stocks, format_results

st.set_page_config(page_title="ML Trading Bot", layout="wide")

# === Title ===
st.title("📈 Machine Learning Trading Bot")
st.markdown("Predict next-day stock movement using a Random Forest classifier and technical indicators.")

# === Sidebar Input ===
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)

@st.cache_data
def get_stock_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    # Flatten column names if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure Close column exists and is clean
    df = df.dropna(subset=["Close"])
    
    # Use explicit column reference to avoid ambiguity
    df['SMA_10'] = df.ta.sma(close=df['Close'], length=10)
    df['RSI'] = df.ta.rsi(close=df['Close'], length=14)
    macd = df.ta.macd(close=df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    
    df['Return'] = df['Close'].pct_change()
    df['Lag1'] = df['Return'].shift(1)
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    return df

df = get_stock_data(ticker, period, interval)

# === Train model ===
features = ['SMA_10', 'RSI', 'MACD', 'Lag1']
X = df[features]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
df = df.loc[X_test.index]
df['Predicted'] = y_pred
df['Trade'] = df['Predicted'].shift(1).fillna(0)
df['Strategy'] = df['Trade'] * df['Return'] - 0.001 * df['Trade'].diff().abs().fillna(0)

# === Performance chart ===
cumulative_returns = (df[['Return', 'Strategy']] + 1).cumprod()

st.subheader("📊 Strategy Performance vs Buy & Hold")
st.line_chart(cumulative_returns)

# === Metrics ===
st.subheader("📋 Model Classification Metrics")
report = classification_report(y_test, y_pred, output_dict=True)
st.json(report)

# === Strategy performance metrics ===
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

strategy_metrics = compute_metrics(df['Strategy'].dropna())
buy_hold_metrics = compute_metrics(df['Return'].dropna())

st.subheader("📈 Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Sharpe Ratio", f"{strategy_metrics['Sharpe Ratio']:.2f}")
col2.metric("Max Drawdown", f"{strategy_metrics['Max Drawdown']:.2%}")
col3.metric("Annualized Return", f"{strategy_metrics['Annualized Return']:.2%}")

with st.expander("🔎 Full Metrics Comparison"):
    st.write("**ML Strategy**")
    for k, v in strategy_metrics.items():
        st.write(f"{k}: {v:.2%}" if 'Return' in k or 'Drawdown' in k else f"{k}: {v:.2f}")
    st.write("**Buy & Hold**")
    for k, v in buy_hold_metrics.items():
        st.write(f"{k}: {v:.2%}" if 'Return' in k or 'Drawdown' in k else f"{k}: {v:.2f}")


# === Most Recent Prediction ===
latest_prediction = "UP 📈" if y_pred[-1] == 1 else "DOWN 📉"
st.subheader("🔮 Latest Prediction")
st.markdown(f"Tomorrow's prediction for **{ticker}**: **{latest_prediction}**")

# === Screener Section ===
st.header("🧠 Stock Screener: Find Top ML-Performing Stocks")

from datetime import date, timedelta
from stock_screener import screen_stocks

st.markdown("Scan a set of top stocks to find where the ML strategy outperforms Buy & Hold.")

# Preloaded top ~50 US tickers
preloaded_tickers = [
    'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JPM',
    'JNJ', 'UNH', 'V', 'PG', 'MA', 'HD', 'DIS', 'BAC', 'PFE', 'KO',
    'PEP', 'MRK', 'ABBV', 'XOM', 'WMT', 'CVX', 'AVGO', 'NFLX', 'ADBE', 'T',
    'INTC', 'CSCO', 'CRM', 'VZ', 'CMCSA', 'ACN', 'ABT', 'NKE', 'MCD', 'TMO',
    'COST', 'QCOM', 'TXN', 'NEE', 'AMD', 'LOW', 'AMGN', 'DHR', 'MDT', 'LMT'
]

# Screener time frame and threshold
screener_start = date.today() - timedelta(days=180)
screener_end = date.today()
sharpe_threshold = 0.2

if st.button("🚀 Run Screener on Top 50 Stocks"):
    with st.spinner("Running strategy..."):
        results = screen_stocks(preloaded_tickers, screener_start, screener_end, sharpe_threshold)

        if not results:
            st.warning("No outperforming stocks found.")
        else:
            df_results = format_results(results)

            st.success(f"Found {len(df_results)} outperforming stocks.")
            st.dataframe(df_results.style.format({
                "ML Sharpe": "{:.2f}",
                "BH Sharpe": "{:.2f}",
                "Sharpe Diff": "{:.2f}",
                "ML Return": "{:.2%}",
                "BH Return": "{:.2%}",
            }))