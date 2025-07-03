import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import streamlit as st

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
    df = yf.download(ticker, period=period, interval=interval)

    # Flatten MultiIndex columns if needed (fix for Streamlit Cloud)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['SMA_10'] = df.ta.sma(length=10)
    df['RSI'] = df.ta.rsi(length=14)
    macd = df.ta.macd()
    df['MACD'] = macd['MACD_12_26_9']
    df['Lag1'] = df['Close'].shift(1)
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
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
df['Strategy'] = df['Predicted'].shift(1) * df['Return']

# === Performance chart ===
cumulative_returns = (df[['Return', 'Strategy']] + 1).cumprod()

st.subheader("📊 Strategy Performance vs Buy & Hold")
st.line_chart(cumulative_returns)

# === Metrics ===
st.subheader("📋 Model Performance Metrics")
report = classification_report(y_test, y_pred, output_dict=True)
st.json(report)

# === Most Recent Prediction ===
latest_prediction = "UP 📈" if y_pred[-1] == 1 else "DOWN 📉"
st.subheader("🔮 Latest Prediction")
st.markdown(f"Tomorrow's prediction for **{ticker}**: **{latest_prediction}**")
