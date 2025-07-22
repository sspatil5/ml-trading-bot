# alpaca_intraday.py

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

def run_intraday_trader():
    st.title("⚡ Intraday Trader (Alpaca)")
    st.markdown("Get real-time minute-level data using Alpaca's market data API.")

    # Load credentials from Streamlit secrets
    ALPACA_KEY = st.secrets["ALPACA_API_KEY"]
    ALPACA_SECRET = st.secrets["ALPACA_API_SECRET"]

    BASE_URL = "https://data.alpaca.markets/v2"

    # User inputs
    symbol = st.text_input("Enter stock symbol", value="AAPL").upper()
    timeframe = st.selectbox("Select timeframe", ["1Min", "5Min", "15Min"], index=0)
    limit = st.slider("Number of bars to retrieve", min_value=50, max_value=500, value=100)

    # Date range for today
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=2)

    # Request headers
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET
    }

    # Fetch bars
    url = f"{BASE_URL}/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start_time.isoformat() + "Z",
        "end": end_time.isoformat() + "Z",
        "limit": limit
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        st.error(f"Failed to fetch data: {response.status_code} - {response.text}")
        return

    bars = pd.DataFrame(response.json()["bars"])

    if bars.empty:
        st.warning("No data returned.")
        return

    # Format and display
    bars["t"] = pd.to_datetime(bars["t"])
    bars.set_index("t", inplace=True)

    st.subheader(f"📉 Price Chart for {symbol} ({timeframe})")
    st.line_chart(bars["c"])  # Closing price

    st.subheader("📊 Latest Data Snapshot")
    st.dataframe(bars.tail(10)[["o", "h", "l", "c", "v"]].rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
    }))