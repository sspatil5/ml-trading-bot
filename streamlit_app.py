import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import streamlit as st
from alpaca import run_intraday_trader
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np

st.set_page_config(page_title="ML Trading Bot", layout="wide")

# === Tab selector ===
mode = st.sidebar.radio("Select Mode", ["📅 Daily ML Bot", "⚡ Intraday Trader (Alpaca)"])

base_model = RandomForestClassifier(n_estimators=100, random_state=42)

@st.cache_data
def get_stock_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    # Flatten column names if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure Close column exists and is clean
    df = df.dropna(subset=["Close"])
    
    # Use explicit column reference to avoid ambiguity
    df['SMA_10'] = df.ta.sma(close=df['Close'], length=10).squeeze()
    df['RSI'] = df.ta.rsi(close=df['Close'], length=14).squeeze()
    macd = df.ta.macd(close=df['Close'])
    df['MACD'] = macd['MACD_12_26_9'].squeeze()
    
    df['Return'] = df['Close'].pct_change()
    df['Lag1'] = df['Return'].shift(1)
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    return df

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

def run_daily_strategy():
    # === Title ===
    st.title("📈 Machine Learning Trading Bot")
    st.markdown("Predict next-day stock movement using a Random Forest classifier and technical indicators.")

    # === Sidebar Input ===
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
    period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y"], index=1)
    interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)

    df = get_stock_data(ticker, period, interval)

    # === Train model ===
    features = ['SMA_10', 'RSI', 'MACD', 'Lag1']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Drop any rows with NaNs in training and testing data
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]

    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }
    model_results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        df_temp = df.loc[X_test.index].copy()
        df_temp['Predicted'] = y_pred
        df_temp['Trade'] = df_temp['Predicted'].shift(1).fillna(0)
        df_temp['Strategy'] = df_temp['Trade'] * df_temp['Return'] - 0.001 * df_temp['Trade'].diff().abs().fillna(0)

        strat_metrics = compute_metrics(df_temp['Strategy'].dropna())

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().dropna()
        report_df[["precision", "recall", "f1-score"]] = report_df[["precision", "recall", "f1-score"]].round(2)
        report_df["support"] = report_df["support"].astype(int)

        model_results.append({
            "Model": name,
            "Sharpe Ratio": strat_metrics["Sharpe Ratio"],
            "Annualized Return": strat_metrics["Annualized Return"],
            "Max Drawdown": strat_metrics["Max Drawdown"],
            "Classification Report": report_dict,
            "Report DF": report_df
        })

    # === Model Comparison Table ===
    st.header("🤖 Model Comparison Results")

    comparison_data = {
        "Model": [],
        "Accuracy": [],
        "F1 (Class 0)": [],
        "F1 (Class 1)": [],
        "Macro F1": [],
        "Weighted F1": []
    }

    for result in model_results:
        comparison_data["Model"].append(result["Model"])
        report = result["Classification Report"]
        comparison_data["Accuracy"].append(report.get("accuracy", 0))
        comparison_data["F1 (Class 0)"].append(report.get("0", {}).get("f1-score", 0))
        comparison_data["F1 (Class 1)"].append(report.get("1", {}).get("f1-score", 0))
        comparison_data["Macro F1"].append(report.get("macro avg", {}).get("f1-score", 0))
        comparison_data["Weighted F1"].append(report.get("weighted avg", {}).get("f1-score", 0))

    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison.style.format({
        "Accuracy": "{:.2%}",
        "F1 (Class 0)": "{:.2f}",
        "F1 (Class 1)": "{:.2f}",
        "Macro F1": "{:.2f}",
        "Weighted F1": "{:.2f}"
    }))

    # Identify best model by Macro F1 score
    best_model = df_comparison.loc[df_comparison["Macro F1"].idxmax()]["Model"]
    st.markdown(f"✅ **Best performing model for {ticker}: `{best_model}` based on Macro F1 score.**")

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
    #st.subheader("📋 Model Classification Metrics")
    #report = classification_report(y_test, y_pred, output_dict=True)
    #st.json(report)

    #strategy_metrics = compute_metrics(df['Strategy'].dropna())
    #buy_hold_metrics = compute_metrics(df['Return'].dropna())

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
    st.markdown("Run the ML strategy across 50 major tickers and compare vs Buy & Hold.")

    top_50_tickers = [
        'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JPM',
        'JNJ', 'UNH', 'V', 'PG', 'MA', 'HD', 'DIS', 'BAC', 'PFE', 'KO',
        'PEP', 'MRK', 'ABBV', 'XOM', 'WMT', 'CVX', 'AVGO', 'NFLX', 'ADBE', 'T',
        'INTC', 'CSCO', 'CRM', 'VZ', 'CMCSA', 'ACN', 'ABT', 'NKE', 'MCD', 'TMO',
        'COST', 'QCOM', 'TXN', 'NEE', 'AMD', 'LOW', 'AMGN', 'DHR', 'MDT', 'LMT'
    ]

    if st.button("🚀 Run Screener on Top 50 Stocks"):
        outperforming = []
        with st.spinner("Running strategy on top 50 stocks..."):
            for symbol in top_50_tickers:
                try:
                    df_temp = get_stock_data(symbol, period, interval)

                    X = df_temp[features]
                    y = df_temp['Target']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    df_temp = df_temp.loc[X_test.index]
                    df_temp['Predicted'] = y_pred
                    df_temp['Trade'] = df_temp['Predicted'].shift(1).fillna(0)
                    df_temp['Strategy'] = df_temp['Trade'] * df_temp['Return'] - 0.001 * df_temp['Trade'].diff().abs().fillna(0)

                    ml_metrics = compute_metrics(df_temp['Strategy'].dropna())
                    bh_metrics = compute_metrics(df_temp['Return'].dropna())

                    prediction_label = "📈 UP" if y_pred[-1] == 1 else "📉 DOWN"

                    if ml_metrics['Sharpe Ratio'] > bh_metrics['Sharpe Ratio'] and ml_metrics['Annualized Return'] > bh_metrics['Annualized Return']:
                        outperforming.append({
                            "Ticker": symbol,
                            "ML Sharpe": ml_metrics['Sharpe Ratio'],
                            "BH Sharpe": bh_metrics['Sharpe Ratio'],
                            "Sharpe Diff": ml_metrics['Sharpe Ratio'] - bh_metrics['Sharpe Ratio'],
                            "ML Return": ml_metrics['Annualized Return'],
                            "BH Return": bh_metrics['Annualized Return'],
                            "Prediction": prediction_label
                        })
                except Exception as e:
                    st.write(f"❌ {symbol} failed: {e}")
                    continue

        if not outperforming:
            st.warning("No outperforming stocks found.")
        else:
            df_screen = pd.DataFrame(outperforming).sort_values(by="Sharpe Diff", ascending=False)
            st.success(f"Found {len(df_screen)} outperforming stocks!")
            st.dataframe(df_screen.style.format({
                "ML Sharpe": "{:.2f}",
                "BH Sharpe": "{:.2f}",
                "Sharpe Diff": "{:.2f}",
                "ML Return": "{:.2%}",
                "BH Return": "{:.2%}",
            }))
    return model_results

if mode == "📅 Daily ML Bot":
    model_results = run_daily_strategy()

    st.header("🤖 Model Comparison Results")

    for result in model_results:
        st.subheader(f"📌 {result['Model']}")
        st.metric("Sharpe Ratio", f"{result['Sharpe Ratio']:.2f}")
        st.metric("Annualized Return", f"{result['Annualized Return']:.2%}")
        st.metric("Max Drawdown", f"{result['Max Drawdown']:.2%}")
        with st.expander("📋 Classification Report"):
            st.dataframe(result["Report DF"].style.format({
                "precision": "{:.2f}",
                "recall": "{:.2f}",
                "f1-score": "{:.2f}",
                "support": "{:d}"
            }))

elif mode == "⚡ Intraday Trader (Alpaca)":
    run_intraday_trader()