import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def compute_metrics(returns):
    cumulative = (1 + returns).prod() - 1
    ann_return = (1 + cumulative) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * (252 ** 0.5)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    drawdown = (1 + returns).cumprod()
    max_dd = ((drawdown.cummax() - drawdown) / drawdown.cummax()).max()
    return {
        "cumulative": cumulative,
        "annualized": ann_return,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd
    }

def run_buy_and_hold(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return compute_metrics(df['Return'])

def run_ml_strategy(df, return_full_df=False):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(subset=["Close"], inplace=True)

    # Feature engineering
    df['SMA_10'] = df.ta.sma(close=df['Close'], length=10)
    df['RSI'] = df.ta.rsi(close=df['Close'], length=14)
    macd = df.ta.macd(close=df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['Lag1'] = df['Close'].shift(1)
    df['Return'] = df['Close'].pct_change()

    # Shift features forward by 1 so they only use info available at time t
    df['SMA_10'] = df['SMA_10'].shift(1)
    df['RSI'] = df['RSI'].shift(1)
    df['MACD'] = df['MACD'].shift(1)
    df['Lag1'] = df['Lag1'].shift(1)

    # Define target: whether next-day return is positive
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)

    # Feature and target selection
    features = ['SMA_10', 'RSI', 'MACD', 'Lag1']
    X = df[features]
    y = df['Target']

    # Split without shuffling to preserve time order
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba > 0.55).astype(int)

    # Apply trading logic
    df = df.loc[X_test.index].copy()
    df['Predicted'] = y_pred
    df['Trade'] = df['Predicted'].shift(1).fillna(0)  # Use signal from previous day
    cost = 0.001
    df['Strategy'] = df['Trade'] * df['Return'] - cost * df['Trade'].diff().abs().fillna(0)

    metrics = compute_metrics(df['Strategy'].dropna())

    return (metrics, df) if return_full_df else metrics