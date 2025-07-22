import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def compute_metrics(returns):
    cumulative = (1 + returns).prod() - 1
    trading_days = returns.count()
    ann_return = (1 + cumulative) ** (252 / trading_days) - 1 if trading_days > 0 else 0
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

def run_buy_and_hold(df, test_index=None):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    if test_index is not None:
        df = df.loc[test_index]
    return compute_metrics(df['Return'])

def run_ml_strategy(df, return_full_df=False):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(subset=["Close"], inplace=True)

    # Compute indicators
    df['SMA_10'] = ta.sma(df['Close'], length=10).shift(1)
    df['RSI'] = ta.rsi(df['Close'], length=14).shift(1)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9'].shift(1)
    df['Lag1'] = df['Close'].shift(1)  # Lag1 of Close, shifted an extra step to match features at t-1
    df['Return'] = df['Close'].pct_change()

    # Define target
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)

    features = ['SMA_10', 'RSI', 'MACD', 'Lag1']
    X = df[features]
    y = df['Target']

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba > 0.55).astype(int)

    df_test = df.loc[X_test.index].copy()
    df_test['Predicted'] = y_pred
    df_test['Trade'] = df_test['Predicted'].shift(1).fillna(0)
    df_test['Strategy'] = df_test['Trade'] * df_test['Return'] - 0.001 * df_test['Trade'].diff().abs().fillna(0)

    df_test.dropna(subset=['Strategy'], inplace=True)
    metrics = compute_metrics(df_test['Strategy'].dropna())

    return (metrics, df_test) if return_full_df else metrics