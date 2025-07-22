import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def get_stock_data(ticker='AAPL', period='1y', interval='1d'):
    df = yf.download(ticker, period=period, interval=interval)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"])
    return df

df = get_stock_data('AAPL')
print(df.tail())

df['SMA_10'] = df.ta.sma(close=df['Close'], length=10) #Adds 10-day simple moving average of closing price
df['RSI'] = df.ta.rsi(close=df['Close'], length=14) #Relative Strength Index, measures momentum, detecting overbought/oversold conditions
macd = df.ta.macd(close=df['Close'])
df['MACD'] = macd['MACD_12_26_9'] # Another momentum-based signal
df['Lag1'] = df['Close'].shift(1) # Basic time dependency: Adds yesterday's closing price as lagged feature
df['Return'] = df['Close'].pct_change() # Adds daily return in percent change
df.dropna(inplace=True) # Removes rows with missing values

df['Target'] = (df['Return'].shift(-1) > 0).astype(int) # Sets target for prediction
# 1 = Stock went up the next day
# 0 = Stock went down or stayed the same

features = ['SMA_10', 'RSI', 'MACD', 'Lag1']
X = df[features] # input data: indicators and price
y = df['Target'] # what you're trying to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
# Splits dataset; 80% to train model
# 20% to test how well it predicts unseen data

model = RandomForestClassifier(n_estimators=100, random_state=42) # Creates random forest with 100 decision trees
model.fit(X_train, y_train) # Fits into training data (Learns patterns)

proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba > 0.55).astype(int) # Attempts to predict test data with confidence threshold
print(classification_report(y_test, y_pred)) # Prints metrics like Accuracy, Precision, Recall

df = df.iloc[X_test.index] # Keeps only data in the test set (Strategy Simulation)
df['Predicted'] = y_pred
df['Strategy'] = df['Predicted'].shift(1) * df['Return']
# If model predicted 1 yesterday, buy today
# If model predicted 0 yesterday, don't buy, return 0

# Strategy logic with transaction cost
cost = 0.001  # 0.1% per trade
df['Trade'] = df['Predicted'].shift(1).fillna(0)
df['Strategy'] = df['Trade'] * df['Return'] - cost * df['Trade'].diff().abs().fillna(0)

# Performance metrics
def compute_metrics(returns, label="Strategy"):
    cumulative = (1 + returns).prod() - 1
    ann_return = (1 + cumulative) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * (252 ** 0.5)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    drawdown = (1 + returns).cumprod()
    max_dd = ((drawdown.cummax() - drawdown) / drawdown.cummax()).max()
    
    print(f"\n--- {label} ---")
    print(f"Cumulative Return: {cumulative:.2%}")
    print(f"Annualized Return: {ann_return:.2%}")
    print(f"Volatility: {ann_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")

# Print metrics
compute_metrics(df['Return'], label="Buy & Hold")
compute_metrics(df['Strategy'], label="ML Strategy")

# Cumulative returns
cumulative_returns = (df[['Return', 'Strategy']] + 1).cumprod()
cumulative_returns.plot(title='ML Strategy vs Buy & Hold')
plt.show()
# Plots what happens if you just hold the stock
# Plots the model's predictions
# Shows if the ML bot is actually useful, or just holding stock
