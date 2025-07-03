# Machine Learning Trading Bot

This is a Python-based trading bot that uses machine learning to predict stock price movements. It applies technical indicators to historical price data and trains a Random Forest classifier to generate simple buy/sell signals.

## What It Does

- Downloads historical stock data from Yahoo Finance (default: Apple - AAPL)
- Calculates technical indicators like:
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - MACD
- Trains a Random Forest model to predict whether the stock will go **up or down** the next day
- Simulates a trading strategy based on the model's predictions
- Compares the strategy to a buy-and-hold baseline using a performance plot

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/sspatil5/ml-trading-bot.git
cd ml-trading-bot
