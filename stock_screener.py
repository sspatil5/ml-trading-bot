import yfinance as yf
from strategy import run_ml_strategy, run_buy_and_hold
import pandas as pd
import streamlit as st

def screen_stocks(stock_list, start_date, end_date, verbose=False):
    results = []

    for ticker in stock_list:
    try:
        if verbose:
            st.write(f"▶️ Processing {ticker}")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data.empty:
            if verbose:
                st.write(f"  ❌ No data for {ticker}")
            continue

        ml_perf = run_ml_strategy(data)
        bh_perf = run_buy_and_hold(data)

        sharpe_diff = ml_perf["sharpe"] - bh_perf["sharpe"]
        annualized_diff = ml_perf["annualized"] - bh_perf["annualized"]

        if sharpe_diff > 0.5 and annualized_diff > 0.01:
            results.append((ticker, ml_perf, bh_perf))
            if verbose:
                st.write(f"  ✅ {ticker} — Sharpe Diff: {sharpe_diff:.2f}, Return Diff: {annualized_diff:.2%}")
        else:
            if verbose:
                st.write(f"  ⚠️ {ticker} skipped — Sharpe Diff: {sharpe_diff:.2f}, Return Diff: {annualized_diff:.2%}")

    except Exception as e:
        if verbose:
            st.write(f"  ❌ Error on {ticker}: {e}")
        continue

    return results

def format_results(results):
    formatted = pd.DataFrame([
        {
            "Ticker": ticker,
            "ML Sharpe": ml["sharpe"],
            "BH Sharpe": bh["sharpe"],
            "Sharpe Diff": ml["sharpe"] - bh["sharpe"],
            "ML Return": ml["annualized"],
            "BH Return": bh["annualized"],
            "ML Max Drawdown": ml["max_drawdown"],
            "BH Max Drawdown": bh["max_drawdown"],
        }
        for ticker, ml, bh in results
    ])
    return formatted.sort_values(by="Sharpe Diff", ascending=False).reset_index(drop=True)