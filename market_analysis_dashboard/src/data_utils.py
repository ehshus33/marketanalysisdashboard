from __future__ import annotations
import pandas as pd
import yfinance as yf

def download_stock_data(ticker: str, start: str) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Please enter a ticker.")

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    df = df.reset_index()

    cleaned_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            col = col[0]
        cleaned_columns.append(str(col).lower().replace(" ", "_"))
    df.columns = cleaned_columns

    if "date" not in df.columns:
        raise ValueError(f"Date column not found. Columns returned: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    needed = ["open", "high", "low", "close", "volume"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Columns returned: {df.columns.tolist()}")

    return df[needed].dropna()
