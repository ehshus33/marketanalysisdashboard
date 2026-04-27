from __future__ import annotations
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

FEATURE_COLUMNS = [
    "return_1d",
    "ma_10",
    "ma_20",
    "ma_gap",
    "volatility_10",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "lag_return_1",
    "lag_return_2",
    "lag_return_3",
    "volume_change",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["return_1d"] = data["close"].pct_change()
    data["ma_10"] = data["close"].rolling(10).mean()
    data["ma_20"] = data["close"].rolling(20).mean()
    data["ma_gap"] = data["ma_10"] - data["ma_20"]
    data["volatility_10"] = data["return_1d"].rolling(10).std()
    data["volume_change"] = data["volume"].pct_change()

    data["rsi_14"] = RSIIndicator(close=data["close"], window=14).rsi()

    macd = MACD(close=data["close"])
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_diff"] = macd.macd_diff()

    data["lag_return_1"] = data["return_1d"].shift(1)
    data["lag_return_2"] = data["return_1d"].shift(2)
    data["lag_return_3"] = data["return_1d"].shift(3)

    data["target_up"] = (data["close"].shift(-1) > data["close"]).astype(int)
    data["target_return"] = data["close"].shift(-1) / data["close"] - 1

    return data.dropna()

def latest_indicator_summary(data: pd.DataFrame) -> dict:
    latest = data.iloc[-1]
    rsi = float(latest["rsi_14"])
    if rsi >= 70:
        rsi_status = "Overbought"
    elif rsi <= 30:
        rsi_status = "Oversold"
    else:
        rsi_status = "Neutral"

    trend_status = "Short MA above Long MA" if float(latest["ma_gap"]) > 0 else "Short MA below Long MA"
    macd_status = "Bullish momentum" if float(latest["macd"]) > float(latest["macd_signal"]) else "Bearish momentum"

    return {
        "rsi_status": rsi_status,
        "trend_status": trend_status,
        "macd_status": macd_status,
    }
