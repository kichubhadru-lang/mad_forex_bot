from __future__ import annotations

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close"}


def validate_ohlc(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("DataFrame is empty.")

    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


def add_trend_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    validate_ohlc(df)

    result = df.copy()

    close = result["Close"].astype(float)
    high = result["High"].astype(float)
    low = result["Low"].astype(float)

    result["EMA20"] = EMAIndicator(
        close=close,
        window=20,
    ).ema_indicator()

    result["EMA50"] = EMAIndicator(
        close=close,
        window=50,
    ).ema_indicator()

    result["EMA200"] = EMAIndicator(
        close=close,
        window=200,
    ).ema_indicator()

    result["RSI"] = RSIIndicator(
        close=close,
        window=14,
    ).rsi()

    result["ADX"] = ADXIndicator(
        high=high,
        low=low,
        close=close,
        window=14,
    ).adx()

    result["ATR"] = AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=14,
    ).average_true_range()

    result["ATR_AVG20"] = (
        result["ATR"]
        .rolling(20)
        .mean()
    )

    result["ATR_EXPANSION"] = (
        result["ATR"]
        > result["ATR_AVG20"] * 1.05
    )

    result["EMA20_SLOPE"] = (
        result["EMA20"]
        - result["EMA20"].shift(3)
    )

    result["EMA50_SLOPE"] = (
        result["EMA50"]
        - result["EMA50"].shift(5)
    )

    return result


def add_trend_state(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = add_trend_indicators(df)

    bullish_alignment = (
        (result["EMA20"] > result["EMA50"])
        & (result["EMA50"] > result["EMA200"])
        & (result["Close"] > result["EMA20"])
    )

    bearish_alignment = (
        (result["EMA20"] < result["EMA50"])
        & (result["EMA50"] < result["EMA200"])
        & (result["Close"] < result["EMA20"])
    )

    bullish_momentum = (
        (result["EMA20_SLOPE"] > 0)
        & (result["EMA50_SLOPE"] > 0)
        & (result["RSI"] >= 50)
        & (result["RSI"] <= 70)
    )

    bearish_momentum = (
        (result["EMA20_SLOPE"] < 0)
        & (result["EMA50_SLOPE"] < 0)
        & (result["RSI"] >= 30)
        & (result["RSI"] <= 50)
    )

    result["BULLISH_TREND"] = (
        bullish_alignment
        & bullish_momentum
    )

    result["BEARISH_TREND"] = (
        bearish_alignment
        & bearish_momentum
    )

    result["TREND_DIRECTION"] = np.select(
        [
            result["BULLISH_TREND"],
            result["BEARISH_TREND"],
        ],
        [
            "BULLISH",
            "BEARISH",
        ],
        default="NEUTRAL",
    )

    result["TREND_STRENGTH"] = 0

    result.loc[
        result["BULLISH_TREND"],
        "TREND_STRENGTH",
    ] += 40

    result.loc[
        result["BEARISH_TREND"],
        "TREND_STRENGTH",
    ] += 40

    result.loc[
        result["ADX"] >= 20,
        "TREND_STRENGTH",
    ] += 15

    result.loc[
        result["ADX"] >= 25,
        "TREND_STRENGTH",
    ] += 15

    result.loc[
        result["ADX"] >= 35,
        "TREND_STRENGTH",
    ] += 10

    result.loc[
        result["ATR_EXPANSION"],
        "TREND_STRENGTH",
    ] += 10

    result.loc[
        (
            (result["BULLISH_TREND"] & (result["RSI"] >= 55))
            | (result["BEARISH_TREND"] & (result["RSI"] <= 45))
        ),
        "TREND_STRENGTH",
    ] += 10

    result["TREND_STRENGTH"] = (
        result["TREND_STRENGTH"]
        .clip(0, 100)
        .astype(int)
    )

    result["RANGING_MARKET"] = (
        (result["ADX"] < 20)
        | (
            (result["EMA20"] - result["EMA50"]).abs()
            < result["ATR"] * 0.20
        )
    )

    return result


def latest_trend_summary(
    df: pd.DataFrame,
) -> dict:
    if df is None or df.empty:
        return {}

    latest = df.iloc[-1]

    return {
        "direction": latest.get(
            "TREND_DIRECTION",
            "UNKNOWN",
        ),
        "strength": int(
            latest.get("TREND_STRENGTH", 0)
        ),
        "bullish": bool(
            latest.get("BULLISH_TREND", False)
        ),
        "bearish": bool(
            latest.get("BEARISH_TREND", False)
        ),
        "ranging": bool(
            latest.get("RANGING_MARKET", False)
        ),
        "adx": round(
            float(latest.get("ADX", 0)),
            2,
        ),
        "rsi": round(
            float(latest.get("RSI", 0)),
            2,
        ),
        "atr_expansion": bool(
            latest.get("ATR_EXPANSION", False)
        ),
        "ema20": latest.get("EMA20"),
        "ema50": latest.get("EMA50"),
        "ema200": latest.get("EMA200"),
    }
