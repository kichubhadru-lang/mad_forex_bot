from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close"}


def validate_ohlc(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("DataFrame is empty.")

    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


def add_liquidity_sweeps(
    df: pd.DataFrame,
    lookback: int = 20,
    wick_ratio: float = 1.5,
) -> pd.DataFrame:
    """
    Detect bullish and bearish liquidity sweeps.

    Bullish sweep:
    - Price trades below a prior swing low.
    - Candle closes back above that level.
    - Lower wick is meaningfully larger than the body.

    Bearish sweep:
    - Price trades above a prior swing high.
    - Candle closes back below that level.
    - Upper wick is meaningfully larger than the body.
    """
    validate_ohlc(df)

    result = df.copy()

    result["PRIOR_HIGH"] = (
        result["High"]
        .rolling(lookback)
        .max()
        .shift(1)
    )

    result["PRIOR_LOW"] = (
        result["Low"]
        .rolling(lookback)
        .min()
        .shift(1)
    )

    body = (result["Close"] - result["Open"]).abs()

    upper_wick = (
        result["High"]
        - result[["Open", "Close"]].max(axis=1)
    )

    lower_wick = (
        result[["Open", "Close"]].min(axis=1)
        - result["Low"]
    )

    result["BODY_SIZE"] = body
    result["UPPER_WICK"] = upper_wick
    result["LOWER_WICK"] = lower_wick

    result["BULLISH_LIQUIDITY_SWEEP"] = (
        (result["Low"] < result["PRIOR_LOW"])
        & (result["Close"] > result["PRIOR_LOW"])
        & (lower_wick >= body * wick_ratio)
    )

    result["BEARISH_LIQUIDITY_SWEEP"] = (
        (result["High"] > result["PRIOR_HIGH"])
        & (result["Close"] < result["PRIOR_HIGH"])
        & (upper_wick >= body * wick_ratio)
    )

    result = add_stop_hunt_detection(result)
    result = add_liquidity_strength(result)
    result = add_recent_liquidity_flags(result)

    return result


def add_stop_hunt_detection(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    bullish_range_rejection = (
        result["Close"] > result["Open"]
    )

    bearish_range_rejection = (
        result["Close"] < result["Open"]
    )

    result["BULLISH_STOP_HUNT"] = (
        result["BULLISH_LIQUIDITY_SWEEP"]
        & bullish_range_rejection
    )

    result["BEARISH_STOP_HUNT"] = (
        result["BEARISH_LIQUIDITY_SWEEP"]
        & bearish_range_rejection
    )

    return result


def add_liquidity_strength(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a 0–10 strength score.
    """
    result = df.copy()

    result["BULLISH_LIQUIDITY_STRENGTH"] = 0
    result["BEARISH_LIQUIDITY_STRENGTH"] = 0

    bullish_mask = result["BULLISH_LIQUIDITY_SWEEP"]
    bearish_mask = result["BEARISH_LIQUIDITY_SWEEP"]

    result.loc[
        bullish_mask,
        "BULLISH_LIQUIDITY_STRENGTH",
    ] += 4

    result.loc[
        bearish_mask,
        "BEARISH_LIQUIDITY_STRENGTH",
    ] += 4

    result.loc[
        result["BULLISH_STOP_HUNT"],
        "BULLISH_LIQUIDITY_STRENGTH",
    ] += 2

    result.loc[
        result["BEARISH_STOP_HUNT"],
        "BEARISH_LIQUIDITY_STRENGTH",
    ] += 2

    bullish_large_wick = (
        result["LOWER_WICK"]
        >= result["BODY_SIZE"] * 2.5
    )

    bearish_large_wick = (
        result["UPPER_WICK"]
        >= result["BODY_SIZE"] * 2.5
    )

    result.loc[
        bullish_mask & bullish_large_wick,
        "BULLISH_LIQUIDITY_STRENGTH",
    ] += 2

    result.loc[
        bearish_mask & bearish_large_wick,
        "BEARISH_LIQUIDITY_STRENGTH",
    ] += 2

    if "BULLISH_CHOCH" in result.columns:
        result.loc[
            bullish_mask & result["BULLISH_CHOCH"],
            "BULLISH_LIQUIDITY_STRENGTH",
        ] += 2

    if "BEARISH_CHOCH" in result.columns:
        result.loc[
            bearish_mask & result["BEARISH_CHOCH"],
            "BEARISH_LIQUIDITY_STRENGTH",
        ] += 2

    result["BULLISH_LIQUIDITY_STRENGTH"] = (
        result["BULLISH_LIQUIDITY_STRENGTH"]
        .clip(0, 10)
        .astype(int)
    )

    result["BEARISH_LIQUIDITY_STRENGTH"] = (
        result["BEARISH_LIQUIDITY_STRENGTH"]
        .clip(0, 10)
        .astype(int)
    )

    return result


def add_recent_liquidity_flags(
    df: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    result = df.copy()

    result["BULLISH_LIQUIDITY_RECENT"] = (
        result["BULLISH_LIQUIDITY_SWEEP"]
        .rolling(window=window, min_periods=1)
        .max()
        .astype(bool)
    )

    result["BEARISH_LIQUIDITY_RECENT"] = (
        result["BEARISH_LIQUIDITY_SWEEP"]
        .rolling(window=window, min_periods=1)
        .max()
        .astype(bool)
    )

    return result


def latest_liquidity_summary(
    df: pd.DataFrame,
) -> dict:
    validate_ohlc(df)

    latest = df.iloc[-1]

    return {
        "bullish_sweep": bool(
            latest.get("BULLISH_LIQUIDITY_SWEEP", False)
        ),
        "bearish_sweep": bool(
            latest.get("BEARISH_LIQUIDITY_SWEEP", False)
        ),
        "bullish_stop_hunt": bool(
            latest.get("BULLISH_STOP_HUNT", False)
        ),
        "bearish_stop_hunt": bool(
            latest.get("BEARISH_STOP_HUNT", False)
        ),
        "bullish_recent": bool(
            latest.get("BULLISH_LIQUIDITY_RECENT", False)
        ),
        "bearish_recent": bool(
            latest.get("BEARISH_LIQUIDITY_RECENT", False)
        ),
        "bullish_strength": int(
            latest.get("BULLISH_LIQUIDITY_STRENGTH", 0)
        ),
        "bearish_strength": int(
            latest.get("BEARISH_LIQUIDITY_STRENGTH", 0)
        ),
        "prior_high": latest.get("PRIOR_HIGH"),
        "prior_low": latest.get("PRIOR_LOW"),
    }
