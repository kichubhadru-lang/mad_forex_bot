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


def add_order_blocks(
    df: pd.DataFrame,
    atr_column: str = "ATR",
    displacement_multiplier: float = 1.0,
    lookahead_bars: int = 3,
) -> pd.DataFrame:
    """
    Detect bullish and bearish order-block candidates.

    Bullish order block:
    - Previous candle is bearish.
    - Current candle is bullish.
    - Current candle body shows strong displacement.
    - Current close breaks above the previous candle high.

    Bearish order block:
    - Previous candle is bullish.
    - Current candle is bearish.
    - Current candle body shows strong displacement.
    - Current close breaks below the previous candle low.
    """
    validate_ohlc(df)

    result = df.copy()

    body = (result["Close"] - result["Open"]).abs()

    if atr_column in result.columns:
        displacement_threshold = (
            result[atr_column] * displacement_multiplier
        )
    else:
        average_body = body.rolling(20).mean()
        displacement_threshold = (
            average_body * displacement_multiplier
        )

    previous_bearish = (
        result["Close"].shift(1)
        < result["Open"].shift(1)
    )

    previous_bullish = (
        result["Close"].shift(1)
        > result["Open"].shift(1)
    )

    current_bullish = (
        result["Close"]
        > result["Open"]
    )

    current_bearish = (
        result["Close"]
        < result["Open"]
    )

    strong_displacement = (
        body >= displacement_threshold
    )

    bullish_break = (
        result["Close"]
        > result["High"].shift(1)
    )

    bearish_break = (
        result["Close"]
        < result["Low"].shift(1)
    )

    result["BULLISH_OB"] = (
        previous_bearish
        & current_bullish
        & strong_displacement
        & bullish_break
    )

    result["BEARISH_OB"] = (
        previous_bullish
        & current_bearish
        & strong_displacement
        & bearish_break
    )

    result["BULLISH_OB_LOW"] = result["Low"].shift(1).where(
        result["BULLISH_OB"]
    )

    result["BULLISH_OB_HIGH"] = result["High"].shift(1).where(
        result["BULLISH_OB"]
    )

    result["BEARISH_OB_LOW"] = result["Low"].shift(1).where(
        result["BEARISH_OB"]
    )

    result["BEARISH_OB_HIGH"] = result["High"].shift(1).where(
        result["BEARISH_OB"]
    )

    result["LAST_BULLISH_OB_LOW"] = (
        result["BULLISH_OB_LOW"].ffill()
    )

    result["LAST_BULLISH_OB_HIGH"] = (
        result["BULLISH_OB_HIGH"].ffill()
    )

    result["LAST_BEARISH_OB_LOW"] = (
        result["BEARISH_OB_LOW"].ffill()
    )

    result["LAST_BEARISH_OB_HIGH"] = (
        result["BEARISH_OB_HIGH"].ffill()
    )

    result = add_order_block_retests(
        result,
        lookahead_bars=lookahead_bars,
    )

    result = add_order_block_strength(result)

    return result


def add_order_block_retests(
    df: pd.DataFrame,
    lookahead_bars: int = 3,
) -> pd.DataFrame:
    """
    Detect whether price returns to the most recent order-block zone.

    This uses current and prior known zones only.
    """
    result = df.copy()

    bullish_zone_valid = (
        result["LAST_BULLISH_OB_LOW"].notna()
        & result["LAST_BULLISH_OB_HIGH"].notna()
    )

    bearish_zone_valid = (
        result["LAST_BEARISH_OB_LOW"].notna()
        & result["LAST_BEARISH_OB_HIGH"].notna()
    )

    result["BULLISH_OB_RETEST"] = (
        bullish_zone_valid
        & (
            result["Low"]
            <= result["LAST_BULLISH_OB_HIGH"]
        )
        & (
            result["High"]
            >= result["LAST_BULLISH_OB_LOW"]
        )
        & (
            result["Close"]
            > result["LAST_BULLISH_OB_HIGH"]
        )
    )

    result["BEARISH_OB_RETEST"] = (
        bearish_zone_valid
        & (
            result["High"]
            >= result["LAST_BEARISH_OB_LOW"]
        )
        & (
            result["Low"]
            <= result["LAST_BEARISH_OB_HIGH"]
        )
        & (
            result["Close"]
            < result["LAST_BEARISH_OB_LOW"]
        )
    )

    result["BULLISH_OB_RETEST_RECENT"] = (
        result["BULLISH_OB_RETEST"]
        .rolling(
            window=max(1, lookahead_bars),
            min_periods=1,
        )
        .max()
        .astype(bool)
    )

    result["BEARISH_OB_RETEST_RECENT"] = (
        result["BEARISH_OB_RETEST"]
        .rolling(
            window=max(1, lookahead_bars),
            min_periods=1,
        )
        .max()
        .astype(bool)
    )

    return result


def add_order_block_strength(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a simple 0–10 strength score for order blocks.
    """
    result = df.copy()

    result["BULLISH_OB_STRENGTH"] = 0
    result["BEARISH_OB_STRENGTH"] = 0

    bullish_mask = result["BULLISH_OB"]
    bearish_mask = result["BEARISH_OB"]

    result.loc[
        bullish_mask,
        "BULLISH_OB_STRENGTH",
    ] += 4

    result.loc[
        bearish_mask,
        "BEARISH_OB_STRENGTH",
    ] += 4

    if "ATR" in result.columns:
        bullish_large_body = (
            bullish_mask
            & (
                (result["Close"] - result["Open"]).abs()
                >= result["ATR"] * 1.25
            )
        )

        bearish_large_body = (
            bearish_mask
            & (
                (result["Close"] - result["Open"]).abs()
                >= result["ATR"] * 1.25
            )
        )

        result.loc[
            bullish_large_body,
            "BULLISH_OB_STRENGTH",
        ] += 2

        result.loc[
            bearish_large_body,
            "BEARISH_OB_STRENGTH",
        ] += 2

    if "BULLISH_BOS" in result.columns:
        result.loc[
            bullish_mask & result["BULLISH_BOS"],
            "BULLISH_OB_STRENGTH",
        ] += 2

    if "BEARISH_BOS" in result.columns:
        result.loc[
            bearish_mask & result["BEARISH_BOS"],
            "BEARISH_OB_STRENGTH",
        ] += 2

    result.loc[
        result["BULLISH_OB_RETEST"],
        "BULLISH_OB_STRENGTH",
    ] += 2

    result.loc[
        result["BEARISH_OB_RETEST"],
        "BEARISH_OB_STRENGTH",
    ] += 2

    result["BULLISH_OB_STRENGTH"] = (
        result["BULLISH_OB_STRENGTH"]
        .clip(0, 10)
        .astype(int)
    )

    result["BEARISH_OB_STRENGTH"] = (
        result["BEARISH_OB_STRENGTH"]
        .clip(0, 10)
        .astype(int)
    )

    return result


def latest_order_block_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Return the latest order-block state for debugging or scoring.
    """
    validate_ohlc(df)

    if df.empty:
        return {}

    latest = df.iloc[-1]

    return {
        "bullish_ob": bool(
            latest.get("BULLISH_OB", False)
        ),
        "bearish_ob": bool(
            latest.get("BEARISH_OB", False)
        ),
        "bullish_retest": bool(
            latest.get("BULLISH_OB_RETEST_RECENT", False)
        ),
        "bearish_retest": bool(
            latest.get("BEARISH_OB_RETEST_RECENT", False)
        ),
        "bullish_strength": int(
            latest.get("BULLISH_OB_STRENGTH", 0)
        ),
        "bearish_strength": int(
            latest.get("BEARISH_OB_STRENGTH", 0)
        ),
        "bullish_zone_low": latest.get(
            "LAST_BULLISH_OB_LOW"
        ),
        "bullish_zone_high": latest.get(
            "LAST_BULLISH_OB_HIGH"
        ),
        "bearish_zone_low": latest.get(
            "LAST_BEARISH_OB_LOW"
        ),
        "bearish_zone_high": latest.get(
            "LAST_BEARISH_OB_HIGH"
        ),
    }
