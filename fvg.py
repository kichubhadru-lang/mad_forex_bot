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


def add_fair_value_gaps(
    df: pd.DataFrame,
    min_gap_atr: float = 0.10,
) -> pd.DataFrame:
    """
    Detect three-candle Fair Value Gaps.

    Bullish FVG:
    Current low is above the high from two candles earlier.

    Bearish FVG:
    Current high is below the low from two candles earlier.
    """
    validate_ohlc(df)

    result = df.copy()

    bullish_gap_size = (
        result["Low"] - result["High"].shift(2)
    )

    bearish_gap_size = (
        result["Low"].shift(2) - result["High"]
    )

    if "ATR" in result.columns:
        minimum_gap = result["ATR"] * min_gap_atr
    else:
        average_range = (
            result["High"] - result["Low"]
        ).rolling(20).mean()

        minimum_gap = average_range * min_gap_atr

    result["BULLISH_FVG"] = (
        bullish_gap_size > minimum_gap
    )

    result["BEARISH_FVG"] = (
        bearish_gap_size > minimum_gap
    )

    result["BULLISH_FVG_LOW"] = (
        result["High"].shift(2)
        .where(result["BULLISH_FVG"])
    )

    result["BULLISH_FVG_HIGH"] = (
        result["Low"]
        .where(result["BULLISH_FVG"])
    )

    result["BEARISH_FVG_LOW"] = (
        result["High"]
        .where(result["BEARISH_FVG"])
    )

    result["BEARISH_FVG_HIGH"] = (
        result["Low"].shift(2)
        .where(result["BEARISH_FVG"])
    )

    result = track_latest_fvg_zones(result)
    result = add_fvg_fill_status(result)
    result = add_fvg_strength(result)

    return result


def track_latest_fvg_zones(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["LAST_BULLISH_FVG_LOW"] = (
        result["BULLISH_FVG_LOW"].ffill()
    )

    result["LAST_BULLISH_FVG_HIGH"] = (
        result["BULLISH_FVG_HIGH"].ffill()
    )

    result["LAST_BEARISH_FVG_LOW"] = (
        result["BEARISH_FVG_LOW"].ffill()
    )

    result["LAST_BEARISH_FVG_HIGH"] = (
        result["BEARISH_FVG_HIGH"].ffill()
    )

    return result


def add_fvg_fill_status(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect whether price has returned into the latest FVG zone.
    """
    result = df.copy()

    bullish_zone_valid = (
        result["LAST_BULLISH_FVG_LOW"].notna()
        & result["LAST_BULLISH_FVG_HIGH"].notna()
    )

    bearish_zone_valid = (
        result["LAST_BEARISH_FVG_LOW"].notna()
        & result["LAST_BEARISH_FVG_HIGH"].notna()
    )

    result["BULLISH_FVG_TOUCHED"] = (
        bullish_zone_valid
        & (
            result["Low"]
            <= result["LAST_BULLISH_FVG_HIGH"]
        )
        & (
            result["High"]
            >= result["LAST_BULLISH_FVG_LOW"]
        )
    )

    result["BEARISH_FVG_TOUCHED"] = (
        bearish_zone_valid
        & (
            result["High"]
            >= result["LAST_BEARISH_FVG_LOW"]
        )
        & (
            result["Low"]
            <= result["LAST_BEARISH_FVG_HIGH"]
        )
    )

    result["BULLISH_FVG_REJECTED"] = (
        result["BULLISH_FVG_TOUCHED"]
        & (
            result["Close"]
            > result["LAST_BULLISH_FVG_HIGH"]
        )
        & (
            result["Close"]
            > result["Open"]
        )
    )

    result["BEARISH_FVG_REJECTED"] = (
        result["BEARISH_FVG_TOUCHED"]
        & (
            result["Close"]
            < result["LAST_BEARISH_FVG_LOW"]
        )
        & (
            result["Close"]
            < result["Open"]
        )
    )

    result["BULLISH_FVG_FILLED"] = (
        bullish_zone_valid
        & (
            result["Low"]
            <= result["LAST_BULLISH_FVG_LOW"]
        )
    )

    result["BEARISH_FVG_FILLED"] = (
        bearish_zone_valid
        & (
            result["High"]
            >= result["LAST_BEARISH_FVG_HIGH"]
        )
    )

    return result


def add_fvg_strength(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a simple 0–10 FVG quality score.
    """
    result = df.copy()

    result["BULLISH_FVG_STRENGTH"] = 0
    result["BEARISH_FVG_STRENGTH"] = 0

    bullish_mask = result["BULLISH_FVG"]
    bearish_mask = result["BEARISH_FVG"]

    result.loc[
        bullish_mask,
        "BULLISH_FVG_STRENGTH",
    ] += 4

    result.loc[
        bearish_mask,
        "BEARISH_FVG_STRENGTH",
    ] += 4

    if "ATR" in result.columns:
        bullish_gap = (
            result["BULLISH_FVG_HIGH"]
            - result["BULLISH_FVG_LOW"]
        )

        bearish_gap = (
            result["BEARISH_FVG_HIGH"]
            - result["BEARISH_FVG_LOW"]
        )

        result.loc[
            bullish_mask
            & (bullish_gap >= result["ATR"] * 0.25),
            "BULLISH_FVG_STRENGTH",
        ] += 2

        result.loc[
            bearish_mask
            & (bearish_gap >= result["ATR"] * 0.25),
            "BEARISH_FVG_STRENGTH",
        ] += 2

    if "BULLISH_BOS" in result.columns:
        result.loc[
            bullish_mask & result["BULLISH_BOS"],
            "BULLISH_FVG_STRENGTH",
        ] += 2

    if "BEARISH_BOS" in result.columns:
        result.loc[
            bearish_mask & result["BEARISH_BOS"],
            "BEARISH_FVG_STRENGTH",
        ] += 2

    result.loc[
        result["BULLISH_FVG_REJECTED"],
        "BULLISH_FVG_STRENGTH",
    ] += 2

    result.loc[
        result["BEARISH_FVG_REJECTED"],
        "BEARISH_FVG_STRENGTH",
    ] += 2

    result["BULLISH_FVG_STRENGTH"] = (
        result["BULLISH_FVG_STRENGTH"]
        .clip(0, 10)
        .astype(int)
    )

    result["BEARISH_FVG_STRENGTH"] = (
        result["BEARISH_FVG_STRENGTH"]
        .clip(0, 10)
        .astype(int)
    )

    return result


def latest_fvg_summary(
    df: pd.DataFrame,
) -> dict:
    validate_ohlc(df)

    latest = df.iloc[-1]

    return {
        "bullish_fvg": bool(
            latest.get("BULLISH_FVG", False)
        ),
        "bearish_fvg": bool(
            latest.get("BEARISH_FVG", False)
        ),
        "bullish_touched": bool(
            latest.get("BULLISH_FVG_TOUCHED", False)
        ),
        "bearish_touched": bool(
            latest.get("BEARISH_FVG_TOUCHED", False)
        ),
        "bullish_rejected": bool(
            latest.get("BULLISH_FVG_REJECTED", False)
        ),
        "bearish_rejected": bool(
            latest.get("BEARISH_FVG_REJECTED", False)
        ),
        "bullish_filled": bool(
            latest.get("BULLISH_FVG_FILLED", False)
        ),
        "bearish_filled": bool(
            latest.get("BEARISH_FVG_FILLED", False)
        ),
        "bullish_strength": int(
            latest.get("BULLISH_FVG_STRENGTH", 0)
        ),
        "bearish_strength": int(
            latest.get("BEARISH_FVG_STRENGTH", 0)
        ),
        "bullish_zone_low": latest.get(
            "LAST_BULLISH_FVG_LOW"
        ),
        "bullish_zone_high": latest.get(
            "LAST_BULLISH_FVG_HIGH"
        ),
        "bearish_zone_low": latest.get(
            "LAST_BEARISH_FVG_LOW"
        ),
        "bearish_zone_high": latest.get(
            "LAST_BEARISH_FVG_HIGH"
        ),
    }
