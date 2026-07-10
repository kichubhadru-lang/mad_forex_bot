import numpy as np
import pandas as pd


def add_swings(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
) -> pd.DataFrame:
    """
    Detect confirmed swing highs and swing lows.

    A swing is confirmed only after right_bars candles,
    which helps avoid look-ahead bias in backtesting.
    """
    result = df.copy()

    window = left_bars + right_bars + 1

    rolling_high = result["High"].rolling(
        window=window,
        center=True,
    ).max()

    rolling_low = result["Low"].rolling(
        window=window,
        center=True,
    ).min()

    raw_swing_high = result["High"].eq(rolling_high)
    raw_swing_low = result["Low"].eq(rolling_low)

    # Move confirmation forward by right_bars candles.
    result["SWING_HIGH"] = (
        result["High"]
        .where(raw_swing_high)
        .shift(right_bars)
    )

    result["SWING_LOW"] = (
        result["Low"]
        .where(raw_swing_low)
        .shift(right_bars)
    )

    result["LAST_SWING_HIGH"] = result["SWING_HIGH"].ffill()
    result["LAST_SWING_LOW"] = result["SWING_LOW"].ffill()

    return result


def add_structure_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect bullish/bearish Break of Structure and CHoCH.
    """
    result = df.copy()

    previous_swing_high = result["LAST_SWING_HIGH"].shift(1)
    previous_swing_low = result["LAST_SWING_LOW"].shift(1)

    result["BULLISH_BOS"] = (
        result["Close"] > previous_swing_high
    ) & (
        result["Close"].shift(1) <= previous_swing_high
    )

    result["BEARISH_BOS"] = (
        result["Close"] < previous_swing_low
    ) & (
        result["Close"].shift(1) >= previous_swing_low
    )

    structure = []
    current_structure = 0

    for bullish_bos, bearish_bos in zip(
        result["BULLISH_BOS"],
        result["BEARISH_BOS"],
    ):
        if bullish_bos:
            current_structure = 1
        elif bearish_bos:
            current_structure = -1

        structure.append(current_structure)

    result["STRUCTURE"] = structure

    previous_structure = result["STRUCTURE"].shift(1).fillna(0)

    result["BULLISH_CHOCH"] = (
        result["BULLISH_BOS"]
        & (previous_structure == -1)
    )

    result["BEARISH_CHOCH"] = (
        result["BEARISH_BOS"]
        & (previous_structure == 1)
    )

    return result


def add_market_structure(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
) -> pd.DataFrame:
    """
    Full structure pipeline.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    required = {"Open", "High", "Low", "Close"}

    if not required.issubset(df.columns):
        raise ValueError(
            "DataFrame must contain Open, High, Low and Close columns."
        )

    result = add_swings(
        df=df,
        left_bars=left_bars,
        right_bars=right_bars,
    )

    result = add_structure_breaks(result)

    result["TREND_LABEL"] = np.select(
        [
            result["STRUCTURE"] == 1,
            result["STRUCTURE"] == -1,
        ],
        [
            "BULLISH",
            "BEARISH",
        ],
        default="NEUTRAL",
    )

    return result
