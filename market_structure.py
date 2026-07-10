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
