from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd


UTC = ZoneInfo("UTC")
LONDON_TZ = ZoneInfo("Europe/London")
NEW_YORK_TZ = ZoneInfo("America/New_York")
TOKYO_TZ = ZoneInfo("Asia/Tokyo")


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("DataFrame is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    result = df.copy()

    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")

    return result


def is_between(
    value: time,
    start: time,
    end: time,
) -> bool:
    if start <= end:
        return start <= value < end

    return value >= start or value < end


def add_trading_sessions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds DST-aware Forex session information.

    Main windows:
    Asian: 08:00–17:00 Tokyo time
    London: 08:00–17:00 London time
    New York: 08:00–17:00 New York time
    """

    result = ensure_utc_index(df)

    london_times = result.index.tz_convert(LONDON_TZ)
    new_york_times = result.index.tz_convert(NEW_YORK_TZ)
    tokyo_times = result.index.tz_convert(TOKYO_TZ)

    result["ASIAN_SESSION"] = [
        is_between(timestamp.time(), time(8, 0), time(17, 0))
        for timestamp in tokyo_times
    ]

    result["LONDON_SESSION"] = [
        is_between(timestamp.time(), time(8, 0), time(17, 0))
        for timestamp in london_times
    ]

    result["NEW_YORK_SESSION"] = [
        is_between(timestamp.time(), time(8, 0), time(17, 0))
        for timestamp in new_york_times
    ]

    result["LONDON_NEW_YORK_OVERLAP"] = (
        result["LONDON_SESSION"]
        & result["NEW_YORK_SESSION"]
    )

    result["LONDON_KILL_ZONE"] = [
        is_between(timestamp.time(), time(7, 0), time(10, 0))
        for timestamp in london_times
    ]

    result["NEW_YORK_KILL_ZONE"] = [
        is_between(timestamp.time(), time(8, 0), time(11, 0))
        for timestamp in new_york_times
    ]

    result["HIGH_LIQUIDITY_SESSION"] = (
        result["LONDON_SESSION"]
        | result["NEW_YORK_SESSION"]
    )

    result["PREFERRED_ENTRY_WINDOW"] = (
        result["LONDON_KILL_ZONE"]
        | result["NEW_YORK_KILL_ZONE"]
        | result["LONDON_NEW_YORK_OVERLAP"]
    )

    result["SESSION_NAME"] = "OFF_SESSION"

    result.loc[
        result["ASIAN_SESSION"],
        "SESSION_NAME",
    ] = "ASIAN"

    result.loc[
        result["LONDON_SESSION"],
        "SESSION_NAME",
    ] = "LONDON"

    result.loc[
        result["NEW_YORK_SESSION"],
        "SESSION_NAME",
    ] = "NEW_YORK"

    result.loc[
        result["LONDON_NEW_YORK_OVERLAP"],
        "SESSION_NAME",
    ] = "LONDON_NEW_YORK_OVERLAP"

    result["SESSION_SCORE"] = 0

    result.loc[
        result["ASIAN_SESSION"],
        "SESSION_SCORE",
    ] = 2

    result.loc[
        result["LONDON_SESSION"],
        "SESSION_SCORE",
    ] = 5

    result.loc[
        result["NEW_YORK_SESSION"],
        "SESSION_SCORE",
    ] = 5

    result.loc[
        result["PREFERRED_ENTRY_WINDOW"],
        "SESSION_SCORE",
    ] = 8

    result.loc[
        result["LONDON_NEW_YORK_OVERLAP"],
        "SESSION_SCORE",
    ] = 10

    return result


def latest_session_summary(
    df: pd.DataFrame,
) -> dict:
    if df is None or df.empty:
        return {}

    latest = df.iloc[-1]

    return {
        "session": latest.get("SESSION_NAME", "UNKNOWN"),
        "asian": bool(latest.get("ASIAN_SESSION", False)),
        "london": bool(latest.get("LONDON_SESSION", False)),
        "new_york": bool(
            latest.get("NEW_YORK_SESSION", False)
        ),
        "overlap": bool(
            latest.get("LONDON_NEW_YORK_OVERLAP", False)
        ),
        "preferred_entry_window": bool(
            latest.get("PREFERRED_ENTRY_WINDOW", False)
        ),
        "session_score": int(
            latest.get("SESSION_SCORE", 0)
        ),
    }
