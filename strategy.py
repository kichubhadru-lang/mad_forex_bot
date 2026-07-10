from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from fvg import add_fair_value_gaps
from liquidity import add_liquidity_sweeps
from market_structure import add_market_structure
from orderblock import add_order_blocks
from scoring import calculate_setup_score
from session import add_trading_sessions
from trend import add_trend_state


PAIRS = {
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
    "NZDUSD": "NZDUSD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
}

MIN_LIVE_SCORE = 90
ATR_STOP_MULTIPLIER = 1.5
TP1_RR = 2.0
TP2_RR = 3.0


def clean_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    if not all(column in df.columns for column in required):
        return None

    result = df[required].copy().dropna()

    if not isinstance(result.index, pd.DatetimeIndex):
        return None

    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")

    return None if result.empty else result


def download_hourly_data(
    ticker: str,
    period: str = "60d",
) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            ticker,
            interval="1h",
            period=period,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        return clean_data(df)

    except Exception as error:
        print(f"Download error for {ticker}: {error}")
        return None


def resample_ohlc(
    df: pd.DataFrame,
    rule: str,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    result = df.resample(
        rule,
        label="right",
        closed="right",
    ).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
    )

    return clean_data(result)


def apply_full_analysis(
    df: pd.DataFrame,
    include_session: bool = False,
) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 220:
        return None

    try:
        result = add_trend_state(df)
        result = add_market_structure(result)
        result = add_order_blocks(result)
        result = add_fair_value_gaps(result)
        result = add_liquidity_sweeps(result)

        if include_session:
            result = add_trading_sessions(result)

        result = result.dropna(
            subset=[
                "EMA20",
                "EMA50",
                "EMA200",
                "RSI",
                "ADX",
                "ATR",
            ]
        )

        return None if result.empty else result

    except Exception as error:
        print("Analysis pipeline error:", error)
        return None


def latest_completed_row(df: pd.DataFrame) -> pd.Series:
    if len(df) >= 2:
        return df.iloc[-2]

    return df.iloc[-1]


def calculate_trade_levels(
    action: str,
    entry: float,
    atr: float,
) -> Dict:
    risk_distance = atr * ATR_STOP_MULTIPLIER

    if action == "BUY":
        stop_loss = entry - risk_distance
        tp1 = entry + risk_distance * TP1_RR
        tp2 = entry + risk_distance * TP2_RR
    else:
        stop_loss = entry + risk_distance
        tp1 = entry - risk_distance * TP1_RR
        tp2 = entry - risk_distance * TP2_RR

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "risk_distance": risk_distance,
        "rr_tp1": TP1_RR,
        "rr_tp2": TP2_RR,
    }


def analyze_pair(
    pair: str,
    ticker: str,
) -> Optional[Dict]:
    hourly_raw = download_hourly_data(ticker)

    if hourly_raw is None or len(hourly_raw) < 500:
        print(pair, "- insufficient hourly data")
        return None

    h4_raw = resample_ohlc(hourly_raw, "4h")
    daily_raw = resample_ohlc(hourly_raw, "1D")
    weekly_raw = resample_ohlc(hourly_raw, "1W")

    h1 = apply_full_analysis(
        hourly_raw,
        include_session=True,
    )

    h4 = apply_full_analysis(h4_raw)
    daily = apply_full_analysis(daily_raw)
    weekly = apply_full_analysis(weekly_raw)

    if any(
        timeframe is None
        for timeframe in [h1, h4, daily, weekly]
    ):
        print(pair, "- timeframe analysis failed")
        return None

    h1_row = latest_completed_row(h1)
    h4_row = latest_completed_row(h4)
    daily_row = latest_completed_row(daily)
    weekly_row = latest_completed_row(weekly)

    weekly_trend = str(
        weekly_row.get("TREND_DIRECTION", "NEUTRAL")
    )

    daily_trend = str(
        daily_row.get("TREND_DIRECTION", "NEUTRAL")
    )

    score_result = calculate_setup_score(
        weekly_trend=weekly_trend,
        daily_trend=daily_trend,
        h4_row=h4_row,
        h1_row=h1_row,
    )

    action = score_result["action"]
    score = int(score_result["score"])

    entry = float(h1_row["Close"])
    atr = float(h4_row["ATR"])

    if atr <= 0:
        return None

    levels = calculate_trade_levels(
        action=action,
        entry=entry,
        atr=atr,
    )

    decimals = 2 if pair == "XAUUSD" else 5

    result = {
        "pair": pair,
        "ticker": ticker,
        "action": action,
        "score": score,
        "confidence": score_result["confidence"],
        "grade": score_result["grade"],
        "qualified": score >= MIN_LIVE_SCORE,
        "weekly_trend": weekly_trend,
        "daily_trend": daily_trend,
        "h4_trend": h4_row.get(
            "TREND_DIRECTION",
            "NEUTRAL",
        ),
        "session": h1_row.get(
            "SESSION_NAME",
            "UNKNOWN",
        ),
        "entry": round(levels["entry"], decimals),
        "stop_loss": round(
            levels["stop_loss"],
            decimals,
        ),
        "tp1": round(levels["tp1"], decimals),
        "tp2": round(levels["tp2"], decimals),
        "rr_tp1": levels["rr_tp1"],
        "rr_tp2": levels["rr_tp2"],
        "adx": round(float(h4_row["ADX"]), 2),
        "rsi": round(float(h4_row["RSI"]), 2),
        "atr_expansion": bool(
            h4_row.get("ATR_EXPANSION", False)
        ),
        "bullish_bos": bool(
            h4_row.get("BULLISH_BOS", False)
        ),
        "bearish_bos": bool(
            h4_row.get("BEARISH_BOS", False)
        ),
        "bullish_choch": bool(
            h4_row.get("BULLISH_CHOCH", False)
        ),
        "bearish_choch": bool(
            h4_row.get("BEARISH_CHOCH", False)
        ),
        "reasons": score_result["reasons"],
    }

    return result


def scan_all_pairs() -> list[Dict]:
    results = []

    for pair, ticker in PAIRS.items():
        print("Scanning", pair)

        result = analyze_pair(
            pair=pair,
            ticker=ticker,
        )

        if result is not None:
            results.append(result)

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results


def get_qualified_signals(
    results: list[Dict],
) -> list[Dict]:
    return [
        result
        for result in results
        if result["qualified"]
    ]
