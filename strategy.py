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


def clean_data(
    df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    if not all(column in df.columns for column in required):
        return None

    result = df[required].copy()
    result = result.dropna()

    if not isinstance(result.index, pd.DatetimeIndex):
        return None

    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")

    return None if result.empty else result


def download_data(
    ticker: str,
    interval: str,
    period: str,
) -> Optional[pd.DataFrame]:
    try:
        print(
            f"Downloading {ticker}: "
            f"interval={interval}, period={period}"
        )

        df = yf.download(
            ticker,
            interval=interval,
            period=period,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        cleaned = clean_data(df)

        if cleaned is None:
            print(
                f"No usable data for {ticker} "
                f"{interval}"
            )
            return None

        print(
            f"{ticker} {interval} rows:",
            len(cleaned),
        )

        return cleaned

    except Exception as error:
        print(
            f"Download error for {ticker} "
            f"{interval}: {error}"
        )
        return None


def resample_ohlc(
    df: pd.DataFrame,
    rule: str,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    try:
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

    except Exception as error:
        print(
            f"Resample error for {rule}:",
            error,
        )
        return None


def apply_full_analysis(
    df: pd.DataFrame,
    include_session: bool = False,
) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    if len(df) < 220:
        print(
            "Insufficient rows for EMA200:",
            len(df),
        )
        return None

    try:
        result = add_trend_state(df)

        result = add_market_structure(
            result,
            left_bars=3,
            right_bars=3,
        )

        result = add_order_blocks(
            result,
            atr_column="ATR",
            displacement_multiplier=1.0,
            lookahead_bars=3,
        )

        result = add_fair_value_gaps(
            result,
            min_gap_atr=0.10,
        )

        result = add_liquidity_sweeps(
            result,
            lookback=20,
            wick_ratio=1.5,
        )

        if include_session:
            result = add_trading_sessions(result)

        required_indicator_columns = [
            "EMA20",
            "EMA50",
            "EMA200",
            "RSI",
            "ADX",
            "ATR",
        ]

        result = result.dropna(
            subset=required_indicator_columns
        )

        if result.empty:
            return None

        return result

    except Exception as error:
        print(
            "Analysis pipeline error:",
            type(error).__name__,
            error,
        )
        return None


def latest_completed_row(
    df: pd.DataFrame,
) -> pd.Series:
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
    print("\n" + "=" * 60)
    print("Analyzing", pair)
    print("=" * 60)

    # 1H data for entry analysis
    hourly_raw = download_data(
        ticker=ticker,
        interval="1h",
        period="60d",
    )

    if hourly_raw is None:
        print(pair, "- hourly data missing")
        return None

    # Build 4H candles from 1H data
    h4_raw = resample_ohlc(
        hourly_raw,
        "4h",
    )

    # Separate daily download gives enough history
    daily_raw = download_data(
        ticker=ticker,
        interval="1d",
        period="2y",
    )

    # Separate weekly download gives enough history
    weekly_raw = download_data(
        ticker=ticker,
        interval="1wk",
        period="5y",
    )

    if any(
        timeframe is None
        for timeframe in [
            h4_raw,
            daily_raw,
            weekly_raw,
        ]
    ):
        print(pair, "- one or more timeframes missing")
        return None

    print(
        pair,
        "raw rows:",
        {
            "1H": len(hourly_raw),
            "4H": len(h4_raw),
            "Daily": len(daily_raw),
            "Weekly": len(weekly_raw),
        },
    )

    h1 = apply_full_analysis(
        hourly_raw,
        include_session=True,
    )

    h4 = apply_full_analysis(
        h4_raw,
        include_session=False,
    )

    daily = apply_full_analysis(
        daily_raw,
        include_session=False,
    )

    weekly = apply_full_analysis(
        weekly_raw,
        include_session=False,
    )

    if h1 is None:
        print(pair, "- 1H analysis failed")
        return None

    if h4 is None:
        print(pair, "- 4H analysis failed")
        return None

    if daily is None:
        print(pair, "- daily analysis failed")
        return None

    if weekly is None:
        print(pair, "- weekly analysis failed")
        return None

    h1_row = latest_completed_row(h1)
    h4_row = latest_completed_row(h4)
    daily_row = latest_completed_row(daily)
    weekly_row = latest_completed_row(weekly)

    weekly_trend = str(
        weekly_row.get(
            "TREND_DIRECTION",
            "NEUTRAL",
        )
    )

    daily_trend = str(
        daily_row.get(
            "TREND_DIRECTION",
            "NEUTRAL",
        )
    )

    score_result = calculate_setup_score(
        weekly_trend=weekly_trend,
        daily_trend=daily_trend,
        h4_row=h4_row,
        h1_row=h1_row,
    )

    action = str(score_result["action"])
    score = int(score_result["score"])

    entry = float(h1_row["Close"])
    atr = float(h4_row["ATR"])

    if pd.isna(entry) or pd.isna(atr):
        print(pair, "- invalid entry or ATR")
        return None

    if atr <= 0:
        print(pair, "- ATR is zero or negative")
        return None

    levels = calculate_trade_levels(
        action=action,
        entry=entry,
        atr=atr,
    )

    if pair == "XAUUSD":
        decimals = 2
    elif "JPY" in pair:
        decimals = 3
    else:
        decimals = 5

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
        "h4_trend": str(
            h4_row.get(
                "TREND_DIRECTION",
                "NEUTRAL",
            )
        ),

        "session": str(
            h1_row.get(
                "SESSION_NAME",
                "UNKNOWN",
            )
        ),

        "entry": round(
            levels["entry"],
            decimals,
        ),

        "stop_loss": round(
            levels["stop_loss"],
            decimals,
        ),

        "tp1": round(
            levels["tp1"],
            decimals,
        ),

        "tp2": round(
            levels["tp2"],
            decimals,
        ),

        "rr_tp1": levels["rr_tp1"],
        "rr_tp2": levels["rr_tp2"],

        "adx": round(
            float(h4_row.get("ADX", 0)),
            2,
        ),

        "rsi": round(
            float(h4_row.get("RSI", 0)),
            2,
        ),

        "atr_expansion": bool(
            h4_row.get(
                "ATR_EXPANSION",
                False,
            )
        ),

        "bullish_bos": bool(
            h4_row.get(
                "BULLISH_BOS",
                False,
            )
        ),

        "bearish_bos": bool(
            h4_row.get(
                "BEARISH_BOS",
                False,
            )
        ),

        "bullish_choch": bool(
            h4_row.get(
                "BULLISH_CHOCH",
                False,
            )
        ),

        "bearish_choch": bool(
            h4_row.get(
                "BEARISH_CHOCH",
                False,
            )
        ),

        "reasons": score_result["reasons"],
    }

    print(
        pair,
        "result:",
        action,
        score,
        score_result["grade"],
    )

    return result


def scan_all_pairs() -> list[Dict]:
    results: list[Dict] = []

    for pair, ticker in PAIRS.items():
        try:
            result = analyze_pair(
                pair=pair,
                ticker=ticker,
            )

            if result is not None:
                results.append(result)

        except Exception as error:
            print(
                pair,
                "unexpected error:",
                type(error).__name__,
                error,
            )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    print(
        "\nPairs successfully analyzed:",
        len(results),
    )

    return results


def get_qualified_signals(
    results: list[Dict],
) -> list[Dict]:
    qualified = [
        result
        for result in results
        if result["qualified"]
    ]

    qualified.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return qualified
