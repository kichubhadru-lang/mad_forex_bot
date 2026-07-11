from collections import Counter

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from fvg import add_fair_value_gaps
from liquidity import add_liquidity_sweeps
from market_structure import add_market_structure
from orderblock import add_order_blocks
from scoring import calculate_setup_score
from session import add_trading_sessions
from trend import add_trend_state


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

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

REPORT_DIR = Path("backtest_reports")

MIN_SCORE = 75

STARTING_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.01
from collections import Counter

REJECTION_COUNTS = Counter()
PAIR_REJECTION_COUNTS = Counter()
MANDATORY_PASS_COUNT = 0
ATR_STOP_MULTIPLIER = 1.5
TP1_R_MULTIPLE = 2.0
TP2_R_MULTIPLE = 3.0

TP1_POSITION_FRACTION = 0.50
TP2_POSITION_FRACTION = 0.50

MAX_HOLDING_BARS = 120
TRADING_COST_R = 0.03

ALLOW_OVERLAPPING_TRADES = False


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(
    df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):
        return None

    result = df[required_columns].copy()
    result = result.dropna()

    if not isinstance(
        result.index,
        pd.DatetimeIndex,
    ):
        return None

    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")

    result = result[
        ~result.index.duplicated(keep="last")
    ]

    return None if result.empty else result


# ============================================================
# MARKET DATA DOWNLOAD
# ============================================================

def download_data(
    ticker: str,
    interval: str,
    period: str,
) -> Optional[pd.DataFrame]:
    try:
        print(
            f"Downloading {ticker} | "
            f"{interval} | {period}"
        )

        data = yf.download(
            ticker,
            interval=interval,
            period=period,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        cleaned = clean_data(data)

        if cleaned is None:
            print(
                f"No usable data: "
                f"{ticker} {interval}"
            )
            return None

        print(
            f"Rows received: {len(cleaned)}"
        )

        return cleaned

    except Exception as error:
        print(
            f"Download error: {ticker} "
            f"{interval}: {error}"
        )
        return None


# ============================================================
# RESAMPLING
# ============================================================

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
            f"Resample error {rule}:",
            error,
        )
        return None


# ============================================================
# FULL ANALYSIS PIPELINE
# ============================================================

def apply_full_analysis(
    df: pd.DataFrame,
    include_session: bool = False,
) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 220:
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

        required_columns = [
            "EMA20",
            "EMA50",
            "EMA200",
            "RSI",
            "ADX",
            "ATR",
        ]

        result = result.dropna(
            subset=required_columns
        )

        return None if result.empty else result

    except Exception as error:
        print(
            "Analysis pipeline error:",
            type(error).__name__,
            error,
        )
        return None


# ============================================================
# ALIGN HIGHER TIMEFRAMES TO 1H
# ============================================================

def align_timeframe(
    higher_df: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
    prefix: str,
) -> pd.DataFrame:
    """
    Shift one completed higher-timeframe candle before
    forward-filling onto 1H candles.

    This reduces look-ahead bias.
    """
    shifted = higher_df.shift(1).copy()

    shifted.columns = [
        f"{prefix}_{column}"
        for column in shifted.columns
    ]

    aligned = shifted.reindex(
        hourly_index,
        method="ffill",
    )

    return aligned


# ============================================================
# PREPARE ONE PAIR
# ============================================================

def prepare_pair_data(
    pair: str,
    ticker: str,
) -> Optional[pd.DataFrame]:
    print("\n" + "=" * 70)
    print("Preparing", pair)
    print("=" * 70)

    hourly_raw = download_data(
        ticker=ticker,
        interval="1h",
        period="730d",
    )

    daily_raw = download_data(
        ticker=ticker,
        interval="1d",
        period="5y",
    )

    weekly_raw = download_data(
        ticker=ticker,
        interval="1wk",
        period="10y",
    )

    if any(
        timeframe is None
        for timeframe in [
            hourly_raw,
            daily_raw,
            weekly_raw,
        ]
    ):
        print(pair, "- missing timeframe data")
        return None

    four_hour_raw = resample_ohlc(
        hourly_raw,
        "4h",
    )

    if four_hour_raw is None:
        print(pair, "- 4H resample failed")
        return None

    h1 = apply_full_analysis(
        hourly_raw,
        include_session=True,
    )

    h4 = apply_full_analysis(
        four_hour_raw,
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

    if any(
        timeframe is None
        for timeframe in [
            h1,
            h4,
            daily,
            weekly,
        ]
    ):
        print(pair, "- analysis failed")
        return None

    h4_aligned = align_timeframe(
        higher_df=h4,
        hourly_index=h1.index,
        prefix="H4",
    )

    daily_aligned = align_timeframe(
        higher_df=daily,
        hourly_index=h1.index,
        prefix="D1",
    )

    weekly_aligned = align_timeframe(
        higher_df=weekly,
        hourly_index=h1.index,
        prefix="W1",
    )

    combined = pd.concat(
        [
            h1,
            h4_aligned,
            daily_aligned,
            weekly_aligned,
        ],
        axis=1,
    )

    combined = combined.dropna()

    if combined.empty:
        print(pair, "- combined data empty")
        return None

    print(
        pair,
        "prepared rows:",
        len(combined),
    )

    return combined
  # ============================================================
# SIGNAL SCORING
# ============================================================

def get_trend_label(
    value: object,
) -> str:
    if value is None or pd.isna(value):
        return "NEUTRAL"

    return str(value)


def calculate_historical_signal(
    row: pd.Series,
) -> Dict:
    weekly_trend = get_trend_label(
        row.get("W1_TREND_DIRECTION")
    )

    daily_trend = get_trend_label(
        row.get("D1_TREND_DIRECTION")
    )

    h4_row = pd.Series(
        {
            column.replace("H4_", ""): value
            for column, value in row.items()
            if column.startswith("H4_")
        }
    )

    h1_row = pd.Series(
        {
            column: value
            for column, value in row.items()
            if not (
                column.startswith("H4_")
                or column.startswith("D1_")
                or column.startswith("W1_")
            )
        }
    )

    result = calculate_setup_score(
        weekly_trend=weekly_trend,
        daily_trend=daily_trend,
        h4_row=h4_row,
        h1_row=h1_row,
    )

    return result


# ============================================================
# TRADE LEVELS
# ============================================================

def calculate_trade_levels(
    direction: str,
    entry: float,
    atr: float,
) -> Optional[Dict]:
    if pd.isna(entry) or pd.isna(atr):
        return None

    if atr <= 0:
        return None

    risk_distance = (
        atr * ATR_STOP_MULTIPLIER
    )

    if direction == "BUY":
        stop_loss = (
            entry - risk_distance
        )

        tp1 = (
            entry
            + risk_distance
            * TP1_R_MULTIPLE
        )

        tp2 = (
            entry
            + risk_distance
            * TP2_R_MULTIPLE
        )

    elif direction == "SELL":
        stop_loss = (
            entry + risk_distance
        )

        tp1 = (
            entry
            - risk_distance
            * TP1_R_MULTIPLE
        )

        tp2 = (
            entry
            - risk_distance
            * TP2_R_MULTIPLE
        )

    else:
        return None

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "risk_distance": risk_distance,
    }


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df: pd.DataFrame,
    signal_bar: int,
    pair: str,
    direction: str,
    score: int,
    grade: str,
    confidence: str,
    reasons: List[str],
) -> Optional[Dict]:
    entry_bar = signal_bar + 1

    if entry_bar >= len(df):
        return None

    signal_row = df.iloc[signal_bar]
    entry_row = df.iloc[entry_bar]

    entry_price = float(
        entry_row["Open"]
    )

    atr = float(
        signal_row.get(
            "H4_ATR",
            np.nan,
        )
    )

    levels = calculate_trade_levels(
        direction=direction,
        entry=entry_price,
        atr=atr,
    )

    if levels is None:
        return None

    original_stop = float(
        levels["stop_loss"]
    )

    current_stop = original_stop

    tp1 = float(levels["tp1"])
    tp2 = float(levels["tp2"])

    risk_distance = float(
        levels["risk_distance"]
    )

    tp1_hit = False

    result_label = "TIME_EXIT"
    r_multiple = 0.0

    exit_price = entry_price
    exit_time = df.index[entry_bar]

    final_bar = min(
        entry_bar + MAX_HOLDING_BARS,
        len(df) - 1,
    )

    actual_exit_bar = final_bar

    for bar_index in range(
        entry_bar,
        final_bar + 1,
    ):
        bar = df.iloc[bar_index]

        high = float(bar["High"])
        low = float(bar["Low"])
        close = float(bar["Close"])

        if direction == "BUY":

            if not tp1_hit:
                stop_touched = (
                    low <= current_stop
                )

                tp1_touched = (
                    high >= tp1
                )

                # Conservative same-candle assumption:
                # stop is counted before target.
                if stop_touched:
                    result_label = "STOP_LOSS"
                    r_multiple = -1.0
                    exit_price = current_stop
                    exit_time = df.index[bar_index]
                    actual_exit_bar = bar_index
                    break

                if tp1_touched:
                    tp1_hit = True

                    r_multiple += (
                        TP1_POSITION_FRACTION
                        * TP1_R_MULTIPLE
                    )

                    current_stop = entry_price

                    # TP2 may also be touched
                    # during the same candle.
                    if high >= tp2:
                        r_multiple += (
                            TP2_POSITION_FRACTION
                            * TP2_R_MULTIPLE
                        )

                        result_label = "TP2"
                        exit_price = tp2
                        exit_time = df.index[bar_index]
                        actual_exit_bar = bar_index
                        break

            else:
                breakeven_touched = (
                    low <= current_stop
                )

                tp2_touched = (
                    high >= tp2
                )

                # Conservative ordering.
                if breakeven_touched:
                    result_label = "TP1_BREAKEVEN"
                    exit_price = current_stop
                    exit_time = df.index[bar_index]
                    actual_exit_bar = bar_index
                    break

                if tp2_touched:
                    r_multiple += (
                        TP2_POSITION_FRACTION
                        * TP2_R_MULTIPLE
                    )

                    result_label = "TP2"
                    exit_price = tp2
                    exit_time = df.index[bar_index]
                    actual_exit_bar = bar_index
                    break

        else:

            if not tp1_hit:
                stop_touched = (
                    high >= current_stop
                )

                tp1_touched = (
                    low <= tp1
                )

                if stop_touched:
                    result_label = "STOP_LOSS"
                    r_multiple = -1.0
                    exit_price = current_stop
                    exit_time = df.index[bar_index]
                    actual_exit_bar = bar_index
                    break

                if tp1_touched:
                    tp1_hit = True

                    r_multiple += (
                        TP1_POSITION_FRACTION
                        * TP1_R_MULTIPLE
                    )

                    current_stop = entry_price

                    if low <= tp2:
                        r_multiple += (
                            TP2_POSITION_FRACTION
                            * TP2_R_MULTIPLE
                        )

                        result_label = "TP2"
                        exit_price = tp2
                        exit_time = df.index[bar_index]
                        actual_exit_bar = bar_index
                        break

            else:
                breakeven_touched = (
                    high >= current_stop
                )

                tp2_touched = (
                    low <= tp2
                )

                if breakeven_touched:
                    result_label = "TP1_BREAKEVEN"
                    exit_price = current_stop
                    exit_time = df.index[bar_index]
                    actual_exit_bar = bar_index
                    break

                if tp2_touched:
                    r_multiple += (
                        TP2_POSITION_FRACTION
                        * TP2_R_MULTIPLE
                    )

                    result_label = "TP2"
                    exit_price = tp2
                    exit_time = df.index[bar_index]
                    actual_exit_bar = bar_index
                    break

    else:
        final_close = float(
            df.iloc[final_bar]["Close"]
        )

        if direction == "BUY":
            remaining_r = (
                final_close - entry_price
            ) / risk_distance
        else:
            remaining_r = (
                entry_price - final_close
            ) / risk_distance

        if tp1_hit:
            r_multiple += (
                TP2_POSITION_FRACTION
                * remaining_r
            )

            result_label = "TP1_TIME_EXIT"
        else:
            r_multiple = remaining_r
            result_label = "TIME_EXIT"

        exit_price = final_close
        exit_time = df.index[final_bar]
        actual_exit_bar = final_bar

    r_multiple -= TRADING_COST_R

    return_percentage = (
        r_multiple
        * RISK_PER_TRADE
        * 100
    )

    return {
        "Pair": pair,
        "Signal Time": df.index[signal_bar],
        "Entry Time": df.index[entry_bar],
        "Exit Time": exit_time,
        "Direction": direction,
        "Score": int(score),
        "Grade": grade,
        "Confidence": confidence,
        "Entry": round(entry_price, 6),
        "Stop Loss": round(original_stop, 6),
        "TP1": round(tp1, 6),
        "TP2": round(tp2, 6),
        "Exit Price": round(exit_price, 6),
        "Result": result_label,
        "TP1 Hit": tp1_hit,
        "R Multiple": round(
            float(r_multiple),
            4,
        ),
        "Return %": round(
            float(return_percentage),
            4,
        ),
        "Bars Held": int(
            actual_exit_bar - entry_bar + 1
        ),
        "Exit Bar": int(actual_exit_bar),
        "Reasons": ", ".join(reasons),
    }


# ============================================================
# BACKTEST ONE PAIR
# ============================================================

def backtest_pair(
    pair: str,
    ticker: str,
) -> pd.DataFrame:
    data = prepare_pair_data(
        pair=pair,
        ticker=ticker,
    )

    if data is None or len(data) < 300:
        print(
            pair,
            "- insufficient prepared data",
        )

        return pd.DataFrame()

    trades: List[Dict] = []

    next_allowed_bar = 0

    for index in range(
        250,
        len(data) - 2,
    ):
        if (
            not ALLOW_OVERLAPPING_TRADES
            and index < next_allowed_bar
        ):
            continue

        row = data.iloc[index]
       score_result = calculate_historical_signal(row)

      direction = str(score_result["action"])
score = int(score_result["score"])

global MANDATORY_PASS_COUNT

if not score_result.get("mandatory_passed", False):
    rejection_data = score_result.get(
        "rejection_reasons",
        {},
    )

    for side in ["BUY", "SELL"]:
        for reason in rejection_data.get(side, []):
            REJECTION_COUNTS[
                f"{side}: {reason}"
            ] += 1

            PAIR_REJECTION_COUNTS[
                f"{pair} | {side}: {reason}"
            ] += 1

    continue

MANDATORY_PASS_COUNT += 1

if score < MIN_SCORE:
    REJECTION_COUNTS[
        "Passed mandatory filters but score below minimum"
    ] += 1

    continue

direction = str(score_result["action"])
score = int(score_result["score"])

global MANDATORY_PASS_COUNT

if not score_result.get("mandatory_passed", False):
    rejection_data = score_result.get(
        "rejection_reasons",
        {},
    )

    for trade_side in ["BUY", "SELL"]:
        for reason in rejection_data.get(trade_side, []):
            key = f"{trade_side}: {reason}"

            REJECTION_COUNTS[key] += 1
            PAIR_REJECTION_COUNTS[
                f"{pair} | {key}"
            ] += 1

    continue

MANDATORY_PASS_COUNT += 1

if score < MIN_SCORE:
    REJECTION_COUNTS[
        "Passed mandatory filters but score below minimum"
    ] += 1

    PAIR_REJECTION_COUNTS[
        f"{pair} | Passed mandatory but score below {MIN_SCORE}"
    ] += 1

    continue

        trade = simulate_trade(
            df=data,
            signal_bar=index,
            pair=pair,
            direction=direction,
            score=score,
            grade=str(
                score_result["grade"]
            ),
            confidence=str(
                score_result["confidence"]
            ),
            reasons=list(
                score_result["reasons"]
            ),
        )

        if trade is None:
            continue

        trades.append(trade)

        if not ALLOW_OVERLAPPING_TRADES:
            next_allowed_bar = (
                int(trade["Exit Bar"]) + 1
            )

    result = pd.DataFrame(trades)

    print(
        pair,
        "backtest trades:",
        len(result),
    )

    return result
# ============================================================
# EQUITY AND DRAWDOWN
# ============================================================

def build_equity_curve(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(
            columns=[
                "Exit Time",
                "R Multiple",
                "Equity",
                "Running Peak",
                "Drawdown %",
            ]
        )

    result = trades.copy()

    result = result.sort_values(
        "Exit Time"
    ).reset_index(drop=True)

    equity_values = []
    current_equity = STARTING_CAPITAL

    for r_multiple in result["R Multiple"]:
        trade_return = (
            float(r_multiple)
            * RISK_PER_TRADE
        )

        current_equity *= (
            1.0 + trade_return
        )

        equity_values.append(
            current_equity
        )

    result["Equity"] = equity_values

    result["Running Peak"] = (
        result["Equity"].cummax()
    )

    result["Drawdown %"] = (
        (
            result["Equity"]
            - result["Running Peak"]
        )
        / result["Running Peak"]
        * 100
    )

    return result[
        [
            "Exit Time",
            "R Multiple",
            "Equity",
            "Running Peak",
            "Drawdown %",
        ]
    ]


def calculate_max_drawdown(
    trades: pd.DataFrame,
) -> float:
    equity_curve = build_equity_curve(
        trades
    )

    if equity_curve.empty:
        return 0.0

    return round(
        float(
            equity_curve[
                "Drawdown %"
            ].min()
        ),
        2,
    )


# ============================================================
# PERFORMANCE STATISTICS
# ============================================================

def calculate_statistics(
    trades: pd.DataFrame,
) -> Dict:
    if trades is None or trades.empty:
        return {
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Breakeven": 0,
            "Win Rate %": 0.0,
            "Average R": 0.0,
            "Median R": 0.0,
            "Profit Factor": 0.0,
            "Expectancy R": 0.0,
            "Total Return %": 0.0,
            "Max Drawdown %": 0.0,
            "Best Trade R": 0.0,
            "Worst Trade R": 0.0,
            "Average Bars Held": 0.0,
            "TP1 Hit Rate %": 0.0,
        }

    result = trades.copy()

    winners = result[
        result["R Multiple"] > 0
    ]

    losers = result[
        result["R Multiple"] < 0
    ]

    breakeven = result[
        result["R Multiple"] == 0
    ]

    gross_profit = float(
        winners["R Multiple"].sum()
    )

    gross_loss = abs(
        float(
            losers["R Multiple"].sum()
        )
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    equity_curve = build_equity_curve(
        result
    )

    if equity_curve.empty:
        final_equity = STARTING_CAPITAL
    else:
        final_equity = float(
            equity_curve["Equity"].iloc[-1]
        )

    total_return_percentage = (
        (
            final_equity
            / STARTING_CAPITAL
        )
        - 1
    ) * 100

    win_rate = (
        len(winners)
        / len(result)
        * 100
    )

    tp1_hit_rate = (
        result["TP1 Hit"]
        .astype(bool)
        .mean()
        * 100
    )

    expectancy_r = float(
        result["R Multiple"].mean()
    )

    return {
        "Trades": int(len(result)),
        "Wins": int(len(winners)),
        "Losses": int(len(losers)),
        "Breakeven": int(len(breakeven)),

        "Win Rate %": round(
            win_rate,
            2,
        ),

        "Average R": round(
            float(
                result["R Multiple"].mean()
            ),
            4,
        ),

        "Median R": round(
            float(
                result["R Multiple"].median()
            ),
            4,
        ),

        "Profit Factor": (
            round(profit_factor, 4)
            if np.isfinite(profit_factor)
            else "Infinity"
        ),

        "Expectancy R": round(
            expectancy_r,
            4,
        ),

        "Total Return %": round(
            total_return_percentage,
            2,
        ),

        "Max Drawdown %": calculate_max_drawdown(
            result
        ),

        "Best Trade R": round(
            float(
                result["R Multiple"].max()
            ),
            4,
        ),

        "Worst Trade R": round(
            float(
                result["R Multiple"].min()
            ),
            4,
        ),

        "Average Bars Held": round(
            float(
                result["Bars Held"].mean()
            ),
            2,
        ),

        "TP1 Hit Rate %": round(
            tp1_hit_rate,
            2,
        ),
    }


# ============================================================
# SCORE-BAND ANALYSIS
# ============================================================

def score_band(
    score: int,
) -> str:
    if score >= 95:
        return "95-100"

    if score >= 90:
        return "90-94"

    if score >= 85:
        return "85-89"

    if score >= 80:
        return "80-84"

    return "75-79"


def build_score_summary(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()

    result = trades.copy()

    result["Score Band"] = (
        result["Score"]
        .astype(int)
        .apply(score_band)
    )

    rows = []

    band_order = [
        "95-100",
        "90-94",
        "85-89",
        "80-84",
        "75-79",
    ]

    for band in band_order:
        group = result[
            result["Score Band"] == band
        ]

        if group.empty:
            continue

        stats = calculate_statistics(
            group
        )

        stats["Score Band"] = band

        rows.append(stats)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    columns = [
        "Score Band",
        "Trades",
        "Wins",
        "Losses",
        "Win Rate %",
        "Average R",
        "Profit Factor",
        "Total Return %",
        "Max Drawdown %",
        "TP1 Hit Rate %",
    ]

    return summary[
        columns
    ]


# ============================================================
# PAIR-BY-PAIR SUMMARY
# ============================================================

def build_pair_summary(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()

    rows = []

    for pair, group in trades.groupby(
        "Pair"
    ):
        group = group.sort_values(
            "Exit Time"
        )

        stats = calculate_statistics(
            group
        )

        stats["Pair"] = pair

        rows.append(stats)

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    columns = [
        "Pair",
        "Trades",
        "Wins",
        "Losses",
        "Breakeven",
        "Win Rate %",
        "Average R",
        "Median R",
        "Profit Factor",
        "Expectancy R",
        "Total Return %",
        "Max Drawdown %",
        "Best Trade R",
        "Worst Trade R",
        "Average Bars Held",
        "TP1 Hit Rate %",
    ]

    summary = summary[
        columns
    ]

    summary = summary.sort_values(
        [
            "Expectancy R",
            "Profit Factor",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return summary


# ============================================================
# DIRECTION SUMMARY
# ============================================================

def build_direction_summary(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()

    rows = []

    for direction, group in trades.groupby(
        "Direction"
    ):
        stats = calculate_statistics(
            group
        )

        stats["Direction"] = direction

        rows.append(stats)

    summary = pd.DataFrame(rows)

    columns = [
        "Direction",
        "Trades",
        "Wins",
        "Losses",
        "Win Rate %",
        "Average R",
        "Profit Factor",
        "Total Return %",
        "Max Drawdown %",
        "TP1 Hit Rate %",
    ]

    return summary[
        columns
    ]


# ============================================================
# RESULT-TYPE SUMMARY
# ============================================================

def build_result_summary(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()

    summary = (
        trades.groupby("Result")
        .agg(
            Trades=(
                "Result",
                "size",
            ),
            Average_R=(
                "R Multiple",
                "mean",
            ),
            Total_R=(
                "R Multiple",
                "sum",
            ),
            Average_Bars=(
                "Bars Held",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["Average_R"] = (
        summary["Average_R"].round(4)
    )

    summary["Total_R"] = (
        summary["Total_R"].round(4)
    )

    summary["Average_Bars"] = (
        summary["Average_Bars"].round(2)
    )

    return summary.sort_values(
        "Trades",
        ascending=False,
    ).reset_index(drop=True)
  # ============================================================
# REPORT HELPERS
# ============================================================

def ensure_report_directory() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_text_summary(
    combined_stats: Dict,
    pair_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    direction_summary: pd.DataFrame,
    result_summary: pd.DataFrame,
) -> Path:
    ensure_report_directory()

    output_path = (
        REPORT_DIR
        / "forex_v2_backtest_summary.txt"
    )

    lines = []

    lines.append("=" * 72)
    lines.append("FOREX V2 BACKTEST SUMMARY")
    lines.append("=" * 72)
    lines.append("")

    for key, value in combined_stats.items():
        lines.append(
            f"{key}: {value}"
        )

    lines.append("")
    lines.append("=" * 72)
    lines.append("PAIR SUMMARY")
    lines.append("=" * 72)

    if pair_summary.empty:
        lines.append("No pair statistics.")
    else:
        lines.append(
            pair_summary.to_string(
                index=False
            )
        )

    lines.append("")
    lines.append("=" * 72)
    lines.append("SCORE BAND SUMMARY")
    lines.append("=" * 72)

    if score_summary.empty:
        lines.append("No score statistics.")
    else:
        lines.append(
            score_summary.to_string(
                index=False
            )
        )

    lines.append("")
    lines.append("=" * 72)
    lines.append("DIRECTION SUMMARY")
    lines.append("=" * 72)

    if direction_summary.empty:
        lines.append(
            "No direction statistics."
        )
    else:
        lines.append(
            direction_summary.to_string(
                index=False
            )
        )

    lines.append("")
    lines.append("=" * 72)
    lines.append("EXIT TYPE SUMMARY")
    lines.append("=" * 72)

    if result_summary.empty:
        lines.append(
            "No result statistics."
        )
    else:
        lines.append(
            result_summary.to_string(
                index=False
            )
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output_path


# ============================================================
# CHARTS
# ============================================================

def save_equity_chart(
    equity_curve: pd.DataFrame,
) -> Optional[Path]:
    if equity_curve is None or equity_curve.empty:
        return None

    try:
        import matplotlib.pyplot as plt

        output_path = (
            REPORT_DIR
            / "forex_v2_equity_curve.png"
        )

        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            equity_curve["Exit Time"],
            equity_curve["Equity"],
        )

        plt.title(
            "Forex V2 Equity Curve"
        )

        plt.xlabel("Date")
        plt.ylabel("Equity")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

        return output_path

    except Exception as error:
        print(
            "Equity chart error:",
            error,
        )
        return None


def save_drawdown_chart(
    equity_curve: pd.DataFrame,
) -> Optional[Path]:
    if equity_curve is None or equity_curve.empty:
        return None

    try:
        import matplotlib.pyplot as plt

        output_path = (
            REPORT_DIR
            / "forex_v2_drawdown.png"
        )

        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            equity_curve["Exit Time"],
            equity_curve["Drawdown %"],
        )

        plt.title(
            "Forex V2 Drawdown"
        )

        plt.xlabel("Date")
        plt.ylabel("Drawdown %")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

        return output_path

    except Exception as error:
        print(
            "Drawdown chart error:",
            error,
        )
        return None


# ============================================================
# SAVE ALL REPORTS
# ============================================================

def save_reports(
    trades: pd.DataFrame,
    pair_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    direction_summary: pd.DataFrame,
    result_summary: pd.DataFrame,
    equity_curve: pd.DataFrame,
    combined_stats: Dict,
) -> None:
    ensure_report_directory()

    trades.to_csv(
        REPORT_DIR
        / "forex_v2_trades.csv",
        index=False,
    )

    pair_summary.to_csv(
        REPORT_DIR
        / "forex_v2_pair_summary.csv",
        index=False,
    )

    score_summary.to_csv(
        REPORT_DIR
        / "forex_v2_score_summary.csv",
        index=False,
    )

    direction_summary.to_csv(
        REPORT_DIR
        / "forex_v2_direction_summary.csv",
        index=False,
    )

    result_summary.to_csv(
        REPORT_DIR
        / "forex_v2_result_summary.csv",
        index=False,
    )

    equity_curve.to_csv(
        REPORT_DIR
        / "forex_v2_equity_curve.csv",
        index=False,
    )

    summary_path = save_text_summary(
        combined_stats=combined_stats,
        pair_summary=pair_summary,
        score_summary=score_summary,
        direction_summary=direction_summary,
        result_summary=result_summary,
    )

    equity_chart = save_equity_chart(
        equity_curve
    )

    drawdown_chart = save_drawdown_chart(
        equity_curve
    )

    print(
        "Summary saved:",
        summary_path,
    )

    if equity_chart:
        print(
            "Equity chart saved:",
            equity_chart,
        )

    if drawdown_chart:
        print(
            "Drawdown chart saved:",
            drawdown_chart,
        )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_combined_results(
    combined_stats: Dict,
) -> None:
    print("\n" + "=" * 72)
    print("COMBINED BACKTEST RESULTS")
    print("=" * 72)

    for key, value in combined_stats.items():
        print(
            f"{key:24}: {value}"
        )


def print_dataframe_section(
    title: str,
    dataframe: pd.DataFrame,
) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    if dataframe.empty:
        print("No data.")
        return

    print(
        dataframe.to_string(
            index=False
        )
    )


# ============================================================
# RUN FULL BACKTEST
# ============================================================

def run_backtest() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    all_trades = []

    for pair, ticker in PAIRS.items():
        try:
            pair_trades = backtest_pair(
                pair=pair,
                ticker=ticker,
            )

            if not pair_trades.empty:
                all_trades.append(
                    pair_trades
                )

        except Exception as error:
            print(
                pair,
                "backtest error:",
                type(error).__name__,
                error,
            )

    if not all_trades:
        print(
            "\nNo trades found for any pair."
        )

        ensure_report_directory()

        empty_message = (
            REPORT_DIR
            / "forex_v2_no_trades.txt"
        )

        empty_message.write_text(
            (
                "No trades found.\n"
                f"Minimum score: {MIN_SCORE}\n"
            ),
            encoding="utf-8",
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    trades = pd.concat(
        all_trades,
        ignore_index=True,
    )

    trades = trades.sort_values(
        "Entry Time"
    ).reset_index(drop=True)

    pair_summary = build_pair_summary(
        trades
    )

    score_summary = build_score_summary(
        trades
    )

    direction_summary = (
        build_direction_summary(
            trades
        )
    )

    result_summary = (
        build_result_summary(
            trades
        )
    )

    equity_curve = build_equity_curve(
        trades
    )

    combined_stats = calculate_statistics(
        trades
    )

    print_combined_results(
        combined_stats
    )

    print_dataframe_section(
        "PAIR RESULTS",
        pair_summary,
    )

    print_dataframe_section(
        "SCORE BAND RESULTS",
        score_summary,
    )

    print_dataframe_section(
        "DIRECTION RESULTS",
        direction_summary,
    )

    print_dataframe_section(
        "EXIT TYPE RESULTS",
        result_summary,
    )

    save_reports(
        trades=trades,
        pair_summary=pair_summary,
        score_summary=score_summary,
        direction_summary=direction_summary,
        result_summary=result_summary,
        equity_curve=equity_curve,
        combined_stats=combined_stats,
    )

    print("\nLatest 20 trades:")

    print(
        trades.tail(20).to_string(
            index=False
        )
    )

    return trades, pair_summary


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 72)
    print("FOREX V2 MULTI-PAIR BACKTEST")
    print("=" * 72)

    print(
        "Pairs:",
        len(PAIRS),
    )

    print(
        "Minimum score:",
        MIN_SCORE,
    )

    print(
        "Risk per trade:",
        f"{RISK_PER_TRADE * 100}%",
    )

    print(
        "Starting capital:",
        STARTING_CAPITAL,
    )

    trades, pair_summary = (
        run_backtest()
    )

    print("\nBacktest finished.")

    print(
        "Total trades:",
        len(trades),
    )

    print(
        "Pairs with trades:",
        len(pair_summary),
    )


if __name__ == "__main__":
    main()
