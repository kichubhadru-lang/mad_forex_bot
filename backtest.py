# ============================================================
# ADVANCED MULTI-PAIR FOREX + GOLD BACKTEST
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


# ============================================================
# SETTINGS
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

DOWNLOAD_PERIOD = "730d"
BASE_INTERVAL = "1h"

MIN_SCORE = 75
MAX_HOLDING_BARS = 120

ATR_STOP_MULTIPLIER = 1.5
TP1_R_MULTIPLE = 2.0
TP2_R_MULTIPLE = 3.0

TP1_POSITION_FRACTION = 0.50
TP2_POSITION_FRACTION = 0.50

SPREAD_COST_R = 0.03

STARTING_CAPITAL = 100_000
RISK_PER_TRADE = 0.01

ALLOW_MULTIPLE_OPEN_TRADES_PER_PAIR = False


# ============================================================
# DATA
# ============================================================

def clean_downloaded_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    if not all(column in df.columns for column in required):
        return None

    df = df[required].copy()
    df = df.dropna()

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return None if df.empty else df


def download_hourly_data(ticker: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            ticker,
            period=DOWNLOAD_PERIOD,
            interval=BASE_INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        return clean_downloaded_data(df)

    except Exception as error:
        print(f"Download failed for {ticker}: {error}")
        return None


# ============================================================
# TIMEFRAME RESAMPLING
# ============================================================

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

    return clean_downloaded_data(result)


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(
    df: pd.DataFrame,
    require_ema200: bool = True,
) -> Optional[pd.DataFrame]:

    minimum_rows = 210 if require_ema200 else 60

    if df is None or len(df) < minimum_rows:
        return None

    try:
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

        if require_ema200:
            result["EMA200"] = EMAIndicator(
                close=close,
                window=200,
            ).ema_indicator()
        else:
            result["EMA200"] = np.nan

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

        result["ATR_AVG20"] = result["ATR"].rolling(20).mean()

        result["SWING_HIGH_20"] = high.rolling(20).max()
        result["SWING_LOW_20"] = low.rolling(20).min()

        result["PREV_SWING_HIGH"] = result["SWING_HIGH_20"].shift(1)
        result["PREV_SWING_LOW"] = result["SWING_LOW_20"].shift(1)

        result["BODY"] = (result["Close"] - result["Open"]).abs()
        result["RANGE"] = result["High"] - result["Low"]

        result["UPPER_WICK"] = (
            result["High"]
            - result[["Open", "Close"]].max(axis=1)
        )

        result["LOWER_WICK"] = (
            result[["Open", "Close"]].min(axis=1)
            - result["Low"]
        )

        result = result.dropna(
            subset=[
                "EMA20",
                "EMA50",
                "RSI",
                "ADX",
                "ATR",
                "ATR_AVG20",
                "PREV_SWING_HIGH",
                "PREV_SWING_LOW",
            ]
        )

        return None if result.empty else result

    except Exception as error:
        print("Indicator error:", error)
        return None


# ============================================================
# MARKET STRUCTURE APPROXIMATIONS
# ============================================================

def add_smart_money_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # Break of structure
    result["BULLISH_BOS"] = (
        result["Close"] > result["PREV_SWING_HIGH"]
    )

    result["BEARISH_BOS"] = (
        result["Close"] < result["PREV_SWING_LOW"]
    )

    # Liquidity sweep:
    # Price breaks prior swing intrabar but closes back inside.
    result["BULLISH_LIQUIDITY_SWEEP"] = (
        (result["Low"] < result["PREV_SWING_LOW"])
        & (result["Close"] > result["PREV_SWING_LOW"])
    )

    result["BEARISH_LIQUIDITY_SWEEP"] = (
        (result["High"] > result["PREV_SWING_HIGH"])
        & (result["Close"] < result["PREV_SWING_HIGH"])
    )

    # Fair Value Gap approximation using a three-candle pattern.
    result["BULLISH_FVG"] = (
        result["Low"] > result["High"].shift(2)
    )

    result["BEARISH_FVG"] = (
        result["High"] < result["Low"].shift(2)
    )

    # Order-block approximation:
    # Previous opposite candle followed by strong displacement.
    bullish_displacement = (
        (result["Close"] > result["Open"])
        & (result["BODY"] > result["ATR"])
    )

    bearish_displacement = (
        (result["Close"] < result["Open"])
        & (result["BODY"] > result["ATR"])
    )

    previous_bearish = (
        result["Close"].shift(1)
        < result["Open"].shift(1)
    )

    previous_bullish = (
        result["Close"].shift(1)
        > result["Open"].shift(1)
    )

    result["BULLISH_ORDER_BLOCK"] = (
        previous_bearish & bullish_displacement
    )

    result["BEARISH_ORDER_BLOCK"] = (
        previous_bullish & bearish_displacement
    )

    # ATR expansion
    result["ATR_EXPANSION"] = (
        result["ATR"] > result["ATR_AVG20"] * 1.05
    )

    return result


# ============================================================
# ALIGN HIGHER TIMEFRAMES WITHOUT LOOKAHEAD
# ============================================================

def align_timeframe_to_hourly(
    higher_df: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
    prefix: str,
) -> pd.DataFrame:

    # Shift one higher-timeframe candle so the backtest uses only
    # information from a fully completed higher-timeframe bar.
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
# 1H RETEST LOGIC
# ============================================================

def add_hourly_retest_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    bullish_retest_level = result["EMA20"]

    result["BULLISH_RETEST"] = (
        (result["Low"] <= bullish_retest_level * 1.001)
        & (result["Close"] > bullish_retest_level)
        & (result["Close"] > result["Open"])
    )

    bearish_retest_level = result["EMA20"]

    result["BEARISH_RETEST"] = (
        (result["High"] >= bearish_retest_level * 0.999)
        & (result["Close"] < bearish_retest_level)
        & (result["Close"] < result["Open"])
    )

    return result


# ============================================================
# PREPARE ONE PAIR
# ============================================================

def prepare_pair_data(
    pair: str,
    ticker: str,
) -> Optional[pd.DataFrame]:

    hourly_raw = download_hourly_data(ticker)

    if hourly_raw is None or len(hourly_raw) < 1000:
        print(pair, "- insufficient hourly data")
        return None

    h1 = add_indicators(
        hourly_raw,
        require_ema200=True,
    )

    h4_raw = resample_ohlc(hourly_raw, "4h")
    daily_raw = resample_ohlc(hourly_raw, "1D")
    weekly_raw = resample_ohlc(hourly_raw, "1W")

    h4 = add_indicators(
        h4_raw,
        require_ema200=True,
    )

    daily = add_indicators(
        daily_raw,
        require_ema200=True,
    )

    weekly = add_indicators(
        weekly_raw,
        require_ema200=False,
    )

    if any(item is None for item in [h1, h4, daily, weekly]):
        print(pair, "- timeframe preparation failed")
        return None

    h4 = add_smart_money_features(h4)
    h1 = add_hourly_retest_features(h1)

    h4_aligned = align_timeframe_to_hourly(
        h4,
        h1.index,
        "H4",
    )

    daily_aligned = align_timeframe_to_hourly(
        daily,
        h1.index,
        "D1",
    )

    weekly_aligned = align_timeframe_to_hourly(
        weekly,
        h1.index,
        "W1",
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

    return None if combined.empty else combined


# ============================================================
# SCORE SIGNAL
# ============================================================

def calculate_signal(
    row: pd.Series,
) -> Tuple[str, int, list]:

    buy_score = 0
    sell_score = 0

    buy_reasons = []
    sell_reasons = []

    # Weekly bias: 15 points
    if row["W1_EMA20"] > row["W1_EMA50"]:
        buy_score += 15
        buy_reasons.append("Weekly bullish")

    if row["W1_EMA20"] < row["W1_EMA50"]:
        sell_score += 15
        sell_reasons.append("Weekly bearish")

    # Daily bias: 15 points
    if (
        row["D1_EMA50"] > row["D1_EMA200"]
        and row["D1_Close"] > row["D1_EMA200"]
    ):
        buy_score += 15
        buy_reasons.append("Daily bullish")

    if (
        row["D1_EMA50"] < row["D1_EMA200"]
        and row["D1_Close"] < row["D1_EMA200"]
    ):
        sell_score += 15
        sell_reasons.append("Daily bearish")

    # 4H EMA alignment: 15 points
    if (
        row["H4_EMA20"]
        > row["H4_EMA50"]
        > row["H4_EMA200"]
    ):
        buy_score += 15
        buy_reasons.append("4H EMA alignment")

    if (
        row["H4_EMA20"]
        < row["H4_EMA50"]
        < row["H4_EMA200"]
    ):
        sell_score += 15
        sell_reasons.append("4H EMA alignment")

    # Break of structure: 15 points
    if bool(row["H4_BULLISH_BOS"]):
        buy_score += 15
        buy_reasons.append("Bullish BOS")

    if bool(row["H4_BEARISH_BOS"]):
        sell_score += 15
        sell_reasons.append("Bearish BOS")

    # Order block: 10 points
    if bool(row["H4_BULLISH_ORDER_BLOCK"]):
        buy_score += 10
        buy_reasons.append("Bullish order block")

    if bool(row["H4_BEARISH_ORDER_BLOCK"]):
        sell_score += 10
        sell_reasons.append("Bearish order block")

    # FVG: 10 points
    if bool(row["H4_BULLISH_FVG"]):
        buy_score += 10
        buy_reasons.append("Bullish FVG")

    if bool(row["H4_BEARISH_FVG"]):
        sell_score += 10
        sell_reasons.append("Bearish FVG")

    # Liquidity sweep: 10 points
    if bool(row["H4_BULLISH_LIQUIDITY_SWEEP"]):
        buy_score += 10
        buy_reasons.append("Bullish liquidity sweep")

    if bool(row["H4_BEARISH_LIQUIDITY_SWEEP"]):
        sell_score += 10
        sell_reasons.append("Bearish liquidity sweep")

    # ADX: 10 points
    if row["H4_ADX"] >= 25:
        buy_score += 10
        sell_score += 10

        buy_reasons.append("Strong ADX")
        sell_reasons.append("Strong ADX")

    # ATR expansion: 5 points
    if bool(row["H4_ATR_EXPANSION"]):
        buy_score += 5
        sell_score += 5

        buy_reasons.append("ATR expansion")
        sell_reasons.append("ATR expansion")

    # 1H retest: 10 points
    if bool(row["BULLISH_RETEST"]):
        buy_score += 10
        buy_reasons.append("1H bullish retest")

    if bool(row["BEARISH_RETEST"]):
        sell_score += 10
        sell_reasons.append("1H bearish retest")

    # 1H momentum: 5 points
    if (
        row["EMA20"] > row["EMA50"]
        and 50 <= row["RSI"] <= 68
    ):
        buy_score += 5
        buy_reasons.append("1H momentum")

    if (
        row["EMA20"] < row["EMA50"]
        and 32 <= row["RSI"] <= 50
    ):
        sell_score += 5
        sell_reasons.append("1H momentum")

    if buy_score >= sell_score:
        return "BUY", buy_score, buy_reasons

    return "SELL", sell_score, sell_reasons


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df: pd.DataFrame,
    signal_index: int,
    direction: str,
    pair: str,
    score: int,
    reasons: list,
) -> Optional[Dict]:

    entry_index = signal_index + 1

    if entry_index >= len(df):
        return None

    entry_row = df.iloc[entry_index]

    entry_price = float(entry_row["Open"])
    atr = float(df.iloc[signal_index]["H4_ATR"])

    if atr <= 0:
        return None

    risk_distance = atr * ATR_STOP_MULTIPLIER

    if direction == "BUY":
        stop_loss = entry_price - risk_distance
        tp1 = entry_price + risk_distance * TP1_R_MULTIPLE
        tp2 = entry_price + risk_distance * TP2_R_MULTIPLE

    else:
        stop_loss = entry_price + risk_distance
        tp1 = entry_price - risk_distance * TP1_R_MULTIPLE
        tp2 = entry_price - risk_distance * TP2_R_MULTIPLE

    tp1_hit = False
    trade_r = 0.0
    result = "TIME_EXIT"
    exit_price = float(entry_price)
    exit_time = df.index[entry_index]

    final_bar = min(
        entry_index + MAX_HOLDING_BARS,
        len(df) - 1,
    )

    for bar_index in range(entry_index, final_bar + 1):
        bar = df.iloc[bar_index]

        high = float(bar["High"])
        low = float(bar["Low"])

        if direction == "BUY":

            if not tp1_hit:
                stop_hit = low <= stop_loss
                tp1_reached = high >= tp1

                # Conservative assumption:
                # if both occur in the same candle, stop is counted first.
                if stop_hit:
                    trade_r = -1.0
                    result = "LOSS"
                    exit_price = stop_loss
                    exit_time = df.index[bar_index]
                    break

                if tp1_reached:
                    tp1_hit = True

                    trade_r += (
                        TP1_POSITION_FRACTION
                        * TP1_R_MULTIPLE
                    )

                    stop_loss = entry_price

            else:
                breakeven_hit = low <= stop_loss
                tp2_reached = high >= tp2

                if breakeven_hit:
                    result = "TP1_BE"
                    exit_price = stop_loss
                    exit_time = df.index[bar_index]
                    break

                if tp2_reached:
                    trade_r += (
                        TP2_POSITION_FRACTION
                        * TP2_R_MULTIPLE
                    )

                    result = "TP2"
                    exit_price = tp2
                    exit_time = df.index[bar_index]
                    break

        else:

            if not tp1_hit:
                stop_hit = high >= stop_loss
                tp1_reached = low <= tp1

                if stop_hit:
                    trade_r = -1.0
                    result = "LOSS"
                    exit_price = stop_loss
                    exit_time = df.index[bar_index]
                    break

                if tp1_reached:
                    tp1_hit = True

                    trade_r += (
                        TP1_POSITION_FRACTION
                        * TP1_R_MULTIPLE
                    )

                    stop_loss = entry_price

            else:
                breakeven_hit = high >= stop_loss
                tp2_reached = low <= tp2

                if breakeven_hit:
                    result = "TP1_BE"
                    exit_price = stop_loss
                    exit_time = df.index[bar_index]
                    break

                if tp2_reached:
                    trade_r += (
                        TP2_POSITION_FRACTION
                        * TP2_R_MULTIPLE
                    )

                    result = "TP2"
                    exit_price = tp2
                    exit_time = df.index[bar_index]
                    break

    else:
        final_close = float(df.iloc[final_bar]["Close"])

        if tp1_hit:
            remaining_r = (
                (final_close - entry_price)
                / risk_distance
            )

            if direction == "SELL":
                remaining_r = -remaining_r

            trade_r += TP2_POSITION_FRACTION * remaining_r
            result = "TP1_TIME_EXIT"

        else:
            trade_r = (
                (final_close - entry_price)
                / risk_distance
            )

            if direction == "SELL":
                trade_r = -trade_r

            result = "TIME_EXIT"

        exit_price = final_close
        exit_time = df.index[final_bar]

    trade_r -= SPREAD_COST_R

    return {
        "Pair": pair,
        "Signal Time": df.index[signal_index],
        "Entry Time": df.index[entry_index],
        "Exit Time": exit_time,
        "Direction": direction,
        "Score": score,
        "Entry": round(entry_price, 5),
        "Stop": round(
            entry_price - risk_distance
            if direction == "BUY"
            else entry_price + risk_distance,
            5,
        ),
        "TP1": round(tp1, 5),
        "TP2": round(tp2, 5),
        "Exit": round(exit_price, 5),
        "Result": result,
        "R Multiple": round(trade_r, 3),
        "Return %": round(
            (
                trade_r
                * RISK_PER_TRADE
                * 100
            ),
            3,
        ),
        "Bars Held": final_bar - entry_index + 1,
        "Reasons": ", ".join(reasons),
    }


# ============================================================
# BACKTEST ONE PAIR
# ============================================================

def backtest_pair(
    pair: str,
    ticker: str,
) -> pd.DataFrame:

    print(f"\nPreparing {pair}...")

    df = prepare_pair_data(pair, ticker)

    if df is None or len(df) < 300:
        print(pair, "- no usable dataset")
        return pd.DataFrame()

    trades = []
    next_allowed_bar = 0

    for index in range(250, len(df) - 2):

        if (
            not ALLOW_MULTIPLE_OPEN_TRADES_PER_PAIR
            and index < next_allowed_bar
        ):
            continue

        direction, score, reasons = calculate_signal(
            df.iloc[index]
        )

        if score < MIN_SCORE:
            continue

        trade = simulate_trade(
            df=df,
            signal_index=index,
            direction=direction,
            pair=pair,
            score=score,
            reasons=reasons,
        )

        if trade is None:
            continue

        trades.append(trade)

        if not ALLOW_MULTIPLE_OPEN_TRADES_PER_PAIR:
            exit_time = trade["Exit Time"]

            exit_location = df.index.get_indexer(
                [exit_time],
                method="nearest",
            )[0]

            next_allowed_bar = exit_location + 1

    result = pd.DataFrame(trades)

    print(pair, "trades:", len(result))

    return result


# ============================================================
# PERFORMANCE STATISTICS
# ============================================================

def calculate_max_drawdown(r_values: pd.Series) -> float:
    equity = STARTING_CAPITAL * (
        1
        + r_values
        * RISK_PER_TRADE
    ).cumprod()

    running_high = equity.cummax()

    drawdown = (
        equity - running_high
    ) / running_high

    return float(drawdown.min() * 100)


def performance_summary(
    trades: pd.DataFrame,
) -> Dict:

    if trades is None or trades.empty:
        return {
            "Trades": 0,
            "Win Rate %": 0,
            "Average R": 0,
            "Profit Factor": 0,
            "Total Return %": 0,
            "Max Drawdown %": 0,
        }

    winners = trades[
        trades["R Multiple"] > 0
    ]

    losers = trades[
        trades["R Multiple"] < 0
    ]

    gross_profit = winners["R Multiple"].sum()
    gross_loss = abs(losers["R Multiple"].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    compounded_equity = STARTING_CAPITAL * (
        1
        + trades["R Multiple"]
        * RISK_PER_TRADE
    ).cumprod()

    total_return = (
        compounded_equity.iloc[-1]
        / STARTING_CAPITAL
        - 1
    ) * 100

    max_drawdown = calculate_max_drawdown(
        trades["R Multiple"]
    )

    return {
        "Trades": len(trades),
        "Wins": len(winners),
        "Losses": len(losers),
        "Win Rate %": round(
            len(winners) / len(trades) * 100,
            2,
        ),
        "Average R": round(
            trades["R Multiple"].mean(),
            3,
        ),
        "Median R": round(
            trades["R Multiple"].median(),
            3,
        ),
        "Profit Factor": round(
            profit_factor,
            3,
        ),
        "Total Return %": round(
            total_return,
            2,
        ),
        "Max Drawdown %": round(
            max_drawdown,
            2,
        ),
        "Best Trade R": round(
            trades["R Multiple"].max(),
            3,
        ),
        "Worst Trade R": round(
            trades["R Multiple"].min(),
            3,
        ),
    }


def pair_summary_table(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    if trades.empty:
        return pd.DataFrame()

    for pair, group in trades.groupby("Pair"):
        stats = performance_summary(
            group.sort_values("Entry Time")
        )

        stats["Pair"] = pair
        rows.append(stats)

    result = pd.DataFrame(rows)

    columns = [
        "Pair",
        "Trades",
        "Wins",
        "Losses",
        "Win Rate %",
        "Average R",
        "Profit Factor",
        "Total Return %",
        "Max Drawdown %",
        "Best Trade R",
        "Worst Trade R",
    ]

    return result[columns].sort_values(
        "Total Return %",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# RUN ALL PAIRS
# ============================================================

def run_multi_pair_backtest():
    all_trades = []

    for pair, ticker in PAIRS.items():
        pair_trades = backtest_pair(
            pair,
            ticker,
        )

        if not pair_trades.empty:
            all_trades.append(pair_trades)

    if not all_trades:
        print("\nNo trades found.")
        return pd.DataFrame(), pd.DataFrame()

    trades = pd.concat(
        all_trades,
        ignore_index=True,
    )

    trades = trades.sort_values(
        "Entry Time"
    ).reset_index(drop=True)

    pair_stats = pair_summary_table(trades)
    combined_stats = performance_summary(trades)

    print("\n" + "=" * 70)
    print("COMBINED BACKTEST RESULTS")
    print("=" * 70)

    for key, value in combined_stats.items():
        print(f"{key:22}: {value}")

    print("\n" + "=" * 70)
    print("RESULTS BY PAIR")
    print("=" * 70)

    print(pair_stats.to_string(index=False))

    trades.to_csv(
        "forex_advanced_backtest_trades.csv",
        index=False,
    )

    pair_stats.to_csv(
        "forex_advanced_pair_summary.csv",
        index=False,
    )

    return trades, pair_stats


# ============================================================
# EXECUTE
# ============================================================

trades, pair_stats = run_multi_pair_backtest()

print("\nLatest trades:")
print(trades.tail(20))
