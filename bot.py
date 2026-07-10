import os
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

MIN_SCORE = 80
MAX_SIGNALS = 3

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


def send_telegram(message):
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN missing")
        return False

    if not CHAT_ID:
        print("ERROR: CHAT_ID missing")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            json={
                "chat_id": CHAT_ID,
                "text": message,
            },
            timeout=30,
        )

        print("Telegram status:", response.status_code)
        print("Telegram response:", response.text[:500])

        return response.status_code == 200

    except Exception as error:
        print("Telegram error:", error)
        return False


def clean_data(df):
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    if not all(column in df.columns for column in required):
        return None

    df = df[required].copy()
    df = df.dropna()

    return None if df.empty else df


def download_data(symbol, interval, period):
    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        return clean_data(df)

    except Exception as error:
        print(f"Download error {symbol} {interval}: {error}")
        return None


def make_four_hour_data(hourly):
    if hourly is None or hourly.empty:
        return None

    try:
        four_hour = hourly.resample("4h").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
            }
        )

        return clean_data(four_hour)

    except Exception as error:
        print("4H resample error:", error)
        return None


def add_indicators(df):
    if df is None or len(df) < 210:
        return None

    try:
        df = df.copy()

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)

        df["EMA20"] = EMAIndicator(close=close, window=20).ema_indicator()
        df["EMA50"] = EMAIndicator(close=close, window=50).ema_indicator()
        df["EMA200"] = EMAIndicator(close=close, window=200).ema_indicator()

        df["RSI"] = RSIIndicator(close=close, window=14).rsi()

        df["ADX"] = ADXIndicator(
            high=high,
            low=low,
            close=close,
            window=14,
        ).adx()

        df["ATR"] = AverageTrueRange(
            high=high,
            low=low,
            close=close,
            window=14,
        ).average_true_range()

        df["HIGH20"] = high.rolling(20).max()
        df["LOW20"] = low.rolling(20).min()

        df = df.dropna()

        return None if df.empty else df

    except Exception as error:
        print("Indicator error:", error)
        return None


def analyze_pair(pair, ticker):
    daily = add_indicators(
        download_data(ticker, interval="1d", period="2y")
    )

    hourly_raw = download_data(
        ticker,
        interval="1h",
        period="60d",
    )

    hourly = add_indicators(hourly_raw)
    four_hour = add_indicators(make_four_hour_data(hourly_raw))

    if daily is None or hourly is None or four_hour is None:
        print(pair, "insufficient data")
        return None

    d = daily.iloc[-1]
    h4 = four_hour.iloc[-1]
    h1 = hourly.iloc[-1]

    buy_score = 0
    sell_score = 0

    buy_reasons = []
    sell_reasons = []

    # Daily trend
    if d["EMA50"] > d["EMA200"] and d["Close"] > d["EMA200"]:
        buy_score += 25
        buy_reasons.append("Daily bullish trend")

    if d["EMA50"] < d["EMA200"] and d["Close"] < d["EMA200"]:
        sell_score += 25
        sell_reasons.append("Daily bearish trend")

    # Four-hour structure
    if h4["EMA20"] > h4["EMA50"] > h4["EMA200"]:
        buy_score += 25
        buy_reasons.append("4H EMA alignment")

    if h4["EMA20"] < h4["EMA50"] < h4["EMA200"]:
        sell_score += 25
        sell_reasons.append("4H EMA alignment")

    # Trend strength
    if h4["ADX"] >= 25:
        buy_score += 15
        sell_score += 15
        buy_reasons.append("Strong ADX")
        sell_reasons.append("Strong ADX")

    # Momentum
    if 50 <= h4["RSI"] <= 68:
        buy_score += 15
        buy_reasons.append("Healthy bullish RSI")

    if 32 <= h4["RSI"] <= 50:
        sell_score += 15
        sell_reasons.append("Healthy bearish RSI")

    # One-hour confirmation
    if h1["Close"] > h1["EMA20"] and h1["EMA20"] > h1["EMA50"]:
        buy_score += 10
        buy_reasons.append("1H bullish confirmation")

    if h1["Close"] < h1["EMA20"] and h1["EMA20"] < h1["EMA50"]:
        sell_score += 10
        sell_reasons.append("1H bearish confirmation")

    # Breakout confirmation
    previous_high = four_hour["HIGH20"].shift(1).iloc[-1]
    previous_low = four_hour["LOW20"].shift(1).iloc[-1]

    if h4["Close"] > previous_high:
        buy_score += 10
        buy_reasons.append("4H breakout")

    if h4["Close"] < previous_low:
        sell_score += 10
        sell_reasons.append("4H breakdown")

    if buy_score >= sell_score:
        action = "BUY"
        score = buy_score
        reasons = buy_reasons
    else:
        action = "SELL"
        score = sell_score
        reasons = sell_reasons

    if score < MIN_SCORE:
        print(pair, "score", score, "- no signal")
        return None

    entry = float(h1["Close"])
    atr = float(h4["ATR"])

    if atr <= 0:
        return None

    if action == "BUY":
        stop_loss = entry - (1.5 * atr)
        tp1 = entry + (2 * atr)
        tp2 = entry + (3 * atr)
    else:
        stop_loss = entry + (1.5 * atr)
        tp1 = entry - (2 * atr)
        tp2 = entry - (3 * atr)

    decimals = 2 if pair == "XAUUSD" else 5

    return {
        "Pair": pair,
        "Action": action,
        "Score": score,
        "Entry": round(entry, decimals),
        "SL": round(stop_loss, decimals),
        "TP1": round(tp1, decimals),
        "TP2": round(tp2, decimals),
        "RSI": round(float(h4["RSI"]), 2),
        "ADX": round(float(h4["ADX"]), 2),
        "Reasons": ", ".join(reasons),
    }


def format_signal(signal):
    action_emoji = "🟢" if signal["Action"] == "BUY" else "🔴"

    return (
        "🏆 FOREX ELITE SIGNAL\n\n"
        f"Pair: {signal['Pair']}\n"
        f"{action_emoji} Action: {signal['Action']}\n"
        f"Score: {signal['Score']}/100\n\n"
        f"Entry: {signal['Entry']}\n"
        f"SL: {signal['SL']}\n"
        f"TP1: {signal['TP1']}\n"
        f"TP2: {signal['TP2']}\n\n"
        f"RSI: {signal['RSI']}\n"
        f"ADX: {signal['ADX']}\n"
        f"Reasons: {signal['Reasons']}\n\n"
        "⚠️ Educational use only. Confirm manually."
    )


def main():
    print("Starting Forex scan")

    signals = []

    for pair, ticker in PAIRS.items():
        print("Scanning", pair)

        signal = analyze_pair(pair, ticker)

        if signal:
            signals.append(signal)

    signals.sort(
        key=lambda item: item["Score"],
        reverse=True,
    )

    if not signals:
        message = (
            "🏆 FOREX SWING BOT\n\n"
            f"Date: {datetime.utcnow().strftime('%d-%b-%Y %H:%M UTC')}\n"
            f"Pairs scanned: {len(PAIRS)}\n"
            "Elite signals: 0\n\n"
            "No high-quality setup currently."
        )

        send_telegram(message)
        print("No qualifying signals")
        return

    for signal in signals[:MAX_SIGNALS]:
        send_telegram(format_signal(signal))

    print("Signals sent:", min(len(signals), MAX_SIGNALS))


if __name__ == "__main__":
    main()
