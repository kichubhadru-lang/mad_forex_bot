import os
import requests
import pandas as pd
import yfinance as yf

from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

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
    "GBPJPY": "GBPJPY=X"
}


def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=30
    )


def download(symbol, interval, period):
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def indicators(df):

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA20"] = EMAIndicator(close,20).ema_indicator()
    df["EMA50"] = EMAIndicator(close,50).ema_indicator()
    df["EMA200"] = EMAIndicator(close,200).ema_indicator()

    df["RSI"] = RSIIndicator(close,14).rsi()

    df["ADX"] = ADXIndicator(
        high,
        low,
        close,
        14
    ).adx()

    df["ATR"] = AverageTrueRange(
        high,
        low,
        close,
        14
    ).average_true_range()

    return df.dropna()
def elite_signal(pair, ticker):

    daily = download(ticker, "1d", "1y")
    h4 = download(ticker, "4h", "180d")
    h1 = download(ticker, "1h", "60d")

    if daily is None or h4 is None or h1 is None:
        return None

    daily = indicators(daily)
    h4 = indicators(h4)
    h1 = indicators(h1)

    if daily.empty or h4.empty or h1.empty:
        return None

    d = daily.iloc[-1]
    h = h4.iloc[-1]
    e = h1.iloc[-1]

    score = 0
    reasons = []

    # Daily trend
    if d.EMA50 > d.EMA200:
        score += 20
        reasons.append("Daily Uptrend")

    if d.Close > d.EMA200:
        score += 10
        reasons.append("Above EMA200")

    # 4H trend
    if h.EMA20 > h.EMA50:
        score += 20
        reasons.append("4H Bullish")

    # 1H confirmation
    if e.Close > e.EMA20:
        score += 15
        reasons.append("1H Confirmation")

    # ADX
    if h.ADX > 25:
        score += 15
        reasons.append("Strong ADX")

    # RSI
    if 50 <= h.RSI <= 65:
        score += 10
        reasons.append("Healthy RSI")

    atr = float(h.ATR)

    entry = float(e.Close)
    sl = round(entry - (1.5 * atr), 4)
    tp1 = round(entry + (2 * atr), 4)
    tp2 = round(entry + (4 * atr), 4)

    if score < 80:
        return None

    return {
        "Pair": pair,
        "Score": score,
        "Entry": round(entry, 4),
        "SL": sl,
        "TP1": tp1,
        "TP2": tp2,
        "Reasons": ", ".join(reasons)
    }
def format_signal(s):
    return f"""
🏆 FOREX ELITE SIGNAL

Pair: {s['Pair']}

🟢 BUY

Score: {s['Score']}/100

Entry : {s['Entry']}
SL    : {s['SL']}
TP1   : {s['TP1']}
TP2   : {s['TP2']}

Reasons:
{s['Reasons']}

⚠️ Educational use only.
"""


def main():
    send_telegram("✅ Forex bot is running successfully!")
    print("Telegram test sent")


if __name__ == "__main__":
    main()
