# pip install yfinance ta requests pandas numpy

import yfinance as yf
import requests
import pandas as pd

from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# ---------------- SETTINGS ---------------- #

BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

ASSETS = {
    "XAU/USD": "XAUUSD=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X"
}

# ------------------------------------------ #

def send(msg):

    requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        params={
            "chat_id": CHAT_ID,
            "text": msg
        },
        timeout=20
    )

for name, symbol in ASSETS.items():

    try:

        # ===============================
        # HIGHER TIMEFRAME TREND (4H)
        # ===============================

        df4h = yf.download(
            symbol,
            period="3mo",
            interval="4h",
            progress=False
        )

        if len(df4h) < 100:
            continue

        close4 = df4h["Close"].squeeze()

        ema50_4h = EMAIndicator(close4,50).ema_indicator()
        ema200_4h = EMAIndicator(close4,200).ema_indicator()

        bullish_trend = ema50_4h.iloc[-1] > ema200_4h.iloc[-1]
        bearish_trend = ema50_4h.iloc[-1] < ema200_4h.iloc[-1]

        # ===============================
        # ENTRY TIMEFRAME (1H)
        # ===============================

        df = yf.download(
            symbol,
            period="1mo",
            interval="1h",
            progress=False
        )

        if len(df) < 200:
            continue

        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        price = float(close.iloc[-1])

        # ===============================
        # INDICATORS
        # ===============================

        ema20 = EMAIndicator(close,20).ema_indicator()
        ema50 = EMAIndicator(close,50).ema_indicator()

        rsi = RSIIndicator(close,14).rsi()

        macd = MACD(close)

        atr = AverageTrueRange(
            high,
            low,
            close,
            window=14
        ).average_true_range()

        adx = ADXIndicator(
            high,
            low,
            close,
            window=14
        ).adx()

        # ===============================
        # STRONG BUY CONDITIONS
        # ===============================

        buy = (

            bullish_trend

            and ema20.iloc[-1] > ema50.iloc[-1]

            and rsi.iloc[-1] > 55
            and rsi.iloc[-1] < 70

            and macd.macd().iloc[-1] >
            macd.macd_signal().iloc[-1]

            and adx.iloc[-1] > 25

            and close.iloc[-1] > ema20.iloc[-1]
        )

        # ===============================
        # STRONG SELL CONDITIONS
        # ===============================

        sell = (

            bearish_trend

            and ema20.iloc[-1] < ema50.iloc[-1]

            and rsi.iloc[-1] < 45
            and rsi.iloc[-1] > 30

            and macd.macd().iloc[-1] <
            macd.macd_signal().iloc[-1]

            and adx.iloc[-1] > 25

            and close.iloc[-1] < ema20.iloc[-1]
        )

        # ===============================
        # RISK MANAGEMENT
        # ===============================

        risk = atr.iloc[-1]

        # ===============================
        # BUY SIGNAL
        # ===============================

        if buy:

            sl = round(price - risk*1.5,4)

            tp1 = round(price + risk,4)
            tp2 = round(price + risk*2,4)
            tp3 = round(price + risk*3,4)

            message = f"""
🟢 STRONG BUY SIGNAL

{name}

Entry : {round(price,4)}

SL : {sl}

TP1 : {tp1}
TP2 : {tp2}
TP3 : {tp3}

CONFIRMATIONS:

✅ 4H Trend Bullish
✅ EMA20 > EMA50
✅ RSI Momentum Strong
✅ MACD Bullish
✅ ADX Strong Trend
"""

            send(message)

        # ===============================
        # SELL SIGNAL
        # ===============================

        elif sell:

            sl = round(price + risk*1.5,4)

            tp1 = round(price - risk,4)
            tp2 = round(price - risk*2,4)
            tp3 = round(price - risk*3,4)

            message = f"""
🔴 STRONG SELL SIGNAL

{name}

Entry : {round(price,4)}

SL : {sl}

TP1 : {tp1}
TP2 : {tp2}
TP3 : {tp3}

CONFIRMATIONS:

✅ 4H Trend Bearish
✅ EMA20 < EMA50
✅ RSI Momentum Weak
✅ MACD Bearish
✅ ADX Strong Trend
"""

            send(message)

    except Exception as e:

        print(name, e)
