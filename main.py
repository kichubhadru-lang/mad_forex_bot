import yfinance as yf
import requests
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
BOT_TOKEN = "8693707625:AAGC0veKlDc0DYTrERRI2P1l1Bon4_Le5wY"
CHAT_ID = "8682661998"
ASSETS = {
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/INR": "INR=X"
}
def send(msg):
    requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        params={
            "chat_id": CHAT_ID,
            "text": msg
        }
    )
for name, symbol in ASSETS.items():
    try:
        df = yf.download(
            symbol,
            period="6mo",
            interval="1h",
            progress=False
        )
        if len(df) < 100:
            continue
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        price = float(close.iloc[-1])
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
        buy = (
            ema20.iloc[-1] > ema50.iloc[-1]
            and rsi.iloc[-1] > 55
            and macd.macd().iloc[-1] >
            macd.macd_signal().iloc[-1]
        )
        sell = (
            ema20.iloc[-1] < ema50.iloc[-1]
            and rsi.iloc[-1] < 45
            and macd.macd().iloc[-1] <
            macd.macd_signal().iloc[-1]
        )
        risk = atr.iloc[-1]
        if buy:
            sl = round(price-risk*1.5,4)
            tp1 = round(price+risk,4)
            tp2 = round(price+risk*2,4)
            tp3 = round(price+risk*3,4)
            message = (
                name+" BUY NOW "+str(round(price,4))+
                "\n\nSL "+str(sl)+
                "\n\nTP "+str(tp1)+
                "\nTP "+str(tp2)+
                "\nTP "+str(tp3)
            )
            send(message)
        elif sell:
            sl = round(price+risk*1.5,4)
            tp1 = round(price-risk,4)
            tp2 = round(price-risk*2,4)
            tp3 = round(price-risk*3,4)
            message = (
                name+" SELL NOW "+str(round(price,4))+
                "\n\nSL "+str(sl)+
                "\n\nTP "+str(tp1)+
                "\nTP "+str(tp2)+
                "\nTP "+str(tp3)
            )
            send(message)
    except Exception as e:
        print(name, e)
