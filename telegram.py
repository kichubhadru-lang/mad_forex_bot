from __future__ import annotations

import os
import requests

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")


def send_message(message: str) -> bool:
    if not BOT_TOKEN:
        print("BOT_TOKEN missing")
        return False

    if not CHAT_ID:
        print("CHAT_ID missing")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=30,
        )

        print(
            "Telegram:",
            response.status_code,
        )

        if response.status_code != 200:
            print(response.text)

        return response.status_code == 200

    except Exception as e:
        print(e)
        return False


def send_signals(signals, report):
    if not signals:
        from report import format_no_signal_message

        send_message(
            format_no_signal_message(report)
        )
        return

    from report import format_signal_message

    for signal in signals:
        send_message(
            format_signal_message(signal)
        )
