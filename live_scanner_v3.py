from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from report import format_reason_lines, save_scan_report
from strategy import analyze_pair
from telegram import send_message


V3_PAIRS = {
    "XAUUSD": "GC=F",
    "USDJPY": "JPY=X",
    "USDCAD": "CAD=X",
}

V3_MIN_SCORE = 45
V3_DIRECTION = "BUY"
SEND_NO_SIGNAL_MESSAGE = True


def is_v3_signal(result: Dict) -> bool:
    return (
        result.get("action") == V3_DIRECTION
        and int(result.get("score", 0)) >= V3_MIN_SCORE
    )


def format_v3_signal(signal: Dict) -> str:
    return (
        "🏆 FOREX V3 PAPER SIGNAL\n\n"
        f"Pair: {signal['pair']}\n"
        "🟢 Action: BUY\n"
        f"Score: {signal['score']}/100\n"
        f"Grade: {signal['grade']}\n\n"
        f"Weekly trend: {signal['weekly_trend']}\n"
        f"Daily trend: {signal['daily_trend']}\n"
        f"4H trend: {signal['h4_trend']}\n"
        f"Session: {signal['session']}\n\n"
        f"Entry: {signal['entry']}\n"
        f"SL: {signal['stop_loss']}\n"
        f"TP1: {signal['tp1']}\n"
        f"TP2: {signal['tp2']}\n"
        f"RR TP1: 1:{signal['rr_tp1']}\n"
        f"RR TP2: 1:{signal['rr_tp2']}\n\n"
        f"ADX: {signal['adx']}\n"
        f"RSI: {signal['rsi']}\n\n"
        "Reasons:\n"
        f"{format_reason_lines(signal.get('reasons', []))}\n\n"
        "🧪 PAPER TEST ONLY — do not place a real-money trade automatically."
    )


def format_v3_no_signal(results: List[Dict]) -> str:
    now = datetime.now(timezone.utc)

    lines = [
        "🏆 FOREX V3 PAPER SCAN",
        "",
        f"Date: {now.strftime('%d-%b-%Y %H:%M UTC')}",
        f"Pairs analyzed: {len(results)}",
        "Qualified BUY signals: 0",
        "",
        "No V3 setup currently.",
        "",
        "Rules: USDCAD / USDJPY / XAUUSD, BUY only, score ≥ 45",
    ]

    if results:
        lines.extend(["", "Current readings:"])

        for result in sorted(
            results,
            key=lambda item: int(item.get("score", 0)),
            reverse=True,
        ):
            lines.extend(
                [
                    "",
                    (
                        f"{result['pair']}: "
                        f"{result['action']} — "
                        f"{result['score']}/100"
                    ),
                    (
                        f"4H: {result['h4_trend']} | "
                        f"ADX: {result['adx']} | "
                        f"RSI: {result['rsi']}"
                    ),
                ]
            )

    return "\n".join(lines)


def main() -> None:
    print("=" * 64)
    print("FOREX V3 PAPER SCANNER")
    print("Pairs: XAUUSD, USDJPY, USDCAD | BUY only | Score >= 45")
    print("=" * 64)

    results: List[Dict] = []

    for pair, ticker in V3_PAIRS.items():
        try:
            result = analyze_pair(
                pair=pair,
                ticker=ticker,
            )

            if result is None:
                print(f"{pair}: analysis returned no result")
                continue

            result["qualified"] = is_v3_signal(result)
            results.append(result)

            print(
                f"{pair}: "
                f"{result.get('action')} | "
                f"Score {result.get('score')}"
            )

        except Exception as exc:
            print(f"{pair}: error — {exc}")

    if results:
        save_scan_report(results)

    qualified = [
        result
        for result in results
        if is_v3_signal(result)
    ]

    qualified.sort(
        key=lambda item: int(item.get("score", 0)),
        reverse=True,
    )

    print(f"Analyzed: {len(results)}")
    print(f"Qualified V3 BUY signals: {len(qualified)}")

    if qualified:
        for signal in qualified:
            message = format_v3_signal(signal)
            print(message)
            send_message(message)

    elif SEND_NO_SIGNAL_MESSAGE:
        message = format_v3_no_signal(results)
        print(message)
        send_message(message)


if __name__ == "__main__":
    main()
