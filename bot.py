from __future__ import annotations

from report import (
    format_no_signal_message,
    format_signal_message,
    print_console_summary,
    save_scan_report,
)
from strategy import (
    MIN_LIVE_SCORE,
    get_qualified_signals,
    scan_all_pairs,
)
from telegram import send_message


MAX_TELEGRAM_SIGNALS = 3


def main() -> None:
    print("=" * 60)
    print("FOREX V2 ELITE BOT")
    print("=" * 60)

    results = scan_all_pairs()

    if not results:
        print("No pairs could be analyzed.")

        send_message(
            "🏆 FOREX V2 ELITE BOT\n\n"
            "No market data could be analyzed."
        )
        return

    print_console_summary(results)
    save_scan_report(results)

    qualified = get_qualified_signals(results)

    if not qualified:
        print("No elite signal found.")

        message = format_no_signal_message(
            results=results,
            minimum_score=MIN_LIVE_SCORE,
        )

        send_message(message)
        return

    qualified = sorted(
        qualified,
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = qualified[:MAX_TELEGRAM_SIGNALS]

    print("Qualified signals:", len(qualified))
    print("Signals being sent:", len(selected))

    for signal in selected:
        message = format_signal_message(signal)
        send_message(message)

    print("V2 scan completed.")


if __name__ == "__main__":
    main()
