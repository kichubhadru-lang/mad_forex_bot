from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPORT_DIR = Path("reports")


def ensure_report_directory() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def format_reason_lines(reasons: List[str]) -> str:
    if not reasons:
        return "• No strong confirmations"

    return "\n".join(
        f"✅ {reason}"
        for reason in reasons
        if "penalty" not in reason.lower()
    )


def format_signal_message(signal: Dict) -> str:
    action_emoji = "🟢" if signal["action"] == "BUY" else "🔴"

    return (
        "🏆 FOREX ELITE SIGNAL\n\n"
        f"Pair: {signal['pair']}\n"
        f"{action_emoji} Action: {signal['action']}\n\n"
        f"Confidence: {signal['score']}%\n"
        f"Class: {signal['confidence']}\n"
        f"Grade: {signal['grade']}\n\n"
        f"Weekly: {signal['weekly_trend']}\n"
        f"Daily: {signal['daily_trend']}\n"
        f"4H: {signal['h4_trend']}\n"
        f"Session: {signal['session']}\n\n"
        f"Entry: {signal['entry']}\n"
        f"SL: {signal['stop_loss']}\n"
        f"TP1: {signal['tp1']}\n"
        f"TP2: {signal['tp2']}\n"
        f"RR TP1: 1:{signal['rr_tp1']}\n"
        f"RR TP2: 1:{signal['rr_tp2']}\n\n"
        f"ADX: {signal['adx']}\n"
        f"RSI: {signal['rsi']}\n"
        f"ATR Expansion: "
        f"{'Yes' if signal['atr_expansion'] else 'No'}\n\n"
        "Reasons:\n"
        f"{format_reason_lines(signal['reasons'])}\n\n"
        "⚠️ Educational use only. Confirm manually."
    )


def format_no_signal_message(
    results: List[Dict],
    minimum_score: int = 90,
) -> str:
    now = datetime.now(timezone.utc)

    message = (
        "🏆 FOREX V2 ELITE SCAN\n\n"
        f"Date: {now.strftime('%d-%b-%Y %H:%M UTC')}\n"
        f"Pairs analyzed: {len(results)}\n"
        "Qualified signals: 0\n\n"
        "No elite setup currently.\n\n"
    )

    if results:
        top_results = sorted(
            results,
            key=lambda item: item["score"],
            reverse=True,
        )[:3]

        message += "Closest setups:\n\n"

        for index, result in enumerate(top_results, start=1):
            emoji = "🟢" if result["action"] == "BUY" else "🔴"

            message += (
                f"{index}. {result['pair']}\n"
                f"{emoji} Bias: {result['action']}\n"
                f"Score: {result['score']}/100\n"
                f"Grade: {result['grade']}\n"
                f"4H Trend: {result['h4_trend']}\n"
                f"ADX: {result['adx']}\n"
                f"RSI: {result['rsi']}\n\n"
            )

    message += f"Minimum live score: {minimum_score}/100"

    return message


def results_to_dataframe(
    results: List[Dict],
) -> pd.DataFrame:
    rows = []

    for result in results:
        rows.append(
            {
                "Pair": result["pair"],
                "Action": result["action"],
                "Score": result["score"],
                "Confidence": result["confidence"],
                "Grade": result["grade"],
                "Qualified": result["qualified"],
                "Weekly Trend": result["weekly_trend"],
                "Daily Trend": result["daily_trend"],
                "4H Trend": result["h4_trend"],
                "Session": result["session"],
                "Entry": result["entry"],
                "Stop Loss": result["stop_loss"],
                "TP1": result["tp1"],
                "TP2": result["tp2"],
                "RR TP1": result["rr_tp1"],
                "RR TP2": result["rr_tp2"],
                "ADX": result["adx"],
                "RSI": result["rsi"],
                "ATR Expansion": result["atr_expansion"],
                "Bullish BOS": result["bullish_bos"],
                "Bearish BOS": result["bearish_bos"],
                "Bullish CHoCH": result["bullish_choch"],
                "Bearish CHoCH": result["bearish_choch"],
                "Reasons": ", ".join(result["reasons"]),
            }
        )

    return pd.DataFrame(rows)


def save_scan_report(
    results: List[Dict],
) -> Path | None:
    if not results:
        print("No results available to save.")
        return None

    ensure_report_directory()

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M"
    )

    output_path = REPORT_DIR / f"forex_v2_scan_{timestamp}.csv"

    dataframe = results_to_dataframe(results)

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print("Report saved:", output_path)

    return output_path


def print_console_summary(
    results: List[Dict],
) -> None:
    if not results:
        print("No pairs analyzed.")
        return

    print("\n" + "=" * 72)
    print("FOREX V2 SCAN RESULTS")
    print("=" * 72)

    dataframe = results_to_dataframe(results)

    columns = [
        "Pair",
        "Action",
        "Score",
        "Grade",
        "Qualified",
        "4H Trend",
        "ADX",
        "RSI",
    ]

    print(
        dataframe[columns]
        .sort_values(
            "Score",
            ascending=False,
        )
        .to_string(index=False)
    )

    print("=" * 72)
