from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def classify_confidence(score: int) -> Tuple[str, str]:
    if score >= 95:
        return "Elite", "A+"
    if score >= 90:
        return "High Probability", "A"
    if score >= 85:
        return "Watchlist", "B+"
    if score >= 75:
        return "Developing", "B"

    return "Ignore", "C"


def calculate_setup_score(
    weekly_trend: str,
    daily_trend: str,
    h4_row: pd.Series,
    h1_row: pd.Series,
) -> Dict:
    buy_score = 0
    sell_score = 0

    buy_reasons: List[str] = []
    sell_reasons: List[str] = []

    # Weekly trend — 15
    if weekly_trend == "BULLISH":
        buy_score += 15
        buy_reasons.append("Weekly bullish")

    if weekly_trend == "BEARISH":
        sell_score += 15
        sell_reasons.append("Weekly bearish")

    # Daily trend — 15
    if daily_trend == "BULLISH":
        buy_score += 15
        buy_reasons.append("Daily bullish")

    if daily_trend == "BEARISH":
        sell_score += 15
        sell_reasons.append("Daily bearish")

    # 4H trend — 15
    if h4_row.get("BULLISH_TREND", False):
        buy_score += 15
        buy_reasons.append("4H bullish trend")

    if h4_row.get("BEARISH_TREND", False):
        sell_score += 15
        sell_reasons.append("4H bearish trend")

    # BOS — 10
    if h4_row.get("BULLISH_BOS", False):
        buy_score += 10
        buy_reasons.append("Bullish BOS")

    if h4_row.get("BEARISH_BOS", False):
        sell_score += 10
        sell_reasons.append("Bearish BOS")

    # CHoCH — 10
    if h4_row.get("BULLISH_CHOCH", False):
        buy_score += 10
        buy_reasons.append("Bullish CHoCH")

    if h4_row.get("BEARISH_CHOCH", False):
        sell_score += 10
        sell_reasons.append("Bearish CHoCH")

    # Order block — 10
    bullish_ob_strength = int(
        h4_row.get("BULLISH_OB_STRENGTH", 0)
    )
    bearish_ob_strength = int(
        h4_row.get("BEARISH_OB_STRENGTH", 0)
    )

    if bullish_ob_strength >= 6:
        buy_score += 10
        buy_reasons.append("Strong bullish order block")
    elif bullish_ob_strength >= 4:
        buy_score += 6
        buy_reasons.append("Bullish order block")

    if bearish_ob_strength >= 6:
        sell_score += 10
        sell_reasons.append("Strong bearish order block")
    elif bearish_ob_strength >= 4:
        sell_score += 6
        sell_reasons.append("Bearish order block")

    # FVG — 10
    bullish_fvg_strength = int(
        h4_row.get("BULLISH_FVG_STRENGTH", 0)
    )
    bearish_fvg_strength = int(
        h4_row.get("BEARISH_FVG_STRENGTH", 0)
    )

    if bullish_fvg_strength >= 6:
        buy_score += 10
        buy_reasons.append("Strong bullish FVG")
    elif bullish_fvg_strength >= 4:
        buy_score += 6
        buy_reasons.append("Bullish FVG")

    if bearish_fvg_strength >= 6:
        sell_score += 10
        sell_reasons.append("Strong bearish FVG")
    elif bearish_fvg_strength >= 4:
        sell_score += 6
        sell_reasons.append("Bearish FVG")

    # Liquidity sweep — 5
    bullish_liquidity = int(
        h4_row.get("BULLISH_LIQUIDITY_STRENGTH", 0)
    )
    bearish_liquidity = int(
        h4_row.get("BEARISH_LIQUIDITY_STRENGTH", 0)
    )

    if bullish_liquidity >= 6:
        buy_score += 5
        buy_reasons.append("Bullish liquidity sweep")

    if bearish_liquidity >= 6:
        sell_score += 5
        sell_reasons.append("Bearish liquidity sweep")

    # Session — 5
    session_score = int(
        h1_row.get("SESSION_SCORE", 0)
    )

    if session_score >= 8:
        buy_score += 5
        sell_score += 5
        buy_reasons.append("Preferred session")
        sell_reasons.append("Preferred session")
    elif session_score >= 5:
        buy_score += 3
        sell_score += 3
        buy_reasons.append("Active session")
        sell_reasons.append("Active session")

    # ADX — 3
    adx = float(h4_row.get("ADX", 0))

    if adx >= 25:
        buy_score += 3
        sell_score += 3
        buy_reasons.append("Strong ADX")
        sell_reasons.append("Strong ADX")

    # ATR expansion — 2
    if h4_row.get("ATR_EXPANSION", False):
        buy_score += 2
        sell_score += 2
        buy_reasons.append("ATR expansion")
        sell_reasons.append("ATR expansion")

    # 1H retest confirmation — 5
    if h1_row.get("BULLISH_OB_RETEST_RECENT", False):
        buy_score += 5
        buy_reasons.append("1H bullish retest")

    if h1_row.get("BEARISH_OB_RETEST_RECENT", False):
        sell_score += 5
        sell_reasons.append("1H bearish retest")

    # Avoid ranging market
    if h4_row.get("RANGING_MARKET", False):
        buy_score -= 15
        sell_score -= 15
        buy_reasons.append("Ranging-market penalty")
        sell_reasons.append("Ranging-market penalty")

    buy_score = max(0, min(100, buy_score))
    sell_score = max(0, min(100, sell_score))

    if buy_score >= sell_score:
        action = "BUY"
        score = buy_score
        reasons = buy_reasons
    else:
        action = "SELL"
        score = sell_score
        reasons = sell_reasons

    confidence, grade = classify_confidence(score)

    return {
        "action": action,
        "score": score,
        "confidence": confidence,
        "grade": grade,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "reasons": reasons,
    }
