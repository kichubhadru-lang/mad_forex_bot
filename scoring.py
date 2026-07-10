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

    # Weekly trend — 15 points
    if weekly_trend == "BULLISH":
        buy_score += 15
        buy_reasons.append("Weekly bullish")

    elif weekly_trend == "BEARISH":
        sell_score += 15
        sell_reasons.append("Weekly bearish")

    # Daily trend — 15 points
    if daily_trend == "BULLISH":
        buy_score += 15
        buy_reasons.append("Daily bullish")

    elif daily_trend == "BEARISH":
        sell_score += 15
        sell_reasons.append("Daily bearish")

    # 4H trend — 15 points
    if bool(h4_row.get("BULLISH_TREND", False)):
        buy_score += 15
        buy_reasons.append("4H bullish trend")

    if bool(h4_row.get("BEARISH_TREND", False)):
        sell_score += 15
        sell_reasons.append("4H bearish trend")

    # Break of Structure — 10 points
    if bool(h4_row.get("BULLISH_BOS", False)):
        buy_score += 10
        buy_reasons.append("Bullish BOS")

    if bool(h4_row.get("BEARISH_BOS", False)):
        sell_score += 10
        sell_reasons.append("Bearish BOS")

    # Change of Character — 10 points
    if bool(h4_row.get("BULLISH_CHOCH", False)):
        buy_score += 10
        buy_reasons.append("Bullish CHoCH")

    if bool(h4_row.get("BEARISH_CHOCH", False)):
        sell_score += 10
        sell_reasons.append("Bearish CHoCH")

    # Order block — maximum 10 points
    bullish_ob_strength = int(
        h4_row.get("BULLISH_OB_STRENGTH", 0)
    )

    bearish_ob_strength = int(
        h4_row.get("BEARISH_OB_STRENGTH", 0)
    )

    if bullish_ob_strength >= 8:
        buy_score += 10
        buy_reasons.append("Elite bullish order block")

    elif bullish_ob_strength >= 6:
        buy_score += 8
        buy_reasons.append("Strong bullish order block")

    elif bullish_ob_strength >= 4:
        buy_score += 5
        buy_reasons.append("Bullish order block")

    if bearish_ob_strength >= 8:
        sell_score += 10
        sell_reasons.append("Elite bearish order block")

    elif bearish_ob_strength >= 6:
        sell_score += 8
        sell_reasons.append("Strong bearish order block")

    elif bearish_ob_strength >= 4:
        sell_score += 5
        sell_reasons.append("Bearish order block")

    # Order-block retest bonus — 5 points
    if bool(
        h1_row.get(
            "BULLISH_OB_RETEST_RECENT",
            False,
        )
    ):
        buy_score += 5
        buy_reasons.append("1H bullish order-block retest")

    if bool(
        h1_row.get(
            "BEARISH_OB_RETEST_RECENT",
            False,
        )
    ):
        sell_score += 5
        sell_reasons.append("1H bearish order-block retest")

    # Fair Value Gap — maximum 10 points
    bullish_fvg_strength = int(
        h4_row.get("BULLISH_FVG_STRENGTH", 0)
    )

    bearish_fvg_strength = int(
        h4_row.get("BEARISH_FVG_STRENGTH", 0)
    )

    if bullish_fvg_strength >= 8:
        buy_score += 10
        buy_reasons.append("Elite bullish FVG")

    elif bullish_fvg_strength >= 6:
        buy_score += 8
        buy_reasons.append("Strong bullish FVG")

    elif bullish_fvg_strength >= 4:
        buy_score += 5
        buy_reasons.append("Bullish FVG")

    if bearish_fvg_strength >= 8:
        sell_score += 10
        sell_reasons.append("Elite bearish FVG")

    elif bearish_fvg_strength >= 6:
        sell_score += 8
        sell_reasons.append("Strong bearish FVG")

    elif bearish_fvg_strength >= 4:
        sell_score += 5
        sell_reasons.append("Bearish FVG")

    # FVG rejection bonus — 3 points
    if bool(h1_row.get("BULLISH_FVG_REJECTED", False)):
        buy_score += 3
        buy_reasons.append("1H bullish FVG rejection")

    if bool(h1_row.get("BEARISH_FVG_REJECTED", False)):
        sell_score += 3
        sell_reasons.append("1H bearish FVG rejection")

    # Liquidity sweep — maximum 5 points
    bullish_liquidity_strength = int(
        h4_row.get(
            "BULLISH_LIQUIDITY_STRENGTH",
            0,
        )
    )

    bearish_liquidity_strength = int(
        h4_row.get(
            "BEARISH_LIQUIDITY_STRENGTH",
            0,
        )
    )

    if bullish_liquidity_strength >= 8:
        buy_score += 5
        buy_reasons.append("Strong bullish liquidity sweep")

    elif bullish_liquidity_strength >= 6:
        buy_score += 3
        buy_reasons.append("Bullish liquidity sweep")

    if bearish_liquidity_strength >= 8:
        sell_score += 5
        sell_reasons.append("Strong bearish liquidity sweep")

    elif bearish_liquidity_strength >= 6:
        sell_score += 3
        sell_reasons.append("Bearish liquidity sweep")
            # Session quality — maximum 5 points
    session_score = int(
        h1_row.get("SESSION_SCORE", 0)
    )

    if session_score >= 10:
        buy_score += 5
        sell_score += 5
        buy_reasons.append("London-New York overlap")
        sell_reasons.append("London-New York overlap")

    elif session_score >= 8:
        buy_score += 4
        sell_score += 4
        buy_reasons.append("Preferred trading window")
        sell_reasons.append("Preferred trading window")

    elif session_score >= 5:
        buy_score += 2
        sell_score += 2
        buy_reasons.append("Active session")
        sell_reasons.append("Active session")

    # ADX strength — maximum 5 points
    adx = float(h4_row.get("ADX", 0))

    if adx >= 35:
        buy_score += 5
        sell_score += 5
        buy_reasons.append("Very strong ADX")
        sell_reasons.append("Very strong ADX")

    elif adx >= 25:
        buy_score += 3
        sell_score += 3
        buy_reasons.append("Strong ADX")
        sell_reasons.append("Strong ADX")

    elif adx >= 20:
        buy_score += 1
        sell_score += 1
        buy_reasons.append("Moderate ADX")
        sell_reasons.append("Moderate ADX")

    # ATR expansion — 3 points
    if bool(h4_row.get("ATR_EXPANSION", False)):
        buy_score += 3
        sell_score += 3
        buy_reasons.append("ATR expansion")
        sell_reasons.append("ATR expansion")

    # 1H trend confirmation — maximum 5 points
    if bool(h1_row.get("BULLISH_TREND", False)):
        buy_score += 5
        buy_reasons.append("1H bullish confirmation")

    if bool(h1_row.get("BEARISH_TREND", False)):
        sell_score += 5
        sell_reasons.append("1H bearish confirmation")

    # RSI direction confirmation and conflict penalties
    h4_rsi = float(h4_row.get("RSI", 50))
    h1_rsi = float(h1_row.get("RSI", 50))

    if 50 <= h4_rsi <= 68:
        buy_score += 4
        buy_reasons.append("Healthy 4H bullish RSI")

    if 32 <= h4_rsi <= 50:
        sell_score += 4
        sell_reasons.append("Healthy 4H bearish RSI")

    if 50 <= h1_rsi <= 68:
        buy_score += 2
        buy_reasons.append("Healthy 1H bullish RSI")

    if 32 <= h1_rsi <= 50:
        sell_score += 2
        sell_reasons.append("Healthy 1H bearish RSI")

    # Conflict penalty:
    # Very high RSI weakens a fresh SELL setup.
    # Very low RSI weakens a fresh BUY setup.
    if h4_rsi >= 65:
        sell_score -= 10
        sell_reasons.append(
            "Overbought RSI conflicts with SELL"
        )

    if h4_rsi <= 35:
        buy_score -= 10
        buy_reasons.append(
            "Oversold RSI conflicts with BUY"
        )

    if h1_rsi >= 70:
        sell_score -= 5
        sell_reasons.append(
            "1H RSI conflicts with SELL"
        )

    if h1_rsi <= 30:
        buy_score -= 5
        buy_reasons.append(
            "1H RSI conflicts with BUY"
        )

    # Higher-timeframe conflict penalties
    if (
        weekly_trend == "BULLISH"
        and daily_trend == "BEARISH"
    ):
        buy_score -= 8
        sell_score -= 8
        buy_reasons.append(
            "Weekly and daily trend conflict"
        )
        sell_reasons.append(
            "Weekly and daily trend conflict"
        )

    if (
        weekly_trend == "BEARISH"
        and daily_trend == "BULLISH"
    ):
        buy_score -= 8
        sell_score -= 8
        buy_reasons.append(
            "Weekly and daily trend conflict"
        )
        sell_reasons.append(
            "Weekly and daily trend conflict"
        )

    # Ranging-market penalty
    if bool(h4_row.get("RANGING_MARKET", False)):
        buy_score -= 15
        sell_score -= 15
        buy_reasons.append("Ranging-market penalty")
        sell_reasons.append("Ranging-market penalty")

    # Opposite-structure penalties
    if bool(h4_row.get("BEARISH_BOS", False)):
        buy_score -= 8
        buy_reasons.append(
            "Bearish BOS conflicts with BUY"
        )

    if bool(h4_row.get("BULLISH_BOS", False)):
        sell_score -= 8
        sell_reasons.append(
            "Bullish BOS conflicts with SELL"
        )

    if bool(h4_row.get("BEARISH_CHOCH", False)):
        buy_score -= 8
        buy_reasons.append(
            "Bearish CHoCH conflicts with BUY"
        )

    if bool(h4_row.get("BULLISH_CHOCH", False)):
        sell_score -= 8
        sell_reasons.append(
            "Bullish CHoCH conflicts with SELL"
        )

    # Clamp scores to 0–100
    buy_score = max(
        0,
        min(100, int(round(buy_score))),
    )

    sell_score = max(
        0,
        min(100, int(round(sell_score))),
    )

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
