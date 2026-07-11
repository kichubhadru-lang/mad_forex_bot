from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def classify_confidence(score: int) -> Tuple[str, str]:
    if score >= 95:
        return "Elite", "A+"
    if score >= 90:
        return "High Probability", "A"
    if score >= 80:
        return "Watchlist", "B+"

    return "Rejected", "C"


def calculate_setup_score(
    weekly_trend: str,
    daily_trend: str,
    h4_row: pd.Series,
    h1_row: pd.Series,
) -> Dict:

    weekly_trend = str(weekly_trend).upper()
    daily_trend = str(daily_trend).upper()

    adx = float(h4_row.get("ADX", 0))
    h4_rsi = float(h4_row.get("RSI", 50))
    h1_rsi = float(h1_row.get("RSI", 50))

    bullish_bos = bool(h4_row.get("BULLISH_BOS", False))
    bearish_bos = bool(h4_row.get("BEARISH_BOS", False))

    bullish_choch = bool(h4_row.get("BULLISH_CHOCH", False))
    bearish_choch = bool(h4_row.get("BEARISH_CHOCH", False))

    bullish_structure = (
        bullish_bos
        or bullish_choch
        or int(h4_row.get("STRUCTURE", 0)) == 1
    )

    bearish_structure = (
        bearish_bos
        or bearish_choch
        or int(h4_row.get("STRUCTURE", 0)) == -1
    )

    bullish_ob_strength = int(
        h4_row.get("BULLISH_OB_STRENGTH", 0)
    )

    bearish_ob_strength = int(
        h4_row.get("BEARISH_OB_STRENGTH", 0)
    )

    bullish_ob_retest = bool(
        h1_row.get("BULLISH_OB_RETEST_RECENT", False)
    )

    bearish_ob_retest = bool(
        h1_row.get("BEARISH_OB_RETEST_RECENT", False)
    )

    bullish_ob_valid = (
        bullish_ob_strength >= 4
        or bullish_ob_retest
    )

    bearish_ob_valid = (
        bearish_ob_strength >= 4
        or bearish_ob_retest
    )

    bullish_rejections: List[str] = []
    bearish_rejections: List[str] = []

    # ========================================================
    # MANDATORY BUY FILTERS
    # ========================================================

    if weekly_trend != "BULLISH":
        bullish_rejections.append("Weekly trend not bullish")

    if daily_trend != "BULLISH":
        bullish_rejections.append("Daily trend not bullish")

    if not bullish_structure:
        bullish_rejections.append(
            "No bullish BOS, CHoCH or structure"
        )

    if not bullish_ob_valid:
        bullish_rejections.append(
            "No valid bullish order block"
        )

    if adx < 25:
        bullish_rejections.append("ADX below 25")

    # ========================================================
    # MANDATORY SELL FILTERS
    # ========================================================

    if weekly_trend != "BEARISH":
        bearish_rejections.append("Weekly trend not bearish")

    if daily_trend != "BEARISH":
        bearish_rejections.append("Daily trend not bearish")

    if not bearish_structure:
        bearish_rejections.append(
            "No bearish BOS, CHoCH or structure"
        )

    if not bearish_ob_valid:
        bearish_rejections.append(
            "No valid bearish order block"
        )

    if adx < 25:
        bearish_rejections.append("ADX below 25")

    buy_passed = len(bullish_rejections) == 0
    sell_passed = len(bearish_rejections) == 0

    # No valid direction
    if not buy_passed and not sell_passed:
        return {
            "action": "NO TRADE",
            "score": 0,
            "confidence": "Rejected",
            "grade": "C",
            "buy_score": 0,
            "sell_score": 0,
            "mandatory_passed": False,
            "reasons": [],
            "rejection_reasons": {
                "BUY": bullish_rejections,
                "SELL": bearish_rejections,
            },
        }

    # ========================================================
    # QUALITY SCORING
    # ========================================================

    buy_score = 0
    sell_score = 0

    buy_reasons: List[str] = []
    sell_reasons: List[str] = []

    # BOS — 15
    if bullish_bos:
        buy_score += 15
        buy_reasons.append("Bullish BOS")

    if bearish_bos:
        sell_score += 15
        sell_reasons.append("Bearish BOS")

    # CHoCH — 10
    if bullish_choch:
        buy_score += 10
        buy_reasons.append("Bullish CHoCH")

    if bearish_choch:
        sell_score += 10
        sell_reasons.append("Bearish CHoCH")

    # Order block — 20
    if bullish_ob_strength >= 8:
        buy_score += 20
        buy_reasons.append("Elite bullish order block")
    elif bullish_ob_strength >= 6:
        buy_score += 17
        buy_reasons.append("Strong bullish order block")
    elif bullish_ob_strength >= 4:
        buy_score += 14
        buy_reasons.append("Valid bullish order block")

    if bearish_ob_strength >= 8:
        sell_score += 20
        sell_reasons.append("Elite bearish order block")
    elif bearish_ob_strength >= 6:
        sell_score += 17
        sell_reasons.append("Strong bearish order block")
    elif bearish_ob_strength >= 4:
        sell_score += 14
        sell_reasons.append("Valid bearish order block")

    # FVG — 10
    bullish_fvg = int(
        h4_row.get("BULLISH_FVG_STRENGTH", 0)
    )

    bearish_fvg = int(
        h4_row.get("BEARISH_FVG_STRENGTH", 0)
    )

    if bullish_fvg >= 6:
        buy_score += 10
        buy_reasons.append("Strong bullish FVG")
    elif bullish_fvg >= 4:
        buy_score += 6
        buy_reasons.append("Bullish FVG")

    if bearish_fvg >= 6:
        sell_score += 10
        sell_reasons.append("Strong bearish FVG")
    elif bearish_fvg >= 4:
        sell_score += 6
        sell_reasons.append("Bearish FVG")

    # Liquidity sweep — 10
    bullish_liquidity = int(
        h4_row.get("BULLISH_LIQUIDITY_STRENGTH", 0)
    )

    bearish_liquidity = int(
        h4_row.get("BEARISH_LIQUIDITY_STRENGTH", 0)
    )

    if bullish_liquidity >= 6:
        buy_score += 10
        buy_reasons.append("Bullish liquidity sweep")

    if bearish_liquidity >= 6:
        sell_score += 10
        sell_reasons.append("Bearish liquidity sweep")

    # 1H retest — 10
    if bullish_ob_retest:
        buy_score += 10
        buy_reasons.append("1H bullish retest")

    if bearish_ob_retest:
        sell_score += 10
        sell_reasons.append("1H bearish retest")

    # EMA alignment — 10
    if bool(h4_row.get("BULLISH_TREND", False)):
        buy_score += 10
        buy_reasons.append("4H EMA alignment")

    if bool(h4_row.get("BEARISH_TREND", False)):
        sell_score += 10
        sell_reasons.append("4H EMA alignment")

    # ATR expansion — 5
    if bool(h4_row.get("ATR_EXPANSION", False)):
        buy_score += 5
        sell_score += 5
        buy_reasons.append("ATR expansion")
        sell_reasons.append("ATR expansion")

    # Session — 5
    session_score = int(h1_row.get("SESSION_SCORE", 0))

    if session_score >= 8:
        buy_score += 5
        sell_score += 5
        buy_reasons.append("Preferred trading session")
        sell_reasons.append("Preferred trading session")
    elif session_score >= 5:
        buy_score += 3
        sell_score += 3
        buy_reasons.append("Active trading session")
        sell_reasons.append("Active trading session")

    # RSI quality — 5
    if 50 <= h4_rsi <= 68 and 50 <= h1_rsi <= 70:
        buy_score += 5
        buy_reasons.append("Healthy bullish RSI")

    if 32 <= h4_rsi <= 50 and 30 <= h1_rsi <= 50:
        sell_score += 5
        sell_reasons.append("Healthy bearish RSI")

    # Base score for passing all mandatory filters
    if buy_passed:
        buy_score += 15
        buy_reasons.insert(0, "All mandatory BUY filters passed")
    else:
        buy_score = 0

    if sell_passed:
        sell_score += 15
        sell_reasons.insert(0, "All mandatory SELL filters passed")
    else:
        sell_score = 0

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
        "mandatory_passed": True,
        "reasons": reasons,
        "rejection_reasons": {
            "BUY": bullish_rejections,
            "SELL": bearish_rejections,
        },
    }
