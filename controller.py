import pandas as pd

def controller(row: pd.Series, pred: float, threshold: float = 0.03) -> dict:
    """
    pred is now an edge score, not raw probability.
    Example:
        prob = 0.58 -> pred = 0.08
        prob = 0.51 -> pred = 0.01
    """
    is_trending = row["is_trending"] == 1
    is_ranging = not is_trending

    vol_ratio = float(row.get("vol_ratio_5_30", 1.0))
    dist_z = float(row.get("dist_ma_15_z", 0.0))
    imbalance_5 = float(row.get("imbalance_5", 0.0))

    is_breakout = vol_ratio > 1.1

    breakout_boost = 1.0
    if vol_ratio > 1.2:
        breakout_boost = 1.15
    elif vol_ratio > 1.1:
        breakout_boost = 1.08

    confirm_boost = 1.0
    if imbalance_5 > 0.10:
        confirm_boost = 1.15
    elif imbalance_5 > 0.05:
        confirm_boost = 1.08
    elif imbalance_5 < -0.05:
        confirm_boost = 0.85

    adjusted_pred = pred * breakout_boost * confirm_boost

    trend_long = (
        is_trending
        and adjusted_pred > threshold
    )

    meanrev_long = (
        is_ranging
        and (not is_breakout)
        and dist_z < -1.0
        and adjusted_pred > threshold
    )

    long_signal_raw = trend_long or meanrev_long
    long_confirm = imbalance_5 > 0.0
    long_signal = long_signal_raw and long_confirm

    position = 0.0
    reason = "no_trade"

    if trend_long and long_confirm:
        position = 1.0
        reason = "trend_long"
    elif meanrev_long and long_confirm:
        position = 0.5
        reason = "meanrev_long"

    return {
        "signal": 1 if position > 0 else 0,
        "position": position,
        "reason": reason,
        "long_signal_raw": long_signal_raw,
        "long_signal": long_signal,
        "is_trending": is_trending,
        "is_breakout": is_breakout,
        "long_confirm": long_confirm,
        "adjusted_pred": adjusted_pred,
        "breakout_boost": breakout_boost,
        "confirm_boost": confirm_boost,
    }