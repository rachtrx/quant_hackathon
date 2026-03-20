import pandas as pd


def controller(
    row: pd.Series,
    pred: float,
    threshold: float = 0.03,
    params: dict | None = None,
) -> dict:
    """
    pred is a centered edge score, not raw probability.

    Example:
        prob = 0.58 -> pred = 0.08
        prob = 0.51 -> pred = 0.01

    `threshold` is the dynamic threshold from rolling history.
    `params` allows hyperoptable controller settings from the strategy.
    """
    params = params or {}

    buy_threshold = float(params.get("buy_threshold", 0.03))
    breakout_level = float(params.get("breakout_level", 1.10))
    strong_breakout_level = float(params.get("strong_breakout_level", 1.20))
    mild_breakout_boost = float(params.get("mild_breakout_boost", 1.08))
    strong_breakout_boost = float(params.get("strong_breakout_boost", 1.15))

    imbalance_soft = float(params.get("imbalance_soft", 0.05))
    imbalance_strong = float(params.get("imbalance_strong", 0.10))
    imbalance_negative = float(params.get("imbalance_negative", -0.05))
    confirm_soft_boost = float(params.get("confirm_soft_boost", 1.08))
    confirm_strong_boost = float(params.get("confirm_strong_boost", 1.15))
    negative_confirm_penalty = float(params.get("negative_confirm_penalty", 0.85))

    meanrev_dist_z = float(params.get("meanrev_dist_z", -1.0))
    confirm_min_imbalance = float(params.get("confirm_min_imbalance", 0.0))
    meanrev_position_size = float(params.get("meanrev_position_size", 0.5))

    is_trending = row["is_trending"] == 1
    is_ranging = not is_trending

    vol_ratio = float(row.get("vol_ratio_5_30", 1.0))
    dist_z = float(row.get("dist_ma_15_z", 0.0))
    imbalance_5 = float(row.get("imbalance_5", 0.0))

    is_breakout = vol_ratio > breakout_level

    breakout_boost = 1.0
    if vol_ratio > strong_breakout_level:
        breakout_boost = strong_breakout_boost
    elif vol_ratio > breakout_level:
        breakout_boost = mild_breakout_boost

    confirm_boost = 1.0
    if imbalance_5 > imbalance_strong:
        confirm_boost = confirm_strong_boost
    elif imbalance_5 > imbalance_soft:
        confirm_boost = confirm_soft_boost
    elif imbalance_5 < imbalance_negative:
        confirm_boost = negative_confirm_penalty

    adjusted_pred = pred * breakout_boost * confirm_boost

    # Use the stricter of:
    # - dynamic rolling threshold from recent prediction history
    # - static minimum threshold from hyperopt
    eff_threshold = max(float(threshold), buy_threshold)

    trend_long = (
        is_trending
        and adjusted_pred > eff_threshold
    )

    meanrev_long = (
        is_ranging
        and (not is_breakout)
        and dist_z < meanrev_dist_z
        and adjusted_pred > eff_threshold
    )

    long_signal_raw = trend_long or meanrev_long
    long_confirm = imbalance_5 > confirm_min_imbalance
    long_signal = long_signal_raw and long_confirm

    position = 0.0
    reason = "no_trade"

    if trend_long and long_confirm:
        position = 1.0
        reason = "trend_long"
    elif meanrev_long and long_confirm:
        position = meanrev_position_size
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