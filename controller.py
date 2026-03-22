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

    threshold is the dynamic threshold from rolling history.
    params contains hyperoptable controller settings from the strategy.
    """
    params = params or {}

    # ----------------------------
    # Core thresholds / boosts
    # ----------------------------
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

    # ----------------------------
    # New confluence / anti-chase params
    # ----------------------------
    meanrev_range_max = float(params.get("meanrev_range_max", 0.35))
    trend_range_max = float(params.get("trend_range_max", 0.80))
    breakout_range_max = float(params.get("breakout_range_max", 0.90))

    meanrev_rsi_max = float(params.get("meanrev_rsi_max", 40.0))
    trend_rsi_min = float(params.get("trend_rsi_min", 50.0))
    trend_rsi_max = float(params.get("trend_rsi_max", 68.0))

    trend_macdhist_min = float(params.get("trend_macdhist_min", 0.0))
    breakout_macdhist_delta_min = float(params.get("breakout_macdhist_delta_min", 0.0))

    # Optional looseness for meanrev. Keep at 0.0 first.
    allow_meanrev_edge_relax = float(params.get("allow_meanrev_edge_relax", 0.0))

    # ----------------------------
    # Raw row inputs
    # ----------------------------
    is_trending = bool(row.get("is_trending", 0) == 1)
    is_ranging = not is_trending

    vol_ratio = float(row.get("vol_ratio_5_30", 1.0))
    dist_z = float(row.get("dist_ma_15_z", 0.0))
    imbalance_5 = float(row.get("imbalance_5", 0.0))

    rsi = float(row.get("rsi", float("nan")))
    rsi_slope = float(row.get("rsi_slope", 0.0))

    macd = float(row.get("macd", 0.0))
    macdsignal = float(row.get("macdsignal", 0.0))
    macdhist = float(row.get("macdhist", 0.0))
    macdhist_delta = float(row.get("macdhist_delta", 0.0))

    range_pos_20 = float(row.get("range_pos_20", 0.5))

    enable_breakout = bool(params.get("enable_breakout", True))
    enable_trend = bool(params.get("enable_trend", True))
    enable_meanrev = bool(params.get("enable_meanrev", True))

    # ----------------------------
    # Breakout / imbalance boosts
    # ----------------------------
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
    # - rolling threshold from recent prediction history
    # - static minimum threshold
    eff_threshold = max(float(threshold), buy_threshold)

    # ----------------------------
    # Regime classification
    # ----------------------------
    regime = "neutral"

    if is_ranging and (not is_breakout) and dist_z < meanrev_dist_z:
        regime = "meanrev"
    elif is_breakout and macd > macdsignal and rsi > trend_rsi_min:
        regime = "breakout"
    elif is_trending and macd > macdsignal and rsi > trend_rsi_min:
        regime = "trend"

    # ----------------------------
    # Raw setup conditions
    # ----------------------------
    trend_long_raw = (
        regime == "trend"
        and adjusted_pred > eff_threshold
    )

    breakout_long_raw = (
        regime == "breakout"
        and adjusted_pred > eff_threshold
    )

    meanrev_long_raw = (
        regime == "meanrev"
        and adjusted_pred > max(eff_threshold - allow_meanrev_edge_relax, 0.0)
    )

    # ----------------------------
    # Confluence filters
    # ----------------------------
    meanrev_confluence = (
        pd.notna(rsi)
        and (range_pos_20 < meanrev_range_max)
        and (rsi < meanrev_rsi_max)
        and (rsi_slope > 0)
        and (macdhist_delta > 0)
    )

    trend_confluence = (
        pd.notna(rsi)
        and (range_pos_20 < trend_range_max)
        and (rsi > trend_rsi_min)
        and (rsi < trend_rsi_max)
        and (macd > macdsignal)
        and (macdhist > trend_macdhist_min)
    )

    breakout_confluence = (
        pd.notna(rsi)
        and (range_pos_20 < breakout_range_max)
        and (rsi > trend_rsi_min)
        and (rsi < trend_rsi_max + 5.0)
        and (macd > macdsignal)
        and (macdhist_delta > breakout_macdhist_delta_min)
    )

    # ----------------------------
    # Final confirmation
    # ----------------------------
    long_confirm = imbalance_5 > confirm_min_imbalance

    trend_long = enable_trend and trend_long_raw and trend_confluence and long_confirm
    breakout_long = enable_breakout and breakout_long_raw and breakout_confluence and long_confirm
    meanrev_long = enable_meanrev and meanrev_long_raw and meanrev_confluence and long_confirm

    # ----------------------------
    # Position / reason
    # ----------------------------
    position = 0.0
    reason = "no_trade"

    if breakout_long:
        position = 1.0
        reason = "breakout_long"
    elif trend_long:
        position = 1.0
        reason = "trend_long"
    elif meanrev_long:
        position = meanrev_position_size
        reason = "meanrev_long"

    # ----------------------------
    # Exit warning flags
    # These are not forced exits by themselves;
    # strategy can combine them with hard exits.
    # ----------------------------
    meanrev_exit_warn = (
        reason == "meanrev_long"
        and (
            (range_pos_20 > 0.60)
            or (rsi > 58.0)
            or (macdhist_delta < 0)
        )
    )

    trend_exit_warn = (
        reason in {"trend_long", "breakout_long"}
        and (
            ((macdhist < 0) and (rsi < 50.0))
            or (macdhist_delta < 0)
        )
    )

    long_signal_raw = trend_long_raw or breakout_long_raw or meanrev_long_raw
    long_signal = position > 0

    return {
        # numeric
        "signal": 1 if position > 0 else 0,
        "position": position,
        "adjusted_pred": adjusted_pred,
        "breakout_boost": breakout_boost,
        "confirm_boost": confirm_boost,

        # booleans
        "long_signal_raw": long_signal_raw,
        "long_signal": long_signal,
        "is_trending": is_trending,
        "is_breakout": is_breakout,
        "long_confirm": long_confirm,
        "meanrev_exit_warn": bool(meanrev_exit_warn),
        "trend_exit_warn": bool(trend_exit_warn),

        # labels
        "regime": regime,
        "reason": reason,
    }