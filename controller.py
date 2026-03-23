import pandas as pd
import numpy as np

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

    def _get_float(*keys, default=0.0) -> float:
        for k in keys:
            if k in row and pd.notna(row[k]):
                try:
                    return float(row[k])
                except Exception:
                    pass
        return float(default)

    def _get_bool_flag(key: str, default: bool = False) -> bool:
        val = row.get(key, int(default))
        try:
            return bool(val == 1 or val is True)
        except Exception:
            return default

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
    # Confluence / anti-chase params
    # ----------------------------
    meanrev_range_max = float(params.get("meanrev_range_max", 0.35))
    trend_range_max = float(params.get("trend_range_max", 0.80))
    breakout_range_max = float(params.get("breakout_range_max", 0.90))

    meanrev_rsi_max = float(params.get("meanrev_rsi_max", 40.0))
    trend_rsi_min = float(params.get("trend_rsi_min", 50.0))
    trend_rsi_max = float(params.get("trend_rsi_max", 68.0))
    breakout_rsi_max = float(params.get("breakout_rsi_max", trend_rsi_max + 5.0))

    trend_macdhist_min = float(params.get("trend_macdhist_min", 0.0))
    breakout_macdhist_delta_min = float(params.get("breakout_macdhist_delta_min", 0.0))

    allow_meanrev_edge_relax = float(params.get("allow_meanrev_edge_relax", 0.0))

    # breakout_requires_trending = bool(params.get("breakout_requires_trending", False))

    # ----------------------------
    # Raw row inputs
    # ----------------------------
    is_trending = _get_bool_flag("is_trending", False)
    is_ranging = not is_trending

    vol_ratio = _get_float("vol_ratio_5_30", default=1.0)
    dist_z = _get_float("dist_ma_15_z", default=0.0)
    imbalance_5 = _get_float("imbalance_5", default=0.0)

    rsi = _get_float("rsi", default=np.nan)
    rsi_slope = _get_float("rsi_slope", default=0.0)

    macd = _get_float("macd", default=0.0)
    macd_signal = _get_float("macd_signal", "macdsignal", default=0.0)
    macd_hist = _get_float("macd_hist", "macdhist", default=0.0)
    macdhist_delta = _get_float("macdhist_delta", default=0.0)

    range_pos_20 = _get_float("range_pos_20", "close_pos_in_bar", default=0.5)

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

    eff_threshold = max(float(threshold), buy_threshold)

    # ----------------------------
    # Regime classification
    # ----------------------------
    regime = "neutral"

    breakout_structure_ok = (
        macd > macd_signal
        and (pd.isna(rsi) or rsi > trend_rsi_min)
    )

    trend_structure_ok = (
        macd > macd_signal
        and (pd.isna(rsi) or rsi > trend_rsi_min)
    )

    if is_ranging and (not is_breakout) and dist_z < meanrev_dist_z:
        regime = "meanrev"
    elif is_breakout and breakout_structure_ok: # and (
    #     (not breakout_requires_trending) or
    #     is_trending
    # ):
        regime = "breakout"
    elif is_trending and trend_structure_ok:
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
        (range_pos_20 < meanrev_range_max)
        and (pd.isna(rsi) or rsi < meanrev_rsi_max)
        and (rsi_slope >= 0)
        and (macdhist_delta >= 0)
    )

    trend_confluence = (
        (range_pos_20 < trend_range_max)
        and (pd.isna(rsi) or ((rsi > trend_rsi_min) and (rsi < trend_rsi_max)))
        and (macd > macd_signal)
        and (macd_hist > trend_macdhist_min)
    )

    breakout_confluence = (
        (range_pos_20 < breakout_range_max)
        and (pd.isna(rsi) or ((rsi > trend_rsi_min) and (rsi < breakout_rsi_max)))
        and (macd > macd_signal)
        and (macdhist_delta > breakout_macdhist_delta_min)
    )

    # ----------------------------
    # Final confirmation
    # ----------------------------
    long_confirm = imbalance_5 > confirm_min_imbalance

    trend_long = trend_long_raw and trend_confluence and long_confirm
    breakout_long = breakout_long_raw and breakout_confluence and long_confirm
    meanrev_long = meanrev_long_raw and meanrev_confluence and long_confirm

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
    # ----------------------------
    meanrev_exit_warn = (
        reason == "meanrev_long"
        and (
            (range_pos_20 > 0.60)
            or (pd.notna(rsi) and rsi > 58.0)
            or (macdhist_delta < 0)
        )
    )

    trend_exit_warn = (
        reason in {"trend_long", "breakout_long"}
        and (
            ((macd_hist < 0) and (pd.isna(rsi) or rsi < 50.0))
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