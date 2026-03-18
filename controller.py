def controller(row: pd.Series, pred: float, threshold: float = 0.001) -> dict:
    is_trending = row["is_trending"] == 1
    is_ranging = not is_trending
    is_breakout = row["vol_ratio_5_30"] > 1.1

    # raw regime signals only
    trend_long = (
        (pred > threshold)
        and is_trending
        and is_breakout
    )
    trend_short = (
        (pred < -threshold)
        and is_trending
        and is_breakout
    )

    meanrev_long = (
        (pred > threshold)
        and is_ranging
        and (not is_breakout)
        and (row["dist_ma_15_z"] < -1.0)
    )
    meanrev_short = (
        (pred < -threshold)
        and is_ranging
        and (not is_breakout)
        and (row["dist_ma_15_z"] > 1.0)
    )

    long_signal_raw = trend_long or meanrev_long
    short_signal_raw = trend_short or meanrev_short

    long_confirm = row["imbalance_5"] > 0.05
    short_confirm = row["imbalance_5"] < -0.05

    long_signal = long_signal_raw and long_confirm
    short_signal = short_signal_raw and short_confirm

    position = 0.0
    reason = "no_trade"

    if trend_long and long_confirm:
        position = 1.0
        reason = "trend_long"
    elif trend_short and short_confirm:
        position = -1.0
        reason = "trend_short"
    elif meanrev_long and long_confirm:
        position = 0.5
        reason = "meanrev_long"
    elif meanrev_short and short_confirm:
        position = -0.5
        reason = "meanrev_short"

    return {
        "signal": 1 if position > 0 else (-1 if position < 0 else 0),
        "position": position,
        "reason": reason,
        "long_signal_raw": long_signal_raw,
        "short_signal_raw": short_signal_raw,
        "long_signal": long_signal,
        "short_signal": short_signal,
        "is_trending": is_trending,
        "is_breakout": is_breakout,
        "long_confirm": long_confirm,
        "short_confirm": short_confirm,
    }