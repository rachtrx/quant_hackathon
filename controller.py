# DETECT REGIME

# TRENDING REGIME
long_ok = (pred > threshold) & (df["is_trending"] == 1) & (df["imbalance_5"] > 0)
short_ok = (pred < -threshold) & (df["is_trending"] == 1) & (df["imbalance_5"] < 0)

# RANGING REGIME
momentum_long = (
    (pred > threshold)
    & (df["is_trending"] == 1)
    & breakout_ok
    & (df["imbalance_5"] > 0)
)

meanrev_long = (
    (pred > threshold)
    & (df["is_trending"] == 0)
    & (~breakout_ok)
    & (df["dist_ma_15_z"] < -1.0)
)

# momentum requires breakout_ok true, ranging (mean rev) requires breakout_ok false
breakout_ok = df["vol_ratio_5_30"] > 1.1

# long only if imbalance positive, short only if negative
long_confirm = df["imbalance_5"] > 0.05
short_confirm = df["imbalance_5"] < -0.05