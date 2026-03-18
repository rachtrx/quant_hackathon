from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def calc_trend_strength(df: pd.DataFrame) -> pd.Series:
    returns = df["close"].pct_change(1)
    mean_20 = returns.rolling(20).mean()
    std_20 = returns.rolling(20).std()
    return mean_20 / (std_20 + EPS)


def calc_vol_regime_ratio(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].pct_change(1)
    vol_short = ret.rolling(10).std()
    vol_long = ret.rolling(50).std()
    return vol_short / (vol_long + EPS)


def calc_is_trending(trend_strength: pd.Series) -> pd.Series:
    thresh = trend_strength.abs().rolling(200).quantile(0.7)
    return (trend_strength.abs() > thresh).astype(int)


def calc_is_high_vol(vol_ratio: pd.Series) -> pd.Series:
    thresh = vol_ratio.rolling(100).quantile(0.7)
    return (vol_ratio > thresh).astype(int)


def add_features(df: pd.DataFrame, target_horizon: int) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    df["ret_1"] = df["close"].pct_change(1)

    mom_windows = [3, 5, 10, 15]
    ret_windows = [3, 5, 15, 30, 60]
    vol_windows = [5, 15, 30, 60]
    ma_windows = [5, 15, 30, 60]

    df["target"] = df["close"].shift(-target_horizon) / df["close"] - 1.0
    df[f"target_ret_fwd_{target_horizon}"] = df["target"]

    for w in ret_windows:
        df[f"ret_{w}"] = df["close"].pct_change(w)

    for w in mom_windows:
        df[f"mom_{w}"] = df["close"].pct_change(w)

    df["volume_mom_5"] = df["volume"].pct_change(5)

    for w in vol_windows:
        df[f"vol_{w}"] = df["ret_1"].rolling(w).std()

    df["vol_ratio_5_30"] = df["vol_5"] / (df["vol_30"] + EPS)
    df["breakout_active"] = (df["vol_ratio_5_30"] > 1.2).astype(int)

    for w in ma_windows:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        df[f"dist_ma_{w}"] = df["close"] / (df[f"ma_{w}"] + EPS) - 1.0

    dist_ma_15_mean_60 = df["dist_ma_15"].rolling(60).mean()
    dist_ma_15_std_60 = df["dist_ma_15"].rolling(60).std()
    df["dist_ma_15_z"] = (df["dist_ma_15"] - dist_ma_15_mean_60) / (dist_ma_15_std_60 + EPS)

    spread = (df["high"] - df["low"]).replace(0, np.nan)

    df["bar_range"] = (df["high"] - df["low"]) / (df["close"] + EPS)
    df["hl_spread"] = df["bar_range"]
    df["co_spread"] = (df["close"] - df["open"]) / (df["open"] + EPS)
    df["close_pos_in_bar"] = (df["close"] - df["low"]) / spread

    df["range_5"] = df["bar_range"].rolling(5).mean()
    df["range_15"] = df["bar_range"].rolling(15).mean()
    df["range_ratio"] = df["range_5"] / (df["range_15"] + EPS)

    df["volume_ret_1"] = df["volume"].pct_change(1)

    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_std_20"] = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - df["volume_ma_20"]) / (df["volume_std_20"] + EPS)

    volume_nonzero = df["volume"].replace(0, np.nan)

    df["taker_sell_base_asset_volume"] = df["volume"] - df["taker_buy_base_asset_volume"]
    df["taker_buy_ratio"] = df["taker_buy_base_asset_volume"] / volume_nonzero

    df["imbalance"] = (
        df["taker_buy_base_asset_volume"] - df["taker_sell_base_asset_volume"]
    ) / volume_nonzero

    df["imbalance_5"] = df["imbalance"].rolling(5).mean()
    df["imbalance_15"] = df["imbalance"].rolling(15).mean()

    df["trend_strength"] = calc_trend_strength(df)
    df["vol_regime_ratio"] = calc_vol_regime_ratio(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_cols = [
        "mom_3", "mom_5", "mom_10", "mom_15",
        "volume_mom_5",
        "dist_ma_5", "dist_ma_15", "dist_ma_30", "dist_ma_15_z",
        "vol_5", "vol_15", "vol_30", "vol_ratio_5_30",
        "bar_range", "range_5", "range_15", "range_ratio",
        "trend_strength", "vol_regime_ratio",
        "imbalance_5", "imbalance_15",
        "volume_z",
    ]

    return df, feature_cols