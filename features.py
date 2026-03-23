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


def add_features(df: pd.DataFrame, target_horizon: int) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------
    fwd_ret = df["close"].shift(-target_horizon) / df["close"] - 1.0
    df[f"target_ret_fwd_{target_horizon}"] = fwd_ret
    df[f"target_{target_horizon}"] = (fwd_ret > 0).astype(int)

    # ------------------------------------------------------------------
    # Base return
    # ------------------------------------------------------------------
    df["ret_1"] = df["close"].pct_change(1)

    # ------------------------------------------------------------------
    # Momentum: keep a few horizons only
    # ------------------------------------------------------------------
    for w in [5, 15, 60]:
        df[f"mom_{w}"] = df["close"].pct_change(w, fill_method=None)

    # ------------------------------------------------------------------
    # Volume / activity momentum
    # ------------------------------------------------------------------
    df["volume_mom_5"] = df["volume"].pct_change(5, fill_method=None)
    df["num_trades_mom_5"] = df["num_trades"].pct_change(5, fill_method=None)

    # ------------------------------------------------------------------
    # Volatility: keep short, medium, and ratio
    # ------------------------------------------------------------------
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_30"] = df["ret_1"].rolling(30).std()
    df["vol_ratio_5_30"] = df["vol_5"] / (df["vol_30"] + EPS)

    # ------------------------------------------------------------------
    # Moving average distance: enough for mean reversion / trend position
    # ------------------------------------------------------------------
    for w in [15, 30]:
        ma = df["close"].rolling(w).mean()
        df[f"dist_ma_{w}"] = df["close"] / (ma + EPS) - 1.0

    # ------------------------------------------------------------------
    # Price action
    # ------------------------------------------------------------------
    spread = (df["high"] - df["low"]).replace(0, np.nan)

    df["bar_range"] = (df["high"] - df["low"]) / (df["close"] + EPS)
    df["co_spread"] = (df["close"] - df["open"]) / (df["open"] + EPS)
    df["close_pos_in_bar"] = (df["close"] - df["low"]) / spread

    df["range_5"] = df["bar_range"].rolling(5).mean()
    df["range_15"] = df["bar_range"].rolling(15).mean()
    df["range_ratio"] = df["range_5"] / (df["range_15"] + EPS)

    # ------------------------------------------------------------------
    # Activity z-scores
    # ------------------------------------------------------------------
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_std_20"] = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - df["volume_ma_20"]) / (df["volume_std_20"] + EPS)

    df["trades_ma_20"] = df["num_trades"].rolling(20).mean()
    df["trades_std_20"] = df["num_trades"].rolling(20).std()
    df["trades_z"] = (df["num_trades"] - df["trades_ma_20"]) / (df["trades_std_20"] + EPS)

    # ------------------------------------------------------------------
    # Order flow
    # ------------------------------------------------------------------
    volume_nonzero = df["volume"].replace(0, np.nan)

    df["taker_sell_base_asset_volume"] = df["volume"] - df["taker_buy_base_asset_volume"]
    df["taker_buy_ratio"] = df["taker_buy_base_asset_volume"] / volume_nonzero

    df["imbalance"] = (
        df["taker_buy_base_asset_volume"] - df["taker_sell_base_asset_volume"]
    ) / volume_nonzero

    df["imbalance_15"] = df["imbalance"].rolling(15).mean()

    imb_mean_30 = df["imbalance"].rolling(30).mean()
    imb_std_30 = df["imbalance"].rolling(30).std()
    df["imbalance_z"] = (df["imbalance"] - imb_mean_30) / (imb_std_30 + EPS)

    # ------------------------------------------------------------------
    # Regime features: keep raw continuous versions, not binary flags
    # ------------------------------------------------------------------
    df["trend_strength"] = calc_trend_strength(df)
    df["vol_regime_ratio"] = calc_vol_regime_ratio(df)

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------
    open_time = pd.to_datetime(df["open_time"])

    hour = open_time.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = open_time.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ------------------------------------------------------------------
    # ATR normalized
    # ------------------------------------------------------------------
    prev_close = df["close"].shift(1)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ),
    )
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_norm"] = df["atr_14"] / (df["close"] + EPS)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ------------------------------------------------------------------
    # Final feature list
    # ------------------------------------------------------------------
    feature_cols = [
        # momentum
        "mom_5",
        "mom_15",
        "mom_60",
        "macd_hist",

        # activity momentum
        "volume_mom_5",
        "num_trades_mom_5",

        # trend / mean reversion positioning
        "dist_ma_15",
        "dist_ma_30",

        # volatility
        "vol_5",
        "vol_30",
        "vol_ratio_5_30",
        "atr_norm",

        # price action
        "bar_range",
        "co_spread",
        "close_pos_in_bar",
        "range_ratio",

        # regime
        "trend_strength",
        "vol_regime_ratio",

        # order flow
        "imbalance",
        "imbalance_15",
        "imbalance_z",
        "taker_buy_ratio",

        # activity
        "volume_z",
        "trades_z",

        # time
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    return df, feature_cols