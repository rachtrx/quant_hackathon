# flake8: noqa: F401
# isort: skip_file

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter

from constants import DATA_DIR, INTERVAL, MODEL_DIR, TARGET_HORIZON
from features import add_features


class MlSignalStrategySimple(IStrategy):
    INTERFACE_VERSION = 3

    can_short = False
    timeframe = INTERVAL

    minimal_roi = {}
    stoploss = -0.99
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 800

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    MODEL_TYPE = "xgb"
    TARGET_HORIZON = TARGET_HORIZON

    # =========================
    # Much smaller hyperopt space
    # =========================
    buy_threshold = DecimalParameter(0.005, 0.060, default=0.020, decimals=3, space="buy", optimize=True, load=True)
    pred_hist_window = IntParameter(10, 80, default=30, space="buy", optimize=True, load=True)
    pred_quantile = DecimalParameter(0.50, 0.90, default=0.65, decimals=2, space="buy", optimize=True, load=True)

    min_imbalance = DecimalParameter(-0.05, 0.10, default=-0.01, decimals=2, space="buy", optimize=True, load=True)
    max_range_pos = DecimalParameter(0.70, 0.98, default=0.92, decimals=2, space="buy", optimize=True, load=True)
    min_rsi = DecimalParameter(40.0, 60.0, default=48.0, decimals=1, space="buy", optimize=True, load=True)

    sell_threshold = DecimalParameter(-0.08, 0.02, default=0.000, decimals=3, space="sell", optimize=True, load=True)

    def bot_start(self, **kwargs) -> None:
        self._artifact_cache: dict[str, dict[str, Any]] = {}
        self._meta_cache: dict[str, dict[str, Any]] = {}
        self._raw_cache: dict[str, pd.DataFrame] = {}

        self.model_root = Path(MODEL_DIR) / self.MODEL_TYPE
        self.raw_root = Path(DATA_DIR)

    def _pair_to_symbol(self, pair: str) -> str:
        return pair.replace("/", "").replace(":", "")

    def _load_pair_artifacts(self, pair: str) -> tuple[dict[str, Any], dict[str, Any]]:
        symbol = self._pair_to_symbol(pair)

        if symbol in self._artifact_cache:
            return self._artifact_cache[symbol], self._meta_cache[symbol]

        meta_path = self.model_root / f"{symbol}__h{self.TARGET_HORIZON}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta file: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_path = Path(meta["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        artifacts = joblib.load(model_path)

        required_keys = ["calibrator", "selected_features"]
        missing_keys = [k for k in required_keys if k not in artifacts]
        if missing_keys:
            raise ValueError(f"Artifact file {model_path} missing keys: {missing_keys}")

        meta["test_start_time"] = pd.to_datetime(meta["test_start_time"], utc=True)
        meta["test_end_time"] = pd.to_datetime(meta["test_end_time"], utc=True)

        self._artifact_cache[symbol] = artifacts
        self._meta_cache[symbol] = meta
        return artifacts, meta

    def _load_raw_pair_data(self, pair: str) -> pd.DataFrame:
        symbol = self._pair_to_symbol(pair)

        if symbol in self._raw_cache:
            return self._raw_cache[symbol]

        raw_path = self.raw_root / f"{symbol}_{INTERVAL}.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw parquet for {symbol}: {raw_path}")

        raw_df = pd.read_parquet(raw_path).copy()

        if "open_time" not in raw_df.columns:
            raise ValueError(f"{raw_path} must contain 'open_time'.")

        raw_df["date"] = pd.to_datetime(raw_df["open_time"], utc=True)

        candidate_cols = [
            "open_time",   # keep this
            "date",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
        keep_cols = [c for c in candidate_cols if c in raw_df.columns]

        raw_df = (
            raw_df[keep_cols]
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        self._raw_cache[symbol] = raw_df
        return raw_df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        artifacts, meta = self._load_pair_artifacts(pair)

        calibrator = artifacts["calibrator"]
        feature_cols = artifacts["selected_features"]

        test_start_time = meta["test_start_time"]
        test_end_time = meta["test_end_time"]

        out = dataframe.copy()
        df = dataframe.copy()

        out["date"] = pd.to_datetime(out["date"], utc=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)

        out["in_test_window"] = (
            (out["date"] >= test_start_time) &
            (out["date"] <= test_end_time)
        )

        raw_df = self._load_raw_pair_data(pair)
        df = df.merge(raw_df, on="date", how="left")

        feat_df, _ = add_features(df, target_horizon=self.TARGET_HORIZON)

        default_num_cols = [
            "pred_proba",
            "pred",
            "threshold",
            "rsi",
            "macd",
            "macdsignal",
            "macdhist",
            "range_pos_20",
            "imbalance_5",
        ]
        default_bool_cols = [
            "buy_signal",
            "sell_signal",
        ]

        if feat_df.empty:
            for c in default_num_cols:
                out[c] = np.nan
            for c in default_bool_cols:
                out[c] = False
            out["signal"] = 0
            out["position"] = 0.0
            out["reason"] = ""
            return out

        missing_feature_cols = [c for c in feature_cols if c not in feat_df.columns]
        if missing_feature_cols:
            raise ValueError(f"Missing feature columns required by model: {missing_feature_cols}")

        valid_mask = ~feat_df[feature_cols].isna().any(axis=1)
        feat_valid = feat_df.loc[valid_mask].copy()

        if feat_valid.empty:
            for c in default_num_cols:
                out[c] = np.nan
            for c in default_bool_cols:
                out[c] = False
            out["signal"] = 0
            out["position"] = 0.0
            out["reason"] = ""
            return out

        X = feat_valid[feature_cols]
        feat_valid["pred_proba"] = calibrator.predict_proba(X)[:, 1]
        feat_valid["pred"] = feat_valid["pred_proba"] - 0.5

        rolling_window = int(self.pred_hist_window.value)
        q = float(self.pred_quantile.value)
        fallback_threshold = float(self.buy_threshold.value)

        feat_valid["threshold"] = (
            feat_valid["pred"]
            .abs()
            .shift(1)
            .rolling(rolling_window)
            .quantile(q)
        ).fillna(fallback_threshold)

        feat_valid["rsi"] = ta.RSI(feat_valid, timeperiod=14)

        macd = ta.MACD(feat_valid)
        feat_valid["macd"] = macd["macd"]
        feat_valid["macdsignal"] = macd["macdsignal"]
        feat_valid["macdhist"] = macd["macdhist"]

        roll_low = feat_valid["low"].rolling(20).min()
        roll_high = feat_valid["high"].rolling(20).max()
        range_span = (roll_high - roll_low).replace(0, np.nan)
        feat_valid["range_pos_20"] = (feat_valid["close"] - roll_low) / range_span

        # -------------------------
        # SIMPLE ENTRY LOGIC
        # -------------------------
        model_ok = feat_valid["pred"] > feat_valid["threshold"]
        imbalance_ok = feat_valid["imbalance_5"] > float(self.min_imbalance.value)
        rsi_ok = feat_valid["rsi"] > float(self.min_rsi.value)
        not_too_extended = feat_valid["range_pos_20"] < float(self.max_range_pos.value)

        feat_valid["buy_signal"] = model_ok & imbalance_ok & rsi_ok & not_too_extended

        # Sell only on signal deterioration
        feat_valid["sell_signal"] = feat_valid["pred"] < float(self.sell_threshold.value)

        feat_valid["signal"] = feat_valid["buy_signal"].astype(int)
        feat_valid["position"] = np.where(feat_valid["buy_signal"], 1.0, 0.0)

        feat_valid["reason"] = np.where(
            feat_valid["buy_signal"],
            "model_long_simple",
            "no_trade"
        )

        feat_valid = feat_valid.set_index("date", drop=False)

        for c in default_num_cols:
            out[c] = out["date"].map(feat_valid[c])

        for c in default_bool_cols:
            out[c] = out["date"].map(feat_valid[c]).astype("boolean").fillna(False).astype(bool)

        out["signal"] = out["date"].map(feat_valid["signal"]).fillna(0).astype(int)
        out["position"] = out["date"].map(feat_valid["position"]).fillna(0.0).astype(float)
        out["reason"] = out["date"].map(feat_valid["reason"]).fillna("")

        return out

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        df["enter_long"] = 0
        df["enter_tag"] = None

        entry_cond = (
            (df["in_test_window"]) &
            (df["volume"] > 0) &
            (df["buy_signal"])
        )

        df.loc[entry_cond, "enter_long"] = 1
        df.loc[entry_cond, "enter_tag"] = "model_long_simple"

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        df["exit_long"] = 0
        df["exit_tag"] = None

        exit_cond = (
            (df["volume"] > 0) &
            (
                (df["sell_signal"]) |
                (~df["in_test_window"])
            )
        )

        df.loc[df["sell_signal"] & (df["volume"] > 0), "exit_tag"] = "signal_off"
        df.loc[(~df["in_test_window"]) & (df["volume"] > 0), "exit_tag"] = "test_window_exit"
        df.loc[exit_cond, "exit_long"] = 1

        return df