# user_data/strategies/MlSignalStrategy.py

# flake8: noqa: F401
# isort: skip_file

from datetime import timedelta
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy

# --- your project imports ---
# Make sure these files are importable from where freqtrade runs.
from features import add_features
from controller import controller


class MlSignalStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Spot / long-only
    can_short = False
    timeframe = "1m"

    # Keep ROI/SL out of the way initially so exits are driven by your logic.
    minimal_roi = {"0": 1000}
    stoploss = -0.99
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 400

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ---- Your ML settings ----
    MODEL_TYPE = "rf"       # or "xgb"
    TARGET_HORIZON = 5
    MIN_PRED_HISTORY = 100
    DEFAULT_THRESHOLD = 0.001
    PRED_HIST_WINDOW = 500

    # Set True if you want a forced sell after TARGET_HORIZON candles/minutes.
    USE_TIME_EXIT = False

    def bot_start(self, **kwargs) -> None:
        self._model_cache: dict[str, Any] = {}
        self._feature_cache: dict[str, list[str]] = {}
        self._meta_cache: dict[str, dict[str, Any]] = {}
        self.model_root = Path("models") / self.MODEL_TYPE
    
    def _pair_to_symbol(self, pair: str) -> str:
        # BTC/USDT -> BTCUSDT
        return pair.replace("/", "").replace(":", "")

    def _load_pair_artifacts(self, pair: str) -> tuple[Any, list[str], dict[str, Any]]:
        symbol = self._pair_to_symbol(pair)

        if symbol in self._model_cache:
            return (
                self._model_cache[symbol],
                self._feature_cache[symbol],
                self._meta_cache[symbol],
            )

        meta_path = self.model_root / f"{symbol}__h{self.TARGET_HORIZON}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta file: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        model_path = Path(meta["model_path"])
        features_path = Path(meta["feature_cols_path"])

        model = joblib.load(model_path)

        with open(features_path, "r") as f:
            feature_cols = json.load(f)

        meta["test_start_time"] = pd.to_datetime(meta["test_start_time"], utc=True)
        meta["test_end_time"] = pd.to_datetime(meta["test_end_time"], utc=True)

        self._model_cache[symbol] = model
        self._feature_cache[symbol] = feature_cols
        self._meta_cache[symbol] = meta

        return model, feature_cols, meta

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        model, feature_cols, meta = self._load_pair_artifacts(pair)

        test_start_time = meta["test_start_time"]
        test_end_time = meta["test_end_time"]

        df = dataframe.copy()
        out = dataframe.copy()

        if "date" not in out.columns:
            raise ValueError("Freqtrade dataframe is expected to contain a 'date' column.")

        df["date"] = pd.to_datetime(df["date"], utc=True)
        out["date"] = pd.to_datetime(out["date"], utc=True)

        out["in_test_window"] = (
            (out["date"] >= test_start_time) &
            (out["date"] <= test_end_time)
        )

        feat_df, _ = add_features(df, target_horizon=self.TARGET_HORIZON)

        extra_cols = [
            "pred",
            "threshold",
            "signal",
            "position",
            "reason",
            "long_signal_raw",
            "long_signal",
            "is_trending",
            "is_breakout",
            "long_confirm",
        ]
        for c in extra_cols:
            out[c] = np.nan

        if feat_df.empty:
            return out

        valid_mask = ~feat_df[feature_cols].isna().any(axis=1)
        feat_valid = feat_df.loc[valid_mask].copy()

        if feat_valid.empty:
            return out

        X = feat_valid[feature_cols]
        feat_valid["pred"] = model.predict(X)

        feat_valid["threshold"] = (
            feat_valid["pred"]
            .abs()
            .shift(1)
            .rolling(self.PRED_HIST_WINDOW, min_periods=self.MIN_PRED_HISTORY)
            .quantile(0.80)
        )
        feat_valid["threshold"] = feat_valid["threshold"].fillna(self.DEFAULT_THRESHOLD)

        def _apply_controller(row: pd.Series) -> pd.Series:
            d = controller(row, float(row["pred"]), threshold=float(row["threshold"]))
            return pd.Series({
                "signal": d.get("signal", 0),
                "position": d.get("position", 0),
                "reason": d.get("reason", ""),
                "long_signal_raw": d.get("long_signal_raw", False),
                "long_signal": d.get("long_signal", False),
                "is_trending": d.get("is_trending", False),
                "is_breakout": d.get("is_breakout", False),
                "long_confirm": d.get("long_confirm", False),
            })

        ctrl_cols = feat_valid.apply(_apply_controller, axis=1)
        feat_valid = pd.concat([feat_valid, ctrl_cols], axis=1)

        if "date" in feat_valid.columns:
            feat_trim = feat_valid[["date"] + extra_cols].copy()
            feat_trim["date"] = pd.to_datetime(feat_trim["date"], utc=True)

            out = out.merge(feat_trim, on="date", how="left", suffixes=("", "_ml"))
            for c in extra_cols:
                mlc = f"{c}_ml"
                if mlc in out.columns:
                    out[c] = out[mlc]
                    out.drop(columns=[mlc], inplace=True)
        else:
            common_idx = feat_valid.index.intersection(out.index)
            out.loc[common_idx, extra_cols] = feat_valid.loc[common_idx, extra_cols]

        out["signal"] = out["signal"].fillna(0).astype(int)
        out["position"] = out["position"].fillna(0).astype(int)

        bool_cols = [
            "in_test_window",
            "long_signal_raw",
            "long_signal",
            "is_trending",
            "is_breakout",
            "long_confirm",
        ]
        for c in bool_cols:
            out[c] = out[c].fillna(False).astype(bool)

        out["reason"] = out["reason"].fillna("")

        return out

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        df["enter_long"] = 0
        df["enter_tag"] = None

        df.loc[
            (
                (df["in_test_window"]) &
                (df["position"] == 1) &
                (df["volume"] > 0)
            ),
            ["enter_long", "enter_tag"]
        ] = (1, "ml_long")

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        df["exit_long"] = 0
        df["exit_tag"] = None

        df.loc[
            (
                (
                    (df["position"] <= 0) |
                    (~df["long_signal"]) |
                    (~df["in_test_window"])
                ) &
                (df["volume"] > 0)
            ),
            ["exit_long", "exit_tag"]
        ] = (1, "ml_exit_long")

        return df

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs
    ):
        """
        Optional fixed-time sell after TARGET_HORIZON minutes.
        Since timeframe is 1m, TARGET_HORIZON maps directly to minutes.
        """
        if not self.USE_TIME_EXIT:
            return None

        max_duration = timedelta(minutes=self.TARGET_HORIZON)

        if current_time - trade.open_date_utc >= max_duration:
            return "time_exit"

        return None