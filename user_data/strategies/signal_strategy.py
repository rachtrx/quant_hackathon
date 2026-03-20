# user_data/strategies/MlSignalStrategy.py

# flake8: noqa: F401
# isort: skip_file

from datetime import timedelta
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy

from constants import DATA_DIR, INTERVAL, MODEL_DIR, TARGET_HORIZON
from features import add_features
from controller import controller


class MlSignalStrategyOld(IStrategy):
    INTERFACE_VERSION = 3

    # Spot / long-only
    can_short = False
    timeframe = INTERVAL

    minimal_roi = {
        "0": 0.02
    }
    stoploss = -0.01
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 800

    use_exit_signal = False # True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    MODEL_TYPE = "xgb"
    TARGET_HORIZON = TARGET_HORIZON
    MIN_PRED_HISTORY = 100
    DEFAULT_THRESHOLD = 0.03
    PRED_HIST_WINDOW = 500

    USE_TIME_EXIT = False

    def bot_start(self, **kwargs) -> None:
        self._model_cache: dict[str, Any] = {}
        self._feature_cache: dict[str, list[str]] = {}
        self._meta_cache: dict[str, dict[str, Any]] = {}
        self._raw_cache: dict[str, pd.DataFrame] = {}

        # models/rf/...
        self.model_root = Path(MODEL_DIR) / self.MODEL_TYPE

        # IMPORTANT:
        # This must NOT be user_data/data/binance
        # Put your rich 11-column raw parquet files somewhere else.
        self.raw_root = Path(DATA_DIR)

    def _pair_to_symbol(self, pair: str) -> str:
        # BTC/USDT -> BTCUSDT
        return pair.replace("/", "").replace(":", "")

    def _pair_to_freqtrade_filename(self, pair: str) -> str:
        # BTC/USDT -> BTC_USDT
        return pair.replace("/", "_").replace(":", "_")

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

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_path = Path(meta["model_path"])
        features_path = Path(meta["feature_cols_path"])

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Missing feature cols file: {features_path}")

        model = joblib.load(model_path)

        with open(features_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

        meta["test_start_time"] = pd.to_datetime(meta["test_start_time"], utc=True)
        meta["test_end_time"] = pd.to_datetime(meta["test_end_time"], utc=True)

        self._model_cache[symbol] = model
        self._feature_cache[symbol] = feature_cols
        self._meta_cache[symbol] = meta

        return model, feature_cols, meta

    def _load_raw_pair_data(self, pair: str) -> pd.DataFrame:
        """
        Load the rich raw market parquet containing Binance-specific fields
        such as num_trades, quote_asset_volume, taker_buy_base_asset_volume, etc.

        These files must be stored OUTSIDE freqtrade's OHLCV data directory.
        """
        symbol = self._pair_to_symbol(pair)

        if symbol in self._raw_cache:
            return self._raw_cache[symbol]

        raw_path = self.raw_root / f"{symbol}_{INTERVAL}.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing raw parquet for {symbol}: {raw_path}\n"
                f"Put your rich raw files in {self.raw_root.resolve()}"
            )

        raw_df = pd.read_parquet(raw_path).copy()

        if "open_time" not in raw_df.columns:
            raise ValueError(f"{raw_path} must contain 'open_time'.")

        raw_df["date"] = pd.to_datetime(raw_df["open_time"], utc=True)

        # Keep only the extra fields you need from the raw dataframe.
        # Do NOT keep OHLCV duplicates from here since freqtrade already provides them.
        candidate_cols = [
            "date",
            "open_time",
            "close_time",
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

        # Merge raw Binance-specific fields from your separate rich parquet.
        raw_df = self._load_raw_pair_data(pair)
        df = df.merge(raw_df, on="date", how="left")

        # Optional debug check
        needed_extra_cols = [
            "num_trades",
            "quote_asset_volume",
            "taker_buy_base_asset_volume",
        ]
        missing_stats = {
            c: float(df[c].isna().mean()) if c in df.columns else 1.0
            for c in needed_extra_cols
        }
        print(f"[{pair}] merged raw-column missing ratios: {missing_stats}")

        feat_df, _ = add_features(df, target_horizon=self.TARGET_HORIZON)

        if feat_df.empty:
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
            for c in ["pred", "threshold"]:
                out[c] = np.nan
            for c in ["signal", "position"]:
                out[c] = 0
            out["reason"] = ""
            for c in ["long_signal_raw", "long_signal", "is_trending", "is_breakout", "long_confirm"]:
                out[c] = False
            return out

        # Ensure all trained feature columns exist
        missing_feature_cols = [c for c in feature_cols if c not in feat_df.columns]
        if missing_feature_cols:
            raise ValueError(
                f"Missing feature columns required by model: {missing_feature_cols}"
            )

        valid_mask = ~feat_df[feature_cols].isna().any(axis=1)
        feat_valid = feat_df.loc[valid_mask].copy()

        if feat_valid.empty:
            print(f"[{pair}] No valid rows after feature filtering.")
            for c in ["pred", "threshold"]:
                out[c] = np.nan
            for c in ["signal", "position"]:
                out[c] = 0
            out["reason"] = ""
            for c in ["long_signal_raw", "long_signal", "is_trending", "is_breakout", "long_confirm"]:
                out[c] = False
            return out

        X = feat_valid[feature_cols]
        feat_valid["pred_proba"] = model.predict_proba(X)[:, 1]
        # Convert probability into centered directional edge
        feat_valid["pred"] = feat_valid["pred_proba"] - 0.5

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

        for col in ctrl_cols.columns:
            feat_valid[col] = ctrl_cols[col]

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

        # Use date mapping instead of merge to avoid duplicate-column issues.
        feat_trim = (
            feat_valid.loc[:, ["date"] + extra_cols]
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")
        )

        # hard guard against duplicate columns
        if feat_trim.columns.duplicated().any():
            dupes = feat_trim.columns[feat_trim.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns in feat_trim: {dupes}")

        for c in ["pred", "threshold", "reason"]:
            out[c] = out["date"].map(feat_trim[c])

        out["signal"] = out["date"].map(feat_trim["signal"]).fillna(0).astype(int)
        out["position"] = out["date"].map(feat_trim["position"]).fillna(0.0).astype(float)

        for c in ["long_signal_raw", "long_signal", "is_trending", "is_breakout", "long_confirm"]:
            out[c] = out["date"].map(feat_trim[c]).astype("boolean").fillna(False).astype(bool)

        out["reason"] = out["reason"].fillna("")

        return out

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        df["enter_long"] = 0
        df["enter_tag"] = None

        df.loc[
            (
                (df["in_test_window"]) &
                (df["position"] > 0) &
                (df["volume"] > 0)
            ),
            ["enter_long", "enter_tag"]
        ] = (1, "ml_long")

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        df["exit_long"] = 0
        df["exit_tag"] = None

        # df.loc[
        #     (
        #         (
        #             (df["position"] <= 0) |
        #             (~df["long_signal"]) |
        #             (~df["in_test_window"])
        #         ) &
        #         (df["volume"] > 0)
        #     ),
        #     ["exit_long", "exit_tag"]
        # ] = (1, "ml_exit_long")

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
        if not self.USE_TIME_EXIT:
            return None

        max_duration = timedelta(minutes=self.TARGET_HORIZON)

        if current_time - trade.open_date_utc >= max_duration:
            return "time_exit"

        return None