# flake8: noqa: F401
# isort: skip_file

from datetime import timedelta
import json
from pathlib import Path
from typing import Any
import talib.abstract as ta

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import (
    BooleanParameter,
    IStrategy,
    DecimalParameter,
    IntParameter,
)

from constants import DATA_DIR, INTERVAL, MODEL_DIR, TARGET_HORIZON
from features import add_features
from controller import controller


class MlSignalStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Spot / long-only
    can_short = False
    timeframe = INTERVAL

    minimal_roi = { }
    stoploss = -0.99
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 800

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    MODEL_TYPE = "xgb"
    TARGET_HORIZON = TARGET_HORIZON

    USE_TIME_EXIT = False

    buy_threshold = DecimalParameter(0.005, 0.080, default=0.015, decimals=3, space="buy", optimize=True, load=True)
    pred_hist_window = IntParameter(5, 100, default=10, space="buy", optimize=True, load=True)
    pred_quantile = DecimalParameter(0.40, 0.95, default=0.80, decimals=2, space="buy", optimize=True, load=True)
    breakout_level = DecimalParameter(1.00, 1.30, default=1.10, decimals=2, space="buy", optimize=True, load=True)
    strong_breakout_level = DecimalParameter(1.05, 1.60, default=1.20, decimals=2, space="buy", optimize=True, load=True)
    mild_breakout_boost = DecimalParameter(1.00, 1.20, default=1.08, decimals=2, space="buy", optimize=True, load=True)
    strong_breakout_boost = DecimalParameter(1.00, 1.30, default=1.15, decimals=2, space="buy", optimize=True, load=True)
    imbalance_soft = DecimalParameter(0.00, 0.15, default=0.05, decimals=2, space="buy", optimize=True, load=True)
    imbalance_strong = DecimalParameter(0.00, 0.25, default=0.10, decimals=2, space="buy", optimize=True, load=True)
    imbalance_negative = DecimalParameter(-0.20, 0.00, default=-0.05, decimals=2, space="buy", optimize=True, load=True)
    confirm_soft_boost = DecimalParameter(1.00, 1.20, default=1.08, decimals=2, space="buy", optimize=True, load=True)
    confirm_strong_boost = DecimalParameter(1.00, 1.30, default=1.15, decimals=2, space="buy", optimize=True, load=True)
    negative_confirm_penalty = DecimalParameter(0.70, 1.00, default=0.85, decimals=2, space="buy", optimize=True, load=True)
    meanrev_dist_z = DecimalParameter(-3.00, -0.50, default=-1.00, decimals=2, space="buy", optimize=True, load=True)
    confirm_min_imbalance = DecimalParameter(-0.10, 0.10, default=0.00, decimals=2, space="buy", optimize=True, load=True)
    meanrev_position_size = DecimalParameter(0.25, 1.00, default=0.50, decimals=2, space="buy", optimize=True, load=True)
    sell_pred_threshold = DecimalParameter(-0.10, 0.05, default=0.00, decimals=3, space="sell", optimize=True, load=True)
    sell_imbalance_threshold = DecimalParameter(-0.20, 0.10, default=0.00, decimals=2, space="sell", optimize=True, load=True)
    sell_dist_z_threshold = DecimalParameter(-0.50, 1.50, default=0.00, decimals=2, space="sell", optimize=True, load=True)

    trend_rsi_min = DecimalParameter(40.0, 60.0, default=50.0, decimals=1, space="buy")
    trend_rsi_max = DecimalParameter(60.0, 80.0, default=68.0, decimals=1, space="buy")

    meanrev_rsi_max = DecimalParameter(30.0, 60.0, default=40.0, decimals=1, space="buy")

    meanrev_range_max = DecimalParameter(0.2, 0.7, default=0.35, decimals=2, space="buy")
    trend_range_max = DecimalParameter(0.55, 0.95, default=0.80, decimals=2, space="buy")
    breakout_range_max = DecimalParameter(0.70, 0.99, default=0.90, decimals=2, space="buy")

    trend_macdhist_min = DecimalParameter(-0.01, 0.05, default=0.00, decimals=3, space="buy")
    breakout_macdhist_delta_min = DecimalParameter(-0.01, 0.05, default=0.00, decimals=3, space="buy")
    allow_meanrev_edge_relax = DecimalParameter(0.0, 0.05, default=0.00, decimals=2, space="buy")

    def bot_start(self, **kwargs) -> None:
        self._artifact_cache: dict[str, dict[str, Any]] = {}
        self._meta_cache: dict[str, dict[str, Any]] = {}
        self._raw_cache: dict[str, pd.DataFrame] = {}

        self.model_root = Path(MODEL_DIR) / self.MODEL_TYPE
        self.raw_root = Path(DATA_DIR)

    def _pair_to_symbol(self, pair: str) -> str:
        # BTC/USDT -> BTCUSDT
        return pair.replace("/", "").replace(":", "")

    def _pair_to_freqtrade_filename(self, pair: str) -> str:
        # BTC/USDT -> BTC_USDT
        return pair.replace("/", "_").replace(":", "_")

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

        if not isinstance(artifacts, dict):
            raise ValueError(
                f"Expected artifact dict in {model_path}, got {type(artifacts)}"
            )

        required_keys = ["base_model", "calibrator", "selected_features"]
        missing_keys = [k for k in required_keys if k not in artifacts]
        if missing_keys:
            raise ValueError(
                f"Artifact file {model_path} missing keys: {missing_keys}"
            )

        meta["test_start_time"] = pd.to_datetime(meta["test_start_time"], utc=True)
        meta["test_end_time"] = pd.to_datetime(meta["test_end_time"], utc=True)

        self._artifact_cache[symbol] = artifacts
        self._meta_cache[symbol] = meta

        return artifacts, meta

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
        artifacts, meta = self._load_pair_artifacts(pair)

        calibrator = artifacts["calibrator"]
        feature_cols = artifacts["selected_features"]

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

        empty_defaults_bool = [
            "long_signal_raw",
            "long_signal",
            "is_trending",
            "is_breakout",
            "long_confirm",
            "meanrev_exit_warn",
            "trend_exit_warn",
        ]

        empty_defaults_num = [
            "pred",
            "pred_proba",
            "threshold",

            "signal",
            "position",
            "adjusted_pred",
            "breakout_boost",
            "confirm_boost",
        ]

        if feat_df.empty:
            for c in empty_defaults_num:
                out[c] = np.nan if c not in ["signal", "position"] else 0
            out["signal"] = 0
            out["position"] = 0.0
            out["reason"] = ""
            for c in empty_defaults_bool:
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
            for c in empty_defaults_num:
                out[c] = np.nan if c not in ["signal", "position"] else 0
            out["signal"] = 0
            out["position"] = 0.0
            out["reason"] = ""
            for c in empty_defaults_bool:
                out[c] = False
            return out

        X = feat_valid[feature_cols]
        feat_valid["pred_proba"] = calibrator.predict_proba(X)[:, 1]
        # Convert probability into centered directional edge
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
        )
        feat_valid["threshold"] = feat_valid["threshold"].fillna(fallback_threshold)

        feat_valid["rsi"] = ta.RSI(feat_valid, timeperiod=14)

        macd = ta.MACD(feat_valid)
        feat_valid["macd"] = macd["macd"]
        feat_valid["macdsignal"] = macd["macdsignal"]
        feat_valid["macdhist"] = macd["macdhist"]

        feat_valid["rsi_slope"] = feat_valid["rsi"] - feat_valid["rsi"].shift(1)
        feat_valid["macdhist_delta"] = feat_valid["macdhist"] - feat_valid["macdhist"].shift(1)

        range_window = 20
        roll_low = feat_valid["low"].rolling(range_window).min()
        roll_high = feat_valid["high"].rolling(range_window).max()
        range_span = (roll_high - roll_low).replace(0, np.nan)
        feat_valid["range_pos_20"] = (feat_valid["close"] - roll_low) / range_span

        controller_params = {
            "buy_threshold": float(self.buy_threshold.value),
            "breakout_level": float(self.breakout_level.value),
            "strong_breakout_level": float(self.strong_breakout_level.value),
            "mild_breakout_boost": float(self.mild_breakout_boost.value),
            "strong_breakout_boost": float(self.strong_breakout_boost.value),
            "imbalance_soft": float(self.imbalance_soft.value),
            "imbalance_strong": float(self.imbalance_strong.value),
            "imbalance_negative": float(self.imbalance_negative.value),
            "confirm_soft_boost": float(self.confirm_soft_boost.value),
            "confirm_strong_boost": float(self.confirm_strong_boost.value),
            "negative_confirm_penalty": float(self.negative_confirm_penalty.value),
            "meanrev_dist_z": float(self.meanrev_dist_z.value),
            "confirm_min_imbalance": float(self.confirm_min_imbalance.value),
            "meanrev_position_size": float(self.meanrev_position_size.value),

            # confluence / anti-peak params
            "meanrev_range_max": float(self.meanrev_range_max.value),
            "trend_range_max": float(self.trend_range_max.value),
            "breakout_range_max": float(self.breakout_range_max.value),
            "meanrev_rsi_max": float(self.meanrev_rsi_max.value),
            "trend_rsi_min": float(self.trend_rsi_min.value),
            "trend_rsi_max": float(self.trend_rsi_max.value),
            "trend_macdhist_min": float(self.trend_macdhist_min.value),
            "breakout_macdhist_delta_min": float(self.breakout_macdhist_delta_min.value),

            # keep 0.0 initially unless you explicitly want looser meanrev entries
            "allow_meanrev_edge_relax": float(self.allow_meanrev_edge_relax.value),
        }

        def _apply_controller(row: pd.Series) -> pd.Series:
            d = controller(
                row=row,
                pred=float(row["pred"]),
                threshold=float(row["threshold"]),
                params=controller_params,
            )
            return pd.Series(d)

        ctrl_cols = feat_valid.apply(_apply_controller, axis=1)

        for col in ctrl_cols.columns:
            feat_valid[col] = ctrl_cols[col]

        # hard guard against duplicate columns
        if feat_valid.columns.duplicated().any():
            dupes = feat_valid.columns[feat_valid.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns in feat_valid: {dupes}")
        
        feat_valid = feat_valid.copy()
        feat_valid = feat_valid.set_index("date", drop=False)

        num_map_cols = [
            "pred_proba",
            "pred",
            "threshold",
            "adjusted_pred",
            "breakout_boost",
            "confirm_boost",

            "rsi",
            "macd",
            "macdsignal",
            "macdhist",
            "rsi_slope",
            "macdhist_delta",
            "range_pos_20",
            
            "imbalance_5",
            "dist_ma_15_z",
        ]

        for c in num_map_cols:
            out[c] = out["date"].map(feat_valid[c])

        out["signal"] = out["date"].map(feat_valid["signal"]).fillna(0).astype(int)
        out["position"] = out["date"].map(feat_valid["position"]).fillna(0.0).astype(float)

        for c in [
            "long_signal_raw",
            "long_signal",
            "is_trending",
            "is_breakout",
            "long_confirm",
            "meanrev_exit_warn",
            "trend_exit_warn",
        ]:
            out[c] = out["date"].map(feat_valid[c]).astype("boolean").fillna(False).astype(bool)

        out["reason"] = out["date"].map(feat_valid["reason"]).fillna("")
        out["regime"] = out["date"].map(feat_valid["regime"]).fillna("neutral")

        # print(out[["date", "pred", "threshold", "position", "signal", "reason"]].tail(20))
        return out

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        df["enter_long"] = 0
        df["enter_tag"] = None

        entry_cond = (
            (df["in_test_window"]) &
            (df["volume"] > 0) &
            (df["signal"] == 1) &
            (df["position"] > 0)
        )

        df.loc[entry_cond, "enter_long"] = 1
        df.loc[entry_cond, "enter_tag"] = df.loc[entry_cond, "reason"]

        return df


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        df["exit_long"] = 0
        df["exit_tag"] = None

        has_volume = df["volume"] > 0

        pred_exit = df["adjusted_pred"] < float(self.sell_pred_threshold.value)
        imbalance_exit = df["imbalance_5"] < float(self.sell_imbalance_threshold.value)

        meanrev_reversion_exit = (
            (df["reason"] == "meanrev_long") &
            (
                (df["dist_ma_15_z"] > float(self.sell_dist_z_threshold.value)) |
                (df["meanrev_exit_warn"])
            )
        )

        trend_momentum_exit = (
            df["reason"].isin(["trend_long", "breakout_long"]) &
            df["trend_exit_warn"]
        )

        test_window_exit = ~df["in_test_window"]

        exit_cond = has_volume & (
            pred_exit |
            imbalance_exit |
            meanrev_reversion_exit |
            trend_momentum_exit |
            test_window_exit
        )

        df.loc[pred_exit & has_volume, "exit_tag"] = "pred_exit"
        df.loc[imbalance_exit & has_volume, "exit_tag"] = "imbalance_exit"
        df.loc[trend_momentum_exit & has_volume, "exit_tag"] = "trend_momentum_exit"
        df.loc[meanrev_reversion_exit & has_volume, "exit_tag"] = "meanrev_revert_exit"
        df.loc[test_window_exit & has_volume, "exit_tag"] = "test_window_exit"

        df.loc[exit_cond, "exit_long"] = 1

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