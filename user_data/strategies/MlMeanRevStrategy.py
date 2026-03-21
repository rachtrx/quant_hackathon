# user_data/strategies/MlMeanRevStrategy.py

# flake8: noqa: F401
# isort: skip_file

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
)

from constants import DATA_DIR, INTERVAL, MODEL_DIR, TARGET_HORIZON
from features import add_features


class MlMeanRevStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = False
    timeframe = INTERVAL

    # We handle exits ourselves via custom_exit()
    minimal_roi = {}
    stoploss = -0.99
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 1000

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # --------------------------------------------------
    # Fixed strategy settings
    # --------------------------------------------------
    MODEL_TYPE = "xgb"
    TARGET_HORIZON = TARGET_HORIZON
    MIN_BAND_STD = 1e-8

    # --------------------------------------------------
    # Hyperoptable parameters
    # --------------------------------------------------
    # 30m band settings
    WINDOW_30M = IntParameter(5, 30, default=10, space="buy")
    ENTRY_DEV_Z = DecimalParameter(-3.5, -1.0, decimals=2, default=-2.20, space="buy")
    SL_DEV_DELTA = DecimalParameter(0.2, 2.0, decimals=2, default=0.50, space="sell")

    # Volatility filter
    VOL_FILTER_THRESH = DecimalParameter(0.70, 2.00, decimals=2, default=1.10, space="buy")

    # Stake sizing
    BASE_SIZE_PCT = DecimalParameter(0.002, 0.010, decimals=3, default=0.005, space="buy")
    MID_SIZE_PCT = DecimalParameter(0.005, 0.020, decimals=3, default=0.010, space="buy")

    # Prediction thresholds
    ABS_MIN_PRED = DecimalParameter(0.001, 0.030, decimals=3, default=0.005, space="buy")
    PRED_ENTRY_QUANTILE = DecimalParameter(0.45, 0.65, decimals=2, default=0.51, space="buy")
    PRED_MID_DELTA = DecimalParameter(0.01, 0.15, decimals=2, default=0.04, space="buy")
    PRED_HIGH_DELTA = DecimalParameter(0.01, 0.20, decimals=2, default=0.05, space="buy")
    PRED_ROLLING_WINDOW = IntParameter(1, 10, default=5, space="buy")
    PRED_MIN_PERIODS_FRAC = DecimalParameter(0.2, 1.0, decimals=2, default=0.50, space="buy")

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            "main_plot": {
                "tema": {},
                "sar": {"color": "white"},
            },
            "subplots": {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    "macd": {"color": "blue"},
                    "macdsignal": {"color": "orange"},
                },
                "RSI": {
                    "rsi": {"color": "red"},
                }
            }
        }

    def bot_start(self, **kwargs) -> None:
        self._model_cache: dict[str, Any] = {}
        self._feature_cache: dict[str, list[str]] = {}
        self._meta_cache: dict[str, dict[str, Any]] = {}
        self._raw_cache: dict[str, pd.DataFrame] = {}

        self.model_root = Path(MODEL_DIR) / self.MODEL_TYPE
        self.raw_root = Path(DATA_DIR)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _pair_to_symbol(self, pair: str) -> str:
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
        Load richer raw parquet containing extra fields needed by feature engineering.
        Keep only non-OHLCV extras to avoid clashing with freqtrade OHLCV columns.
        """
        symbol = self._pair_to_symbol(pair)

        if symbol in self._raw_cache:
            return self._raw_cache[symbol]

        raw_path = self.raw_root / f"{symbol}_1m.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing raw parquet for {symbol}: {raw_path}\n"
                f"Put your rich raw files in {self.raw_root.resolve()}"
            )

        raw_df = pd.read_parquet(raw_path).copy()

        if "open_time" not in raw_df.columns:
            raise ValueError(f"{raw_path} must contain 'open_time'.")

        raw_df["date"] = pd.to_datetime(raw_df["open_time"], utc=True)

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

    def _get_last_row_before_time(self, pair: str, current_time: datetime) -> pd.Series | None:
        if not self.dp:
            return None

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None

        temp = df.copy()
        temp["date"] = pd.to_datetime(temp["date"], utc=True)

        ts = pd.Timestamp(current_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        temp = temp[temp["date"] <= ts]
        if temp.empty:
            return None

        return temp.iloc[-1]

    # --------------------------------------------------
    # Indicators
    # --------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        model, feature_cols, meta = self._load_pair_artifacts(pair)

        test_start_time = meta["test_start_time"]
        test_end_time = meta["test_end_time"]

        # Resolve hyperopt params once
        window_30m = int(self.WINDOW_30M.value)
        entry_dev_z = float(self.ENTRY_DEV_Z.value)
        sl_dev_z = entry_dev_z - float(self.SL_DEV_DELTA.value)
        vol_filter_thresh = float(self.VOL_FILTER_THRESH.value)

        base_size_pct = float(self.BASE_SIZE_PCT.value)
        mid_size_pct = float(self.MID_SIZE_PCT.value)

        abs_min_pred = float(self.ABS_MIN_PRED.value)
        pred_entry_q = float(self.PRED_ENTRY_QUANTILE.value)
        pred_mid_q = pred_entry_q + float(self.PRED_MID_DELTA.value)
        pred_high_q = pred_mid_q + float(self.PRED_HIGH_DELTA.value)

        # Safety clip (important)
        pred_mid_q = min(pred_mid_q, 0.95)
        pred_high_q = min(pred_high_q, 0.99)

        pred_rolling_window = int(self.PRED_ROLLING_WINDOW.value)
        pred_min_periods = max(1, int(pred_rolling_window * float(self.PRED_MIN_PERIODS_FRAC.value)))
        pred_min_periods = min(pred_min_periods, pred_rolling_window)

        df = dataframe.copy()
        out = dataframe.copy()

        if "date" not in df.columns:
            raise ValueError("Freqtrade dataframe is expected to contain a 'date' column.")

        df["date"] = pd.to_datetime(df["date"], utc=True)
        out["date"] = pd.to_datetime(out["date"], utc=True)

        out["in_test_window"] = (
            (out["date"] >= test_start_time) &
            (out["date"] <= test_end_time)
        )

        # --------------------------------------------------
        # Merge richer raw fields for feature generation
        # --------------------------------------------------
        raw_df = self._load_raw_pair_data(pair)
        df = df.merge(raw_df, on="date", how="left")

        # --------------------------------------------------
        # Feature engineering
        # --------------------------------------------------
        feat_df, _ = add_features(df.copy(), target_horizon=self.TARGET_HORIZON)

        numeric_cols = [
            "pred_proba",
            "pred",
            "threshold",
            "pred_mid_threshold",
            "pred_high_threshold",
            "band_mean_30m",
            "band_std_30m",
            "sl_price_30m",
            "dev_z",
            "vol_5bar",
            "vol_30_30m",
            "vol_ratio_5_30",
            "stake_pct",
        ]
        bool_cols = [
            "enter_logic",
            "pass_vol_filter",
            "pass_pred_filter",
        ]
        text_cols = [
            "enter_reason",
        ]

        for c in numeric_cols:
            out[c] = np.nan
        for c in bool_cols:
            out[c] = False
        for c in text_cols:
            out[c] = ""

        if feat_df.empty:
            return out

        missing_feature_cols = [c for c in feature_cols if c not in feat_df.columns]
        if missing_feature_cols:
            raise ValueError(
                f"Missing feature columns required by model: {missing_feature_cols}"
            )

        valid_mask = ~feat_df[feature_cols].isna().any(axis=1)
        feat_valid = feat_df.loc[valid_mask].copy()

        if feat_valid.empty:
            return out

        # --------------------------------------------------
        # Model prediction
        # --------------------------------------------------
        X = feat_valid[feature_cols]
        feat_valid["pred_proba"] = model.predict_proba(X)[:, 1]
        feat_valid["pred"] = feat_valid["pred_proba"] - 0.5

        # --------------------------------------------------
        # 30m rolling band from completed 30m bars only
        # --------------------------------------------------
        df_30 = (
            df.set_index("date")
            .resample("30min")
            .agg(close_30m=("close", "last"))
            .dropna()
            .sort_index()
            .reset_index()
        )

        if df_30.empty:
            return out

        df_30["band_mean_30m"] = df_30["close_30m"].rolling(window_30m).mean()
        df_30["band_std_30m"] = df_30["close_30m"].rolling(window_30m).std()
        df_30["vol_30_30m"] = df_30["close_30m"].pct_change().rolling(30).std()

        df_30["band_std_30m"] = df_30["band_std_30m"].clip(lower=self.MIN_BAND_STD)
        df_30["vol_30_30m"] = df_30["vol_30_30m"].clip(lower=self.MIN_BAND_STD)

        # completed 30m bar becomes available only after it closes
        df_30["date_available"] = df_30["date"] + pd.Timedelta(minutes=30)

        feat_valid = pd.merge_asof(
            feat_valid.sort_values("date"),
            df_30[
                [
                    "date_available",
                    "band_mean_30m",
                    "band_std_30m",
                    "vol_30_30m",
                ]
            ].sort_values("date_available"),
            left_on="date",
            right_on="date_available",
            direction="backward",
        )

        # --------------------------------------------------
        # Short-term vol and band metrics
        # --------------------------------------------------
        feat_valid["vol_5bar"] = feat_valid["close"].pct_change().rolling(5).std()

        feat_valid["dev_z"] = (
            (feat_valid["close"] - feat_valid["band_mean_30m"]) /
            feat_valid["band_std_30m"]
        )

        feat_valid["vol_ratio_5_30"] = (
            feat_valid["vol_5bar"] / feat_valid["vol_30_30m"]
        )

        feat_valid["sl_price_30m"] = (
            feat_valid["band_mean_30m"] + sl_dev_z * feat_valid["band_std_30m"]
        )

        # --------------------------------------------------
        # Vol filter
        # --------------------------------------------------
        feat_valid["pass_vol_filter"] = (
            feat_valid["vol_ratio_5_30"] < vol_filter_thresh
        )

        # --------------------------------------------------
        # Adaptive prediction thresholds
        # --------------------------------------------------
        pred_shifted = feat_valid["pred"].shift(1)

        feat_valid["threshold"] = (
            pred_shifted
            .rolling(pred_rolling_window, min_periods=pred_min_periods)
            .quantile(pred_entry_q)
        )

        feat_valid["pred_mid_threshold"] = (
            pred_shifted
            .rolling(pred_rolling_window, min_periods=pred_min_periods)
            .quantile(pred_mid_q)
        )

        feat_valid["pred_high_threshold"] = (
            pred_shifted
            .rolling(pred_rolling_window, min_periods=pred_min_periods)
            .quantile(pred_high_q)
        )

        for c in ["threshold", "pred_mid_threshold", "pred_high_threshold"]:
            feat_valid[c] = feat_valid[c].fillna(abs_min_pred)
            feat_valid[c] = feat_valid[c].clip(lower=abs_min_pred)

        feat_valid["pred_mid_threshold"] = np.maximum(
            feat_valid["pred_mid_threshold"],
            feat_valid["threshold"],
        )
        feat_valid["pred_high_threshold"] = np.maximum(
            feat_valid["pred_high_threshold"],
            feat_valid["pred_mid_threshold"],
        )

        # --------------------------------------------------
        # Prediction filter
        # High pred = skip
        # --------------------------------------------------
        feat_valid["pass_pred_filter"] = (
            (feat_valid["pred"] >= feat_valid["threshold"]) &
            (feat_valid["pred"] < feat_valid["pred_high_threshold"])
        )

        # --------------------------------------------------
        # Entry logic
        # --------------------------------------------------
        feat_valid["enter_logic"] = (
            (feat_valid["dev_z"] <= entry_dev_z) &
            feat_valid["pass_vol_filter"] &
            feat_valid["pass_pred_filter"]
        )

        # --------------------------------------------------
        # Stake sizing from adaptive pred strength
        # Only small and mid buckets.
        # High pred bucket is skipped entirely.
        # --------------------------------------------------
        feat_valid["stake_pct"] = 0.0

        feat_valid.loc[
            feat_valid["enter_logic"] &
            (feat_valid["pred"] >= feat_valid["threshold"]) &
            (feat_valid["pred"] < feat_valid["pred_mid_threshold"]),
            "stake_pct"
        ] = base_size_pct

        feat_valid.loc[
            feat_valid["enter_logic"] &
            (feat_valid["pred"] >= feat_valid["pred_mid_threshold"]) &
            (feat_valid["pred"] < feat_valid["pred_high_threshold"]),
            "stake_pct"
        ] = mid_size_pct

        feat_valid["enter_reason"] = np.select(
            [
                feat_valid["stake_pct"] == base_size_pct,
                feat_valid["stake_pct"] == mid_size_pct,
                (~feat_valid["pass_pred_filter"]) & (feat_valid["pred"] >= feat_valid["pred_high_threshold"]),
            ],
            [
                "meanrev_long_small",
                "meanrev_long_mid",
                "pred_too_high_skip",
            ],
            default="no_trade",
        )

        # --------------------------------------------------
        # Optional debug
        # --------------------------------------------------
        if len(feat_valid) > 0:
            dev_pass = float((feat_valid["dev_z"] <= entry_dev_z).mean())
            vol_pass = float(feat_valid["pass_vol_filter"].mean())
            pred_pass = float(feat_valid["pass_pred_filter"].mean())
            enter_pass = float(feat_valid["enter_logic"].mean())
            too_high = float((feat_valid["pred"] >= feat_valid["pred_high_threshold"]).mean())

            print(
                f"[{pair}] rows={len(feat_valid)} "
                f"dev_pass={dev_pass:.4f} "
                f"vol_pass={vol_pass:.4f} "
                f"pred_pass={pred_pass:.4f} "
                f"too_high={too_high:.4f} "
                f"enter_pass={enter_pass:.4f}"
            )

        # --------------------------------------------------
        # Map back to freqtrade output df
        # --------------------------------------------------
        keep_cols = [
            "date",
            "pred_proba",
            "pred",
            "threshold",
            "pred_mid_threshold",
            "pred_high_threshold",
            "band_mean_30m",
            "band_std_30m",
            "sl_price_30m",
            "dev_z",
            "vol_5bar",
            "vol_30_30m",
            "vol_ratio_5_30",
            "stake_pct",
            "enter_logic",
            "pass_vol_filter",
            "pass_pred_filter",
            "enter_reason",
        ]

        feat_trim = (
            feat_valid[keep_cols]
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")
        )

        for c in [
            "pred_proba",
            "pred",
            "threshold",
            "pred_mid_threshold",
            "pred_high_threshold",
            "band_mean_30m",
            "band_std_30m",
            "sl_price_30m",
            "dev_z",
            "vol_5bar",
            "vol_30_30m",
            "vol_ratio_5_30",
            "stake_pct",
            "enter_reason",
        ]:
            out[c] = out["date"].map(feat_trim[c])

        for c in ["enter_logic", "pass_vol_filter", "pass_pred_filter"]:
            out[c] = (
                out["date"]
                .map(feat_trim[c])
                .astype("boolean")
                .fillna(False)
                .astype(bool)
            )

        out["enter_reason"] = out["enter_reason"].fillna("")

        return out

    # --------------------------------------------------
    # Entry / Exit
    # --------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        df["enter_long"] = 0
        df["enter_tag"] = None

        df.loc[
            (
                df["in_test_window"] &
                df["enter_logic"] &
                (df["stake_pct"] > 0) &
                (df["volume"] > 0)
            ),
            ["enter_long", "enter_tag"]
        ] = (1, "ml_meanrev_long")

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        df["exit_long"] = 0
        df["exit_tag"] = None

        df.loc[
            (
                (~df["in_test_window"]) &
                (df["volume"] > 0)
            ),
            ["exit_long", "exit_tag"]
        ] = (1, "outside_test_window")

        return df

    # --------------------------------------------------
    # TP / SL from 30m bands
    # --------------------------------------------------
    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        row = self._get_last_row_before_time(pair, current_time)
        if row is None:
            return None

        in_test_window = bool(row.get("in_test_window", True))
        if not in_test_window:
            return "outside_test_window"

        band_mean = row.get("band_mean_30m", np.nan)
        sl_price = row.get("sl_price_30m", np.nan)

        # Take profit: revert back to mean
        if pd.notna(band_mean) and current_rate >= float(band_mean):
            return "tp_mean_revert"

        # Stop loss: deeper move to configured band level
        if pd.notna(sl_price) and current_rate <= float(sl_price):
            return "sl_band"

        return None