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

from freqtrade.strategy import IStrategy

from constants import DATA_DIR, MODEL_DIR, INTERVAL
from features import add_features


class MlMeanRevOldStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = False

    # We handle exits ourselves via custom_exit().
    minimal_roi = {}
    stoploss = -0.99
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 1000

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ----------------------------
    # Strategy parameters
    # ----------------------------
    MODEL_TYPE = "xgb"
    print("using xgb")
    TARGET_HORIZON = 5

    WINDOW_30M = 10
    VOL_FILTER_THRESH = 1.1

    BASE_SIZE_PCT = 0.005   # 0.5%
    MID_SIZE_PCT = 0.01     # 1.0%
    HIGH_SIZE_PCT = 0.02    # 2.0%

    ENTRY_DEV_Z = -2.0
    SL_DEV = -2.5           # sl_price = mean + SL_DEV * std = mean - 2.5 * std

    def bot_start(self, **kwargs) -> None:
        self._model_cache: dict[str, Any] = {}
        self._feature_cache: dict[str, list[str]] = {}
        self._meta_cache: dict[str, dict[str, Any]] = {}
        self._raw_cache: dict[str, pd.DataFrame] = {}

        self.model_root = Path(MODEL_DIR) / self.MODEL_TYPE
        self.raw_root = Path(DATA_DIR)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
        Load richer raw parquet containing Binance-specific fields that your
        feature engineering may require. We only keep non-OHLCV extras to avoid
        clashing with freqtrade's own OHLCV columns.
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

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        model, feature_cols, meta = self._load_pair_artifacts(pair)

        test_start_time = meta["test_start_time"]
        test_end_time = meta["test_end_time"]

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

        # Merge richer raw fields for feature generation
        raw_df = self._load_raw_pair_data(pair)
        df = df.merge(raw_df, on="date", how="left")

        # ----------------------------
        # Feature engineering + model prediction
        # ----------------------------
        feat_df, _ = add_features(df.copy(), target_horizon=self.TARGET_HORIZON)

        # Default output columns
        numeric_cols = [
            "pred",
            "band_mean_30m",
            "band_std_30m",
            "sl_price_30m",
            "dev_z",
            "pred_price",
            "pred_dev_z",
            "vol_5_5m",
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

        X = feat_valid[feature_cols]
        feat_valid["pred"] = model.predict(X)

        # ----------------------------
        # 30m rolling mean/std using COMPLETED 30m bars only
        # to avoid lookahead leakage in backtests
        # ----------------------------
        df_30 = (
            df.set_index("date")
            .resample("30min")
            .agg(close_30m=("close", "last"))
            .dropna()
            .sort_index()
            .reset_index()
        )

        if not df_30.empty:
            df_30["band_mean_30m"] = (
                df_30["close_30m"].rolling(self.WINDOW_30M).mean()
            )
            df_30["band_std_30m"] = (
                df_30["close_30m"].rolling(self.WINDOW_30M).std()
            )
            df_30["vol_30_30m"] = (
                df_30["close_30m"].pct_change().rolling(30).std()
            )

            # Use only completed 30m bars:
            # shift the bar timestamp forward by 30m so a 10:00-10:29 bar becomes
            # available from 10:30 onward.
            df_30["date_available"] = df_30["date"] + pd.Timedelta(minutes=30)

            feat_valid = pd.merge_asof(
                feat_valid.sort_values("date"),
                df_30[[
                    "date_available",
                    "band_mean_30m",
                    "band_std_30m",
                    "vol_30_30m",
                ]].sort_values("date_available"),
                left_on="date",
                right_on="date_available",
                direction="backward",
            )

        # 5m short vol
        feat_valid["vol_5_5m"] = feat_valid["close"].pct_change().rolling(5).std()

        # Main band logic
        feat_valid["dev_z"] = (
            (feat_valid["close"] - feat_valid["band_mean_30m"]) /
            feat_valid["band_std_30m"]
        )

        feat_valid["vol_ratio_5_30"] = (
            feat_valid["vol_5_5m"] / feat_valid["vol_30_30m"]
        )

        feat_valid["pred_price"] = feat_valid["close"] * (1.0 + feat_valid["pred"])
        feat_valid["pred_dev_z"] = (
            (feat_valid["pred_price"] - feat_valid["band_mean_30m"]) /
            feat_valid["band_std_30m"]
        )

        feat_valid["sl_price_30m"] = (
            feat_valid["band_mean_30m"] + self.SL_DEV * feat_valid["band_std_30m"]
        )

        feat_valid["pass_vol_filter"] = feat_valid["vol_ratio_5_30"] < self.VOL_FILTER_THRESH
        feat_valid["pass_pred_filter"] = feat_valid["pred"] > 0.0

        feat_valid["enter_logic"] = (
            (feat_valid["dev_z"] <= self.ENTRY_DEV_Z) &
            feat_valid["pass_vol_filter"] &
            feat_valid["pass_pred_filter"]
        )

        # Stake sizing from predicted reversion strength
        feat_valid["stake_pct"] = 0.0

        feat_valid.loc[
            feat_valid["enter_logic"] &
            (feat_valid["pred_dev_z"] >= -2.0) &
            (feat_valid["pred_dev_z"] < -1.0),
            "stake_pct"
        ] = self.BASE_SIZE_PCT

        feat_valid.loc[
            feat_valid["enter_logic"] &
            (feat_valid["pred_dev_z"] >= -1.0) &
            (feat_valid["pred_dev_z"] < 0.0),
            "stake_pct"
        ] = self.MID_SIZE_PCT

        feat_valid.loc[
            feat_valid["enter_logic"] &
            (feat_valid["pred_dev_z"] >= 0.0),
            "stake_pct"
        ] = self.HIGH_SIZE_PCT

        feat_valid["enter_reason"] = np.where(
            feat_valid["enter_logic"] & (feat_valid["stake_pct"] > 0),
            "meanrev_long",
            "no_trade",
        )

        # keep only rows with everything we need
        keep_cols = [
            "date",
            "pred",
            "band_mean_30m",
            "band_std_30m",
            "sl_price_30m",
            "dev_z",
            "pred_price",
            "pred_dev_z",
            "vol_5_5m",
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
            "pred",
            "band_mean_30m",
            "band_std_30m",
            "sl_price_30m",
            "dev_z",
            "pred_price",
            "pred_dev_z",
            "vol_5_5m",
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

    # ------------------------------------------------------------------
    # Entry / Exit
    # ------------------------------------------------------------------
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

        # Exits are mainly handled in custom_exit().
        # Keep these columns here because freqtrade expects them.
        df["exit_long"] = 0
        df["exit_tag"] = None

        # Safety exit if we leave the model's intended test window.
        df.loc[
            (
                (~df["in_test_window"]) &
                (df["volume"] > 0)
            ),
            ["exit_long", "exit_tag"]
        ] = (1, "outside_test_window")

        return df

    # ------------------------------------------------------------------
    # Dynamic stake sizing
    # ------------------------------------------------------------------
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        row = self._get_last_row_before_time(pair, current_time)
        if row is None:
            return proposed_stake

        stake_pct = float(row.get("stake_pct", 0.0) or 0.0)

        if stake_pct <= 0:
            return 0.0

        # Best used with stake_amount = "unlimited" so max_stake ~= available wallet.
        # Then returning max_stake * stake_pct approximates "X% of wallet".
        stake = max_stake * stake_pct

        if min_stake is not None:
            stake = max(stake, min_stake)

        stake = min(stake, max_stake)

        return float(stake)

    # ------------------------------------------------------------------
    # TP / SL from rolling 30m bands
    # ------------------------------------------------------------------
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

        if pd.notna(band_mean) and current_rate >= float(band_mean):
            return "tp_mean_revert"

        if pd.notna(sl_price) and current_rate <= float(sl_price):
            return "sl_band"

        return None