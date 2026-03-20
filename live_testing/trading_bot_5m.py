#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import sys
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Any

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from roostoo import python_demo
from features import add_features
from data_retrieval import fetch_recent_klines_rest
from live_testing import tele_update, logger
from constants import TARGET_HORIZON, INTERVAL


# =========================================================
# GLOBAL CONFIG
# =========================================================

MODEL_TYPE = "xgb"
MODEL_DIR = f"models/{MODEL_TYPE}"
DATA_DIR = "live_testing/data"
STATE_DIR = "live_testing/state"

MIN_BAND_STD = 1e-8

# These mirror the STRATEGY DEFAULTS
WINDOW_30M = 20
ENTRY_DEV_Z = -2.27
SL_DEV_DELTA = 1.81

VOL_FILTER_THRESH = 0.96

BASE_SIZE_PCT = 0.002
MID_SIZE_PCT = 0.019

ABS_MIN_PRED = 0.003
PRED_ENTRY_QUANTILE = 0.45
PRED_MID_DELTA = 0.09
PRED_HIGH_DELTA = 0.04
PRED_ROLLING_WINDOW = 656
PRED_MIN_PERIODS_FRAC = 0.63

MAX_BARS_5M = 2500
MIN_BARS_5M = 1200

BUY_FEE_RATE = 0.000 # TO CHECK

MAX_OPEN_POSITIONS = 5

PAIR_CONFIGS = [
    {"pair": "AVAX/USD", "symbol": "AVAXUSDT", "coin": "AVAX"},
    {"pair": "DOT/USD", "symbol": "DOTUSDT", "coin": "DOT"},
    {"pair": "XRP/USD", "symbol": "XRPUSDT", "coin": "XRP"},
    {"pair": "LINK/USD", "symbol": "LINKUSDT", "coin": "LINK"},
    {"pair": "LTC/USD", "symbol": "LTCUSDT", "coin": "LTC"},
    {"pair": "ADA/USD", "symbol": "ADAUSDT", "coin": "ADA"},
    {"pair": "SOL/USD", "symbol": "SOLUSDT", "coin": "SOL"},
]

order_lock = threading.Lock()


# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class PairConfig:
    pair: str
    symbol: str
    coin: str


@dataclass
class PositionState:
    qty: float
    entry_price: float
    entry_time: str
    band_mean_30m: float
    band_std_30m: float
    tp_price: float
    sl_price: float
    stake_pct: float
    pred: float
    pred_proba: float
    entry_reason: str


# =========================================================
# TRADER
# =========================================================
class CoinTrader:
    def __init__(self, cfg: PairConfig):
        self.cfg = cfg
        self.logger = logger.setup_logger(name=f"bot_{cfg.symbol}")

        self.meta_path = os.path.join(MODEL_DIR, f"{cfg.symbol}__h{TARGET_HORIZON}_meta.json")
        self.meta = self.load_meta()

        self.model_path = self.meta["model_path"]
        self.features_path = self.meta["feature_cols_path"]
        self.target_horizon = TARGET_HORIZON

        self.parquet_path = os.path.join(DATA_DIR, f"{cfg.symbol}_{INTERVAL}_live.parquet")
        self.position_state_path = os.path.join(
            STATE_DIR, f"{cfg.symbol.lower()}_position_state.json"
        )

        self.ensure_dirs()
        self.model, self.feature_cols = self.load_model_and_features()
        self.df = self.load_initial_data()
        self.position = self.load_position_state()

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------
    def log(self, msg: str, level: str = "info", send_tele: bool = False):
        line = f"[{self.cfg.symbol}] {msg}"

        if level == "info":
            self.logger.info(line)
        elif level == "warning":
            self.logger.warning(line)
        elif level == "error":
            self.logger.error(line)
        else:
            self.logger.info(line)

        if send_tele:
            try:
                tele_update.send(line)
            except Exception as e:
                self.logger.error(f"[{self.cfg.symbol}] TELEGRAM FAILED: {e}")

    # -----------------------------------------------------
    # Setup / Persistence
    # -----------------------------------------------------
    def ensure_dirs(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(STATE_DIR, exist_ok=True)

    def load_meta(self) -> dict[str, Any]:
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Meta file not found: {self.meta_path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return meta

    def load_model_and_features(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Feature cols file not found: {self.features_path}")

        model = joblib.load(self.model_path)
        with open(self.features_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        return model, feature_cols

    def load_initial_data(self) -> pd.DataFrame:
        if os.path.exists(self.parquet_path):
            df = pd.read_parquet(self.parquet_path)
            if not df.empty:
                df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
                if "close_time" in df.columns:
                    df["close_time"] = pd.to_datetime(
                        df["close_time"], utc=True, errors="coerce"
                    )
                df = (
                    df.sort_values("open_time")
                    .drop_duplicates(subset=["open_time"])
                    .tail(MAX_BARS_5M)
                    .reset_index(drop=True)
                )
                return df
        return pd.DataFrame()

    def save_data(self):
        if not self.df.empty:
            self.df.to_parquet(self.parquet_path, index=False)

    def save_position_state(self):
        if self.position is None:
            if os.path.exists(self.position_state_path):
                os.remove(self.position_state_path)
            return

        with open(self.position_state_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.position), f, indent=2)

    def load_position_state(self) -> Optional[PositionState]:
        if not os.path.exists(self.position_state_path):
            return None
        with open(self.position_state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PositionState(**data)

    # -----------------------------------------------------
    # Position limits
    # -----------------------------------------------------
    def count_open_positions(self) -> int:
        count = 0
        for cfg in PAIR_CONFIGS:
            path = os.path.join(STATE_DIR, f"{cfg['symbol'].lower()}_position_state.json")
            if os.path.exists(path):
                count += 1
        return count

    def can_open_new_position(self) -> bool:
        if self.position is not None:
            return False
        return self.count_open_positions() < MAX_OPEN_POSITIONS

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------
    def refresh_data(self):
        if self.df.empty:
            recent = fetch_recent_klines_rest(
                self.cfg.symbol,
                INTERVAL,
                limit=MAX_BARS_5M,
            )
            if not recent.empty:
                self.df = recent.copy()
        else:
            last_open_time = pd.to_datetime(self.df["open_time"].max(), utc=True)
            start_time_ms = int(last_open_time.timestamp() * 1000) + 1
            recent = fetch_recent_klines_rest(
                self.cfg.symbol,
                INTERVAL,
                start_time_ms=start_time_ms,
                limit=1000,
            )
            if not recent.empty:
                self.df = pd.concat([self.df, recent], ignore_index=True)

        if not self.df.empty:
            self.df["open_time"] = pd.to_datetime(self.df["open_time"], utc=True)
            if "close_time" in self.df.columns:
                self.df["close_time"] = pd.to_datetime(
                    self.df["close_time"], utc=True, errors="coerce"
                )

            self.df = (
                self.df.drop_duplicates(subset=["open_time"])
                .sort_values("open_time")
                .tail(MAX_BARS_5M)
                .reset_index(drop=True)
            )

    # -----------------------------------------------------
    # Build signal frame to mirror Freqtrade strategy
    # -----------------------------------------------------
    def _normalize_utc_ns(self, s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")
    
    def build_signal_frame(self) -> pd.DataFrame:
        df = self.df.copy()

        if df.empty:
            return pd.DataFrame()

        df["date"] = self._normalize_utc_ns(df["open_time"])

        feat_df, _ = add_features(df.copy(), target_horizon=self.target_horizon)

        if feat_df.empty:
            return pd.DataFrame()

        missing_feature_cols = [c for c in self.feature_cols if c not in feat_df.columns]
        if missing_feature_cols:
            raise ValueError(
                f"Missing feature columns required by model: {missing_feature_cols}"
            )

        valid_mask = ~feat_df[self.feature_cols].isna().any(axis=1)
        feat_valid = feat_df.loc[valid_mask].copy()

        if feat_valid.empty:
            return pd.DataFrame()

        feat_valid["date"] = self._normalize_utc_ns(feat_valid["date"])

        X = feat_valid[self.feature_cols]
        feat_valid["pred_proba"] = self.model.predict_proba(X)[:, 1]
        feat_valid["pred"] = feat_valid["pred_proba"] - 0.5

        df_30 = (
            df.set_index("date")
            .resample("30min")
            .agg(close_30m=("close", "last"))
            .dropna()
            .sort_index()
            .reset_index()
        )

        if df_30.empty:
            return pd.DataFrame()

        df_30["date"] = self._normalize_utc_ns(df_30["date"])

        df_30["band_mean_30m"] = df_30["close_30m"].rolling(WINDOW_30M).mean()
        df_30["band_std_30m"] = df_30["close_30m"].rolling(WINDOW_30M).std()
        df_30["vol_30_30m"] = df_30["close_30m"].pct_change().rolling(30).std()

        df_30["band_std_30m"] = df_30["band_std_30m"].clip(lower=MIN_BAND_STD)
        df_30["vol_30_30m"] = df_30["vol_30_30m"].clip(lower=MIN_BAND_STD)

        df_30["date_available"] = self._normalize_utc_ns(
            df_30["date"] + pd.Timedelta(minutes=30)
        )

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

        feat_valid["vol_5bar"] = feat_valid["close"].pct_change().rolling(5).std()

        feat_valid["dev_z"] = (
            (feat_valid["close"] - feat_valid["band_mean_30m"]) /
            feat_valid["band_std_30m"]
        )

        feat_valid["vol_ratio_5_30"] = (
            feat_valid["vol_5bar"] / feat_valid["vol_30_30m"]
        )

        sl_dev_z = ENTRY_DEV_Z - SL_DEV_DELTA

        feat_valid["sl_price_30m"] = (
            feat_valid["band_mean_30m"] + sl_dev_z * feat_valid["band_std_30m"]
        )

        feat_valid["pass_vol_filter"] = (
            feat_valid["vol_ratio_5_30"] < VOL_FILTER_THRESH
        )

        pred_shifted = feat_valid["pred"].shift(1)

        pred_mid_q = min(PRED_ENTRY_QUANTILE + PRED_MID_DELTA, 0.95)
        pred_high_q = min(pred_mid_q + PRED_HIGH_DELTA, 0.99)

        pred_min_periods = max(
            1,
            int(PRED_ROLLING_WINDOW * PRED_MIN_PERIODS_FRAC)
        )
        pred_min_periods = min(pred_min_periods, PRED_ROLLING_WINDOW)

        feat_valid["threshold"] = (
            pred_shifted
            .rolling(PRED_ROLLING_WINDOW, min_periods=pred_min_periods)
            .quantile(PRED_ENTRY_QUANTILE)
        )

        feat_valid["pred_mid_threshold"] = (
            pred_shifted
            .rolling(PRED_ROLLING_WINDOW, min_periods=pred_min_periods)
            .quantile(pred_mid_q)
        )

        feat_valid["pred_high_threshold"] = (
            pred_shifted
            .rolling(PRED_ROLLING_WINDOW, min_periods=pred_min_periods)
            .quantile(pred_high_q)
        )

        for c in ["threshold", "pred_mid_threshold", "pred_high_threshold"]:
            feat_valid[c] = feat_valid[c].fillna(ABS_MIN_PRED)
            feat_valid[c] = feat_valid[c].clip(lower=ABS_MIN_PRED)

        feat_valid["pred_mid_threshold"] = np.maximum(
            feat_valid["pred_mid_threshold"],
            feat_valid["threshold"],
        )
        feat_valid["pred_high_threshold"] = np.maximum(
            feat_valid["pred_high_threshold"],
            feat_valid["pred_mid_threshold"],
        )

        feat_valid["pass_pred_filter"] = (
            (feat_valid["pred"] >= feat_valid["threshold"]) &
            (feat_valid["pred"] < feat_valid["pred_high_threshold"])
        )

        feat_valid["enter_logic"] = (
            (feat_valid["dev_z"] <= ENTRY_DEV_Z) &
            feat_valid["pass_vol_filter"] &
            feat_valid["pass_pred_filter"]
        )

        feat_valid["stake_pct"] = 0.0

        feat_valid.loc[
            feat_valid["enter_logic"] &
            (feat_valid["pred"] >= feat_valid["threshold"]) &
            (feat_valid["pred"] < feat_valid["pred_mid_threshold"]),
            "stake_pct"
        ] = BASE_SIZE_PCT

        feat_valid.loc[
            feat_valid["enter_logic"] &
            (feat_valid["pred"] >= feat_valid["pred_mid_threshold"]) &
            (feat_valid["pred"] < feat_valid["pred_high_threshold"]),
            "stake_pct"
        ] = MID_SIZE_PCT

        feat_valid["enter_reason"] = np.select(
            [
                feat_valid["stake_pct"] == BASE_SIZE_PCT,
                feat_valid["stake_pct"] == MID_SIZE_PCT,
                (~feat_valid["pass_pred_filter"]) &
                (feat_valid["pred"] >= feat_valid["pred_high_threshold"]),
            ],
            [
                "meanrev_long_small",
                "meanrev_long_mid",
                "pred_too_high_skip",
            ],
            default="no_trade",
        )

        return feat_valid.sort_values("date").reset_index(drop=True)

    def get_latest_signal_values(self, sig_df: pd.DataFrame) -> Optional[dict[str, Any]]:
        if sig_df.empty:
            return None

        latest = sig_df.iloc[-1].copy()

        required = [
            "close",
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

        for c in required:
            if c not in latest.index:
                return None
            if c not in ["enter_logic", "pass_vol_filter", "pass_pred_filter", "enter_reason"]:
                if pd.isna(latest[c]):
                    return None

        return {
            "date": latest["date"],
            "current_price": float(latest["close"]),
            "pred_proba": float(latest["pred_proba"]),
            "pred": float(latest["pred"]),
            "threshold": float(latest["threshold"]),
            "pred_mid_threshold": float(latest["pred_mid_threshold"]),
            "pred_high_threshold": float(latest["pred_high_threshold"]),
            "band_mean_30m": float(latest["band_mean_30m"]),
            "band_std_30m": float(latest["band_std_30m"]),
            "sl_price_30m": float(latest["sl_price_30m"]),
            "dev_z": float(latest["dev_z"]),
            "vol_5bar": float(latest["vol_5bar"]),
            "vol_30_30m": float(latest["vol_30_30m"]),
            "vol_ratio_5_30": float(latest["vol_ratio_5_30"]),
            "stake_pct": float(latest["stake_pct"]),
            "enter_logic": bool(latest["enter_logic"]),
            "pass_vol_filter": bool(latest["pass_vol_filter"]),
            "pass_pred_filter": bool(latest["pass_pred_filter"]),
            "enter_reason": str(latest["enter_reason"]),
        }

    # -----------------------------------------------------
    # Balances / Prices
    # -----------------------------------------------------
    def get_usd_balance(self) -> float:
        balance = python_demo.get_balance()
        try:
            return float(balance["Wallet"]["USD"]["Free"])
        except Exception:
            return 0.0

    def get_coin_balance(self) -> float:
        balance = python_demo.get_balance()
        try:
            return float(balance["Wallet"][self.cfg.coin]["Free"])
        except Exception:
            return 0.0

    def get_live_price(self) -> float:
        ticker = python_demo.get_ticker(self.cfg.pair)
        if not ticker.get("Success"):
            return 0.0

        try:
            return float(ticker["Data"][self.cfg.pair]["LastPrice"])
        except Exception:
            return 0.0

    # -----------------------------------------------------
    # Orders
    # -----------------------------------------------------
    def place_buy(
        self,
        stake_pct: float,
        current_price: float,
        band_mean_30m: float,
        band_std_30m: float,
        sl_price_30m: float,
        pred: float,
        pred_proba: float,
        entry_reason: str,
    ) -> Optional[PositionState]:
        usd_balance = self.get_usd_balance()
        if usd_balance <= 0:
            self.log("No USD balance available.", level="warning")
            return None

        qty = (usd_balance * stake_pct * (1 - BUY_FEE_RATE)) / current_price

        buy_resp = python_demo.place_order(self.cfg.pair, "BUY", qty)

        if not isinstance(buy_resp, dict) or not buy_resp.get("Success"):
            self.log(f"BUY failed: {buy_resp}", level="error", send_tele=True)
            return None

        tp_price = band_mean_30m

        self.log(
            f"ENTRY BUY qty={qty:.8f} {self.cfg.coin} "
            f"entry={current_price:.6f} "
            f"tp_mean={tp_price:.6f} "
            f"sl_band={sl_price_30m:.6f} "
            f"stake_pct={stake_pct * 100:.2f}% "
            f"pred={pred:.6f} "
            f"pred_proba={pred_proba:.6f} "
            f"reason={entry_reason}",
            send_tele=True,
        )

        return PositionState(
            qty=float(qty),
            entry_price=float(current_price),
            entry_time=pd.Timestamp.utcnow().isoformat(),
            band_mean_30m=float(band_mean_30m),
            band_std_30m=float(band_std_30m),
            tp_price=float(tp_price),
            sl_price=float(sl_price_30m),
            stake_pct=float(stake_pct),
            pred=float(pred),
            pred_proba=float(pred_proba),
            entry_reason=str(entry_reason),
        )

    def market_sell_position(self) -> bool:
        if self.position is None:
            return False

        sell_resp = python_demo.place_order(
            self.cfg.pair,
            "SELL",
            self.position.qty,
        )

        ok = isinstance(sell_resp, dict) and sell_resp.get("Success")
        if not ok:
            self.log(f"SELL failed: {sell_resp}", level="error", send_tele=True)
        return ok

    def check_position_exit(self) -> Optional[str]:
        if self.position is None:
            return None

        live_price = self.get_live_price()
        if live_price <= 0:
            return None

        # Mirrors custom_exit:
        # TP if current_rate >= band_mean
        if live_price >= self.position.tp_price:
            ok = self.market_sell_position()
            if ok:
                approx_pnl = (
                    (live_price - self.position.entry_price) * self.position.qty
                )
                usd_balance = self.get_usd_balance()

                self.log(
                    f"TP MEAN REVERT qty={self.position.qty:.8f} {self.cfg.coin} "
                    f"entry={self.position.entry_price:.6f} "
                    f"exit≈{live_price:.6f} "
                    f"tp_mean={self.position.tp_price:.6f} "
                    f"pnl≈{approx_pnl:.6f} "
                    f"usd_balance={usd_balance:.2f}",
                    send_tele=True,
                )
                return "tp_mean_revert"

        # Mirrors custom_exit:
        # SL if current_rate <= sl_price
        if live_price <= self.position.sl_price:
            ok = self.market_sell_position()
            if ok:
                approx_pnl = (
                    (live_price - self.position.entry_price) * self.position.qty
                )
                usd_balance = self.get_usd_balance()

                self.log(
                    f"SL BAND qty={self.position.qty:.8f} {self.cfg.coin} "
                    f"entry={self.position.entry_price:.6f} "
                    f"exit≈{live_price:.6f} "
                    f"sl_band={self.position.sl_price:.6f} "
                    f"pnl≈{approx_pnl:.6f} "
                    f"usd_balance={usd_balance:.2f}",
                    send_tele=True,
                )
                return "sl_band"

        return None

    # -----------------------------------------------------
    # Main step
    # -----------------------------------------------------
    def step(self):
        self.refresh_data()

        if self.df.empty or len(self.df) < MIN_BARS_5M:
            self.save_data()
            return

        self.save_data()

        if self.position is not None:
            exit_reason = self.check_position_exit()
            if exit_reason is not None:
                self.position = None
                self.save_position_state()
            return

        sig_df = self.build_signal_frame()
        sig = self.get_latest_signal_values(sig_df)

        if sig is None:
            return

        # Optional debug like strategy
        too_high = sig["pred"] >= sig["pred_high_threshold"]
        self.log(
            "latest_signal "
            f"dev_z={sig['dev_z']:.4f} "
            f"vol_ratio={sig['vol_ratio_5_30']:.4f} "
            f"pred={sig['pred']:.6f} "
            f"thr={sig['threshold']:.6f} "
            f"mid_thr={sig['pred_mid_threshold']:.6f} "
            f"high_thr={sig['pred_high_threshold']:.6f} "
            f"vol_pass={sig['pass_vol_filter']} "
            f"pred_pass={sig['pass_pred_filter']} "
            f"too_high={too_high} "
            f"enter_logic={sig['enter_logic']} "
            f"stake_pct={sig['stake_pct']:.4f} "
            f"reason={sig['enter_reason']}"
        )

        if not sig["enter_logic"]:
            return

        if sig["stake_pct"] <= 0:
            return

        with order_lock:
            if not self.can_open_new_position():
                return

            new_position = self.place_buy(
                stake_pct=sig["stake_pct"],
                current_price=sig["current_price"],
                band_mean_30m=sig["band_mean_30m"],
                band_std_30m=sig["band_std_30m"],
                sl_price_30m=sig["sl_price_30m"],
                pred=sig["pred"],
                pred_proba=sig["pred_proba"],
                entry_reason=sig["enter_reason"],
            )

            if new_position is not None:
                self.position = new_position
                self.save_position_state()

    def run_forever(self):
        self.log(
            f"strategy started | loaded_rows={len(self.df)} | "
            f"open_position={self.position is not None}"
        )

        while True:
            try:
                self.step()
            except Exception as e:
                self.log(f"ERROR: {e}", level="error", send_tele=True)
            sleep_to_next_candle(5)

# =========================================================
# RUNNERS
# =========================================================
def run_trader(cfg_dict):
    trader = CoinTrader(PairConfig(**cfg_dict))
    trader.run_forever()


def sleep_to_next_candle(interval_minutes: int = 5):
    now = time.time()
    next_boundary = (int(now // (interval_minutes * 60)) + 1) * (interval_minutes * 60)
    time.sleep(max(0, next_boundary - now))


def main():
    threads = []
    for cfg in PAIR_CONFIGS:
        t = threading.Thread(target=run_trader, args=(cfg,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(2)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()