#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from decimal import Decimal
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
import talib.abstract as ta

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from roostoo import python_demo
from features import add_features
from data_retrieval import fetch_recent_klines_rest
from live_testing import tele_update, logger
from constants import TARGET_HORIZON, INTERVAL
from controller import controller


# =========================================================
# GLOBAL CONFIG
# =========================================================

MODEL_TYPE = "xgb"
MODEL_DIR = f"models/{MODEL_TYPE}"
DATA_DIR = "live_testing/data"
STATE_DIR = "live_testing/state"

MAX_BARS = 2500
MIN_BARS = 800

PAIR_CONFIGS = [
    {"pair": "AVAX/USD", "symbol": "AVAXUSDT", "coin": "AVAX"},
    {"pair": "LINK/USD", "symbol": "LINKUSDT", "coin": "LINK"},
    {"pair": "SOL/USD", "symbol": "SOLUSDT", "coin": "SOL"},
]

BUY_FEE_RATE = 0.000  # TO CHECK
MAX_OPEN_POSITIONS = len(PAIR_CONFIGS)

####################
# HYPEROPT PARAMS
####################

BUY_THRESHOLD = 0.011
PRED_HIST_WINDOW = 26
PRED_QUANTILE = 0.61

BREAKOUT_LEVEL = 1.07
STRONG_BREAKOUT_LEVEL = 1.18
MILD_BREAKOUT_BOOST = 1.18
STRONG_BREAKOUT_BOOST = 1.04

IMBALANCE_SOFT = 0.03
IMBALANCE_STRONG = 0.02
IMBALANCE_NEGATIVE = -0.2
CONFIRM_SOFT_BOOST = 1.2
CONFIRM_STRONG_BOOST = 1.04
NEGATIVE_CONFIRM_PENALTY = 0.86

MEANREV_DIST_Z = -2.05
CONFIRM_MIN_IMBALANCE = 0.01
MEANREV_POSITION_SIZE = 0.92

SELL_PRED_THRESHOLD = 0.00
SELL_IMBALANCE_THRESHOLD = 0.00
SELL_DIST_Z_THRESHOLD = 0.00

TREND_RSI_MIN = 55.2
TREND_RSI_MAX = 59.2
MEANREV_RSI_MAX = 45.0

MEANREV_RANGE_MAX = 0.26
TREND_RANGE_MAX = 0.86
BREAKOUT_RANGE_MAX = 0.79

TREND_MACDHIST_MIN = 0.028
BREAKOUT_MACDHIST_DELTA_MIN = -0.008

ALLOW_MEANREV_EDGE_RELAX = 0.0

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
                    .tail(MAX_BARS)
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
                limit=MAX_BARS,
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
                .tail(MAX_BARS)
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

        # ----------------------------
        # Model predictions
        # ----------------------------
        X = feat_valid[self.feature_cols]
        feat_valid["pred_proba"] = self.model.predict_proba(X)[:, 1]
        feat_valid["pred"] = feat_valid["pred_proba"] - 0.5

        # ----------------------------
        # Rolling threshold
        # Mirrors strategy:
        # feat_valid["pred"].abs().shift(1).rolling(window).quantile(q)
        # ----------------------------
        feat_valid["threshold"] = (
            feat_valid["pred"]
            .abs()
            .shift(1)
            .rolling(PRED_HIST_WINDOW)
            .quantile(PRED_QUANTILE)
        )
        feat_valid["threshold"] = feat_valid["threshold"].fillna(BUY_THRESHOLD)

        # ----------------------------
        # Extra indicator fields used by controller
        # ----------------------------
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

        # ----------------------------
        # Controller params
        # ----------------------------
        controller_params = {
            "buy_threshold": BUY_THRESHOLD,
            "breakout_level": BREAKOUT_LEVEL,
            "strong_breakout_level": STRONG_BREAKOUT_LEVEL,
            "mild_breakout_boost": MILD_BREAKOUT_BOOST,
            "strong_breakout_boost": STRONG_BREAKOUT_BOOST,
            "imbalance_soft": IMBALANCE_SOFT,
            "imbalance_strong": IMBALANCE_STRONG,
            "imbalance_negative": IMBALANCE_NEGATIVE,
            "confirm_soft_boost": CONFIRM_SOFT_BOOST,
            "confirm_strong_boost": CONFIRM_STRONG_BOOST,
            "negative_confirm_penalty": NEGATIVE_CONFIRM_PENALTY,
            "meanrev_dist_z": MEANREV_DIST_Z,
            "confirm_min_imbalance": CONFIRM_MIN_IMBALANCE,
            "meanrev_position_size": MEANREV_POSITION_SIZE,
            "meanrev_range_max": MEANREV_RANGE_MAX,
            "trend_range_max": TREND_RANGE_MAX,
            "breakout_range_max": BREAKOUT_RANGE_MAX,
            "meanrev_rsi_max": MEANREV_RSI_MAX,
            "trend_rsi_min": TREND_RSI_MIN,
            "trend_rsi_max": TREND_RSI_MAX,
            "trend_macdhist_min": TREND_MACDHIST_MIN,
            "breakout_macdhist_delta_min": BREAKOUT_MACDHIST_DELTA_MIN,
            "allow_meanrev_edge_relax": ALLOW_MEANREV_EDGE_RELAX,
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

        if feat_valid.columns.duplicated().any():
            dupes = feat_valid.columns[feat_valid.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns in feat_valid: {dupes}")

        return feat_valid.sort_values("date").reset_index(drop=True)

    def get_latest_signal_values(self, sig_df: pd.DataFrame) -> Optional[dict[str, Any]]:
        if sig_df.empty:
            return None

        latest = sig_df.iloc[-1].copy()

        required = [
            "date",
            "close",
            "pred_proba",
            "pred",
            "threshold",
            "adjusted_pred",
            "breakout_boost",
            "confirm_boost",
            "signal",
            "position",
            "reason",
            "regime",
            "long_signal_raw",
            "long_signal",
            "is_trending",
            "is_breakout",
            "long_confirm",
            "meanrev_exit_warn",
            "trend_exit_warn",
            "imbalance_5",
            "dist_ma_15_z",
            "meanrev_confluence",
            "trend_confluence",
            "breakout_confluence",
        ]

        for c in required:
            if c not in latest.index:
                return None

        numeric_required = [
            "close",
            "pred_proba",
            "pred",
            "threshold",
            "adjusted_pred",
            "breakout_boost",
            "confirm_boost",
            "position",
            "imbalance_5",
            "dist_ma_15_z",
        ]
        for c in numeric_required:
            if pd.isna(latest[c]):
                return None

        return {
            "date": latest["date"],
            "current_price": float(latest["close"]),
            "pred_proba": float(latest["pred_proba"]),
            "pred": float(latest["pred"]),
            "threshold": float(latest["threshold"]),
            "adjusted_pred": float(latest["adjusted_pred"]),
            "breakout_boost": float(latest["breakout_boost"]),
            "confirm_boost": float(latest["confirm_boost"]),
            "signal": int(latest["signal"]),
            "position": float(latest["position"]),
            "reason": str(latest["reason"]),
            "regime": str(latest["regime"]),
            "long_signal_raw": bool(latest["long_signal_raw"]),
            "long_signal": bool(latest["long_signal"]),
            "is_trending": bool(latest["is_trending"]),
            "is_breakout": bool(latest["is_breakout"]),
            "long_confirm": bool(latest["long_confirm"]),
            "meanrev_exit_warn": bool(latest["meanrev_exit_warn"]),
            "trend_exit_warn": bool(latest["trend_exit_warn"]),
            "imbalance_5": float(latest["imbalance_5"]),
            "dist_ma_15_z": float(latest["dist_ma_15_z"]),
            "meanrev_confluence": bool(latest["meanrev_confluence"]),
            "trend_confluence": bool(latest["trend_confluence"]),
            "breakout_confluence": bool(latest["breakout_confluence"]),
        }

    # -----------------------------------------------------
    # Balances / Prices
    # -----------------------------------------------------
    def get_usd_balance(self) -> float:
        balance = python_demo.get_balance()
        try:
            return float(balance["SpotWallet"]["USD"]["Free"])
        except Exception:
            return 0.0

    def get_coin_balance(self) -> float:
        balance = python_demo.get_balance()
        try:
            return float(balance["SpotWallet"][self.cfg.coin]["Free"])
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
    def get_remaining_slots(self) -> int:
        open_positions = self.count_open_positions()
        remaining = len(PAIR_CONFIGS) - open_positions
        return max(1, remaining)

    @staticmethod
    def round_to_step(qty: float, step: float) -> float:
        qty_dec = Decimal(str(qty))
        step_dec = Decimal(str(step))

        return float((qty_dec // step_dec) * step_dec)
    
    def place_buy(
        self,
        stake_pct: float,
        current_price: float,
        pred: float,
        pred_proba: float,
        entry_reason: str,
    ) -> Optional[PositionState]:
        usd_balance = self.get_usd_balance()
        if usd_balance <= 0:
            self.log("No USD balance available.", level="warning")
            return None

        remaining_slots = self.get_remaining_slots()
        slot_usd = usd_balance / remaining_slots
        alloc_usd = slot_usd * stake_pct

        step_size_map = {
            "SOL": 0.01,
            "LINK": 0.01,
            "AVAX": 0.01,
        }

        qty_raw = (alloc_usd * (1 - BUY_FEE_RATE)) / current_price
        qty = self.round_to_step(qty_raw, step_size_map[self.cfg.coin])

        if qty <= 0:
            self.log(f"Rounded qty became zero. raw_qty={qty_raw}", level="warning")
            return None

        buy_resp = python_demo.place_order(self.cfg.coin, "BUY", qty)

        if not isinstance(buy_resp, dict) or not buy_resp.get("Success"):
            self.log(
                f"BUY failed: {buy_resp} | raw_qty={qty_raw} | rounded_qty={qty}",
                level="error",
                send_tele=True,
            )
            return None

        position = PositionState(
            qty=float(qty),
            entry_price=float(current_price),
            entry_time=pd.Timestamp.utcnow().isoformat(),
            stake_pct=float(stake_pct),
            pred=float(pred),
            pred_proba=float(pred_proba),
            entry_reason=str(entry_reason),
        )

        usd_balance_after = self.get_usd_balance()

        self.log(
            f"BUY filled qty={position.qty:.8f} {self.cfg.coin} "
            f"entry≈{position.entry_price:.6f} "
            f"stake_pct={position.stake_pct:.4f} "
            f"pred={position.pred:.6f} "
            f"pred_proba={position.pred_proba:.6f} "
            f"reason={position.entry_reason} "
            f"usd_balance={usd_balance_after:.2f}",
            send_tele=True,
        )

        return position

    def market_sell_position(self) -> bool:
        if self.position is None:
            return False

        sell_resp = python_demo.place_order(
            self.cfg.coin,
            "SELL",
            self.position.qty,
        )

        ok = isinstance(sell_resp, dict) and sell_resp.get("Success")
        if not ok:
            self.log(f"SELL failed: {sell_resp}", level="error", send_tele=True)
        return ok

    def check_position_exit(self, sig: dict[str, Any]) -> Optional[str]:
        if self.position is None:
            return None

        live_price = self.get_live_price()
        if live_price <= 0:
            return None

        entry_reason = self.position.entry_reason

        pred_exit = sig["adjusted_pred"] < SELL_PRED_THRESHOLD
        imbalance_exit = sig["imbalance_5"] < SELL_IMBALANCE_THRESHOLD

        meanrev_reversion_exit = (
            (entry_reason == "meanrev_long")
            and (
                (sig["dist_ma_15_z"] > SELL_DIST_Z_THRESHOLD)
                or sig["meanrev_exit_warn"]
            )
        )

        trend_momentum_exit = (
            entry_reason in {"trend_long", "breakout_long"}
            and sig["trend_exit_warn"]
        )

        if not (pred_exit or imbalance_exit or meanrev_reversion_exit or trend_momentum_exit):
            return None

        ok = self.market_sell_position()
        if not ok:
            return None

        approx_pnl = (live_price - self.position.entry_price) * self.position.qty
        usd_balance = self.get_usd_balance()

        if pred_exit:
            exit_reason = "pred_exit"
        elif imbalance_exit:
            exit_reason = "imbalance_exit"
        elif meanrev_reversion_exit:
            exit_reason = "meanrev_revert_exit"
        else:
            exit_reason = "trend_momentum_exit"

        self.log(
            f"EXIT {exit_reason} qty={self.position.qty:.8f} {self.cfg.coin} "
            f"entry={self.position.entry_price:.6f} "
            f"exit≈{live_price:.6f} "
            f"pnl≈{approx_pnl:.6f} "
            f"adj_pred={sig['adjusted_pred']:.6f} "
            f"imbalance_5={sig['imbalance_5']:.6f} "
            f"dist_z={sig['dist_ma_15_z']:.6f} "
            f"usd_balance={usd_balance:.2f}",
            send_tele=True,
        )

        return exit_reason

    # -----------------------------------------------------
    # Main step
    # -----------------------------------------------------
    def step(self):
        self.refresh_data()

        if self.df.empty or len(self.df) < MIN_BARS:
            self.save_data()
            return

        self.save_data()

        sig_df = self.build_signal_frame()
        sig = self.get_latest_signal_values(sig_df)

        if sig is None:
            return

        # Exit first if already in position
        if self.position is not None:
            exit_reason = self.check_position_exit(sig)
            if exit_reason is not None:
                self.position = None
                self.save_position_state()
            return

        self.log(
            "latest_signal "
            f"pred={sig['pred']:.6f} "
            f"thr={sig['threshold']:.6f} "
            f"adj_pred={sig['adjusted_pred']:.6f} "
            f"signal={sig['signal']} "
            f"position={sig['position']:.4f} "
            f"reason={sig['reason']} "
            f"regime={sig['regime']} "
            f"confirm={sig['long_confirm']} "
            f"is_breakout={sig['is_breakout']} "
            f"mr_conf={sig['meanrev_confluence']} "
            f"trend_conf={sig['trend_confluence']} "
            f"bo_conf={sig['breakout_confluence']} "
            f"meanrev_warn={sig['meanrev_exit_warn']} "
            f"trend_warn={sig['trend_exit_warn']}"
        )

        if sig["signal"] != 1:
            return

        if sig["position"] <= 0:
            return

        with order_lock:
            if not self.can_open_new_position():
                return

            new_position = self.place_buy(
                stake_pct=sig["position"],
                current_price=sig["current_price"],
                pred=sig["pred"],
                pred_proba=sig["pred_proba"],
                entry_reason=sig["reason"],
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