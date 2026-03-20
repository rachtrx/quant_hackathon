#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from roostoo import python_demo
from features import add_features
from data_retrieval import fetch_recent_klines_rest
from live_testing import tele_update

# =========================================================
# GLOBAL CONFIG
# =========================================================
INTERVAL = "1m"

MODEL_DIR = "models/rf"
DATA_DIR = "live_testing/data"
STATE_DIR = "live_testing/state"

TARGET_HORIZON = 5  # fallback only if meta does not contain target_horizon

RESAMPLE_RULE = "30min"
WINDOW_30M = 10
ENTRY_DEV_THRESHOLD = -2.0
STOP_DEV = -2.5
TP_DEV = 0.0

VOL_FILTER_THRESH = 1.1
VOL_SHORT_WINDOW_1M = 5
VOL_LONG_WINDOW_30M = 30

BASE_SIZE_SMALL = 0.005
BASE_SIZE_MED = 0.01
BASE_SIZE_LARGE = 0.02

MAX_BARS_1M = 2000
MIN_BARS_1M = 1200

BUY_FEE_RATE = 0.001
MIN_ORDER_VALUE_USD = 1.0

PAIR_CONFIGS = [
    {"pair": "ADA/USD", "symbol": "ADAUSDT", "coin": "ADA"},
    {"pair": "XRP/USD", "symbol": "XRPUSDT", "coin": "XRP"},
    {"pair": "LINK/USD", "symbol": "LINKUSDT", "coin": "LINK"},
    # {"pair": "DOT/USD", "symbol": "DOTUSDT", "coin": "DOT"},
    # {"pair": "AVAX/USD", "symbol": "AVAXUSDT", "coin": "AVAX"},
    # {"pair": "SOL/USD", "symbol": "SOLUSDT", "coin": "SOL"},
    # {"pair": "LTC/USD", "symbol": "LTCUSDT", "coin": "LTC"},
    # {"pair": "BNB/USD", "symbol": "BNBUSDT", "coin": "BNB"},
    # {"pair": "ETH/USD", "symbol": "ETHUSDT", "coin": "ETH"},
    # {"pair": "BTC/USD", "symbol": "BTCUSDT", "coin": "BTC"},
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
    entry_mean: float
    entry_std: float
    tp_price: float
    sl_price: float
    tp_order_id: Optional[str] = None


# =========================================================
# TRADER
# =========================================================
class CoinTrader:
    def __init__(self, cfg: PairConfig):
        self.cfg = cfg

        self.meta_path = os.path.join(MODEL_DIR, f"{cfg.symbol}__h5_meta.json")
        self.meta = self.load_meta()

        self.model_path = self.meta["model_path"]
        self.features_path = self.meta["feature_cols_path"]
        self.target_horizon = self.meta.get("target_horizon", TARGET_HORIZON)

        self.parquet_path = os.path.join(DATA_DIR, f"{cfg.symbol}_1m_live.parquet")
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
    def log(self, msg: str, send_tele: bool = False, force_print: bool = False):
        full = f"[{self.cfg.symbol}] {msg}"

        if force_print:
            print(full)

        if send_tele:
            try:
                tele_update.send(full)
            except Exception:
                pass

    # -----------------------------------------------------
    # Setup / Persistence
    # -----------------------------------------------------
    def ensure_dirs(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(STATE_DIR, exist_ok=True)

    def load_meta(self):
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Meta file not found: {self.meta_path}")

        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        return meta

    def load_model_and_features(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Feature cols file not found: {self.features_path}")

        model = joblib.load(self.model_path)
        with open(self.features_path, "r") as f:
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
                    .tail(MAX_BARS_1M)
                )
                return df.reset_index(drop=True)
        return pd.DataFrame()

    def save_data(self):
        self.df.to_parquet(self.parquet_path, index=False)

    def save_position_state(self):
        if self.position is None:
            if os.path.exists(self.position_state_path):
                os.remove(self.position_state_path)
            return

        with open(self.position_state_path, "w") as f:
            json.dump(asdict(self.position), f, indent=2)

    def load_position_state(self) -> Optional[PositionState]:
        if not os.path.exists(self.position_state_path):
            return None
        with open(self.position_state_path, "r") as f:
            data = json.load(f)
        return PositionState(**data)

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------
    def refresh_1m_data(self):
        if self.df.empty:
            recent = fetch_recent_klines_rest(
                self.cfg.symbol, INTERVAL, limit=MAX_BARS_1M
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
            self.df = (
                self.df.drop_duplicates(subset=["open_time"])
                .sort_values("open_time")
                .tail(MAX_BARS_1M)
                .reset_index(drop=True)
            )

    def build_signal_frame(self):
        df = self.df.copy()

        df_features, _ = add_features(df, self.target_horizon)

        df_30 = (
            df.set_index("open_time")
            .resample(RESAMPLE_RULE)
            .agg({"close": "last"})
            .dropna()
            .copy()
        )

        df_30["mean"] = df_30["close"].rolling(WINDOW_30M).mean()
        df_30["std"] = df_30["close"].rolling(WINDOW_30M).std()

        df["ret_1m"] = df["close"].pct_change()
        vol_5 = df["ret_1m"].rolling(VOL_SHORT_WINDOW_1M).std().iloc[-1]

        df_30["ret_30m"] = df_30["close"].pct_change()
        df_30["vol_30m"] = df_30["ret_30m"].rolling(VOL_LONG_WINDOW_30M).std()
        df_30["vol_5_latest"] = vol_5

        return df_features, df_30

    def get_latest_signal_values(
        self, df_features: pd.DataFrame, df_30: pd.DataFrame
    ):
        if df_features.empty or df_30.empty:
            return None

        latest_30 = df_30.iloc[-1]

        current_price = float(latest_30["close"])
        current_mean = float(latest_30["mean"])
        current_std = float(latest_30["std"])
        vol_30 = (
            float(latest_30["vol_30m"])
            if pd.notna(latest_30["vol_30m"])
            else np.nan
        )
        vol_5 = (
            float(latest_30["vol_5_latest"])
            if pd.notna(latest_30["vol_5_latest"])
            else np.nan
        )

        if np.isnan(current_mean) or np.isnan(current_std) or current_std <= 0:
            return None
        if np.isnan(vol_30) or vol_30 <= 0:
            return None
        if np.isnan(vol_5):
            return None

        dev = (current_price - current_mean) / current_std
        vol_ratio = vol_5 / vol_30

        missing = [c for c in self.feature_cols if c not in df_features.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")

        latest_feat_row = df_features.iloc[-1][self.feature_cols]
        if latest_feat_row.isna().any():
            return None

        pred = float(self.model.predict(latest_feat_row.values.reshape(1, -1))[0])
        pred_price = current_price * (1.0 + pred)
        pred_dev = (pred_price - current_mean) / current_std

        return {
            "current_price": current_price,
            "current_mean": current_mean,
            "current_std": current_std,
            "dev": dev,
            "vol_5": vol_5,
            "vol_30": vol_30,
            "vol_ratio": vol_ratio,
            "pred": pred,
            "pred_price": pred_price,
            "pred_dev": pred_dev,
        }

    # -----------------------------------------------------
    # Sizing / Balances / Prices
    # -----------------------------------------------------
    def choose_position_size_pct(self, pred_dev: float) -> float:
        if pred_dev < -1:
            return 0.0
        elif -1 <= pred_dev < 0:
            return BASE_SIZE_SMALL
        elif 0 <= pred_dev < 1:
            return BASE_SIZE_MED
        else:
            return BASE_SIZE_LARGE

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
    def place_buy_and_tp(
        self,
        size_pct: float,
        current_price: float,
        current_mean: float,
        current_std: float,
    ):
        with order_lock:
            usd_balance = self.get_usd_balance()
            if usd_balance <= 0:
                return None

            qty = (usd_balance * size_pct * (1 - BUY_FEE_RATE)) / current_price

            if qty * current_price < MIN_ORDER_VALUE_USD:
                qty = (MIN_ORDER_VALUE_USD + 0.1) / current_price

            buy_resp = python_demo.place_order(self.cfg.pair, "BUY", qty)

            if not isinstance(buy_resp, dict) or not buy_resp.get("Success"):
                self.log(
                    f"BUY failed: {buy_resp}",
                    send_tele=True,
                    force_print=True,
                )
                return None

            tp_price = current_mean + TP_DEV * current_std
            sl_price = current_mean + STOP_DEV * current_std

            tp_resp = python_demo.place_order(
                self.cfg.pair,
                "SELL",
                qty,
                tp_price,
                order_type="LIMIT",
            )

            tp_order_id = None
            if isinstance(tp_resp, dict):
                tp_order_id = tp_resp.get("OrderDetail", {}).get("OrderID")

            self.log(
                f"ENTRY BUY qty={qty:.8f} {self.cfg.coin} "
                f"entry={current_price:.2f} tp={tp_price:.2f} sl={sl_price:.2f} "
                f"size_pct={size_pct * 100:.2f}% "
                f"dev={(current_price - current_mean) / current_std:.3f}",
                send_tele=True,
                force_print=True,
            )

            return PositionState(
                qty=float(qty),
                entry_price=float(current_price),
                entry_time=pd.Timestamp.utcnow().isoformat(),
                entry_mean=float(current_mean),
                entry_std=float(current_std),
                tp_price=float(tp_price),
                sl_price=float(sl_price),
                tp_order_id=str(tp_order_id) if tp_order_id is not None else None,
            )

    def cancel_tp_if_needed(self):
        if self.position and self.position.tp_order_id:
            try:
                python_demo.cancel_order(order_id=self.position.tp_order_id)
            except Exception as e:
                self.log(
                    f"Failed to cancel TP order {self.position.tp_order_id}: {e}",
                    send_tele=True,
                    force_print=True,
                )

    def is_tp_filled(self) -> bool:
        if self.position is None:
            return False

        coin_balance = self.get_coin_balance()

        # If most/all of the coin is gone, assume TP limit sell filled.
        # Using 10% leftover threshold to allow for minor residuals.
        return coin_balance < max(1e-8, self.position.qty * 0.1)

    def check_position_exit(self) -> Optional[str]:
        if self.position is None:
            return None

        live_price = self.get_live_price()
        if live_price <= 0:
            return None

        # TP: only count it as closed if we can confirm the coin was actually sold.
        if live_price >= self.position.tp_price:
            if self.is_tp_filled():
                usd_balance = self.get_usd_balance()
                approx_pnl = (
                    (self.position.tp_price - self.position.entry_price)
                    * self.position.qty
                )

                self.log(
                    f"TP FILLED qty={self.position.qty:.8f} {self.cfg.coin} "
                    f"entry={self.position.entry_price:.2f} "
                    f"exit≈{self.position.tp_price:.2f} "
                    f"pnl≈{approx_pnl:.2f} "
                    f"usd_balance={usd_balance:.2f}",
                    send_tele=True,
                    force_print=True,
                )
                return "tp"

        # SL: we actively market-sell, so can log right away.
        if live_price <= self.position.sl_price:
            self.cancel_tp_if_needed()

            sell_resp = python_demo.place_order(
                self.cfg.pair,
                "SELL",
                self.position.qty,
            )

            usd_balance = self.get_usd_balance()
            approx_pnl = (
                (live_price - self.position.entry_price) * self.position.qty
            )

            self.log(
                f"STOPPED OUT qty={self.position.qty:.8f} {self.cfg.coin} "
                f"entry={self.position.entry_price:.2f} "
                f"exit≈{live_price:.2f} "
                f"sl={self.position.sl_price:.2f} "
                f"pnl≈{approx_pnl:.2f} "
                f"usd_balance={usd_balance:.2f} "
                f"resp={sell_resp}",
                send_tele=True,
                force_print=True,
            )
            return "sl"

        return None

    # -----------------------------------------------------
    # Main step
    # -----------------------------------------------------
    def step(self):
        self.refresh_1m_data()

        if self.df.empty or len(self.df) < MIN_BARS_1M:
            self.save_data()
            return

        self.save_data()

        if self.position is not None:
            exit_reason = self.check_position_exit()
            if exit_reason is not None:
                self.position = None
                self.save_position_state()
            return

        df_features, df_30 = self.build_signal_frame()
        sig = self.get_latest_signal_values(df_features, df_30)

        if sig is None:
            return

        entry_ok = (
            sig["dev"] <= ENTRY_DEV_THRESHOLD
            and sig["vol_ratio"] < VOL_FILTER_THRESH
            and sig["pred"] > 0
        )

        if not entry_ok:
            return

        size_pct = self.choose_position_size_pct(sig["pred_dev"])
        if size_pct <= 0:
            return

        new_position = self.place_buy_and_tp(
            size_pct=size_pct,
            current_price=sig["current_price"],
            current_mean=sig["current_mean"],
            current_std=sig["current_std"],
        )

        if new_position is not None:
            self.position = new_position
            self.save_position_state()

    def run_forever(self):
        self.log(
            f"strategy started | loaded_rows={len(self.df)} | "
            f"open_position={self.position is not None}",
            force_print=True,
        )

        while True:
            try:
                self.step()
            except Exception as e:
                self.log(f"ERROR: {e}", send_tele=True, force_print=True)
            sleep_to_next_minute()

def run_trader(cfg_dict):
    trader = CoinTrader(PairConfig(**cfg_dict))
    trader.run_forever()

def sleep_to_next_minute():
    now = time.time()
    next_minute = (int(now // 60) + 1) * 60
    sleep_time = next_minute - now
    time.sleep(sleep_time)

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