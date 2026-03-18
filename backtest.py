import os
import joblib
import json
import numpy as np
import pandas as pd
from features import add_features
from controller import controller
from constants import DATA_DIR, MODEL_DIR, MODEL_TYPES, SYMBOLS, TARGET_HORIZON
from utils import time_split

model_type=MODEL_TYPES["rf"]

def generate_live_style_signals(
    symbol: str,
    target_horizon: int,
    min_pred_history: int = 100,
    default_threshold: float = 0.001,
    pred_hist_window: int = 500,
):
    """
    Simulate live-style signal generation using only past data up to each step.

    Parameters
    ----------
    symbol : str
        Example: "BTCUSDT"
    target_horizon : int
        Same horizon used in training
    model_type : str
        "rf" or "xgb"
    min_pred_history : int
        Minimum past predictions needed before dynamic thresholding kicks in
    default_threshold : float
        Threshold used before enough pred history exists
    pred_hist_window : int
        Number of recent predictions to use for percentile threshold
    test_start_idx : int | None
        Index in raw chronological df where test period starts.
        If None, starts after a safe warmup.

    Returns
    -------
    pd.DataFrame
        Signal dataframe
    """

    parquet_path = os.path.join(DATA_DIR, f"{symbol}_1m.parquet")

    if model_type not in ["rf", "xgb"]:
        raise NotImplementedError(f"Unsupported model_type: {model_type}")

    meta_path = os.path.join(MODEL_DIR, model_type, f"{symbol}__h{target_horizon}_meta.json")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)

    model_path = meta["model_path"]
    features_path = meta["feature_importance_path"]

    # load raw data
    df = pd.read_parquet(parquet_path).sort_values("open_time").reset_index(drop=True)

    # load trained model
    model = joblib.load(model_path)

    # load saved feature columns
    with open(features_path, "r") as f:
        feature_cols = json.load(f)

    pred_history = []
    rows_out = []

    test_start_time = pd.to_datetime(meta["test_start_time"], utc=True)
    test_end_time = pd.to_datetime(meta["test_end_time"], utc=True)

    matches = df.index[df["open_time"] == test_start_time]
    if len(matches) == 0:
        raise ValueError(f"Could not find test_start_time {test_start_time} in raw parquet for {symbol}")

    test_start_idx = int(matches[0])

    for end_idx in range(test_start_idx + 1, len(df) + 1):
        df_slice = df.iloc[:end_idx].copy()

        feat_df, _ = add_features(df_slice, target_horizon=target_horizon)

        # shouldnt happen
        if len(feat_df) == 0:
            continue

        row = feat_df.iloc[-1]

        if row["open_time"] > test_end_time:
            break

        # shouldnt happen
        if row[feature_cols].isna().any():
            continue

        X = row[feature_cols].to_frame().T
        pred = model.predict(X)[0]

        recent_preds = np.array(pred_history[-pred_hist_window:])
        if len(recent_preds) >= min_pred_history:
            threshold = np.percentile(np.abs(recent_preds), 80)
        else:
            threshold = default_threshold

        decision = controller(row, pred, threshold=threshold)

        rows_out.append({
            "open_time": row["open_time"],
            "close_time": row["close_time"],
            "pred": pred,
            "threshold": threshold,
            "signal": decision["signal"],
            "position": decision["position"],
            "reason": decision["reason"],
            "long_signal_raw": decision["long_signal_raw"],
            "short_signal_raw": decision["short_signal_raw"],
            "long_signal": decision["long_signal"],
            "short_signal": decision["short_signal"],
            "is_trending": decision["is_trending"],
            "is_breakout": decision["is_breakout"],
            "long_confirm": decision["long_confirm"],
            "short_confirm": decision["short_confirm"],
        })

        pred_history.append(pred)

    out = pd.DataFrame(rows_out)
    return out

if __name__ == "__main__":
    for symbol in SYMBOLS:
        print(f"\n===== {symbol} =====")

        parquet_path = f"{DATA_DIR}/{symbol}_1m.parquet"

        # raw chronological data
        raw_df = pd.read_parquet(parquet_path).sort_values("open_time").reset_index(drop=True)

        # rebuild feature dataframe exactly like training
        feat_df, feature_cols = add_features(raw_df.copy(), TARGET_HORIZON)
        target_col = f"target_ret_fwd_{TARGET_HORIZON}"

        model_df = feat_df[["open_time"] + feature_cols + [target_col]].copy()
        model_df = model_df.replace([np.inf, -np.inf], np.nan)
        model_df = model_df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

        print(f"[info] raw rows:    {len(raw_df):,}")
        print(f"[info] usable rows: {len(model_df):,}")

        # exact same chronological split as training
        train_df, test_df = time_split(model_df, train_frac=0.8)

        if len(test_df) == 0:
            print("[warn] empty test_df, skipping")
            continue

        first_test_time = test_df["open_time"].iloc[0]

        matches = raw_df.index[raw_df["open_time"] == first_test_time]
        if len(matches) == 0:
            print("[warn] could not map first_test_time back to raw_df, skipping")
            continue

        test_start_idx = int(matches[0])

        print(f"[info] test rows:         {len(test_df):,}")
        print(f"[info] first test time:   {first_test_time}")
        print(f"[info] raw test start idx:{test_start_idx}")

        signals_df = generate_live_style_signals(
            symbol=symbol,
            target_horizon=TARGET_HORIZON,
            test_start_idx=test_start_idx,
        )

        # re-create full feature df for execution backtest
        raw_feat_df, _ = add_features(raw_df.copy(), TARGET_HORIZON)
        raw_feat_df = raw_feat_df.replace([np.inf, -np.inf], np.nan)
        raw_feat_df = raw_feat_df.dropna().reset_index(drop=True)

        # optional: restrict to true test period only
        raw_feat_df = raw_feat_df[raw_feat_df["open_time"] >= first_test_time].reset_index(drop=True)

        signal_map = signals_df.set_index("open_time")[["position", "reason"]].to_dict("index")

        trades = []
        i = 0
        while i < len(raw_feat_df) - TARGET_HORIZON:
            row = raw_feat_df.iloc[i]
            ot = row["open_time"]

            if ot not in signal_map:
                i += 1
                continue

            pos = signal_map[ot]["position"]
            reason = signal_map[ot]["reason"]

            if pos == 0:
                i += 1
                continue

            entry_price = row["close"]
            exit_row = raw_feat_df.iloc[i + TARGET_HORIZON]
            exit_price = exit_row["close"]

            gross_ret = (exit_price / entry_price - 1.0) * pos

            trades.append({
                "open_time": ot,
                "exit_time": exit_row["open_time"],
                "position": pos,
                "reason": reason,
                "trade_ret": gross_ret,
            })

            i += TARGET_HORIZON

        trades_df = pd.DataFrame(trades)

        if len(trades_df) == 0:
            print("[info] no trades")
            continue

        trades_df["equity"] = (1 + trades_df["trade_ret"]).cumprod()

        print(trades_df.tail())
        print("num trades:", len(trades_df))
        print("win rate:", (trades_df["trade_ret"] > 0).mean())
        print("avg trade ret:", trades_df["trade_ret"].mean())
        print("final equity:", trades_df["equity"].iloc[-1])
        print(trades_df.groupby("reason")["trade_ret"].agg(["count", "mean", "std"]))