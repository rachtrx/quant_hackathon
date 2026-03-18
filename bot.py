import os
import time
import pandas as pd
import numpy as np
import joblib
import json

import controller
from data_retrieval import fetch_recent_klines_rest
from constants import DATA_DIR, MODEL_DIR, INTERVAL, SYMBOLS, TARGET_HORIZON
from features import add_features

BUFFER_SIZE = 500

model_dict = {}
feature_cols_dict = {}
meta_dict = {}
pred_history_dict = {symbol: [] for symbol in SYMBOLS}
last_processed_time = {symbol: None for symbol in SYMBOLS}

for symbol in SYMBOLS:
    model_path = os.path.join(MODEL_DIR, f"{symbol}__h{TARGET_HORIZON}_rf_model.joblib")
    features_path = os.path.join(MODEL_DIR, f"{symbol}__h{TARGET_HORIZON}_feature_cols.json")
    meta_path = os.path.join(MODEL_DIR, f"{symbol}__h{TARGET_HORIZON}_meta.json")

    model_dict[symbol] = joblib.load(model_path)

    with open(features_path, "r") as f:
        feature_cols_dict[symbol] = json.load(f)

    with open(meta_path, "r") as f:
        meta_dict[symbol] = json.load(f)

df_dict = {
    symbol: (
        pd.read_parquet(os.path.join(DATA_DIR, f"{symbol}_{INTERVAL}.parquet"))
        .sort_values("open_time")
        .drop_duplicates(subset=["open_time"])
        .tail(BUFFER_SIZE)
        .reset_index(drop=True)
    )
    for symbol in SYMBOLS
}

while True:
    try:
        sleep_time = 60 - (time.time() % 60)
        time.sleep(sleep_time + 1)

        for symbol in SYMBOLS:
            latest = fetch_recent_klines_rest(symbol, INTERVAL, limit=2)

            df = pd.concat([df_dict[symbol], latest], ignore_index=True)
            df = df.drop_duplicates(subset=["open_time"])
            df = df.sort_values("open_time")
            df = df.tail(BUFFER_SIZE).reset_index(drop=True)
            df_dict[symbol] = df

            target_horizon = meta_dict[symbol]["target_horizon"]
            feature_cols = feature_cols_dict[symbol]
            model = model_dict[symbol]

            feat_df, _ = add_features(df, target_horizon=target_horizon)

            if len(feat_df) < 2:
                continue

            row = feat_df.iloc[-2]

            if row[feature_cols].isna().any():
                continue

            if last_processed_time[symbol] == row["close_time"]:
                continue

            last_processed_time[symbol] = row["close_time"]

            X = row[feature_cols].to_frame().T
            pred = model.predict(X)[0]

            pred_history_dict[symbol].append(pred)
            recent_preds = np.array(pred_history_dict[symbol][-500:])

            threshold = np.percentile(np.abs(recent_preds), 80) if len(recent_preds) >= 100 else 0.001
            decision = controller(row, pred, threshold=threshold)

            print(
                f"{symbol} | "
                f"time={row['close_time']} | "
                f"pred={pred:.6f} | "
                f"signal={decision['signal']} | "
                f"reason={decision['reason']}"
            )

    except Exception as e:
        print("error:", e)
        time.sleep(5)