import os
import io
import time
import zipfile
import requests
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from constants import PARQUET_PATH
from data_retrieval import load_or_build_data

def main() -> None:
    # load data
    df = load_or_build_data()
    print(f"[info] raw rows: {len(df):,}")

    # add features + target
    df = add_features(df)

    feature_cols = [
        "ret_1",
        "ret_3",
        "ret_5",
        "ret_15",
        "ret_30",
        "ret_60",
        "vol_5",
        "vol_15",
        "vol_30",
        "vol_60",
        "close_over_ma_5",
        "close_over_ma_15",
        "close_over_ma_60",
        "hl_spread",
        "co_spread",
        "close_pos_in_bar",
        "volume_ret_1",
        "volume_zscore_20",
        "num_trades_zscore_20",
        "taker_buy_ratio",
    ]

    target_col = f"target_ret_fwd_{TARGET_HORIZON}"

    model_df = df[["open_time"] + feature_cols + [target_col]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    print(f"[info] usable rows after features: {len(model_df):,}")

    train_df, test_df = time_split(model_df, train_frac=0.8)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"[info] train rows: {len(train_df):,}")
    print(f"[info] test rows:  {len(test_df):,}")

    # train model
    model = RandomForestRegressor(**RF_PARAMS)
    start = time.time()
    print("[training] fitting random forest...")
    model.fit(X_train, y_train)
    print(f"[training] done in {time.time() - start:.2f}s")

    # predict
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # evaluate
    print("[eval] computing metrics...")
    train_ic = information_coefficient(y_train.values, train_pred)
    test_ic = information_coefficient(y_test.values, test_pred)

    train_rank_ic = rank_information_coefficient(y_train.values, train_pred)
    test_rank_ic = rank_information_coefficient(y_test.values, test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    print("\n===== RESULTS =====")
    print(f"Train IC:      {train_ic:.6f}")
    print(f"Test IC:       {test_ic:.6f}")
    print(f"Train Rank IC: {train_rank_ic:.6f}")
    print(f"Test Rank IC:  {test_rank_ic:.6f}")
    print(f"Train RMSE:    {train_rmse:.6f}")
    print(f"Test RMSE:     {test_rmse:.6f}")

    # feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n===== FEATURE IMPORTANCE =====")
    print(importances)

    # save predictions
    out = test_df[["open_time", target_col]].copy()
    out["prediction"] = test_pred

    # convert predictions into simple signals
    out["signal"] = 0
    out.loc[out["prediction"] > BUY_THRESHOLD, "signal"] = 1
    out.loc[out["prediction"] < SELL_THRESHOLD, "signal"] = -1

    # naive realised strategy return:
    # use signal at time t to earn next-horizon forward return
    out["strategy_return"] = out["signal"] * out[target_col]

    print("\n===== SIGNAL COUNTS =====")
    print(out["signal"].value_counts(dropna=False).sort_index())

    avg_trade_return = out.loc[out["signal"] != 0, "strategy_return"].mean()
    total_strategy_return = out["strategy_return"].sum()

    print("\n===== SIMPLE SIGNAL STATS =====")
    print(f"Average return on signalled trades: {avg_trade_return:.6f}")
    print(f"Sum of strategy returns:            {total_strategy_return:.6f}")

    pred_path = os.path.join(DATA_DIR, f"{SYMBOL}_{INTERVAL}_rf_predictions.csv")
    out.to_csv(pred_path, index=False)
    print(f"\n[saved] predictions -> {pred_path}")


if __name__ == "__main__":
    main()