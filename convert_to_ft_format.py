import os
import pandas as pd
from constants import SYMBOLS

DATA_DIR = "binance_data"
OUT_DIR = "user_data/data/binance"

INTERVAL = "1m"

def symbol_to_pair(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT"
    raise ValueError(f"Unsupported symbol: {symbol}")

os.makedirs(OUT_DIR, exist_ok=True)

for symbol in SYMBOLS:
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{symbol}_{INTERVAL}.parquet"))

    # keep only required columns
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()

    # convert to datetime (IMPORTANT)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    # rename to freqtrade format
    df = df.rename(columns={"open_time": "date"})

    # sort + clean
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # convert symbol → BTC_USDT
    pair = symbol_to_pair(symbol)
    base, quote = pair.split("/")
    pair_name = f"{base}_{quote}"

    # correct filename
    out_path = os.path.join(OUT_DIR, f"{pair_name}-{INTERVAL}.parquet")

    # save parquet
    df.to_parquet(out_path, index=False)

    print("saved:", out_path)