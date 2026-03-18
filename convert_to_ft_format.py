import os
import pandas as pd

DATA_DIR = "binance_data"
OUT_DIR = "user_data/data/binance"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
INTERVAL = "1m"

def symbol_to_pair(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT"
    raise ValueError(f"Unsupported symbol: {symbol}")

for symbol in SYMBOLS:
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{symbol}_{INTERVAL}.parquet"))

    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    pair = symbol_to_pair(symbol)   # BTC/USDT
    base, quote = pair.split("/")

    out_dir = os.path.join(OUT_DIR, base)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{quote}-{INTERVAL}.json")

    export_df = df.rename(columns={"open_time": "date"})
    export_df["date"] = export_df["date"].astype("int64") // 10**6

    export_df.to_json(out_path, orient="values")
    print("saved:", out_path)