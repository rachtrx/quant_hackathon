import os

MODEL_TYPES = ["rf"]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT"]
MARKET = "spot"          # "spot" or "futures/um"
INTERVAL = "1m"
MONTHS = [
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
]

BINANCE_SPOT_KLINES_API_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_SPOT_KLINES_VISION_URL = "https://data.binance.vision/data/spot"
TARGET_HORIZON = 5

DATA_DIR = "binance_data"
MODEL_DIR = "models"
RAW_DIR = os.path.join(DATA_DIR, "raw")
