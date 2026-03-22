import os

MODEL_TYPES = ["rf", "xgb"]

# ADA - slow, low edge
# DOT - low participation
# LTC - dead liquidity vs others
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "LINKUSDT",
    "AVAXUSDT",

    # alpha layer
    "APTUSDT",
    "ARBUSDT",

    # optional
    "OPUSDT",
    "MATICUSDT",
    "SUIUSDT",

    # "ADAUSDT",
    # "DOTUSDT",
    # "LTCUSDT"
]

MARKET = "spot"          # "spot" or "futures/um"
INTERVAL = "5m"
MONTHS = [
    "2025-06",
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
]

BINANCE_SPOT_KLINES_API_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_SPOT_KLINES_VISION_URL = "https://data.binance.vision/data/spot"
TARGET_HORIZON = 6

DATA_DIR = "binance_data"
MODEL_DIR = "models"
RAW_DIR = os.path.join(DATA_DIR, "raw")
