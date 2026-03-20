#!/usr/bin/env bash
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: ./backtest.sh <StrategyName> <RunName>"
  echo "Example: ./backtest.sh MlMeanRevStrategy z2_vol1.1"
  exit 1
fi

STRATEGY="$1"
RUN_NAME="$2"

# Generate timestamp: YYYYMMDD_HHMMSS
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Final filename
EXPORT_NAME="bt_${STRATEGY}_${RUN_NAME}_${TIMESTAMP}.json"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

freqtrade backtesting \
  --config user_data/config.json \
  --strategy "$STRATEGY" \
  --data-format-ohlcv parquet \
  --timerange 20260201-20260316 \
  --export trades \
  --export-filename "$EXPORT_NAME"

echo "Saved as: $EXPORT_NAME"