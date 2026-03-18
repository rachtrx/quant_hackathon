#!/usr/bin/env bash
set -e

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

freqtrade backtesting \
  --config user_data/config.json \
  --strategy MlSignalStrategy \
  --data-format-ohlcv parquet