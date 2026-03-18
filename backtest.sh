#!/usr/bin/env bash
set -e

export PYTHONPATH="${PYTHONPATH}:."

freqtrade backtesting \
  --config user_data/config.json \
  --strategy MlSignalStrategy \
  --timeframe 1m \
  --timerange 20260301-20260316