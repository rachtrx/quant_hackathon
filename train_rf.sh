#!/bin/bash

set -e  # stop on error

NOTEBOOK="train_rf.ipynb"
OUTPUT_DIR="outputs"
TARGET_HORIZON=5

SYMBOLS=(
  BTCUSDT
  ETHUSDT
  BNBUSDT
  ADAUSDT
  XRPUSDT
)

mkdir -p "$OUTPUT_DIR/rf"

echo "Starting papermill runs..."

for SYMBOL in "${SYMBOLS[@]}"; do
  echo "Running for $SYMBOL..."

  papermill "$NOTEBOOK" \
    "$OUTPUT_DIR/out_${SYMBOL}.ipynb" \
    -p SYMBOL "$SYMBOL" \
    -p TARGET_HORIZON "$TARGET_HORIZON"

  echo "Finished $SYMBOL"
done

echo "All runs completed."