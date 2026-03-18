from pathlib import Path
import subprocess
import sys

from constants import SYMBOLS, TARGET_HORIZON, MODEL_TYPES

NOTEBOOK = "train.ipynb"
OUTPUT_DIR = Path("outputs")


def run_one(symbol: str, model_type: str, target_horizon: int) -> None:
    run_dir = OUTPUT_DIR / model_type
    run_dir.mkdir(parents=True, exist_ok=True)

    output_notebook = run_dir / f"{symbol}.ipynb"

    cmd = [
        "papermill",
        NOTEBOOK,
        str(output_notebook),
        "-p", "SYMBOL", symbol,
        "-p", "TARGET_HORIZON", str(target_horizon),
        "-p", "MODEL_TYPE", model_type,
    ]

    print(f"Running {model_type} for {symbol}...")
    subprocess.run(cmd, check=True)
    print(f"Finished {model_type} - {symbol}")


def main() -> None:
    print("Starting papermill runs...")

    for symbol in SYMBOLS:
        for model_type in MODEL_TYPES:
            run_one(symbol, model_type, TARGET_HORIZON)

    print("All runs completed.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {e.cmd}", file=sys.stderr)
        raise