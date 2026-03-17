from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import zipfile

import pandas as pd
import requests

from constants import BINANCE_SPOT_KLINES_API_URL, BINANCE_SPOT_KLINES_VISION_URL, DATA_DIR, RAW_DIR, SYMBOLS, INTERVAL, MONTHS, MARKET

def make_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)


def build_binance_vision_url(
    symbol: str,
    interval: str,
    date_str: str,
    market: str = "spot",
    freq: str = "monthly"
) -> str:
    if market == "spot":
        base = f"{BINANCE_SPOT_KLINES_VISION_URL}/{freq}/klines/{symbol}/{interval}"
    else:
        raise NotImplementedError()
        # base = f"https://data.binance.vision/data/futures/um/{freq}/klines/{symbol}/{interval}"

    filename = f"{symbol}-{interval}-{date_str}.zip"
    return f"{base}/{filename}"

def try_roll_previous_month_daily_into_monthly(
    symbol: str,
    interval: str,
    market: str = "spot",
) -> None:
    now_utc = pd.Timestamp.now(tz="UTC")
    prev_month = (now_utc.to_period("M") - 1)
    prev_month_str = str(prev_month)

    # Only meaningful once we're no longer in that month
    monthly_out_dir = os.path.join(RAW_DIR, "monthly")
    daily_out_dir = os.path.join(RAW_DIR, "daily")
    os.makedirs(monthly_out_dir, exist_ok=True)
    os.makedirs(daily_out_dir, exist_ok=True)

    try:
        monthly_zip = download_one_archive(
            symbol=symbol,
            interval=interval,
            date_str=prev_month_str,
            market=market,
            freq="monthly",
        )
        print(f"[rollup] monthly archive ready: {monthly_zip}")
    except Exception as exc:
        print(f"[rollup] previous month monthly zip not available yet: {exc}")
        return

    # Only delete previous-month daily zips after monthly download succeeds
    prefix = f"{symbol}-{interval}-{prev_month_str}-"
    deleted = 0
    if os.path.exists(daily_out_dir):
        for fname in os.listdir(daily_out_dir):
            if fname.startswith(prefix) and fname.endswith(".zip"):
                path = os.path.join(daily_out_dir, fname)
                try:
                    os.remove(path)
                    deleted += 1
                except Exception as exc:
                    print(f"[warn] failed deleting {path}: {exc}")

    print(f"[rollup] deleted {deleted} previous-month daily zips for {prev_month_str}")

def download_one_archive(
    symbol: str,
    interval: str,
    date_str: str,
    market: str = "spot",
    freq: str = "monthly",
) -> str:
    url = build_binance_vision_url(symbol, interval, date_str, market, freq)
    out_dir = os.path.join(RAW_DIR, freq)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}-{interval}-{date_str}.zip")

    if os.path.exists(out_path):
        print(f"[skip] already downloaded: {out_path}")
        return out_path

    print(f"[download] {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


def download_many_archives(
    symbol: str,
    interval: str,
    date_strs: list[str],
    market: str = "spot",
    freq: str = "monthly",
    max_workers: int = 8,
) -> list[str]:
    paths: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_one_archive, symbol, interval, d, market, freq): d
            for d in date_strs
        }
        for future in as_completed(futures):
            d = futures[future]
            try:
                path = future.result()
                paths.append(path)
                print(f"[done] {freq} {d}")
            except Exception as exc:
                print(f"[error] failed for {freq} {d}: {exc}")

    return sorted(paths)


def _infer_public_time_unit(series: pd.Series, market: str) -> str:
    """
    Binance public spot data moved to microseconds from 2025-01-01 onward.
    Heuristic:
      - 16-digit-ish -> microseconds
      - 13-digit-ish -> milliseconds
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "ms"

    sample = int(s.iloc[0])

    # ~2025 in microseconds is around 1.7e15, milliseconds around 1.7e12
    if market == "spot" and sample >= 10**15:
        return "us"
    return "ms"


def read_kline_zip(zip_path: str, market: str = "spot") -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            raise ValueError(f"No files inside zip: {zip_path}")

        with zf.open(names[0]) as f:
            df = pd.read_csv(f, header=None, names=columns)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Public ZIP timestamps can be ms or us depending on source/date.
    time_unit = _infer_public_time_unit(df["open_time"], market=market)
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit=time_unit, utc=True)
    df["close_time"] = pd.to_datetime(pd.to_numeric(df["close_time"], errors="coerce"), unit=time_unit, utc=True)

    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def combine_and_save_parquet(frames: list[pd.DataFrame], parquet_path: str) -> pd.DataFrame:
    if not frames:
        raise ValueError("No dataframes were loaded.")

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    # drop useless mixed-type column
    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    print(f"[saved] parquet -> {parquet_path}")

    return df


def month_strings_between(start_ym: str, end_ym: str) -> list[str]:
    start = pd.Period(start_ym, freq="M")
    end = pd.Period(end_ym, freq="M")
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def day_strings_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[str]:
    if end_date < start_date:
        return []
    days = pd.date_range(start_date.normalize(), end_date.normalize(), freq="D", tz="UTC")
    return [d.strftime("%Y-%m-%d") for d in days]


def fetch_recent_klines_rest(
    symbol: str,
    interval: str,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if start_time_ms is not None:
        params["startTime"] = start_time_ms
    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    r = requests.get(BINANCE_SPOT_KLINES_API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if not data:
        return pd.DataFrame(columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "num_trades", "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume", "ignore",
        ])

    df = pd.DataFrame(data, columns=[
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ])

    numeric_cols = [
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # REST klines are standard epoch milliseconds.
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(pd.to_numeric(df["close_time"]), unit="ms", utc=True)
    return df.sort_values("open_time").reset_index(drop=True)

def refresh_latest_data(
    symbol: str,
    interval: str,
    market: str,
    parquet_path: str,
    base_months: list[str],
) -> pd.DataFrame:
    """
    Refresh strategy:
      1) load existing parquet if present
      2) ensure base monthly history exists
      3) fetch current-month completed DAILY archives from Vision
      4) fetch latest REST klines after the latest known open_time
      5) merge and save
    """
    make_dirs()
    try_roll_previous_month_daily_into_monthly(
        symbol=symbol,
        interval=interval,
        market=market,
    )

    frames: list[pd.DataFrame] = []

    # Existing parquet
    existing = None
    if os.path.exists(parquet_path):
        print(f"[load] existing parquet: {parquet_path}")
        existing = pd.read_parquet(parquet_path)
        if not existing.empty:
            existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True)
            existing["close_time"] = pd.to_datetime(existing["close_time"], utc=True, errors="coerce")
            frames.append(existing)

    # Base monthly archives
    monthly_paths = download_many_archives(
        symbol=symbol,
        interval=interval,
        date_strs=base_months,
        market=market,
        freq="monthly",
    )
    for path in monthly_paths:
        print(f"[read] {path}")
        frames.append(read_kline_zip(path, market=market))

    # Current month daily archives (completed days only)
    now_utc = pd.Timestamp.now(tz="UTC")
    first_day_of_month = now_utc.normalize().replace(day=1)
    yesterday = (now_utc - pd.Timedelta(days=1)).normalize()

    if yesterday >= first_day_of_month:
        daily_strs = day_strings_between(first_day_of_month, yesterday)
        daily_paths = download_many_archives(
            symbol=symbol,
            interval=interval,
            date_strs=daily_strs,
            market=market,
            freq="daily",
        )
        for path in daily_paths:
            print(f"[read] {path}")
            frames.append(read_kline_zip(path, market=market))

    # Merge what we have so far to determine last timestamp
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not merged.empty:
        merged = merged.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        last_open_time = pd.to_datetime(merged["open_time"].max(), utc=True)
        start_ms = int(last_open_time.timestamp() * 1000)
    else:
        start_ms = None

    # Pull latest recent klines from REST
    # For spot, REST endpoint is /api/v3/klines.
    # If you later use futures, switch endpoint accordingly.
    latest_rest = fetch_recent_klines_rest(
        symbol=symbol,
        interval=interval,
        start_time_ms=start_ms,
        limit=1000,
    )
    if not latest_rest.empty:
        print(f"[info] pulled {len(latest_rest):,} recent REST klines")
        frames.append(latest_rest)

    final_df = combine_and_save_parquet(frames, parquet_path)
    return final_df


def main() -> None:
    for symbol in SYMBOLS:
        parquet_path=os.path.join(DATA_DIR, f"{symbol}_{INTERVAL}.parquet")
        df = refresh_latest_data(
            symbol=symbol,
            interval=INTERVAL,
            market=MARKET,
            parquet_path=parquet_path,
            base_months=MONTHS,
        )
        print(f"[info] rows after refresh: {len(df):,}")
        print(df.tail())
        time.sleep(10)

if __name__ == "__main__":
    main()