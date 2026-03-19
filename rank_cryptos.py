import os
import pandas as pd
import numpy as np
from features import add_features
from constants import DATA_DIR, TARGET_HORIZON, SYMBOLS, SYMBOLS

def compute_metrics(symbol):
    """
    Compute key metrics for a given symbol based on features.
    """
    parquet_path = os.path.join(DATA_DIR, f"{symbol}_1m.parquet")
    if not os.path.exists(parquet_path):
        print(f"Data file for {symbol} not found: {parquet_path}")
        return None

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows for {symbol}")

    # Add features (using TARGET_HORIZON from constants)
    df, feature_cols = add_features(df, TARGET_HORIZON)

    # Compute metrics
    # Volatility: average of vol_15 (15-period volatility)
    volatility = df['vol_15'].mean()

    # Liquidity: average volume
    liquidity = df['volume'].mean()

    # Mean reverting behavior: inverse of average absolute trend strength (lower trend = more reversion)
    # Also, average absolute distance from MA as oscillation measure
    trend_strength_avg = df['trend_strength'].abs().mean()
    reversion_score = 1 / (trend_strength_avg + 1e-6)  # Higher when less trending
    oscillation = df['dist_ma_15_z'].abs().mean()  # Average absolute z-score from 15-MA

    return {
        'symbol': symbol,
        'volatility': volatility,
        'liquidity': liquidity,
        'reversion_score': reversion_score,
        'oscillation': oscillation
    }

def rank_cryptos(symbols):
    """
    Compute metrics for each symbol and rank them based on suitability for mean reversion.
    """
    metrics_list = []
    for symbol in symbols:
        metrics = compute_metrics(symbol)
        if metrics:
            metrics_list.append(metrics)

    if not metrics_list:
        print("No valid data found for any symbols.")
        return

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Normalize metrics (min-max scaling)
    for col in ['volatility', 'liquidity', 'reversion_score', 'oscillation']:
        min_val = metrics_df[col].min()
        max_val = metrics_df[col].max()
        if max_val > min_val:
            metrics_df[f'{col}_norm'] = (metrics_df[col] - min_val) / (max_val - min_val)
        else:
            metrics_df[f'{col}_norm'] = 0.5  # If all same, neutral

    # Composite score: weighted sum favoring volatility, liquidity, and reversion/oscillation
    # Weights: volatility 0.25, liquidity 0.25, reversion_score 0.25, oscillation 0.25
    metrics_df['suitability_score'] = (
        0.25 * metrics_df['volatility_norm'] +
        0.25 * metrics_df['liquidity_norm'] +
        0.25 * metrics_df['reversion_score_norm'] +
        0.25 * metrics_df['oscillation_norm']
    )

    # Rank by suitability score descending
    ranked_df = metrics_df.sort_values('suitability_score', ascending=False).reset_index(drop=True)
    ranked_df['rank'] = ranked_df.index + 1

    # Print results
    print("\nCrypto Suitability Ranking for Mean Reversion Strategy:")
    print("=" * 60)
    for _, row in ranked_df.iterrows():
        print(f"Rank {int(row['rank']):2d}: {row['symbol']} (Score: {row['suitability_score']:.3f})")
        print(f"  Volatility: {row['volatility']:.3f}")
        print(f"  Liquidity: {row['liquidity']:.3f}")
        print(f"  Reversion Score: {row['reversion_score']:.3f}")
        print(f"  Oscillation: {row['oscillation']:.3f}")
        print()

    # Save to CSV
    output_path = "crypto_ranking.csv"
    ranked_df.to_csv(output_path, index=False)
    print(f"Full results saved to {output_path}")

if __name__ == "__main__":
    # List of cryptos to evaluate (can be modified or taken from args)
    symbols_to_rank = SYMBOLS  # Use all symbols from constants

    rank_cryptos(symbols_to_rank)