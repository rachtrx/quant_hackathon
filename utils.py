import numpy as np
import pandas as pd

# =========================
# IC metrics
# =========================

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return np.nan
    return np.corrcoef(y_true[valid], y_pred[valid])[0, 1]


def rank_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()

    if len(df) < 2:
        return np.nan

    r1 = df["y_true"].rank()
    r2 = df["y_pred"].rank()

    if r1.nunique() <= 1 or r2.nunique() <= 1:
        return np.nan

    return r1.corr(r2)

# SPLIT
def time_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df