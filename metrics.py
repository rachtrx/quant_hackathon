# =========================
# IC metrics
# =========================
import numpy as np
import pandas as pd


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return np.nan
    return np.corrcoef(y_true[valid], y_pred[valid])[0, 1]


def rank_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s1 = pd.Series(y_true)
    s2 = pd.Series(y_pred)
    return s1.rank().corr(s2.rank())