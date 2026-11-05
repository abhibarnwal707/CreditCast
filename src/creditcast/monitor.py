"""Population Stability Index and rolling-AUC monitoring helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PSI_WATCH, PSI_SHIFT = 0.10, 0.25


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI with bin edges at the quantiles of the EXPECTED (development) sample. Identical samples give 0."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:                                   # (near) constant variable
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_flag(v: float) -> str:
    return "shifted" if v > PSI_SHIFT else "watch" if v > PSI_WATCH else "stable"


def rolling_quarterly_auc(df: pd.DataFrame, score: str, min_defaults: int = 3) -> pd.DataFrame:
    rows = []
    for q, g in df.groupby("qidx"):
        d = int(g["default"].sum())
        auc = roc_auc_score(g["default"], g[score]) if d >= min_defaults and d < len(g) else np.nan
        rows.append({"qidx": int(q), "n": len(g), "defaults": d, "actual_rate": g["default"].mean(), "pred_rate": g[score].mean(), "auc": auc})
    return pd.DataFrame(rows)
