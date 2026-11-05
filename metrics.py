"""Goodness-of-fit metrics used across the diagnostic/manuscript figures."""
import numpy as np


def nse(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 2:
        return float("nan")
    o, s = obs[m].astype(np.float64), sim[m].astype(np.float64)
    denom = np.sum((o - o.mean()) ** 2)
    return float("nan") if denom <= 0 else float(1.0 - np.sum((o - s) ** 2) / denom)


def kge(obs, sim):
    """Kling-Gupta Efficiency: 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)."""
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 2:
        return float("nan")
    o, s = obs[m].astype(np.float64), sim[m].astype(np.float64)
    if o.std() == 0 or s.std() == 0 or o.mean() == 0:
        return float("nan")
    r = np.corrcoef(o, s)[0, 1]
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def r2(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 2:
        return float("nan")
    o, s = obs[m].astype(np.float64), sim[m].astype(np.float64)
    if o.std() == 0 or s.std() == 0:
        return float("nan")
    return float(np.corrcoef(o, s)[0, 1] ** 2)
