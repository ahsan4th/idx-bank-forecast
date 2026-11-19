"""Forecast evaluation: error metrics, Diebold-Mariano tests, intervals, coverage.

Conventions
-----------
actual, pred: (N, H) arrays of prices (Rupiah); row = origin, column = horizon h-1.
log residual: e = log(actual) - log(pred). An interval [pred*exp(q_lo), pred*exp(q_hi)]
covers the actual price exactly when q_lo <= e <= q_hi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


def point_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """MAE and RMSE (Rupiah) and MAPE (%) per horizon."""
    if actual.shape != pred.shape:
        raise ValueError(f"Shape mismatch {actual.shape} vs {pred.shape}")
    err = actual - pred
    return {
        "mae": np.mean(np.abs(err), axis=0),
        "rmse": np.sqrt(np.mean(err**2, axis=0)),
        "mape": np.mean(np.abs(err) / actual, axis=0) * 100.0,
    }


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    n: int
    horizon: int
    note: str


def dm_test(loss_reference: np.ndarray, loss_model: np.ndarray, horizon: int) -> DMResult:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold (1997) correction.

    d_t = L_reference,t - L_model,t, so a positive statistic means the model has the
    lower loss. The long-run variance sums autocovariances up to lag h-1 (h-step
    errors are MA(h-1) under optimal forecasting). Two-sided p-value from Student t
    with n-1 degrees of freedom. When the variance estimate is not positive the
    test falls back to h = 1, as R's forecast::dm.test does, and says so in `note`.
    """
    d = np.asarray(loss_reference, dtype=float) - np.asarray(loss_model, dtype=float)
    if d.ndim != 1:
        raise ValueError("Loss series must be 1-D")
    n = len(d)
    if n < 3:
        raise ValueError("Need at least 3 paired losses")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    centered = d - d.mean()
    gamma = [float(np.dot(centered[k:], centered[: n - k]) / n) for k in range(min(horizon, n))]
    h_used, note = horizon, ""
    variance = (gamma[0] + 2.0 * sum(gamma[1:])) / n
    if variance <= 0.0:
        if gamma[0] <= 0.0:
            return DMResult(math.nan, math.nan, n, horizon, "selisih loss konstan; uji tidak terdefinisi")
        variance, h_used = gamma[0] / n, 1
        note = "varians jangka panjang tidak positif; memakai h=1"
    dm = d.mean() / math.sqrt(variance)
    correction = math.sqrt((n + 1 - 2 * h_used + h_used * (h_used - 1) / n) / n)
    statistic = dm * correction
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
    return DMResult(float(statistic), p_value, n, horizon, note)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values; NaN entries are excluded and stay NaN."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m = len(valid)
    if m == 0:
        return out
    order = valid[np.argsort(p[valid], kind="mergesort")]
    adjusted = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adjusted)
    return out


def wilson_interval(successes: int, trials: int, level: float) -> tuple[float, float]:
    if trials == 0:
        return math.nan, math.nan
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    phat = successes / trials
    denom = 1.0 + z**2 / trials
    center = (phat + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2)) / denom
    return center - half, center + half


def directional_hits(
    actual: np.ndarray, pred: np.ndarray, origin_price: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Correct up/down calls per horizon. Unchanged prices (either side) are excluded."""
    actual_move = np.sign(actual - origin_price[:, None])
    pred_move = np.sign(pred - origin_price[:, None])
    valid = (actual_move != 0) & (pred_move != 0)
    hits = ((actual_move == pred_move) & valid).sum(axis=0)
    return hits.astype(int), valid.sum(axis=0).astype(int)


def _conformal_rank(n: int, level: float) -> int:
    alpha = 1.0 - level
    # The small epsilon keeps float noise (e.g. 54.000000000001) from adding a rank.
    return math.ceil((n + 1) * (1.0 - alpha / 2.0) - 1e-9)


def min_calibration_size(level: float) -> int:
    """Smallest n for which `conformal_offsets` gives a finite interval at `level`."""
    n = 1
    while _conformal_rank(n, level) > n:
        n += 1
    return n


def conformal_offsets(log_residuals: np.ndarray, level: float) -> tuple[float, float]:
    """Split-conformal bounds (q_lo, q_hi) on log(actual/forecast) for one horizon.

    Each tail gets alpha/2: q_hi is the k-th smallest residual with
    k = ceil((n+1)(1 - alpha/2)), and q_lo is the k-th largest. Under exchangeable
    residuals each one-sided statement holds with probability >= 1 - alpha/2, so
    the two-sided interval covers with probability >= 1 - alpha. Daily stock
    residuals are only approximately exchangeable (volatility clusters).
    """
    e = np.asarray(log_residuals, dtype=float)
    if e.ndim != 1 or not np.all(np.isfinite(e)):
        raise ValueError("Residuals must be a finite 1-D array")
    n = len(e)
    k = _conformal_rank(n, level)
    if k > n:
        raise ValueError(
            f"Interval {level:.0%} butuh minimal {min_calibration_size(level)} residual, "
            f"tersedia {n}. Tambah jumlah origin backtest."
        )
    e_sorted = np.sort(e)
    return float(e_sorted[n - k]), float(e_sorted[k - 1])


@dataclass(frozen=True)
class CoverageResult:
    level: float
    per_horizon: np.ndarray
    pooled: float
    n_calibration: int
    n_evaluation: int


def split_conformal_coverage(
    log_residuals: np.ndarray, level: float, calibration_fraction: float
) -> CoverageResult | None:
    """Out-of-sample coverage: calibrate on early origins, score on later origins.

    Returns None when the calibration block is too small for `level`.
    """
    n_total, horizon = log_residuals.shape
    n_cal = int(math.floor(n_total * calibration_fraction))
    if n_cal < min_calibration_size(level) or n_total - n_cal < 1:
        return None
    calibration, evaluation = log_residuals[:n_cal], log_residuals[n_cal:]
    covered = np.empty(evaluation.shape, dtype=bool)
    for h in range(horizon):
        q_lo, q_hi = conformal_offsets(calibration[:, h], level)
        covered[:, h] = (evaluation[:, h] >= q_lo) & (evaluation[:, h] <= q_hi)
    return CoverageResult(
        level=level,
        per_horizon=covered.mean(axis=0),
        pooled=float(covered.mean()),
        n_calibration=n_cal,
        n_evaluation=n_total - n_cal,
    )


def native_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Share of actual prices inside a model's own quantile band, per horizon."""
    return ((actual >= lower) & (actual <= upper)).mean(axis=0)
