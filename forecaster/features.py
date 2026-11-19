"""Causal features and log-return targets for the gradient-boosting model.

Every feature in row t is computed from bars 0..t only (rolling windows end at t,
shifts look backwards). tests/test_features.py checks this by perturbing the bars
after t and asserting that row t does not change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.config import GBMSettings


def warmup_length(g: GBMSettings) -> int:
    """First row index at which every price feature can be defined."""
    windows = [
        g.return_lags,
        *g.momentum_windows,
        *g.volatility_windows,
        *g.ma_windows,
        g.rsi_window + 1,
        g.volume_window,
        g.range_window,
    ]
    return max(windows) + 1


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """RSI with simple moving averages of gains and losses (Cutler's variant).

    A window without losses gives RSI = 100; a window without any price change has
    no defined RSI and stays NaN, which HistGradientBoosting treats as missing.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window, min_periods=window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
    return 100.0 - 100.0 / (1.0 + rs)


def build_gbm_features(
    ohlcv: pd.DataFrame, g: GBMSettings, sentiment: pd.DataFrame | None
) -> pd.DataFrame:
    """Feature matrix aligned to `ohlcv.index` (rows before the warm-up contain NaN)."""
    close = ohlcv["Close"]
    log_close = np.log(close)
    ret = log_close.diff()  # daily log return r_t = log(P_t / P_{t-1})

    cols: dict[str, pd.Series] = {}
    for k in range(g.return_lags):
        cols[f"ret_lag{k}"] = ret.shift(k)
    for w in g.momentum_windows:
        cols[f"mom_{w}"] = log_close - log_close.shift(w)
    for w in g.volatility_windows:
        cols[f"vol_{w}"] = ret.rolling(w, min_periods=w).std(ddof=1)
    for w in g.ma_windows:
        cols[f"ma_gap_{w}"] = log_close - np.log(close.rolling(w, min_periods=w).mean())
    cols[f"rsi_{g.rsi_window}"] = _rsi(close, g.rsi_window)

    hl_range = np.log(ohlcv["High"] / ohlcv["Low"])
    cols["hl_range"] = hl_range
    cols[f"hl_range_mean_{g.range_window}"] = hl_range.rolling(
        g.range_window, min_periods=g.range_window
    ).mean()

    log_volume = np.log1p(ohlcv["Volume"])
    vol_mean = log_volume.rolling(g.volume_window, min_periods=g.volume_window).mean()
    vol_std = log_volume.rolling(g.volume_window, min_periods=g.volume_window).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cols[f"volume_z_{g.volume_window}"] = (log_volume - vol_mean) / vol_std
    cols["day_of_week"] = pd.Series(
        ohlcv.index.dayofweek.astype("float64"), index=ohlcv.index
    )

    features = pd.DataFrame(cols, index=ohlcv.index)
    if sentiment is not None:
        if not sentiment.index.equals(ohlcv.index):
            raise ValueError("Sentiment features must share the price index")
        overlap = set(features.columns) & set(sentiment.columns)
        if overlap:
            raise ValueError(f"Sentiment columns clash with price features: {overlap}")
        features = features.join(sentiment)
    # A zero-variance volume window divides by zero. Modelling decision: an undefined
    # z-score is a missing value (NaN), never +/-inf.
    return features.replace([np.inf, -np.inf], np.nan)


def log_return_targets(close: pd.Series, horizon: int) -> pd.DataFrame:
    """Column h holds log(P_{t+h} / P_t); NaN where t + h is past the data."""
    log_close = np.log(close)
    return pd.DataFrame(
        {h: log_close.shift(-h) - log_close for h in range(1, horizon + 1)},
        index=close.index,
    )
