"""Shared fixtures.

All price data in the test suite is SYNTHETIC (simulated log returns). It exists
only to check that the code computes what it claims; it is never a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecaster.config import load_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings():
    return load_settings(ROOT / "settings.toml")


def synthetic_ohlcv(
    n: int, seed: int, ar: float = 0.0, vol: float = 0.015, start: str = "2020-01-06"
) -> pd.DataFrame:
    """SYNTHETIC daily bars: r_t = ar * r_{t-1} + N(0, vol^2), weekdays only."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, vol, n)
    ret = np.empty(n)
    ret[0] = eps[0]
    for t in range(1, n):
        ret[t] = ar * ret[t - 1] + eps[t]
    close = 5000.0 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0.0, vol / 2, n)) + 1e-4
    high = close * np.exp(spread)
    low = close * np.exp(-spread)
    open_ = np.clip(close * np.exp(rng.normal(0.0, vol / 4, n)), low, high)
    volume = rng.lognormal(17.0, 0.3, n)
    index = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index
    )


class WeekdayCalendar:
    """Minimal stand-in for exchange_calendars: every weekday is a session."""

    def is_session(self, date: pd.Timestamp) -> bool:
        return pd.Timestamp(date).dayofweek < 5

    def date_to_session(self, date: pd.Timestamp, direction: str) -> pd.Timestamp:
        day = pd.Timestamp(date)
        step = {"previous": -1, "next": 1}[direction]
        while not self.is_session(day):
            day += pd.Timedelta(days=step)
        return day

    def next_session(self, session: pd.Timestamp) -> pd.Timestamp:
        if not self.is_session(session):
            raise ValueError("not a session")
        day = pd.Timestamp(session) + pd.Timedelta(days=1)
        while not self.is_session(day):
            day += pd.Timedelta(days=1)
        return day


class EchoForecastObject:
    def __init__(self, quantiles: np.ndarray, levels: tuple[float, ...]):
        self.quantiles = quantiles
        self.quantile_levels = levels


class FakeChronos2Pipeline:
    """Mimics Chronos2Pipeline.predict_quantiles: median = last context value."""

    def __init__(self):
        self.calls: list[dict] = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        self.calls.append({"inputs": inputs, "kwargs": kwargs})
        out = []
        for item in inputs:
            target = item["target"] if isinstance(item, dict) else item[0]
            last = float(target[-1])
            q = np.empty((1, prediction_length, len(quantile_levels)))
            for j, level in enumerate(quantile_levels):
                q[0, :, j] = last * (1.0 + (level - 0.5) * 0.1)
            out.append(q)
        means = [q[..., 1] for q in out]
        return out, means


class FakeTimestralForecaster:
    LEVELS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)

    def __init__(self):
        self.calls: list[dict] = []

    def forecast(self, context, prediction_length, past_covariates=None, symmetric_averaging=True):
        self.calls.append({"context": context, "past_covariates": past_covariates})
        last = np.asarray(context)[:, -1]
        levels = np.asarray(self.LEVELS)
        q = last[:, None, None, None] * (1.0 + (levels[None, None, None, :] - 0.5) * 0.1)
        q = np.broadcast_to(q, (len(last), 1, prediction_length, len(levels))).copy()
        return EchoForecastObject(q, self.LEVELS)


class FakeTimesFM:
    def forecast(self, horizon, inputs):
        last = np.array([float(x[-1]) for x in inputs])
        deciles = np.arange(1, 10) / 10.0
        quant = np.empty((len(inputs), horizon, 10))
        quant[..., 0] = last[:, None]
        quant[..., 1:] = last[:, None, None] * (1.0 + (deciles - 0.5) * 0.1)
        return last[:, None].repeat(horizon, axis=1), quant
