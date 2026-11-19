"""Headless run of the Streamlit script with mocked prices (skipped without Streamlit).

Run on a machine with the requirements installed:  pytest tests/test_app_smoke.py
Prices are SYNTHETIC; foundation models are switched off so nothing is downloaded.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("exchange_calendars")
pytest.importorskip("yfinance")

from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import synthetic_ohlcv  # noqa: E402

WIB = "Asia/Jakarta"


def _fake_daily(ticker, period, data, market):
    yesterday = pd.Timestamp.now(tz=WIB).normalize().tz_localize(None) - pd.Timedelta(days=1)
    frame = synthetic_ohlcv(1300, seed=sum(map(ord, ticker)))  # stable across processes
    frame.index = pd.bdate_range(end=yesterday, periods=len(frame), name="date")
    return frame


def _fake_intraday(ticker, data, market):
    day = pd.Timestamp.now(tz=WIB).normalize() - pd.Timedelta(days=1)
    idx = pd.date_range(day + pd.Timedelta(hours=9), periods=30, freq="1min")
    close = 5000 + np.cumsum(np.ones(30))
    return pd.DataFrame(
        {"Open": close, "High": close + 5, "Low": close - 5, "Close": close, "Volume": 100.0},
        index=idx,
    )


def _only_core_models():
    return {"naive": True, "gbm": True, "chronos2": False, "chronos_bolt": False,
            "timestral": False, "timesfm": False}


def test_app_runs_end_to_end_with_mocked_data():
    with (
        patch("forecaster.data.fetch_daily_history", side_effect=_fake_daily),
        patch("forecaster.data.fetch_intraday", side_effect=_fake_intraday),
        patch("forecaster.models.model_availability", side_effect=_only_core_models),
    ):
        at = AppTest.from_file("../streamlit_app.py", default_timeout=300)
        at.run()
    assert not at.exception, at.exception
    assert len(at.get("echarts_chart")) >= 2
    assert any("Prakiraan model terbaik" in s.value for s in at.subheader)
