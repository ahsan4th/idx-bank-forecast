from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from forecaster.charts import fan_chart_option, model_color, skill_chart_option, tone_chart_option
from forecaster.models import Chronos2Runner, TimestralRunner
from forecaster.pipeline import backtest_origin_indices, run_backtest, run_final_forecast
from forecaster.sentiment import align_to_trading_days
from tests.conftest import (
    FakeChronos2Pipeline,
    FakeTimestralForecaster,
    WeekdayCalendar,
    synthetic_ohlcv,
)


def _loader(key):
    if key == "chronos2":
        return Chronos2Runner(FakeChronos2Pipeline(), 32, 0.1, 0.9)
    if key == "timestral":
        return TimestralRunner(FakeTimestralForecaster(), True, 0.1, 0.9)
    raise RuntimeError(f"download failed for {key}")  # simulated unavailable model


def test_origin_indices_balanced_panel():
    origins = backtest_origin_indices(100, 10, 5)
    assert origins[-1] + 5 == 99 and len(origins) == 10
    with pytest.raises(ValueError):
        backtest_origin_indices(10, 10, 5)


@pytest.fixture(scope="module")
def backtest_result(settings):
    frame = synthetic_ohlcv(900, seed=30)
    keys = ("naive", "gbm", "chronos2", "timestral", "timesfm")
    return frame, run_backtest("TEST.JK", frame, keys, settings, 60, _loader, None, None)


def test_backtest_structure(backtest_result, settings):
    frame, bt = backtest_result
    assert set(bt.forecasts) == {"naive", "gbm", "chronos2", "timestral", "ensemble"}
    assert "timesfm" in bt.failures and "download failed" in bt.failures["timesfm"]
    assert bt.actual.shape == (60, 5)
    naive_rows = bt.metrics[bt.metrics["model"] == "naive"]
    np.testing.assert_allclose(naive_rows["rel_mae"], 1.0)
    # The fake foundation models echo the last price, so they equal the naive forecast.
    np.testing.assert_allclose(bt.forecasts["chronos2"].median, bt.forecasts["naive"].median)
    assert bt.best_model in bt.forecasts
    assert bt.summary["mean_mae"].is_monotonic_increasing
    assert bt.metrics["dm_p_holm"].dropna().between(0, 1).all()
    assert set(bt.ensemble_members) == {"gbm", "chronos2", "timestral"}


def test_gbm_learns_a_planted_signal(settings):
    # SYNTHETIC check that the ML pipeline can find structure when it exists:
    # strongly autocorrelated returns (phi = 0.6) are predictable one day ahead.
    frame = synthetic_ohlcv(900, seed=31, ar=0.6)
    bt = run_backtest("AR.JK", frame, ("naive", "gbm"), settings, 120, _loader, None, None)
    row = bt.metrics[(bt.metrics["model"] == "gbm") & (bt.metrics["h"] == 1)].iloc[0]
    assert row["rel_mae"] < 0.9
    assert row["dm_p"] < 0.05


def test_gbm_does_not_invent_skill_on_a_random_walk(settings):
    frame = synthetic_ohlcv(900, seed=32, ar=0.0)
    bt = run_backtest("RW.JK", frame, ("naive", "gbm"), settings, 120, _loader, None, None)
    gbm = bt.metrics[bt.metrics["model"] == "gbm"]
    assert gbm["rel_mae"].mean() > 0.95


def test_final_forecast_dates_and_intervals(backtest_result, settings):
    frame, bt = backtest_result
    out = run_final_forecast(
        "TEST.JK", frame, frame, bt, settings, 0.8, WeekdayCalendar(), _loader, None, None
    )
    table = out.table
    assert set(table["model"]) == set(bt.forecasts)
    assert (table["lower"] <= table["median"]).all() and (table["median"] <= table["upper"]).all()
    dates = sorted(set(table["date"]))
    assert len(dates) == 5 and dates[0] > frame.index[-1]
    assert all(d.dayofweek < 5 for d in dates)
    assert not out.provisional
    naive = table[table["model"] == "naive"]
    np.testing.assert_allclose(naive["median"], frame["Close"].iloc[-1])


def test_sentiment_variants_and_ablation(settings):
    frame = synthetic_ohlcv(900, seed=33)
    utc_days = pd.date_range(frame.index[0] - pd.Timedelta(days=5), frame.index[-1], freq="D")
    rng = np.random.default_rng(1)
    daily = pd.DataFrame(
        {"tone": rng.normal(size=len(utc_days)), "articles": rng.integers(0, 6, len(utc_days)).astype(float)},
        index=utc_days,
    )
    feats = align_to_trading_days(daily, frame.index, 1, 5.0, utc_days[0])
    keys = ("naive", "gbm", "gbm+sent", "chronos2", "chronos2+sent")
    bt = run_backtest("S.JK", frame, keys, settings, 60, _loader, feats, None)
    assert {"gbm+sent", "chronos2+sent"} <= set(bt.forecasts)
    assert set(bt.ensemble_members) == {"gbm", "chronos2"}
    assert set(bt.ablation["model"]) == {"gbm", "chronos2"}
    assert len(bt.ablation) == 10
    with pytest.raises(ValueError):
        run_backtest("S.JK", frame, ("naive", "gbm+sent"), settings, 60, _loader, None, None)


def test_fan_chart_option_is_consistent(backtest_result, settings):
    frame, bt = backtest_result
    out = run_final_forecast(
        "TEST.JK", frame, frame, bt, settings, 0.8, WeekdayCalendar(), _loader, None, None
    )
    focus = out.table[out.table["model"] == "gbm"]
    compare = {
        k: (g["label"].iloc[0], g) for k, g in out.table.groupby("model") if k not in ("gbm",)
    }
    history = frame["Close"].iloc[-120:]
    for mode in ("light", "dark"):
        option = fan_chart_option(
            history, "gbm", "Gradient boosting", focus, compare, 0.8, mode, settings.ui.watermark
        )
        assert [g["style"]["text"] for g in option["graphic"]] == [settings.ui.watermark]
        data = option["dataset"]["source"]
        for s in option["series"]:
            for column in np.atleast_1d(s["encode"]["y"]):
                assert column in data.columns
        names = {s["name"] for s in option["series"]}
        assert set(option["legend"]["data"]) <= names
        lower_plus_band = data["lower"] + data["band"]
        mask = data["upper"].notna()
        np.testing.assert_allclose(lower_plus_band[mask], data["upper"][mask])
        # Option must serialise once the dataframe is converted (Streamlit sends Arrow).
        json.dumps({k: v for k, v in option.items() if k != "dataset"})
    assert model_color("gbm", "light") != model_color("gbm", "dark")
    assert model_color("gbm+sent", "light") == model_color("gbm", "light")


def test_skill_chart_option():
    rel = pd.DataFrame([[1.0, 1.0], [0.98, 1.02]], index=["naive", "gbm"], columns=[1, 2])
    option = skill_chart_option(rel, {"naive": "Random walk", "gbm": "GB"}, "light", "Muhammad Ahsan · ITS")
    assert list(option["dataset"]["source"].columns) == ["horizon", "Random walk", "GB"]
    assert option["series"][0]["markLine"]["data"] == [{"yAxis": 1}]
    watermark = option["graphic"][0]
    assert watermark["style"]["text"] == "Muhammad Ahsan · ITS"
    assert watermark["silent"] is True  # never intercepts hover or clicks
    assert watermark["z"] < 2  # ECharts series sit at z = 2, so data draws over the text
    # An empty watermark setting turns the chart watermark off.
    assert skill_chart_option(rel, {"naive": "Random walk", "gbm": "GB"}, "dark", "")["graphic"] == []


def test_tone_chart_option_carries_watermark():
    index = pd.bdate_range("2026-01-05", periods=10)
    features = pd.DataFrame({"sent_tone": np.linspace(-1, 1, 10), "sent_tone_ewm": np.zeros(10)}, index=index)
    option = tone_chart_option(features, "dark", "Muhammad Ahsan · ITS")
    assert [g["style"]["text"] for g in option["graphic"]] == ["Muhammad Ahsan · ITS"]
    assert tone_chart_option(features, "light", "")["graphic"] == []
