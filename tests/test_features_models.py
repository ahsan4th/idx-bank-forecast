from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecaster.features import build_gbm_features, log_return_targets, warmup_length
from forecaster.models import (
    Chronos2Runner,
    ForecastBatch,
    TimesFMRunner,
    TimestralRunner,
    ensemble_forecast,
    gbm_backtest,
    gbm_final,
    make_contexts,
    naive_forecast,
    parse_variant,
    variant_key,
)
from tests.conftest import FakeChronos2Pipeline, FakeTimesFM, FakeTimestralForecaster, synthetic_ohlcv


def _perturb_after(frame: pd.DataFrame, t: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    factor = np.exp(rng.normal(0, 0.05, len(out) - t - 1))
    for col in ("Open", "High", "Low", "Close"):
        out.iloc[t + 1 :, out.columns.get_loc(col)] *= factor
    out.iloc[t + 1 :, out.columns.get_loc("Volume")] *= 3.0
    return out


def test_features_are_causal(settings):
    frame = synthetic_ohlcv(300, seed=10)
    t = 200
    original = build_gbm_features(frame, settings.models.gbm, None)
    perturbed = build_gbm_features(_perturb_after(frame, t, 11), settings.models.gbm, None)
    pd.testing.assert_frame_equal(original.iloc[: t + 1], perturbed.iloc[: t + 1])
    # The check can fail: the first perturbed row must differ.
    assert not original.iloc[t + 1].equals(perturbed.iloc[t + 1])


def test_features_defined_after_warmup(settings):
    frame = synthetic_ohlcv(300, seed=12)
    feats = build_gbm_features(frame, settings.models.gbm, None)
    warm = warmup_length(settings.models.gbm)
    assert feats.iloc[warm:].notna().all().all()
    assert feats.iloc[: warm - 2].isna().any(axis=1).all()
    assert not np.isinf(feats.to_numpy()).any()


def test_targets_alignment():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    y = log_return_targets(close, 2)
    assert y.loc[0, 1] == pytest.approx(np.log(1.1))
    assert y.loc[0, 2] == pytest.approx(np.log(1.21))
    assert np.isnan(y.loc[3, 1]) and np.isnan(y.loc[2, 2])


def test_naive_forecast_equals_origin_price():
    close = np.arange(1.0, 21.0)
    batch = naive_forecast(close, np.array([5, 9]), 3)
    np.testing.assert_array_equal(batch.median, [[6, 6, 6], [10, 10, 10]])


def test_forecast_batch_validation():
    with pytest.raises(ValueError):
        ForecastBatch(np.array([[1.0, np.nan]]), None, None)
    with pytest.raises(ValueError):
        ForecastBatch(np.array([[1.0, -2.0]]), None, None)
    with pytest.raises(ValueError):
        ForecastBatch(np.ones((2, 3)), np.ones((2, 3)), None)


def test_gbm_backtest_has_no_lookahead(settings):
    frame = synthetic_ohlcv(700, seed=20)
    origins = np.arange(600, 690)
    mid = 640
    a = gbm_backtest(frame, origins, 5, settings.models.gbm, 7, None)
    b = gbm_backtest(_perturb_after(frame, mid, 21), origins, 5, settings.models.gbm, 7, None)
    upto = origins <= mid
    np.testing.assert_array_equal(a.median[upto], b.median[upto])
    assert not np.allclose(a.median[~upto], b.median[~upto])


def test_gbm_is_deterministic(settings):
    frame = synthetic_ohlcv(600, seed=22)
    origins = np.arange(560, 590)
    a = gbm_backtest(frame, origins, 5, settings.models.gbm, 7, None)
    b = gbm_backtest(frame, origins, 5, settings.models.gbm, 7, None)
    np.testing.assert_array_equal(a.median, b.median)


def test_gbm_final_uses_provisional_origin(settings):
    frame = synthetic_ohlcv(601, seed=23)
    final, provisional = frame.iloc[:-1], frame.iloc[[-1]]
    origin_frame = pd.concat([final, provisional])
    out_final = gbm_final(final, final, 5, settings.models.gbm, 7, None, None)
    out_prov = gbm_final(final, origin_frame, 5, settings.models.gbm, 7, None, None)
    assert out_final.median.shape == (1, 5)
    assert not np.allclose(out_final.median, out_prov.median)
    with pytest.raises(ValueError):
        gbm_final(final, _perturb_after(origin_frame, 100, 1), 5, settings.models.gbm, 7, None, None)


def test_gbm_too_little_data_fails_loudly(settings):
    frame = synthetic_ohlcv(200, seed=24)
    with pytest.raises(ValueError, match="baris latih"):
        gbm_backtest(frame, np.arange(150, 160), 5, settings.models.gbm, 7, None)


def test_make_contexts_end_at_origin():
    values = np.arange(100.0)
    ctx = make_contexts(values, np.array([10, 50]), 5)
    np.testing.assert_array_equal(ctx, [[6, 7, 8, 9, 10], [46, 47, 48, 49, 50]])
    cov = make_contexts(np.vstack([values, -values]), np.array([10]), 3)
    assert cov.shape == (1, 2, 3)
    np.testing.assert_array_equal(cov[0], [[8, 9, 10], [-8, -9, -10]])
    with pytest.raises(ValueError):
        make_contexts(values, np.array([3]), 5)


def test_chronos2_runner_univariate_and_covariates():
    pipe = FakeChronos2Pipeline()
    runner = Chronos2Runner(pipe, batch_size=16, q_low=0.1, q_high=0.9)
    contexts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    out = runner.predict(contexts, 4, None, ())
    np.testing.assert_allclose(out.median, [[3] * 4, [6] * 4])
    assert np.all(out.lower < out.median) and np.all(out.upper > out.median)
    assert pipe.calls[-1]["kwargs"]["cross_learning"] is False
    cov = np.zeros((2, 2, 3), dtype=np.float32)
    runner.predict(contexts, 4, cov, ("a", "b"))
    item = pipe.calls[-1]["inputs"][0]
    assert set(item["past_covariates"]) == {"a", "b"}


def test_timestral_and_timesfm_runners_pick_requested_levels():
    contexts = np.array([[1.0, 2.0, 10.0]], dtype=np.float32)
    ts = TimestralRunner(FakeTimestralForecaster(), True, 0.1, 0.9).predict(contexts, 3, None, ())
    np.testing.assert_allclose(ts.median, [[10, 10, 10]])
    np.testing.assert_allclose(ts.lower, [[10 * 0.96] * 3])
    np.testing.assert_allclose(ts.upper, [[10 * 1.04] * 3])
    tf = TimesFMRunner(FakeTimesFM(), 0.1, 0.9).predict(contexts, 3, None, ())
    np.testing.assert_allclose(tf.lower, [[10 * 0.96] * 3])
    np.testing.assert_allclose(tf.upper, [[10 * 1.04] * 3])
    with pytest.raises(ValueError):
        TimesFMRunner(FakeTimesFM(), 0.05, 0.95)


def test_ensemble_and_variant_keys():
    a = ForecastBatch(np.full((2, 3), 10.0), None, None)
    b = ForecastBatch(np.full((2, 3), 20.0), None, None)
    c = ForecastBatch(np.full((2, 3), 12.0), None, None)
    np.testing.assert_array_equal(ensemble_forecast([a, b, c]).median, np.full((2, 3), 12.0))
    with pytest.raises(ValueError):
        ensemble_forecast([a])
    assert parse_variant(variant_key("gbm", True)) == ("gbm", True)
    assert parse_variant("chronos2") == ("chronos2", False)
