from __future__ import annotations

import numpy as np
import pytest

from forecaster.evaluation import (
    conformal_offsets,
    directional_hits,
    dm_test,
    holm_adjust,
    min_calibration_size,
    point_metrics,
    split_conformal_coverage,
    wilson_interval,
)


def _ma_errors(rng, n, h, size):
    """MA(h-1) errors with unit innovations: the dependence of overlapping h-step errors."""
    eps = rng.normal(size=(size, n + h - 1))
    kernel = np.ones(h)
    return np.stack([np.convolve(row, kernel, mode="valid") for row in eps])


def _rejection_rate(h_data, h_test, n=60, reps=1500, seed=0):
    rng = np.random.default_rng(seed)
    e1 = _ma_errors(rng, n, h_data, reps)
    e2 = _ma_errors(rng, n, h_data, reps)
    pvals = np.array([dm_test(np.abs(a), np.abs(b), h_test).p_value for a, b in zip(e1, e2)])
    return np.mean(pvals < 0.05)


def test_dm_size_close_to_nominal_for_h1():
    assert 0.03 <= _rejection_rate(1, 1) <= 0.075


def test_dm_hac_lags_control_size_for_overlapping_errors():
    # Ignoring the MA(4) dependence (testing as if h=1) must over-reject clearly;
    # the h=5 test must stay far closer to the nominal 5%.
    wrong = _rejection_rate(5, 1, seed=1)
    right = _rejection_rate(5, 5, seed=1)
    assert wrong > 0.12
    assert right < 0.10
    assert right < wrong


def test_dm_power_and_symmetry():
    rng = np.random.default_rng(3)
    e_bad = rng.normal(0, 2.0, 80)
    e_good = rng.normal(0, 1.0, 80)
    res = dm_test(np.abs(e_bad), np.abs(e_good), 1)
    assert res.statistic > 0 and res.p_value < 0.01
    flipped = dm_test(np.abs(e_good), np.abs(e_bad), 1)
    assert flipped.statistic == pytest.approx(-res.statistic)
    assert flipped.p_value == pytest.approx(res.p_value)


def test_dm_identical_losses_is_undefined():
    x = np.ones(20)
    res = dm_test(x, x, 2)
    assert np.isnan(res.p_value) and res.note


def test_holm_known_example():
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03, 0.005, np.nan]))
    np.testing.assert_allclose(adjusted[:4], [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(adjusted[4])


def test_wilson_interval_reference_value():
    lo, hi = wilson_interval(5, 10, 0.95)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)


def test_point_metrics_and_hits():
    actual = np.array([[110.0, 90.0], [100.0, 100.0]])
    pred = np.array([[105.0, 95.0], [102.0, 98.0]])
    m = point_metrics(actual, pred)
    np.testing.assert_allclose(m["mae"], [3.5, 3.5])
    hits, trials = directional_hits(actual, pred, np.array([100.0, 100.0]))
    # Row 2 has an unchanged price -> excluded; row 1: up/up hit, down/down hit.
    np.testing.assert_array_equal(trials, [1, 1])
    np.testing.assert_array_equal(hits, [1, 1])


def test_conformal_rank_and_minimum_sizes():
    assert min_calibration_size(0.80) == 9
    assert min_calibration_size(0.90) == 19
    assert min_calibration_size(0.95) == 39
    e = np.arange(60, dtype=float)  # sorted residuals 0..59
    q_lo, q_hi = conformal_offsets(e, 0.80)
    # k = ceil(61 * 0.9) = 55: q_hi is the 55th smallest (54), q_lo the 55th largest (5).
    assert (q_lo, q_hi) == (5.0, 54.0)
    with pytest.raises(ValueError, match="minimal 39"):
        conformal_offsets(np.arange(30, dtype=float), 0.95)


def test_conformal_coverage_on_exchangeable_residuals():
    rng = np.random.default_rng(5)
    coverages = []
    for _ in range(400):
        cal = rng.standard_t(df=4, size=60)
        q_lo, q_hi = conformal_offsets(cal, 0.80)
        test = rng.standard_t(df=4, size=500)
        coverages.append(np.mean((test >= q_lo) & (test <= q_hi)))
    # Expected coverage is 1 - 2 * 6/61 = 0.803 for n = 60.
    assert 0.785 <= np.mean(coverages) <= 0.825


def test_split_coverage_shapes_and_small_samples():
    rng = np.random.default_rng(6)
    res = rng.normal(size=(60, 5))
    cov = split_conformal_coverage(res, 0.80, 0.5)
    assert cov is not None and cov.per_horizon.shape == (5,)
    assert cov.n_calibration == 30 and cov.n_evaluation == 30
    assert split_conformal_coverage(res, 0.95, 0.5) is None  # 30 < 39
