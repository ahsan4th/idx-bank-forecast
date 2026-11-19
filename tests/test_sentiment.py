from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecaster.sentiment import (
    SENTIMENT_COLUMNS,
    align_to_trading_days,
    covariate_matrix,
    daily_from_timelines,
    fetch_gdelt_daily,
)


def _timelines(days: pd.DatetimeIndex, tone: np.ndarray, counts: np.ndarray):
    stamps = pd.to_datetime([d.strftime("%Y%m%dT000000Z") for d in days], utc=True)
    tone_df = pd.DataFrame({"datetime": stamps, "Average Tone": tone})
    vol_df = pd.DataFrame({"datetime": stamps, "Article Count": counts, "All Articles": counts * 100})
    return tone_df, vol_df


def test_daily_from_timelines_weights_subdaily_bins():
    stamps = pd.to_datetime(
        ["20261101T000000Z", "20261101T120000Z", "20261102T000000Z"], utc=True
    )
    tone = pd.DataFrame({"datetime": stamps, "Average Tone": [1.0, -2.0, 0.5]})
    vol = pd.DataFrame({"datetime": stamps, "Article Count": [3.0, 1.0, 0.0], "All Articles": [1, 1, 1]})
    daily = daily_from_timelines(tone, vol)
    assert daily.loc["2026-11-01", "articles"] == 4
    assert daily.loc["2026-11-01", "tone"] == pytest.approx((3 * 1.0 + 1 * -2.0) / 4)
    assert np.isnan(daily.loc["2026-11-02", "tone"])  # no articles -> undefined tone


def test_daily_from_timelines_rejects_coarse_bins():
    days = pd.date_range("2026-01-01", periods=4, freq="7D")
    tone, vol = _timelines(days, np.zeros(4), np.ones(4))
    with pytest.raises(ValueError, match="daily or finer"):
        daily_from_timelines(tone, vol)


class FakeGdelt:
    def __init__(self):
        self.requests = []

    def timeline_search(self, mode, filters):
        self.requests.append((mode, filters))
        start = pd.Timestamp(filters["start_date"])
        end = pd.Timestamp(filters["end_date"]) - pd.Timedelta(days=1)
        days = pd.date_range(start, end, freq="D")
        tone, vol = _timelines(days, np.full(len(days), 2.0), np.full(len(days), 5.0))
        return tone if mode == "timelinetone" else vol


def test_fetch_gdelt_daily_chunks_and_fills():
    client = FakeGdelt()
    out = fetch_gdelt_daily(
        ("Bank X",), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-25"), 10, 0.0,
        client, lambda **kw: kw, None,
    )
    assert len(out) == 25 and (out["articles"] == 5).all()
    assert len(client.requests) == 6  # 3 chunks x (tone + volume)
    first_filters = client.requests[0][1]
    assert first_filters["keyword"] == ["Bank X"]
    assert first_filters["end_date"] == "2026-01-11"  # exclusive end of chunk 1


def test_alignment_has_no_lookahead_and_weekend_rolls_into_monday():
    utc_days = pd.date_range("2026-10-01", "2026-11-30", freq="D")
    rng = np.random.default_rng(0)
    daily = pd.DataFrame(
        {"tone": rng.normal(size=len(utc_days)), "articles": rng.integers(1, 9, len(utc_days)).astype(float)},
        index=utc_days,
    )
    trading = pd.bdate_range("2026-10-05", "2026-11-27")
    base = align_to_trading_days(daily, trading, 1, 5.0, pd.Timestamp("2026-10-01"))
    assert list(base.columns) == list(SENTIMENT_COLUMNS)

    # Changing the UTC bin of day d itself must not move the features of day d.
    d = pd.Timestamp("2026-11-18")
    changed = daily.copy()
    changed.loc[d, "tone"] += 50.0
    after = align_to_trading_days(changed, trading, 1, 5.0, pd.Timestamp("2026-10-01"))
    pd.testing.assert_frame_equal(base.loc[:d], after.loc[:d])
    assert after.loc[pd.Timestamp("2026-11-19"), "sent_tone"] != base.loc[pd.Timestamp("2026-11-19"), "sent_tone"]

    # Monday 2026-11-23 aggregates the UTC days Fri 20, Sat 21, Sun 22.
    window = daily.loc["2026-11-20":"2026-11-22"]
    expected = (window["tone"] * window["articles"]).sum() / window["articles"].sum()
    assert base.loc[pd.Timestamp("2026-11-23"), "sent_tone"] == pytest.approx(expected)


def test_alignment_marks_missing_coverage_as_nan_not_zero():
    utc_days = pd.date_range("2026-11-01", "2026-11-10", freq="D")
    daily = pd.DataFrame({"tone": 1.0, "articles": 2.0}, index=utc_days)
    trading = pd.bdate_range("2026-10-26", "2026-11-13")
    out = align_to_trading_days(daily, trading, 1, 5.0, pd.Timestamp("2026-11-01"))
    assert out.loc["2026-10-26":"2026-11-02", "sent_log_count"].isna().all()  # before coverage
    assert out.loc[pd.Timestamp("2026-11-03"), "sent_log_count"] == pytest.approx(np.log1p(2.0))
    # Wed Nov 11 reads the Nov 10 bin, the last one fetched; Thu Nov 12 would need Nov 11.
    assert out.loc[pd.Timestamp("2026-11-11"), "sent_log_count"] == pytest.approx(np.log1p(2.0))
    assert np.isnan(out.loc[pd.Timestamp("2026-11-12"), "sent_log_count"])
    cov = covariate_matrix(out)
    assert cov.shape == (2, len(trading)) and np.isfinite(cov).all()
