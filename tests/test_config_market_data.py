from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecaster.config import load_settings
from forecaster.data import (
    build_live_quote,
    normalize_daily_frame,
    provisional_bar_from_intraday,
    split_provisional,
    validate_daily,
)
from forecaster.market import is_daily_bar_final, market_phase, next_sessions
from tests.conftest import ROOT, WeekdayCalendar, synthetic_ohlcv

WIB = "Asia/Jakarta"


def test_missing_key_fails_loudly(tmp_path: Path):
    text = (ROOT / "settings.toml").read_text()
    broken = tmp_path / "settings.toml"
    broken.write_text(text.replace('history_period = "5y"\n', ""))
    with pytest.raises(KeyError):
        load_settings(broken)


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2026-11-16 08:30", "belum_buka"),  # Monday
        ("2026-11-16 08:50", "pra_pembukaan"),
        ("2026-11-16 10:00", "sesi_1"),
        ("2026-11-16 12:30", "istirahat"),
        ("2026-11-16 14:00", "sesi_2"),
        ("2026-11-16 15:55", "pra_penutupan"),
        ("2026-11-16 16:05", "tutup"),
        ("2026-11-20 11:45", "istirahat"),  # Friday: session I ends 11:30
        ("2026-11-20 13:45", "istirahat"),  # Friday: session II starts 14:00
        ("2026-11-20 14:10", "sesi_2"),
    ],
)
def test_market_phase(settings, stamp, expected):
    now = pd.Timestamp(stamp, tz=WIB)
    assert market_phase(now, True, settings.market) == expected


def test_market_phase_holiday_and_timezone(settings):
    assert market_phase(pd.Timestamp("2026-11-16 10:00", tz=WIB), False, settings.market) == "libur"
    # 03:00 UTC is 10:00 WIB.
    assert market_phase(pd.Timestamp("2026-11-16 03:00", tz="UTC"), True, settings.market) == "sesi_1"
    with pytest.raises(ValueError):
        market_phase(pd.Timestamp("2026-11-16 10:00"), True, settings.market)


def test_daily_bar_finality(settings):
    day = pd.Timestamp("2026-11-19")
    assert not is_daily_bar_final(day, pd.Timestamp("2026-11-19 15:00", tz=WIB), settings.market)
    assert is_daily_bar_final(day, pd.Timestamp("2026-11-19 16:30", tz=WIB), settings.market)
    assert is_daily_bar_final(day, pd.Timestamp("2026-11-20 08:00", tz=WIB), settings.market)
    with pytest.raises(ValueError):
        is_daily_bar_final(day, pd.Timestamp("2026-11-18 20:00", tz=WIB), settings.market)


def test_next_sessions_skip_weekend():
    cal = WeekdayCalendar()
    dates, is_session = next_sessions(cal, pd.Timestamp("2026-11-20"), 5)  # Friday
    assert is_session
    assert [d.strftime("%a %d") for d in dates] == ["Mon 23", "Tue 24", "Wed 25", "Thu 26", "Fri 27"]
    # A data date the calendar does not know as a session is flagged, not hidden.
    dates, is_session = next_sessions(cal, pd.Timestamp("2026-11-21"), 1)  # Saturday
    assert not is_session and dates[0] == pd.Timestamp("2026-11-23")


def test_normalize_daily_frame_maps_to_wib_dates():
    idx = pd.DatetimeIndex(["2026-11-17 00:00", "2026-11-18 00:00"], tz=WIB)
    raw = pd.DataFrame(
        {"Open": [1, 2], "High": [2, 3], "Low": [1, 1], "Close": [1.5, 2.5], "Volume": [10, 20],
         "Extra": [0, 0]},
        index=idx,
    )
    out = normalize_daily_frame(raw, WIB)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.tz is None and list(out.index.day) == [17, 18]
    with pytest.raises(ValueError):
        normalize_daily_frame(raw.tz_localize(None), WIB)


def test_validate_daily_reports_and_cleans(settings):
    frame = synthetic_ohlcv(30, seed=1)
    dup = frame.iloc[[5]]
    frame = pd.concat([frame, dup]).sort_index(kind="mergesort")
    frame.iloc[10, frame.columns.get_loc("Close")] = np.nan
    frame.iloc[20, frame.columns.get_loc("Close")] *= 2.0  # +100% jump
    clean, report = validate_daily(frame, "TEST.JK", settings.data)
    assert report.n_raw == 31 and report.n_clean == 29
    assert len(report.duplicate_dates) == 1 and len(report.dropped_missing_close) == 1
    assert any(abs(r) > 0.9 for _, r in report.large_moves)
    assert clean.index.is_unique and clean["Close"].notna().all()
    assert report.issues()


def test_validate_daily_rejects_nonpositive(settings):
    frame = synthetic_ohlcv(10, seed=2)
    frame.iloc[3, frame.columns.get_loc("Close")] = 0.0
    with pytest.raises(ValueError):
        validate_daily(frame, "TEST.JK", settings.data)


def test_split_provisional(settings):
    frame = synthetic_ohlcv(10, seed=3, start="2026-11-06")  # last bar Thu 2026-11-19
    assert frame.index[-1] == pd.Timestamp("2026-11-19")
    final, prov = split_provisional(frame, pd.Timestamp("2026-11-19 11:00", tz=WIB), settings.market)
    assert len(final) == 9 and prov is not None and prov.index[0] == frame.index[-1]
    final, prov = split_provisional(frame, pd.Timestamp("2026-11-19 17:00", tz=WIB), settings.market)
    assert len(final) == 10 and prov is None


def _intraday(day: str, closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:00", periods=len(closes), freq="1min", tz=WIB)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"Open": c - 5, "High": c + 10, "Low": c - 10, "Close": c, "Volume": np.full(len(c), 100.0)},
        index=idx,
    )


def test_provisional_bar_and_live_quote():
    intraday = _intraday("2026-11-19", [9000, 9025, 9050])
    bar = provisional_bar_from_intraday(intraday, pd.Timestamp("2026-11-19"))
    assert bar is not None
    row = bar.iloc[0]
    assert row["Open"] == 8995 and row["High"] == 9060 and row["Low"] == 8990
    assert row["Close"] == 9050 and row["Volume"] == 300
    assert provisional_bar_from_intraday(intraday, pd.Timestamp("2026-11-20")) is None

    daily = synthetic_ohlcv(5, seed=4, start="2026-11-13")  # ends Wed 2026-11-19
    quote = build_live_quote("TEST.JK", intraday, daily)
    prev = daily.loc[daily.index < pd.Timestamp("2026-11-19"), "Close"].iloc[-1]
    assert quote.previous_close == pytest.approx(prev)
    assert quote.change_pct == pytest.approx((9050 / prev - 1) * 100)
    assert len(quote.intraday_close) == 3
