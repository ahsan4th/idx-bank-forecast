"""IDX market clock: trading phase, daily-bar finality, and future session dates.

All times are wall-clock times in the exchange time zone (WIB, Asia/Jakarta).
Session boundaries come from settings.toml, not from this module.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from forecaster.config import MarketSettings

PHASE_LABELS_ID: dict[str, str] = {
    "libur": "Libur bursa",
    "belum_buka": "Belum dibuka",
    "pra_pembukaan": "Pra-pembukaan",
    "sesi_1": "Sesi I berjalan",
    "istirahat": "Istirahat siang",
    "sesi_2": "Sesi II berjalan",
    "pra_penutupan": "Pra-penutupan",
    "tutup": "Perdagangan selesai",
}

# Phases during which prices move; used to decide whether a quote is "live".
TRADING_PHASES = frozenset({"pra_pembukaan", "sesi_1", "sesi_2", "pra_penutupan"})


class TradingCalendar(Protocol):
    """The subset of exchange_calendars.ExchangeCalendar this app relies on."""

    def is_session(self, date: pd.Timestamp) -> bool: ...

    def date_to_session(self, date: pd.Timestamp, direction: str) -> pd.Timestamp: ...

    def next_session(self, session: pd.Timestamp) -> pd.Timestamp: ...


def load_exchange_calendar(name: str) -> TradingCalendar:
    import exchange_calendars as xcals

    return xcals.get_calendar(name)


def to_local(now: pd.Timestamp, market: MarketSettings) -> pd.Timestamp:
    if now.tzinfo is None:
        raise ValueError("`now` must be timezone-aware")
    return now.tz_convert(market.timezone)


def market_phase(now: pd.Timestamp, is_session_day: bool, market: MarketSettings) -> str:
    """Phase key of the IDX regular market at `now` (see PHASE_LABELS_ID)."""
    local = to_local(now, market)
    if not is_session_day:
        return "libur"
    t = local.time()
    session_1, session_2 = (
        market.sessions_fri if local.weekday() == 4 else market.sessions_mon_thu
    )
    if t < market.pre_opening[0]:
        return "belum_buka"
    if t < market.pre_opening[1]:
        return "pra_pembukaan"
    if session_1[0] <= t < session_1[1]:
        return "sesi_1"
    if session_1[1] <= t < session_2[0]:
        return "istirahat"
    if session_2[0] <= t < session_2[1]:
        return "sesi_2"
    if market.pre_closing[0] <= t < market.pre_closing[1]:
        return "pra_penutupan"
    if t >= market.pre_closing[1]:
        return "tutup"
    # Only reachable if settings leave a gap between session II and pre-closing,
    # which config.load_settings rejects.
    raise AssertionError(f"Unclassified market time {t}")


def is_daily_bar_final(
    bar_date: pd.Timestamp, now: pd.Timestamp, market: MarketSettings
) -> bool:
    """True when the daily bar for `bar_date` can no longer change."""
    local = to_local(now, market)
    today = pd.Timestamp(local.date())
    bar_day = pd.Timestamp(bar_date.date())
    if bar_day > today:
        raise ValueError(f"Daily bar dated {bar_day.date()} is in the future of {today.date()}")
    if bar_day < today:
        return True
    return local.time() >= market.daily_bar_final_time


def next_sessions(
    calendar: TradingCalendar, origin_date: pd.Timestamp, count: int
) -> tuple[list[pd.Timestamp], bool]:
    """The `count` trading sessions strictly after `origin_date`.

    Returns the dates and whether `origin_date` is itself a session in the
    calendar. A False flag means the price data and the holiday calendar disagree,
    which the app reports instead of hiding.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    day = pd.Timestamp(origin_date.date())
    origin_is_session = bool(calendar.is_session(day))
    session = calendar.date_to_session(day, direction="previous")
    sessions: list[pd.Timestamp] = []
    for _ in range(count):
        session = calendar.next_session(session)
        stamp = pd.Timestamp(session)
        sessions.append(stamp.tz_localize(None) if stamp.tzinfo is not None else stamp)
    if not sessions[0] > day:
        raise AssertionError(f"First future session {sessions[0]} is not after {day}")
    return sessions, origin_is_session
