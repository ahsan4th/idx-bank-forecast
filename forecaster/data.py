"""Yahoo Finance prices (through yfinance), data-quality reports, and live quotes.

Raw frames are never modified in place: every step returns a new frame, and each
cleaning action is recorded in a DataQualityReport that the app displays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecaster.config import DataSettings, MarketSettings
from forecaster.market import is_daily_bar_final

OHLCV: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Price frame is missing columns {missing}; got {list(frame.columns)}")


def normalize_daily_frame(raw: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Daily bars -> float OHLCV on a tz-naive date index named 'date'."""
    _require_columns(raw, OHLCV)
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise TypeError("Daily price index must be a DatetimeIndex")
    if raw.index.tz is None:
        raise ValueError("Daily price index has no time zone; cannot map bars to WIB dates")
    dates = raw.index.tz_convert(timezone).normalize().tz_localize(None)
    out = raw.loc[:, list(OHLCV)].astype("float64").copy()
    out.index = pd.DatetimeIndex(dates, name="date")
    return out.sort_index()


def normalize_intraday_frame(raw: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Intraday bars -> float OHLCV on a tz-aware (exchange time zone) index."""
    _require_columns(raw, OHLCV)
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.tz is None:
        raise ValueError("Intraday index must be a timezone-aware DatetimeIndex")
    out = raw.loc[:, list(OHLCV)].astype("float64").copy()
    out.index = raw.index.tz_convert(timezone)
    out.index.name = "time"
    return out.sort_index()


def fetch_daily_history(
    ticker: str, period: str, data: DataSettings, market: MarketSettings
) -> pd.DataFrame:
    """Daily bars from Yahoo. Errors propagate (raise_errors=True): no empty fallbacks."""
    import yfinance as yf

    raw = yf.Ticker(ticker).history(
        period=period,
        interval="1d",
        auto_adjust=data.auto_adjust,
        repair=data.yahoo_repair,
        actions=False,
        timeout=data.request_timeout_seconds,
        raise_errors=True,
    )
    if raw.empty:
        raise ValueError(f"Yahoo returned no daily bars for {ticker}")
    return normalize_daily_frame(raw, market.timezone)


def fetch_intraday(ticker: str, data: DataSettings, market: MarketSettings) -> pd.DataFrame:
    """Bars of the latest trading day (Yahoo: about 10 minutes delayed for .JK).

    Intraday prices are requested unadjusted: they are the traded prices, and the
    latest adjusted daily close equals the traded close.
    """
    import yfinance as yf

    raw = yf.Ticker(ticker).history(
        period="1d",
        interval=data.intraday_interval,
        auto_adjust=False,
        prepost=False,
        actions=False,
        timeout=data.request_timeout_seconds,
        raise_errors=True,
    )
    if raw.empty:
        raise ValueError(f"Yahoo returned no intraday bars for {ticker}")
    return normalize_intraday_frame(raw, market.timezone)


@dataclass(frozen=True)
class DataQualityReport:
    ticker: str
    n_raw: int
    n_clean: int
    duplicate_dates: tuple[str, ...]
    dropped_missing_close: tuple[str, ...]
    zero_volume_days: int
    high_below_low_days: tuple[str, ...]
    large_moves: tuple[tuple[str, float], ...]
    stale_runs: tuple[tuple[str, str, int], ...]
    first_date: str
    last_date: str

    def issues(self) -> list[str]:
        """Human-readable list of findings (Indonesian), empty when clean."""
        notes: list[str] = []
        if self.duplicate_dates:
            notes.append(
                f"{len(self.duplicate_dates)} tanggal ganda (bar terakhir dipakai): "
                + ", ".join(self.duplicate_dates[:5])
            )
        if self.dropped_missing_close:
            notes.append(
                f"{len(self.dropped_missing_close)} bar tanpa harga close dibuang: "
                + ", ".join(self.dropped_missing_close[:5])
            )
        if self.high_below_low_days:
            notes.append(
                f"{len(self.high_below_low_days)} bar dengan High < Low: "
                + ", ".join(self.high_below_low_days[:5])
            )
        if self.large_moves:
            moves = ", ".join(f"{d} ({r:+.1%})" for d, r in self.large_moves[:5])
            notes.append(f"Lonjakan harian besar (cek kebenarannya): {moves}")
        if self.stale_runs:
            runs = ", ".join(f"{a} s/d {b} ({n} hari)" for a, b, n in self.stale_runs[:5])
            notes.append(f"Harga tidak berubah dengan volume nol: {runs}")
        if self.zero_volume_days:
            notes.append(
                f"{self.zero_volume_days} hari dengan volume 0 "
                "(yfinance mengisi volume kosong dengan 0)"
            )
        return notes


def _stale_runs(close: pd.Series, volume: pd.Series, min_length: int) -> list[tuple[str, str, int]]:
    flag = ((volume == 0) & (close.diff() == 0)).to_numpy()
    runs: list[tuple[str, str, int]] = []
    start = None
    for i, value in enumerate(np.append(flag, False)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            length = i - start
            if length >= min_length:
                runs.append(
                    (str(close.index[start].date()), str(close.index[i - 1].date()), length)
                )
            start = None
    return runs


def validate_daily(
    frame: pd.DataFrame, ticker: str, data: DataSettings
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Check and minimally clean daily bars; every action is reported."""
    _require_columns(frame, OHLCV)
    n_raw = len(frame)
    duplicated = frame.index.duplicated(keep="last")
    duplicate_dates = tuple(str(d.date()) for d in frame.index[duplicated])
    clean = frame.loc[~duplicated]

    missing_close = clean["Close"].isna()
    dropped = tuple(str(d.date()) for d in clean.index[missing_close])
    clean = clean.loc[~missing_close].copy()
    if len(clean) < 2:
        raise ValueError(f"{ticker}: fewer than two valid daily closes")

    for column in ("Open", "High", "Low", "Close"):
        values = clean[column]
        bad = values.notna() & (values <= 0)
        if bad.any():
            dates = [str(d.date()) for d in clean.index[bad]]
            raise ValueError(f"{ticker}: non-positive {column} on {dates[:5]}")
    if (clean["Volume"] < 0).any():
        raise ValueError(f"{ticker}: negative volume")

    hl_bad = clean["High"].notna() & clean["Low"].notna() & (clean["High"] < clean["Low"])
    log_return = np.log(clean["Close"]).diff()
    large = log_return.abs() > data.max_abs_daily_log_return
    report = DataQualityReport(
        ticker=ticker,
        n_raw=n_raw,
        n_clean=len(clean),
        duplicate_dates=duplicate_dates,
        dropped_missing_close=dropped,
        zero_volume_days=int((clean["Volume"] == 0).sum()),
        high_below_low_days=tuple(str(d.date()) for d in clean.index[hl_bad]),
        large_moves=tuple(
            (str(d.date()), float(np.expm1(r)))
            for d, r in log_return[large].items()
        ),
        stale_runs=tuple(_stale_runs(clean["Close"], clean["Volume"], data.stale_run_length)),
        first_date=str(clean.index[0].date()),
        last_date=str(clean.index[-1].date()),
    )
    return clean, report


def split_provisional(
    frame: pd.DataFrame, now: pd.Timestamp, market: MarketSettings
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Separate a still-changing bar for today from the final bars."""
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Daily index must be strictly increasing")
    if is_daily_bar_final(frame.index[-1], now, market):
        return frame, None
    final = frame.iloc[:-1]
    if len(final) and not is_daily_bar_final(final.index[-1], now, market):
        raise AssertionError("More than one non-final daily bar")
    return final, frame.iloc[[-1]]


def provisional_bar_from_intraday(
    intraday: pd.DataFrame, session_date: pd.Timestamp
) -> pd.DataFrame | None:
    """Aggregate today's intraday bars into one provisional daily bar (or None)."""
    day = pd.Timestamp(session_date.date())
    bar_days = intraday.index.normalize().tz_localize(None)
    today = intraday.loc[bar_days == day]
    closes = today["Close"].dropna()
    if closes.empty:
        return None
    opens = today["Open"].dropna()
    row = {
        "Open": float(opens.iloc[0]) if len(opens) else float(closes.iloc[0]),
        "High": float(today["High"].max()),
        "Low": float(today["Low"].min()),
        "Close": float(closes.iloc[-1]),
        "Volume": float(today["Volume"].fillna(0).sum()),
    }
    return pd.DataFrame([row], index=pd.DatetimeIndex([day], name="date"))


@dataclass(frozen=True)
class LiveQuote:
    ticker: str
    price: float
    bar_time: pd.Timestamp
    session_date: pd.Timestamp
    previous_close: float
    previous_close_date: pd.Timestamp
    change: float
    change_pct: float
    intraday_close: pd.Series


def build_live_quote(ticker: str, intraday: pd.DataFrame, daily: pd.DataFrame) -> LiveQuote:
    """Latest price vs. the close of the previous session."""
    closes = intraday["Close"].dropna()
    if closes.empty:
        raise ValueError(f"{ticker}: intraday bars have no prices")
    bar_time = closes.index[-1]
    session_date = pd.Timestamp(bar_time.date())
    prior = daily.loc[daily.index < session_date, "Close"]
    if prior.empty:
        raise ValueError(f"{ticker}: no daily close before {session_date.date()}")
    previous_close = float(prior.iloc[-1])
    price = float(closes.iloc[-1])
    day_mask = closes.index.normalize().tz_localize(None) == session_date
    return LiveQuote(
        ticker=ticker,
        price=price,
        bar_time=bar_time,
        session_date=session_date,
        previous_close=previous_close,
        previous_close_date=prior.index[-1],
        change=price - previous_close,
        change_pct=(price / previous_close - 1.0) * 100.0,
        intraday_close=closes.loc[day_mask],
    )
