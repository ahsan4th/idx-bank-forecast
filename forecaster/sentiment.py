"""News sentiment from the GDELT DOC 2.0 API, aligned to IDX trading days.

Tone is GDELT's article tone (positive minus negative word share, computed on
machine-translated text), averaged per UTC day and weighted by article counts.

Timing rule (no look-ahead): GDELT's daily bins are UTC days, which run from
07:00 WIB to 07:00 WIB the next day. The bin for UTC day t therefore contains
news published after the IDX close of day t. A forecast made at the close of
trading day d_i may only use bins up to UTC day d_i - lag_days (lag_days >= 1,
enforced in config). Feature row i aggregates the bins in
(d_{i-1} - lag_days, d_i - lag_days], so weekend news lands on Monday.
"""

from __future__ import annotations

import time as _time
from typing import Any, Callable

import numpy as np
import pandas as pd

SENTIMENT_COLUMNS: tuple[str, ...] = ("sent_tone", "sent_tone_ewm", "sent_log_count")
# Past covariates handed to foundation models (NaN filled with 0, see `covariate_matrix`).
FM_COVARIATES: tuple[str, ...] = ("sent_tone", "sent_log_count")


def daily_from_timelines(tone: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """gdeltdoc timeline frames -> one row per UTC day: tone, articles.

    `tone` has columns [datetime, <average tone>]; `volume` (timelinevolraw) has
    [datetime, <article count>, "All Articles"]. Sub-daily bins are aggregated with
    article-count weights; bins coarser than a day are rejected.
    """
    if tone.empty and volume.empty:
        return pd.DataFrame(columns=["tone", "articles"], dtype="float64")
    tone_cols = [c for c in tone.columns if c != "datetime"]
    vol_cols = [c for c in volume.columns if c not in ("datetime", "All Articles")]
    if len(tone_cols) != 1 or len(vol_cols) != 1:
        raise ValueError(
            f"Unexpected GDELT timeline columns: tone={list(tone.columns)}, volume={list(volume.columns)}"
        )
    merged = pd.merge(
        tone[["datetime", tone_cols[0]]].rename(columns={tone_cols[0]: "tone"}),
        volume[["datetime", vol_cols[0]]].rename(columns={vol_cols[0]: "articles"}),
        on="datetime",
        how="outer",
        validate="one_to_one",
    ).sort_values("datetime")
    stamps = pd.DatetimeIndex(pd.to_datetime(merged["datetime"], utc=True))
    if len(stamps) > 1:
        spacing = pd.Series(stamps).diff().dropna().median()
        if spacing > pd.Timedelta(days=1):
            raise ValueError(
                f"GDELT returned {spacing} bins; daily or finer is required. "
                "Lower sentiment.chunk_days in settings.toml."
            )
    articles = merged["articles"].fillna(0.0).to_numpy(dtype=float)
    tone_values = merged["tone"].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "weighted": np.where(articles > 0, np.nan_to_num(tone_values) * articles, 0.0),
            "articles": articles,
        },
        index=stamps.tz_convert("UTC").normalize().tz_localize(None),
    )
    daily = frame.groupby(level=0).sum()
    daily["tone"] = np.where(daily["articles"] > 0, daily["weighted"] / daily["articles"], np.nan)
    daily.index.name = "utc_date"
    return daily[["tone", "articles"]]


def fetch_gdelt_daily(
    terms: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int,
    pause_seconds: float,
    client: Any,
    filters_factory: Callable[..., Any],
    on_chunk: Callable[[int, int], None] | None,
) -> pd.DataFrame:
    """Daily UTC tone and article counts for `terms` (OR-ed) over [start, end].

    Requests run in chunks with a pause between calls to respect GDELT's rate
    limits. Days absent from the response have 0 articles and undefined tone.
    """
    start, end = pd.Timestamp(start.date()), pd.Timestamp(end.date())
    if end < start:
        raise ValueError("end must not precede start")
    chunks = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + pd.Timedelta(days=chunk_days - 1), end)
        chunks.append((cursor, stop))
        cursor = stop + pd.Timedelta(days=1)
    parts = []
    for i, (a, b) in enumerate(chunks):
        filters = filters_factory(
            keyword=list(terms),
            start_date=a.strftime("%Y-%m-%d"),
            end_date=(b + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        tone = client.timeline_search("timelinetone", filters)
        _time.sleep(pause_seconds)
        volume = client.timeline_search("timelinevolraw", filters)
        _time.sleep(pause_seconds)
        parts.append(daily_from_timelines(tone, volume))
        if on_chunk is not None:
            on_chunk(i + 1, len(chunks))
    combined = pd.concat([p for p in parts if not p.empty]) if any(not p.empty for p in parts) else None
    full_range = pd.date_range(start, end, freq="D", name="utc_date")
    if combined is None:
        return pd.DataFrame({"tone": np.nan, "articles": 0.0}, index=full_range)
    combined = combined.groupby(level=0).agg({"tone": "mean", "articles": "sum"})
    combined = combined.reindex(full_range)
    combined["articles"] = combined["articles"].fillna(0.0)
    return combined


def align_to_trading_days(
    daily_utc: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    lag_days: int,
    ewm_halflife_days: float,
    coverage_start: pd.Timestamp,
) -> pd.DataFrame:
    """Sentiment features per trading day, using only bins known before its close.

    Row i aggregates UTC days in (d_{i-1} - lag, d_i - lag] (row 0: the single day
    d_0 - lag). Windows reaching before `coverage_start` or past the fetched data
    are NaN, not zero: "no data" is different from "no articles".
    """
    if lag_days < 1:
        raise ValueError("lag_days must be >= 1")
    if not trading_dates.is_monotonic_increasing:
        raise ValueError("trading_dates must be increasing")
    lag = pd.Timedelta(days=lag_days)
    data_end = daily_utc.index.max() if len(daily_utc) else pd.Timestamp.min
    tone_x_articles = (daily_utc["tone"].fillna(0.0) * daily_utc["articles"]).to_numpy()
    articles = daily_utc["articles"].to_numpy(dtype=float)
    index = daily_utc.index

    rows = []
    previous_end = None
    for date in trading_dates:
        window_end = pd.Timestamp(date.date()) - lag
        window_start = (window_end - pd.Timedelta(days=1)) if previous_end is None else previous_end
        previous_end = window_end
        # Window is (window_start, window_end].
        if window_start + pd.Timedelta(days=1) < coverage_start or window_end > data_end:
            rows.append((np.nan, np.nan))
            continue
        mask = (index > window_start) & (index <= window_end)
        n_articles = float(articles[mask].sum())
        tone = float(tone_x_articles[mask].sum() / n_articles) if n_articles > 0 else np.nan
        rows.append((tone, float(np.log1p(n_articles))))

    out = pd.DataFrame(rows, index=trading_dates, columns=["sent_tone", "sent_log_count"])
    # Exponentially weighted tone over trading days; uses past rows only.
    out["sent_tone_ewm"] = out["sent_tone"].ewm(halflife=ewm_halflife_days, ignore_na=True).mean()
    return out[list(SENTIMENT_COLUMNS)]


def covariate_matrix(features: pd.DataFrame) -> np.ndarray:
    """(K, T) past covariates for foundation models.

    Modelling decision: undefined tone (no articles, or no coverage) becomes 0,
    GDELT's neutral tone, and a missing count becomes 0 articles.
    """
    return features[list(FM_COVARIATES)].fillna(0.0).to_numpy(dtype=np.float32).T


def recent_articles(
    terms: tuple[str, ...], client: Any, filters_factory: Callable[..., Any], timespan: str, max_records: int
) -> pd.DataFrame:
    """Latest matching headlines (GDELT ArtList), newest first."""
    filters = filters_factory(keyword=list(terms), timespan=timespan, num_records=max_records)
    articles = client.article_search(filters)
    if articles.empty:
        return articles
    articles = articles.assign(
        seen=pd.to_datetime(articles["seendate"], format="%Y%m%dT%H%M%SZ", utc=True)
    ).sort_values("seen", ascending=False)
    return articles[["seen", "title", "domain", "language", "sourcecountry", "url"]]
