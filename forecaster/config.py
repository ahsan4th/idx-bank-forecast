"""Load settings.toml into frozen dataclasses.

Every parameter that can change a result lives in settings.toml. Nothing in this
package carries a default for such a parameter: a missing key raises KeyError when
the settings are loaded, before any data is fetched.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any


def parse_hhmm(value: str) -> time:
    """Parse "HH:MM" into a time; raise on anything else."""
    hours, minutes = value.split(":")
    parsed = time(int(hours), int(minutes))
    if f"{parsed:%H:%M}" != value:
        raise ValueError(f"Expected HH:MM, got {value!r}")
    return parsed


def _window(pair: list[str]) -> tuple[time, time]:
    if len(pair) != 2:
        raise ValueError(f"A time window needs [start, end], got {pair!r}")
    start, end = parse_hhmm(pair[0]), parse_hhmm(pair[1])
    if not start < end:
        raise ValueError(f"Window start must precede end: {pair!r}")
    return start, end


@dataclass(frozen=True)
class DataSettings:
    tickers_default: tuple[str, ...]
    tickers_optional: tuple[str, ...]
    history_period: str
    history_period_options: tuple[str, ...]
    auto_adjust: bool
    yahoo_repair: bool
    daily_cache_ttl_seconds: int
    intraday_cache_ttl_seconds: int
    intraday_interval: str
    request_timeout_seconds: int
    max_abs_daily_log_return: float
    stale_run_length: int


@dataclass(frozen=True)
class MarketSettings:
    timezone: str
    calendar: str
    yahoo_delay_minutes: int
    daily_bar_final_time: time
    pre_opening: tuple[time, time]
    sessions_mon_thu: tuple[tuple[time, time], tuple[time, time]]
    sessions_fri: tuple[tuple[time, time], tuple[time, time]]
    pre_closing: tuple[time, time]


@dataclass(frozen=True)
class ForecastSettings:
    horizon: int
    interval_levels: tuple[float, ...]
    interval_level_default: float
    seed: int


@dataclass(frozen=True)
class BacktestSettings:
    n_origins_options: tuple[int, ...]
    n_origins_default: int
    dm_alpha: float
    hit_rate_ci_level: float
    coverage_calibration_fraction: float


@dataclass(frozen=True)
class GBMSettings:
    refit_every: int
    min_train_samples: int
    loss: str
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    return_lags: int
    momentum_windows: tuple[int, ...]
    volatility_windows: tuple[int, ...]
    ma_windows: tuple[int, ...]
    rsi_window: int
    volume_window: int
    range_window: int


@dataclass(frozen=True)
class FoundationSettings:
    context_length: int
    device: str
    native_quantile_low: float
    native_quantile_high: float
    chronos2_model_id: str
    chronos2_batch_size: int
    chronos_bolt_model_id: str
    timestral_model_id: str
    timestral_symmetric_averaging: bool
    timesfm_model_id: str
    timesfm_torch_compile: bool
    timesfm_per_core_batch_size: int
    timesfm_max_horizon: int
    timesfm_normalize_inputs: bool
    timesfm_use_continuous_quantile_head: bool
    timesfm_force_flip_invariance: bool
    timesfm_infer_is_positive: bool
    timesfm_fix_quantile_crossing: bool


@dataclass(frozen=True)
class ModelSettings:
    default_selection: tuple[str, ...]
    gbm: GBMSettings
    foundation: FoundationSettings


@dataclass(frozen=True)
class SentimentSettings:
    enabled_default: bool
    source: str
    earliest_date: str
    chunk_days: int
    request_pause_seconds: float
    lag_days: int
    ewm_halflife_days: float
    cache_ttl_seconds: int
    headlines_timespan: str
    headlines_max_records: int
    queries: tuple[tuple[str, tuple[str, ...]], ...]

    def query_for(self, ticker: str) -> tuple[str, ...]:
        for key, terms in self.queries:
            if key == ticker:
                return terms
        raise KeyError(
            f"No news query for {ticker!r} in [sentiment.queries] of settings.toml"
        )


@dataclass(frozen=True)
class UISettings:
    chart_history_days: int
    refresh_options_seconds: tuple[int, ...]
    refresh_default_seconds: int


@dataclass(frozen=True)
class Settings:
    data: DataSettings
    market: MarketSettings
    forecast: ForecastSettings
    backtest: BacktestSettings
    models: ModelSettings
    sentiment: SentimentSettings
    ui: UISettings


def _load_market(raw: dict[str, Any]) -> MarketSettings:
    mon_thu = tuple(_window(w) for w in raw["sessions_mon_thu"])
    fri = tuple(_window(w) for w in raw["sessions_fri"])
    pre_opening = _window(raw["pre_opening"])
    pre_closing = _window(raw["pre_closing"])
    for name, sessions in (("sessions_mon_thu", mon_thu), ("sessions_fri", fri)):
        if len(sessions) != 2:
            raise ValueError(f"{name} must list exactly two sessions")
        if not (pre_opening[1] <= sessions[0][0] < sessions[0][1] <= sessions[1][0]):
            raise ValueError(f"{name} sessions are out of order: {sessions}")
        if not sessions[1][1] <= pre_closing[0]:
            raise ValueError(f"{name} session II must end before pre-closing")
    return MarketSettings(
        timezone=raw["timezone"],
        calendar=raw["calendar"],
        yahoo_delay_minutes=int(raw["yahoo_delay_minutes"]),
        daily_bar_final_time=parse_hhmm(raw["daily_bar_final_time"]),
        pre_opening=pre_opening,
        sessions_mon_thu=mon_thu,  # type: ignore[arg-type]
        sessions_fri=fri,  # type: ignore[arg-type]
        pre_closing=pre_closing,
    )


def load_settings(path: Path) -> Settings:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    d = raw["data"]
    data = DataSettings(
        tickers_default=tuple(d["tickers_default"]),
        tickers_optional=tuple(d["tickers_optional"]),
        history_period=d["history_period"],
        history_period_options=tuple(d["history_period_options"]),
        auto_adjust=bool(d["auto_adjust"]),
        yahoo_repair=bool(d["yahoo_repair"]),
        daily_cache_ttl_seconds=int(d["daily_cache_ttl_seconds"]),
        intraday_cache_ttl_seconds=int(d["intraday_cache_ttl_seconds"]),
        intraday_interval=d["intraday_interval"],
        request_timeout_seconds=int(d["request_timeout_seconds"]),
        max_abs_daily_log_return=float(d["max_abs_daily_log_return"]),
        stale_run_length=int(d["stale_run_length"]),
    )
    if data.history_period not in data.history_period_options:
        raise ValueError("data.history_period must be one of history_period_options")

    f = raw["forecast"]
    forecast = ForecastSettings(
        horizon=int(f["horizon"]),
        interval_levels=tuple(float(x) for x in f["interval_levels"]),
        interval_level_default=float(f["interval_level_default"]),
        seed=int(f["seed"]),
    )
    if forecast.horizon < 1:
        raise ValueError("forecast.horizon must be >= 1")
    if forecast.interval_level_default not in forecast.interval_levels:
        raise ValueError("forecast.interval_level_default must be in interval_levels")
    if not all(0.0 < level < 1.0 for level in forecast.interval_levels):
        raise ValueError("Interval levels must lie strictly between 0 and 1")

    b = raw["backtest"]
    backtest = BacktestSettings(
        n_origins_options=tuple(int(x) for x in b["n_origins_options"]),
        n_origins_default=int(b["n_origins_default"]),
        dm_alpha=float(b["dm_alpha"]),
        hit_rate_ci_level=float(b["hit_rate_ci_level"]),
        coverage_calibration_fraction=float(b["coverage_calibration_fraction"]),
    )
    if backtest.n_origins_default not in backtest.n_origins_options:
        raise ValueError("backtest.n_origins_default must be in n_origins_options")
    if not 0.0 < backtest.coverage_calibration_fraction < 1.0:
        raise ValueError("coverage_calibration_fraction must lie in (0, 1)")

    m = raw["models"]
    g = m["gbm"]
    gbm = GBMSettings(
        refit_every=int(g["refit_every"]),
        min_train_samples=int(g["min_train_samples"]),
        loss=g["loss"],
        learning_rate=float(g["learning_rate"]),
        max_iter=int(g["max_iter"]),
        max_leaf_nodes=int(g["max_leaf_nodes"]),
        min_samples_leaf=int(g["min_samples_leaf"]),
        l2_regularization=float(g["l2_regularization"]),
        return_lags=int(g["return_lags"]),
        momentum_windows=tuple(int(x) for x in g["momentum_windows"]),
        volatility_windows=tuple(int(x) for x in g["volatility_windows"]),
        ma_windows=tuple(int(x) for x in g["ma_windows"]),
        rsi_window=int(g["rsi_window"]),
        volume_window=int(g["volume_window"]),
        range_window=int(g["range_window"]),
    )
    fo, c2, cb, ts, tf = (
        m["foundation"],
        m["chronos2"],
        m["chronos_bolt"],
        m["timestral"],
        m["timesfm"],
    )
    foundation = FoundationSettings(
        context_length=int(fo["context_length"]),
        device=fo["device"],
        native_quantile_low=float(fo["native_quantile_low"]),
        native_quantile_high=float(fo["native_quantile_high"]),
        chronos2_model_id=c2["model_id"],
        chronos2_batch_size=int(c2["batch_size"]),
        chronos_bolt_model_id=cb["model_id"],
        timestral_model_id=ts["model_id"],
        timestral_symmetric_averaging=bool(ts["symmetric_averaging"]),
        timesfm_model_id=tf["model_id"],
        timesfm_torch_compile=bool(tf["torch_compile"]),
        timesfm_per_core_batch_size=int(tf["per_core_batch_size"]),
        timesfm_max_horizon=int(tf["max_horizon"]),
        timesfm_normalize_inputs=bool(tf["normalize_inputs"]),
        timesfm_use_continuous_quantile_head=bool(tf["use_continuous_quantile_head"]),
        timesfm_force_flip_invariance=bool(tf["force_flip_invariance"]),
        timesfm_infer_is_positive=bool(tf["infer_is_positive"]),
        timesfm_fix_quantile_crossing=bool(tf["fix_quantile_crossing"]),
    )
    if foundation.device not in ("cpu", "cuda", "mps"):
        raise ValueError("models.foundation.device must be cpu, cuda, or mps")
    if foundation.timesfm_max_horizon < forecast.horizon:
        raise ValueError("models.timesfm.max_horizon must be >= forecast.horizon")
    models = ModelSettings(
        default_selection=tuple(m["default_selection"]),
        gbm=gbm,
        foundation=foundation,
    )

    s = raw["sentiment"]
    sentiment = SentimentSettings(
        enabled_default=bool(s["enabled_default"]),
        source=s["source"],
        earliest_date=s["earliest_date"],
        chunk_days=int(s["chunk_days"]),
        request_pause_seconds=float(s["request_pause_seconds"]),
        lag_days=int(s["lag_days"]),
        ewm_halflife_days=float(s["ewm_halflife_days"]),
        cache_ttl_seconds=int(s["cache_ttl_seconds"]),
        headlines_timespan=s["headlines_timespan"],
        headlines_max_records=int(s["headlines_max_records"]),
        queries=tuple(
            (ticker, tuple(terms)) for ticker, terms in sorted(s["queries"].items())
        ),
    )
    if sentiment.source != "gdelt":
        raise ValueError("Only sentiment.source = 'gdelt' is implemented")
    if sentiment.lag_days < 1:
        raise ValueError(
            "sentiment.lag_days must be >= 1: a UTC-day bin ends after the IDX close"
        )

    u = raw["ui"]
    ui = UISettings(
        chart_history_days=int(u["chart_history_days"]),
        refresh_options_seconds=tuple(int(x) for x in u["refresh_options_seconds"]),
        refresh_default_seconds=int(u["refresh_default_seconds"]),
    )
    if ui.refresh_default_seconds not in ui.refresh_options_seconds:
        raise ValueError("ui.refresh_default_seconds must be in refresh_options_seconds")
    shortest_refresh = min(s for s in ui.refresh_options_seconds if s > 0)
    if data.intraday_cache_ttl_seconds >= shortest_refresh:
        raise ValueError("data.intraday_cache_ttl_seconds must be shorter than every refresh option")

    return Settings(
        data=data,
        market=_load_market(raw["market"]),
        forecast=forecast,
        backtest=backtest,
        models=models,
        sentiment=sentiment,
        ui=ui,
    )
