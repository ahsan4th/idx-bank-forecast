"""Rolling-origin backtest and final forecast for one ticker.

Design (fixed before looking at results):
- Origins: the last N bars whose five horizons are all observed, so every model
  and horizon is scored on the same N origins (a balanced panel).
- Benchmark: the random walk (naive). Each other model is compared with it by
  relative MAE and a Diebold-Mariano test on absolute errors; Holm's correction
  is applied across all model x horizon tests of the ticker.
- Ranking: lowest MAE averaged over horizons. The selection uses the same
  backtest it is reported on, so the winner's skill is slightly optimistic.
- Intervals: split-conformal offsets from the backtest log residuals per horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from forecaster import evaluation as ev
from forecaster.config import Settings
from forecaster.market import TradingCalendar, next_sessions
from forecaster.models import (
    ENSEMBLE_KEY,
    MODEL_SPECS,
    NAIVE_KEY,
    ForecastBatch,
    FoundationRunner,
    ensemble_forecast,
    gbm_backtest,
    gbm_final,
    make_contexts,
    naive_forecast,
    parse_variant,
    variant_label,
)
from forecaster.sentiment import FM_COVARIATES, covariate_matrix

RunnerLoader = Callable[[str], FoundationRunner]


def backtest_origin_indices(n_obs: int, n_origins: int, horizon: int) -> np.ndarray:
    last = n_obs - 1 - horizon
    first = last - n_origins + 1
    if n_origins < 1 or first < 0:
        raise ValueError(
            f"{n_obs} bar tidak cukup untuk {n_origins} origin dan horizon {horizon}"
        )
    return np.arange(first, last + 1)


@dataclass(frozen=True)
class BacktestResult:
    ticker: str
    horizon: int
    origin_index: np.ndarray
    origin_dates: tuple[str, ...]
    origin_price: np.ndarray
    actual: np.ndarray
    forecasts: dict[str, ForecastBatch]
    failures: dict[str, str]
    ensemble_members: tuple[str, ...]
    metrics: pd.DataFrame
    summary: pd.DataFrame
    ablation: pd.DataFrame
    best_model: str
    data_last_date: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def log_residuals(self, key: str) -> np.ndarray:
        return np.log(self.actual) - np.log(self.forecasts[key].median)


def _foundation_batch(
    base: str,
    loader: RunnerLoader,
    close: np.ndarray,
    origins: np.ndarray,
    horizon: int,
    context_length: int,
    covariates: np.ndarray | None,
) -> ForecastBatch:
    runner = loader(base)
    contexts = make_contexts(close, origins, context_length)
    cov = make_contexts(covariates, origins, context_length) if covariates is not None else None
    names = FM_COVARIATES if covariates is not None else ()
    return runner.predict(contexts, horizon, cov, names)


def run_backtest(
    ticker: str,
    final: pd.DataFrame,
    model_keys: tuple[str, ...],
    settings: Settings,
    n_origins: int,
    loader: RunnerLoader,
    sentiment: pd.DataFrame | None,
    on_model: Callable[[str], None] | None,
) -> BacktestResult:
    """Score every requested model on the same origins.

    Foundation models depend on downloads and optional packages, so their
    failures are caught, recorded in `failures`, and shown; the model is then
    excluded. Errors in the naive and gradient-boosting code are not caught.
    """
    if NAIVE_KEY not in model_keys:
        raise ValueError("The naive benchmark must always be part of the backtest")
    horizon = settings.forecast.horizon
    fs, g = settings.models.foundation, settings.models.gbm
    origins = backtest_origin_indices(len(final), n_origins, horizon)
    close = final["Close"].to_numpy(dtype=float)
    actual = close[origins[:, None] + np.arange(1, horizon + 1)[None, :]]
    cov_matrix = covariate_matrix(sentiment) if sentiment is not None else None

    forecasts: dict[str, ForecastBatch] = {}
    failures: dict[str, str] = {}
    for key in model_keys:
        base, with_sent = parse_variant(key)
        if with_sent and sentiment is None:
            raise ValueError(f"{key} needs sentiment features")
        if with_sent and not MODEL_SPECS[base].supports_sentiment:
            raise ValueError(f"{base} does not take sentiment input")
        if on_model is not None:
            on_model(key)
        if base == NAIVE_KEY:
            forecasts[key] = naive_forecast(close, origins, horizon)
        elif base == "gbm":
            forecasts[key] = gbm_backtest(
                final, origins, horizon, g, settings.forecast.seed,
                sentiment if with_sent else None,
            )
        else:
            try:
                forecasts[key] = _foundation_batch(
                    base, loader, close, origins, horizon, fs.context_length,
                    cov_matrix if with_sent else None,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, model excluded
                failures[key] = f"{type(exc).__name__}: {exc}"

    members = tuple(
        k for k in forecasts if k != NAIVE_KEY and not parse_variant(k)[1]
    )
    if len(members) >= 2:
        forecasts[ENSEMBLE_KEY] = ensemble_forecast([forecasts[k] for k in members])

    metrics, summary, ablation, best = _score(
        forecasts, actual, close[origins], horizon, settings
    )
    return BacktestResult(
        ticker=ticker,
        horizon=horizon,
        origin_index=origins,
        origin_dates=tuple(str(d.date()) for d in final.index[origins]),
        origin_price=close[origins],
        actual=actual,
        forecasts=forecasts,
        failures=failures,
        ensemble_members=members if len(members) >= 2 else (),
        metrics=metrics,
        summary=summary,
        ablation=ablation,
        best_model=best,
        data_last_date=str(final.index[-1].date()),
    )


def _score(
    forecasts: dict[str, ForecastBatch],
    actual: np.ndarray,
    origin_price: np.ndarray,
    horizon: int,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    bt = settings.backtest
    naive_err = np.abs(actual - forecasts[NAIVE_KEY].median)
    naive_mae = naive_err.mean(axis=0)

    rows = []
    for key, batch in forecasts.items():
        point = ev.point_metrics(actual, batch.median)
        err = np.abs(actual - batch.median)
        hits, trials = ev.directional_hits(actual, batch.median, origin_price)
        native = (
            ev.native_coverage(actual, batch.lower, batch.upper)
            if batch.lower is not None and batch.upper is not None
            else np.full(horizon, np.nan)
        )
        for h in range(horizon):
            if key == NAIVE_KEY:
                dm = ev.DMResult(np.nan, np.nan, len(actual), h + 1, "")
            else:
                dm = ev.dm_test(naive_err[:, h], err[:, h], h + 1)
            lo, hi = ev.wilson_interval(int(hits[h]), int(trials[h]), bt.hit_rate_ci_level)
            rows.append(
                {
                    "model": key,
                    "h": h + 1,
                    "mae": point["mae"][h],
                    "rmse": point["rmse"][h],
                    "mape": point["mape"][h],
                    "rel_mae": point["mae"][h] / naive_mae[h],
                    "dm_stat": dm.statistic,
                    "dm_p": dm.p_value,
                    "dm_note": dm.note,
                    "hits": int(hits[h]),
                    "trials": int(trials[h]),
                    "hit_rate": hits[h] / trials[h] if trials[h] else np.nan,
                    "hit_lo": lo,
                    "hit_hi": hi,
                    "native_cov": native[h],
                }
            )
    metrics = pd.DataFrame(rows)
    metrics["dm_p_holm"] = ev.holm_adjust(metrics["dm_p"].to_numpy())

    summary_rows = []
    for key, batch in forecasts.items():
        m = metrics[metrics["model"] == key]
        residuals = np.log(actual) - np.log(batch.median)
        better = m[(m["dm_p_holm"] < bt.dm_alpha) & (m["dm_stat"] > 0)]
        worse = m[(m["dm_p_holm"] < bt.dm_alpha) & (m["dm_stat"] < 0)]
        row = {
            "model": key,
            "label": variant_label(key),
            "mean_mae": m["mae"].mean(),
            "mean_rel_mae": m["rel_mae"].mean(),
            "skill_pct": (1.0 - m["mae"].mean() / naive_mae.mean()) * 100.0,
            "n_sig_better": len(better),
            "n_sig_worse": len(worse),
            "native_cov": np.nanmean(m["native_cov"]) if m["native_cov"].notna().any() else np.nan,
        }
        for level in settings.forecast.interval_levels:
            cov = ev.split_conformal_coverage(residuals, level, bt.coverage_calibration_fraction)
            row[f"conformal_cov_{int(round(level * 100))}"] = cov.pooled if cov else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("mean_mae", kind="mergesort")
    best = str(summary["model"].iloc[0])

    ablation_rows = []
    for key in forecasts:
        base, with_sent = parse_variant(key)
        if not with_sent or base not in forecasts:
            continue
        err_base = np.abs(actual - forecasts[base].median)
        err_sent = np.abs(actual - forecasts[key].median)
        for h in range(horizon):
            dm = ev.dm_test(err_base[:, h], err_sent[:, h], h + 1)
            ablation_rows.append(
                {
                    "model": base,
                    "h": h + 1,
                    "mae_without": err_base[:, h].mean(),
                    "mae_with": err_sent[:, h].mean(),
                    "ratio": err_sent[:, h].mean() / err_base[:, h].mean(),
                    "dm_stat": dm.statistic,
                    "dm_p": dm.p_value,
                }
            )
    ablation = pd.DataFrame(ablation_rows)
    if len(ablation):
        ablation["dm_p_holm"] = ev.holm_adjust(ablation["dm_p"].to_numpy())
    return metrics, summary.reset_index(drop=True), ablation, best


@dataclass(frozen=True)
class FinalForecast:
    ticker: str
    origin_date: pd.Timestamp
    origin_price: float
    provisional: bool
    origin_is_session: bool
    interval_level: float
    table: pd.DataFrame
    failures: dict[str, str]


def run_final_forecast(
    ticker: str,
    final: pd.DataFrame,
    origin_frame: pd.DataFrame,
    backtest: BacktestResult,
    settings: Settings,
    interval_level: float,
    calendar: TradingCalendar,
    loader: RunnerLoader,
    sentiment_final: pd.DataFrame | None,
    sentiment_origin: pd.DataFrame | None,
) -> FinalForecast:
    """h+1..h+H forecasts from the last bar of `origin_frame` for every scored model."""
    horizon = settings.forecast.horizon
    fs, g = settings.models.foundation, settings.models.gbm
    close_origin = origin_frame["Close"].to_numpy(dtype=float)
    origin = np.array([len(origin_frame) - 1])
    origin_price = float(close_origin[-1])
    origin_date = origin_frame.index[-1]
    dates, origin_is_session = next_sessions(calendar, origin_date, horizon)
    cov_origin = covariate_matrix(sentiment_origin) if sentiment_origin is not None else None

    medians: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    for key in backtest.forecasts:
        if key == ENSEMBLE_KEY:
            continue
        base, with_sent = parse_variant(key)
        if base == NAIVE_KEY:
            medians[key] = naive_forecast(close_origin, origin, horizon).median[0]
        elif base == "gbm":
            medians[key] = gbm_final(
                final, origin_frame, horizon, g, settings.forecast.seed,
                sentiment_final if with_sent else None,
                sentiment_origin if with_sent else None,
            ).median[0]
        else:
            try:
                medians[key] = _foundation_batch(
                    base, loader, close_origin, origin, horizon, fs.context_length,
                    cov_origin if with_sent else None,
                ).median[0]
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, model excluded
                failures[key] = f"{type(exc).__name__}: {exc}"
    if backtest.ensemble_members and all(k in medians for k in backtest.ensemble_members):
        medians[ENSEMBLE_KEY] = np.median(
            np.stack([medians[k] for k in backtest.ensemble_members]), axis=0
        )

    rows = []
    for key, median in medians.items():
        residuals = backtest.log_residuals(key)
        for h in range(horizon):
            q_lo, q_hi = ev.conformal_offsets(residuals[:, h], interval_level)
            rows.append(
                {
                    "model": key,
                    "label": variant_label(key),
                    "h": h + 1,
                    "date": dates[h],
                    "median": median[h],
                    "lower": median[h] * np.exp(q_lo),
                    "upper": median[h] * np.exp(q_hi),
                    "change_pct": (median[h] / origin_price - 1.0) * 100.0,
                }
            )
    return FinalForecast(
        ticker=ticker,
        origin_date=origin_date,
        origin_price=origin_price,
        provisional=len(origin_frame) > len(final),
        origin_is_session=origin_is_session,
        interval_level=interval_level,
        table=pd.DataFrame(rows),
        failures=failures,
    )
