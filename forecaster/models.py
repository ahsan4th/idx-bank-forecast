"""Forecasting models behind one contract.

For origin index o (the last bar a model may see) and horizon h, row i, column h-1
of `ForecastBatch.median` is the forecast of Close[o_i + h] in Rupiah. No model
reads a bar after its origin; tests/test_models.py checks this by perturbing the
future and asserting identical forecasts.

Foundation models are loaded lazily. Their packages are optional: a model whose
package is missing is reported as unavailable, never silently replaced.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from forecaster.config import FoundationSettings, GBMSettings
from forecaster.features import build_gbm_features, log_return_targets, warmup_length

SENTIMENT_SUFFIX = "+sent"
ENSEMBLE_KEY = "ensemble"
NAIVE_KEY = "naive"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str  # "baseline", "ml", "foundation"
    modules: tuple[str, ...]
    install_hint: str
    supports_sentiment: bool
    reference: str


TIMESTRAL_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"

MODEL_SPECS: dict[str, ModelSpec] = {
    "naive": ModelSpec(
        "naive", "Random walk (naive)", "baseline", (), "", False,
        "Harga besok = harga terakhir; tolok ukur wajib untuk harga saham.",
    ),
    "gbm": ModelSpec(
        "gbm", "Gradient boosting", "ml", ("sklearn",), "pip install scikit-learn", True,
        "HistGradientBoosting (scikit-learn), strategi direct per horizon.",
    ),
    "chronos2": ModelSpec(
        "chronos2", "Chronos-2", "foundation", ("torch", "chronos"),
        'pip install "chronos-forecasting>=2.1"', True,
        "Amazon, 120M parameter, Okt 2025; mendukung past covariates.",
    ),
    "chronos_bolt": ModelSpec(
        "chronos_bolt", "Chronos-Bolt", "foundation", ("torch", "chronos"),
        'pip install "chronos-forecasting>=2.1"', False,
        "Amazon, varian cepat (small, 48M parameter).",
    ),
    "timestral": ModelSpec(
        "timestral", "Timestral", "foundation", ("torch", "timestral"),
        f"pip install https://github.com/mistralai/timestral/archive/{TIMESTRAL_COMMIT}.zip",
        True,
        "Mistral, 209M parameter, Okt 2026; lisensi bobot Modified MIT.",
    ),
    "timesfm": ModelSpec(
        "timesfm", "TimesFM 2.5", "foundation", ("torch", "timesfm"),
        'pip install "timesfm[torch]>=3.0"', False,
        "Google, 200M parameter, bobot 2.5 berlisensi Apache-2.0.",
    ),
}


def variant_key(base: str, with_sentiment: bool) -> str:
    return f"{base}{SENTIMENT_SUFFIX}" if with_sentiment else base


def parse_variant(key: str) -> tuple[str, bool]:
    if key.endswith(SENTIMENT_SUFFIX):
        return key[: -len(SENTIMENT_SUFFIX)], True
    return key, False


def variant_label(key: str) -> str:
    if key == ENSEMBLE_KEY:
        return "Ensembel (median)"
    base, with_sentiment = parse_variant(key)
    label = MODEL_SPECS[base].label
    return f"{label} + sentimen" if with_sentiment else label


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def model_availability() -> dict[str, bool]:
    return {
        key: all(module_available(m) for m in spec.modules)
        for key, spec in MODEL_SPECS.items()
    }


@dataclass(frozen=True)
class ForecastBatch:
    """Price forecasts for N origins and H horizons (Rupiah)."""

    median: np.ndarray
    lower: np.ndarray | None  # model's own lower quantile, when it has one
    upper: np.ndarray | None

    def __post_init__(self) -> None:
        if self.median.ndim != 2:
            raise ValueError(f"median must be (N, H), got shape {self.median.shape}")
        if not np.all(np.isfinite(self.median)):
            raise ValueError("Forecasts contain non-finite values")
        if np.any(self.median <= 0):
            raise ValueError("Price forecasts must be positive")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("lower and upper must both be given or both be None")
        if self.lower is not None and self.upper is not None:
            for name, arr in (("lower", self.lower), ("upper", self.upper)):
                if arr.shape != self.median.shape:
                    raise ValueError(f"{name} shape {arr.shape} != median {self.median.shape}")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"{name} quantiles contain non-finite values")


def _check_origins(origins: np.ndarray, n_obs: int) -> None:
    if origins.ndim != 1 or len(origins) == 0:
        raise ValueError("origins must be a non-empty 1-D array")
    if not np.all(np.diff(origins) > 0):
        raise ValueError("origins must be strictly increasing")
    if origins[0] < 0 or origins[-1] >= n_obs:
        raise ValueError("origins out of range")


# ---------------------------------------------------------------- naive


def naive_forecast(close: np.ndarray, origins: np.ndarray, horizon: int) -> ForecastBatch:
    """Random walk: every horizon equals the price at the origin."""
    _check_origins(origins, len(close))
    last = close[origins].astype(float)
    return ForecastBatch(np.repeat(last[:, None], horizon, axis=1), None, None)


# ---------------------------------------------------------------- gradient boosting


def _new_gbm(g: GBMSettings, seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss=g.loss,
        learning_rate=g.learning_rate,
        max_iter=g.max_iter,
        max_leaf_nodes=g.max_leaf_nodes,
        min_samples_leaf=g.min_samples_leaf,
        l2_regularization=g.l2_regularization,
        early_stopping=False,
        random_state=seed,
    )


def _fit_horizon_models(
    features: np.ndarray,
    targets: pd.DataFrame,
    cutoff: int,
    horizon: int,
    g: GBMSettings,
    seed: int,
) -> list[Any]:
    """One model per horizon, trained only on targets observed by row `cutoff`.

    Row s may train the horizon-h model only if s + h <= cutoff, so its target
    log(P_{s+h} / P_s) was known at the cutoff.
    """
    warm = warmup_length(g)
    fitted = []
    for h in range(1, horizon + 1):
        rows = np.arange(warm, cutoff - h + 1)
        if len(rows) < g.min_train_samples:
            raise ValueError(
                f"Gradient boosting h={h}: {len(rows)} baris latih, butuh "
                f"{g.min_train_samples}. Pilih periode data yang lebih panjang."
            )
        y = targets[h].to_numpy()[rows]
        if np.isnan(y).any():
            raise AssertionError("Training target would use a bar after the cutoff")
        fitted.append(_new_gbm(g, seed).fit(features[rows], y))
    return fitted


def gbm_backtest(
    ohlcv: pd.DataFrame,
    origins: np.ndarray,
    horizon: int,
    g: GBMSettings,
    seed: int,
    sentiment: pd.DataFrame | None,
) -> ForecastBatch:
    """Rolling-origin forecasts; models are refit every `g.refit_every` origins.

    A model fitted at origin r is reused for origins r..r+refit_every-1. It saw
    only data up to r, which is never later than the origin it predicts from.
    """
    _check_origins(origins, len(ohlcv))
    features = build_gbm_features(ohlcv, g, sentiment).to_numpy(dtype=float)
    targets = log_return_targets(ohlcv["Close"], horizon)
    close = ohlcv["Close"].to_numpy(dtype=float)
    median = np.empty((len(origins), horizon))
    fitted: list[Any] | None = None
    fitted_at = -1
    for i, origin in enumerate(origins):
        if fitted is None or origin - fitted_at >= g.refit_every:
            fitted = _fit_horizon_models(features, targets, int(origin), horizon, g, seed)
            fitted_at = int(origin)
        x = features[origin : origin + 1]
        for h in range(horizon):
            median[i, h] = close[origin] * np.exp(fitted[h].predict(x)[0])
    return ForecastBatch(median, None, None)


def gbm_final(
    final: pd.DataFrame,
    origin_frame: pd.DataFrame,
    horizon: int,
    g: GBMSettings,
    seed: int,
    sentiment_final: pd.DataFrame | None,
    sentiment_origin: pd.DataFrame | None,
) -> ForecastBatch:
    """Fit on every final bar, then forecast from the last row of `origin_frame`.

    `origin_frame` is `final` itself or `final` plus one provisional bar for today.
    """
    if not origin_frame.iloc[: len(final)].equals(final):
        raise ValueError("origin_frame must start with the final bars")
    features_final = build_gbm_features(final, g, sentiment_final).to_numpy(dtype=float)
    targets = log_return_targets(final["Close"], horizon)
    fitted = _fit_horizon_models(features_final, targets, len(final) - 1, horizon, g, seed)
    x = build_gbm_features(origin_frame, g, sentiment_origin).to_numpy(dtype=float)[-1:]
    origin_price = float(origin_frame["Close"].iloc[-1])
    median = np.array([[origin_price * np.exp(m.predict(x)[0]) for m in fitted]])
    return ForecastBatch(median, None, None)


# ---------------------------------------------------------------- foundation models


def make_contexts(values: np.ndarray, origins: np.ndarray, length: int) -> np.ndarray:
    """Windows of `length` values ending at each origin (inclusive).

    `values` is (T,) for a target or (K, T) for covariates; the result is (N, L) or
    (N, K, L).
    """
    if origins[0] - length + 1 < 0:
        raise ValueError(
            f"Konteks {length} bar tidak tersedia untuk origin pertama (indeks {origins[0]}). "
            "Pilih periode data yang lebih panjang atau kurangi jumlah origin."
        )
    idx = origins[:, None] + np.arange(-length + 1, 1)[None, :]
    if values.ndim == 1:
        return values[idx].astype(np.float32)
    if values.ndim == 2:
        return np.transpose(values[:, idx], (1, 0, 2)).astype(np.float32)
    raise ValueError("values must be 1-D or 2-D")


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=float)


def _level_index(levels: Any, level: float) -> int:
    levels = np.asarray(levels, dtype=float)
    hits = np.flatnonzero(np.isclose(levels, level, atol=1e-9))
    if len(hits) != 1:
        raise ValueError(f"Quantile level {level} not among model levels {levels.tolist()}")
    return int(hits[0])


class FoundationRunner(Protocol):
    key: str
    supports_covariates: bool

    def predict(
        self,
        contexts: np.ndarray,
        horizon: int,
        covariates: np.ndarray | None,
        covariate_names: tuple[str, ...],
    ) -> ForecastBatch: ...


class Chronos2Runner:
    key = "chronos2"
    supports_covariates = True

    def __init__(self, pipeline: Any, batch_size: int, q_low: float, q_high: float):
        self.pipeline, self.batch_size = pipeline, batch_size
        self.levels = [q_low, 0.5, q_high]

    def predict(self, contexts, horizon, covariates, covariate_names):
        n = len(contexts)
        if covariates is None:
            inputs: Any = contexts[:, None, :]  # (N, 1, L): univariate items
        else:
            if covariates.shape[:2] != (n, len(covariate_names)):
                raise ValueError("covariates must be (N, K, L) with K names")
            inputs = [
                {
                    "target": contexts[i],
                    "past_covariates": {
                        name: covariates[i, k] for k, name in enumerate(covariate_names)
                    },
                }
                for i in range(n)
            ]
        # cross_learning stays False: every window is forecast on its own. Sharing
        # information across backtest windows would leak later prices into earlier ones.
        quantiles, _ = self.pipeline.predict_quantiles(
            inputs,
            prediction_length=horizon,
            quantile_levels=self.levels,
            batch_size=self.batch_size,
            cross_learning=False,
        )
        q = np.stack([_to_numpy(t) for t in quantiles])  # (N, 1, H, 3)
        if q.shape != (n, 1, horizon, 3):
            raise ValueError(f"Unexpected Chronos-2 output shape {q.shape}")
        q = q[:, 0]
        return ForecastBatch(q[..., 1], q[..., 0], q[..., 2])


class ChronosBoltRunner:
    key = "chronos_bolt"
    supports_covariates = False

    def __init__(self, pipeline: Any, q_low: float, q_high: float):
        self.pipeline = pipeline
        self.levels = [q_low, 0.5, q_high]

    def predict(self, contexts, horizon, covariates, covariate_names):
        if covariates is not None:
            raise ValueError("Chronos-Bolt does not accept covariates")
        import torch

        quantiles, _ = self.pipeline.predict_quantiles(
            torch.as_tensor(contexts, dtype=torch.float32),
            prediction_length=horizon,
            quantile_levels=self.levels,
        )
        q = _to_numpy(quantiles)  # (N, H, 3)
        if q.shape != (len(contexts), horizon, 3):
            raise ValueError(f"Unexpected Chronos-Bolt output shape {q.shape}")
        return ForecastBatch(q[..., 1], q[..., 0], q[..., 2])


class TimestralRunner:
    key = "timestral"
    supports_covariates = True

    def __init__(self, forecaster: Any, symmetric_averaging: bool, q_low: float, q_high: float):
        self.forecaster = forecaster
        self.symmetric_averaging = symmetric_averaging
        self.q_low, self.q_high = q_low, q_high

    def predict(self, contexts, horizon, covariates, covariate_names):
        if covariates is not None and covariates.shape[:2] != (len(contexts), len(covariate_names)):
            raise ValueError("covariates must be (N, K, L) with K names")
        forecast = self.forecaster.forecast(
            contexts,  # (B, T): one target variate per series
            prediction_length=horizon,
            past_covariates=covariates,  # (B, C_past, T) or None
            symmetric_averaging=self.symmetric_averaging,
        )
        q = _to_numpy(forecast.quantiles)  # (B, V, H, Q)
        if q.shape[:3] != (len(contexts), 1, horizon):
            raise ValueError(f"Unexpected Timestral output shape {q.shape}")
        levels = forecast.quantile_levels
        lo, med, hi = (
            q[:, 0, :, _level_index(levels, level)] for level in (self.q_low, 0.5, self.q_high)
        )
        return ForecastBatch(med, lo, hi)


class TimesFMRunner:
    key = "timesfm"
    supports_covariates = False
    # forecast() returns quantiles with index 0 = mean and 1..9 = levels 0.1..0.9
    # (timesfm.configs.DEFAULT_QUANTILES); the point forecast is the median.
    _DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def __init__(self, model: Any, q_low: float, q_high: float):
        self.model = model
        self.lo_idx = 1 + _level_index(self._DECILES, q_low)
        self.hi_idx = 1 + _level_index(self._DECILES, q_high)

    def predict(self, contexts, horizon, covariates, covariate_names):
        if covariates is not None:
            raise ValueError("TimesFM past-only covariates are not wired in this app")
        point, quantiles = self.model.forecast(horizon=horizon, inputs=list(contexts))
        point = np.asarray(point, dtype=float)[:, :horizon]
        quantiles = np.asarray(quantiles, dtype=float)[:, :horizon, :]
        if point.shape != (len(contexts), horizon) or quantiles.shape[2] != 10:
            raise ValueError(f"Unexpected TimesFM output shapes {point.shape}, {quantiles.shape}")
        return ForecastBatch(point, quantiles[..., self.lo_idx], quantiles[..., self.hi_idx])


def load_foundation_runner(key: str, fs: FoundationSettings) -> FoundationRunner:
    """Download (first call) and load one foundation model. Needs network + torch."""
    import torch

    q_low, q_high = fs.native_quantile_low, fs.native_quantile_high
    if key in ("chronos2", "chronos_bolt"):
        from chronos import BaseChronosPipeline

        model_id = fs.chronos2_model_id if key == "chronos2" else fs.chronos_bolt_model_id
        pipeline = BaseChronosPipeline.from_pretrained(
            model_id, device_map=fs.device, dtype=torch.float32
        )
        if key == "chronos2":
            return Chronos2Runner(pipeline, fs.chronos2_batch_size, q_low, q_high)
        return ChronosBoltRunner(pipeline, q_low, q_high)
    if key == "timestral":
        from timestral import TimestralForecaster

        forecaster = TimestralForecaster.from_pretrained(fs.timestral_model_id, device=fs.device)
        return TimestralRunner(forecaster, fs.timestral_symmetric_averaging, q_low, q_high)
    if key == "timesfm":
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            fs.timesfm_model_id, torch_compile=fs.timesfm_torch_compile
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=fs.context_length,
                max_horizon=fs.timesfm_max_horizon,
                normalize_inputs=fs.timesfm_normalize_inputs,
                use_continuous_quantile_head=fs.timesfm_use_continuous_quantile_head,
                force_flip_invariance=fs.timesfm_force_flip_invariance,
                infer_is_positive=fs.timesfm_infer_is_positive,
                fix_quantile_crossing=fs.timesfm_fix_quantile_crossing,
                per_core_batch_size=fs.timesfm_per_core_batch_size,
            )
        )
        return TimesFMRunner(model, q_low, q_high)
    raise KeyError(f"Unknown foundation model {key!r}")


# ---------------------------------------------------------------- ensemble


def ensemble_forecast(batches: list[ForecastBatch]) -> ForecastBatch:
    """Element-wise median of member forecasts (no interval of its own)."""
    if len(batches) < 2:
        raise ValueError("An ensemble needs at least two members")
    stacked = np.stack([b.median for b in batches])
    return ForecastBatch(np.median(stacked, axis=0), None, None)
