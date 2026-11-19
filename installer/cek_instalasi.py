"""Periksa instalasi: paket wajib, versi paket, kalender bursa, model foundation, dan Yahoo.

Pemasang menjalankan skrip ini dengan HF_HUB_OFFLINE=1 setelah bobot model diunduh. Setiap
model foundation dimuat dari cache lokal lalu diminta memprakirakan deret SINTETIS (random
walk geometris), hanya untuk memastikan bobotnya termuat dan keluarannya berupa harga positif
yang hingga. Ini bukan uji akurasi. Koneksi ke Yahoo Finance diperiksa terakhir, karena
aplikasi tetap butuh internet untuk harga saham; kegagalannya hanya peringatan.

Kode keluar: 0 siap, 1 paket wajib belum lengkap, 3 ada model foundation yang gagal dimuat.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecaster import market, models  # noqa: E402
from forecaster import data as price_data  # noqa: E402
from forecaster.config import FoundationSettings, Settings, load_settings  # noqa: E402

# Import names of the packages the app cannot start without.
REQUIRED_MODULES = ("streamlit", "yfinance", "exchange_calendars", "sklearn", "scipy", "pandas", "numpy")
# Distribution names whose versions are printed for the record.
REPORTED_PACKAGES = (
    "streamlit", "yfinance", "pandas", "numpy", "scikit-learn", "scipy", "exchange_calendars",
    "chronos-forecasting", "timesfm", "timestral", "torch", "huggingface_hub", "gdeltdoc",
)
EXIT_OK, EXIT_MISSING_REQUIRED, EXIT_FOUNDATION_FAILED = 0, 1, 3


def missing_required(module_available: Callable[[str], bool]) -> list[str]:
    return [name for name in REQUIRED_MODULES if not module_available(name)]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "tidak terpasang"


def synthetic_contexts(seed: int, length: int, n_series: int) -> np.ndarray:
    """Geometric random walks starting near Rp9.000: a stand-in series, not market data."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, 0.015, size=(n_series, length))
    return (9000.0 * np.exp(np.cumsum(log_returns, axis=1))).astype(np.float32)


def check_foundation_models(
    keys: Sequence[str],
    fs: FoundationSettings,
    horizon: int,
    seed: int,
    loader: Callable[[str, FoundationSettings], models.FoundationRunner],
) -> dict[str, str]:
    """Load each model and forecast synthetic series; map key -> "OK" or the failure."""
    contexts = synthetic_contexts(seed, fs.context_length, 2)
    status: dict[str, str] = {}
    for key in keys:
        try:
            batch = loader(key, fs).predict(contexts, horizon, None, ())
        except Exception as exc:  # noqa: BLE001 - reported per model, exactly as the app does
            status[key] = f"GAGAL: {type(exc).__name__}: {exc}"
            continue
        expected = (len(contexts), horizon)
        if batch.median.shape != expected:
            status[key] = f"GAGAL: bentuk keluaran {batch.median.shape}, seharusnya {expected}"
            continue
        status[key] = "OK"
    return status


def fetch_errors() -> tuple[type[BaseException], ...]:
    """The errors the app shows per ticker when Yahoo cannot be reached (same set)."""
    from yfinance.exceptions import YFException

    errors: list[type[BaseException]] = [ValueError, ImportError, OSError, TimeoutError, YFException]
    if models.module_available("requests"):
        import requests

        errors.append(requests.exceptions.RequestException)
    if models.module_available("curl_cffi"):
        from curl_cffi.requests.exceptions import RequestException as CurlRequestException

        errors.append(CurlRequestException)
    return tuple(errors)


def check_yahoo(settings: Settings) -> str:
    ticker = settings.data.tickers_default[0]
    try:
        bars = price_data.fetch_daily_history(ticker, "5d", settings.data, settings.market)
    except fetch_errors() as exc:
        return f"GAGAL ({type(exc).__name__}: {exc}). Aplikasi butuh internet untuk harga saham."
    return f"OK ({ticker}, {len(bars)} bar harian terakhir)"


def main() -> int:
    settings = load_settings(ROOT / "settings.toml")
    print(f"Python {platform.python_version()} di {sys.executable}")
    for name in REPORTED_PACKAGES:
        print(f"  {name:<22} {package_version(name)}")

    missing = missing_required(models.module_available)
    if missing:
        print("Paket wajib belum terpasang: " + ", ".join(missing))
        return EXIT_MISSING_REQUIRED

    calendar = market.load_exchange_calendar(settings.market.calendar)
    print(f"Kalender bursa {settings.market.calendar}: OK ({calendar.name})")

    if os.environ.get("HF_HUB_OFFLINE") != "1":
        print("Catatan: HF_HUB_OFFLINE bukan 1, jadi pemeriksaan ini boleh mengunduh bobot yang belum ada.")
    availability = models.model_availability()
    foundation = [
        key for key, spec in models.MODEL_SPECS.items() if spec.family == "foundation" and availability[key]
    ]
    exit_code = EXIT_OK
    if foundation:
        print("Model foundation (dimuat dari cache, diuji pada deret sintetis):")
        status = check_foundation_models(
            foundation, settings.models.foundation, settings.forecast.horizon,
            settings.forecast.seed, models.load_foundation_runner,
        )
        for key, text in status.items():
            print(f"  {models.MODEL_SPECS[key].label:<14} {text}")
        if any(text != "OK" for text in status.values()):
            exit_code = EXIT_FOUNDATION_FAILED
    else:
        print("Model foundation tidak terpasang (mode ringan): random walk dan gradient boosting saja.")

    print(f"Koneksi Yahoo Finance: {check_yahoo(settings)}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
