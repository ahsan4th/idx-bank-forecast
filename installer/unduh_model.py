"""Unduh bobot model foundation ke cache Hugging Face lokal (HF_HOME) saat instalasi.

Pemasang menjalankan skrip ini dengan HF_HOME menunjuk ke folder instalasi. Setelah itu
aplikasi berjalan dengan HF_HUB_OFFLINE=1, jadi bobot tidak diunduh lagi. Hanya model yang
paketnya terpasang yang diunduh, dan id model dibaca dari settings.toml, sumber yang sama
dengan yang dipakai aplikasi.

Kode keluar: 0 semua berhasil, 1 ada model yang gagal, 2 lingkungan belum siap.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecaster import models  # noqa: E402
from forecaster.config import FoundationSettings, load_settings  # noqa: E402

# Field of FoundationSettings holding each foundation model's Hugging Face repo id.
MODEL_ID_FIELDS = {
    "chronos2": "chronos2_model_id",
    "chronos_bolt": "chronos_bolt_model_id",
    "timestral": "timestral_model_id",
    "timesfm": "timesfm_model_id",
}

EXIT_OK, EXIT_SOME_FAILED, EXIT_NOT_READY = 0, 1, 2


def planned_downloads(fs: FoundationSettings, availability: dict[str, bool]) -> list[tuple[str, str]]:
    """(model key, repo id) for every foundation model whose packages are installed."""
    foundation = [key for key, spec in models.MODEL_SPECS.items() if spec.family == "foundation"]
    unmapped = sorted(set(foundation) - set(MODEL_ID_FIELDS))
    if unmapped:
        raise KeyError(f"No model-id field for foundation model(s) {unmapped}; extend MODEL_ID_FIELDS")
    return [(key, getattr(fs, MODEL_ID_FIELDS[key])) for key in foundation if availability[key]]


def download_errors() -> tuple[type[BaseException], ...]:
    """Errors a failed download raises: network, HTTP, gated or missing repository."""
    from huggingface_hub.errors import HfHubHTTPError

    errors: list[type[BaseException]] = [OSError, ValueError, HfHubHTTPError]
    if models.module_available("httpx"):
        import httpx

        errors.append(httpx.HTTPError)
    if models.module_available("requests"):
        import requests

        errors.append(requests.exceptions.RequestException)
    return tuple(errors)


def folder_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def download_all(
    plan: Sequence[tuple[str, str]],
    snapshot_download: Callable[[str], str],
    errors: tuple[type[BaseException], ...],
    out: Callable[[str], None],
) -> list[tuple[str, str]]:
    """Download every repo in `plan`; return (key, message) for each failure.

    A failed model does not stop the others: the app reports a model whose weights are
    missing as an error and leaves it out, so the remaining models stay usable.
    """
    failures: list[tuple[str, str]] = []
    for key, repo_id in plan:
        label = models.MODEL_SPECS[key].label
        out(f"Mengunduh {label} ({repo_id}) ...")
        try:
            path = Path(snapshot_download(repo_id))
        except errors as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((key, message))
            out(f"  GAGAL: {message}")
            continue
        out(f"  OK, {folder_size_mb(path):.0f} MB di {path}")
    return failures


def main() -> int:
    if not os.environ.get("HF_HOME"):
        print("HF_HOME belum diatur. Jalankan skrip ini lewat pemasang.", file=sys.stderr)
        return EXIT_NOT_READY
    settings = load_settings(ROOT / "settings.toml")
    plan = planned_downloads(settings.models.foundation, models.model_availability())
    if not plan:
        print("Tidak ada model foundation yang terpasang; tidak ada bobot yang diunduh.")
        return EXIT_OK
    from huggingface_hub import snapshot_download

    print(f"Cache model: {os.environ['HF_HOME']}")
    failures = download_all(plan, snapshot_download, download_errors(), print)
    print(f"Selesai: {len(plan) - len(failures)} dari {len(plan)} model tersedia tanpa internet.")
    return EXIT_SOME_FAILED if failures else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
