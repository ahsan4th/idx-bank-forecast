"""Installer helpers (installer/*.py) and the Windows scripts they belong with.

The Windows scripts cannot run here, so these tests check what can be checked off
Windows: the Python helpers' logic with fakes, and the scripts' encoding, line endings
and the files they reference.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import fields
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from forecaster import models
from forecaster.config import FoundationSettings, load_settings
from tests.conftest import ROOT

WINDOWS_SCRIPTS = [ROOT / "pasang_windows.bat", *sorted((ROOT / "installer" / "windows").glob("*.*"))]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "installer" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


unduh = _load("unduh_model")
cek = _load("cek_instalasi")
FOUNDATION = [k for k, spec in models.MODEL_SPECS.items() if spec.family == "foundation"]


@pytest.fixture(scope="module")
def foundation_settings() -> FoundationSettings:
    return load_settings(ROOT / "settings.toml").models.foundation


def test_every_foundation_model_has_a_settings_field_for_its_repo_id():
    assert set(unduh.MODEL_ID_FIELDS) == set(FOUNDATION)
    field_names = {f.name for f in fields(FoundationSettings)}
    assert set(unduh.MODEL_ID_FIELDS.values()) <= field_names


def test_planned_downloads_reads_repo_ids_from_settings(foundation_settings):
    plan = unduh.planned_downloads(foundation_settings, {k: True for k in models.MODEL_SPECS})
    assert [key for key, _ in plan] == FOUNDATION
    assert dict(plan)["chronos2"] == foundation_settings.chronos2_model_id
    assert dict(plan)["timesfm"] == foundation_settings.timesfm_model_id


def test_planned_downloads_skips_models_without_packages(foundation_settings):
    availability = {k: k == "chronos2" for k in models.MODEL_SPECS}
    assert unduh.planned_downloads(foundation_settings, availability) == [
        ("chronos2", foundation_settings.chronos2_model_id)
    ]


def test_download_all_reports_a_failure_and_continues(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 2_000_000)
    calls, lines = [], []

    def fake_snapshot(repo_id: str) -> str:
        calls.append(repo_id)
        if repo_id == "org/broken":
            raise OSError("connection reset")
        return str(tmp_path)

    plan = [("chronos2", "org/ok"), ("timestral", "org/broken"), ("timesfm", "org/also-ok")]
    failures = unduh.download_all(plan, fake_snapshot, (OSError,), lines.append)
    assert calls == ["org/ok", "org/broken", "org/also-ok"]
    assert failures == [("timestral", "OSError: connection reset")]
    assert any("GAGAL" in line for line in lines)
    assert any("OK, 2 MB" in line for line in lines)


def test_download_all_lets_unexpected_errors_crash(tmp_path):
    def buggy_snapshot(repo_id: str) -> str:
        raise AttributeError("API changed")

    with pytest.raises(AttributeError):
        unduh.download_all([("chronos2", "org/x")], buggy_snapshot, (OSError,), lambda line: None)


def test_synthetic_contexts_are_positive_and_reproducible():
    a = cek.synthetic_contexts(20261119, 512, 2)
    b = cek.synthetic_contexts(20261119, 512, 2)
    assert a.shape == (2, 512) and a.dtype == np.float32
    assert np.all(a > 0) and np.all(np.isfinite(a))
    np.testing.assert_array_equal(a, b)


class _Runner:
    def __init__(self, behaviour: str):
        self.behaviour = behaviour

    def predict(self, contexts, horizon, covariates, covariate_names):
        assert covariates is None and covariate_names == ()
        n = len(contexts)
        if self.behaviour == "raise":
            raise RuntimeError("weights missing from cache")
        width = horizon + 1 if self.behaviour == "wrong_shape" else horizon
        median = np.full((n, width), 9000.0)
        return models.ForecastBatch(median, median * 0.98, median * 1.02)


def test_check_foundation_models_reports_each_model(foundation_settings):
    behaviour = {"chronos2": "ok", "timestral": "raise", "timesfm": "wrong_shape"}
    status = cek.check_foundation_models(
        list(behaviour), foundation_settings, 5, 1, lambda key, fs: _Runner(behaviour[key])
    )
    assert status["chronos2"] == "OK"
    assert status["timestral"] == "GAGAL: RuntimeError: weights missing from cache"
    assert status["timesfm"].startswith("GAGAL: bentuk keluaran (2, 6)")


def test_missing_required_lists_absent_modules():
    assert cek.missing_required(lambda name: name != "yfinance") == ["yfinance"]
    assert cek.missing_required(lambda name: True) == []


@pytest.mark.parametrize("path", WINDOWS_SCRIPTS, ids=lambda p: p.name)
def test_windows_scripts_are_ascii_with_crlf_line_endings(path: Path):
    raw = path.read_bytes()
    raw.decode("ascii")  # Windows PowerShell 5.1 misreads non-ASCII in BOM-less UTF-8 scripts
    assert raw.count(b"\n") == raw.count(b"\r\n"), "every line must end with CRLF for cmd.exe"


def test_installer_references_files_that_exist():
    script = (ROOT / "installer" / "windows" / "pasang.ps1").read_text(encoding="ascii")
    helpers = sorted(set(re.findall(r"installer\\([a-z_]+\.py)", script)))
    assert helpers == ["cek_instalasi.py", "unduh_model.py"]
    for helper in helpers:
        assert (ROOT / "installer" / helper).is_file(), helper
    copied = re.findall(r"'([^']+)'", re.search(r"\$Scripts\s*=\s*@\(([^)]*)\)", script).group(1))
    assert copied == ["jalankan.bat", "hapus.bat", "hapus.ps1"]
    for name in copied:
        assert (ROOT / "installer" / "windows" / name).is_file(), name
    assert (ROOT / "requirements.txt").is_file() and (ROOT / "requirements-lite.txt").is_file()


def test_install_and_uninstall_never_wait_for_typing():
    for name in ("pasang.ps1", "hapus.ps1"):
        script = (ROOT / "installer" / "windows" / name).read_text(encoding="ascii")
        assert "Read-Host" not in script, name
    launcher = (ROOT / "pasang_windows.bat").read_text(encoding="ascii")
    success_path, failure_path = launcher.split(":failed", 1)
    assert "pause" not in success_path.lower()
    assert "pause" in failure_path.lower()  # an error stays on screen until it has been read


def test_install_ends_by_opening_the_app():
    script = (ROOT / "installer" / "windows" / "pasang.ps1").read_text(encoding="ascii")
    main = script[script.index("Start-Transcript"):]
    # The mark of the web is removed before anything is launched from the installed files.
    assert main.index("Unblock-InstalledFiles") < main.index("Start-Process -FilePath $Launcher")


def test_launcher_downloads_this_repository_when_run_on_its_own():
    launcher = (ROOT / "pasang_windows.bat").read_text(encoding="ascii")
    assert r"%~dp0installer\windows\pasang.ps1" in launcher
    assert "https://github.com/ahsan4th/idx-bank-forecast/archive/refs/heads/main.zip" in launcher
    assert r"idx-bank-forecast-main\installer\windows\pasang.ps1" in launcher


def test_timestral_pin_is_the_same_commit_everywhere():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert f"timestral/archive/{models.TIMESTRAL_COMMIT}.zip" in requirements
    assert models.TIMESTRAL_COMMIT in models.MODEL_SPECS["timestral"].install_hint
    assert "git+" not in requirements  # Windows PCs often have no git
