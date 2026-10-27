"""Package-level contracts: version metadata, lazy exports, the CLI, and data.

The lazy-export check runs in a subprocess so that modules imported by other
tests cannot mask an accidental eager import in ``camsim/__init__.py``.

The spectral-data checks pin a packaging contract that is easy to break while
everything still passes from a source checkout: the flagship preset's QE, CFA,
and IR-cut CSVs must live inside the installed ``camsim.data`` package and be
resolved without relying on the repository root or the process working
directory. A wheel built from this tree must therefore carry the CSVs, and a
sensor must construct from the package data alone.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import resources
from importlib.metadata import version
from pathlib import Path

import pytest

import camsim
from camsim.config.io import load_sensor

REPO_ROOT = Path(__file__).resolve().parents[1]
SENSOR_PRESET = (
    REPO_ROOT / "src" / "camsim" / "config" / "presets" / "sensors" / "flagship_phone_50mp.json"
)
PACKAGED_SPECTRA = ("qe_flagship.csv", "cfa_rggb.csv", "ir_cut.csv")


def test_import_and_version() -> None:
    assert camsim.__version__ == "0.1.0"
    assert hasattr(camsim, "__all__")


def test_installed_metadata_matches_package_version() -> None:
    assert version("camsim") == camsim.__version__


def test_unknown_attribute_raises_attribute_error() -> None:
    name = "definitely_not_exported"
    with pytest.raises(AttributeError, match=f"no attribute '{name}'"):
        getattr(camsim, name)


def test_lazy_export_map_resolves_and_caches() -> None:
    camsim._LAZY_EXPORTS["_Probe"] = ("camsim._version", "__version__")
    try:
        assert "_Probe" in dir(camsim)
        assert camsim._Probe == camsim.__version__
        assert "_Probe" in vars(camsim)  # resolved values are cached on the module
    finally:
        camsim._LAZY_EXPORTS.pop("_Probe", None)
        camsim.__dict__.pop("_Probe", None)


def test_imports_are_lazy() -> None:
    code = "\n".join(
        [
            "import sys",
            "import camsim",
            "assert camsim.__version__",
            "for name in (",
            "    'camsim.sensor', 'camsim.isp', 'camsim.scene',",
            "    'camsim.optics', 'camsim.metrics',",
            "):",
            "    assert name not in sys.modules, name",
            "import camsim.metrics",
            "for name in ('camsim.sensor', 'camsim.isp', 'camsim.scene', 'camsim.optics'):",
            "    assert name not in sys.modules, name",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_info_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    from camsim.cli import main

    assert main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "camsim"
    assert payload["version"] == camsim.__version__
    assert "python" in payload


# ---------------------------------------------------------------------------
# Bundled spectral data
# ---------------------------------------------------------------------------


def test_spectral_csvs_ship_inside_the_package() -> None:
    """The sensor CSVs are importable package resources, not repo-root files."""
    spectra = resources.files("camsim.data").joinpath("spectra")
    for name in PACKAGED_SPECTRA:
        resource = spectra.joinpath(name)
        assert resource.is_file(), f"missing packaged spectral data: {name}"
        text = resource.read_text(encoding="utf-8")
        assert "wavelength_nm" in text
        assert "camsim" in text.lower()


def test_preset_uses_package_relative_spectral_paths() -> None:
    """The preset names curves inside ``camsim.data`` rather than the repo tree."""
    config = load_sensor(SENSOR_PRESET)
    for field_name in ("qe_csv", "cfa_csv", "ir_cut_csv"):
        value = getattr(config.spectral, field_name)
        assert value.startswith("spectra/"), (field_name, value)


def test_sensor_constructs_from_package_data_in_a_subprocess(tmp_path: Path) -> None:
    """A sensor builds from an installed-style package layout outside the repo.

    Running from ``tmp_path`` proves the resolver does not depend on the
    working directory and that the preset itself is a package resource. This is
    the executable form of the wheel check: the same files are what a wheel
    installs under ``site-packages/camsim``.
    """
    code = "\n".join(
        [
            "import numpy as np",
            "import camsim",
            "from importlib import resources",
            "from camsim.config.io import load_sensor",
            "from camsim.core.spectral import SpectralIrradiance",
            "preset = resources.files('camsim.config').joinpath(",
            "    'presets/sensors/flagship_phone_50mp.json')",
            "config = load_sensor(str(preset))",
            "sensor = camsim.Sensor(config, seed=1)",
            "wavelengths = np.arange(360.0, 831.0, 5.0)",
            "data = np.full((8, 8, wavelengths.size), 1e-5, dtype=np.float64)",
            "cube = SpectralIrradiance.from_array(",
            "    data, wavelengths, 'W/m^2/nm', pixel_pitch_um=1.12)",
            "raw = sensor.expose(cube, 0.01, mode='LCG', seed=1)",
            "assert raw.data.shape == (8, 8)",
            "assert raw.data.dtype == np.uint16",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
