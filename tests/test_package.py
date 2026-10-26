"""Package-level contracts: version metadata, lazy exports, and the CLI skeleton.

The lazy-export check runs in a subprocess so that modules imported by other
tests cannot mask an accidental eager import in ``camsim/__init__.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version

import pytest

import camsim


def test_import_and_version() -> None:
    assert camsim.__version__ == "0.1.0.dev0"
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
