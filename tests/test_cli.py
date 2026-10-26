"""Tests for the ``camsim`` command-line skeleton.

The CLI is the first contact most users have with the configuration layer, so
its exit codes are a contract: ``0`` means the file validated (warnings are
still reported), ``1`` means the file was read but violates the schema, and
``2`` means the input could not be read at all. Scripts and CI jobs branch on
these codes, so they are asserted explicitly rather than inferred from printed
text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from camsim import __version__
from camsim.cli import EXIT_INVALID, EXIT_OK, EXIT_UNREADABLE, main

REPO_ROOT = Path(__file__).resolve().parents[1]
SENSOR_PRESET = (
    REPO_ROOT / "src" / "camsim" / "config" / "presets" / "sensors" / "flagship_phone_50mp.json"
)
ISP_PRESET = REPO_ROOT / "src" / "camsim" / "config" / "presets" / "isp" / "default.json"


def _preset_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(tmp_path: Path, payload: dict[str, Any], name: str = "config.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_validate_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(SENSOR_PRESET)]) == EXIT_OK
    assert "valid" in capsys.readouterr().out

    bad = _preset_payload(SENSOR_PRESET)
    bad["unknown_field"] = 1
    bad_path = _write_json(tmp_path, bad, "bad.json")
    assert main(["validate", str(bad_path)]) == EXIT_INVALID
    result = capsys.readouterr()
    assert "invalid" in result.out
    assert "error" in result.out

    assert main(["validate", str(tmp_path / "missing.json")]) == EXIT_UNREADABLE
    assert "cannot read" in capsys.readouterr().err


def test_cli_validate_reports_warnings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data["adc"]["full_scale_uv"] = 100.0
    path = _write_json(tmp_path, data, "adc_limited.json")
    assert main(["validate", str(path)]) == EXIT_OK
    output = capsys.readouterr().out
    assert "warning" in output
    assert "ADC" in output


def test_cli_validate_malformed_json_is_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert main(["validate", str(path)]) == EXIT_INVALID
    assert "JSON" in capsys.readouterr().out


def test_cli_schema_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema", "sensor"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["title"] == "SensorConfig"
    assert "pixel" in payload["properties"]
    assert "modes" in payload["properties"]

    assert main(["schema", "isp"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["title"] == "ISPConfig"
    assert "dpc" in payload["properties"]
    assert "output" in payload["properties"]


def test_cli_schema_rejects_unknown_kind() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["schema", "lens"])
    assert excinfo.value.code == 2


def test_cli_info_lists_presets(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "camsim"
    assert payload["version"] == __version__
    assert payload["python"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}")
    assert "sensors/flagship_phone_50mp.json" in payload["presets"]["sensors"]
    assert "isp/default.json" in payload["presets"]["isp"]
