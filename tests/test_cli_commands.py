"""Tests for the render, characterize, measure, and figures CLI commands.

The command line is the reproducible entry point to the simulator: every
command writes files that another command, a report, or a figure consumes, so
these tests check the artifacts as well as the exit codes. The contract is the
same as the configuration commands: ``0`` means the command completed, ``2``
means an input or validation failure that a batch script should stop on, and
``1`` means the command ran but its work failed (the gallery scripts are
missing or an example exited non-zero).

The commands compose the public layers rather than reimplementing them. In
particular ``render`` exposes the generated scene once and passes that same RAW
frame to the ISP, so the saved RAW, the rendered PNG, and the layout sidecar
describe one exposure and one sensor grid. ``measure`` then reloads the PNG and
layout from disk, which verifies the IO sidecars and the metrics entry point
together.

Failure artefacts guarded here: a raw file saved next to a rendered image that
silently overwrites its sidecar (the image uses a distinct ``_render`` stem); a
layout whose patch coordinates or reference do not match the image; an
auto-exposure capture whose RAW and image disagree about exposure or gain; a
colorchecker measurement without a colour reference; and a figures command that
runs example scripts from the wrong directory or overwrites committed results.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import camsim.cli as cli_module
from camsim.cli import EXIT_INVALID, EXIT_OK, EXIT_UNREADABLE, main

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Brief: the verbatim render contract
# ---------------------------------------------------------------------------


def test_cli_render_writes_output(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "camsim.cli",
            "render",
            "--scene",
            "colorchecker",
            "--lux",
            "200",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.glob("*.png"))
    assert list(tmp_path.glob("*.npy"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _render(tmp_path: Path, *extra: str) -> tuple[int, Path]:
    """Render a small chart in process and return the exit code and output dir."""
    out = tmp_path / "captures"
    code = main(
        [
            "render",
            "--scene",
            "colorchecker",
            "--lux",
            "2000",
            "--out",
            str(out),
            *extra,
        ]
    )
    return code, out


def _artifacts(out: Path) -> tuple[Path, Path, Path, Path]:
    """Return the raw npy, raw sidecar, rendered PNG, and image sidecar paths."""
    raw_path = next(out.glob("raw_*.npy"))
    raw_sidecar = raw_path.with_suffix(".json")
    image_path = next(out.glob("*_render.png"))
    image_sidecar = image_path.with_suffix(".json")
    return raw_path, raw_sidecar, image_path, image_sidecar


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_cli_render_layout_connects_scene_to_metrics(tmp_path: Path) -> None:
    code, out = _render(tmp_path, "--seed", "3", "--shape", "128x128", "--oversampling", "2")
    assert code == EXIT_OK

    layout = _read_json(out / "layout.json")
    assert layout["name"] == "colorchecker"
    assert len(layout["patches"]) == 24
    assert layout["edge_roi"] is None
    assert layout["reference"]["illuminant_name"] == "D65"
    assert layout["pixel_pitch_um"] == pytest.approx(1.12)
    assert layout["focal_length_mm"] == pytest.approx(6.4)

    # The saved layout must parse back into the metrics value object.
    from camsim.metrics.chart import ChartLayout

    restored = ChartLayout.from_dict(layout)
    assert restored.name == "colorchecker"
    assert len(restored.patches) == 24
    assert restored.reference is not None


def test_cli_render_raw_and_image_share_one_exposure(tmp_path: Path) -> None:
    code, out = _render(tmp_path, "--seed", "4")
    assert code == EXIT_OK

    raw_path, raw_sidecar, image_path, image_sidecar = _artifacts(out)
    assert raw_path.name.startswith("raw_")
    assert image_path.name.startswith("raw_")
    assert image_path.name.endswith("_render.png")

    raw = _read_json(raw_sidecar)
    image = _read_json(image_sidecar)
    frame = raw["frame"]
    assert frame["cfa"] == "RGGB"
    assert frame["bit_depth"] == 10
    assert frame["black_level_dn"] == pytest.approx(64.0)
    assert frame["seed"] == 4
    assert frame["frame_index"] == 0
    assert frame["mode"] in {"LCG", "HCG"}
    assert frame["sensor_config_hash"]

    assert image["channels"] == "RGB"
    assert image["color_space"] == "sRGB"
    assert image["bit_depth"] == 8
    assert image["metadata"]["exposure_s"] == pytest.approx(frame["exposure_s"])
    assert image["metadata"]["pixel_pitch_um"] == pytest.approx(frame["pixel_pitch_um"])

    # The two sidecars are distinct files: the image stem carries ``_render``.
    assert raw_sidecar != image_sidecar


def test_cli_render_writes_a_slanted_edge_layout(tmp_path: Path) -> None:
    out = tmp_path / "edge"
    code = main(
        [
            "render",
            "--scene",
            "slanted_edge",
            "--out",
            str(out),
            "--shape",
            "96x96",
            "--oversampling",
            "2",
        ]
    )
    assert code == EXIT_OK
    layout = _read_json(out / "layout.json")
    assert layout["name"] == "slanted_edge"
    assert layout["edge_roi"]["name"] == "edge"
    assert layout["patches"] == []


def test_cli_render_auto_exposure_uses_the_measured_settings(tmp_path: Path) -> None:
    out = tmp_path / "auto"
    code = main(
        [
            "render",
            "--scene",
            "flat_patch",
            "--lux",
            "20",
            "--out",
            str(out),
            "--shape",
            "64x64",
            "--oversampling",
            "1",
            "--auto-exposure",
            "--seed",
            "5",
        ]
    )
    assert code == EXIT_OK

    _, raw_sidecar, _, image_sidecar = _artifacts(out)
    raw = _read_json(raw_sidecar)
    image = _read_json(image_sidecar)
    frame = raw["frame"]

    # Auto exposure at 20 lux cannot leave the preset defaults untouched.
    assert (
        frame["exposure_s"] != pytest.approx(0.01)
        or frame["analog_gain_db"] > 0.0
        or frame["digital_gain"] > 1.0
        or frame["mode"] == "HCG"
    )
    assert 1e-4 <= frame["exposure_s"] <= 0.1
    assert frame["seed"] == 5
    assert image["metadata"]["exposure_s"] == pytest.approx(frame["exposure_s"])


def test_cli_render_rejects_missing_presets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "render",
            "--sensor-preset",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == EXIT_UNREADABLE
    assert "sensor" in capsys.readouterr().err.lower()


def test_cli_render_rejects_a_malformed_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["render", "--shape", "128", "--out", str(tmp_path / "out")])
    assert code == EXIT_UNREADABLE
    assert "shape" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("shape", ["axb", "0x10"])
def test_cli_render_rejects_bad_shape_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], shape: str
) -> None:
    code = main(["render", "--shape", shape, "--out", str(tmp_path / "out")])
    assert code == EXIT_UNREADABLE
    assert "shape" in capsys.readouterr().err.lower()


def test_cli_render_rejects_a_missing_isp_preset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "render",
            "--isp-preset",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == EXIT_UNREADABLE
    assert "isp" in capsys.readouterr().err.lower()


def test_cli_render_rejects_an_unknown_scene() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["render", "--scene", "not_a_chart"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# characterize
# ---------------------------------------------------------------------------


def test_cli_characterize_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["characterize", "--quick", "--out", str(tmp_path), "--seed", "0"])
    assert code == EXIT_OK

    report_path = tmp_path / "emva_LCG.json"
    assert report_path.is_file()
    payload = _read_json(report_path)
    assert payload["schema_version"]
    assert payload["kind"]
    assert payload["system_gain_dn_per_e"] > 0.0
    assert payload["read_noise_e"] > 0.0
    assert payload["saturation_capacity_e"] > 0.0
    assert payload["provenance"]["mode_resolved"] == "LCG"
    assert payload["provenance"]["frames_per_level"] == 4

    output = capsys.readouterr().out
    assert "read noise" in output
    assert "quick" in output
    assert "emva_LCG.json" in output


def test_cli_characterize_rejects_too_few_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["characterize", "--frames", "1", "--out", str(tmp_path)])
    assert code == EXIT_UNREADABLE
    assert "frames" in capsys.readouterr().err.lower()


def test_cli_characterize_rejects_a_missing_preset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "characterize",
            "--preset",
            str(tmp_path / "missing.json"),
            "--quick",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == EXIT_UNREADABLE
    assert "preset" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


def test_cli_measure_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _render(tmp_path)
    assert code == EXIT_OK
    image_path = next(out.glob("*_render.png"))
    layout_path = out / "layout.json"
    report_path = tmp_path / "metrics.json"

    code = main(
        [
            "measure",
            "--image",
            str(image_path),
            "--layout",
            str(layout_path),
            "--out",
            str(report_path),
        ]
    )
    assert code == EXIT_OK, capsys.readouterr().err

    payload = _read_json(report_path)
    assert payload["layout_id"] == "colorchecker"
    assert payload["snr"] is not None
    assert payload["color"] is not None
    assert payload["mtf"] is None  # a colorchecker layout has no edge ROI
    assert payload["metadata"]["channels"] == "RGB"
    assert "layout:" in capsys.readouterr().out


def test_cli_measure_defaults_to_the_image_stem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _render(tmp_path)
    assert code == EXIT_OK
    image_path = next(out.glob("*_render.png"))

    code = main(["measure", "--image", str(image_path), "--layout", str(out / "layout.json")])
    assert code == EXIT_OK, capsys.readouterr().err
    assert (out / f"{image_path.stem}_metrics.json").is_file()


def test_cli_measure_requires_a_reference_for_a_colorchecker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _render(tmp_path)
    assert code == EXIT_OK
    image_path = next(out.glob("*_render.png"))
    layout_path = out / "layout.json"
    layout = _read_json(layout_path)
    layout["reference"] = None
    layout_path.write_text(json.dumps(layout), encoding="utf-8")

    report_path = tmp_path / "metrics.json"
    code = main(
        [
            "measure",
            "--image",
            str(image_path),
            "--layout",
            str(layout_path),
            "--out",
            str(report_path),
        ]
    )
    assert code == EXIT_UNREADABLE
    assert "reference" in capsys.readouterr().err.lower()
    assert not report_path.exists()


def test_cli_measure_accepts_an_explicit_reference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _render(tmp_path)
    assert code == EXIT_OK
    image_path = next(out.glob("*_render.png"))
    layout_path = out / "layout.json"
    reference_path = tmp_path / "reference.json"
    reference = _read_json(layout_path)["reference"]
    reference_path.write_text(json.dumps(reference), encoding="utf-8")

    report_path = tmp_path / "explicit_reference.json"
    code = main(
        [
            "measure",
            "--image",
            str(image_path),
            "--layout",
            str(layout_path),
            "--reference",
            str(reference_path),
            "--out",
            str(report_path),
        ]
    )
    assert code == EXIT_OK, capsys.readouterr().err
    assert _read_json(report_path)["color"] is not None


def test_cli_measure_reports_mtf_for_a_slanted_edge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "edge"
    code = main(
        [
            "render",
            "--scene",
            "slanted_edge",
            "--lux",
            "200",
            "--out",
            str(out),
            "--shape",
            "96x96",
            "--oversampling",
            "2",
        ]
    )
    assert code == EXIT_OK
    image_path = next(out.glob("*_render.png"))
    report_path = tmp_path / "edge_metrics.json"

    code = main(
        [
            "measure",
            "--image",
            str(image_path),
            "--layout",
            str(out / "layout.json"),
            "--out",
            str(report_path),
        ]
    )
    assert code == EXIT_OK, capsys.readouterr().err

    payload = _read_json(report_path)
    assert payload["mtf"] is not None
    assert payload["acutance"] is not None
    assert payload["snr"] is None
    assert payload["color"] is None
    output = capsys.readouterr().out
    assert "MTF50" in output
    assert "acutance" in output
    assert "colour: n/a" in output


def test_cli_measure_reports_missing_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "measure",
            "--image",
            str(tmp_path / "missing.png"),
            "--layout",
            str(tmp_path / "missing.json"),
        ]
    )
    assert code == EXIT_UNREADABLE
    assert "image" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def test_cli_figures_smoke(tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    reports = tmp_path / "reports"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "camsim.cli",
            "figures",
            "--quick",
            "--out",
            str(figures),
            "--report-dir",
            str(reports),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert list(figures.glob("*.png"))
    assert "01_" in result.stdout
    assert "04_" in result.stdout
    assert (reports / "g3_isp_results_quick.json").is_file()
    assert (reports / "g4_metrics_results_quick.json").is_file()
    assert (reports / "g2_sensor_results_quick.json").is_file()


def test_cli_figures_requires_the_example_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli_module, "REPOSITORY_ROOT", tmp_path)
    code = main(["figures", "--out", str(tmp_path / "figures")])
    assert code == EXIT_INVALID
    assert "examples" in capsys.readouterr().err.lower()


def _install_fake_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    returncodes: list[int],
    calls: list[tuple[list[str], dict[str, Any]]],
) -> None:
    """Replace subprocess.run with a recorder that returns the given codes."""

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        index = len(calls)
        calls.append((list(command), dict(kwargs)))
        code = returncodes[index] if index < len(returncodes) else 0
        return subprocess.CompletedProcess(command, code, "", f"simulated stderr {index + 1}")

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)


def test_cli_figures_runs_the_examples_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    _install_fake_subprocess(monkeypatch, [0, 0, 0, 0], calls)
    figures = tmp_path / "figures"
    reports = tmp_path / "reports"

    code = main(
        [
            "figures",
            "--quick",
            "--out",
            str(figures),
            "--report-dir",
            str(reports),
        ]
    )
    assert code == EXIT_OK
    assert len(calls) == 4
    assert [Path(command[1]).name for command, _ in calls] == [
        "01_spectral_scene_to_raw.py",
        "02_isp_stage_gallery.py",
        "03_measure_iq.py",
        "04_emva_characterization.py",
    ]
    first_command, first_kwargs = calls[0]
    assert first_command[0] == sys.executable
    assert "--report-dir" not in first_command
    assert "--quick" not in first_command
    for command, _ in calls[1:]:
        assert command[-3:] == ["--report-dir", str(reports.resolve()), "--quick"]
    assert first_kwargs["cwd"] == cli_module.REPOSITORY_ROOT
    environment = first_kwargs["env"]
    assert environment["MPLBACKEND"] == "Agg"
    assert environment["CAMSIM_FIG_DIR"] == str(figures.resolve())
    assert figures.is_dir()
    assert reports.is_dir()


def test_cli_figures_stops_on_the_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    _install_fake_subprocess(monkeypatch, [0, 3, 0, 0], calls)

    code = main(["figures", "--out", str(tmp_path / "figures")])
    assert code == EXIT_INVALID
    assert len(calls) == 2
    error = capsys.readouterr().err
    assert "02_isp_stage_gallery.py" in error
    assert "status 3" in error
    assert "simulated stderr 2" in error


def test_cli_figures_reports_a_launch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("cannot start interpreter")

    monkeypatch.setattr(cli_module.subprocess, "run", failing_run)
    code = main(["figures", "--out", str(tmp_path / "figures")])
    assert code == EXIT_INVALID
    assert "cannot run" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Entry point and existing contract
# ---------------------------------------------------------------------------


def test_cli_module_entrypoint_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "camsim.cli", "info"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["name"] == "camsim"


def test_cli_parser_lists_every_command() -> None:
    help_text = cli_module.build_parser().format_help()
    for command in ("validate", "schema", "info", "render", "characterize", "measure", "figures"):
        assert command in help_text
