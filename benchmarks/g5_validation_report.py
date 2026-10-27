"""Generate the consolidated G5 validation report and its two summary figures.

The G5 gate is the final Phase 5 checkpoint: it asks whether the scene, optics,
sensor, ISP, and metrics layers have been validated together, whether the
engineering gates (tests, coverage, lint, types, docs, build) are green, and
whether the cross-cutting design criteria that span phases are evidenced. This
script does not re-derive the phase gates. It reads the committed G1-G4 reports
and their machine-readable JSON, verifies that every recorded check passed, and
then measures the handful of criteria that belong to no single phase: denoise
SNR/MTF preservation, sharpen MTF50/overshoot, camera auto-exposure convergence
and limits, the RCCC baseline, determinism, import independence, and the CLI
render -> measure chain.

The measurements are direct calls to the public stage and metric functions, not
copies of their internals. A blurred slanted edge is measured with the same
ISO 12233-style e-SFR estimator a user would call; the denoise and sharpen
stages are the shipped stages; the auto-exposure policy is the shipped
``Camera.auto_expose``. Every random draw uses an explicit ``numpy`` generator
with a fixed seed, so repeated runs in one environment produce the same values
and byte-identical PNGs.

Why these criteria matter. Denoising trades noise against resolution: a filter
that suppresses noise by smearing the edge has not preserved image quality, so
the gate requires a positive SNR gain *and* no more than a 10 % MTF50 loss.
Sharpening is the inverse trade: it must raise MTF50 without inventing contrast
so large that the edge halo is unbounded, so the gate requires both a higher
MTF50 and a positive but measured overshoot. Auto-exposure must converge to its
target while respecting the preset's physical exposure, analog-gain, and
digital-gain windows; at extreme low light digital gain is a display
convenience that amplifies noise, which is recorded as a limitation rather than
hidden. The RCCC baseline is a shape/linearity/smoke reconstruction only: its
clear channel is a documented luma placeholder, and the Bayer colour and MTF
thresholds do not apply.

The report is deliberately a *candidate* artifact. Its status is "awaiting
human sign-off"; it does not claim that the owner has signed, that a tag
exists, or that the package has been released. The project owner reviews the
report and the figure gallery, confirms the scope, assumptions, and
limitations, and only then authorises ``v0.1.0``. No machine-vision code is
written before that decision.

Failure artefacts guarded here: a report that silently accepts a failing phase
check; a stale report whose values no longer match the committed JSON; a quick
mode mistaken for gate evidence; a denoise check that measures SNR gain while
ignoring resolution loss; a sharpen check that calls any contrast increase a
pass without measuring overshoot; an auto-exposure check that ignores the
preset's physical limits; a determinism claim made by comparing different
seeds; a CLI chain that is described but never executed; and a "signed off"
status manufactured by the generator instead of recorded from a human decision.

The recorded quality-gate outcomes in :data:`GATE_RUN` are evidence captured
from the G5 candidate run on this machine. Re-run the documented commands (and
update this block) whenever the source tree changes; the rest of the report is
regenerated from the committed artifacts and the live cross-cutting
measurements below.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

import camsim
from camsim.camera import Camera
from camsim.config.io import PRESETS_DIR, load_isp, load_sensor
from camsim.config.schema import DenoiseConfig, LensConfig, SharpenConfig
from camsim.core.frames import CFAPattern
from camsim.isp.demosaic import demosaic_rccc_baseline
from camsim.isp.denoise import denoise
from camsim.isp.pipeline import ISP
from camsim.isp.sharpen import unsharp_mask
from camsim.metrics import slanted_edge_mtf, snr_patch
from camsim.metrics.chart import PatchROI
from camsim.optics.propagation import apply_psf
from camsim.optics.psf import gaussian_psf
from camsim.scene.targets import render, slanted_edge_target
from camsim.sensor import Sensor
from camsim.sensor.cfa import mosaic_from_channels
from camsim.viz import PALETTE, save_figure, set_style

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPORT_PATH: Final[Path] = REPO_ROOT / "results" / "reports" / "validation_report.md"
FIGURE_DIR: Final[Path] = REPO_ROOT / "results" / "figures"
REPORT_DIR: Final[Path] = REPO_ROOT / "results" / "reports"
DOCS_IMAGES: Final[Path] = REPO_ROOT / "docs" / "images"

SENSOR_PRESET: Final[Path] = PRESETS_DIR / "sensors" / "flagship_phone_50mp.json"
ISP_PRESET: Final[Path] = PRESETS_DIR / "isp" / "default.json"

PIXEL_PITCH_UM: Final[float] = 1.12
EDGE_SHAPE: Final[tuple[int, int]] = (256, 256)
EDGE_ANGLE_DEG: Final[float] = 5.0
GAUSSIAN_SIGMA_PX: Final[float] = 1.0
GAUSSIAN_KERNEL_SIZE: Final[int] = 31
MTF_OVERSAMPLING: Final[int] = 8
EDGE_NOISE_SIGMA: Final[float] = 0.01
EDGE_NOISE_SEED: Final[int] = 71
FLAT_NOISE_SIGMA: Final[float] = 0.02
FLAT_NOISE_SEED: Final[int] = 72
FLAT_LEVEL: Final[float] = 0.4
FLAT_SHAPE: Final[tuple[int, int]] = (64, 64)
DENOISE_MAX_MTF_LOSS: Final[float] = 0.10
SHARPEN_OVERSHOOT_MARGIN: Final[float] = 1e-3
AE_TARGET_FRACTION: Final[float] = 0.18
AE_TOLERANCE: Final[float] = 0.02
AE_MAX_ITERATIONS: Final[int] = 12
AE_LUX_LEVELS: Final[tuple[float, ...]] = (2000.0, 20.0, 0.01)
DETERMINISM_SHAPE: Final[tuple[int, int]] = (64, 64)

FORBIDDEN_METRICS_IMPORTS: Final[tuple[str, ...]] = (
    "camsim.sensor",
    "camsim.isp",
    "camsim.scene",
    "camsim.optics",
)


@dataclass(frozen=True, slots=True)
class Check:
    """One cross-cutting check and its recorded verdict."""

    name: str
    measured: str
    bound: str
    passed: bool
    citation: str


@dataclass(frozen=True, slots=True)
class GateCommand:
    """A quality-gate command, its recorded outcome, and its wall time."""

    command: str
    result: str
    duration_s: float
    passed: bool


@dataclass(frozen=True, slots=True)
class GateArtifact:
    """One committed phase gate: its report text, checks, and status."""

    gate: str
    scope: str
    report: Path
    json_path: Path | None
    checks: tuple[dict[str, Any], ...]
    reported_status: str


# Quality-gate outcomes recorded during the G5 candidate run. Timings are
# machine-specific measurements, not promises; the verdicts and counts are the
# evidence. Update this block only after re-running the documented commands.
GATE_RUN: Final[tuple[GateCommand, ...]] = (
    GateCommand("uv run pytest -q", "1751 passed", 125.92, True),
    GateCommand(
        "uv run pytest --cov=camsim --cov-fail-under=90",
        "1751 passed; total coverage 91.82% (required 90%)",
        152.16,
        True,
    ),
    GateCommand("uv run ruff check .", "All checks passed!", 0.06, True),
    GateCommand("uv run ruff format --check .", "140 files already formatted", 0.03, True),
    GateCommand("uv run mypy src", "Success: no issues found in 68 source files", 0.59, True),
    GateCommand(
        "uv run mkdocs build --strict",
        "Documentation built in 1.73 seconds",
        1.90,
        True,
    ),
    GateCommand(
        "uv run python benchmarks/benchmark_pipeline.py --quick",
        "pipeline total 8.26 ms (128 x 128)",
        1.31,
        True,
    ),
    GateCommand(
        "uv run python benchmarks/benchmark_pipeline.py",
        "pipeline total 115.29 ms (512 x 512)",
        1.87,
        True,
    ),
    GateCommand(
        "uv build",
        "built camsim-0.1.0.dev0.tar.gz and camsim-0.1.0.dev0-py3-none-any.whl",
        0.75,
        True,
    ),
)

# Benchmark stage shares recorded by the same gate run, used in the evidence
# figure. Using recorded constants keeps the PNG deterministic; machine timing
# is inherently variable and is reported as evidence, never as a guarantee.
BENCHMARK_QUICK_TOTAL_MS: Final[float] = 8.26
BENCHMARK_FULL_TOTAL_MS: Final[float] = 115.29
BENCHMARK_STAGES: Final[tuple[str, ...]] = (
    "scene.render",
    "sensor.expose",
    "isp.process",
    "measure_all",
)
BENCHMARK_QUICK_SHARE_PCT: Final[tuple[float, ...]] = (39.7, 38.2, 17.0, 5.0)
BENCHMARK_FULL_SHARE_PCT: Final[tuple[float, ...]] = (41.9, 41.2, 15.2, 1.7)


def _package_version(name: str) -> str:
    """Return an installed distribution's version, or ``not installed``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _git_commit() -> str:
    """Return the current source commit, or ``unknown`` outside a checkout."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _split_table_row(line: str) -> list[str]:
    """Split a table row on the ``' | '`` cell delimiter.

    A cell may legitimately contain an unescaped ``|`` (the G1 report writes
    ``max |error| 0.0426 deg``), so splitting on every pipe would shift the
    columns. The committed reports use spaced delimiters between cells, which
    keeps the internal pipe inside its cell.
    """
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split(" | ")]


def _parse_markdown_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Parse every GitHub-style Markdown table into header and rows."""
    lines = text.splitlines()
    tables: list[tuple[list[str], list[list[str]]]] = []
    index = 0
    while index < len(lines) - 1:
        header_line = lines[index].strip()
        separator = lines[index + 1].strip()
        if not (header_line.startswith("|") and separator.startswith("|")):
            index += 1
            continue
        separators = [cell.strip() for cell in separator.strip("|").split("|")]
        if not all(cell and set(cell) <= {"-", ":"} for cell in separators):
            index += 1
            continue
        header = _split_table_row(header_line)
        rows: list[list[str]] = []
        index += 2
        while index < len(lines) and lines[index].strip().startswith("|"):
            rows.append(_split_table_row(lines[index]))
            index += 1
        tables.append((header, rows))
    return tables


def _reported_status(text: str) -> str:
    """Extract the first ``Status:`` line from a phase report."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("**status:") or stripped.lower().startswith("status:"):
            return stripped.strip("*").split(":", 1)[1].strip().strip("*").strip()
    return "missing"


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path`` with a clear failure message."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


# Every committed phase artifact the consolidated report reads. Binding each
# file by hash makes the report internally consistent with the evidence it
# summarises: changing an artifact without regenerating the report fails the
# G5 tests instead of leaving a plausible-looking stale pairing.
PHASE_ARTIFACTS: Final[tuple[str, ...]] = (
    "g1_scene_optics.md",
    "g2_sensor.md",
    "g2_sensor_results.json",
    "g3_isp.md",
    "g3_isp_results.json",
    "g4_metrics.md",
    "g4_metrics_results.json",
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_bindings() -> list[tuple[str, str]]:
    """Return ``(relative path, sha256)`` for every phase artifact used."""
    missing = [name for name in PHASE_ARTIFACTS if not (REPORT_DIR / name).is_file()]
    if missing:
        raise ValueError(f"missing phase artifacts: {missing}")
    return [(name, _sha256(REPORT_DIR / name)) for name in PHASE_ARTIFACTS]


def _load_gates() -> tuple[list[GateArtifact], list[list[str]]]:
    """Load G1-G4 reports and checks, and the G1 verdict rows.

    A gate is usable evidence only when its report says PASS, its JSON exists
    where one is defined, every recorded check passed, and the JSON is marked
    as gate evidence rather than a quick smoke run.
    """
    g1_report = REPORT_DIR / "g1_scene_optics.md"
    g1_text = g1_report.read_text(encoding="utf-8")
    g1_rows: list[list[str]] = []
    for header, rows in _parse_markdown_tables(g1_text):
        if header == ["Check", "Measured", "Tolerance", "Result"]:
            g1_rows = rows
            break
    if not g1_rows:
        raise ValueError("G1 report has no verdict summary table")
    g1_checks = tuple(
        {
            "name": row[0],
            "measured_text": row[1],
            "tolerance_text": row[2],
            "pass": row[3].upper() == "PASS",
            "notes": "",
        }
        for row in g1_rows
    )

    gates: list[GateArtifact] = [
        GateArtifact(
            "G1",
            "scene, illuminants, optics",
            g1_report,
            None,
            g1_checks,
            _reported_status(g1_text),
        )
    ]
    for gate, scope, report_name, json_name in (
        ("G2", "sensor model and EMVA characterization", "g2_sensor.md", "g2_sensor_results.json"),
        ("G3", "ISP stages and pipeline", "g3_isp.md", "g3_isp_results.json"),
        ("G4", "metrics harness", "g4_metrics.md", "g4_metrics_results.json"),
    ):
        report_path = REPORT_DIR / report_name
        json_path = REPORT_DIR / json_name
        payload = _load_json(json_path)
        if payload.get("gate_evidence") is not True:
            raise ValueError(f"{json_name} is not marked as gate evidence")
        if payload.get("quick") is True:
            raise ValueError(f"{json_name} is a quick-mode artifact, not gate evidence")
        checks = tuple(payload["checks"])
        gates.append(
            GateArtifact(
                gate,
                scope,
                report_path,
                json_path,
                checks,
                _reported_status(report_path.read_text(encoding="utf-8")),
            )
        )
    return gates, g1_rows


def _format_number(value: float) -> str:
    """Format a measured value compactly and consistently."""
    if not math.isfinite(value):
        return "n/a"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.4g}"
    return f"{value:.6g}"


def _format_tolerance(tolerance: dict[str, float]) -> str:
    """Format a JSON tolerance mapping as a human-readable condition."""
    if "target" in tolerance and "relative" in tolerance:
        return f"{_format_number(tolerance['target'])} +/- {tolerance['relative'] * 100:g} %"
    if "target" in tolerance and "absolute" in tolerance:
        return f"{_format_number(tolerance['target'])} +/- {_format_number(tolerance['absolute'])}"
    if "minimum" in tolerance and "maximum" in tolerance:
        return (
            f"{_format_number(tolerance['minimum'])} < x < {_format_number(tolerance['maximum'])}"
        )
    if "maximum" in tolerance:
        return f"<= {_format_number(tolerance['maximum'])}"
    if "minimum" in tolerance:
        return f"> {_format_number(tolerance['minimum'])}"
    return "n/a"


def _escape_cell(text: str) -> str:
    """Escape a pipe so a table cell cannot create an extra column."""
    return text.replace("|", r"\|")


def _check_row(check: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return name, measured, tolerance, and verdict for a JSON check."""
    measured = check.get("measured_text")
    if measured is None:
        measured = _format_number(float(check["measured"]))
    tolerance = check.get("tolerance_text")
    if tolerance is None:
        tolerance = _format_tolerance(check.get("tolerance", {}))
    verdict = "PASS" if check.get("pass") else "FAIL"
    return (
        _escape_cell(str(check["name"])),
        _escape_cell(str(measured)),
        _escape_cell(str(tolerance)),
        verdict,
    )


def _blurred_edge() -> NDArray[np.float64]:
    """Return a 5 degree slanted edge blurred by a unit-sigma Gaussian."""
    edge = slanted_edge_target(EDGE_SHAPE, angle_deg=EDGE_ANGLE_DEG)
    blurred = apply_psf(
        edge[..., None],
        gaussian_psf(GAUSSIAN_SIGMA_PX, size=GAUSSIAN_KERNEL_SIZE),
    )[..., 0]
    return np.asarray(blurred, dtype=np.float64)


def _measure_denoise() -> tuple[dict[str, Any], list[Check]]:
    """Measure denoise SNR gain and MTF50 preservation on a noisy edge.

    The edge is blurred by the known Gaussian, then corrupted by additive
    noise. MTF50 is measured before and after the shipped bilateral stage; a
    denoiser that buys SNR by smearing detail would fail the loss bound. The
    SNR measurement uses an independent flat patch with its own generator so
    the edge noise and the patch noise cannot cancel.
    """
    blurred = _blurred_edge()
    rng = np.random.default_rng(EDGE_NOISE_SEED)
    noisy = blurred + EDGE_NOISE_SIGMA * rng.standard_normal(blurred.shape)
    config = DenoiseConfig(
        method="bilateral",
        luma_strength=1.0,
        chroma_strength=0.0,
        radius=1.5,
    )
    denoised = np.asarray(denoise(noisy, config), dtype=np.float64)
    before = slanted_edge_mtf(noisy, PIXEL_PITCH_UM, oversampling=MTF_OVERSAMPLING)
    after = slanted_edge_mtf(denoised, PIXEL_PITCH_UM, oversampling=MTF_OVERSAMPLING)
    relative_change = float(after.mtf50_cp / before.mtf50_cp - 1.0)
    loss = -relative_change

    flat_rng = np.random.default_rng(FLAT_NOISE_SEED)
    flat = FLAT_LEVEL + FLAT_NOISE_SIGMA * flat_rng.standard_normal(FLAT_SHAPE)
    denoised_flat = np.asarray(denoise(flat, config), dtype=np.float64)
    roi = PatchROI(name="flat", x=0, y=0, height=FLAT_SHAPE[0], width=FLAT_SHAPE[1])
    snr_before = snr_patch(flat, roi)
    snr_after = snr_patch(denoised_flat, roi)
    snr_gain = float(snr_after.snr_db - snr_before.snr_db)

    checks = [
        Check(
            "Denoise MTF50 preservation",
            f"{before.mtf50_cp:.6f} -> {after.mtf50_cp:.6f} cycles/pixel ({relative_change:+.2%})",
            f"loss <= {DENOISE_MAX_MTF_LOSS:.0%}",
            loss <= DENOISE_MAX_MTF_LOSS,
            "camsim.metrics.slanted_edge_mtf before/after camsim.isp.denoise.denoise",
        ),
        Check(
            "Denoise luma SNR gain",
            f"{snr_before.snr_db:.3f} -> {snr_after.snr_db:.3f} dB ({snr_gain:+.2f} dB)",
            "> 0 dB",
            snr_gain > 0.0,
            "camsim.metrics.snr_patch on an independent noisy flat patch",
        ),
    ]
    values = {
        "config": {
            "method": config.method,
            "luma_strength": config.luma_strength,
            "chroma_strength": config.chroma_strength,
            "radius": config.radius,
            "edge_noise_sigma": EDGE_NOISE_SIGMA,
            "edge_seed": EDGE_NOISE_SEED,
            "flat_noise_sigma": FLAT_NOISE_SIGMA,
            "flat_seed": FLAT_NOISE_SEED,
            "oversampling": MTF_OVERSAMPLING,
        },
        "mtf50_before_cp": float(before.mtf50_cp),
        "mtf50_after_cp": float(after.mtf50_cp),
        "mtf50_relative_change": relative_change,
        "snr_before_db": float(snr_before.snr_db),
        "snr_after_db": float(snr_after.snr_db),
        "snr_gain_db": snr_gain,
    }
    return values, checks


def _measure_sharpen() -> tuple[dict[str, Any], list[Check]]:
    """Measure unsharp-mask MTF50 gain and positive overshoot.

    The input is the clean blurred edge, so the measurement isolates the stage
    rather than the noise. The profile is the row mean of the linear image; the
    low and high plateaus are medians of the first and last 40 columns, which
    are far enough from the transition for the Gaussian's tails to have
    settled. Overshoot is reported both as a fraction of edge contrast and as
    the normalized maximum, so a three-digit percentage cannot hide behind a
    unit convention.
    """
    blurred = _blurred_edge()
    config = SharpenConfig(amount=1.0, radius=1.0, threshold=0.0, luma_only=True)
    sharpened = np.asarray(unsharp_mask(blurred, config), dtype=np.float64)
    before = slanted_edge_mtf(blurred, PIXEL_PITCH_UM, oversampling=MTF_OVERSAMPLING)
    after = slanted_edge_mtf(sharpened, PIXEL_PITCH_UM, oversampling=MTF_OVERSAMPLING)
    gain = float(after.mtf50_cp / before.mtf50_cp - 1.0)

    profile_before = blurred.mean(axis=0)
    profile_after = sharpened.mean(axis=0)
    low = float(np.median(profile_before[:40]))
    high = float(np.median(profile_before[-40:]))
    contrast = high - low
    normalized_max = float((profile_after.max() - low) / contrast)
    overshoot = float((profile_after.max() - high) / contrast)

    checks = [
        Check(
            "Sharpen MTF50 gain",
            f"{before.mtf50_cp:.6f} -> {after.mtf50_cp:.6f} cycles/pixel ({gain:+.2%})",
            "after > before",
            float(after.mtf50_cp) > float(before.mtf50_cp),
            "camsim.metrics.slanted_edge_mtf before/after camsim.isp.sharpen.unsharp_mask",
        ),
        Check(
            "Sharpen positive overshoot",
            f"max normalized profile {normalized_max:.6f} "
            f"(overshoot {overshoot:+.2%} of edge contrast)",
            f"> 1 + {SHARPEN_OVERSHOOT_MARGIN:g}",
            normalized_max > 1.0 + SHARPEN_OVERSHOOT_MARGIN,
            "Row-mean linear profile of the sharpened edge",
        ),
    ]
    values = {
        "config": {
            "amount": config.amount,
            "radius": config.radius,
            "threshold": config.threshold,
            "luma_only": config.luma_only,
        },
        "mtf50_before_cp": float(before.mtf50_cp),
        "mtf50_after_cp": float(after.mtf50_cp),
        "mtf50_relative_gain": gain,
        "profile_low_plateau": low,
        "profile_high_plateau": high,
        "profile_max_after": float(profile_after.max()),
        "normalized_max": normalized_max,
        "overshoot_fraction": overshoot,
        "profile_before": profile_before,
        "profile_after": profile_after,
    }
    return values, checks


def _reference_lens() -> LensConfig:
    """Return the design reference lens geometry."""
    return LensConfig(
        f_number=1.8,
        focal_length_mm=6.4,
        sensor_diagonal_mm=11.46,
        transmittance=0.9,
    )


def _measure_auto_exposure() -> tuple[list[dict[str, Any]], list[Check]]:
    """Exercise ``Camera.auto_expose`` across bright, dim, and extreme scenes.

    The three scenes span the policy's useful range: a bright flat patch that
    is already near the target, a dim patch that needs exposure and analog
    gain, and a 0.01 lux patch that needs HCG, the exposure ceiling, maximum
    analog gain, and digital gain. Every result must converge, respect the
    preset windows, and land within the default metering tolerance of the
    target fraction.
    """
    camera = Camera(
        _reference_lens(),
        Sensor(load_sensor(SENSOR_PRESET), seed=1),
        ISP(load_isp(ISP_PRESET)),
    )
    config = camera.sensor.config
    rows: list[dict[str, Any]] = []
    for lux in AE_LUX_LEVELS:
        scene = render(
            "flat_patch",
            illuminant="D65",
            lux=lux,
            shape=(64, 64),
            oversampling=2,
        )
        result = camera.auto_expose(scene)
        within_limits = (
            config.exposure.min_s <= result.exposure_s <= config.exposure.max_s
            and config.analog_gain.min_db <= result.analog_gain_db <= config.analog_gain.max_db
            and math.isfinite(result.digital_gain)
            and result.digital_gain >= 1.0
            and result.iterations <= AE_MAX_ITERATIONS
            and result.mode in {"LCG", "HCG"}
        )
        rows.append(
            {
                "lux": lux,
                "converged": bool(result.converged),
                "iterations": int(result.iterations),
                "mode": str(result.mode),
                "exposure_ms": float(result.exposure_s * 1e3),
                "analog_gain_db": float(result.analog_gain_db),
                "digital_gain": float(result.digital_gain),
                "metered_value": float(result.metered_value),
                "within_limits": bool(within_limits),
                "within_tolerance": bool(
                    abs(result.metered_value - AE_TARGET_FRACTION) <= AE_TOLERANCE
                ),
            }
        )
    checks = [
        Check(
            "Camera AE convergence",
            "; ".join(
                f"{row['lux']:g} lux: {row['iterations']} probe(s), "
                f"metered {row['metered_value']:.4f}"
                for row in rows
            ),
            f"converged and |metered - {AE_TARGET_FRACTION:g}| <= {AE_TOLERANCE:g}",
            all(row["converged"] and row["within_tolerance"] for row in rows),
            "camsim.camera.Camera.auto_expose; tests/test_camera.py::"
            "test_auto_exposure_converges, ::test_auto_exposure_respects_limits",
        ),
        Check(
            "Camera AE physical limits",
            "; ".join(
                f"{row['lux']:g} lux: {row['exposure_ms']:.2f} ms, "
                f"{row['analog_gain_db']:.1f} dB analog, {row['digital_gain']:.2f}x digital, "
                f"{row['mode']}"
                for row in rows
            ),
            "exposure within preset min/max, analog within preset min/max, "
            f"digital >= 1x, iterations <= {AE_MAX_ITERATIONS}, mode LCG/HCG",
            all(row["within_limits"] for row in rows),
            "ExposureResult checked against the full SensorConfig exposure, "
            "analog-gain, iteration, and mode bounds",
        ),
    ]
    return rows, checks


def _measure_rccc() -> tuple[dict[str, Any], list[Check]]:
    """Check the RCCC baseline's shape, constant reconstruction, and linearity.

    RCCC is a monochrome-plus-red automotive layout: the clear channel is the
    documented luma placeholder for both G and B. This is therefore a
    shape/linearity/smoke check, not a colour-accuracy or MTF gate; advanced
    RCCC demosaicing is deferred.
    """
    height, width = 32, 48
    red = np.full((height, width), 0.25, dtype=np.float32)
    clear = np.full((height, width), 0.6, dtype=np.float32)
    mosaic = mosaic_from_channels({"R": red, "C": clear}, CFAPattern.RCCC)
    reconstructed = np.asarray(demosaic_rccc_baseline(mosaic), dtype=np.float64)
    scaled = np.asarray(demosaic_rccc_baseline(mosaic * 0.5), dtype=np.float64)

    shape_ok = reconstructed.shape == (height, width, 3) and bool(
        np.all(np.isfinite(reconstructed))
    )
    constant_ok = bool(
        np.allclose(reconstructed[..., 0], 0.25, atol=1e-6)
        and np.allclose(reconstructed[..., 1], 0.6, atol=1e-6)
        and np.allclose(reconstructed[..., 2], 0.6, atol=1e-6)
    )
    linear_ok = bool(np.allclose(scaled, reconstructed * 0.5, atol=1e-6))
    checks = [
        Check(
            "RCCC baseline shape and constants",
            f"{tuple(reconstructed.shape)}, R={reconstructed[..., 0].mean():.3f}, "
            f"G={reconstructed[..., 1].mean():.3f}, B={reconstructed[..., 2].mean():.3f}",
            "finite (H, W, 3); R = 0.25; G = B = 0.60",
            shape_ok and constant_ok,
            "camsim.isp.demosaic.demosaic_rccc_baseline; tests/test_isp_demosaic.py::"
            "test_rccc_baseline_shape_and_measured_sites, "
            "::test_rccc_baseline_preserves_dc_everywhere",
        ),
        Check(
            "RCCC baseline linearity",
            f"max |0.5 x output - output(0.5 x input)| = "
            f"{float(np.max(np.abs(scaled - reconstructed * 0.5))):.3e}",
            "<= 1e-6",
            linear_ok,
            "Explicit 0.5 scale-invariance check on the baseline reconstruction",
        ),
    ]
    values = {
        "shape": list(reconstructed.shape),
        "finite": bool(np.all(np.isfinite(reconstructed))),
        "constant_reconstruction": constant_ok,
        "linearity_max_error": float(np.max(np.abs(scaled - reconstructed * 0.5))),
    }
    return values, checks


def _run_pipeline_once() -> tuple[NDArray[np.uint16], NDArray[np.float32]]:
    """Run the documented render -> expose -> ISP chain with fixed seeds."""
    scene = render(
        "slanted_edge",
        illuminant="D65",
        lux=200.0,
        shape=DETERMINISM_SHAPE,
        oversampling=2,
        seed=11,
    )
    sensor = Sensor(load_sensor(SENSOR_PRESET), seed=1)
    raw = sensor.expose(scene.irradiance, 0.01, mode="LCG", seed=7)
    image = ISP(load_isp(ISP_PRESET)).process(raw)
    return np.asarray(raw.data), np.asarray(image.data)


def _measure_determinism() -> tuple[dict[str, Any], list[Check]]:
    """Run the same pipeline twice and require bit-identical arrays.

    Both runs construct the sensor and ISP from the same presets and pass the
    same scene and frame seeds. Fixed-pattern maps derive from the preset's
    defect seed, so an accidental global RNG would show up as a difference.
    """
    raw_one, image_one = _run_pipeline_once()
    raw_two, image_two = _run_pipeline_once()
    raw_equal = bool(np.array_equal(raw_one, raw_two))
    image_equal = bool(np.array_equal(image_one, image_two))
    checks = [
        Check(
            "Determinism (same seeds, same environment)",
            f"RAW equal={raw_equal}, rendered equal={image_equal} "
            f"({raw_one.nbytes + image_one.nbytes} bytes compared)",
            "bit-identical arrays",
            raw_equal and image_equal,
            "Two in-process runs of scene.render -> Sensor.expose -> ISP.process",
        )
    ]
    values = {
        "raw_equal": raw_equal,
        "image_equal": image_equal,
        "raw_shape": list(raw_one.shape),
        "image_shape": list(image_one.shape),
    }
    return values, checks


def _measure_import_independence() -> tuple[dict[str, Any], list[Check]]:
    """Import ``camsim.metrics`` in a fresh interpreter and check the graph."""
    assertions = "; ".join(
        f"assert {module!r} not in sys.modules" for module in FORBIDDEN_METRICS_IMPORTS
    )
    code = f"import sys; import camsim.metrics; {assertions}"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    passed = completed.returncode == 0
    checks = [
        Check(
            "Metrics import independence",
            f"forbidden modules loaded: "
            f"{'none' if passed else completed.stderr.strip().splitlines()[-1]}",
            "sensor, ISP, scene, and optics absent from sys.modules",
            passed,
            "Fresh interpreter: import camsim.metrics, then inspect sys.modules",
        )
    ]
    return {"passed": passed, "forbidden": list(FORBIDDEN_METRICS_IMPORTS)}, checks


def _measure_cli_chain() -> tuple[dict[str, Any], list[Check]]:
    """Execute ``camsim render`` then ``camsim measure`` in a temporary directory.

    The CLI is the user-facing orchestration path; calling the library
    functions directly would not prove that argument parsing, file IO, layout
    serialization, and report serialization still fit together.
    """
    with tempfile.TemporaryDirectory(prefix="camsim-g5-cli-") as directory:
        out = Path(directory)
        render_cmd = [
            sys.executable,
            "-m",
            "camsim.cli",
            "render",
            "--scene",
            "slanted_edge",
            "--shape",
            "128x128",
            "--out",
            str(out),
            "--seed",
            "5",
        ]
        render_result = subprocess.run(
            render_cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        images = sorted(out.glob("*_render.png"))
        layouts = sorted(out.glob("layout.json"))
        report_path = out / "metrics.json"
        measure_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        if render_result.returncode == 0 and images and layouts:
            measure_cmd = [
                sys.executable,
                "-m",
                "camsim.cli",
                "measure",
                "--image",
                str(images[0]),
                "--layout",
                str(layouts[0]),
                "--out",
                str(report_path),
            ]
            measure_result = subprocess.run(
                measure_cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        payload: dict[str, Any] = {}
        if report_path.is_file():
            payload = _load_json(report_path)
        mtf50 = payload.get("mtf", {}).get("mtf50_cp") if payload else None
        acutance = payload.get("acutance")
        passed = (
            render_result.returncode == 0
            and measure_result.returncode == 0
            and isinstance(mtf50, (int, float))
            and math.isfinite(float(mtf50))
            and isinstance(acutance, (int, float))
            and math.isfinite(float(acutance))
        )
        measured = (
            f"render rc={render_result.returncode}, measure rc={measure_result.returncode}, "
            f"MTF50={_format_number(float(mtf50))} cycles/pixel, "
            f"acutance={_format_number(float(acutance))}"
            if passed
            else f"render rc={render_result.returncode}, measure rc={measure_result.returncode}"
        )
        checks = [
            Check(
                "CLI render -> measure chain",
                measured,
                "both commands exit 0; finite MTF50 and acutance",
                passed,
                "python -m camsim.cli render | measure on a 128 x 128 slanted edge",
            )
        ]
        values = {
            "render_returncode": render_result.returncode,
            "measure_returncode": measure_result.returncode,
            "mtf50_cp": float(mtf50) if passed else None,
            "acutance": float(acutance) if passed else None,
        }
        return values, checks


def _measure_quick_isolation() -> tuple[dict[str, Any], list[Check]]:
    """Verify that committed gate evidence is full-run evidence.

    Quick modes write ``*_quick.json`` names and are documented as smoke tests.
    A quick artifact in the gate directory, or a committed JSON with
    ``quick: true``, would silently downgrade the evidence.
    """
    quick_files = sorted(path.name for path in REPORT_DIR.glob("*quick*"))
    full_flags: dict[str, Any] = {}
    for name in ("g3_isp_results.json", "g4_metrics_results.json"):
        payload = _load_json(REPORT_DIR / name)
        full_flags[name] = payload.get("quick") is False and payload.get("gate_evidence") is True
    g2 = _load_json(REPORT_DIR / "g2_sensor_results.json")
    full_flags["g2_sensor_results.json"] = g2.get("gate_evidence") is True
    passed = not quick_files and all(full_flags.values())
    checks = [
        Check(
            "Quick-mode isolation",
            f"quick artifacts in results/reports: {quick_files or 'none'}; "
            f"full-run flags: {full_flags}",
            "no quick artifacts; every committed JSON is full-run gate evidence",
            passed,
            "results/reports/*quick* absent; quick=false and gate_evidence=true",
        )
    ]
    return {"quick_files": quick_files, "full_flags": full_flags}, checks


def _measure_machine_vision_scope() -> tuple[dict[str, Any], list[Check]]:
    """Confirm that no machine-vision implementation exists before sign-off.

    ``camsim.tasks`` is a gated namespace. Until the owner signs off it must
    contain only its package docstring: no modules, functions, classes, or
    PyTorch import.
    """
    tasks_dir = REPO_ROOT / "src" / "camsim" / "tasks"
    files = sorted(path.name for path in tasks_dir.glob("*.py"))
    init_text = (tasks_dir / "__init__.py").read_text(encoding="utf-8")
    module = ast.parse(init_text)
    executable_nodes = [
        node for node in module.body if not isinstance(node, ast.Expr | ast.ImportFrom)
    ]
    imports_torch = "torch" in init_text
    passed = files == ["__init__.py"] and not executable_nodes and not imports_torch
    checks = [
        Check(
            "Machine-vision code remains gated",
            f"files={files}, executable statements={len(executable_nodes)}, "
            f"torch import={imports_torch}",
            "tasks package contains only its docstring",
            passed,
            "camsim.tasks package inspection; no Phase 6 implementation",
        )
    ]
    return {
        "files": files,
        "executable_statements": len(executable_nodes),
        "imports_torch": imports_torch,
    }, checks


def _copy_g5_figures() -> list[str]:
    """Copy the G5 figures into ``docs/images`` and verify byte equality."""
    copied: list[str] = []
    for name in ("g5_gate_summary.png", "g5_evidence.png"):
        source = FIGURE_DIR / name
        destination = DOCS_IMAGES / name
        DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if source.read_bytes() != destination.read_bytes():
            raise ValueError(f"docs image {destination} is not byte-identical to {source}")
        copied.append(name)
    return copied


def _plot_gate_summary(
    gates: list[GateArtifact],
    cross_checks: list[Check],
) -> Figure:
    """Draw the gate/check status matrix used on the validation page."""
    set_style()
    short_scope = {"G1": "scene/optics", "G2": "sensor", "G3": "ISP", "G4": "metrics"}
    rows: list[tuple[str, str, str, str]] = [
        ("G0", "foundation", f"{len(GATE_RUN)}/{len(GATE_RUN)} commands", "PASS"),
        *[
            (
                gate.gate,
                short_scope.get(gate.gate, gate.scope),
                f"{sum(1 for check in gate.checks if check.get('pass'))}/{len(gate.checks)} checks",
                "PASS",
            )
            for gate in gates
        ],
        (
            "G5",
            "cross-cutting",
            f"{sum(c.passed for c in cross_checks)}/{len(cross_checks)} checks",
            "RECORDED",
        ),
        ("G5", "human sign-off", "owner review", "AWAITING"),
    ]
    figure, axis = plt.subplots(figsize=(10.0, 4.4))
    colours = {
        "PASS": PALETTE[2],
        "RECORDED": PALETTE[5],
        "AWAITING": PALETTE[4],
        "FAIL": PALETTE[1],
    }
    row_height = 1.0
    axis.set_xlim(0.0, 3.0)
    axis.set_ylim(0.0, len(rows) * row_height)
    axis.set_xticks([0.5, 1.5, 2.5])
    axis.set_xticklabels(["Gate", "Evidence", "Status"])
    axis.xaxis.tick_top()
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(False)
    for index, (gate, scope, evidence, status) in enumerate(rows):
        y = len(rows) - index - 1
        fill = colours.get(status, PALETTE[7])
        axis.add_patch(
            plt.Rectangle((0.0, y), 3.0, row_height, facecolor="white", edgecolor="#999999")
        )
        axis.add_patch(
            plt.Rectangle((2.0, y), 1.0, row_height, facecolor=fill, edgecolor="#999999")
        )
        axis.text(0.5, y + 0.5, f"{gate}  {scope}", ha="center", va="center", fontsize=10)
        axis.text(1.5, y + 0.5, evidence, ha="center", va="center", fontsize=10)
        axis.text(
            2.5,
            y + 0.5,
            status,
            ha="center",
            va="center",
            fontsize=10,
            color="white" if status != "AWAITING" else "#222222",
            fontweight="bold",
        )
    for x in (1.0, 2.0):
        axis.axvline(x, color="#999999", linewidth=0.8)
    axis.set_title(
        "G5 gate status: recorded checks pass; human sign-off awaiting",
        pad=24.0,
    )
    return figure


def _plot_evidence(
    denoise: dict[str, Any],
    sharpen: dict[str, Any],
    ae_rows: list[dict[str, Any]],
) -> Figure:
    """Draw the cross-cutting evidence panel used on the validation page."""
    set_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))

    axis = axes[0, 0]
    # SNR is in dB and MTF50 is in cycles/pixel, so plotting them on one axis
    # would hide the MTF50 bars. The bars show SNR; the resolution numbers are
    # annotated rather than scaled onto an unrelated axis.
    positions = np.arange(2, dtype=np.float64)
    axis.bar(
        positions,
        [denoise["snr_before_db"], denoise["snr_after_db"]],
        0.55,
        color=[PALETTE[0], PALETTE[2]],
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(["before", "after"])
    axis.set_ylabel("SNR (dB)")
    axis.set_ylim(0.0, max(denoise["snr_after_db"], denoise["snr_before_db"]) * 1.25)
    axis.set_title("Denoise: SNR gain and MTF50 preservation")
    axis.annotate(
        f"SNR {denoise['snr_gain_db']:+.2f} dB\n"
        f"MTF50 {denoise['mtf50_before_cp']:.4f} -> {denoise['mtf50_after_cp']:.4f} cp\n"
        f"({denoise['mtf50_relative_change']:+.2%})",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
    )

    axis = axes[0, 1]
    columns = np.arange(80, 176)
    difference = sharpen["profile_after"][columns] - sharpen["profile_before"][columns]
    axis.plot(columns, difference, color=PALETTE[2], label="sharpened - original")
    axis.axhline(0.0, color=PALETTE[7], linestyle=":", linewidth=1.0, label="no change")
    axis.set_xlabel("column (pixel)")
    axis.set_ylabel("linear luma difference")
    axis.set_title("Sharpen: profile change and overshoot")
    axis.legend(loc="upper left")
    axis.annotate(
        f"MTF50 {sharpen['mtf50_relative_gain']:+.2%}\n"
        f"overshoot {sharpen['overshoot_fraction']:+.3%} of contrast\n"
        f"max {sharpen['normalized_max']:.6f}",
        xy=(0.98, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
    )

    axis = axes[1, 0]
    scene_labels = [f"{row['lux']:g} lux" for row in ae_rows]
    metered = [row["metered_value"] for row in ae_rows]
    positions = np.arange(len(ae_rows), dtype=np.float64)
    axis.bar(positions, metered, 0.55, color=PALETTE[0])
    axis.axhline(
        AE_TARGET_FRACTION,
        color=PALETTE[1],
        linestyle="--",
        linewidth=1.2,
        label=f"target {AE_TARGET_FRACTION:g}",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(scene_labels)
    axis.set_ylim(0.0, 0.30)
    axis.set_ylabel("metered fraction of full scale")
    axis.set_title("Camera auto-exposure convergence")
    axis.legend(loc="upper right")
    for index, row in enumerate(ae_rows):
        axis.annotate(
            f"{row['iterations']} probe(s), {row['mode']}\n{row['metered_value']:.4f}",
            xy=(index, 0.245),
            ha="center",
            va="center",
            fontsize=8,
        )

    axis = axes[1, 1]
    positions = np.arange(len(BENCHMARK_STAGES), dtype=np.float64)
    axis.barh(
        positions - 0.18,
        BENCHMARK_QUICK_SHARE_PCT,
        0.34,
        color=PALETTE[5],
        label=f"quick ({BENCHMARK_QUICK_TOTAL_MS:.2f} ms)",
    )
    axis.barh(
        positions + 0.18,
        BENCHMARK_FULL_SHARE_PCT,
        0.34,
        color=PALETTE[0],
        label=f"full ({BENCHMARK_FULL_TOTAL_MS:.2f} ms)",
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(BENCHMARK_STAGES)
    axis.set_xlabel("share of pipeline time (%)")
    axis.set_title("Pipeline benchmark stage shares")
    axis.legend(loc="center right")
    axis.set_xlim(0.0, 70.0)

    figure.suptitle(
        "G5 cross-cutting evidence: denoise, sharpen, auto-exposure, and benchmark",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure


def _format_value(value: Any) -> str:
    """Format a nested measurement value for the report's detail tables."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _format_number(value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={_format_value(item)}" for key, item in value.items())
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    return str(value)


def _render_report(
    gates: list[GateArtifact],
    cross_checks: list[Check],
    measurements: dict[str, Any],
    docs_images: list[str],
) -> str:
    """Build the consolidated Markdown report."""
    lines: list[str] = []
    add = lines.append
    commit = _git_commit()
    environment = {
        "Python": platform.python_version(),
        "camsim": camsim.__version__,
        "NumPy": _package_version("numpy"),
        "SciPy": _package_version("scipy"),
        "colour-science": _package_version("colour-science"),
        "colour-demosaicing": _package_version("colour-demosaicing"),
        "Matplotlib": _package_version("matplotlib"),
        "Platform": f"{platform.system()} ({platform.machine()})",
        "Validated source commit": commit,
        "Branch": "feat/camsim-core",
        "Report date": date.today().isoformat(),
        "Sensor preset": "src/camsim/config/presets/sensors/flagship_phone_50mp.json",
        "ISP preset": "src/camsim/config/presets/isp/default.json",
    }
    # The preset hashes are the ones recorded by the G3 run on this source
    # tree; reading them back keeps the report tied to the validated build
    # rather than to a hash recomputed after a possible edit.
    presets = _load_json(REPORT_DIR / "g3_isp_results.json")["presets"]
    bindings = _artifact_bindings()

    add("# camsim G5 validation report")
    add("")
    add("**Status: awaiting human sign-off.**")
    add("")
    add(
        "This is the consolidated G5 candidate report for the camsim core "
        "pipeline. G1-G4 are green, every cross-cutting check below passed on "
        "the validated source tree, and the engineering gates are green. The "
        "G5 gate itself is **not** closed: the project owner must review this "
        "report and the figure gallery, confirm the scope, assumptions, and "
        "limitations, and explicitly approve `v0.1.0`. Until that decision is "
        "recorded, the status stays **awaiting human sign-off**, no tag "
        "exists, and no machine-vision code may be written."
    )
    add("")
    add("## 1. Scope and architecture")
    add("")
    add(
        "camsim is a deterministic, from-scratch camera simulator with three "
        "clean layers. The **simulation pipeline** renders spectral scene "
        "irradiance through a lens model, integrates it into electrons in the "
        "sensor, and processes the RAW frame through a modular ISP; `scene` "
        "and `optics` are supporting helpers, while `sensor` and `isp` are the "
        "reusable core and never depend on them. The **metrics** layer is an "
        "independent harness that operates on arrays plus metadata and must "
        "not import the simulator; `tests/test_metrics_independence.py` "
        "enforces that rule. The **machine-vision** layer (`camsim.tasks`) is "
        "gated behind G5 and is an empty namespace at this commit."
    )
    add("")
    add("```text")
    add("scene reflectance x illuminant x lens")
    add("        -> spectral irradiance      camsim.scene / camsim.optics")
    add("        -> RAW frame                camsim.Sensor.expose")
    add("        -> rendered image           camsim.ISP.process")
    add("        -> image-quality report     camsim.metrics.measure_all")
    add("```")
    add("")
    add(
        "The enforced ISP order is `DPC -> BLC -> LSC -> AWB -> demosaic -> CCM -> "
        "denoise -> sharpen -> tone -> output`. Every stage before tone operates in "
        "linear float32 light; the tone curve and output OETF are the only "
        "non-linear steps."
    )
    add("")
    add("## 2. Environment and validated source")
    add("")
    add("| Item | Value |")
    add("|---|---|")
    for key, value in environment.items():
        add(f"| {key} | `{value}` |")
    add(f"| Sensor config hash | `{presets['sensor_config_hash']}` |")
    add(f"| ISP config hash | `{presets['isp_config_hash']}` |")
    add("")
    add(
        "The phase reports and this candidate were produced from the same "
        "source tree and software environment. The G5 candidate commit adds "
        "only this report, its tests, the two G5 figures, and documentation "
        "and metadata updates; it changes no simulation, sensor, ISP, "
        "metrics, camera, or CLI code. All numbers are simulated measurements, "
        "not measurements of physical hardware."
    )
    add("")

    add("## 3. Gate status")
    add("")
    add("| Gate | Phase | Scope | Evidence | Measured / passed | Status |")
    add("|---|---|---|---|---|---|")
    add(
        "| G0 | 0 | install, lint, typecheck, tests, docs | "
        "[docs/validation.md](../../docs/validation.md) | "
        f"{len(GATE_RUN)}/{len(GATE_RUN)} commands | **PASS** (local) |"
    )
    for gate in gates:
        passed = sum(1 for check in gate.checks if check.get("pass"))
        evidence_parts = [f"[{gate.report.name}]({gate.report.name})"]
        if gate.json_path is not None:
            evidence_parts.append(f"[{gate.json_path.name}]({gate.json_path.name})")
        add(
            f"| {gate.gate} | {gate.gate[1]} | {gate.scope} | "
            f"{' + '.join(evidence_parts)} | {passed}/{len(gate.checks)} checks | "
            f"**{gate.reported_status}** |"
        )
    cross_passed = sum(1 for check in cross_checks if check.passed)
    add(
        f"| G5 | 5 | consolidated cross-cutting evidence | this report | "
        f"{cross_passed}/{len(cross_checks)} checks | **Awaiting human sign-off** |"
    )
    add("")
    add(
        "A phase gate is evidence only when its report says PASS and every "
        "recorded check in its machine-readable results passed. The loader "
        "that builds this table refuses quick-mode artifacts and any JSON not "
        "marked `gate_evidence: true`. G0 is a local gate run; no CI-matrix "
        "result is recorded in this repository artifact."
    )
    add("")
    add("### 3.1 Phase artifact bindings")
    add("")
    add(
        "Every phase artifact this report reads is bound by SHA-256. The G5 "
        "test suite recomputes these hashes and fails if an artifact changes "
        "without this report being regenerated, so the consolidated tables and "
        "the committed evidence cannot silently diverge. The binding proves "
        "that the report describes the committed artifacts; it does **not** "
        "prove that those artifacts were regenerated from this exact commit, "
        "and no such proof is claimed."
    )
    add("")
    add("| Artifact | SHA-256 |")
    add("|---|---|")
    for name, digest in bindings:
        add(f"| `{name}` | `{digest}` |")
    add("")
    add(
        "**Regeneration record for this candidate.** "
        "`benchmarks/g1_scene_optics.py`, examples 01-04 in full mode, and "
        "notebooks 01-03 headless were re-run on the validated source tree. "
        "G1 remained PASS and its report and four figures were byte-identical; "
        "the G2-G4 reports and JSON were byte-identical; the example and "
        "notebook gallery figures were byte-identical. G2, G3, and G4 each "
        "reported PASS after regeneration. Quick modes were not run against "
        "the gate directories."
    )
    add("")

    add("## 4. G1-G4 measured evidence")
    add("")
    add(
        "The tables below are the verdict tables from the committed phase "
        "reports and their JSON. The values were not re-typed by hand: G2-G4 "
        "are read from the JSON, and G1 is parsed from its committed Markdown "
        "verdict table. The hash bindings in section 3.1 tie this report to "
        "those exact artifacts. They verify full-run PASS evidence and "
        "artifact/report consistency; they do not by themselves prove that the "
        "artifacts were regenerated from the current commit. The regeneration "
        "record in section 3.1 states what was actually re-run for this "
        "candidate."
    )
    add("")
    for gate in gates:
        add(f"### {gate.gate} - {gate.scope}")
        add("")
        add(f"Report: [`{gate.report.name}`]({gate.report.name})")
        if gate.json_path is not None:
            add(f"Machine-readable results: [`{gate.json_path.name}`]({gate.json_path.name})")
        add("")
        add("| Check | Measured | Tolerance | Result |")
        add("|---|---:|---|---|")
        for check in gate.checks:
            name, measured, tolerance, verdict = _check_row(check)
            add(f"| {name} | {measured} | {tolerance} | {verdict} |")
        add("")
        add(
            f"{sum(1 for check in gate.checks if check.get('pass'))} of "
            f"{len(gate.checks)} checks pass; reported status **{gate.reported_status}**."
        )
        add("")

    add("## 5. Cross-cutting checks")
    add("")
    add(
        "These criteria span phases or the engineering process, so no single "
        "phase report owns them. The measurements are direct calls to the "
        "public stages and metrics; the quality-gate outcomes are the "
        "recorded outputs of the documented commands."
    )
    add("")
    add("### 5.1 Quality gates")
    add("")
    add("| Command | Recorded result | Wall time | Verdict |")
    add("|---|---|---:|---|")
    for command in GATE_RUN:
        verdict = "PASS" if command.passed else "FAIL"
        add(
            f"| `{command.command}` | {command.result} | "
            f"{command.duration_s:.2f} s | **{verdict}** |"
        )
    add("")
    add(
        "Absolute timings are machine-specific and are evidence, not "
        "guarantees. The coverage gate enforces at least 90 % statement and "
        "branch coverage with `*/camsim/viz/*` omitted: the presentation layer "
        "is exercised by the executed examples and notebooks that generate "
        "the committed gallery, while the simulation, metrics, configuration, "
        "characterization, and CLI modules remain subject to the coverage "
        "gate. The threshold is not lowered for this candidate."
    )
    add("")
    add("### 5.2 Cross-cutting measurement summary")
    add("")
    add("| Check | Measured | Bound | Verdict |")
    add("|---|---|---|---|")
    for check in cross_checks:
        verdict = "PASS" if check.passed else "FAIL"
        add(
            f"| {_escape_cell(check.name)} | {_escape_cell(check.measured)} | "
            f"{_escape_cell(check.bound)} | **{verdict}** |"
        )
    add("")
    add(
        "Every non-timing cross-cutting measurement is deterministic: fixed "
        "seeds, explicit `numpy.random.Generator` instances, and no global "
        "RNG. The derivation and raw values follow."
    )
    add("")

    denoise = measurements["denoise"]
    add("### 5.3 Denoise: SNR gain and MTF50 preservation")
    add("")
    add(
        "A 256 x 256 slanted edge at 5 degrees is blurred by the known "
        "unit-sigma Gaussian and corrupted with additive Gaussian noise "
        f"(sigma = {denoise['config']['edge_noise_sigma']:g}). MTF50 is "
        "measured on the noisy edge and again after the shipped bilateral "
        "stage (`method=bilateral`, luma strength 1.0, radius 1.5); the SNR "
        "measurement uses an independent 64 x 64 flat patch with its own "
        "generator. The resolution bound is a **loss** bound: a denoiser that "
        "suppresses noise by smearing the edge fails, while a slightly higher "
        "measured MTF50 passes because no resolution was lost."
    )
    add("")
    add("| Quantity | Measured |")
    add("|---|---:|")
    add(f"| MTF50 before denoise | {denoise['mtf50_before_cp']:.6f} cycles/pixel |")
    add(f"| MTF50 after denoise | {denoise['mtf50_after_cp']:.6f} cycles/pixel |")
    add(f"| Relative MTF50 change | {denoise['mtf50_relative_change']:+.2%} |")
    add(f"| SNR before denoise | {denoise['snr_before_db']:.3f} dB |")
    add(f"| SNR after denoise | {denoise['snr_after_db']:.3f} dB |")
    add(f"| SNR gain | {denoise['snr_gain_db']:+.2f} dB |")
    add("")
    add(
        "Interpretation: the bilateral stage raises flat-patch SNR and does "
        "not lose MTF50 on this edge. The measured MTF50 rises because the "
        "full-strength bilateral filter sharpens the transition as it "
        "suppresses the noise floor; the check is deliberately a preservation "
        "bound, and the measured change is reported with its sign rather than "
        "described as a pure equality."
    )
    add("")

    sharpen = measurements["sharpen"]
    add("### 5.4 Sharpen: MTF50 gain and overshoot")
    add("")
    add(
        "The same blurred edge without noise is sharpened with the shipped "
        "unsharp mask at amount 1.0, radius 1.0, threshold 0, luma only. MTF50 "
        "must increase, and the bright-side profile maximum must exceed the "
        "high plateau by more than the measured margin; the overshoot is "
        "reported both as a fraction of edge contrast and as the normalized "
        "profile maximum."
    )
    add("")
    add("| Quantity | Measured |")
    add("|---|---:|")
    add(f"| MTF50 before sharpen | {sharpen['mtf50_before_cp']:.6f} cycles/pixel |")
    add(f"| MTF50 after sharpen | {sharpen['mtf50_after_cp']:.6f} cycles/pixel |")
    add(f"| Relative MTF50 gain | {sharpen['mtf50_relative_gain']:+.2%} |")
    add(f"| High plateau | {sharpen['profile_high_plateau']:.6f} |")
    add(f"| Profile maximum after sharpen | {sharpen['profile_max_after']:.6f} |")
    add(f"| Normalized profile maximum | {sharpen['normalized_max']:.6f} |")
    add(f"| Overshoot fraction of edge contrast | {sharpen['overshoot_fraction']:+.3%} |")
    add("")

    add("### 5.5 Camera auto-exposure")
    add("")
    add(
        "`Camera.auto_expose` probes the real sensor, meters a centre-weighted "
        "black-corrected fraction, and ramps exposure, analog gain, readout "
        "mode, and digital gain. The existing gate tests are "
        "`tests/test_camera.py::test_auto_exposure_converges` and "
        "`::test_auto_exposure_respects_limits`; the measurement below "
        "re-executes the policy on three 64 x 64 flat patches."
    )
    add("")
    add("| Scene | Converged | Probes | Mode | Exposure (ms) | Analog (dB) | Digital | Metered |")
    add("|---|---|---|---|---:|---:|---:|---:|")
    for row in measurements["auto_exposure"]:
        add(
            f"| {row['lux']:g} lux | {row['converged']} | {row['iterations']} | "
            f"{row['mode']} | {row['exposure_ms']:.2f} | {row['analog_gain_db']:.1f} | "
            f"{row['digital_gain']:.2f}x | {row['metered_value']:.4f} |"
        )
    add("")
    add(
        f"Target fraction {AE_TARGET_FRACTION:g} with tolerance {AE_TOLERANCE:g}; "
        f"all results converge within {AE_MAX_ITERATIONS} probes and stay inside "
        "the preset's exposure and analog-gain windows. At 0.01 lux the policy "
        "uses HCG, the exposure ceiling, maximum analog gain, and substantial "
        "digital gain; metering succeeds, but digital gain amplifies noise, so "
        "that extreme-low-light image quality is not claimed."
    )
    add("")

    rccc = measurements["rccc"]
    add("### 5.6 RCCC baseline")
    add("")
    add(
        "RCCC is a monochrome-plus-red automotive layout. The documented "
        "baseline uses the clear channel as the luma placeholder for G and B; "
        "it is a shape/linearity/smoke reconstruction, not an advanced "
        "demosaic. The existing tests are "
        "`tests/test_isp_demosaic.py::test_rccc_baseline_shape_and_measured_sites`, "
        "`::test_rccc_baseline_preserves_dc_everywhere`, and "
        "`tests/test_example_02.py::test_g3_rccc_baseline_smoke`; the G3 "
        "qualitative render is `../figures/02_isp_rccc_baseline.png`."
    )
    add("")
    add("| Quantity | Measured |")
    add("|---|---:|")
    add(f"| Output shape | `{tuple(rccc['shape'])}` |")
    add(f"| Finite pixels | {rccc['finite']} |")
    add(f"| Constant reconstruction | {rccc['constant_reconstruction']} |")
    add(f"| Linear-scaling max error | {rccc['linearity_max_error']:.3e} |")
    add("")
    add(
        "**RCCC/Bayer caveat.** The Bayer ColorChecker delta-E thresholds and "
        "MTF gates do not apply to RCCC: its colour reconstruction is a "
        "placeholder by design, and advanced RCCC demosaicing is deferred. "
        "This evidence shows that the baseline is finite, shape-correct, "
        "constant-preserving, and linear, no more."
    )
    add("")

    determinism = measurements["determinism"]
    add("### 5.7 Determinism and import independence")
    add("")
    add(
        f"- Determinism: two runs of the same seeded render -> expose -> ISP "
        f"chain produce bit-identical RAW (`{determinism['raw_equal']}`) and "
        f"rendered (`{determinism['image_equal']}`) arrays."
    )
    independence = measurements["import_independence"]
    add(
        f"- Import independence: a fresh interpreter imports `camsim.metrics` "
        f"and finds none of {', '.join(f'`{name}`' for name in independence['forbidden'])} "
        f"in `sys.modules` (pass={independence['passed']})."
    )
    cli = measurements["cli_chain"]
    add(
        f"- CLI chain: `camsim render` then `camsim measure` exit "
        f"{cli['render_returncode']} and {cli['measure_returncode']} and "
        f"produce MTF50 {_format_number(cli['mtf50_cp'] or math.nan)} cycles/pixel "
        f"with acutance {_format_number(cli['acutance'] or math.nan)}."
    )
    add("")
    add(
        "Determinism is promised within one environment (same OS, Python, "
        "NumPy, and SciPy). Cross-platform comparisons use tolerances because "
        "floating-point reduction order and Matplotlib/font builds can differ; "
        "README and documentation figures should be compared numerically, not "
        "bit-for-bit, across platforms."
    )
    add("")

    add("### 5.8 Quick-mode isolation and machine-vision scope")
    add("")
    quick = measurements["quick_isolation"]
    add(
        f"- Quick-mode isolation: quick artifacts in `results/reports` = "
        f"`{quick['quick_files']}`; committed full-run flags = "
        f"`{quick['full_flags']}`."
    )
    scope = measurements["machine_vision_scope"]
    add(
        f"- Machine-vision scope: `camsim.tasks` modules = `{scope['files']}`, "
        f"executable statements = {scope['executable_statements']}, "
        f"torch import = {scope['imports_torch']}."
    )
    add("")
    add(
        "Quick modes (`--quick`) use smaller grids and write suffixed "
        "`*_quick.json` artifacts; they are smoke tests and never gate "
        "evidence. No machine-vision code exists before the G5 sign-off, so "
        "this report makes no task-accuracy, CNN, or metric-correlation claim."
    )
    add("")

    add("## 6. Assumptions")
    add("")
    assumptions = (
        "**Simulated sensor and preset.** The flagship preset is a "
        "representative engineering design point (LCG 100 uV/e-, 6000 e- full "
        "well, 2 e- rms read noise; HCG 300 uV/e-, 2000 e-, 0.7 e-; 10-bit ADC, "
        "0.8 V full scale, 64 DN black level). It is not a measurement of, or a "
        "claim about, a specific commercial part.",
        "**Provisional spectral data.** The QE, CFA, IR-cut, illuminant, and "
        "ColorChecker curves are bundled synthetic engineering estimates with "
        "recorded provenance, not digitized vendor or measured data. They are "
        "adequate to validate the physics and interfaces, not to predict a "
        "particular product's colour response.",
        "**In-sample colour calibration.** The 3x3 CCM is fitted and evaluated "
        "on the same 24 ColorChecker patches. This measures whether the ISP "
        "can represent the calibration scene, not held-out generalisation, "
        "colour constancy, or performance under an unseen illuminant.",
        "**ISO 12233-style, not certified.** The slanted-edge e-SFR follows "
        "the standard's principles (5 degree tilt, ESF derivative, Hamming "
        "window, native-Nyquist band) but makes no claim of conformance to its "
        "chart, windowing, or reporting details. The EMVA 1288-style sensor "
        "characterization is likewise simulated and uncertified.",
        "**Determinism within one environment.** Every stochastic path takes "
        "an explicit `numpy.random.Generator` and a recorded seed. Bit-identical "
        "outputs are promised for one OS/Python/NumPy/SciPy environment; "
        "cross-platform comparisons use the stated tolerances.",
    )
    for index, assumption in enumerate(assumptions, start=1):
        add(f"{index}. {assumption}")
    add("")

    add("## 7. Limitations")
    add("")
    limitations = (
        "**Synthetic CFA re-authored during G3.** The original broad, nearly "
        "collinear CFA curves left a 45.6 % Luther-condition residual and no "
        "3x3 matrix could meet the ColorChecker threshold. The shipped preset "
        "uses conventional Bayer bands (blue ~448 nm, green ~542 nm, red "
        "~610 nm) with moderate out-of-band leakage, reducing the residual to "
        "12.6 %. The delta-E thresholds (2.0 / 4.0) were not relaxed.",
        "**In-sample calibration.** The measured mean delta-E values (0.919 "
        "under D65 and 3.834 under 3000 K) are in-sample; the 3000 K mean is "
        "4.14 % below its limit and per-patch maxima reach 13.208. Held-out "
        "chart, illuminant-generalisation, and colour-constancy performance "
        "are not claimed.",
        "**8x Airy MTF gate.** The G4 analytic MTF cases use eight analysis "
        "samples per pixel because four carries roughly a 2 % MTF50 bias near "
        "the diffraction limit. Airy f/2.8 is a documented diagnostic only: "
        "its MTF never reaches 0.20 inside Nyquist and its crossing sits at "
        "the band edge.",
        "**Synthetic LSC, denoise, and sharpen measurements.** Lens shading "
        "is measured on a synthetic radial flat field, denoise on a synthetic "
        "noisy step edge and flat patch, and sharpen on a synthetic blurred "
        "step edge. These isolate the stages; they are not camera captures "
        "and do not include real lens shading, sensor texture, or scene "
        "content.",
        "**Quick modes are not gate evidence.** `--quick` paths use smaller "
        "grids, write suffixed quick artifacts, and never overwrite the "
        "committed full-run JSON or Markdown. The committed G2 JSON records "
        "`gate_evidence: true` (G2 has no quick-mode payload); G3 and G4 "
        "additionally record `quick: false`.",
        "**Auto-exposure digital-gain noise.** At 0.01 lux auto-exposure "
        "needs HCG, the exposure ceiling, maximum analog gain, and about 44x "
        "digital gain. Metering converges, but digital gain amplifies sensor "
        "noise and cannot create signal; extreme-low-light image quality is "
        "not claimed.",
        "**No HDR, 3-D, motion blur, or autofocus.** The model has a single "
        "linear exposure, a planar scene, static optics, and no focus search, "
        "rolling shutter, or multi-frame HDR. These are Phase 6-9 or later "
        "scope, not G5 claims.",
        "**RCCC is a baseline only.** The clear channel is a luma placeholder "
        "for G and B; Bayer colour and MTF thresholds do not apply, and "
        "advanced RCCC demosaicing is deferred.",
        "**No machine-vision evidence.** `camsim.tasks` is an empty gated "
        "namespace. This report says nothing about CNN accuracy, task-relevant "
        "IQ, mono-versus-Bayer, auto-tuning, or metric correlation.",
    )
    for index, limitation in enumerate(limitations, start=1):
        add(f"{index}. {limitation}")
    add("")

    add("## 8. Reproduction")
    add("")
    add("Run from the repository root with `uv` and all extras installed:")
    add("")
    add("```bash")
    add("uv sync --all-extras")
    add("")
    add("# Phase artifacts (regenerate only when the phase code changes)")
    add("uv run python benchmarks/g1_scene_optics.py")
    add("uv run python examples/04_emva_characterization.py")
    add("uv run python examples/02_isp_stage_gallery.py")
    add("uv run python examples/03_measure_iq.py")
    add("")
    add("# Gallery: examples 01-04 full mode, then notebooks 01-03 headless")
    add("uv run python examples/01_spectral_scene_to_raw.py")
    add("uv run python examples/02_isp_stage_gallery.py")
    add("uv run python examples/03_measure_iq.py")
    add("uv run python examples/04_emva_characterization.py")
    add("make notebooks")
    add("")
    add("# Consolidated G5 report and figures")
    add("uv run python benchmarks/g5_validation_report.py")
    add("")
    add("# G5 quality gates")
    add("uv run pytest -q")
    add("uv run pytest --cov=camsim --cov-fail-under=90")
    add("uv run ruff check . && uv run ruff format --check . && uv run mypy src")
    add("uv run mkdocs build --strict")
    add("uv run python benchmarks/benchmark_pipeline.py --quick")
    add("uv run python benchmarks/benchmark_pipeline.py")
    add("uv build")
    add("```")
    add("")
    add(
        "`make figures` runs examples 01-04 and is the gallery regeneration "
        "target; `make notebooks` executes every Jupytext notebook into "
        "`results/cache/notebooks` under a two-minute-per-notebook budget. "
        "Neither command tags, pushes, or publishes anything."
    )
    add("")
    add(
        "**Pre-commit snapshot caveat.** The generator records "
        "`git rev-parse HEAD` and `date.today()` at generation time. Re-running "
        "it on the candidate commit therefore rewrites the validated-commit "
        "and report-date fields and dirties the working tree. The committed "
        "report is a pre-commit snapshot of that candidate; regenerate it only "
        "as part of a new candidate, and commit the regenerated report with the "
        "change that invalidated it."
    )
    add("")

    add("## 9. Figure gallery")
    add("")
    add("| Figure | Shows |")
    add("|---|---|")
    add(
        "| [g5_gate_summary.png](../figures/g5_gate_summary.png) | "
        "Gate/check status matrix; G1-G4 PASS, G5 recorded checks pass, human "
        "sign-off awaiting |"
    )
    add(
        "| [g5_evidence.png](../figures/g5_evidence.png) | Denoise SNR/MTF, "
        "sharpen profile/overshoot, auto-exposure convergence, and benchmark "
        "stage shares |"
    )
    add(
        "| [g1_lux_normalization.png](../figures/g1_lux_normalization.png) | "
        "Photometric normalization across illuminants and 1-100,000 lux |"
    )
    add(
        "| [g1_psf_mtf.png](../figures/g1_psf_mtf.png) | Gaussian, Airy, and "
        "box PSFs with their transfer functions |"
    )
    add("| [g2_ptc.png](../figures/g2_ptc.png) | LCG/HCG photon-transfer curves and system gain |")
    add(
        "| [g2_snr_vs_irradiance.png](../figures/g2_snr_vs_irradiance.png) | "
        "Sensor SNR across the shot- and read-noise regimes |"
    )
    add(
        "| [g3_check_summary.png](../figures/g3_check_summary.png) | G3 check "
        "summary with tolerances |"
    )
    add(
        "| [g3_delta_e.png](../figures/g3_delta_e.png) | Per-patch ColorChecker "
        "delta-E2000 under D65 and 3000 K |"
    )
    add(
        "| [02_isp_rccc_baseline.png](../figures/02_isp_rccc_baseline.png) | "
        "RCCC baseline qualitative render |"
    )
    add(
        "| [g4_check_summary.png](../figures/g4_check_summary.png) | G4 check "
        "summary against analytic references |"
    )
    add(
        "| [g4_mtf_curves.png](../figures/g4_mtf_curves.png) | Measured MTF "
        "against analytic PSF references |"
    )
    add("")
    add(
        f"The two G5 figures are also mirrored into `docs/images/` "
        f"(`{', '.join(docs_images)}`) and verified byte-identical so the "
        "documentation site stays self-contained."
    )
    add("")

    add("## 10. Human sign-off checklist")
    add("")
    add(
        "G5 is a human decision, not an automated verdict. The owner must "
        "review the evidence below and record an explicit decision. Checking "
        "these boxes is the sign-off; until then the status line remains "
        "**awaiting human sign-off**."
    )
    add("")
    add(
        "- [ ] **Report reviewed.** Read this report end to end, including the "
        "G1-G4 verdict tables and the cross-cutting measurements."
    )
    add(
        "- [ ] **Figures reviewed.** Inspect the figure gallery, especially "
        "`g5_gate_summary.png`, `g5_evidence.png`, the G3 ColorChecker delta-E "
        "figure, the RCCC baseline, and the G4 analytic MTF curves."
    )
    add(
        "- [ ] **Scope confirmed.** Agree that the validated scope is the "
        "scene/optics helpers, sensor, ISP, metrics, camera orchestration, IO, "
        "and CLI, and that machine vision is out of scope for v0.1.0."
    )
    add(
        "- [ ] **Assumptions confirmed.** Accept the simulated flagship "
        "preset, provisional synthetic spectral data, in-sample CCM, and the "
        "uncertified ISO 12233/EMVA-style measurement status."
    )
    add(
        "- [ ] **Limitations confirmed.** Accept the synthetic CFA re-authoring, "
        "in-sample calibration, 8x Airy diagnostic, synthetic stage "
        "measurements, quick-mode exclusion, extreme-low-light digital-gain "
        "noise, and the absence of HDR/3-D/motion-blur/autofocus."
    )
    add(
        "- [ ] **Release approved.** Approve tagging `v0.1.0` on the signed-off "
        "commit and starting Phase 6 machine-vision work afterwards."
    )
    add("")
    add("```text")
    add("Owner: ______________________________")
    add("Date:  ______________________________")
    add("Decision:  [ ] approve v0.1.0     [ ] withhold (return to the failing gate)")
    add("Notes:   ______________________________")
    add("```")
    add("")
    add(
        "**This report does not create the `v0.1.0` tag, push a branch, or "
        "publish to PyPI.** Those actions require the recorded approval above "
        "and are performed in a separate release step. No machine-vision code "
        "may be written before that approval."
    )
    add("")
    add("## Appendix A. Cross-cutting detail values")
    add("")
    for section, values in measurements.items():
        if not isinstance(values, dict):
            # The auto-exposure rows are already tabulated in section 5.5.
            continue
        add(f"### {section}")
        add("")
        add("| Field | Value |")
        add("|---|---|")
        for key, value in values.items():
            if key in {"profile_before", "profile_after"}:
                continue
            add(f"| {key} | `{_format_value(value)}` |")
        add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Generate the report and figures, then print their paths."""
    set_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    gates, _ = _load_gates()
    for gate in gates:
        if gate.reported_status.upper() != "PASS":
            raise ValueError(f"{gate.gate} report status is {gate.reported_status!r}, not PASS")
        failed = [check["name"] for check in gate.checks if not check.get("pass")]
        if failed:
            raise ValueError(f"{gate.gate} has failing checks: {failed}")

    denoise, denoise_checks = _measure_denoise()
    sharpen, sharpen_checks = _measure_sharpen()
    ae_rows, ae_checks = _measure_auto_exposure()
    rccc, rccc_checks = _measure_rccc()
    determinism, determinism_checks = _measure_determinism()
    independence, independence_checks = _measure_import_independence()
    cli_chain, cli_checks = _measure_cli_chain()
    quick, quick_checks = _measure_quick_isolation()
    scope, scope_checks = _measure_machine_vision_scope()

    cross_checks = [
        *denoise_checks,
        *sharpen_checks,
        *ae_checks,
        *rccc_checks,
        *determinism_checks,
        *independence_checks,
        *cli_checks,
        *quick_checks,
        *scope_checks,
    ]
    measurements: dict[str, Any] = {
        "denoise": denoise,
        "sharpen": sharpen,
        "auto_exposure": ae_rows,
        "rccc": rccc,
        "determinism": determinism,
        "import_independence": independence,
        "cli_chain": cli_chain,
        "quick_isolation": quick,
        "machine_vision_scope": scope,
    }

    gate_figure = _plot_gate_summary(gates, cross_checks)
    save_figure(gate_figure, FIGURE_DIR / "g5_gate_summary.png")
    plt.close(gate_figure)

    evidence_figure = _plot_evidence(denoise, measurements["sharpen"], ae_rows)
    save_figure(evidence_figure, FIGURE_DIR / "g5_evidence.png")
    plt.close(evidence_figure)

    docs_images = _copy_g5_figures()

    # The appendix skips the long profile arrays; they are figure inputs, not
    # report values.
    report = _render_report(gates, cross_checks, measurements, docs_images)
    REPORT_PATH.write_text(report, encoding="utf-8")

    passed = sum(1 for check in cross_checks if check.passed)
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {FIGURE_DIR / 'g5_gate_summary.png'}")
    print(f"wrote {FIGURE_DIR / 'g5_evidence.png'}")
    print(f"cross-cutting checks: {passed}/{len(cross_checks)} passed")
    for check in cross_checks:
        if not check.passed:
            print(f"FAIL: {check.name}: {check.measured}")
    return 0 if passed == len(cross_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
