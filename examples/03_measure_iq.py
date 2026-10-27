"""Example 03: metrics gallery and the G4 validation report.

This walkthrough demonstrates the independent image-quality layer. It measures
the quantities a camera engineer uses to judge an image, and pairs every figure
with the analytic or known-noise reference that makes the number meaningful.

* **MTF.** A slanted edge is the ISO 12233 structure for estimating resolution
  from one image: the tilt lets every sub-pixel phase of the transition be
  reconstructed, the derivative of the edge-spread function is the line-spread
  function, and the normalized magnitude of its Fourier transform is the MTF.
  The figure overlays the measured e-SFR on the exact transfer function of the
  known PSF and marks the interpolated MTF50 and MTF20 crossings.
* **SNR.** A flat patch's signal-to-noise ratio is ``20 log10(mean / std)`` in
  linear light. The spatial estimator measures one frame and includes
  fixed-pattern variation; the temporal estimator pools per-pixel variances and
  excludes a static pattern. The lux sweep runs the shipped sensor and ISP at
  constant exposure, so the curve shows the transition from read-noise-limited
  to shot-noise-limited behaviour.
* **Colour difference.** CIEDE2000 measures perceived colour error in CIE Lab.
  The heatmap shows per-patch error for a synthetic ColorChecker-like chart
  built from the scene's spectrally computed reference, with a deterministic
  perturbation so the chart carries real structure rather than a numerical
  floor.
* **Acutance.** The contrast-normalized RMS gradient of one edge ROI is a cheap
  sharpness proxy for correlation studies. It falls as a Gaussian blur widens;
  it is not a calibrated MTF.

The example measures the same quantities as ``tests/test_example_03.py`` via
the shared :func:`camsim.characterization.measure_g4`; the tests and the report
therefore cannot disagree. Full mode writes
``results/reports/g4_metrics_results.json`` and ``results/reports/g4_metrics.md``
plus the ``03_*.png`` and ``g4_*.png`` gallery under ``CAMSIM_FIG_DIR``
(default ``results/figures``). ``--quick`` uses smaller MTF, SNR, chart, and
acutance grids, writes only ``g4_metrics_results_quick.json``, and is **not
gate evidence**; it never overwrites the committed gate JSON or Markdown.

The SNR-versus-lux sweep is a gallery illustration, not a gate check. It runs
the flagship sensor at a fixed 10 ms exposure over a log-spaced illuminance
range at 64 x 64 pixels, disables PDAF (its sites have reduced quantum
efficiency and the metrics harness has no mask parameter), and measures the
linear ``RenderedImage.linear_rgb`` ROI with :func:`camsim.metrics.snr_channels`.

This is a simulated validation, not a certified ISO 12233 measurement. The
gate cases use eight analysis samples per pixel because four carries roughly a
2 % MTF50 bias for near-diffraction-limited systems; Airy f/2.8 is a documented
diagnostic only.

Failure artefacts guarded here: a quick run that overwrites gate evidence; a
measured MTF curve drawn without its analytic reference; MTF plotted past
Nyquist; SNR measured on display-encoded or PDAF-contaminated pixels; a delta-E
heatmap that hides the per-patch structure behind one mean; a stale report that
no longer matches the JSON; and a report that claims a check that never ran.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from camsim.characterization import measure_g4, render_g4_report
from camsim.config.io import PRESETS_DIR, load_isp, load_sensor
from camsim.isp.pipeline import ISP
from camsim.metrics import PatchROI as MetricsPatchROI
from camsim.metrics import snr_channels
from camsim.scene.targets import render
from camsim.sensor import Sensor
from camsim.viz import (
    PALETTE,
    MTFPlotSeries,
    figure_dir,
    plot_acutance,
    plot_delta_e_heatmap,
    plot_mtf_curve,
    plot_snr_vs_lux,
    save_figure,
    set_style,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR: Final[Path] = REPO_ROOT / "results" / "reports"
REPORT_NAME: Final[str] = "g4_metrics_results.json"
QUICK_REPORT_NAME: Final[str] = "g4_metrics_results_quick.json"
REPORT_MARKDOWN_NAME: Final[str] = "g4_metrics.md"

_SENSOR_PRESET: Final[Path] = PRESETS_DIR / "sensors" / "flagship_phone_50mp.json"
_ISP_PRESET: Final[Path] = PRESETS_DIR / "isp" / "default.json"

# The flat-field sweep holds exposure fixed and varies illuminance, so the
# signal and shot noise both change. At 10 ms the highest level remains below
# the ADC ceiling for the 0.5-reflectance flat patch.
_SWEEP_SHAPE: Final[tuple[int, int]] = (64, 64)
_SWEEP_EXPOSURE_S: Final[float] = 0.01
_FULL_SWEEP_LUX: Final[tuple[float, ...]] = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
_QUICK_SWEEP_LUX: Final[tuple[float, ...]] = (50.0, 200.0, 800.0, 3200.0)
_SWEEP_SCENE_SEED: Final[int] = 900
_SWEEP_SENSOR_SEED: Final[int] = 901
_SWEEP_FRAME_SEED: Final[int] = 902
_SWEEP_MARGIN: Final[int] = 8

_PASS_COLOUR: Final[str] = PALETTE[2]
_FAIL_COLOUR: Final[str] = PALETTE[1]


def _mtf_series(payload: dict[str, Any]) -> list[MTFPlotSeries]:
    """Wrap each measured MTF case in the plot value object."""
    return [
        MTFPlotSeries(
            label=case["name"],
            frequency_cp=case["frequency_cp"],
            measured_mtf=case["measured_mtf"],
            analytic_mtf=case["analytic_mtf"],
            mtf50_cp=case["measured_mtf50_cp"],
            mtf20_cp=case["measured_mtf20_cp"],
        )
        for case in payload["mtf"]["cases"]
    ]


def _flat_field_snr_sweep(
    *,
    quick: bool,
) -> tuple[list[float], dict[str, list[float]], list[dict[str, float]]]:
    """Measure flat-field SNR at several illuminance levels with sensor and ISP.

    The sensor is rebuilt for every level from the same seed, so the fixed
    pattern is identical across the sweep and only the light level changes.
    PDAF is disabled because its reduced-aperture sites bias a noise statistic
    and ``snr_channels`` has no mask parameter. The measurement uses the
    pre-tone linear buffer, not the display-encoded output.
    """
    sensor_config = load_sensor(_SENSOR_PRESET)
    sensor_config = sensor_config.model_copy(
        update={
            "pdaf": sensor_config.pdaf.model_copy(update={"pattern": "none"}),
        }
    )
    isp_config = load_isp(_ISP_PRESET)
    levels = list(_QUICK_SWEEP_LUX if quick else _FULL_SWEEP_LUX)
    roi = MetricsPatchROI(
        name="flat",
        x=_SWEEP_MARGIN,
        y=_SWEEP_MARGIN,
        width=_SWEEP_SHAPE[1] - 2 * _SWEEP_MARGIN,
        height=_SWEEP_SHAPE[0] - 2 * _SWEEP_MARGIN,
    )
    series: dict[str, list[float]] = {"R": [], "G": [], "B": [], "luma": []}
    rows: list[dict[str, float]] = []
    for lux in levels:
        scene = render(
            "flat_patch",
            illuminant="D65",
            lux=lux,
            shape=_SWEEP_SHAPE,
            oversampling=2,
            seed=_SWEEP_SCENE_SEED,
        )
        sensor = Sensor(sensor_config, seed=_SWEEP_SENSOR_SEED)
        frame = sensor.expose(
            scene.irradiance,
            _SWEEP_EXPOSURE_S,
            mode="LCG",
            seed=_SWEEP_FRAME_SEED,
        )
        rendered = ISP(isp_config).process(frame)
        linear = rendered.linear_rgb
        if linear is None:  # pragma: no cover - the ISP always sets it
            raise RuntimeError("the flat-field sweep produced no linear buffer")
        results = snr_channels(np.asarray(linear, dtype=np.float64), roi)
        for channel in series:
            series[channel].append(float(results[channel].snr_db))
        clipped = float(np.mean(frame.data >= 2**frame.bit_depth - 1))
        rows.append(
            {
                "lux": float(lux),
                "exposure_s": _SWEEP_EXPOSURE_S,
                "clipped_fraction": clipped,
                "luma_snr_db": float(results["luma"].snr_db),
            }
        )
    return levels, series, rows


def _plot_check_summary(payload: dict[str, Any]) -> Figure:
    """Draw a compact PASS/FAIL summary of every G4 check."""
    checks = payload["checks"]
    figure, axis = plt.subplots(figsize=(11.0, 0.28 * len(checks) + 1.0))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.5, len(checks) - 0.5)
    axis.axis("off")
    for index, check in enumerate(reversed(checks)):
        passed = bool(check["pass"])
        colour = _PASS_COLOUR if passed else _FAIL_COLOUR
        axis.text(0.0, index, check["name"], va="center", ha="left", fontsize=7.5)
        axis.text(0.62, index, check["measured_text"], va="center", ha="right", fontsize=7.5)
        axis.text(0.66, index, check["tolerance_text"], va="center", ha="left", fontsize=7.5)
        axis.text(
            0.97,
            index,
            "PASS" if passed else "FAIL",
            va="center",
            ha="right",
            fontsize=7.5,
            color=colour,
            fontweight="bold",
        )
    axis.set_title(f"G4 checks: {payload['status']}", loc="left", fontsize=10.0)
    figure.tight_layout()
    return figure


def _save_and_close(figure: Figure, path: Path) -> Path:
    """Save a figure and release it."""
    written = save_figure(figure, path)
    plt.close(figure)
    return written


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic, strictly finite JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_gallery(
    figures: Path,
    payload: dict[str, Any],
    sweep_lux: list[float],
    sweep_series: dict[str, list[float]],
    *,
    quick: bool,
) -> list[Path]:
    """Write the four metric figures and the gate-replica figures.

    Quick-mode names carry a ``_quick`` suffix so a smoke run with the default
    figure directory cannot overwrite the committed full gallery.
    """
    suffix = "_quick" if quick else ""
    written: list[Path] = []

    mtf_figure, _axes = plot_mtf_curve(_mtf_series(payload))
    written.append(_save_and_close(mtf_figure, figures / f"03_mtf_curves{suffix}.png"))

    snr_figure, _axes = plot_snr_vs_lux(sweep_lux, sweep_series)
    written.append(_save_and_close(snr_figure, figures / f"03_snr_vs_lux{suffix}.png"))

    chart = payload["measure_all"]["chart"]
    heatmap_figure, _axes = plot_delta_e_heatmap(
        chart["patch_names"],
        chart["per_patch_delta_e2000"],
        title=f"Synthetic ColorChecker CIEDE2000 (mean {chart['mean_delta_e2000']:.2f})",
    )
    written.append(_save_and_close(heatmap_figure, figures / f"03_delta_e_heatmap{suffix}.png"))

    acutance_figure, _axes = plot_acutance(
        payload["acutance_sweep"]["sigma_px"],
        payload["acutance_sweep"]["acutance"],
    )
    written.append(_save_and_close(acutance_figure, figures / f"03_acutance_vs_blur{suffix}.png"))

    # Gate-replica figures carry the g4 prefix. Quick mode suffixes them too;
    # it must never overwrite a committed full-resolution figure.
    written.append(_save_and_close(_mtf_replica(payload), figures / f"g4_mtf_curves{suffix}.png"))
    written.append(
        _save_and_close(
            _snr_replica(sweep_lux, sweep_series),
            figures / f"g4_snr_vs_lux{suffix}.png",
        )
    )
    written.append(
        _save_and_close(_delta_e_replica(payload), figures / f"g4_delta_e_heatmap{suffix}.png")
    )
    written.append(
        _save_and_close(_acutance_replica(payload), figures / f"g4_acutance_vs_blur{suffix}.png")
    )
    written.append(
        _save_and_close(
            _plot_check_summary(payload),
            figures / f"g4_check_summary{suffix}.png",
        )
    )
    return written


def _mtf_replica(payload: dict[str, Any]) -> Figure:
    """Draw a fresh MTF figure for the gate-replica name."""
    figure, _axes = plot_mtf_curve(_mtf_series(payload))
    return figure


def _snr_replica(lux: list[float], series: dict[str, list[float]]) -> Figure:
    """Draw a fresh SNR-versus-lux figure for the gate-replica name."""
    figure, _axes = plot_snr_vs_lux(lux, series)
    return figure


def _delta_e_replica(payload: dict[str, Any]) -> Figure:
    """Draw a fresh delta-E heatmap for the gate-replica name."""
    chart = payload["measure_all"]["chart"]
    figure, _axes = plot_delta_e_heatmap(
        chart["patch_names"],
        chart["per_patch_delta_e2000"],
        title=f"Synthetic ColorChecker CIEDE2000 (mean {chart['mean_delta_e2000']:.2f})",
    )
    return figure


def _acutance_replica(payload: dict[str, Any]) -> Figure:
    """Draw a fresh acutance figure for the gate-replica name."""
    figure, _axes = plot_acutance(
        payload["acutance_sweep"]["sigma_px"],
        payload["acutance_sweep"]["acutance"],
    )
    return figure


def _relative_preset(path: Path) -> str:
    """Return a repository-relative preset path when the checkout allows it."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:  # pragma: no cover - installed-package fallback
        return path.as_posix()


def _lux_sweep_section(
    *,
    quick: bool,
    levels: list[float],
    series: dict[str, list[float]],
    rows: list[dict[str, float]],
) -> dict[str, Any]:
    """Package the illustrative lux sweep for the JSON report and Markdown."""
    snr_by_channel: dict[str, list[float]] = {}
    for channel, values in series.items():
        snr_by_channel[channel] = [float(value) for value in values]
    return {
        "gate_evidence": False,
        "note": (
            "Gallery illustration, not a G4 check: fixed exposure, flagship "
            "sensor and default ISP, PDAF disabled, linear RenderedImage.linear_rgb "
            "ROI measured with snr_channels."
        ),
        "quick": quick,
        "shape": list(_SWEEP_SHAPE),
        "exposure_s": _SWEEP_EXPOSURE_S,
        "sensor_preset": _relative_preset(_SENSOR_PRESET),
        "isp_preset": _relative_preset(_ISP_PRESET),
        "pdaf_disabled": True,
        "lux": [float(value) for value in levels],
        "snr_db": snr_by_channel,
        "clipped_fraction": [float(row["clipped_fraction"]) for row in rows],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gallery and G4 measurements, write the artifacts, return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use smaller MTF/SNR/chart grids for a smoke test (not gate evidence)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=(
            f"directory for {REPORT_NAME} and {REPORT_MARKDOWN_NAME} "
            f"(default: results/reports); quick mode writes {QUICK_REPORT_NAME}"
        ),
    )
    args = parser.parse_args(argv)

    set_style()
    figures = figure_dir(REPO_ROOT / "results" / "figures")
    reports = args.report_dir.expanduser()
    reports.mkdir(parents=True, exist_ok=True)

    print("camsim example 03: metrics gallery and G4 validation")
    print(f"figure directory: {figures}")
    print(f"report directory: {reports}")
    print(f"mode: {'quick (not gate evidence)' if args.quick else 'full gate evidence'}")

    payload = measure_g4(quick=args.quick)
    sweep_lux, sweep_series, sweep_rows = _flat_field_snr_sweep(quick=args.quick)
    # The lux sweep is gallery-only, so it travels in the report payload with an
    # explicit `gate_evidence: false` marker; measure_g4's gate checks never see it.
    report_payload = dict(payload)
    report_payload["lux_sweep"] = _lux_sweep_section(
        quick=args.quick,
        levels=sweep_lux,
        series=sweep_series,
        rows=sweep_rows,
    )
    gate_passed = payload["status"] == "PASS"

    print("\nAnalytic MTF cases:")
    for case in payload["mtf"]["cases"]:
        print(
            f"  {case['name']}: MTF50 {case['measured_mtf50_cp']:.5f} vs "
            f"{case['reference_mtf50_cp']:.5f} cp "
            f"({100.0 * case['relative_error_mtf50']:+.3f} %); MTF20 "
            f"{case['measured_mtf20_cp']:.5f} vs {case['reference_mtf20_cp']:.5f} cp "
            f"({100.0 * case['relative_error_mtf20']:+.3f} %)"
        )
    print("\nFlat-field SNR sweep (PDAF disabled, linear RGB ROI):")
    for row in sweep_rows:
        print(
            f"  {row['lux']:6.0f} lux, {1000.0 * row['exposure_s']:.1f} ms: "
            f"luma {row['luma_snr_db']:6.2f} dB, "
            f"clipped {100.0 * row['clipped_fraction']:.2f} %"
        )

    written = _write_gallery(
        figures,
        payload,
        sweep_lux,
        sweep_series,
        quick=args.quick,
    )

    if args.quick:
        report_path = reports / QUICK_REPORT_NAME
        _write_json(report_path, report_payload)
        print(f"\nQuick results JSON: {report_path}")
    else:
        report_path = reports / REPORT_NAME
        _write_json(report_path, report_payload)
        markdown_path = reports / REPORT_MARKDOWN_NAME
        markdown_path.write_text(render_g4_report(report_payload), encoding="utf-8")
        print(f"\nResults JSON: {report_path}")
        print(f"Report: {markdown_path}")

    print("\nWritten figures:")
    for path in written:
        print(f"  {path.name} ({path.stat().st_size / 1024:.0f} KiB)")

    print("\nG4 checks:")
    for check in payload["checks"]:
        status = "PASS" if bool(check["pass"]) else "FAIL"
        print(f"  [{status}] {check['name']}: {check['measured_text']}")

    if args.quick:
        print("\nQuick mode: checks are informational and are not G4 gate evidence.")
        return 0
    if not gate_passed:
        failures = sum(1 for check in payload["checks"] if not bool(check["pass"]))
        print(f"\nG4 gate: FAIL ({failures} check(s) outside tolerance)")
        return 1
    print("\nG4 gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
