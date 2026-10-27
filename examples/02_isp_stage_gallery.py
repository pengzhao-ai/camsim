"""Example 02: ISP stage gallery and the G3 validation report.

This walkthrough runs the shipped camera from a spectral ColorChecker scene to
a rendered image and opens the ISP's black box stage by stage. The physics in
the figures and the report:

* **Linear light until the tone stage.** DPC, black-level correction, lens
  shading, white balance, demosaicing, colour correction, denoise, and
  sharpening all operate on values proportional to radiance. The tone stage is
  the first non-linear step; it applies an exposure trim in linear light, a
  change of primaries when Display P3 is selected, a display clamp, and the
  sRGB opto-electronic transfer function.
* **CFA reconstruction.** A Bayer mosaic records one colour per site. Bilinear
  interpolation averages same-colour neighbours independently and produces
  zipper and false-colour artefacts on edges; Malvar-He-Cutler adds a
  gradient-dependent correction borrowed from the measured green plane and
  reconstructs detail with the same 5x5 support. The star comparison shows
  both against the full-resolution reference.
* **White balance.** A neutral surface under a coloured illuminant does not
  produce equal camera channels. The gray-world estimator equalises per-CFA
  channel means, and a known injected cast should be inverted by the same
  estimator.
* **Colour correction and ΔE.** The camera's broad overlapping filters are not
  a colorimetric basis. A 3x3 matrix fitted on the 24 ColorChecker patches
  maps camera RGB to linear sRGB; CIEDE2000 between the rendered display and
  the spectrally computed reference measures the residual. A 3x3 matrix can be
  exact only when the sensor's spectral sensitivities are a linear combination
  of the CIE observer (the Luther condition).
* **Tone and trace.** The pre-tone linear trace stage, the tone configuration,
  the exposure compensation, and the output primaries must reproduce both the
  rendered display array and the unclipped ``linear_rgb`` buffer.

The example measures the same quantities as ``tests/test_example_02.py`` via
the shared :func:`camsim.characterization.measure_g3`; the tests and the report
therefore cannot disagree. Full mode writes
``results/reports/g3_isp_results.json`` and ``results/reports/g3_isp.md`` plus
the ``02_*.png`` and ``g3_*.png`` gallery under ``CAMSIM_FIG_DIR`` (default
``results/figures``). ``--quick`` uses a smaller chart and a smaller star,
writes only ``g3_isp_results_quick.json``, and is **not gate evidence**; it
never overwrites the committed gate JSON or Markdown.

**CFA colour correction.** The shipped flagship preset uses synthetic
conventional Bayer bands (blue ~448 nm, green ~542 nm, red ~610 nm) with
moderate out-of-band leakage. The original broad, nearly collinear CFA curves
were replaced after the G3 gate showed that no 3x3 matrix could colour-correct
them. With the conventional bands the in-sample 3x3 CCM meets the ΔE2000
thresholds at both illuminants, and the Luther-condition residual falls to
about 0.13. The fit is still in-sample, which is stated as a limitation.

This is a simulated validation, not a certified ISO 12233 or EMVA measurement.
The formal ISO 12233 MTF50 check and the ``Camera.auto_expose`` convergence
check are deferred: the metrics harness does not exist until Phase 4, and the
``Camera`` orchestration arrives with the Phase 5 consolidation.

Failure artefacts guarded here: a quick run that overwrites gate evidence; a
contact sheet that auto-normalises each panel and hides a white-balance or
black-level error; a ColorChecker target at the wrong exposure (clipped
highlights) or wrong white point (a colour cast); a CCM fitted against an
un-normalised target (a matrix that cannot reach the target white); a demosaic
comparison on a grey image (trivially exact); a stale report that no longer
matches the JSON; and a report that claims an MTF check that was never run.
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
from numpy.typing import NDArray

from camsim.characterization import G3Gallery, build_gallery, measure_g3, render_g3_report
from camsim.viz import (
    PALETTE,
    figure_dir,
    plot_demosaic_comparison,
    plot_pipeline_trace,
    plot_zoom_crops,
    save_figure,
    set_style,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR: Final[Path] = REPO_ROOT / "results" / "reports"
REPORT_NAME: Final[str] = "g3_isp_results.json"
QUICK_REPORT_NAME: Final[str] = "g3_isp_results_quick.json"
REPORT_MARKDOWN_NAME: Final[str] = "g3_isp.md"

_PASS_COLOUR: Final[str] = PALETTE[2]
_FAIL_COLOUR: Final[str] = PALETTE[1]


def _plot_delta_e(payload: dict[str, Any]) -> Figure:
    """Draw per-patch ΔE2000 for both illuminants with the gate thresholds."""
    d65 = payload["colorchecker"]["illuminants"]["D65"]
    warm = payload["colorchecker"]["illuminants"]["3000K"]
    patches = d65["patch_names"]
    positions = np.arange(len(patches), dtype=np.float64)
    width = 0.4
    figure, axis = plt.subplots(figsize=(12.0, 4.8))
    axis.bar(
        positions - width / 2,
        d65["delta_e2000"],
        width,
        color=PALETTE[0],
        label=f"D65 (mean {d65['delta_e2000_mean']:.2f})",
    )
    axis.bar(
        positions + width / 2,
        warm["delta_e2000"],
        width,
        color=PALETTE[4],
        label=f"3000 K (mean {warm['delta_e2000_mean']:.2f})",
    )
    axis.axhline(2.0, color=PALETTE[0], linestyle="--", linewidth=1.0, label="D65 gate 2.0")
    axis.axhline(4.0, color=PALETTE[4], linestyle=":", linewidth=1.0, label="3000 K gate 4.0")
    axis.set_xticks(positions)
    axis.set_xticklabels(patches, rotation=55, ha="right", fontsize=7.5)
    axis.set_ylabel("CIEDE2000")
    axis.set_title("ColorChecker ΔE2000 per patch (in-sample 3x3 CCM)")
    axis.legend(loc="upper right", fontsize=8.0)
    axis.set_ylim(bottom=0.0)
    figure.tight_layout()
    return figure


def _plot_awb(payload: dict[str, Any]) -> Figure:
    """Draw measured and analytic white-balance gains for both illuminants."""
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    channels = ("R", "G", "B")
    positions = np.arange(3, dtype=np.float64)
    width = 0.36
    handles: list[Any] = []
    for axis, name, title in zip(axes, ("D65", "3000K"), ("D65", "3000 K"), strict=True):
        section = payload["awb"]["illuminants"][name]
        expected = axis.bar(
            positions - width / 2,
            section["expected_gains"],
            width,
            color=PALETTE[0],
            label="analytic expectation",
        )
        measured = axis.bar(
            positions + width / 2,
            section["measured_gains"],
            width,
            color=PALETTE[4],
            label="gray-world estimate",
        )
        if not handles:
            handles = [expected, measured]
        peak = max(
            max(section["expected_gains"]),
            max(section["measured_gains"]),
        )
        ylim = 1.18 * peak
        for index, error in enumerate(section["relative_errors"]):
            axis.text(
                positions[index],
                max(section["expected_gains"][index], section["measured_gains"][index])
                + 0.035 * ylim,
                f"{100.0 * error:.2f}%",
                ha="center",
                fontsize=7.5,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(channels)
        axis.set_title(title)
        axis.set_ylim(0.0, ylim)
    axes[0].set_ylabel("White-balance gain (green = 1)")
    figure.suptitle("AWB: analytic spectral expectation versus the gray-world estimate")
    figure.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5)
    figure.subplots_adjust(top=0.86, bottom=0.20, left=0.09, right=0.98, wspace=0.10)
    return figure


def _plot_check_summary(payload: dict[str, Any]) -> Figure:
    """Draw a compact PASS/FAIL summary of every G3 check."""
    checks = payload["checks"]
    figure, axis = plt.subplots(figsize=(11.0, 0.28 * len(checks) + 1.0))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.5, len(checks) - 0.5)
    axis.axis("off")
    for index, check in enumerate(reversed(checks)):
        passed = bool(check["pass"])
        colour = _PASS_COLOUR if passed else _FAIL_COLOUR
        axis.text(0.0, index, check["name"], va="center", ha="left", fontsize=7.5)
        axis.text(0.66, index, check["measured_text"], va="center", ha="right", fontsize=7.5)
        axis.text(0.72, index, check["tolerance_text"], va="center", ha="left", fontsize=7.5)
        axis.text(
            0.95,
            index,
            "PASS" if passed else "FAIL",
            va="center",
            ha="right",
            fontsize=7.5,
            color=colour,
            fontweight="bold",
        )
    axis.set_title(f"G3 checks: {payload['status']}", loc="left", fontsize=10.0)
    figure.tight_layout()
    return figure


def _plot_rccc(image: NDArray[np.float32]) -> Figure:
    """Draw the qualitative RCCC baseline reconstruction."""
    figure, axis = plt.subplots(figsize=(4.8, 4.8))
    axis.imshow(image, interpolation="nearest", vmin=0.0, vmax=1.0)
    axis.set_title("RCCC baseline reconstruction\n(shape/linearity smoke only)")
    axis.set_xticks([])
    axis.set_yticks([])
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
    gallery: G3Gallery,
    *,
    quick: bool,
) -> list[Path]:
    """Write the stage-gallery figures and the gate-summary figures.

    Quick-mode figure names carry a ``_quick`` suffix so a smoke run with the
    default figure directory cannot overwrite the committed full-resolution
    gallery. The test only requires ``02_*.png`` files, which the suffixed
    names satisfy.
    """
    written: list[Path] = []
    suffix = "_quick" if quick else ""

    figure, _axes = plot_pipeline_trace(gallery.stages)
    written.append(_save_and_close(figure, figures / f"02_isp_stage_contact_sheet{suffix}.png"))

    figure, _axes = plot_zoom_crops(gallery.stages)
    written.append(_save_and_close(figure, figures / f"02_isp_stage_zoom{suffix}.png"))

    figure, _axes = plot_demosaic_comparison(
        gallery.star_reference,
        gallery.star_bilinear,
        gallery.star_mhc,
    )
    written.append(_save_and_close(figure, figures / f"02_isp_demosaic_comparison{suffix}.png"))

    delta_e_figure = _plot_delta_e(payload)
    written.append(
        _save_and_close(
            delta_e_figure,
            figures / f"02_isp_colorchecker_delta_e{suffix}.png",
        )
    )
    if not quick:
        # The gate figures reuse the same deterministic plot helpers. A second
        # save of an already-closed figure would fail, so draw a fresh one.
        written.append(_save_and_close(_plot_delta_e(payload), figures / "g3_delta_e.png"))

    awb_figure = _plot_awb(payload)
    written.append(_save_and_close(awb_figure, figures / f"02_isp_awb_gains{suffix}.png"))
    if not quick:
        written.append(_save_and_close(_plot_awb(payload), figures / "g3_awb_gains.png"))

    written.append(
        _save_and_close(
            _plot_rccc(gallery.rccc_image),
            figures / f"02_isp_rccc_baseline{suffix}.png",
        )
    )
    if not quick:
        written.append(
            _save_and_close(_plot_check_summary(payload), figures / "g3_check_summary.png")
        )
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gallery and G3 measurements, write the artifacts, return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a smaller chart and star for a smoke test (not gate evidence)",
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

    print("camsim example 02: ISP stage gallery and G3 validation")
    print(f"figure directory: {figures}")
    print(f"report directory: {reports}")
    print(f"mode: {'quick (not gate evidence)' if args.quick else 'full gate evidence'}")

    payload = measure_g3(quick=args.quick)
    gallery = build_gallery(quick=args.quick)
    gate_passed = payload["status"] == "PASS"

    print("\nISP stage order:")
    print("  " + " -> ".join(gallery.stages))
    print("\nColorChecker mean deltaE2000:")
    for name in ("D65", "3000K"):
        section = payload["colorchecker"]["illuminants"][name]
        print(f"  {name}: {section['delta_e2000_mean']:.3f} (max {section['delta_e2000_max']:.3f})")

    written = _write_gallery(figures, payload, gallery, quick=args.quick)

    if args.quick:
        report_path = reports / QUICK_REPORT_NAME
        _write_json(report_path, payload)
        print(f"\nQuick results JSON: {report_path}")
    else:
        report_path = reports / REPORT_NAME
        _write_json(report_path, payload)
        quick_payload = measure_g3(quick=True)
        report_payload = dict(payload)
        report_payload["quick_reference"] = {
            "status": quick_payload["status"],
            "delta_e2000_mean": {
                name: quick_payload["colorchecker"]["illuminants"][name]["delta_e2000_mean"]
                for name in ("D65", "3000K")
            },
            "note": "quick smoke run; not gate evidence",
        }
        markdown_path = reports / REPORT_MARKDOWN_NAME
        markdown_path.write_text(render_g3_report(report_payload), encoding="utf-8")
        print(f"\nResults JSON: {report_path}")
        print(f"Report: {markdown_path}")

    print("\nWritten figures:")
    for path in written:
        print(f"  {path.name} ({path.stat().st_size / 1024:.0f} KiB)")

    print("\nG3 checks:")
    for check in payload["checks"]:
        status = "PASS" if bool(check["pass"]) else "FAIL"
        print(f"  [{status}] {check['name']}: {check['measured_text']}")

    if args.quick:
        print("\nQuick mode: checks are informational and are not G3 gate evidence.")
        return 0
    if not gate_passed:
        failures = sum(1 for check in payload["checks"] if not bool(check["pass"]))
        print(f"\nG3 gate: FAIL ({failures} check(s) outside tolerance)")
        return 1
    print("\nG3 gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
