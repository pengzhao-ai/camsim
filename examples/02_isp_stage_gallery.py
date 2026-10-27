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

**Honest limitation.** The shipped flagship preset uses synthetic, very broad
CFA curves. Its red and green spectral sensitivities are highly correlated and
the best in-sample 3x3 CCM cannot reproduce saturated ColorChecker patches; the
ColorChecker ΔE2000 criterion fails at both illuminants. The report records the
measured FAIL instead of relaxing the design threshold. The other G3 criteria
(white balance, demosaic, tone, trace, linearity, LSC, denoise, sharpen, RCCC)
pass their stated bounds.

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

from camsim.characterization import G3Gallery, build_gallery, measure_g3
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


def _format_matrix_row(row: Sequence[float]) -> str:
    """Format one CCM row as comma-separated signed coefficients."""
    return ", ".join(f"{value:+.6f}" for value in row)


def _render_report(payload: dict[str, Any]) -> str:
    """Render the deterministic G3 Markdown report from the measured payload."""
    checks = payload["checks"]
    d65 = payload["colorchecker"]["illuminants"]["D65"]
    warm = payload["colorchecker"]["illuminants"]["3000K"]
    environment = payload["environment"]
    presets = payload["presets"]
    scene = payload["scene"]
    luther = payload["sensor_colorimetry"]
    awb = payload["awb"]
    demosaic = payload["demosaic"]
    tone = payload["tone"]
    trace = payload["trace"]
    linearity = payload["linearity"]
    lsc = payload["lsc"]
    denoise = payload["denoise"]
    sharpen = payload["sharpen"]
    rccc = payload["rccc"]

    lines: list[str] = [
        "# G3 - ISP validation",
        "",
        f"**Status: {payload['status']}**",
        "",
        "G3 closes Phase 3 by running the complete image-signal-processing chain",
        "from a spectral ColorChecker scene through the sensor to a rendered",
        "display image, then measuring colour accuracy, white balance, demosaic",
        "agreement, tone, stage-trace consistency, linearity, lens shading,",
        "denoising, sharpening, and the RCCC baseline. The measurements are",
        "produced by `camsim.characterization.measure_g3`; this report and the",
        "machine-readable JSON are generated by the same full example run, so a",
        "stale report is detectable.",
        "",
        "**The ColorChecker colour-accuracy criterion is not met.** The design",
        "gate is mean ΔE2000 < 2.0 under D65 and < 4.0 under 3000 K after an",
        "in-sample 3x3 CCM. The measured means are "
        f"{d65['delta_e2000_mean']:.3f} and {warm['delta_e2000_mean']:.3f}, so the",
        "two colour checks are FAIL. The root cause is the shipped sensor preset:",
        "its synthetic CFA curves are deliberately broad, the red and green",
        "spectral sensitivities are highly correlated, and the best 3x3 linear",
        "fit to the CIE 1931 observer leaves a "
        f"{100.0 * luther['luther_relative_residual']:.1f} % spectral residual",
        "(the Luther condition). No 3x3 matrix can reproduce saturated patches",
        "with these sensitivities. The thresholds are **not** relaxed in this",
        "report; a controller ruling is needed to re-author the CFA curves or to",
        "revisit the design threshold. Every other criterion measured in this",
        "task passes; the checks deferred to later phases are listed under",
        "limitations.",
        "",
        "All evidence is simulated. It is not a certified ISO 12233 or EMVA 1288",
        "measurement, and the flagship preset is a representative engineering",
        "design point rather than a measurement of one commercial part.",
        "",
        "## Environment and reproduction",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Python | `{environment['python']}` |",
        f"| camsim | `{environment['camsim']}` |",
        f"| NumPy | `{environment['numpy']}` |",
        f"| SciPy | `{environment['scipy']}` |",
        f"| colour-science | `{environment['colour_science']}` |",
        f"| colour-demosaicing | `{environment['colour_demosaicing']}` |",
        f"| Platform | `{environment['platform']}` |",
        f"| Sensor preset | `{presets['sensor']}` |",
        f"| ISP preset | `{presets['isp']}` |",
        f"| Sensor config hash | `{presets['sensor_config_hash']}` |",
        f"| ISP config hash | `{presets['isp_config_hash']}` |",
        "| Artifact generator | `uv run python examples/02_isp_stage_gallery.py` |",
        "| Machine-readable results | `results/reports/g3_isp_results.json` |",
        "| Gate tests | `uv run pytest tests/test_example_02.py -v` |",
        "| Quick smoke | `CAMSIM_FIG_DIR=/tmp/g3_quick_figures uv run python "
        "examples/02_isp_stage_gallery.py --quick --report-dir /tmp/g3_quick_reports` "
        "(not gate evidence; suffixed figures) |",
        "",
        "The full run uses a "
        f"{scene['colorchecker_shape'][1]} x {scene['colorchecker_shape'][0]}"
        f" ColorChecker at {scene['lux']:.0f} lux with {scene['oversampling']}x",
        "oversampling, the shipped flagship sensor at LCG, and the D65 and 3000 K",
        "illuminants. Exposures are calibrated from a dim probe so the brightest",
        "white-patch channel reaches about "
        f"{100.0 * scene['target_white_fraction']:.0f} % of the ADC white span:",
        f"D65 {1000.0 * scene['exposures_s']['D65']:.2f} ms and 3000 K "
        f"{1000.0 * scene['exposures_s']['3000K']:.2f} ms. The achieved white",
        "fractions are "
        f"{100.0 * scene['white_fractions']['D65']:.1f} % and "
        f"{100.0 * scene['white_fractions']['3000K']:.1f} %.",
        "",
        "## ISP chain and linear-space invariant",
        "",
        "The enforced order is `DPC -> BLC -> LSC -> AWB -> demosaic -> CCM ->",
        "denoise -> sharpen -> tone -> output`. DPC, BLC, LSC, AWB, demosaicing,",
        "the CCM, denoise, and sharpen operate in linear float32 light. BLC",
        "subtracts the 64 DN pedestal and divides by the 959 DN white span, so",
        "ADC saturation maps to 1.0. The tone stage applies the exposure trim in",
        "linear light, converts to Display P3 when requested, clamps to the",
        "display range, and applies the sRGB OETF; it is the only non-linear",
        "step. The linear pre-tone scaling check verifies this invariant directly.",
        "",
        "## Measurement methodology",
        "",
        "**Spectral scene to RAW.** `camsim.scene.render` multiplies ColorChecker",
        "reflectance by the photometrically normalised illuminant and the lens",
        "term, then boxes the oversampled target down to the sensor grid. The",
        "sensor integrates QE x IR-cut x CFA x photon flux, adds shot, read, and",
        "fixed-pattern noise, applies conversion and analog gain, and quantises",
        "to 10-bit codes with a 64 DN pedestal. Every exposure uses a fixed seed.",
        "",
        "**White balance and camera RGB.** The analytic per-channel response to a",
        "perfect diffuser under each illuminant gives green-normalised manual AWB",
        "gains. Using the perfect diffuser, rather than the chart's white patch,",
        "is what makes a row-sum CCM neutral-preserving: the reference Lab scale",
        "is defined by the perfect diffuser. The pipeline then runs with the CCM",
        "disabled and the trace enabled, and the 24 patch means are taken from",
        "the demosaiced, white-balanced linear RGB buffer.",
        "",
        "**Target linear sRGB.** The reference Lab is converted back to XYZ with",
        "the scene illuminant's white point, adapted to CIE D65 with the Bradford",
        "transform, and converted to **linear** sRGB with",
        "`colour.XYZ_to_RGB(..., apply_cctf_encoding=False)`. The perfect-diffuser",
        "white therefore maps to (1, 1, 1); the camera RGB is divided by the",
        "measured perfect-diffuser level before fitting, and the tone stage's",
        "exposure compensation restores the absolute scale.",
        "",
        "**CCM and ΔE2000.** One ridge-regularised 3x3 matrix per illuminant is",
        "fitted with `camsim.isp.ccm.fit_ccm(..., ridge=1e-4, row_sum=True)` on",
        "the same 24 patches used for evaluation (in-sample calibration-scene",
        "validation). The rendered sRGB display is decoded with the sRGB EOTF to",
        "XYZ D65 and CIE Lab, and CIEDE2000 is computed per patch against the",
        "reference Lab. In-sample fitting measures whether the ISP can represent",
        "the calibration scene at all; it does not measure generalisation to a",
        "held-out chart or a different illuminant.",
        "",
        "**AWB, demosaic, tone, and trace.** The gray-world estimator runs on a",
        "50 % neutral flat field under each illuminant; expected gains come from",
        "the analytic spectral response, and a second check injects a known",
        "per-CFA cast (R x1.6, G x1.0, B x0.7) and requires the estimator to",
        "invert it. Bilinear and MHC are compared with `colour-demosaicing` on a",
        "synthetic Siemens-star RGB mosaic with independent R, G, and B",
        "structure. The sRGB OETF is compared with `colour.cctf_encoding`. The",
        "stage trace is validated by applying the tone configuration to the last",
        "linear trace stage and comparing both the display array and the",
        "unclipped `linear_rgb` buffer at +0.5 EV in Display P3.",
        "",
        "## Verdict summary",
        "",
        "| Check | Measured | Tolerance | Result |",
        "|---|---|---|---|",
    ]
    for check in checks:
        status = "PASS" if bool(check["pass"]) else "FAIL"
        lines.append(
            f"| {check['name']} | {check['measured_text']} | {check['tolerance_text']} | {status} |"
        )

    lines.extend(
        [
            "",
            "## 1. End-to-end ColorChecker colour",
            "",
            "The fitted matrices (rows sum to one to machine precision) are:",
            "",
            "| Illuminant | Row 1 | Row 2 | Row 3 |",
            "|---|---|---|---|",
        ]
    )
    lines.append(
        f"| D65 | {_format_matrix_row(d65['ccm'][0])} | "
        f"{_format_matrix_row(d65['ccm'][1])} | {_format_matrix_row(d65['ccm'][2])} |"
    )
    lines.append(
        f"| 3000K | {_format_matrix_row(warm['ccm'][0])} | "
        f"{_format_matrix_row(warm['ccm'][1])} | {_format_matrix_row(warm['ccm'][2])} |"
    )
    lines.extend(
        [
            "",
            "Per-patch CIEDE2000 between the rendered display and the spectrally",
            "computed reference:",
            "",
            "| Patch | D65 deltaE2000 | 3000 K deltaE2000 |",
            "|---|---:|---:|",
        ]
    )
    for index, patch in enumerate(d65["patch_names"]):
        lines.append(
            f"| {patch} | {d65['delta_e2000'][index]:.3f} | {warm['delta_e2000'][index]:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The means are {d65['delta_e2000_mean']:.3f} (D65) and",
            f"{warm['delta_e2000_mean']:.3f} (3000 K); the design limits are 2.0",
            "and 4.0. The pre-clip means are",
            f"{d65['delta_e2000_preclip_mean']:.3f} and",
            f"{warm['delta_e2000_preclip_mean']:.3f}, so display clipping is not",
            "the cause. The CCM fit RMS residuals are "
            f"{d65['ccm_residual']:.4f} and {warm['ccm_residual']:.4f} linear RGB;",
            "the diagonal/off-diagonal coefficients reach magnitudes well outside",
            "the usual 0.5-1.5 / -0.5-0.5 range, which is itself a symptom of the",
            "spectral mismatch.",
            "",
            "**Diagnostic.** The Luther-condition relative residual is "
            f"{luther['luther_relative_residual']:.3f}; the CFA projection",
            "condition number is "
            f"{luther['condition_number']:.1f} and the R/G correlation is "
            f"{luther['channel_correlation'][0][1]:.3f}. The sensor's red and",
            "green filters overlap almost completely, so a 3x3 matrix cannot",
            "separate saturated red, orange, and yellow patches from the CIE",
            "observer. This is a sensor-CFA design finding, not an ISP",
            "implementation error: the CCM is fitted and evaluated in-sample.",
            "",
            "![ColorChecker delta E](../figures/g3_delta_e.png)",
            "",
            "## 2. White balance",
            "",
            "Expected gains are the analytic spectral response ratios; measured",
            "gains are the gray-world estimate on a uniform 50 % flat field.",
            "",
            "| Illuminant | Channel | Expected gain | Measured gain | Relative error | Result |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    check_by_name = {check["name"]: check for check in checks}
    for name, label in (("D65", "D65"), ("3000K", "3000 K")):
        section = awb["illuminants"][name]
        for index, channel in enumerate(("R", "G", "B")):
            if channel == "G":
                result = "PASS"
            else:
                result = (
                    "PASS"
                    if check_by_name[f"AWB gain relative error, {channel} ({label})"]["pass"]
                    else "FAIL"
                )
            lines.append(
                f"| {name} | {channel} | {section['expected_gains'][index]:.6f} | "
                f"{section['measured_gains'][index]:.6f} | "
                f"{section['relative_errors'][index]:.6f} | {result} |"
            )
    injected = awb["injected_scaling"]
    lines.extend(
        [
            "",
            f"The injected-cast check (R x{injected['injection'][0]:g}, "
            f"G x{injected['injection'][1]:g}, B x{injected['injection'][2]:g})",
            "recovers the expected gains with a maximum relative error of "
            f"{100.0 * injected['max_relative_error']:.2f} %. All non-green",
            "channels are within the 5 % tolerance.",
            "",
            "![White-balance gains](../figures/g3_awb_gains.png)",
            "",
            "## 3. Demosaic, tone, trace, and linearity",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| Bilinear max abs vs colour-demosaicing | {demosaic['bilinear_max_abs']:.3e} |",
            f"| MHC max abs vs colour-demosaicing | {demosaic['mhc_max_abs']:.3e} |",
            f"| sRGB OETF max abs vs colour-science | {tone['max_abs']:.3e} |",
            f"| Trace display reproduction max abs | {trace['tone_max_abs']:.3e} |",
            f"| Trace linear_rgb reproduction max abs | {trace['linear_rgb_max_abs']:.3e} |",
            "| Linear pre-tone scaling max relative error | "
            f"{linearity['max_relative_error']:.3e} |",
            "",
            "The trace stage order in the Display P3 validation is "
            f"`{' -> '.join(trace['stage_names'])}`. The tone and linear-buffer",
            "comparisons are exact to float32 because the trace records the same",
            "buffers the pipeline consumed; they validate the stage bookkeeping",
            "and the exposure/P3 order, while the independent sRGB OETF check",
            "validates the transfer function against the published standard.",
            "",
            "## 4. Lens shading, denoise, and sharpen",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| LSC flat-field residual before correction | {lsc['residual_before']:.4f} |",
            f"| LSC flat-field residual after correction | {lsc['residual_after']:.4f} |",
            "| Denoise SNR before / after (dB) | "
            f"{denoise['snr_before_db']:.2f} / {denoise['snr_after_db']:.2f} |",
            f"| Denoise SNR gain (dB) | {denoise['snr_gain_db']:.2f} |",
            f"| Denoise edge-contrast relative change | {denoise['edge_preservation_error']:.4f} |",
            f"| Sharpen edge-contrast gain | {sharpen['contrast_gain']:.4f} |",
            f"| Sharpen positive overshoot | {sharpen['overshoot']:.4f} |",
            "",
            "The LSC measurement calibrates a gain map from one noisy flat field",
            "and applies it to an independent flat field; the residual is the",
            "max-minus-min range divided by the mean. The exact-reciprocal check",
            "is the Task 18 unit test",
            "`tests/test_isp_lsc.py::test_lsc_flat_field_corrects_vignetting`.",
            "Denoise and sharpen are measured on synthetic step and flat regions",
            "with explicit seeds. **Formal ISO 12233 MTF50 and MTF20 validation is",
            "deferred to G4**: the independent metrics harness does not exist yet,",
            "so G3 claims edge-contrast preservation and positive overshoot, not",
            "an MTF number.",
            "",
            "## 5. RCCC baseline",
            "",
            f"The RCCC frame is {rccc['shape'][1]} x {rccc['shape'][0]} pixels and",
            f"renders to shape {rccc['shape']} with {rccc['channels']} channels,",
            f"values in [{rccc['min']:.3f}, {rccc['max']:.3f}], all finite. The",
            "baseline uses the documented `(R, C, C)` luma placeholder; AWB and",
            "CCM are skipped by the pipeline. It is a shape/linearity/smoke test",
            "only, and the Bayer ΔE and MTF thresholds do not apply.",
            "",
            "![RCCC baseline](../figures/02_isp_rccc_baseline.png)",
            "",
            "## 6. Stage unit-test evidence",
            "",
            "DPC, BLC, AWB, and CCM are covered by focused unit tests rather than",
            "re-measured here: `tests/test_isp_dpc.py` checks same-CFA",
            "interpolation, PDAF safety, every Bayer orientation, and exact",
            "preservation of unaffected pixels; `tests/test_isp_blc.py` checks",
            "pedestal subtraction, the 2x2 pattern, affine behaviour about the",
            "pedestal, and the white-span normalisation; `tests/test_isp_awb.py`",
            "checks the three estimators, per-CFA statistics, and gain",
            "normalisation; `tests/test_isp_ccm.py` checks recovery of a known",
            "matrix, machine-precision row sums, ridge regularisation, and ΔE",
            "improvement. The pipeline suite",
            "`tests/test_isp_pipeline.py::test_blc_stage_normalizes_to_the_white_span`",
            "ties the BLC contract to the rendered output.",
            "",
            "## Assumptions and limitations",
            "",
            "- **Simulated data.** This is a simulated ISP validation, not a",
            "  certified measurement of a physical camera. The sensor preset is a",
            "  representative engineering design point; its QE/CFA/IR-cut curves",
            "  are synthetic estimates bundled in `camsim.data`.",
            "- **In-sample CCM.** The 3x3 matrix is fitted and evaluated on the",
            "  same 24 patches. It measures calibration-scene representability,",
            "  not held-out generalisation or colour constancy under a different",
            "  illuminant.",
            "- **CFA colour limitation.** The shipped CFA cannot meet the 2.0/4.0",
            "  ΔE2000 gate; the measured values above are recorded as FAIL. A",
            "  controller ruling is required to revise the CFA curves or the",
            "  threshold.",
            "- **Quick mode is not gate evidence.** `--quick` uses a smaller chart",
            "  and star and writes `g3_isp_results_quick.json`; it never writes or",
            "  overwrites the committed `g3_isp_results.json` or `g3_isp.md`.",
            "- **Formal MTF is deferred to G4.** ISO 12233-style e-SFR, MTF50, and",
            "  MTF20 require the independent metrics harness; G3 reports",
            "  edge-contrast preservation and overshoot instead.",
            "- **Camera auto-exposure is deferred to G5.** The `Camera`",
            "  orchestration and `auto_expose()` convergence check are Phase 5",
            "  consolidation work; G3 calibrates exposure explicitly from a probe.",
            "- **Determinism.** Seeds are explicit and outputs are reproducible",
            "  within one software environment; PNG bytes can differ across",
            "  Matplotlib/font builds, so cross-platform comparisons use numbers",
            "  and tolerances.",
            "",
            "## Figures",
            "",
            "| Figure | Path |",
            "|---|---|",
            "| ISP stage contact sheet | `results/figures/02_isp_stage_contact_sheet.png` |",
            "| Stage zoom crops | `results/figures/02_isp_stage_zoom.png` |",
            "| Demosaic comparison | `results/figures/02_isp_demosaic_comparison.png` |",
            "| ColorChecker ΔE2000 | `results/figures/g3_delta_e.png` |",
            "| AWB gains | `results/figures/g3_awb_gains.png` |",
            "| Check summary | `results/figures/g3_check_summary.png` |",
            "| RCCC baseline | `results/figures/02_isp_rccc_baseline.png` |",
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run python examples/02_isp_stage_gallery.py",
            "uv run pytest tests/test_example_02.py -v",
            "uv run pytest -q",
            "uv run ruff check . && uv run ruff format --check . && uv run mypy src",
            "```",
            "",
            "The full example writes `results/reports/g3_isp_results.json` and",
            "`results/reports/g3_isp.md`; the gate tests parse the committed",
            "report tables back and compare every numeric cell with the JSON, so",
            "a stale report fails instead of being accepted. The `--quick` flag",
            "is a smoke run and is not gate evidence.",
            "",
        ]
    )
    return "\n".join(lines)


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
        markdown_path = reports / REPORT_MARKDOWN_NAME
        markdown_path.write_text(_render_report(payload), encoding="utf-8")
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
    if payload["status"] != "PASS":
        failures = sum(1 for check in payload["checks"] if not bool(check["pass"]))
        print(f"\nG3 gate: FAIL ({failures} check(s) outside tolerance)")
        return 1
    print("\nG3 gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
