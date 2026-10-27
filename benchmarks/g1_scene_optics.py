"""Generate the G1 scene/optics validation report and its figures.

The G1 gate asks whether a simulated scene can be trusted before any sensor
model consumes it. This script evaluates the gate's analytic invariants with
the same public functions the test suite uses, writes four figures to
``results/figures/g1_*.png``, and writes a Markdown report to
``results/reports/g1_scene_optics.md`` that states every measured value, its
tolerance, and the assumptions behind it.

The physics follows four lines of evidence:

* **Photometry.** With a per-nanometre spectral irradiance density
  ``E(lambda)`` in ``W/m^2/nm`` and the wavelength axis in nanometres,
  ``683 * integral(E(lambda) V(lambda) dlambda)`` equals the requested
  illuminance in lux. The axis and the density share the same nanometre unit;
  scaling only one of them is the classic 1e9 photometric error.
* **Optics.** A sampled Gaussian of ``sigma = 1`` pixel has MTF50
  ``sqrt(ln 2 / (2 pi**2)) = 0.18739`` cycles/pixel. An ideal circular pupil
  gives an Airy pattern ``(2 J1(x) / x)**2`` with
  ``x = pi r / (lambda N / pitch)``; the reference f/1.8 lens at 550 nm and a
  1.12 um pitch has its first dark ring near 1.078 pixels, which is below the
  pixel sampling interval and therefore checked against the closed-form
  profile rather than a resolved sampled minimum.
* **Propagation.** A lens redistributes but does not create light. A symmetric
  unit-sum PSF with half-sample reflection, followed by a block mean that
  models pixel-aperture integration, must preserve total energy within 1 %.
* **Geometry and interfaces.** The slanted-edge target and its gradient
  centroid estimator must recover tilts within 0.5 degrees, the external
  spectral cube must reject shape/unit/value mismatches, and the reference
  lens must reproduce its 84 degree field of view and 99.73 pixels/degree.

Real-world ranges represented here: illuminance 1-100,000 lux, Gaussian blur
sigma 0.4-3 pixels, f-numbers f/1.4-f/16, pixel pitches 0.6-10 um, and
slanted-edge tilts of a few degrees. The gate tolerances are 0.1 % for
photometric normalization, 0.01 cycles/pixel for MTF50, 1 % for energy,
0.5 degrees for edge angle, and 0.1 % for pixels per degree.

Failure artefacts guarded here: a metre-scaled wavelength integral; an
unnormalized kernel or a wrong Airy scale; zero-padded convolution or a
summing block reducer; a transposed edge convention; a cube whose grid length
disagrees with its data; a resampler that extrapolates outside the measured
band; and a report or figure that is not regenerated when the code changes.
The output contains no timestamps or commit hashes, so repeated runs in one
environment produce identical Markdown and byte-identical PNGs.
"""

from __future__ import annotations

import math
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import colour
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.special import j1

import camsim
from camsim.config.schema import LensConfig
from camsim.core.spectral import SpectralIrradiance
from camsim.optics.propagation import apply_psf, box_downsample
from camsim.optics.psf import airy_psf, gaussian_psf, psf_mtf, psf_mtf50
from camsim.scene.illuminants import (
    K_M,
    illuminant,
    lux_to_spectral_irradiance,
    photopic_luminous_efficiency,
)
from camsim.scene.targets import estimate_edge_angle, slanted_edge_target
from camsim.viz import PALETTE, save_figure, set_style

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPORT_PATH: Final[Path] = REPO_ROOT / "results" / "reports" / "g1_scene_optics.md"
FIGURE_DIR: Final[Path] = REPO_ROOT / "results" / "figures"

PIXEL_PITCH_UM: Final[float] = 1.12
WAVELENGTH_NM: Final[float] = 550.0
F_NUMBER: Final[float] = 1.8
FOCAL_LENGTH_MM: Final[float] = 6.4
SENSOR_DIAGONAL_MM: Final[float] = 11.46
TRANSMITTANCE: Final[float] = 0.9

ILLUMINANT_NAMES: Final[tuple[str, ...]] = ("D65", "3000K", "4000K")
LUX_LEVELS: Final[tuple[float, ...]] = (0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
EDGE_ANGLES_DEG: Final[tuple[float, ...]] = (
    -20.0,
    -15.0,
    -10.0,
    -7.0,
    -5.0,
    -3.0,
    -2.0,
    -1.0,
    -0.5,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
)
EDGE_SHAPE: Final[tuple[int, int]] = (256, 256)
ENERGY_FACTOR: Final[int] = 4

LUX_TOLERANCE: Final[float] = 1e-3
MTF50_TOLERANCE: Final[float] = 1e-2
ENERGY_TOLERANCE: Final[float] = 1e-2
EDGE_TOLERANCE_DEG: Final[float] = 0.5
FOV_TOLERANCE_DEG: Final[float] = 1.0
PPD_TOLERANCE_REL: Final[float] = 1e-3

# Exact first zero of J1; the conventional 1.22 approximation differs by 0.03 %.
J1_FIRST_ZERO: Final[float] = 3.8317059702075125
GAUSSIAN_MTF50_ANALYTIC: Final[float] = math.sqrt(math.log(2.0) / (2.0 * math.pi**2))

FIGURE_NAMES: Final[tuple[str, ...]] = (
    "g1_lux_normalization.png",
    "g1_psf_mtf.png",
    "g1_energy_conservation.png",
    "g1_edge_angle.png",
)


@dataclass(frozen=True)
class LuxRecord:
    """One requested/measured illuminance pair and its relative error."""

    illuminant_name: str
    requested_lux: float
    measured_lux: float
    relative_error: float


@dataclass(frozen=True)
class GaussianSummary:
    """Measured and analytic properties of the unit-sigma Gaussian PSF."""

    kernel_sum: float
    mtf50_measured: float
    mtf50_analytic: float
    mtf50_absolute_error: float


@dataclass(frozen=True)
class AirySummary:
    """Reference-lens Airy scale and the sampled kernel's closed-form error."""

    airy_scale_px: float
    first_zero_exact_px: float
    first_zero_conventional_px: float
    kernel_sum: float
    max_closed_form_deviation: float


@dataclass(frozen=True)
class EnergyRecord:
    """One propagation stage's compensated energy and its relative error."""

    stage: str
    compensated_sum: float
    reference_sum: float
    relative_error: float


@dataclass(frozen=True)
class EnergySummary:
    """Energy checks through PSF convolution and block averaging."""

    records: tuple[EnergyRecord, ...]
    constant_field_mean: float
    constant_field_error: float


@dataclass(frozen=True)
class EdgeRecord:
    """One slanted-edge tilt and its recovered angle."""

    requested_deg: float
    recovered_deg: float
    error_deg: float


@dataclass(frozen=True)
class SpectralRecord:
    """One spectral-contract check and its observed outcome."""

    check: str
    expected: str
    observed: str
    passed: bool


@dataclass(frozen=True)
class SpectralSummary:
    """Interface checks plus resampling energy errors."""

    records: tuple[SpectralRecord, ...]
    spatial_energy_error: float
    spatial_pitch_um: float
    wavelength_integral_error: float


@dataclass(frozen=True)
class LensSummary:
    """Reference-lens geometry and the resulting cycles/degree scale."""

    field_of_view_deg: float
    pixels_per_degree: float
    mtf50_cycles_per_degree: float


@dataclass(frozen=True)
class GateResult:
    """All G1 measurements in one object consumed by figures and the report."""

    lux_records: tuple[LuxRecord, ...]
    gaussian: GaussianSummary
    airy: AirySummary
    energy: EnergySummary
    edges: tuple[EdgeRecord, ...]
    spectral: SpectralSummary
    lens: LensSummary
    max_lux_error: float
    max_energy_error: float
    max_edge_error_deg: float
    check_status: dict[str, bool]
    gate_passed: bool


def _relative_error(measured: float, reference: float) -> float:
    """Return the absolute relative error, treating an exact zero as zero error."""
    if reference == 0.0:
        return 0.0 if measured == 0.0 else math.inf
    return abs(measured - reference) / abs(reference)


def _reference_lens() -> LensConfig:
    """Return the reference phone lens used by the scene renderer."""
    return LensConfig(
        f_number=F_NUMBER,
        focal_length_mm=FOCAL_LENGTH_MM,
        sensor_diagonal_mm=SENSOR_DIAGONAL_MM,
        transmittance=TRANSMITTANCE,
    )


def _photometric_records() -> tuple[LuxRecord, ...]:
    """Measure the luminous integral for every illuminant and lux level."""
    records: list[LuxRecord] = []
    for illuminant_name in ILLUMINANT_NAMES:
        spd = illuminant(illuminant_name)
        efficiency = photopic_luminous_efficiency(spd.wavelengths_nm.nm)
        for lux in LUX_LEVELS:
            irradiance = lux_to_spectral_irradiance(lux, spd)
            measured = K_M * float(
                np.trapezoid(irradiance.data[0, 0] * efficiency, spd.wavelengths_nm.nm)
            )
            records.append(
                LuxRecord(
                    illuminant_name=illuminant_name,
                    requested_lux=lux,
                    measured_lux=measured,
                    relative_error=_relative_error(measured, lux),
                )
            )
    return tuple(records)


def _gaussian_summary() -> GaussianSummary:
    """Measure the Gaussian kernel sum and MTF50 against the closed form."""
    psf = gaussian_psf(1.0, size=31)
    measured = psf_mtf50(psf)
    return GaussianSummary(
        kernel_sum=float(psf.sum()),
        mtf50_measured=measured,
        mtf50_analytic=GAUSSIAN_MTF50_ANALYTIC,
        mtf50_absolute_error=abs(measured - GAUSSIAN_MTF50_ANALYTIC),
    )


def _airy_summary() -> AirySummary:
    """Compare the reference-lens Airy kernel with the corrected closed form."""
    airy_scale_px = WAVELENGTH_NM * 1e-3 * F_NUMBER / PIXEL_PITCH_UM
    size = 31
    psf = airy_psf(F_NUMBER, WAVELENGTH_NM, PIXEL_PITCH_UM, size=size)
    coordinates = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    x = math.pi * np.hypot(xx, yy) / airy_scale_px
    reference = np.ones_like(x)
    nonzero = x != 0.0
    reference[nonzero] = (2.0 * j1(x[nonzero]) / x[nonzero]) ** 2
    reference /= reference.sum()
    return AirySummary(
        airy_scale_px=airy_scale_px,
        first_zero_exact_px=J1_FIRST_ZERO / math.pi * airy_scale_px,
        first_zero_conventional_px=1.22 * airy_scale_px,
        kernel_sum=float(psf.sum()),
        max_closed_form_deviation=float(np.max(np.abs(psf - reference))),
    )


def _energy_summary() -> EnergySummary:
    """Measure energy conservation through PSF convolution and block means."""
    source = np.zeros((64, 64, 3))
    source[30:34, 30:34, :] = 1.0
    kernel = gaussian_psf(1.0, size=11)
    reference_sum = float(source.sum())

    blurred = apply_psf(source, kernel)
    downsampled_only = box_downsample(source, ENERGY_FACTOR)
    through_psf_and_box = box_downsample(blurred, ENERGY_FACTOR)
    through_psf_and_eight = box_downsample(blurred, 8)

    stages = (
        ("Gaussian PSF only", float(blurred.sum())),
        (
            f"{ENERGY_FACTOR}x box mean only",
            float(downsampled_only.sum()) * ENERGY_FACTOR**2,
        ),
        (
            f"PSF + {ENERGY_FACTOR}x box mean",
            float(through_psf_and_box.sum()) * ENERGY_FACTOR**2,
        ),
        ("PSF + 8x box mean", float(through_psf_and_eight.sum()) * 8**2),
    )
    records = tuple(
        EnergyRecord(
            stage=stage,
            compensated_sum=compensated,
            reference_sum=reference_sum,
            relative_error=_relative_error(compensated, reference_sum),
        )
        for stage, compensated in stages
    )
    constant_mean = float(box_downsample(np.ones((8, 8, 3)), 2).mean())
    return EnergySummary(
        records=records,
        constant_field_mean=constant_mean,
        constant_field_error=_relative_error(constant_mean, 1.0),
    )


def _edge_records() -> tuple[EdgeRecord, ...]:
    """Recover a sweep of slanted-edge tilts with the production estimator."""
    records: list[EdgeRecord] = []
    for angle_deg in EDGE_ANGLES_DEG:
        target = slanted_edge_target(EDGE_SHAPE, angle_deg=angle_deg)
        recovered = estimate_edge_angle(target)
        records.append(
            EdgeRecord(
                requested_deg=angle_deg,
                recovered_deg=recovered,
                error_deg=recovered - angle_deg,
            )
        )
    return tuple(records)


def _first_line(exc: BaseException) -> str:
    """Return the first line of an exception message, truncated for a table."""
    message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return message if len(message) <= 150 else message[:147] + "..."


def _expect_value_error(
    check: str,
    expected: str,
    action: Callable[[], object],
) -> SpectralRecord:
    """Run a rejection check, recording whether it raised ``ValueError``."""
    try:
        action()
    except ValueError as exc:
        return SpectralRecord(check, expected, f"ValueError: {_first_line(exc)}", True)
    return SpectralRecord(check, expected, "no error raised", False)


def _spectral_summary() -> SpectralSummary:
    """Exercise the cube contract and both explicit resampling helpers."""
    wavelengths_nm = np.arange(400.0, 701.0, 50.0)
    data = np.ones((8, 8, wavelengths_nm.size))
    cube = SpectralIrradiance.from_array(
        data,
        wavelengths_nm,
        "W/m^2/nm",
        pixel_pitch_um=4.48,
    )
    records: list[SpectralRecord] = [
        SpectralRecord(
            check="Matching grid accepted",
            expected="shape, unit, and pitch preserved",
            observed=f"{cube.data.shape}, {cube.unit}, {cube.pixel_pitch_um:g} um",
            passed=cube.data.shape == (8, 8, 7) and cube.unit == "W/m^2/nm",
        ),
        _expect_value_error(
            "Data/grid length mismatch",
            "ValueError",
            lambda: SpectralIrradiance.from_array(
                np.ones((8, 8, wavelengths_nm.size + 1)),
                wavelengths_nm,
                "W/m^2/nm",
            ),
        ),
        _expect_value_error(
            "Unsupported unit",
            "ValueError",
            lambda: SpectralIrradiance.from_array(data, wavelengths_nm, "W/m^2/um"),
        ),
        _expect_value_error(
            "Negative irradiance",
            "ValueError",
            lambda: SpectralIrradiance.from_array(-data, wavelengths_nm, "W/m^2/nm"),
        ),
        _expect_value_error(
            "Uncovered wavelength range",
            "ValueError",
            lambda: cube.resample_wavelengths(np.arange(350.0, 701.0, 50.0)),
        ),
    ]

    spatial = cube.resample_spatial((2, 2))
    # The 8x8 grid becomes 2x2, so each output sample averages a 4x4 block:
    # total energy is recovered by multiplying the reduced sum by 4**2, and the
    # scalar pitch grows from 4.48 um to 4.48 * 4 = 17.92 um.
    block_factor = cube.data.shape[0] // spatial.data.shape[0]
    spatial_error = _relative_error(
        float(spatial.data.sum()) * block_factor**2,
        float(cube.data.sum()),
    )
    records.append(
        SpectralRecord(
            check="Spatial box resampling",
            expected=f"energy preserved; pitch x{block_factor}",
            observed=(f"relative error {spatial_error:.2e}; pitch {spatial.pixel_pitch_um:g} um"),
            passed=(spatial_error <= 1e-12 and spatial.pixel_pitch_um == 4.48 * block_factor),
        )
    )

    target_nm = np.arange(400.0, 701.0, 25.0)
    integrated = cube.resample_wavelengths(target_nm)
    source_integral = float(np.trapezoid(cube.data[0, 0], cube.wavelengths_nm.nm))
    target_integral = float(np.trapezoid(integrated.data[0, 0], integrated.wavelengths_nm.nm))
    integral_error = _relative_error(target_integral, source_integral)
    records.append(
        SpectralRecord(
            check="Wavelength integration",
            expected="same-domain integral preserved",
            observed=f"relative error {integral_error:.2e}",
            passed=integral_error <= 1e-12,
        )
    )

    return SpectralSummary(
        records=tuple(records),
        spatial_energy_error=spatial_error,
        spatial_pitch_um=float(spatial.pixel_pitch_um or math.nan),
        wavelength_integral_error=integral_error,
    )


def _lens_summary(gaussian: GaussianSummary) -> LensSummary:
    """Measure the reference lens's geometry and cycles/degree scale."""
    lens = _reference_lens()
    ppd = lens.pixels_per_degree(PIXEL_PITCH_UM)
    return LensSummary(
        field_of_view_deg=lens.field_of_view_deg,
        pixels_per_degree=ppd,
        mtf50_cycles_per_degree=gaussian.mtf50_measured * ppd,
    )


def measure_gate() -> GateResult:
    """Evaluate every G1 check and decide the gate verdict."""
    lux_records = _photometric_records()
    gaussian = _gaussian_summary()
    airy = _airy_summary()
    energy = _energy_summary()
    edges = _edge_records()
    spectral = _spectral_summary()
    lens = _lens_summary(gaussian)

    max_lux_error = max(record.relative_error for record in lux_records)
    max_energy_error = max(record.relative_error for record in energy.records)
    max_edge_error = max(abs(record.error_deg) for record in edges)

    lux_ok = max_lux_error <= LUX_TOLERANCE
    mtf_ok = abs(gaussian.mtf50_measured - 0.1874) <= MTF50_TOLERANCE
    airy_ok = (
        abs(airy.first_zero_exact_px - 1.078) <= 1e-3 and airy.max_closed_form_deviation <= 1e-12
    )
    energy_ok = max_energy_error <= ENERGY_TOLERANCE
    edge_ok = max_edge_error <= EDGE_TOLERANCE_DEG
    spectral_ok = all(record.passed for record in spectral.records)
    lens_ok = (
        abs(lens.field_of_view_deg - 84.0) <= FOV_TOLERANCE_DEG
        and _relative_error(lens.pixels_per_degree, 99.73) <= PPD_TOLERANCE_REL
    )
    check_status = {
        "photometric": lux_ok,
        "mtf": mtf_ok,
        "airy": airy_ok,
        "energy": energy_ok,
        "edge": edge_ok,
        "spectral": spectral_ok,
        "lens": lens_ok,
    }
    return GateResult(
        lux_records=lux_records,
        gaussian=gaussian,
        airy=airy,
        energy=energy,
        edges=edges,
        spectral=spectral,
        lens=lens,
        max_lux_error=max_lux_error,
        max_energy_error=max_energy_error,
        max_edge_error_deg=max_edge_error,
        check_status=check_status,
        gate_passed=all(check_status.values()),
    )


def _style_axes(axes: Axes) -> None:
    """Apply the house grid and tick styling to a single panel."""
    axes.grid(True, which="major")
    axes.tick_params(labelsize=8.5)


def _figure_lux(gate: GateResult) -> Path:
    """Plot the D65 SPD, the photopic weighting, and the lux error sweep."""
    spd = illuminant("D65")
    wavelengths_nm = spd.wavelengths_nm.nm
    efficiency = photopic_luminous_efficiency(wavelengths_nm)
    scene = lux_to_spectral_irradiance(100.0, spd)
    weighted = scene.data[0, 0] * efficiency
    weighted /= float(weighted.max())

    figure, (ax_spd, ax_v, ax_error) = plt.subplots(
        1, 3, figsize=(13.2, 4.0), constrained_layout=True
    )
    ax_spd.plot(wavelengths_nm, spd.values, color=PALETTE[0], lw=1.7)
    ax_spd.set_title("D65 relative SPD")
    ax_spd.set_xlabel("wavelength (nm)")
    ax_spd.set_ylabel("relative spectral power")
    ax_spd.set_xlim(float(wavelengths_nm[0]), float(wavelengths_nm[-1]))
    _style_axes(ax_spd)

    ax_v.plot(wavelengths_nm, efficiency, color=PALETTE[2], lw=1.7, label="V(lambda)")
    ax_v.plot(
        wavelengths_nm,
        weighted,
        color=PALETTE[1],
        lw=1.4,
        label="E(lambda) V(lambda), peak-normalized",
    )
    ax_v.set_title("Photopic weighting (100 lux)")
    ax_v.set_xlabel("wavelength (nm)")
    ax_v.set_ylabel("normalized response")
    ax_v.set_xlim(float(wavelengths_nm[0]), float(wavelengths_nm[-1]))
    ax_v.legend(loc="upper right", fontsize=7.5)
    _style_axes(ax_v)

    # A log error axis would send exact-zero measurements to negative infinity.
    # The panel's job is tolerance compliance, so use a linear axis focused on
    # the 1e-3 budget: all measured errors are at or within round-off of zero on
    # this scale, and the report table carries the exact values.
    ax_error.axhspan(0.0, LUX_TOLERANCE, color="#b0b0b0", alpha=0.22, lw=0.0)
    for index, illuminant_name in enumerate(ILLUMINANT_NAMES):
        levels = [
            record
            for record in gate.lux_records
            if record.illuminant_name == illuminant_name and record.requested_lux > 0.0
        ]
        ax_error.plot(
            [record.requested_lux for record in levels],
            [record.relative_error for record in levels],
            marker="o",
            ms=4,
            lw=1.2,
            color=PALETTE[index],
            label=illuminant_name,
        )
    ax_error.axhline(
        LUX_TOLERANCE,
        color=PALETTE[1],
        lw=1.0,
        ls="--",
        label=f"{LUX_TOLERANCE:g} tolerance",
    )
    ax_error.set_xscale("log")
    ax_error.set_xlim(0.8, 2.0e5)
    ax_error.set_ylim(0.0, 1.6 * LUX_TOLERANCE)
    ax_error.set_title("Lux normalization error")
    ax_error.set_xlabel("requested illuminance (lux)")
    ax_error.set_ylabel("|measured - requested| / requested")
    ax_error.text(
        0.03,
        0.55,
        f"max error {_format_error(gate.max_lux_error)}\n"
        "all points coincide with zero\non this tolerance scale",
        transform=ax_error.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    ax_error.legend(loc="upper right", fontsize=7.5)
    _style_axes(ax_error)

    path = save_figure(figure, FIGURE_DIR / FIGURE_NAMES[0])
    plt.close(figure)
    return path


def _figure_psf(gate: GateResult) -> Path:
    """Compare the Gaussian MTF with its analytic curve and show Airy rings."""
    figure, (ax_mtf, ax_airy) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    frequencies = np.linspace(0.0, 0.5, 501)
    measured = psf_mtf(gaussian_psf(1.0, size=31), frequencies)
    analytic = np.exp(-2.0 * math.pi**2 * frequencies**2)
    ax_mtf.plot(frequencies, measured, color=PALETTE[0], lw=1.7, label="Sampled Gaussian PSF")
    ax_mtf.plot(
        frequencies,
        analytic,
        color=PALETTE[1],
        lw=1.2,
        ls="--",
        label="Analytic exp(-2 pi^2 sigma^2 f^2)",
    )
    ax_mtf.axhline(0.5, color="#666666", lw=0.8, ls=":")
    ax_mtf.axvline(
        gate.gaussian.mtf50_measured,
        color=PALETTE[0],
        lw=0.8,
        ls=":",
        label=f"MTF50 = {gate.gaussian.mtf50_measured:.6f} cycles/pixel",
    )
    ax_mtf.set_xlim(0.0, 0.5)
    ax_mtf.set_ylim(0.0, 1.02)
    ax_mtf.set_title("Gaussian MTF, sigma = 1 pixel")
    ax_mtf.set_xlabel("frequency (cycles/pixel)")
    ax_mtf.set_ylabel("MTF")
    ax_mtf.legend(loc="lower left", fontsize=7.5)
    _style_axes(ax_mtf)

    radii = np.linspace(1e-6, 6.0, 600)
    x = math.pi * radii / gate.airy.airy_scale_px
    profile = (2.0 * j1(x) / x) ** 2
    sampled_radii = np.arange(0.0, 7.0)
    sampled_x = math.pi * sampled_radii / gate.airy.airy_scale_px
    sampled = np.ones_like(sampled_x)
    nonzero = sampled_x != 0.0
    sampled[nonzero] = (2.0 * j1(sampled_x[nonzero]) / sampled_x[nonzero]) ** 2
    ax_airy.plot(radii, profile, color=PALETTE[0], lw=1.7, label="Closed form (2 J1(x)/x)^2")
    ax_airy.plot(
        sampled_radii,
        sampled,
        ls="none",
        marker="o",
        ms=4,
        color=PALETTE[1],
        label="Sampled at pixel centres",
    )
    ax_airy.axvline(
        gate.airy.first_zero_exact_px,
        color="#555555",
        lw=0.9,
        ls=":",
        label=f"first zero {gate.airy.first_zero_exact_px:.4f} px",
    )
    ax_airy.set_xlim(0.0, 6.0)
    ax_airy.set_ylim(0.0, 1.05)
    ax_airy.set_title("Airy pattern, f/1.8, 550 nm, 1.12 um pitch")
    ax_airy.set_xlabel("radius (pixels)")
    ax_airy.set_ylabel("normalized intensity")
    ax_airy.legend(loc="upper right", fontsize=7.5)
    _style_axes(ax_airy)

    path = save_figure(figure, FIGURE_DIR / FIGURE_NAMES[1])
    plt.close(figure)
    return path


def _figure_energy(gate: GateResult) -> Path:
    """Show the delta target through the PSF chain and the energy error budget."""
    source = np.zeros((64, 64, 3))
    source[30:34, 30:34, :] = 1.0
    blurred = apply_psf(source, gaussian_psf(1.0, size=11))
    downsampled = box_downsample(blurred, ENERGY_FACTOR)

    figure = plt.figure(figsize=(12.0, 6.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.0), hspace=0.22)
    images: tuple[tuple[NDArray[np.float64], str], ...] = (
        (source[:, :, 0], f"Source 4x4 delta x 3 planes\nsum = {source.sum():.3f}"),
        (blurred[:, :, 0], f"After Gaussian PSF\nsum = {blurred.sum():.3f}"),
        (
            downsampled[:, :, 0],
            (
                f"After {ENERGY_FACTOR}x box mean ({downsampled.shape[0]}x"
                f"{downsampled.shape[1]})\n{ENERGY_FACTOR**2} x sum = "
                f"{downsampled.sum() * ENERGY_FACTOR**2:.3f}"
            ),
        ),
    )
    for index, (image, caption) in enumerate(images):
        axes = figure.add_subplot(grid[0, index])
        axes.imshow(image, cmap="viridis", interpolation="nearest")
        axes.set_title(caption, fontsize=9.5)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.grid(False)
        axes.set_frame_on(True)

    ax = figure.add_subplot(grid[1, :])
    labels = [record.stage for record in gate.energy.records]
    values = [max(record.relative_error, 1e-18) for record in gate.energy.records]
    bars = ax.bar(labels, values, color=PALETTE[0], width=0.55)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-1)
    ax.axhline(
        ENERGY_TOLERANCE,
        color=PALETTE[1],
        lw=1.1,
        ls="--",
        label=f"{ENERGY_TOLERANCE:g} tolerance",
    )
    for bar, record in zip(bars, gate.energy.records, strict=True):
        label = "0 (exact)" if record.relative_error == 0.0 else f"{record.relative_error:.1e}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(record.relative_error, 1e-18) * 4.0,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )
    ax.set_ylabel("relative energy error")
    ax.legend(loc="upper right", fontsize=8)
    _style_axes(ax)

    path = save_figure(figure, FIGURE_DIR / FIGURE_NAMES[2])
    plt.close(figure)
    return path


def _figure_edge(gate: GateResult) -> Path:
    """Show a recovered slanted edge and the angle sweep against tolerance."""
    requested = 5.0
    target = slanted_edge_target(EDGE_SHAPE, angle_deg=requested)
    recovered = estimate_edge_angle(target)
    rows = np.array([0.0, float(EDGE_SHAPE[0] - 1)])
    center_x = (EDGE_SHAPE[1] - 1) / 2.0
    center_y = (EDGE_SHAPE[0] - 1) / 2.0
    true_columns = center_x + (rows - center_y) * math.tan(math.radians(requested))
    recovered_columns = center_x + (rows - center_y) * math.tan(math.radians(recovered))

    figure, (ax_image, ax_sweep) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    ax_image.imshow(
        target,
        cmap="gray",
        origin="upper",
        extent=(0.0, float(EDGE_SHAPE[1] - 1), float(EDGE_SHAPE[0] - 1), 0.0),
    )
    ax_image.plot(
        true_columns,
        rows,
        color=PALETTE[1],
        lw=1.4,
        ls="--",
        label=f"requested {requested:.2f} deg",
    )
    ax_image.plot(
        recovered_columns,
        rows,
        color=PALETTE[0],
        lw=1.1,
        label=f"estimated {recovered:.3f} deg",
    )
    ax_image.set_title("Slanted-edge target and recovered line")
    ax_image.set_xlabel("column (pixel)")
    ax_image.set_ylabel("row (pixel)")
    ax_image.legend(loc="lower right", fontsize=8)
    ax_image.grid(False)
    ax_image.set_frame_on(True)

    requested_values = np.asarray([record.requested_deg for record in gate.edges])
    recovered_values = np.asarray([record.recovered_deg for record in gate.edges])
    ax_sweep.fill_between(
        requested_values,
        requested_values - EDGE_TOLERANCE_DEG,
        requested_values + EDGE_TOLERANCE_DEG,
        color="#b0b0b0",
        alpha=0.22,
        lw=0.0,
        label=f"+/-{EDGE_TOLERANCE_DEG:g} deg tolerance",
    )
    ax_sweep.plot(
        requested_values,
        requested_values,
        color="#444444",
        lw=1.0,
        ls="--",
        label="1:1",
    )
    ax_sweep.plot(
        requested_values,
        recovered_values,
        color=PALETTE[0],
        lw=1.2,
        marker="o",
        ms=4,
        label="estimated",
    )
    ax_sweep.set_xlim(-21.0, 21.0)
    ax_sweep.set_ylim(-21.0, 21.0)
    ax_sweep.set_aspect("equal")
    ax_sweep.set_title(f"Angle recovery, max |error| = {gate.max_edge_error_deg:.3f} deg")
    ax_sweep.set_xlabel("requested angle (deg)")
    ax_sweep.set_ylabel("recovered angle (deg)")
    ax_sweep.legend(loc="upper left", fontsize=8)
    _style_axes(ax_sweep)

    path = save_figure(figure, FIGURE_DIR / FIGURE_NAMES[3])
    plt.close(figure)
    return path


def write_figures(gate: GateResult) -> tuple[Path, ...]:
    """Write all four G1 figures and return their resolved paths."""
    return (
        _figure_lux(gate),
        _figure_psf(gate),
        _figure_energy(gate),
        _figure_edge(gate),
    )


def _format_error(value: float) -> str:
    """Format a relative error compactly, keeping exact zero readable."""
    return "0" if value == 0.0 else f"{value:.2e}"


def _lux_table(gate: GateResult) -> str:
    """Render the requested/measured lux table."""
    lines = [
        "| Requested lux | D65 measured | 3000 K measured | 4000 K measured | Max rel. error |",
        "|---:|---:|---:|---:|---:|",
    ]
    by_key = {(record.illuminant_name, record.requested_lux): record for record in gate.lux_records}
    for lux in LUX_LEVELS:
        row = [by_key[(name, lux)] for name in ILLUMINANT_NAMES]
        measured = " | ".join(f"{record.measured_lux:.9g}" for record in row)
        error = max(record.relative_error for record in row)
        lines.append(f"| {lux:g} | {measured} | {_format_error(error)} |")
    return "\n".join(lines)


def _report_environment() -> str:
    """Return the environment table used by the reproducibility section."""
    return "\n".join(
        (
            "| Item | Value |",
            "|---|---|",
            f"| Python | `{platform.python_version()}` |",
            f"| camsim | `{camsim.__version__}` |",
            f"| NumPy | `{np.__version__}` |",
            f"| SciPy | `{scipy.__version__}` |",
            f"| colour-science | `{colour.__version__}` |",
            f"| Platform | `{platform.system()}` |",
            "| Command | `uv run python benchmarks/g1_scene_optics.py` |",
        )
    )


def write_report(gate: GateResult) -> Path:
    """Write the deterministic Markdown G1 report and return its path."""
    status = "PASS" if gate.gate_passed else "FAIL"
    gaussian = gate.gaussian
    airy = gate.airy
    lens = gate.lens
    fov_ok = abs(lens.field_of_view_deg - 84.0) <= FOV_TOLERANCE_DEG
    ppd_ok = _relative_error(lens.pixels_per_degree, 99.73) <= PPD_TOLERANCE_REL

    summary_rows = (
        "| Check | Measured | Tolerance | Result |",
        "|---|---|---|---|",
        (
            "| Photometric normalization | max relative error "
            f"{_format_error(gate.max_lux_error)} | < {LUX_TOLERANCE:g} | "
            f"{'PASS' if gate.check_status['photometric'] else 'FAIL'} |"
        ),
        (
            f"| Gaussian PSF MTF50 | {gaussian.mtf50_measured:.6f} cycles/pixel | "
            f"within {MTF50_TOLERANCE:g} of 0.1874 | "
            f"{'PASS' if gate.check_status['mtf'] else 'FAIL'} |"
        ),
        (
            f"| Airy first zero, reference lens | {airy.first_zero_exact_px:.5f} px | "
            "within 1e-3 of 1.078 px; closed form within 1e-12 | "
            f"{'PASS' if gate.check_status['airy'] else 'FAIL'} |"
        ),
        (
            f"| PSF + {ENERGY_FACTOR}x box energy | max relative error "
            f"{_format_error(gate.max_energy_error)} | < {ENERGY_TOLERANCE:g} | "
            f"{'PASS' if gate.check_status['energy'] else 'FAIL'} |"
        ),
        (
            f"| Slanted-edge angle | max |error| {gate.max_edge_error_deg:.4f} deg | "
            f"< {EDGE_TOLERANCE_DEG:g} deg | "
            f"{'PASS' if gate.check_status['edge'] else 'FAIL'} |"
        ),
        (
            f"| Spectral cube contract | {sum(record.passed for record in gate.spectral.records)}"
            f"/{len(gate.spectral.records)} checks behaved as required | all | "
            f"{'PASS' if gate.check_status['spectral'] else 'FAIL'} |"
        ),
        (
            f"| Reference lens | {lens.field_of_view_deg:.3f} deg, "
            f"{lens.pixels_per_degree:.3f} px/deg | 84 +/- 1 deg; 99.73 +/- 0.1 % | "
            f"{'PASS' if gate.check_status['lens'] else 'FAIL'} |"
        ),
    )

    energy_rows = [
        "| Stage | Compensated sum | Reference sum | Relative error |",
        "|---|---:|---:|---:|",
    ]
    for record in gate.energy.records:
        energy_rows.append(
            f"| {record.stage} | {record.compensated_sum:.6f} | "
            f"{record.reference_sum:.6f} | {_format_error(record.relative_error)} |"
        )

    edge_rows = [
        "| Requested angle (deg) | Recovered angle (deg) | Error (deg) |",
        "|---:|---:|---:|",
    ]
    for record in gate.edges:
        edge_rows.append(
            f"| {record.requested_deg:+.2f} | {record.recovered_deg:+.4f} | "
            f"{record.error_deg:+.4f} |"
        )

    spectral_rows = ["| Check | Expected | Observed | Result |", "|---|---|---|---|"]
    for record in gate.spectral.records:
        spectral_rows.append(
            f"| {record.check} | {record.expected} | {record.observed} | "
            f"{'PASS' if record.passed else 'FAIL'} |"
        )

    lines = [
        "# G1 - Scene and optics validation",
        "",
        f"**Status: {status}**",
        "",
        "G1 closes Phase 1 by checking that the scene, illuminant, and optics",
        "layers are physically consistent before any sensor model consumes them.",
        "It covers the photometric scale of a synthetic scene, the analytic",
        "behaviour of the PSF models, energy conservation through propagation",
        "and 4x pixel integration, the slanted-edge geometric convention, the",
        "external spectral-cube contract, and the reference-lens geometry that",
        "converts cycles/pixel into cycles/degree. No sensor QE/CFA, noise, ADC,",
        "or RAW behaviour is exercised here; those belong to G2.",
        "",
        "The numbers apply only to the source tree that generated them. Re-run",
        "`benchmarks/g1_scene_optics.py` after any change to the scene or optics",
        "layers; a stale report is not gate evidence.",
        "",
        "## Environment and reproduction",
        "",
        _report_environment(),
        "",
        "## Verdict summary",
        "",
        "\n".join(summary_rows),
        "",
        "## 1. Photometric normalization",
        "",
        "The scene's absolute scale is a target illuminance. With wavelengths in",
        "nanometres and a per-nanometre spectral irradiance density in W/m^2/nm,",
        "",
        "```text",
        "E_scene(lambda) = lux * S(lambda) / (K_M * integral S(lambda) V(lambda) dlambda)",
        "K_M = 683 lm/W",
        "683 * integral E_scene(lambda) V(lambda) dlambda = lux",
        "```",
        "",
        "The wavelength axis and the density share the nanometre unit. Converting",
        "the axis to metres without also converting the density from per-nm to",
        "per-metre would make the measured illuminance 1e9 times too small; the",
        "implementation and the tests integrate over `e.wavelengths_nm.nm` with no",
        "such factor. `V(lambda)` is the CIE 1924 photopic luminous efficiency",
        "function and `S(lambda)` is the relative illuminant SPD.",
        "",
        _lux_table(gate),
        "",
        f"Tolerance: {LUX_TOLERANCE:g} relative. Observed maximum: "
        f"{_format_error(gate.max_lux_error)}. Zero lux is included as a valid",
        "dark scene and returns exactly zero irradiance.",
        "",
        "![Lux normalization](../figures/g1_lux_normalization.png)",
        "",
        "## 2. Analytic PSF MTF and Airy first zero",
        "",
        "A sampled Gaussian of `sigma = 1` pixel has the continuous MTF",
        "`exp(-2 pi^2 sigma^2 f^2)`, so its MTF50 is",
        f"`sqrt(ln 2 / (2 pi^2)) = {GAUSSIAN_MTF50_ANALYTIC:.6f}` cycles/pixel.",
        "The discrete kernel is normalized to unit sum before its MTF is evaluated.",
        "",
        "| Quantity | Measured | Reference | Absolute difference |",
        "|---|---:|---:|---:|",
        f"| Gaussian kernel sum | {gaussian.kernel_sum:.12f} | 1 | "
        f"{abs(gaussian.kernel_sum - 1.0):.2e} |",
        f"| Gaussian MTF50 | {gaussian.mtf50_measured:.9f} | "
        f"{gaussian.mtf50_analytic:.9f} | {gaussian.mtf50_absolute_error:.2e} |",
        f"| Airy kernel sum | {airy.kernel_sum:.12f} | 1 | {abs(airy.kernel_sum - 1.0):.2e} |",
        f"| Airy first zero (exact J1) | {airy.first_zero_exact_px:.5f} px | "
        f"{airy.first_zero_exact_px:.5f} px | 0 |",
        f"| Airy first zero (1.22 convention) | {airy.first_zero_conventional_px:.5f} px | "
        f"{airy.first_zero_exact_px:.5f} px | "
        f"{abs(airy.first_zero_conventional_px - airy.first_zero_exact_px):.2e} px |",
        f"| Airy kernel vs closed form | {airy.max_closed_form_deviation:.2e} max abs | "
        "0 | "
        f"{airy.max_closed_form_deviation:.2e} |",
        "",
        "The Airy profile is `(2 J1(x) / x)^2` with",
        "`x = pi r / (lambda N / pitch)`. For the reference lens, the Airy scale",
        f"is `lambda N / pitch = {airy.airy_scale_px:.6f}` px and the exact J1",
        f"zero puts the first dark ring at {airy.first_zero_exact_px:.5f} px.",
        "At f/1.8 that ring lies between pixel centres (spacing 1 px), so the",
        "kernel is validated against the closed-form samples; the resolved-ring",
        "check at f/8 lives in `tests/test_psf.py`.",
        "",
        f"Tolerance: Gaussian MTF50 within {MTF50_TOLERANCE:g} cycles/pixel of",
        "0.1874; Airy first zero within 1e-3 px of 1.078 px; sampled Airy kernel",
        "within 1e-12 of the closed form.",
        "",
        "![PSF MTF and Airy first zero](../figures/g1_psf_mtf.png)",
        "",
        "## 3. Energy conservation through PSF and 4x downsampling",
        "",
        "Convolution cannot create or destroy light, and the block mean models",
        "pixel-aperture integration, so the total energy must survive both. The",
        "input is a 4x4 delta in three spectral planes (total 48.0). A block mean",
        "reduces the raw sum by `factor**2`, so the table compares each stage",
        "after multiplying by that factor; the tolerance is on the compensated",
        "total.",
        "",
        "\n".join(energy_rows),
        "",
        "The constant-field check returns a block-mean value of",
        f"{gate.energy.constant_field_mean:.12f} for a field of ones (relative",
        f"error {_format_error(gate.energy.constant_field_error)}), confirming",
        "that the reducer averages rather than sums.",
        "",
        f"Tolerance: {ENERGY_TOLERANCE:g} relative. Observed maximum: "
        f"{_format_error(gate.max_energy_error)}.",
        "",
        "![Energy conservation](../figures/g1_energy_conservation.png)",
        "",
        "## 4. Slanted-edge angle recovery",
        "",
        "The target rasterizes a binary reflectance step through the image centre",
        "at a requested tilt. `estimate_edge_angle` takes the horizontal Sobel",
        "gradient, keeps the columns near each row's gradient peak, forms an",
        "energy-weighted centroid per row, and fits a weighted line whose slope",
        "is `tan(theta)`. Positive angles move the edge rightwards as the row",
        "index increases.",
        "",
        "\n".join(edge_rows),
        "",
        f"Tolerance: {EDGE_TOLERANCE_DEG:g} deg absolute. Observed maximum: "
        f"{gate.max_edge_error_deg:.4f} deg. This check validates the geometric",
        "convention shared by the target and estimator. A full ISO 12233-style",
        "e-SFR measurement on rendered, blurred, noisy edges belongs to the G4",
        "metrics gate and is not claimed here.",
        "",
        "![Slanted-edge angle recovery](../figures/g1_edge_angle.png)",
        "",
        "## 5. Spectral cube and unit contract",
        "",
        "The external `SpectralIrradiance` boundary must accept a cube whose",
        "wavelength grid matches its data and reject every mismatch rather than",
        "silently returning a mis-coloured capture. Explicit resampling helpers",
        "must conserve the represented energy and update pixel-pitch metadata.",
        "",
        "\n".join(spectral_rows),
        "",
        "The spatial box resample reports a relative energy error of",
        f"{_format_error(gate.spectral.spatial_energy_error)} and a pitch of",
        f"{gate.spectral.spatial_pitch_um:g} um (4.48 um x 4). The wavelength",
        "integration error over the same 400-700 nm domain is",
        f"{_format_error(gate.spectral.wavelength_integral_error)}. Sensor-side",
        "pitch and shape compatibility is enforced when a sensor consumes a",
        "cube, which is part of G2.",
        "",
        "## 6. Reference lens geometry",
        "",
        "| Quantity | Measured | Tolerance | Result |",
        "|---|---:|---|---|",
        f"| Diagonal field of view | {lens.field_of_view_deg:.4f} deg | "
        f"84 +/- {FOV_TOLERANCE_DEG:g} deg | "
        f"{'PASS' if fov_ok else 'FAIL'} |",
        f"| Angular sampling | {lens.pixels_per_degree:.4f} px/deg | "
        f"99.73 +/- {PPD_TOLERANCE_REL:.1%} | "
        f"{'PASS' if ppd_ok else 'FAIL'} |",
        f"| Gaussian MTF50 at the reference scale | {lens.mtf50_cycles_per_degree:.4f} "
        "cycles/deg | informational | - |",
        "",
        "A pixel of pitch `p` subtends `atan(p / f)` radians, so",
        "`pixels_per_degree = 1 / degrees(atan(p / f))`; equivalently",
        "`pi / (180 * atan(p / f))`. At 1.12 um and 6.4 mm this is 99.7331",
        "px/deg, which turns the unit-sigma Gaussian's 0.187391 cycles/pixel into",
        "18.689 cycles/degree.",
        "",
        "## Assumptions and limitations",
        "",
        "- **Scope.** G1 covers `camsim.scene` and `camsim.optics` only. Sensor QE,",
        "  CFA, shot/read/dark noise, ADC clipping, and the RAW frame contract are",
        "  G2 evidence and are not validated here.",
        "- **Optics is a supporting model, not a calibrated lens.** The Gaussian is",
        "  an aberration/defocus approximation, the Airy kernel is an ideal",
        "  circular pupil, and the box is the pixel aperture. There is no",
        "  wavefront aberration, pupil apodization, wavelength-dependent PSF,",
        "  polychromatic diffraction, field-dependent blur, or measured lens",
        "  prescription, so these numbers demonstrate physical consistency rather",
        "  than a specific lens's performance.",
        "- **ISO 12233-style e-SFR is deferred.** The edge check validates only the",
        "  target/estimator angle convention; edge-spread-function MTF on rendered,",
        "  blurred, noisy images belongs to the G4 metrics gate.",
        "- **D65 endpoint extrapolation.** `colour-science` tabulates D65 from 300",
        "  to 780 nm. camsim interpolates it onto the 360-830 nm grid and holds the",
        "  780 nm value constant from 780 to 830 nm. V(lambda) is about 1.5e-5 at",
        "  780 nm and 4.5e-7 at 830 nm, so this tail changes the photometric",
        "  normalization by far less than the 1e-3 tolerance; the flat red tail is",
        "  visible in spectral plots.",
        "- **Photometric band.** The luminous integral covers the tabulated",
        "  360-830 nm band. `photopic_luminous_efficiency` returns zero outside the",
        "  CIE 1924 table by contract, so missing ultraviolet and infrared bands",
        "  contribute no luminous flux. Radiometric energy outside the band is",
        "  intentionally not modelled.",
        "- **Reference lens.** The configuration uses a scalar transmittance of",
        f"  {TRANSMITTANCE:g}, no vignetting, a {FOCAL_LENGTH_MM:g} mm focal length, an",
        f"  {SENSOR_DIAGONAL_MM:g} mm diagonal, and a {PIXEL_PITCH_UM:g} um pitch. The",
        f"  simulated {lens.field_of_view_deg:.2f} deg field rounds to the documented",
        "  approximately 84 deg, not an exact 84.0 deg.",
        "- **Energy budget.** Symmetric kernels with half-sample-reflection borders",
        "  conserve energy to floating-point round-off. An asymmetric kernel wraps",
        "  some mass at the frame border and is guaranteed only within the",
        "  documented 1 % energy budget.",
        "- **Spectral resampling.** Wavelength resampling is overlap-weighted",
        "  piecewise-constant integration and refuses to extrapolate outside the",
        "  measured range. Spatial resampling requires an exact block division, so",
        "  no partial block represents a fractional pixel area.",
        "- **Determinism.** The Markdown report and the numerical results are",
        "  deterministic for a fixed environment. PNG metadata is fixed, but",
        "  Matplotlib and font builds can change rendered bytes across platforms;",
        "  cross-platform comparisons should use the reported numbers and a",
        "  pixel tolerance rather than byte equality.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run python benchmarks/g1_scene_optics.py",
        "uv run pytest tests/test_validation_g1.py -v",
        "uv run pytest tests/test_validation_g1.py tests/test_units.py tests/test_spectral.py \\",
        "  tests/test_lens.py tests/test_psf.py tests/test_targets.py "
        "tests/test_propagation.py tests/test_glyphs.py -v",
        "uv run pytest -q",
        "uv run ruff check . && uv run mypy src",
        "```",
        "",
        "The four figures live at `results/figures/g1_lux_normalization.png`,",
        "`g1_psf_mtf.png`, `g1_energy_conservation.png`, and `g1_edge_angle.png`.",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def main() -> int:
    """Generate figures and report, print the verdict, and return an exit code."""
    set_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    gate = measure_gate()
    figures = write_figures(gate)
    report = write_report(gate)
    print(f"G1 status: {'PASS' if gate.gate_passed else 'FAIL'}")
    print(f"Report: {report}")
    for figure in figures:
        print(f"Figure: {figure}")
    return 0 if gate.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
