"""G1 gate: scene, photometry, optics, and spectral-contract validation.

The G1 gate closes Phase 1 by proving that the scene and optics layers are
physically consistent before any sensor model consumes them. It checks four
independent invariants plus two interface contracts:

* **Photometry.** A radiometric scene needs an absolute scale. The luminous
  efficacy of radiation links radiometry to photometry through
  ``K_m = 683 lm/W``: an illuminance in lux is
  ``683 * integral(E(lambda) V(lambda) dlambda)`` when the wavelength axis is
  in nanometres and ``E`` is a per-nanometre density in ``W/m^2/nm``. Scaling
  the grid to metres without also scaling the density produces a value 1e9
  times too small; that bug is exactly what this test would catch.
* **Optics.** A normalized Gaussian PSF of ``sigma = 1`` pixel has the
  closed-form MTF ``exp(-2 pi**2 sigma**2 f**2)``, so MTF50 must be
  ``sqrt(ln 2 / (2 pi**2)) = 0.18739`` cycles/pixel. A circular pupil produces
  the Airy pattern ``(2 J1(x) / x)**2`` with
  ``x = pi r / (lambda N / pitch)``; for the reference f/1.8 lens at 550 nm and
  1.12 um pitch the first dark ring is approximately 1.078 pixels from the
  axis, below the two-pixel sampling interval, so the kernel is checked
  against the closed-form profile rather than by locating a sampled ring.
* **Propagation.** A lens redistributes light but does not create or destroy
  it. Convolution with a symmetric, unit-sum kernel followed by a 4x pixel
  block average must preserve total energy to within 1 %, and the block mean
  preserves per-unit-area irradiance.
* **Edge geometry.** The slanted-edge target and its gradient-centroid
  estimator must share one sign convention and recover the requested tilt
  within 0.5 degrees, because every downstream e-SFR measurement depends on
  that geometry.
* **Spectral contract.** A cube on a matching grid is accepted; a length,
  unit, or non-negativity mismatch raises instead of silently producing a
  mis-coloured RAW; explicit spatial and wavelength resampling conserve the
  represented energy.
* **Reference lens.** The 6.4 mm, f/1.8, 11.46 mm-diagonal reference lens sees
  about 84 degrees diagonally and 99.73 pixels per degree at a 1.12 um pitch,
  the metadata that turns cycles/pixel into cycles/degree.

Real-world ranges behind these checks: illuminance 1-100,000 lux; Gaussian
blur sigma 0.4-3 pixels; f-numbers f/1.4-f/16; pixel pitches 0.6-10 um; and
slanted-edge tilts 2-7 degrees (ISO 12233 recommends about 5). The tolerances
match the project's G1 gate: 0.1 % for photometric normalization, 0.01
cycles/pixel for MTF50, 1 % for energy, 0.5 degrees for edge angle, and 0.1 %
for pixels per degree.

Failure artefacts guarded here: integrating a per-nanometre density over a
metre axis; using MTF50 from an unnormalized kernel; reading the Airy scale
without the ``pi r`` normalization; zero-padded convolution or a summing block
reducer that silently loses or multiplies energy; a transposed edge convention
that reports the wrong sign; accepting a cube whose wavelength axis and data
length disagree; and allowing a spectral resampler to extrapolate outside its
measured range. The final two tests also ensure the visible gate deliverables
exist: the Markdown report records the measured values and limitations, and
every ``g1_*.png`` figure is a non-empty PNG.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import j1

from camsim.config.schema import LensConfig
from camsim.core.spectral import SpectralIrradiance
from camsim.optics.propagation import apply_psf, box_downsample
from camsim.optics.psf import airy_psf, box_psf, gaussian_psf, psf_mtf50
from camsim.scene.illuminants import (
    K_M,
    illuminant,
    lux_to_spectral_irradiance,
    photopic_luminous_efficiency,
)
from camsim.scene.targets import estimate_edge_angle, slanted_edge_target

REPO_ROOT = Path(__file__).resolve().parents[1]
G1_REPORT_PATH = REPO_ROOT / "results" / "reports" / "g1_scene_optics.md"
G1_FIGURE_NAMES = (
    "g1_lux_normalization.png",
    "g1_psf_mtf.png",
    "g1_energy_conservation.png",
    "g1_edge_angle.png",
)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

REFERENCE_PIXEL_PITCH_UM = 1.12
REFERENCE_WAVELENGTH_NM = 550.0
REFERENCE_F_NUMBER = 1.8

# The exact first zero of the Bessel function J1. Conventionally rounded to
# 1.22, which shifts the reference-lens zero from 1.07810 to 1.07839 pixels.
J1_FIRST_ZERO = 3.8317059702075125


def _reference_lens() -> LensConfig:
    """Return the reference phone lens configuration used throughout G1."""
    return LensConfig(
        f_number=REFERENCE_F_NUMBER,
        focal_length_mm=6.4,
        sensor_diagonal_mm=11.46,
        transmittance=0.9,
    )


def test_g1_lux_normalization() -> None:
    """683 lm/W times the V-weighted irradiance integral equals the lux value."""
    e = lux_to_spectral_irradiance(100.0, illuminant("D65"))
    v = photopic_luminous_efficiency(e.wavelengths_nm.nm)

    # The integral is over nanometres and E is a per-nanometre density. This is
    # the corrected plan test: no 1e-9 metre conversion belongs here.
    assert K_M * np.trapezoid(e.data[0, 0] * v, e.wavelengths_nm.nm) == pytest.approx(
        100.0, rel=1e-3
    )


@pytest.mark.parametrize("illuminant_name", ["D65", "3000K", "4000K"])
@pytest.mark.parametrize("lux", [1.0, 100.0, 100_000.0])
def test_g1_lux_normalization_across_illuminants_and_levels(
    illuminant_name: str, lux: float
) -> None:
    """Every supported illuminant and a 1-100,000 lux span normalizes exactly."""
    e = lux_to_spectral_irradiance(lux, illuminant(illuminant_name))
    v = photopic_luminous_efficiency(e.wavelengths_nm.nm)
    measured = K_M * np.trapezoid(e.data[0, 0] * v, e.wavelengths_nm.nm)

    assert measured == pytest.approx(lux, rel=1e-3)


def test_g1_zero_lux_is_a_valid_dark_scene() -> None:
    """Zero illuminance is legal and must produce exactly zero irradiance."""
    e = lux_to_spectral_irradiance(0.0, illuminant("D65"))

    assert np.all(e.data == 0.0)


def test_g1_analytic_mtf() -> None:
    """A unit-sigma Gaussian's discrete MTF50 matches the closed-form value."""
    measured = psf_mtf50(gaussian_psf(1.0, size=31))
    analytic = math.sqrt(math.log(2.0) / (2.0 * math.pi**2))

    assert measured == pytest.approx(0.1874, abs=0.01)
    # The gate tolerance above is deliberately loose; the numerical evaluation
    # of the closed form is far tighter, so record both.
    assert measured == pytest.approx(analytic, abs=1e-4)


@pytest.mark.parametrize(
    "psf",
    [
        gaussian_psf(1.0, size=31),
        airy_psf(REFERENCE_F_NUMBER, REFERENCE_WAVELENGTH_NM, REFERENCE_PIXEL_PITCH_UM, size=31),
        box_psf(4.0),
    ],
    ids=["gaussian", "airy", "box"],
)
def test_g1_psf_normalization(psf: np.ndarray) -> None:
    """Every PSF model returns a non-negative kernel whose sum is one."""
    assert psf.sum() == pytest.approx(1.0)
    assert np.all(psf >= 0.0)


def test_g1_airy_first_zero_for_reference_lens() -> None:
    """The reference lens's Airy first zero uses x = pi r / (lambda N / pitch)."""
    airy_scale_px = REFERENCE_WAVELENGTH_NM * 1e-3 * REFERENCE_F_NUMBER / REFERENCE_PIXEL_PITCH_UM
    first_zero_px = J1_FIRST_ZERO / math.pi * airy_scale_px

    assert first_zero_px == pytest.approx(1.078, abs=1e-3)
    # The conventional 1.22 approximation agrees with the exact zero to 0.03 %.
    assert 1.22 * airy_scale_px == pytest.approx(first_zero_px, rel=1e-3)

    size = 31
    psf = airy_psf(REFERENCE_F_NUMBER, REFERENCE_WAVELENGTH_NM, REFERENCE_PIXEL_PITCH_UM, size=size)
    coordinates = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    x = math.pi * np.hypot(xx, yy) / airy_scale_px
    reference = np.ones_like(x)
    nonzero = x != 0.0
    reference[nonzero] = (2.0 * j1(x[nonzero]) / x[nonzero]) ** 2
    reference /= reference.sum()

    # At f/1.8 the first dark ring (1.078 px) falls between pixel centres, so
    # the sampled kernel is validated against the closed-form profile instead
    # of by finding a resolved ring minimum.
    np.testing.assert_allclose(psf, reference, atol=1e-12)
    assert psf.sum() == pytest.approx(1.0)


def test_g1_downsample_mean() -> None:
    """A 2x box average of a constant cube preserves the mean exactly."""
    assert box_downsample(np.ones((8, 8, 3)), 2).mean() == pytest.approx(1.0)


def test_g1_energy_conservation_through_psf_and_downsampling() -> None:
    """PSF convolution plus 4x pixel integration conserves energy within 1 %."""
    source = np.zeros((64, 64, 3))
    source[30:34, 30:34, :] = 1.0
    blurred = apply_psf(source, gaussian_psf(1.0, size=11))

    # A symmetric kernel with reflected borders conserves energy to round-off,
    # before the block mean; the block mean then preserves the total because
    # sum(y) * factor**2 == sum(x) by construction.
    assert blurred.sum() == pytest.approx(source.sum(), rel=1e-12)
    downsampled = box_downsample(blurred, 4)
    assert downsampled.sum() * 4**2 == pytest.approx(source.sum(), rel=1e-2)


@pytest.mark.parametrize("angle_deg", [-20.0, -5.0, 0.5, 5.0, 20.0])
def test_g1_slanted_edge_angle_recovery(angle_deg: float) -> None:
    """The target/estimator pair recovers near-vertical tilts within 0.5 deg."""
    target = slanted_edge_target((256, 256), angle_deg=angle_deg)

    assert estimate_edge_angle(target) == pytest.approx(angle_deg, abs=0.5)


def test_g1_spectral_cube_contract() -> None:
    """Matching cubes are accepted and shape/unit/value mismatches raise."""
    wavelengths_nm = np.arange(400.0, 701.0, 50.0)
    data = np.ones((8, 8, wavelengths_nm.size))
    cube = SpectralIrradiance.from_array(
        data,
        wavelengths_nm,
        "W/m^2/nm",
        pixel_pitch_um=4.48,
    )

    assert cube.data.shape == (8, 8, 7)
    assert cube.unit == "W/m^2/nm"
    assert cube.pixel_pitch_um == pytest.approx(4.48)

    with pytest.raises(ValueError, match="wavelength"):
        SpectralIrradiance.from_array(
            np.ones((8, 8, wavelengths_nm.size + 1)),
            wavelengths_nm,
            "W/m^2/nm",
        )
    with pytest.raises(ValueError, match="unit"):
        SpectralIrradiance.from_array(data, wavelengths_nm, "W/m^2/um")
    with pytest.raises(ValueError, match="non-negative"):
        SpectralIrradiance.from_array(-data, wavelengths_nm, "W/m^2/nm")

    with pytest.raises(ValueError, match="not covered"):
        cube.resample_wavelengths(np.arange(350.0, 701.0, 50.0))


def test_g1_explicit_resampling_conserves_energy() -> None:
    """Spatial and wavelength resampling preserve the represented integral."""
    wavelengths_nm = np.arange(400.0, 701.0, 50.0)
    cube = SpectralIrradiance.from_array(
        np.ones((8, 8, wavelengths_nm.size)),
        wavelengths_nm,
        "W/m^2/nm",
        pixel_pitch_um=4.48,
    )

    spatial = cube.resample_spatial((2, 2))
    assert spatial.data.shape == (2, 2, wavelengths_nm.size)
    assert spatial.data.sum() * 4**2 == pytest.approx(cube.data.sum())
    assert spatial.pixel_pitch_um == pytest.approx(4.48 * 4)

    # Resample onto a finer grid spanning the same 400-700 nm domain, so the
    # integrals cover the same physical range and must agree.
    integrated = cube.resample_wavelengths(np.arange(400.0, 701.0, 25.0))
    source_integral = np.trapezoid(cube.data[0, 0], cube.wavelengths_nm.nm)
    target_integral = np.trapezoid(integrated.data[0, 0], integrated.wavelengths_nm.nm)
    assert target_integral == pytest.approx(source_integral, rel=1e-12)


def test_g1_reference_lens_geometry() -> None:
    """The reference lens reproduces its 84 deg FOV and 99.73 px/deg scale."""
    lens = _reference_lens()

    assert lens.field_of_view_deg == pytest.approx(84.0, abs=1.0)
    assert lens.pixels_per_degree(1.12) == pytest.approx(99.73, rel=1e-3)


@pytest.mark.parametrize("figure_name", G1_FIGURE_NAMES)
def test_g1_figure_exists_and_is_a_nonempty_png(figure_name: str) -> None:
    """Each committed G1 figure must exist, be non-empty, and be a real PNG."""
    path = REPO_ROOT / "results" / "figures" / figure_name

    assert path.is_file(), f"missing G1 figure {path}"
    assert path.stat().st_size > 0
    assert path.read_bytes()[: len(PNG_MAGIC)] == PNG_MAGIC


def test_g1_report_records_measurements_and_limitations() -> None:
    """The report must state the verdict, key values, tolerances, and limits."""
    assert G1_REPORT_PATH.is_file(), f"missing G1 report {G1_REPORT_PATH}"
    text = G1_REPORT_PATH.read_text(encoding="utf-8").lower()

    for required in (
        "pass",
        "photometric normalization",
        "analytic psf mtf",
        "energy conservation",
        "slanted-edge angle",
        "spectral cube",
        "reference lens",
        "tolerance",
        "assumptions and limitations",
        "683",
        "0.187",
        "1.078",
        "99.73",
    ):
        assert required in text, f"G1 report does not record {required!r}"
