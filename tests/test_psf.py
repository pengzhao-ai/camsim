"""Tests for sampled Gaussian, box, and diffraction-limited Airy PSFs.

A point-spread function is the irradiance pattern a lens forms from an ideal
point source, sampled on the pixel grid. Three models bracket real optics:
an aberration-dominated Gaussian of width ``sigma_px``; the pixel aperture
itself, a box whose Fourier transform is a sinc; and diffraction from a
circular pupil, the Airy pattern ``(2 J1(x) / x)**2`` whose first dark ring
sits at ``1.22 * wavelength * f_number`` in object space, or
``1.22 * wavelength * f_number / pixel_pitch`` pixels.

The modulation transfer function is the normalized magnitude of the PSF's
Fourier transform. At zero frequency the MTF is exactly one (energy is
conserved); it falls with frequency and reaches zero at the diffraction or
aperture cutoff. MTF50, the frequency where MTF crosses 0.5, is the standard
single-number sharpness metric. The analytic Gaussian MTF is
``exp(-2 pi**2 sigma**2 f**2)``, so ``sigma = 1`` pixel gives
``MTF50 = 0.18739`` cycles/pixel; that value anchors these tests against a
closed-form result rather than against the implementation itself.

Failure artefacts guarded here: an unnormalized PSF changes image brightness;
a non-positive sigma has no physical meaning; a negative or zero-sum kernel
cannot be a radiometric PSF; requesting frequencies beyond the 0.5 cycles/pixel
Nyquist limit aliases the MTF; and a PSF whose MTF never crosses one half (for
example a single-pixel delta) has no finite MTF50.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import j1

from camsim.optics.psf import airy_psf, box_psf, gaussian_psf, psf_mtf, psf_mtf50


def test_gaussian_psf_normalized() -> None:
    psf = gaussian_psf(1.0, size=21)

    assert psf.shape == (21, 21)
    assert psf.sum() == pytest.approx(1.0)
    assert np.all(psf >= 0.0)
    center = 21 // 2
    assert psf[center, center] == psf.max()
    np.testing.assert_allclose(psf, psf.T)


def test_gaussian_psf_default_size_covers_six_sigma() -> None:
    sigma_px = 1.5
    psf = gaussian_psf(sigma_px)

    # Six sigma is three sigma either side of the peak, which contains 99.7 %
    # of the continuous Gaussian; the default grid must be odd so it is centred.
    assert psf.shape[0] % 2 == 1
    assert psf.shape[0] >= math.ceil(6.0 * sigma_px)
    assert psf.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("sigma_px", [0.0, -1.0, math.nan, math.inf])
def test_gaussian_psf_rejects_invalid_sigma(sigma_px: float) -> None:
    with pytest.raises(ValueError, match="sigma_px"):
        gaussian_psf(sigma_px)


@pytest.mark.parametrize("size", [0, -3])
def test_gaussian_psf_rejects_invalid_size(size: int) -> None:
    with pytest.raises(ValueError, match="size"):
        gaussian_psf(1.0, size=size)


def test_gaussian_mtf50_analytic() -> None:
    # Continuous Gaussian MTF: exp(-2 pi^2 sigma^2 f^2) = 0.5 at
    # f = sqrt(ln 2 / (2 pi^2)) = 0.18739 cycles/pixel for sigma = 1 pixel.
    assert psf_mtf50(gaussian_psf(1.0, size=31)) == pytest.approx(0.1874, abs=0.01)


def test_airy_first_zero_radius() -> None:
    f_number = 1.8
    wavelength_nm = 550.0
    pixel_pitch_um = 1.12
    size = 41
    psf = airy_psf(f_number, wavelength_nm, pixel_pitch_um, size=size)

    assert psf.shape == (size, size)
    assert psf.sum() == pytest.approx(1.0)
    assert np.all(psf >= 0.0)
    # The pupil is rotationally symmetric, so the sampled PSF must be too.
    np.testing.assert_allclose(psf, psf[::-1, :], atol=1e-15)
    np.testing.assert_allclose(psf, psf[:, ::-1], atol=1e-15)

    # Closed-form reference: the first zero of 2 J1(x)/x is at x = 3.8317,
    # hence at radius 1.22 * wavelength * N / pitch pixels. At 1x sampling the
    # dark ring falls between pixel centres, so compare the whole profile
    # against the analytic expression rather than looking for a sampled zero.
    first_zero_px = 1.22 * (wavelength_nm * 1e-3) * f_number / pixel_pitch_um
    assert 1.0 < first_zero_px < 2.0  # a phone lens is diffraction-limited to ~1 px
    coordinates = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    x = np.pi * np.hypot(xx, yy) / first_zero_px
    reference = np.ones_like(x)
    nonzero = x != 0.0
    reference[nonzero] = (2.0 * j1(x[nonzero]) / x[nonzero]) ** 2
    reference /= reference.sum()
    np.testing.assert_allclose(psf, reference, atol=1e-12)


@pytest.mark.parametrize(
    ("f_number", "wavelength_nm", "pixel_pitch_um", "field"),
    [
        (0.0, 550.0, 1.12, "f_number"),
        (1.8, 0.0, 1.12, "wavelength_nm"),
        (1.8, 550.0, 0.0, "pixel_pitch_um"),
        (math.inf, 550.0, 1.12, "f_number"),
    ],
)
def test_airy_psf_rejects_invalid_optics(
    f_number: float, wavelength_nm: float, pixel_pitch_um: float, field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        airy_psf(f_number, wavelength_nm, pixel_pitch_um, size=11)


def test_box_psf_sinc() -> None:
    width_px = 4.0
    psf = box_psf(width_px)
    assert psf.sum() == pytest.approx(1.0)
    assert np.all(psf >= 0.0)

    frequencies = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
    mtf = psf_mtf(psf, frequencies)
    # A continuous box of width w has MTF |sinc(w f)|. Area sampling the box on
    # the pixel grid adds a small pixel-aperture term and aliases the tails, so
    # the sampled curve is only an approximation; 0.05 covers that error.
    np.testing.assert_allclose(mtf, np.sinc(width_px * frequencies), atol=0.05)
    # The first zero sits at f = 1 / width = Nyquist for a four-pixel box.
    assert mtf[-1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("width_px", [0.0, -1.0, math.nan])
def test_box_psf_rejects_invalid_width(width_px: float) -> None:
    with pytest.raises(ValueError, match="width_px"):
        box_psf(width_px)


def test_box_psf_rejects_width_larger_than_grid() -> None:
    with pytest.raises(ValueError, match="exceed"):
        box_psf(7.0, size=5)


@pytest.mark.parametrize(
    "psf",
    [
        gaussian_psf(1.0, size=11),
        airy_psf(2.0, 550.0, 1.12, size=21),
        box_psf(4.0),
    ],
)
def test_psf_mtf_dc_is_one(psf: NDArray[np.float64]) -> None:
    mtf = psf_mtf(psf, np.array([0.0, 0.25, 0.5]))

    assert mtf.shape == (3,)
    assert mtf[0] == pytest.approx(1.0)
    assert np.all(mtf <= 1.0 + 1e-12)


@pytest.mark.parametrize("frequency_cp", [-0.01, 0.51, math.nan])
def test_psf_mtf_rejects_invalid_frequencies(frequency_cp: float) -> None:
    with pytest.raises(ValueError, match="frequenc"):
        psf_mtf(gaussian_psf(1.0, size=11), [frequency_cp])


def test_psf_mtf_rejects_invalid_psf() -> None:
    with pytest.raises(ValueError, match="2-D"):
        psf_mtf(np.ones(5), [0.0])
    with pytest.raises(ValueError, match="non-negative"):
        psf_mtf(np.array([[-1.0, 1.0], [0.0, 0.0]]), [0.0])
    with pytest.raises(ValueError, match="positive"):
        psf_mtf(np.zeros((3, 3)), [0.0])


def test_psf_mtf50_requires_a_crossing() -> None:
    delta = np.zeros((11, 11))
    delta[5, 5] = 1.0
    with pytest.raises(ValueError, match="Nyquist"):
        psf_mtf50(delta)
