"""Sampled point-spread functions and their modulation transfer functions.

A point-spread function (PSF) is the normalized irradiance pattern that a lens
and pixel form from an ideal point source. Three models cover the useful range:

* **Gaussian.** An aberration- and defocus-dominated blur, written
  ``exp(-r**2 / (2 sigma**2))`` with ``r`` in pixels. A real lens at moderate
  apertures often measures close to Gaussian in the core; ``sigma`` of 0.4-3
  pixels spans a sharp phone lens to a visibly soft one.
* **Box.** The pixel aperture itself. A square pixel of width ``w`` pixels
  integrates light over a box, whose continuous transfer function is
  ``sinc(w f)`` with its first zero at ``f = 1 / w``. On the sampled grid the
  box is area-weighted across partially covered pixels so that non-integer
  widths remain meaningful and the kernel sums to one.
* **Airy.** Diffraction from a circular pupil: ``(2 J1(x) / x)**2`` with
  ``x = pi * r_px * pixel_pitch / (wavelength * f_number)``. The first zero of
  ``J1`` at ``x = 3.8317`` puts the first dark ring at the customary
  ``r0 = 1.22 * wavelength * f_number / pixel_pitch`` pixels. A phone lens at
  f/1.8 and 550 nm with a 1.12 um pitch has ``r0`` just above one pixel, so
  diffraction is already visible at the pixel scale; stopping down to f/8 puts
  the first zero near five pixels and dominates the MTF.

The modulation transfer function is the magnitude of the normalized Fourier
transform of the PSF. ``psf_mtf`` evaluates the exact discrete-time transform
at arbitrary frequencies rather than interpolating an FFT grid, which keeps
MTF50 accurate: with a two-dimensional kernel, summing along one axis gives
the projection whose one-dimensional transform is the axis cut of the 2-D OTF.
Frequencies are in cycles/pixel and therefore cannot exceed the Nyquist limit
of 0.5. ``psf_mtf50`` reports where that curve first crosses one half, the
standard single-number sharpness metric; for a unit-sigma Gaussian the
analytic value is ``sqrt(ln 2 / (2 pi**2)) = 0.18739`` cycles/pixel.

Real-world ranges: phone lenses reach MTF50 of roughly 0.2-0.4 cycles/pixel at
the image centre; diffraction-limited systems at f/8 fall below 0.1. Astigmatic
or misaligned lenses are not rotationally symmetric, which is why every
function here returns a full 2-D kernel rather than a radial profile.

Failure artefacts guarded here: an unnormalized kernel changes image
brightness; a non-positive sigma or f-number has no physical meaning; negative
or zero-sum kernels are not radiometric PSFs; truncating a wide Gaussian or
Airy core into too small a grid discards tails that renormalization then
redistributes, biasing MTF high and adding aliasing/oscillation; frequencies
beyond Nyquist alias and can make MTF appear to rise again; and a kernel whose
MTF never crosses 0.5 lacks a finite MTF50.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import j1  # type: ignore[import-untyped]

__all__ = ["airy_psf", "box_psf", "gaussian_psf", "psf_mtf", "psf_mtf50"]

# Six sigma across the kernel is approximately three sigma either side of the
# peak, which contains 99.7 % of a continuous Gaussian's energy; the tiny
# missing tails are renormalized away.
_GAUSSIAN_TAIL_SIGMA: Final[float] = 6.0

# Twelve first-zero radii across the kernel (about six either side of the peak)
# put the truncation far into the faint rings while keeping the default kernel
# small enough to convolve.
_AIRY_TAIL_FIRST_ZERO_RADII: Final[float] = 12.0
_AIRY_MIN_SIZE: Final[int] = 31

# A 2.5e-5 cycles/pixel grid makes linear interpolation of the 0.5 crossing far
# more accurate than the 0.01 cycles/pixel tolerance required of the metric.
_MTF50_SAMPLES: Final[int] = 20001
_NYQUIST_CP: Final[float] = 0.5


def _positive(value: float, name: str) -> float:
    """Return ``value`` as a finite positive float or raise ``ValueError``."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
    return number


def _positive_size(size: int) -> int:
    """Validate a user-supplied kernel side length."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"size must be a positive integer; got {size!r}")
    return size


def _odd_at_least(minimum: int) -> int:
    """Return the smallest odd integer greater than or equal to ``minimum``."""
    return minimum if minimum % 2 else minimum + 1


def _centred_coordinates(size: int) -> NDArray[np.float64]:
    """Return ``size`` pixel-centre coordinates with the array centred on zero.

    Odd sizes place one pixel exactly at the origin; even sizes place the
    origin between the two central pixels, preserving symmetry for both.
    """
    return np.arange(size, dtype=np.float64) - (size - 1) / 2.0


def _normalized(psf: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Normalize a non-negative kernel to unit sum, rejecting degenerate input."""
    total = float(psf.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} collapsed to a kernel with no positive energy")
    return psf / total


def gaussian_psf(sigma_px: float, size: int | None = None) -> NDArray[np.float64]:
    """Return a normalized, sampled Gaussian PSF.

    Parameters
    ----------
    sigma_px:
        Standard deviation in pixels; must be finite and positive. At the
        reference 1.12 um pitch, ``sigma_px = 1`` corresponds to 1.12 um of
        blur at the sensor plane.
    size:
        Odd or even side length of the square kernel. ``None`` uses
        ``ceil(6 * sigma_px)`` made odd, i.e. approximately three sigma either
        side of the peak.

    Returns
    -------
    A ``(size, size)`` float64 kernel whose sum is exactly one (up to
    floating-point round-off).

    A non-positive or non-finite sigma describes no physical blur; a sigma
    larger than roughly ``size / 6`` is truncated by the grid, and
    renormalizing the remaining core biases MTF high.
    """
    sigma = _positive(sigma_px, "sigma_px")
    if size is None:
        resolved_size = _odd_at_least(max(1, math.ceil(_GAUSSIAN_TAIL_SIGMA * sigma)))
    else:
        resolved_size = _positive_size(size)

    coordinates = _centred_coordinates(resolved_size)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius_squared = xx * xx + yy * yy
    kernel = np.exp(-radius_squared / (2.0 * sigma * sigma))
    return _normalized(kernel, "gaussian_psf")


def box_psf(width_px: float, size: int | None = None) -> NDArray[np.float64]:
    """Return a normalized square pixel-aperture (box) PSF.

    The box of ``width_px`` pixels is centred on the array and its overlap with
    each pixel cell is integrated, so fractional widths spread weight across
    the partially covered pixels instead of being thresholded. That makes the
    discrete kernel a sampled continuous box and keeps its transfer function
    close to ``|sinc(width_px * f)|`` up to the first zero.

    Parameters
    ----------
    width_px:
        Aperture width in pixels; must be finite and positive and may not
        exceed ``size`` or part of the box would fall outside the kernel.
    size:
        Side length of the square kernel. ``None`` uses the smallest odd size
        that contains the box.

    Boxes wider than one pixel produce a sampled transfer function close to
    ``|sinc(width_px * f)|``. A one-pixel box exactly covers one pixel centre,
    so it samples to a delta kernel and its continuous ``sinc(f)`` response is
    applied by the pixel integration itself, not by this sampled kernel.
    """
    width = _positive(width_px, "width_px")
    if size is None:
        resolved_size = _odd_at_least(max(1, math.ceil(width)))
    else:
        resolved_size = _positive_size(size)
    if width > resolved_size:
        raise ValueError(f"box width_px ({width:g}) must not exceed the PSF size ({resolved_size})")

    # Pixel i occupies the unit cell [i, i + 1); integrate the continuous box
    # over each cell. The clip discards cells the box does not reach.
    edges = np.arange(resolved_size + 1, dtype=np.float64)
    low = 0.5 * (resolved_size - width)
    high = 0.5 * (resolved_size + width)
    weights = np.clip(np.minimum(edges[1:], high) - np.maximum(edges[:-1], low), 0.0, None)
    kernel = np.outer(weights, weights)
    return _normalized(kernel, "box_psf")


def airy_psf(
    f_number: float,
    wavelength_nm: float,
    pixel_pitch_um: float,
    size: int | None = None,
) -> NDArray[np.float64]:
    """Return a normalized, sampled Airy (diffraction-limited) PSF.

    The intensity profile is ``(2 J1(x) / x)**2`` where
    ``x = pi * r_px * pixel_pitch / (wavelength * f_number)``, with all lengths
    in the same units. Since ``J1`` first vanishes at ``x = 3.8317``, the first
    dark ring lies at ``r0 = 3.8317 / pi * wavelength * f_number / pitch``,
    conventionally rounded to ``1.22 * wavelength * f_number / pitch`` pixels.
    Wavelength is converted from nanometres to micrometres to match the pitch,
    both lengths being measured at the sensor.

    Parameters
    ----------
    f_number:
        Relative aperture ``N``; must be finite and positive.
    wavelength_nm:
        Free-space wavelength in nanometres; must be finite and positive.
        550 nm is the photopic peak used for reference calculations.
    pixel_pitch_um:
        Sensor pixel pitch in micrometres; must be finite and positive.
    size:
        Side length of the square kernel. ``None`` uses at least 31 pixels and
        twelve first-zero radii across the kernel (about six either side),
        enough for the first several rings.

    Returns
    -------
    A ``(size, size)`` float64 kernel normalized to unit sum.

    The Airy pattern carries a finite fraction of its energy outside any
    truncation, so normalization conserves the captured energy rather than the
    infinite-plane total; renormalizing a grid much smaller than a few
    first-zero radii removes the faint rings, biasing MTF high and introducing
    aliasing/oscillation.
    """
    aperture = _positive(f_number, "f_number")
    wavelength_um = _positive(wavelength_nm, "wavelength_nm") * 1e-3
    pitch_um = _positive(pixel_pitch_um, "pixel_pitch_um")

    # x = pi * r_px / (lambda_um * N / pitch_um); the Airy scale below is the
    # radius in pixels at which x = pi, and 1.22 times it is the first zero.
    airy_scale_px = wavelength_um * aperture / pitch_um
    first_zero_px = 1.22 * airy_scale_px
    if size is None:
        resolved_size = _odd_at_least(
            max(_AIRY_MIN_SIZE, math.ceil(_AIRY_TAIL_FIRST_ZERO_RADII * first_zero_px))
        )
    else:
        resolved_size = _positive_size(size)

    coordinates = _centred_coordinates(resolved_size)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius_px = np.hypot(xx, yy)
    x = math.pi * radius_px / airy_scale_px

    # J1(x)/x tends to 1/2 at the origin, so the profile is one there; compute
    # the ratio only away from x == 0 to avoid a 0/0 warning.
    profile = np.ones_like(x)
    nonzero = x != 0.0
    profile[nonzero] = 2.0 * j1(x[nonzero]) / x[nonzero]
    kernel = profile * profile
    return _normalized(kernel, "airy_psf")


def psf_mtf(psf: ArrayLike, frequencies_cp: ArrayLike) -> NDArray[np.float64]:
    """Return the modulation transfer function of ``psf`` at given frequencies.

    The OTF at ``(fx, fy)`` is the normalized discrete Fourier transform of the
    kernel evaluated exactly at that frequency. Summing the 2-D kernel along
    one axis produces the projection whose 1-D transform is the ``fy = 0`` cut
    of the OTF, which is the conventional one-dimensional MTF for a
    rotationally symmetric or square-separable PSF. Evaluating the sum directly
    avoids FFT-grid interpolation error.

    Parameters
    ----------
    psf:
        A two-dimensional, finite, non-negative kernel with positive sum.
    frequencies_cp:
        One-dimensional frequencies in cycles/pixel in ``[0, 0.5]``. Values
        above Nyquist alias and are rejected; negative frequencies are the
        mirror image of positive ones and are not needed.

    Returns
    -------
    An array with the same shape as ``frequencies_cp``; the zero-frequency
    value is exactly one after DC normalization.
    """
    kernel = np.asarray(psf, dtype=np.float64)
    if kernel.ndim != 2 or kernel.size == 0:
        raise ValueError(f"psf must be a non-empty 2-D array; got shape {kernel.shape}")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("psf values must be finite")
    if np.any(kernel < 0.0):
        raise ValueError("psf values must be non-negative")
    total = float(kernel.sum())
    if total <= 0.0:
        raise ValueError("psf must sum to a positive value for DC normalization")

    frequencies = np.asarray(frequencies_cp, dtype=np.float64)
    if frequencies.ndim != 1:
        raise ValueError("frequencies_cp must be a 1-D array")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies_cp must be finite")
    if np.any((frequencies < 0.0) | (frequencies > _NYQUIST_CP)):
        raise ValueError(
            "frequencies_cp must lie within [0, 0.5] cycles/pixel (Nyquist); "
            f"got {frequencies.min():g} to {frequencies.max():g}"
        )

    projection = kernel.sum(axis=0)
    coordinates = _centred_coordinates(kernel.shape[1])
    phase = 2.0 * math.pi * np.outer(frequencies, coordinates)
    real = np.cos(phase) @ projection
    imaginary = -np.sin(phase) @ projection
    mtf: NDArray[np.float64] = np.hypot(real, imaginary) / total
    return mtf


def psf_mtf50(psf: ArrayLike) -> float:
    """Return the first frequency where the PSF's MTF crosses one half.

    The crossing is located on a dense ``[0, 0.5]`` cycles/pixel grid and
    refined by linear interpolation, which is accurate because MTF is smooth
    near 0.5 for the kernels in this module. Airy and box MTFs oscillate and
    can cross 0.5 more than once; the first crossing is the conventional
    MTF50.

    Raises
    ------
    ValueError
        If the MTF never falls to 0.5 by Nyquist, as for a single-pixel delta
        whose MTF is one at every sampled frequency.
    """
    frequencies = np.linspace(0.0, _NYQUIST_CP, _MTF50_SAMPLES)
    mtf = psf_mtf(psf, frequencies)
    crossings = np.flatnonzero(mtf < 0.5)
    if crossings.size == 0:
        raise ValueError(
            "MTF does not fall to 0.5 by Nyquist (0.5 cycles/pixel); MTF50 is undefined"
        )

    index = int(crossings[0])
    if index == 0:  # DC is exactly one, so only a degenerate input reaches here.
        return 0.0
    low_frequency = float(frequencies[index - 1])
    high_frequency = float(frequencies[index])
    low_mtf = float(mtf[index - 1])
    high_mtf = float(mtf[index])
    fraction = (0.5 - low_mtf) / (high_mtf - low_mtf)
    return low_frequency + (high_frequency - low_frequency) * float(fraction)
