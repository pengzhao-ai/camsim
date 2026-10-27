"""PSF propagation and pixel-aperture downsampling for spectral image cubes.

A lens does not create or destroy light. For a point-spread function
:math:`h(x, y)` normalized to unit integral, the image irradiance is the
convolution ``E_img = E_obj * h``, and integrating over the image plane returns
the object-plane total. The discrete implementation must preserve that
statement: the kernel is renormalized to unit sum, and the frame border is
extended by reflection (half-sample symmetric, so the edge sample is repeated)
rather than padded with zeros. Zero padding absorbs a fraction of the energy
proportional to the kernel area at the border; for a compact seven-pixel kernel
on a 256-pixel frame that is a few per cent, and it masquerades as a
field-dependent exposure error.

Convolution is performed as one FFT over all wavelength planes at once, with an
explicitly reflected pad so the result is equivalent to direct convolution with
reflecting boundaries. The kernel origin is the sample at index ``size // 2``
in each axis, which is the exact centre for the odd-sized kernels returned by
:mod:`camsim.optics.psf`. An even-sized kernel has no single-sample centre:
index ``size // 2`` is the second of its two central samples, so a delta kernel
there is an exact identity while a symmetric even-sized Gaussian splits its
response across the central pair (a half-pixel phase). References that need
sub-pixel centring should therefore use odd kernel sizes.

The second operation models the pixel aperture. A sensor pixel integrates
irradiance over its footprint, so rendering a scene on a grid that is ``N``
times finer than the sensor and averaging each ``N x N`` block gives the
sensor-plane sample. The block mean preserves per-unit-area irradiance; a block
sum would scale the image by ``N**2``. The factor must divide both spatial
dimensions exactly because a partial block would represent a fractional pixel
area, which the scalar-pitch frame metadata cannot describe.

Real-world sampling factors are 1 (native render), 2-4 (typical anti-aliasing
for MTF and edge measurements), and up to 8 for diffraction-limited or very
sharp systems. Kernel support is usually 5-31 pixels; a Gaussian is truncated
near three sigma, and an Airy pattern at several first-zero radii.

Failure artefacts guarded here: an unnormalized kernel changes image
brightness; zero or wrap padding introduces a border brightness ramp or
teleports energy across the frame; a non-reflecting downsample edge leaves a
partial-block seam; summing instead of averaging multiplies every irradiance
by the block area; and non-finite, negative, or zero-sum kernels are not
radiometric PSFs. FFT round-off can leave values near ``-1e-17`` in an
otherwise non-negative cube; those round-off artefacts are clipped so the cube
can be stored as a :class:`~camsim.core.spectral.SpectralIrradiance`.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve  # type: ignore[import-untyped]

__all__ = ["apply_psf", "box_downsample"]

# FFT convolution of non-negative data can produce negative values at the level
# of floating-point round-off. Clipping at exactly zero would also clip genuine
# signed data, so the clip is applied only when the input itself is non-negative.
_NEGATIVE_TOLERANCE: Final[float] = 0.0


def _as_cube(cube: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a finite float64 2-D image or 3-D ``(H, W, L)`` cube."""
    array = np.asarray(cube, dtype=np.float64)
    if array.ndim not in (2, 3):
        raise ValueError(f"{name} must be a 2-D or 3-D array; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty; got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite (no NaN or infinity)")
    return array


def _as_kernel(psf: ArrayLike) -> NDArray[np.float64]:
    """Return a finite, non-negative, unit-sum 2-D PSF kernel.

    Normalizing here makes energy conservation independent of the caller's
    kernel scaling. A PSF is defined up to a positive scale factor, and a
    kernel whose sum is not one would either invent or absorb light.
    """
    kernel = np.asarray(psf, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError(f"PSF must be a 2-D kernel; got shape {kernel.shape}")
    if kernel.size == 0:
        raise ValueError("PSF must not be empty")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("PSF values must be finite")
    if np.any(kernel < 0.0):
        raise ValueError("PSF values must be non-negative")
    total = float(kernel.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("PSF must sum to a positive value")
    return kernel / total


def _reflect_convolution(
    cube: NDArray[np.float64],
    kernel: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convolve every plane of ``cube`` with ``kernel`` using reflected borders."""
    kernel_height, kernel_width = kernel.shape
    # Pad by (size - 1) // 2 in front and the remainder behind. With the
    # "valid" crop this puts the kernel origin at index size // 2: the exact
    # centre for odd sizes, and the second of the two central samples for even
    # sizes. A delta kernel at that index is therefore an exact identity.
    top = (kernel_height - 1) // 2
    left = (kernel_width - 1) // 2
    padding: tuple[tuple[int, int], ...] = (
        (top, kernel_height - 1 - top),
        (left, kernel_width - 1 - left),
    )
    if cube.ndim == 3:
        padding = (*padding, (0, 0))
        spectral_kernel = kernel[:, :, None]
    else:
        spectral_kernel = kernel
    padded: NDArray[np.float64] = np.pad(cube, padding, mode="symmetric")
    convolved: NDArray[np.float64] = np.asarray(
        fftconvolve(padded, spectral_kernel, mode="valid"), dtype=np.float64
    )
    return convolved


def apply_psf(cube: ArrayLike, psf: ArrayLike) -> NDArray[np.float64]:
    """Convolve an image or spectral cube with a 2-D point-spread function.

    Parameters
    ----------
    cube:
        A 2-D ``(H, W)`` image or a 3-D ``(H, W, L)`` cube whose last axis is
        wavelength or colour channel. Values must be finite. The same spatial
        kernel is applied independently to every plane, which is the
        wavelength-independent PSF approximation; a wavelength-dependent PSF
        requires a separate kernel per plane.
    psf:
        A 2-D, finite, non-negative kernel with positive sum. It is normalized
        to unit sum internally, so a PSF defined up to scale behaves
        identically. The origin is the sample at index ``size // 2`` in each
        axis: exact for odd sizes, one sample right/below the geometric centre
        for even sizes.

    Returns
    -------
    A float64 array with the same shape as ``cube``. For non-negative input,
    values are guaranteed non-negative (FFT round-off below zero is clipped);
    signed input is left signed.

    Notes
    -----
    The border is extended by half-sample symmetry (reflection that repeats the
    edge value), so a constant field stays exactly constant for any normalized
    kernel. For a symmetric kernel, total energy is conserved at the border to
    floating-point precision; an asymmetric kernel's reflected wrap can change
    the total by a fraction of a per cent, within the 1 % energy budget. A
    kernel wider than the image is legal: reflection is repeated as needed, and
    the output keeps the input shape.

    Real images are usually convolved once with a kernel sampled on the same
    grid. When a scene is rendered oversampled, the kernel must be sampled on
    that oversampled grid too: a sensor-grid Gaussian of ``sigma`` pixels is
    passed as ``gaussian_psf(sigma * oversampling)``.
    """
    image = _as_cube(cube, "cube")
    kernel = _as_kernel(psf)
    non_negative_input = bool(np.all(image >= 0.0))

    convolved = _reflect_convolution(image, kernel)

    if non_negative_input:
        # Round-off alone can cross zero; clipping restores the physical
        # invariant that irradiance cannot be negative.
        convolved = np.clip(convolved, _NEGATIVE_TOLERANCE, None)
    return convolved


def box_downsample(cube: ArrayLike, factor: int) -> NDArray[np.float64]:
    """Average ``factor x factor`` blocks to model pixel-aperture integration.

    Parameters
    ----------
    cube:
        A 2-D ``(H, W)`` image or a 3-D ``(H, W, L)`` spectral cube. Values must
        be finite.
    factor:
        Positive integer block size. Both spatial dimensions must be exact
        multiples of it.

    Returns
    -------
    A float64 array of shape ``(H / factor, W / factor)`` or
    ``(H / factor, W / factor, L)``. Each output sample is the arithmetic mean
    of its block, so per-unit-area irradiance and the image mean are preserved.

    Raises
    ------
    ValueError
        If ``cube`` is not 2-D/3-D and finite, if ``factor`` is not a positive
        integer, or if the spatial dimensions do not divide exactly.

    A partial block at the frame edge would cover a fractional pixel area that
    the scalar-pitch ``SpectralIrradiance`` metadata cannot represent, and it
    would bias edge pixels towards whatever padding was used, so non-divisible
    shapes are rejected rather than silently cropped.
    """
    image = _as_cube(cube, "cube")
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)):
        raise ValueError(f"factor must be a positive integer; got {factor!r}")
    block = int(factor)
    if block < 1:
        raise ValueError(f"factor must be a positive integer; got {factor!r}")

    height, width = image.shape[:2]
    if height % block != 0 or width % block != 0:
        raise ValueError(f"spatial shape {(height, width)} must divide evenly by factor {block}")

    if image.ndim == 3:
        reduced = image.reshape(
            height // block,
            block,
            width // block,
            block,
            image.shape[2],
        ).mean(axis=(1, 3))
    else:
        reduced = image.reshape(height // block, block, width // block, block).mean(axis=(1, 3))
    return np.asarray(reduced, dtype=np.float64)
