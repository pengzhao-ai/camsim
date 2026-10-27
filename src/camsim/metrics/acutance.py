"""Edge acutance: contrast-normalized gradient energy as an image-sharpness proxy.

Acutance is the perceptual correlate of a sharp edge. A lens, a pixel aperture,
demosaicing, and any denoiser all low-pass the image, spreading a transition that
left the scene as a near-step over several pixels; a sharpener does the reverse
and concentrates it. This module reduces the profile to one number through a
short, reproducible chain:

1. **Crop.** The ROI is a half-open sensor-pixel rectangle, extracted with
   :func:`camsim.metrics.chart.extract_patch`, which also enforces bounds and
   rejects non-real or non-finite samples.
2. **Luma.** A 2-D image is already the plane to measure. An ``(H, W, 3)`` RGB
   image is reduced with the Rec. 709 weights
   ``Y = 0.2126 R + 0.7152 G + 0.0722 B``, applied to the values as supplied.
   The weights sum to one, so a neutral patch keeps its level.
3. **Contrast normalization.** The profile is mapped to
   ``(profile - min) / (max - min)``. The edge's own swing therefore cancels:
   doubling the exposure or halving the chart contrast leaves the number
   unchanged. A flat ROI (``max == min``) contains no edge and returns exactly
   ``0.0`` rather than dividing by zero.
4. **Gradient energy.** Central differences along both axes (``np.gradient``)
   approximate ``dY/dx`` and ``dY/dy`` per pixel. Their root-mean-square,
   ``sqrt(mean(gx**2 + gy**2))``, is the acutance: the normalized edge energy
   averaged over the ROI.

What the number means, and what it does not
-------------------------------------------
This is a **sharpness proxy for correlation studies**, not a calibrated physical
quantity, and it is not ISO 12233 or any standard acutance definition. It has no
units beyond "normalized value per pixel". Its magnitude depends on the ROI
size: a perfect one-pixel step in a ``W``-pixel-wide ROI measures
``1 / sqrt(2 W)`` under this estimator, so comparing values across differently
sized ROIs is meaningless. A sharp but noisy edge can measure higher than a clean
one, because noise is gradient energy too, and a sharpening halo raises the
number even when it damages a machine-vision task. For quantitative resolution,
use :func:`camsim.metrics.mtf.slanted_edge_mtf`; acutance is the cheap summary
that can be correlated with rendering parameters across a sweep.

Real-world ranges: edge regions are typically 100-400 pixels tall and inset from
the frame border; the normalized gradient RMS for a clean optical edge generally
falls between about 0.005 (a heavily blurred, wide ROI) and 0.5 (a nearly ideal
step in a narrow ROI). Values are dimensionless and only comparable between
images measured with the same ROI geometry.

Failure artefacts guarded here
------------------------------
* **Display-encoded input.** A tone curve is non-linear, so the same optical edge
  produces a different gradient profile after the OETF; acutance measured on
  display values is not comparable with acutance measured on linear light.
* **Flat ROI.** No transition means no sharpness; returning zero is explicit and
  documented, not a silent division by zero.
* **Noise and compression.** Sensor noise, JPEG blocks, and denoiser smoothing
  all change gradient energy; a single acutance number cannot separate them.
* **ROI size and placement.** The estimator averages per pixel, and a ROI that
  clips the edge or includes two edges measures neither cleanly. The ROI should
  be inset and the same for every image in a comparison.
* **Border effects.** ``np.gradient`` uses one-sided differences at the ROI
  boundary, so a region ending exactly on the transition undercounts it; inset
  the ROI by at least the blur width on every side.

The module imports NumPy and :mod:`camsim.metrics.chart` only - never the
sensor, ISP, scene, or optics packages - so the measurement harness cannot
depend on the model it is meant to judge. The function is deterministic and does
not modify its input.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.metrics.chart import PatchROI, extract_patch

__all__ = ["edge_acutance"]

# Rec. 709 luminance weights. They sum to one, so a neutral ROI keeps its level;
# applying them to linear light is what the standard's derivation presumes.
_REC709_WEIGHTS: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)

_RGB_CHANNELS: Final[int] = 3


def _luma_plane(patch: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return Rec. 709 luma for an ``(H, W)`` or ``(H, W, 3)`` ROI as float64."""
    if patch.ndim == 2:
        return patch
    return np.asarray(
        _REC709_WEIGHTS[0] * patch[..., 0]
        + _REC709_WEIGHTS[1] * patch[..., 1]
        + _REC709_WEIGHTS[2] * patch[..., 2],
        dtype=np.float64,
    )


def _axis_gradient(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Return the central-difference gradient along ``axis``, zero if degenerate.

    ``np.gradient`` refuses an axis of length one. A one-row or one-column ROI is
    still meaningful along its other axis, and the missing direction genuinely
    has no variation, so it contributes zeros rather than an error.
    """
    if values.shape[axis] < 2:
        return np.zeros_like(values)
    return np.asarray(np.gradient(values, axis=axis), dtype=np.float64)


def edge_acutance(image: ArrayLike, roi: PatchROI) -> float:
    """Return the contrast-normalized RMS gradient energy of one edge ROI.

    Parameters
    ----------
    image:
        Real, finite ``(H, W)`` mono or ``(H, W, 3)`` RGB array-like. RGB input
        is reduced to Rec. 709 luma with the customary ``0.2126``, ``0.7152``,
        and ``0.0722`` weights. Boolean, complex, non-numeric, and non-finite
        input is rejected by :func:`~camsim.metrics.chart.extract_patch`, as is
        any image that is not 2-D or 3-D. A 3-D image that is not exactly three
        channels wide is rejected here: there is no standard weighting for an
        arbitrary channel count.

        Values may be linear light or display-encoded, but the choice changes
        the result: a tone curve is non-linear, so the same optics measured
        through it does not produce the same gradient profile. Comparisons
        should use one encoding throughout. The image is never modified.
    roi:
        The edge region as a :class:`~camsim.metrics.chart.PatchROI`, lying fully
        inside the image. The ROI supplies the analysis window; it is not
        re-centred, and it should be inset from the edge transition so the
        one-sided gradients at its border do not clip the blur.

    Returns
    -------
    float
        ``sqrt(mean(gx**2 + gy**2))`` computed on the luma profile after mapping
        it to ``[0, 1]`` with ``(profile - min) / (max - min)``. Higher is
        sharper. The result is invariant to exposure and chart contrast by
        construction. A flat ROI (``max == min``) has no edge and returns
        ``0.0``.

    Raises
    ------
    TypeError
        If ``roi`` is not a :class:`~camsim.metrics.chart.PatchROI`.
    ValueError
        If the image is not a finite real 2-D or 3-D array, has a 3-D channel
        count other than three, or the ROI extends past the image boundary.

    Notes
    -----
    This is an image-sharpness proxy for correlation studies, not a calibrated
    physical quantity and not a standard definition. The estimator averages per
    pixel, so its magnitude depends on the ROI size (a one-pixel step in a
    ``W``-pixel ROI measures ``1 / sqrt(2 W)``); compare values only between
    measurements that use the same ROI geometry. Noise, compression, and
    sharpening halos all raise the number, so it must not be read as a
    noise-free resolution measurement - use
    :func:`camsim.metrics.mtf.slanted_edge_mtf` for that. The computation is
    deterministic and allocates its own working arrays.
    """
    patch = extract_patch(image, roi)
    if patch.ndim == 3 and patch.shape[2] != _RGB_CHANNELS:
        raise ValueError(
            "edge_acutance requires a 2-D mono image or an (H, W, 3) RGB image; "
            f"got {patch.shape[2]} channels"
        )
    profile = _luma_plane(np.asarray(patch, dtype=np.float64))

    minimum = float(profile.min())
    maximum = float(profile.max())
    if maximum == minimum:
        # A constant ROI has no transition: sharpness is defined as zero rather
        # than the limit of a division whose denominator has vanished.
        return 0.0
    normalized = np.asarray((profile - minimum) / (maximum - minimum), dtype=np.float64)

    gradient_y = _axis_gradient(normalized, axis=0)
    gradient_x = _axis_gradient(normalized, axis=1)
    energy = np.mean(gradient_x * gradient_x + gradient_y * gradient_y)
    return float(math.sqrt(float(energy)))
