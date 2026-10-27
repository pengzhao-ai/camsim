"""Thresholded unsharp-mask sharpening in linear light.

No real camera delivers the scene's full contrast at every spatial frequency.
The lens PSF (diffraction, aberrations), the pixel aperture, the
anti-aliasing/OLPF filter, and demosaicing all attenuate high frequencies, so
the rendered image looks softer than the scene. Unsharp masking restores
apparent contrast by adding back a scaled high-pass residual computed from a
blurred copy of the image::

    highpass = x - gaussian_blur(x, radius)
    gated    = highpass where |highpass| > threshold, else 0
    out      = x + amount * gated

The blur is the same separable Gaussian used by the denoise stage:
``sigma = radius``, truncated at four sigma, with symmetric reflection at the
borders. Subtracting it leaves the detail the blur removed; adding it back
amplifies that detail by ``amount``. Real arithmetic makes the stage linear in
``amount`` for a fixed gate, and the gate plus the addition are its two
non-linearities. All computation is in linear float32 light, before tone
mapping, because sharpening gamma-encoded data applies a gain that depends on
the signal level and produces halos with the wrong shape.

The threshold is a hard gate in linear intensity units: residuals with
``|highpass| <= threshold`` are set exactly to zero, which suppresses
sharpening of sensor noise and low-contrast texture. Residuals above the
threshold pass unchanged -- hard thresholding, not soft shrinkage -- so the
gate is discontinuous at the boundary; a residual exactly at the threshold is
suppressed. ``threshold = 0`` sharpens everything, including noise.

``amount`` is the high-pass gain. ``1`` doubles the residual, the useful
photographic range is roughly 0.2-1.5, and values above about 2 make halos and
grain obvious in a typical render. Output-referred workflows use smaller
amounts than scene-referred ones because tone mapping later stretches the
shadows. Blur radii of 0.5-3 px are typical: a sub-pixel radius produces fine
crispening, a radius near 2-3 px produces local-contrast (clarity) effects
with visible halos of the same width. Thresholds of 0-0.02 in linear units
span "sharpen everything" to "only strong edges", scaled to the noise level of
the frame. Nothing is clipped here: undershoot below zero and overshoot above
one are preserved because the tone stage owns the display range, and a
strong radius/amount pair around a hard edge is the classic bright-dark halo.
``amount = 0`` or ``enabled = False`` returns an exact float32 copy without
blurring at all.

By default the stage sharpens luminance only. For RGB input it computes luma
``Y = 0.2126 R + 0.7152 G + 0.0722 B`` in the reversible BT.709 opponent
space, gates and amplifies ``Y``'s high-pass, and adds the resulting detail
equally to R, G, and B. That construction is exact rather than an
approximation: raising ``Y`` by ``d`` while holding ``Cb`` and ``Cr`` fixed
implies ``R' = R + d`` and ``B' = B + d``, and the luma equation then gives
``G' = G + d`` because the BT.709 weights sum to one. Chroma is therefore
untouched, which matters because sharpening R, G, and B independently
amplifies demosaic error and paints coloured fringes on high-contrast edges.
``luma_only=False`` deliberately takes the per-channel path instead, for
comparisons and for deliberately aggressive looks. A 2-D mono image is a
single channel with no chroma, so it is sharpened directly and ``luma_only`` is
ignored.

Failure artefacts guarded or documented here: halos and overshoot (large
amount or radius), amplified sensor noise and grain (mitigated by the
threshold), chroma fringing from per-channel sharpening (avoided by the
luma-only default), ringing on fine textures, sharpening in a non-linear
domain, clipping (deliberately not done here), and silently accepting
non-finite, boolean, complex, empty, or wrongly shaped input or a mutated
config whose values no longer satisfy the schema. Outputs are new
C-contiguous float32 arrays of the input shape; the input is never mutated.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.config.schema import SharpenConfig
from camsim.isp.denoise import (
    _as_checked_float32,
    _as_float32_image,
    _validate_bool,
    _validate_non_negative,
    _validate_radius,
    gaussian_blur,
    rgb_to_opponent,
)

__all__ = ["SharpenConfig", "unsharp_mask"]


def unsharp_mask(image: ArrayLike, config: SharpenConfig) -> NDArray[np.float32]:
    """Sharpen a linear image with a thresholded unsharp mask.

    Parameters
    ----------
    image:
        A real numeric ``(H, W)`` mono image or ``(H, W, 3)`` linear RGB image.
        Values must be finite; negatives are accepted because earlier linear
        stages can leave excursions below zero. The input is never modified.
    config:
        Validated :class:`~camsim.config.schema.SharpenConfig`. ``amount``
        scales the gated high-pass, ``radius`` is the Gaussian sigma in
        pixels, ``threshold`` is the inclusive low-contrast cut-off in linear
        intensity units, and ``luma_only`` selects luminance-only sharpening
        for RGB input. ``enabled=False`` returns an exact copy once the config
        and image have been validated.

    Returns
    -------
    A new C-contiguous float32 array with the same shape as ``image``. RGB
    input with ``luma_only=True`` is sharpened in BT.709 luma and the detail
    is added to all three channels, leaving chroma unchanged. With
    ``luma_only=False`` each RGB channel is sharpened independently. Mono
    input is sharpened directly. Values are returned unclipped, so overshoot
    and undershoot survive.

    Raises
    ------
    ValueError
        If ``config`` is not a :class:`SharpenConfig`; if the image is not a
        non-empty 2-D or 3-D real numeric array with finite values and, for
        3-D, exactly three channels; if a config value is outside its schema
        domain (a mutated config can bypass pydantic validation); or if the
        result does not fit in float32. Each condition would otherwise produce
        silently wrong or non-finite pixels.

    Notes
    -----
    The gate is inclusive: samples with ``|highpass| <= threshold`` are set to
    zero exactly, and the comparison happens in the image's own float32
    precision. When the gate removes every sample the stage returns an exact
    copy of the input, skipping the opponent round trip, so a fully suppressed
    image is bit-identical to its input. The high-pass and the threshold are in
    linear units, not relative to the local mean, so the threshold does not
    scale with exposure; choose it from the frame's noise level.
    """
    if not isinstance(config, SharpenConfig):
        raise ValueError(f"config must be a SharpenConfig; got {type(config).__name__}")
    values = _as_float32_image(image, "image")
    amount = _validate_non_negative(config.amount, "config.amount")
    radius = _validate_radius(config.radius, "config.radius")
    threshold = _validate_non_negative(config.threshold, "config.threshold")
    luma_only = _validate_bool(config.luma_only, "config.luma_only")
    enabled = _validate_bool(config.enabled, "config.enabled")

    if not enabled or amount == 0.0:
        return values.copy()

    if values.ndim == 3 and luma_only:
        luma = rgb_to_opponent(values)[..., 0]
        detail = _scaled_detail(luma, radius, threshold, amount)
        if not np.any(detail):
            return values.copy()
        # Chroma (Cb, Cr) is untouched by construction: adding the same delta
        # to R, G, and B raises Y by delta and leaves both differences intact.
        sharpened = values.astype(np.float64) + detail.astype(np.float64)[..., np.newaxis]
    else:
        detail = _scaled_detail(values, radius, threshold, amount)
        sharpened = values.astype(np.float64) + detail.astype(np.float64)

    return _as_checked_float32(
        sharpened,
        "unsharp_mask produced non-finite values; the amount or input exceeds float32 range",
    )


def _scaled_detail(
    channel: NDArray[np.float32],
    radius: float,
    threshold: float,
    amount: float,
) -> NDArray[np.float32]:
    """Return ``amount`` times the threshold-gated high-pass of ``channel``.

    ``np.where`` implements the hard gate literally: the comparison is strict
    (``>``), so a sample exactly at the threshold is suppressed, and the gated
    residual is exactly zero rather than merely small. Gating in float32 keeps
    the boundary behaviour identical to what a caller computing the same
    ``channel - gaussian_blur(channel, radius)`` in float32 observes.
    """
    highpass = channel - gaussian_blur(channel, radius)
    gated = np.where(np.abs(highpass) > threshold, highpass, np.float32(0.0))
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = amount * np.asarray(gated, dtype=np.float64)
    return _as_checked_float32(
        scaled,
        "unsharp_mask produced non-finite values; the amount or input exceeds float32 range",
    )
