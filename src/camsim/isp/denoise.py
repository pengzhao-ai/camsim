"""Luma/chroma denoising in a linear opponent space.

Noise enters a camera as fluctuations of light, so it is additive in linear
light: photon shot noise has a standard deviation proportional to the square
root of the signal, read noise is roughly independent of it, and fixed-pattern
terms sit in the RAW data. Demosaicing interpolates two of every three samples,
and the colour-correction matrix mixes the channels, so by the time noise
reaches the denoise stage it is spatially correlated and spread across R, G,
and B. Denoising therefore runs after demosaicing (and the CCM) and before tone
mapping, in linear float32 light. Applying it to gamma-encoded data is a
failure mode: the filter then averages a non-linear function of radiance, so
the effective strength changes with scene brightness and the filter no longer
corresponds to any physical smoothing of photons.

Human vision resolves luminance detail far better than chroma detail, and most
demosaicing error is chromatic, so the stage works in the reversible BT.709
opponent space::

    Y  = 0.2126 R + 0.7152 G + 0.0722 B
    Cb = (B - Y) / 1.8556
    Cr = (R - Y) / 1.5748

    R = Y + 1.5748 Cr
    B = Y + 1.8556 Cb
    G = (Y - 0.2126 R - 0.0722 B) / 0.7152

``Y`` is luma (the BT.709 weights sum to one, so a neutral input has ``Y`` equal
to its level), ``Cb`` and ``Cr`` are blue-difference and red-difference chroma.
The divisors 1.8556 and 1.5748 are the reciprocals of the BT.709 chroma scale
factors: for unit RGB both chroma channels stay within roughly +/-0.5, a
different scale from luma, which is exactly why one strength cannot serve all
three: chroma can be smoothed much harder than luma before visible detail is
lost. The transform is linear, exactly invertible, and does not clip; negative
linear values from earlier stages round-trip unchanged.

Two filters are provided:

* ``gaussian`` -- a separable Gaussian blur with ``sigma = radius``, truncated
  at four sigma with symmetric reflection at the frame borders. It is the
  cheapest useful low-pass, and its fixed spatial kernel smooths detail as well
  as noise.
* ``bilateral`` -- an edge-preserving filter. Each neighbour is weighted by a
  spatial Gaussian of ``sigma = radius`` and a range Gaussian whose sigma is
  set per channel (floored at float32 epsilon):

  * luma: ``0.1 * percentile(|Y|, 99)``;
  * chroma: ``0.1 * max(percentile(|Y|, 99), percentile(|channel|, 99))``.

  Neighbours whose intensity differs by much more than a tenth of the reference
  amplitude receive almost no weight, so a plateau is averaged but an edge is
  not crossed. Chroma references the luma amplitude because Cb and Cr centre on
  zero: on a neutral scene a chroma channel's own percentile is its own noise,
  and a purely self-referential gate would preserve exactly the noise this
  stage exists to remove. Genuine colour edges, whose chroma differences are
  large relative to the luma scale, are still excluded. Every reference
  amplitude follows the signal scale, so the filter is scale equivariant:
  ``denoise(a * x) == a * denoise(x)`` for any real ``a``.

Both methods blend the filtered result with the original::

    out = (1 - s) * x + s * filter(x)

using ``luma_strength`` for Y and ``chroma_strength`` for Cb and Cr
independently. For ``0 <= s <= 1`` this is a convex combination: it never
clips, and it preserves non-negative input. ``s`` is not capped at one. Values
above one extrapolate past the filtered image, inverting the residual
high-frequency component, which produces sharpening-like overshoot and
amplifies noise; that is a documented escape hatch for deblurring-adjacent
experiments, not a normal setting. A strength of exactly zero skips the filter
for that channel and returns it unchanged, so a config with both strengths zero
is an exact identity.

Typical parameter ranges. After normalisation to linear ``[0, 1]`` float, a
well-exposed base-ISO frame has a noise sigma of roughly 0.001-0.01, while a
low-light or high-gain frame reaches 0.02-0.05; the optimal strength depends on
the scene, and over-smoothing produces the "plastic" look in which texture and
skin detail disappear. Blur radii of 0.5-3 px cover the useful range: below
half a pixel the filter is barely distinguishable from the identity, above
three pixels luma detail visibly smears. Luma strengths of 0.3-1.0 and chroma
strengths of 0.5-1.0 are typical, with chroma often filtered more aggressively
because colour noise is both uglier and less informative. The bilateral window
spans ``ceil(3 * radius)`` pixels in each direction, so its cost is
``O(k^2)`` per pixel compared with the Gaussian's separable ``O(k)``; it is the
method of choice only when edges must survive.

Mono sensors skip the opponent transform: 2-D ``(H, W)`` input is filtered
directly using ``luma_strength`` and ``chroma_strength`` is ignored, because
there is no chroma to treat independently. The pipeline keeps mono
single-channel through the whole ISP.

The chroma reference scale is a deliberate compromise, not a pure
self-reference. In a near-black scene the luma amplitude itself is tiny, so
chroma variation comparable to the residual luma level is treated as structure
and left alone; gating chroma on its own magnitude instead is strictly worse,
because on a neutral patch that magnitude is the noise itself.

Failure artefacts guarded or documented here: over-smoothing and the plastic
look (too large a radius or strength), chroma bleeding and loss of colour
detail (too strong a chroma filter), smoothing across edges (the Gaussian's
inherent behaviour, mitigated by the bilateral method), noise amplification
when ``s > 1``, filtering in a non-linear domain, mixing channels inside the
blur (which would create false colour), non-finite, boolean, complex, empty, or
wrongly shaped input, and a mutated config whose values no longer satisfy the
schema. No clipping is applied: negative values and overshoot survive to the
tone stage. Outputs are new C-contiguous float32 arrays of the input shape, and
the input is never mutated.
"""

from __future__ import annotations

import math
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

from camsim.config.schema import DenoiseConfig

__all__ = [
    "DenoiseConfig",
    "DenoiseMethod",
    "denoise",
    "gaussian_blur",
    "opponent_to_rgb",
    "rgb_to_opponent",
]

DenoiseMethod: TypeAlias = Literal["gaussian", "bilateral"]
"""Filter families accepted by :func:`denoise`."""

_METHODS: Final[tuple[str, ...]] = ("gaussian", "bilateral")

# BT.709 luma weights and the reversible chroma divisors. The weights sum to
# one so a neutral input keeps its level in Y; the divisors are chosen so that
# Cb and Cr span roughly [-0.5, 0.5] for in-gamut RGB.
_LUMA_RED: Final[float] = 0.2126
_LUMA_GREEN: Final[float] = 0.7152
_LUMA_BLUE: Final[float] = 0.0722
_CB_SCALE: Final[float] = 1.8556
_CR_SCALE: Final[float] = 1.5748

# Truncating the Gaussian at four sigma (scipy's default) keeps the kernel
# small while leaving a relative error below 1e-4; the same rule is used by the
# sharpen stage so both stages agree on what "radius" means.
_BLUR_TRUNCATE: Final[float] = 4.0

# The bilateral range sigma follows the 99th percentile of |channel|. The
# fraction sets the edge threshold: at 0.1, intensity differences below ~10% of
# the bright range are averaged while larger jumps are preserved. The epsilon
# floor keeps the divisor finite for an all-zero channel; it is the float32
# epsilon, so it only binds at magnitudes below the representation's own noise
# floor.
_RANGE_PERCENTILE: Final[float] = 99.0
_RANGE_SIGMA_FRACTION: Final[float] = 0.1
_RANGE_EPS: Final[float] = float(np.finfo(np.float32).eps)

# The bilateral window radius: three spatial sigmas, matching the Gaussian's
# effective support closely enough that the two methods smooth comparably.
_WINDOW_SIGMAS: Final[float] = 3.0


def rgb_to_opponent(rgb: ArrayLike) -> NDArray[np.float32]:
    """Convert linear RGB to the reversible BT.709 luma/chroma basis.

    Parameters
    ----------
    rgb:
        A real numeric ``(H, W, 3)`` array in linear R, G, B order. Values must
        be finite; negatives are accepted because earlier linear stages can
        leave excursions below zero. The input is never modified.

    Returns
    -------
    A new C-contiguous float32 ``(H, W, 3)`` array with channels ``(Y, Cb, Cr)``
    computed as

    ``Y = 0.2126 R + 0.7152 G + 0.0722 B``,
    ``Cb = (B - Y) / 1.8556``, and ``Cr = (R - Y) / 1.5748``.

    The transform is linear and invertible; :func:`opponent_to_rgb` recovers the
    input to float32 round-off (about 1e-6 absolute). ``Cb`` and ``Cr`` are on
    different scales from ``Y`` (their divisors differ), so a strength tuned
    for luma is not a strength tuned for chroma.

    Raises
    ------
    ValueError
        If ``rgb`` is not a finite real numeric ``(H, W, 3)`` array, or if the
        opponent values do not fit in float32.

    Notes
    -----
    Arithmetic is done in float64 from the float32 input and rounded once at
    the end, so the result is the correctly rounded float32 value of the double
    precision expression. Nothing is clipped.
    """
    values = _as_float32_image(rgb, "rgb")
    if values.ndim != 3:
        raise ValueError(f"rgb must be an (H, W, 3) linear RGB image; got shape {values.shape}")
    return _rgb_to_opponent(values)


def opponent_to_rgb(opponent: ArrayLike) -> NDArray[np.float32]:
    """Convert ``(Y, Cb, Cr)`` opponent values back to linear RGB.

    Parameters
    ----------
    opponent:
        A real numeric ``(H, W, 3)`` array whose channels are ``(Y, Cb, Cr)`` as
        produced by :func:`rgb_to_opponent`. Values must be finite; negatives
        are accepted. The input is never modified.

    Returns
    -------
    A new C-contiguous float32 ``(H, W, 3)`` array in linear R, G, B order,
    computed as

    ``R = Y + 1.5748 Cr``, ``B = Y + 1.8556 Cb``, and
    ``G = (Y - 0.2126 R - 0.0722 B) / 0.7152``.

    The inverse is exact in real arithmetic, so
    ``opponent_to_rgb(rgb_to_opponent(x))`` recovers ``x`` to float32 round-off.

    Raises
    ------
    ValueError
        If ``opponent`` is not a finite real numeric ``(H, W, 3)`` array, or if
        the resulting RGB values do not fit in float32.

    Notes
    -----
    No clipping is applied; out-of-gamut and negative values are preserved.
    Arithmetic is done in float64 and rounded once at the end.
    """
    values = _as_float32_image(opponent, "opponent")
    if values.ndim != 3:
        raise ValueError(
            f"opponent must be an (H, W, 3) luma/chroma image; got shape {values.shape}"
        )
    return _opponent_to_rgb(values)


def gaussian_blur(image: ArrayLike, sigma: float) -> NDArray[np.float32]:
    """Blur a linear image (or one channel) with a separable Gaussian.

    Parameters
    ----------
    image:
        A real numeric ``(H, W)`` or ``(H, W, 3)`` array. Values must be finite;
        negatives are accepted. The input is never modified.
    sigma:
        Gaussian standard deviation in pixels, a finite real number greater
        than or equal to zero. ``0`` is an exact identity. ``1`` is the usual
        starting point; 0.5-3 px covers the useful range for these ISP stages.

    Returns
    -------
    A new C-contiguous float32 array with the same shape as ``image``. For a
    3-D image the filter is applied with ``sigma = (sigma, sigma, 0)``, so the
    three channels are filtered independently and never mixed; for 2-D input a
    scalar sigma is used. Borders use symmetric reflection (``mode="reflect"``
    in ``scipy.ndimage``), so a constant image stays exactly constant, and the
    kernel is truncated at four sigma.

    Raises
    ------
    ValueError
        If ``image`` is not a finite real numeric ``(H, W)`` or ``(H, W, 3)``
        array, if ``sigma`` is not a finite real number at least zero (booleans
        are rejected), or if the result does not fit in float32.

    Notes
    -----
    This helper is shared with :mod:`camsim.isp.sharpen`, so the blur radius
    means the same thing (a Gaussian sigma) in both stages. Filtering is
    linear, so ``gaussian_blur(a * image) == a * gaussian_blur(image)`` and
    constants are preserved by the normalised kernel.
    """
    values = _as_float32_image(image, "image")
    width = _validate_sigma(sigma)
    if width == 0.0:
        return values.copy()

    # A zero sigma on the channel axis is scipy's "do not filter this axis"
    # marker; a tuple is required because a scalar would blur channels together
    # along every axis including the colour one.
    sigmas: float | tuple[float, float, float] = (width, width, 0.0) if values.ndim == 3 else width
    blurred = gaussian_filter(values, sigma=sigmas, mode="reflect", truncate=_BLUR_TRUNCATE)
    output = np.ascontiguousarray(blurred, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError(
            "gaussian_blur produced non-finite values; the input or sigma exceeds float32 range"
        )
    return output


def denoise(image: ArrayLike, config: DenoiseConfig) -> NDArray[np.float32]:
    """Denoise a linear image in the BT.709 luma/chroma space.

    Parameters
    ----------
    image:
        A real numeric ``(H, W)`` mono image or ``(H, W, 3)`` linear RGB image.
        Values must be finite; negatives are accepted because earlier linear
        stages can leave excursions below zero. The input is never modified.
    config:
        Validated :class:`~camsim.config.schema.DenoiseConfig`.
        ``method`` selects the Gaussian or bilateral filter; ``radius`` is the
        spatial sigma in pixels; ``luma_strength`` and ``chroma_strength`` are
        the blend fractions applied to ``Y`` and to both of ``Cb`` and ``Cr``.
        ``enabled=False`` returns an exact copy once the config and image have
        been validated.

    Returns
    -------
    A new C-contiguous float32 array with the same shape as ``image``. For RGB
    input the three channels are transformed to ``(Y, Cb, Cr)``, filtered with
    their respective strengths, and transformed back. For mono input the single
    channel is filtered with ``luma_strength`` and ``chroma_strength`` is
    ignored. Nothing is clipped: strengths above one may overshoot, and
    negative values are preserved.

    Raises
    ------
    ValueError
        If ``config`` is not a :class:`DenoiseConfig`; if the image is not a
        non-empty 2-D or 3-D real numeric array with finite values and, for
        3-D, exactly three channels; if a config value is outside its schema
        domain (a mutated config can bypass pydantic validation); or if the
        blended result does not fit in float32. Each condition would otherwise
        produce silently wrong or non-finite pixels.

    Notes
    -----
    For 3-D input each channel follows
    ``out = (1 - s) * channel + s * filter(channel)`` independently. Zero
    strengths return that channel unchanged (and both zero strengths return the
    input without any transform at all), so the disabled/no-op cases are exact
    identities rather than round-trip approximations. All filters run in linear
    light; the opponent transform is linear, so the stage preserves the
    linear-space invariant. A 2-D image is never expanded to three channels:
    the mono pipeline keeps its single channel through every ISP stage.
    """
    if not isinstance(config, DenoiseConfig):
        raise ValueError(f"config must be a DenoiseConfig; got {type(config).__name__}")
    values = _as_float32_image(image, "image")
    method = _validate_method(config.method)
    luma_strength = _validate_non_negative(config.luma_strength, "config.luma_strength")
    chroma_strength = _validate_non_negative(config.chroma_strength, "config.chroma_strength")
    radius = _validate_radius(config.radius, "config.radius")
    enabled = _validate_bool(config.enabled, "config.enabled")

    if not enabled:
        return values.copy()
    if values.ndim == 2:
        # A mono sensor has no chroma; luma_strength is the whole operation.
        if luma_strength == 0.0:
            return values.copy()
        return _filter_channel(values, method, radius, luma_strength)
    if luma_strength == 0.0 and chroma_strength == 0.0:
        # Skipping the opponent round trip keeps the documented identity exact.
        return values.copy()

    opponent = _rgb_to_opponent(values)
    # Chroma is gated against the luma scale as well as its own, so near-neutral
    # chroma (whose own percentile is just its noise) is actually smoothed.
    chroma_reference: float | None = None
    if method == "bilateral" and chroma_strength != 0.0:
        chroma_reference = _percentile_amplitude(opponent[..., 0])
    filtered = np.empty_like(opponent)
    filtered[..., 0] = _filter_channel(opponent[..., 0], method, radius, luma_strength)
    filtered[..., 1] = _filter_channel(
        opponent[..., 1], method, radius, chroma_strength, chroma_reference
    )
    filtered[..., 2] = _filter_channel(
        opponent[..., 2], method, radius, chroma_strength, chroma_reference
    )
    return _opponent_to_rgb(filtered)


def _rgb_to_opponent(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Apply the BT.709 opponent matrix to an already validated float32 image."""
    working = values.astype(np.float64)
    red = working[..., 0]
    green = working[..., 1]
    blue = working[..., 2]
    luma = _LUMA_RED * red + _LUMA_GREEN * green + _LUMA_BLUE * blue
    cb = (blue - luma) / _CB_SCALE
    cr = (red - luma) / _CR_SCALE
    return _as_checked_float32(
        np.stack((luma, cb, cr), axis=-1),
        "rgb_to_opponent produced non-finite values; the input exceeds float32 range",
    )


def _opponent_to_rgb(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Invert the opponent matrix on an already validated float32 image."""
    working = values.astype(np.float64)
    luma = working[..., 0]
    cb = working[..., 1]
    cr = working[..., 2]
    # Red and blue follow directly from their chroma definitions; green is then
    # fixed by the luma equation, whose weights sum to one.
    red = luma + _CR_SCALE * cr
    blue = luma + _CB_SCALE * cb
    green = (luma - _LUMA_RED * red - _LUMA_BLUE * blue) / _LUMA_GREEN
    return _as_checked_float32(
        np.stack((red, green, blue), axis=-1),
        "opponent_to_rgb produced non-finite values; the input exceeds float32 range",
    )


def _filter_channel(
    channel: NDArray[np.float32],
    method: DenoiseMethod,
    radius: float,
    strength: float,
    reference_amplitude: float | None = None,
) -> NDArray[np.float32]:
    """Filter one channel and blend it with the original at ``strength``.

    ``reference_amplitude`` is the additional scale reference used by the
    bilateral range gate; chroma channels pass the luma 99th percentile so a
    near-neutral chroma plane is still smoothed. It is ignored by the Gaussian
    method, whose kernel does not depend on intensity.
    """
    if strength == 0.0:
        # Do not even blur: a zero blend must be an exact identity.
        return np.array(channel, dtype=np.float32, copy=True)
    if method == "gaussian":
        filtered = gaussian_blur(channel, radius)
    else:
        filtered = _bilateral_filter(channel, radius, reference_amplitude)
    return _blend(channel, filtered, strength)


def _blend(
    original: NDArray[np.float32],
    filtered: NDArray[np.float32],
    strength: float,
) -> NDArray[np.float32]:
    """Return ``(1 - s) * original + s * filtered`` rounded to float32.

    The blend is evaluated in float64 because ``s`` is not capped: an extreme
    but finite strength could overflow float32 part-way through, and raising a
    clear error is better than returning infinities.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        blended = (1.0 - strength) * original.astype(np.float64) + strength * filtered.astype(
            np.float64
        )
    return _as_checked_float32(
        blended,
        "denoise produced non-finite values; the strength or input exceeds float32 range",
    )


def _percentile_amplitude(channel: NDArray[np.float32] | NDArray[np.float64]) -> float:
    """Return the 99th percentile of ``|channel|``, the bilateral range scale."""
    return float(np.percentile(np.abs(np.asarray(channel, dtype=np.float64)), _RANGE_PERCENTILE))


def _bilateral_filter(
    channel: NDArray[np.float32],
    radius: float,
    reference_amplitude: float | None = None,
) -> NDArray[np.float32]:
    """Filter one channel with a scale-adaptive bilateral kernel.

    The weight of neighbour ``j`` for centre ``i`` is the product of a spatial
    Gaussian ``exp(-||p_j - p_i||^2 / (2 radius^2))`` and a range Gaussian
    ``exp(-(x_j - x_i)^2 / (2 sigma_r^2))`` with
    ``sigma_r = 0.1 * max(percentile(|channel|, 99), reference_amplitude)`` and
    a float32-epsilon floor. Luma passes no reference and uses its own
    percentile; chroma passes the luma percentile, because a neutral chroma
    channel's own percentile is just its noise. Every reference amplitude
    scales with the signal, so the weights (and therefore the output) scale
    with it -- the filter is scale equivariant. The window spans
    ``ceil(3 * radius)`` pixels in each direction and borders are extended by
    symmetric reflection, the same rule as the Gaussian blur.
    """
    working = np.asarray(channel, dtype=np.float64)
    height, width = working.shape
    amplitude = _percentile_amplitude(working)
    if reference_amplitude is not None:
        amplitude = max(amplitude, reference_amplitude)
    sigma_range = _RANGE_SIGMA_FRACTION * max(amplitude, _RANGE_EPS)
    window = max(1, math.ceil(_WINDOW_SIGMAS * radius))

    offsets = np.arange(-window, window + 1, dtype=np.float64)
    squared_distance = offsets[:, np.newaxis] ** 2 + offsets[np.newaxis, :] ** 2
    spatial = np.exp(-squared_distance / (2.0 * radius * radius))
    padded = np.pad(working, window, mode="symmetric")

    numerator = np.zeros_like(working)
    denominator = np.zeros_like(working)
    range_denominator = 2.0 * sigma_range * sigma_range
    for row in range(-window, window + 1):
        row_slice = slice(window + row, window + row + height)
        for column in range(-window, window + 1):
            column_slice = slice(window + column, window + column + width)
            shifted = padded[row_slice, column_slice]
            difference = shifted - working
            weight = spatial[row + window, column + window] * np.exp(
                -(difference * difference) / range_denominator
            )
            numerator += weight * shifted
            denominator += weight

    # The centre tap has both weights equal to one, so the denominator is at
    # least one and the division can never see a zero.
    return _as_checked_float32(
        numerator / denominator,
        "denoise produced non-finite values; the radius or input exceeds float32 range",
    )


def _as_float32_image(image: Any, field: str) -> NDArray[np.float32]:
    """Return ``image`` as a finite 2-D or 3-D three-channel float32 array.

    The same contract is shared by the denoise and sharpen stages: 2-D mono or
    3-D RGB with exactly three channels, real numeric, finite, and non-empty.
    A C-contiguous float32 copy is returned, so callers may treat the result as
    private scratch space without aliasing the caller's buffer.
    """
    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a 2-D or 3-D numeric array; {exc}") from exc
    if array.ndim not in (2, 3):
        raise ValueError(
            f"{field} must be a 2-D (H, W) mono image or a 3-D (H, W, 3) RGB image; "
            f"got shape {array.shape}"
        )
    if array.ndim == 3 and array.shape[2] != 3:
        raise ValueError(
            f"{field} must have exactly three channels in R, G, B order; got shape {array.shape}"
        )
    if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{field} must contain at least one pixel")
    if (
        array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{field} must contain real numeric values; got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{field} values must be finite; NaN or infinity is not a linear intensity"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        # A real copy, not ascontiguousarray: a C-contiguous float32 input would
        # otherwise be returned as an alias despite the documented contract.
        converted = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.all(np.isfinite(converted)):
        raise ValueError(
            f"{field} values must fit in float32; conversion produced a non-finite value"
        )
    return converted


def _as_checked_float32(values: NDArray[np.float64], message: str) -> NDArray[np.float32]:
    """Return a C-contiguous float32 copy, rejecting non-finite values.

    Both the float64 intermediate and the float32 result are checked: a value
    can be finite in float64 yet overflow float32 when rounded, and either case
    would silently poison every downstream metric.
    """
    if not np.all(np.isfinite(values)):
        raise ValueError(message)
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.ascontiguousarray(values, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError(message)
    return output


def _validate_method(method: Any) -> DenoiseMethod:
    """Return a supported filter name, accepting only the exact string."""
    if not isinstance(method, str) or method not in _METHODS:
        allowed = ", ".join(_METHODS)
        raise ValueError(f"config.method must be one of {allowed}; got {method!r}")
    return cast(DenoiseMethod, method)


def _validate_non_negative(value: Any, field: str) -> float:
    """Return a finite real value at least zero (booleans are rejected).

    Pydantic validates the config on construction, but a model built with
    ``model_copy(update=...)`` or mutated afterwards can bypass that; every
    stage re-checks the values it consumes instead of trusting the type.
    """
    message = f"{field} must be a finite real number greater than or equal to 0; got {value!r}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(message)
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(message)
    return resolved


def _validate_radius(value: Any, field: str) -> float:
    """Return a finite positive pixel radius (booleans are rejected)."""
    message = f"{field} must be a finite real number greater than 0; got {value!r}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(message)
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(message)
    return resolved


def _validate_bool(value: Any, field: str) -> bool:
    """Return a strict bool, rejecting strings and numbers that merely coerce.

    Shared by both stages for ``enabled`` and ``luma_only``: a mutated config
    with, say, ``enabled="false"`` must raise rather than run the stage because
    a non-empty string is truthy.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a bool; got {value!r}")
    return bool(value)


def _validate_sigma(value: Any) -> float:
    """Return a finite non-negative Gaussian sigma (booleans are rejected)."""
    message = f"sigma must be a finite real number greater than or equal to 0; got {value!r}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(message)
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(message)
    return resolved
