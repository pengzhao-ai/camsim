"""Final tone mapping: exposure, output colour space, and sRGB OETF.

Everything upstream of this module is linear in radiance. Human vision and
display hardware are not: a display's code value 128 is not half the light of
255, and a scene's dark tones need far more code values than the bright ones
to avoid visible banding. The last ISP stage therefore applies the display
transfer function, optionally after an exposure trim and a change of output
colour space::

    linear_srgb --(x 2**ev)--> exposed --(SRGB_TO_P3?)--> target linear RGB
                --(clamp to [0, 1])--> OETF --> display-referred code values

The standard curve is the sRGB opto-electronic transfer function (OETF) from
IEC 61966-2-1::

    OETF(x) = 12.92 x                          for x <= 0.0031308
              1.055 x**(1/2.4) - 0.055         for x >  0.0031308

and its exact inverse, the electro-optical transfer function (EOTF)::

    EOTF(v) = v / 12.92                        for v <= 0.04045
              ((v + 0.055) / 1.055)**2.4       for v >  0.04045

The linear toe exists so the curve is invertible and well conditioned near
black: a pure power curve has an infinite derivative at zero, which amplifies
sensor noise in shadows into banding and makes exact round trips
ill-conditioned. The power segment approximates the eye's brightness response
over most of the range. The two branches meet at the published break points
(``OETF(0.0031308) = 0.04045`` to display precision) and the whole function is
strictly increasing. Display P3 uses exactly the same transfer curve as sRGB;
only the primaries differ.

The transfer functions here are pure pointwise maps. Their intended domain is
the display range ``[0, 1]``, but they accept any finite real value and apply
the defining formulas without clamping: negatives map through the linear
branch (``OETF(-0.1) = -1.292``), and values above one remain above one. That
keeps them usable for round trips and for analysing out-of-range data. The
image-level :func:`apply_tone` is where the pipeline's final clamp lives.

Exposure compensation is a gain of ``2**ev`` applied in *linear* light, and
the target-space matrix -- the identity for sRGB, ``SRGB_TO_P3`` for Display P3
-- is also applied before the curve. The order matters. Encoding first and
matrixing second mixes three different non-linear functions of radiance and
shifts hues; applying the exposure gain in encoded space does not correspond
to any change of scene illumination. Typical trims are +/-3 EV, and the gain
is applied before the clamp, so a positive trim deliberately lets more
highlights reach white. After the matrix the values are clamped to ``[0, 1]``
and then encoded. The clamp is the pipeline's final clip: highlight overshoot
from sharpening becomes pure white, negative excursions from colour correction
become black, and every downstream metric sees a display-referred image.

A ``gamma`` override replaces the sRGB OETF with the plain power curve
``x**(1/gamma)``, applied after the same exposure, matrix, and clamp. The
useful range is roughly 1.8-2.4. It is not equivalent to sRGB at the same
nominal exponent: without the linear toe it crushes shadows, which is why
sRGB replaced the earlier pure-2.2 practice. ``gamma=None`` selects sRGB.

Output is float32 in ``[0, 1]`` with the input shape: ``(H, W, 3)`` RGB stays
RGB and 2-D mono stays single-channel. Mono input skips the colour matrix --
there is no colour space to convert -- but still receives exposure and the
transfer curve. ``enabled=False`` returns an exact copy without gain, clamp,
or encoding, which is the escape hatch for linear-domain analysis.

Failure artefacts guarded or documented here: clipping highlights (deliberate,
after exposure), hue shifts from encoding before matrixing, crushed or
banded shadows from a gamma curve without a linear toe, exposure applied to
encoded values, saturation loss when a wider input gamut is squeezed into a
smaller output gamut (out-of-gamut components reach negative linear values
and are clipped to black here), a non-finite exposure gain or product, and
silently accepting non-finite, boolean, complex, empty, or wrongly shaped
images or a mutated config whose values no longer satisfy the schema. Outputs
are new C-contiguous float32 arrays; inputs are never mutated.
"""

from __future__ import annotations

import math
from typing import Any, Final, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.config.schema import ToneConfig
from camsim.core.frames import ColorSpace
from camsim.isp.colorspace import SRGB_TO_P3
from camsim.isp.denoise import _as_checked_float32, _as_float32_image, _validate_bool

__all__ = [
    "ToneConfig",
    "apply_tone",
    "linear_to_p3",
    "linear_to_srgb",
    "srgb_to_linear",
]

# sRGB transfer-function constants from IEC 61966-2-1. The thresholds are
# expressed separately for each direction because they are the published
# break points, not exact rounding of one another: the two branches meet at
# those values to better than one 8-bit code.
_OETF_THRESHOLD: Final[float] = 0.0031308
_EOTF_THRESHOLD: Final[float] = 0.04045
_LINEAR_SLOPE: Final[float] = 12.92
_POWER_GAIN: Final[float] = 1.055
_POWER_OFFSET: Final[float] = 0.055
_DISPLAY_GAMMA: Final[float] = 2.4

_COLOR_SPACES: Final[tuple[str, ...]] = ("sRGB", "DisplayP3")


def linear_to_srgb(linear_rgb: ArrayLike) -> NDArray[np.float32]:
    """Encode linear values with the exact sRGB OETF.

    Parameters
    ----------
    linear_rgb:
        Any real numeric array-like with finite values. The intended domain is
        linear ``[0, 1]``, but negatives and values above one are accepted:
        they map through the linear branch or the power branch respectively
        and are not clamped. The input is never modified.

    Returns
    -------
    A new C-contiguous float32 array with the same shape as ``linear_rgb``,
    computed as ``12.92 x`` for ``x <= 0.0031308`` and
    ``1.055 x**(1/2.4) - 0.055`` otherwise. ``OETF(0) = 0`` and
    ``OETF(1) = 1``; the curve is strictly increasing and exactly invertible
    by :func:`srgb_to_linear` in real arithmetic.

    Raises
    ------
    ValueError
        If the input contains bool, complex, or non-finite values, or if the
        result cannot be represented in float32. A non-finite code value
        would poison every downstream metric.

    Notes
    -----
    The map is applied pointwise in float64 and rounded to float32 once. It is
    the standard sRGB encoding, identical to ``colour.cctf_encoding(x,
    function="sRGB")``. Display P3 uses this same curve.
    """
    values = _as_float64_values(linear_rgb, "linear_rgb")
    return _as_checked_float32(
        _srgb_oetf(values),
        "linear_to_srgb produced non-finite values; the input exceeds float32 range",
    )


def srgb_to_linear(display_rgb: ArrayLike) -> NDArray[np.float32]:
    """Decode display-encoded sRGB values with the exact sRGB EOTF.

    Parameters
    ----------
    display_rgb:
        Any real numeric array-like with finite values, normally the output
        of :func:`linear_to_srgb`. Negatives and values above one are accepted
        and are not clamped. The input is never modified.

    Returns
    -------
    A new C-contiguous float32 array with the same shape as ``display_rgb``,
    computed as ``v / 12.92`` for ``v <= 0.04045`` and
    ``((v + 0.055) / 1.055)**2.4`` otherwise. It is the exact inverse of
    :func:`linear_to_srgb` up to float32 round-off (about 1e-7 absolute over
    ``[0, 1]``).

    Raises
    ------
    ValueError
        If the input contains bool, complex, or non-finite values, or if the
        result cannot be represented in float32.

    Notes
    -----
    The map is applied pointwise in float64 and rounded once. Values above
    one are valid inputs to the decoding formula even though a physical
    display cannot produce them; they are useful when reversing an out-of-range
    encode.
    """
    values = _as_float64_values(display_rgb, "display_rgb")
    return _as_checked_float32(
        _srgb_eotf(values),
        "srgb_to_linear produced non-finite values; the input exceeds float32 range",
    )


def linear_to_p3(linear_rgb: ArrayLike) -> NDArray[np.float32]:
    """Convert linear sRGB to *display-encoded* Display P3.

    Parameters
    ----------
    linear_rgb:
        Any real numeric array-like whose trailing axis is 3 in R, G, B order:
        a ``(3,)`` colour, ``(N, 3)`` samples, an ``(H, W, 3)`` image, or a
        larger leading shape. Values must be finite; negatives and values
        above one are accepted. The input is never modified.

    Returns
    -------
    A new C-contiguous float32 array with the same shape as ``linear_rgb``.
    The conversion happens in two steps: linear sRGB is multiplied by
    ``SRGB_TO_P3`` and the result is encoded with the sRGB OETF. The output is
    therefore **display-referred Display P3** -- code values ready for a
    P3-tagged container -- not linear P3. For the pure-linear conversion use
    :func:`camsim.isp.colorspace.linear_srgb_to_p3`.

    Raises
    ------
    ValueError
        If the input does not have a trailing axis of 3, contains bool,
        complex, or non-finite values, or if the result cannot be represented
        in float32.

    Notes
    -----
    Display P3 shares the sRGB transfer curve, so only the matrix differs from
    :func:`linear_to_srgb`. The matrix is applied before the curve: doing it
    the other way round would mix three non-linear transfer functions and
    shift hue. Nothing is clamped; values outside ``[0, 1]`` after encoding
    remain outside.
    """
    values = _as_float64_tristimulus(linear_rgb, "linear_rgb")
    converted = values @ SRGB_TO_P3.T
    if not np.all(np.isfinite(converted)):
        raise ValueError("linear_to_p3 produced non-finite values; the input exceeds float32 range")
    return _as_checked_float32(
        _srgb_oetf(converted),
        "linear_to_p3 produced non-finite values; the input exceeds float32 range",
    )


def apply_tone(linear_rgb: ArrayLike, config: ToneConfig) -> NDArray[np.float32]:
    """Tone-map a linear image into display-referred output.

    Parameters
    ----------
    linear_rgb:
        A real numeric ``(H, W)`` mono image or ``(H, W, 3)`` linear sRGB
        image. Values must be finite; negatives are accepted because earlier
        linear stages can leave excursions below zero. The image is
        interpreted as linear sRGB regardless of the target space. The input
        is never modified.
    config:
        Validated :class:`~camsim.config.schema.ToneConfig`.
        ``exposure_compensation_ev`` is a gain of ``2**ev`` applied in linear
        light; ``color_space`` selects sRGB (identity) or Display P3
        (``SRGB_TO_P3``), applied before the curve; ``gamma=None`` selects the
        sRGB OETF and a positive value replaces it with ``x**(1/gamma)``.
        ``enabled=False`` returns an exact copy once the config and image have
        been validated.

    Returns
    -------
    A new C-contiguous float32 array with the same shape as ``linear_rgb``,
    in ``[0, 1]``. The processing order is: exposure gain, target-space
    matrix (3-channel input only), clamp to ``[0, 1]``, then the transfer
    function. Highlight overshoot becomes exactly 1.0 and negative values
    become exactly 0.0. Mono input stays 2-D and skips the colour matrix but
    still receives exposure and the curve.

    Raises
    ------
    ValueError
        If ``config`` is not a :class:`ToneConfig`; if the image is not a
        non-empty 2-D or 3-D real numeric array with finite values and, for
        3-D, exactly three channels; if a config value is outside its schema
        domain (a mutated config can bypass pydantic validation); if the
        exposure gain or the exposed image overflows; or if the result cannot
        be represented in float32.

    Notes
    -----
    Exposure is applied before the curve, and the colour matrix before the
    curve, because both are linear operations that must happen in the linear
    domain. The clamp after the matrix is the pipeline's final clip: it is
    what makes the output display-referred and guarantees the documented
    range. A ``gamma`` override applies to sRGB and Display P3 outputs alike;
    it is the plain power curve with no linear toe, so shadows are darker
    than the sRGB curve at the same nominal exponent.
    """
    if not isinstance(config, ToneConfig):
        raise ValueError(f"config must be a ToneConfig; got {type(config).__name__}")
    values = _as_float32_image(linear_rgb, "linear_rgb")
    enabled = _validate_bool(config.enabled, "config.enabled")
    color_space = _validate_color_space(config.color_space)
    exposure_ev = _validate_finite(
        config.exposure_compensation_ev, "config.exposure_compensation_ev"
    )
    gamma = _validate_gamma(config.gamma)

    if not enabled:
        return values.copy()

    scale = _exposure_scale(exposure_ev)
    with np.errstate(over="ignore", invalid="ignore"):
        working = values.astype(np.float64) * scale
    if not np.all(np.isfinite(working)):
        raise ValueError(
            "apply_tone exposure compensation produced non-finite values; "
            "exposure_compensation_ev is too large for this image"
        )

    if values.ndim == 3 and color_space == "DisplayP3":
        # The matrix mixes linear light; the curve must come afterwards.
        with np.errstate(over="ignore", invalid="ignore"):
            working = working @ SRGB_TO_P3.T
        if not np.all(np.isfinite(working)):
            raise ValueError(
                "apply_tone colour conversion produced non-finite values; "
                "the exposed image exceeds float64 range"
            )

    clipped = np.clip(working, 0.0, 1.0)
    # ``clipped`` is non-negative, so the plain power curve is real-valued.
    tone_mapped = _srgb_oetf(clipped) if gamma is None else np.power(clipped, 1.0 / gamma)

    return _as_checked_float32(
        tone_mapped,
        "apply_tone produced non-finite values; the input or config exceeds float32 range",
    )


def _srgb_oetf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the exact sRGB encoding formula to a float64 array.

    ``np.where`` evaluates both branches, so the power branch briefly sees
    negative values; the resulting NaN is discarded by the selection and the
    invalid-operation warning is suppressed locally rather than leaking to a
    caller who deliberately passes negative linear values.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        return np.where(
            values <= _OETF_THRESHOLD,
            _LINEAR_SLOPE * values,
            _POWER_GAIN * np.power(values, 1.0 / _DISPLAY_GAMMA) - _POWER_OFFSET,
        )


def _srgb_eotf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the exact sRGB decoding formula to a float64 array."""
    with np.errstate(invalid="ignore", over="ignore"):
        return np.where(
            values <= _EOTF_THRESHOLD,
            values / _LINEAR_SLOPE,
            np.power((values + _POWER_OFFSET) / _POWER_GAIN, _DISPLAY_GAMMA),
        )


def _as_float64_values(values: Any, field: str) -> NDArray[np.float64]:
    """Return a finite real float64 array of any shape (including 0-d).

    The transfer functions are pointwise mathematical maps, so any shape is
    valid: scalars, samples, images, or empty arrays. Only the value type and
    finiteness are constrained.
    """
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a real numeric array; {exc}") from exc
    if (
        array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{field} must contain real numeric values; got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{field} values must be finite; NaN or infinity has no transfer-function value"
        )
    return np.asarray(array, dtype=np.float64)


def _as_float64_tristimulus(values: Any, field: str) -> NDArray[np.float64]:
    """Return a finite real float64 array whose trailing axis is 3."""
    array = _as_float64_values(values, field)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(
            f"{field} must be an array with a trailing axis of 3 (R, G, B); got shape {array.shape}"
        )
    return array


def _validate_color_space(value: Any) -> ColorSpace:
    """Return a supported output colour-space name, accepting only the exact strings."""
    if not isinstance(value, str) or value not in _COLOR_SPACES:
        allowed = " and ".join(_COLOR_SPACES)
        raise ValueError(f"config.color_space must be {allowed}; got {value!r}")
    return cast(ColorSpace, value)


def _validate_finite(value: Any, field: str) -> float:
    """Return a finite real value, rejecting booleans and coercible strings."""
    message = f"{field} must be a finite real number; got {value!r}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(message)
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(message)
    return resolved


def _validate_gamma(value: Any) -> float | None:
    """Return ``None`` or a finite positive gamma, rejecting booleans."""
    if value is None:
        return None
    message = f"config.gamma must be None or a finite real number greater than 0; got {value!r}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(message)
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(message)
    return resolved


def _exposure_scale(exposure_ev: float) -> float:
    """Return ``2**exposure_ev``, rejecting a non-finite gain.

    A gain that overflows to infinity would turn every pixel into NaN once it
    met a finite value; raising names the offending parameter instead.
    """
    try:
        scale = float(2.0**exposure_ev)
    except OverflowError as exc:
        raise ValueError(
            f"config.exposure_compensation_ev={exposure_ev!r} is too large; "
            "the 2**ev exposure gain overflows float"
        ) from exc
    if not math.isfinite(scale):
        raise ValueError(
            f"config.exposure_compensation_ev={exposure_ev!r} is too large; "
            "the 2**ev exposure gain is not finite"
        )
    return scale
