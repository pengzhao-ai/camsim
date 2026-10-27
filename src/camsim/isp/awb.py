"""Auto white balance: estimate and remove an illuminant colour cast.

A scene's spectral power distribution changes with the illuminant, but the
human visual system adapts and still perceives a white sheet as white. A camera
does not: its RAW channels integrate the scene radiance through three fixed
colour filters, so a low-colour-temperature (reddish) source raises the red
channel and lowers blue, while daylight does the opposite. The ISP compensates
with the von Kries diagonal model, multiplying each linear channel by one gain::

    balanced(y, x, c) = gain_c * image(y, x, c)

``gain_c = s_G / s_c`` equalises the per-channel statistic ``s`` so that a scene
that averages to grey does so in all three channels. Gains are normalised so
``gain_G == 1``; a common scale would change exposure, and white balance must
adjust colour, not brightness.

Three classical estimators are provided, each making a different assumption
about scene content:

* ``gray_world`` (Buchsbaum, 1980): the average reflectance of a natural scene
  is achromatic, so the channel means should be equal. It is cheap and works
  well on diverse scenes, but a frame dominated by one colour (a grass field, a
  red wall) is mistaken for an illuminant cast.
* ``white_patch`` (Land and McCann's Retinex): the brightest sample in each
  channel is a white surface. Each channel's maximum is used independently, so
  the white reference may come from different pixels. A scene with no white
  patch, or a saturated specular highlight that clips, biases the estimate.
* ``shades_of_gray`` (Finlayson and Trezzi, 2004): the Minkowski ``p``-norm of
  each channel should be equal, ``s_c = (mean(x_c**p))**(1/p)``. ``p=1`` is
  gray-world, ``p -> infinity`` is white-patch, and ``p`` around 4-8 is a robust
  compromise; the default is 6. Larger ``p`` weights the brightest samples more,
  so sensitivity to highlights increases.

Real-world white-balance gains are typically 0.5-4. Tungsten light near 2800 K
can require a red gain several times the blue gain, while shade near 8000 K
inverts that direction. Gain spread therefore spans roughly one stop for a
near-neutral source to more than two stops for extreme illuminants. The
confidence reported here is ``1 - (max_gain - min_gain) / (max_gain + min_gain)``
in ``[0, 1]``: it is one when the channels already agree and falls as the
required correction grows, because a large correction relies more heavily on the
scene assumption being true. Mono sensors and disabled configs report confidence
zero because no colour estimate exists.

A Bayer mosaic records one spectral band per site. Statistics are therefore
collected per CFA colour, using the same masks as the sensor layout: both green
phases share one G channel, and red and blue samples are never averaged
together. Mixing CFA colours would blend scene colour with illuminant colour and
bias every later demosaic and colour-correction step. For an already demosaiced
``(H, W, 3)`` RGB image the three channels are used directly and ``cfa`` must be
``None``. RCCC white balance (an R plus clear-channel sensor) is deferred and
rejected rather than guessed.

The ``illuminant_estimate`` is a simple documented chromaticity: the per-channel
statistic divided by the arithmetic mean of the three statistics, so the triple
averages to one. It is proportional to the reciprocal of the gains and describes
the illuminant in the sensor's R, G, B basis, not a calibrated CIE xy point.

All computation happens in linear float32 light after black-level correction.
No clipping is applied: gains can legitimately push a scene above the old
maximum, and the later tone stage owns the display range. Scaling the input
scales the output (``apply_awb(a * x) == a * apply_awb(x)``), and the estimators
are scale invariant because every statistic is homogeneous of degree one.
Applying white balance to gamma-encoded data is a failure mode, because the
gains then act on a non-linear function of radiance.

Failure artefacts guarded here: mixing CFA colours (colour-dependent cast),
using a demosaiced image with a mosaic layout (silently wrong channel mapping),
zero or negative channel statistics (infinite or sign-flipped gains), negative,
NaN, or infinite input (a poisoned estimate), gains that do not fit float32
(overflow on application), and zero-length or malformed images. A zero channel
statistic cannot be white balanced and raises rather than inventing a gain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.config.schema import AWBConfig
from camsim.core.frames import CFAPattern
from camsim.sensor.cfa import cfa_color_at, cfa_masks

__all__ = ["AWBConfig", "AWBMethod", "AWBResult", "apply_awb", "estimate_awb"]

AWBMethod: TypeAlias = Literal[
    "gray_world",
    "white_patch",
    "shades_of_gray",
    "manual",
    "none",
]
"""Methods an :class:`AWBResult` can report.

``"none"`` records a mono or disabled no-op, and ``"manual"`` records gains
supplied through :class:`~camsim.config.schema.AWBConfig`.
"""

# Canonical channel order shared by the gains tuple and every statistic.
_CHANNELS: Final[tuple[str, str, str]] = ("R", "G", "B")

_ESTIMATION_METHODS: Final[tuple[str, ...]] = (
    "gray_world",
    "white_patch",
    "shades_of_gray",
)

# Gains are multiplied into float32 images. Values outside float32's normal
# range would overflow to infinity or underflow to zero, so reject them at the
# boundary instead of silently producing a black or white frame.
_MAX_FLOAT32_GAIN: Final[float] = float(np.finfo(np.float32).max)
_MIN_FLOAT32_GAIN: Final[float] = float(np.finfo(np.float32).smallest_normal)


@dataclass(frozen=True)
class AWBResult:
    """Estimated white-balance parameters for one image.

    Parameters
    ----------
    gains:
        ``(R, G, B)`` multipliers in the sensor's channel order, normalised so
        the green gain is exactly ``1.0``. Multiply them into linear data with
        :func:`apply_awb`.
    illuminant_estimate:
        Simple illuminant chromaticity in the sensor's R, G, B basis: the
        per-channel statistic (or the reciprocal of manual gains) divided by
        the arithmetic mean of the three values, so the triple averages to one.
        This is not a calibrated CIE chromaticity.
    method:
        Estimator that produced the gains: one of the estimator names,
        ``"manual"`` for pinned gains, or ``"none"`` for a mono or disabled
        no-op.
    confidence:
        Bounded ``[0, 1]`` measure of gain dispersion. It is ``1.0`` when the
        channels already agree, decreases as the required correction grows,
        and is ``0.0`` for mono or disabled stages, where no colour estimate
        exists.

    The gains are frozen and in canonical R, G, B order; a caller cannot
    accidentally swap them between estimation and application.
    """

    gains: tuple[float, float, float]
    illuminant_estimate: tuple[float, float, float]
    method: AWBMethod
    confidence: float


def estimate_awb(
    raw_or_rgb: ArrayLike,
    cfa: CFAPattern | str | None,
    config: AWBConfig,
) -> AWBResult:
    """Estimate per-channel white-balance gains from linear image data.

    Parameters
    ----------
    raw_or_rgb:
        A 2-D ``(H, W)`` Bayer mosaic (one spectral band per site) or a 3-D
        ``(H, W, 3)`` demosaiced image in linear R, G, B order. Values must be
        finite, real, and non-negative; white balance belongs after black-level
        correction, where the signal is proportional to light. The input is
        never modified.
    cfa:
        The CFA pattern for a 2-D mosaic, as a
        :class:`~camsim.core.frames.CFAPattern` or its string value. It must be
        a Bayer layout or ``MONO``; ``RCCC`` is rejected because its colour
        balance is deferred. For a 3-D RGB image this must be ``None``, because
        the spatial mosaic has already been interpolated away.
    config:
        Validated :class:`~camsim.config.schema.AWBConfig`. ``enabled=False``
        returns the identity result. When ``gains`` is supplied it takes
        precedence over ``method`` and is recorded as ``"manual"``; the values
        are rescaled so green is one. Otherwise ``method`` selects gray-world,
        white-patch, or shades-of-gray, using ``p`` for the last.

    Returns
    -------
    An :class:`AWBResult` whose ``gains`` are positive and normalised to
    ``(g_R, 1, g_B)``. Mono and disabled configs return ``(1, 1, 1)`` with
    ``method="none"`` and ``confidence=0.0``.

    Raises
    ------
    ValueError
        If ``config`` is not an :class:`AWBConfig`, the image is not a non-empty
        2-D or 3-D real numeric array, contains non-finite or negative values,
        a required CFA is missing or conflicts with the image rank, the CFA is
        unsupported, a channel statistic is zero (an infinite gain), the method
        or ``p`` is invalid, or manual gains are not three positive finite
        values representable in float32.

    Notes
    -----
    Statistics are computed in float64 for a stable accumulation. The
    shades-of-gray norm is evaluated after dividing by the channel maximum, so
    large ``p`` cannot overflow even though ``x**p`` would. A 2-D image smaller
    than its CFA tile can leave a colour with no samples; that is rejected
    rather than balanced with a missing channel.
    """
    if not isinstance(config, AWBConfig):
        raise ValueError(f"config must be an AWBConfig; got {type(config).__name__}")
    values = _as_float32_image(raw_or_rgb, "raw_or_rgb")
    if np.any(values < 0.0):
        raise ValueError(
            "raw_or_rgb values must be non-negative linear intensities; "
            "white balance runs after black-level correction"
        )
    pattern = _resolve_pattern(values, cfa)

    if not config.enabled or pattern is CFAPattern.MONO:
        # A disabled stage and a sensor without colour filters have no colour
        # estimate to make. Disabling also ignores pinned gains deliberately.
        return _identity_result("none")

    if config.gains is not None:
        manual = _validate_gains(config.gains, "config.gains")
        gains = _validate_gains(_normalise_to_green(manual), "gains")
        return AWBResult(
            gains=gains,
            illuminant_estimate=_illuminant_from_gains(gains),
            method="manual",
            confidence=_gain_confidence(gains),
        )

    method = _validate_method(config.method)
    p = _validate_p(config.p)
    statistics = _channel_statistics(values, pattern, method, p)
    estimated = _validate_gains(_normalise_to_green(_gains_from_statistics(statistics)), "gains")
    return AWBResult(
        gains=estimated,
        illuminant_estimate=_illuminant_from_statistics(statistics),
        method=method,
        confidence=_gain_confidence(estimated),
    )


def apply_awb(
    image: ArrayLike,
    gains: ArrayLike,
    cfa: CFAPattern | str | None = None,
) -> NDArray[np.float32]:
    """Apply per-channel white-balance gains to a linear image.

    Parameters
    ----------
    image:
        A 2-D ``(H, W)`` Bayer or mono mosaic, or a 3-D ``(H, W, 3)`` RGB image
        in canonical channel order. Values must be finite and real; negative
        values are accepted because this is a pure multiplication and earlier
        linear stages may leave small negative excursions. The input is never
        modified.
    gains:
        Three positive finite multipliers in R, G, B order. They are applied
        exactly as given, so callers who need a fixed exposure should pass
        green-normalised gains (as :func:`estimate_awb` returns). A 2-D mosaic
        receives the gain for the CFA colour at each site; both green phases
        receive the green gain.
    cfa:
        The CFA pattern for a 2-D mosaic. It must be given for 2-D input and
        must be ``None`` for 3-D RGB input. ``MONO`` is a pass-through and
        ``RCCC`` is rejected because its colour mapping is deferred.

    Returns
    -------
    A new C-contiguous float32 array with the same shape as ``image``. Nothing
    is clipped: values above the previous maximum are preserved, and the
    operation is linear, so ``apply_awb(a * image) == a * apply_awb(image)``
    for a fixed gain triple.

    Raises
    ------
    ValueError
        If the image is not a non-empty 2-D or 3-D real numeric array, contains
        non-finite values, the CFA layout is missing or unsupported, the gains
        are not three positive finite float32-representable numbers, or a huge
        gain multiplies a large value beyond float32 range. Each condition
        would otherwise produce silently wrong or non-finite pixels.
    """
    values = _as_float32_image(image, "image")
    checked = _validate_gains(gains, "gains")
    pattern = _resolve_pattern(values, cfa)

    if pattern is CFAPattern.MONO:
        return np.array(values, dtype=np.float32, copy=True)

    if pattern is None:
        gain_vector = np.asarray(checked, dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            output = values * gain_vector
    else:
        tile = _cfa_gain_tile(pattern, checked)
        output = np.empty_like(values)
        with np.errstate(over="ignore", invalid="ignore"):
            output[0::2, 0::2] = values[0::2, 0::2] * tile[0, 0]
            output[0::2, 1::2] = values[0::2, 1::2] * tile[0, 1]
            output[1::2, 0::2] = values[1::2, 0::2] * tile[1, 0]
            output[1::2, 1::2] = values[1::2, 1::2] * tile[1, 1]

    if not np.all(np.isfinite(output)):
        raise ValueError(
            "apply_awb produced non-finite values; the gain or input exceeds float32 range"
        )
    return np.ascontiguousarray(output, dtype=np.float32)


def _as_float32_image(image: Any, field: str) -> NDArray[np.float32]:
    """Return ``image`` as a finite 2-D mosaic or 3-D RGB float32 array."""
    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a 2-D or 3-D numeric array; {exc}") from exc
    if array.ndim not in (2, 3):
        raise ValueError(
            f"{field} must be a 2-D (H, W) mosaic or a 3-D (H, W, 3) RGB array; "
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
    converted = np.ascontiguousarray(array, dtype=np.float32)
    if not np.all(np.isfinite(converted)):
        raise ValueError(
            f"{field} values must fit in float32; conversion produced a non-finite value"
        )
    return converted


def _resolve_pattern(values: NDArray[np.float32], cfa: Any) -> CFAPattern | None:
    """Return the CFA pattern for ``values``, or ``None`` for 3-D RGB input.

    Rank and layout must agree: a demosaiced RGB image has no mosaic geometry,
    and a 2-D image cannot be interpreted without a declared CFA.
    """
    if values.ndim == 3:
        if cfa is not None:
            raise ValueError(
                "cfa must be None for a 3-D RGB image; the mosaic has already been demosaiced"
            )
        return None
    if cfa is None:
        raise ValueError("cfa is required for a 2-D mosaic image")
    pattern = _coerce_pattern(cfa)
    if pattern is CFAPattern.RCCC:
        raise ValueError("RCCC white balance is not supported; RCCC-specific AWB is deferred")
    return pattern


def _coerce_pattern(cfa: Any) -> CFAPattern:
    """Return ``cfa`` as a :class:`CFAPattern`, accepting member strings."""
    if isinstance(cfa, CFAPattern):
        return cfa
    if isinstance(cfa, str):
        try:
            return CFAPattern(cfa)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in CFAPattern)
            raise ValueError(f"cfa pattern must be one of {allowed}; got {cfa!r}") from exc
    raise ValueError(
        f"cfa pattern must be a CFAPattern or its string value; got {type(cfa).__name__}"
    )


def _validate_gains(gains: Any, field: str) -> tuple[float, float, float]:
    """Return three positive finite gain values that fit in float32."""
    try:
        array = np.asarray(gains)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be three positive finite numbers; {exc}") from exc
    if array.shape != (3,):
        raise ValueError(
            f"{field} must contain exactly three values in R, G, B order; got shape {array.shape}"
        )
    if (
        array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{field} must contain real numbers; got dtype {array.dtype}")
    numeric = np.asarray(array, dtype=np.float64).reshape(3)
    values = (float(numeric[0]), float(numeric[1]), float(numeric[2]))
    for value in values:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field} must be positive and finite; got {value!r}")
        if not _MIN_FLOAT32_GAIN <= value <= _MAX_FLOAT32_GAIN:
            raise ValueError(
                f"{field} values must be representable in float32's normal range; got {value:g}"
            )
    return values


def _normalise_to_green(gains: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rescale a gain triple so green is exactly one, preserving its balance."""
    green = gains[1]
    return (gains[0] / green, 1.0, gains[2] / green)


def _gains_from_statistics(
    statistics: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return green-normalised gains that equalise the three statistics."""
    red, green, blue = statistics
    return (green / red, 1.0, green / blue)


def _illuminant_from_statistics(
    statistics: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the channel statistics normalised to an arithmetic mean of one."""
    mean = (statistics[0] + statistics[1] + statistics[2]) / 3.0
    return (
        statistics[0] / mean,
        statistics[1] / mean,
        statistics[2] / mean,
    )


def _illuminant_from_gains(
    gains: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the illuminant implied by manual gains, mean-normalised."""
    reciprocals = (1.0 / gains[0], 1.0 / gains[1], 1.0 / gains[2])
    mean = (reciprocals[0] + reciprocals[1] + reciprocals[2]) / 3.0
    return (
        reciprocals[0] / mean,
        reciprocals[1] / mean,
        reciprocals[2] / mean,
    )


def _gain_confidence(gains: tuple[float, float, float]) -> float:
    """Map gain dispersion to a bounded confidence in ``[0, 1]``.

    ``1 - (max - min) / (max + min)`` is scale invariant, equals one when all
    channels need the same gain, and approaches zero as the correction becomes
    extreme. It is deliberately pessimistic: a large correction is the case in
    which a wrong scene assumption hurts most.
    """
    lowest = min(gains)
    highest = max(gains)
    spread = (highest - lowest) / (highest + lowest)
    return float(min(1.0, max(0.0, 1.0 - spread)))


def _identity_result(method: AWBMethod) -> AWBResult:
    """Return the documented no-op result."""
    return AWBResult(
        gains=(1.0, 1.0, 1.0),
        illuminant_estimate=(1.0, 1.0, 1.0),
        method=method,
        confidence=0.0,
    )


def _validate_method(method: Any) -> AWBMethod:
    """Return a supported estimation method, accepting only its string name."""
    if not isinstance(method, str) or method not in _ESTIMATION_METHODS:
        allowed = ", ".join(_ESTIMATION_METHODS)
        raise ValueError(f"method must be one of {allowed}; got {method!r}")
    return cast(AWBMethod, method)


def _validate_p(p: Any) -> float:
    """Return the shades-of-gray Minkowski exponent, finite and at least one."""
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise ValueError(f"p must be a finite number greater than or equal to 1; got {p!r}")
    value = float(p)
    if not math.isfinite(value) or value < 1.0:
        raise ValueError(f"p must be a finite number greater than or equal to 1; got {p!r}")
    return value


def _channel_statistics(
    values: NDArray[np.float32],
    pattern: CFAPattern | None,
    method: str,
    p: float,
) -> tuple[float, float, float]:
    """Return one statistic per channel, never mixing CFA colours."""
    if pattern is None:
        red = values[..., 0]
        green = values[..., 1]
        blue = values[..., 2]
    else:
        masks = cfa_masks(pattern, values.shape)
        red = values[masks["R"]]
        green = values[masks["G"]]
        blue = values[masks["B"]]
    return (
        _statistic(red, "R", method, p),
        _statistic(green, "G", method, p),
        _statistic(blue, "B", method, p),
    )


def _statistic(
    samples: NDArray[np.float32],
    colour: str,
    method: str,
    p: float,
) -> float:
    """Return a positive finite channel statistic for one estimator."""
    if samples.size == 0:
        raise ValueError(f"channel {colour} has no samples; the image is smaller than its CFA tile")
    data = np.asarray(samples, dtype=np.float64)
    if method == "gray_world":
        value = float(data.mean())
    elif method == "white_patch":
        value = float(data.max())
    elif method == "shades_of_gray":
        value = _minkowski_mean(data, p)
    else:
        raise ValueError(f"method is not an estimator: {method!r}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"channel {colour} has a non-positive {method} statistic ({value!r}); "
            "white balance would require an infinite gain"
        )
    return value


def _minkowski_mean(data: NDArray[np.float64], p: float) -> float:
    """Return ``(mean(x**p))**(1/p)`` without overflowing for large ``p``.

    Dividing by the channel maximum keeps every base in ``[0, 1]``, so the
    power is bounded by one and the result is exactly the unscaled Minkowski
    mean. At least one scaled sample equals one, so the mean cannot underflow to
    zero unless every sample was zero.
    """
    peak = float(data.max())
    if peak <= 0.0:
        return 0.0
    scaled = data / peak
    mean_power = float(np.mean(scaled**p))
    return peak * math.pow(mean_power, 1.0 / p)


def _cfa_gain_tile(pattern: CFAPattern, gains: tuple[float, float, float]) -> NDArray[np.float32]:
    """Return the 2x2 positional gain tile for a Bayer pattern."""
    lookup = {"R": gains[0], "G": gains[1], "B": gains[2]}
    tile = np.empty((2, 2), dtype=np.float32)
    for row in range(2):
        for column in range(2):
            tile[row, column] = lookup[cfa_color_at(pattern, row, column)]
    return tile
