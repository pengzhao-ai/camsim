"""Lens-shading correction: radial gain or a measured flat-field gain map.

Lens shading is the spatial falloff of image-plane illuminance. For a thin lens
the oblique ray bundle is foreshortened and farther from the exit pupil, giving
the natural ``cos**4(theta)`` law; pupil aberration, microlens crosstalk, and
mechanical vignetting usually make a real lens fall off faster still. At the
corner of a wide phone lens only 30-50 % of the axial irradiance may remain,
which is one to two stops of shading. Because the falloff multiplies the scene,
it is corrected in the linear RAW domain by multiplying each sample by a
position-dependent gain

``g(y, x) = F_ref / F(y, x)``

where ``F`` is a flat-field response measured through the same lens, CFA, and
sensor. The gain map is an illumination ratio, not a colour-balance operation:
AWB must still estimate the illuminant afterwards, and demosaicing and the
colour matrix must see a uniformly illuminated mosaic so their spatial
assumptions hold.

Two gain representations are supported. ``mode="radial"`` evaluates an even
polynomial in normalised radius,

``g(r) = c0 + c1 r^2 + c2 r^4 + ...``

with ``r`` equal to one at the farthest image corner from ``center``. The
``center`` is given as ``(row, column)`` pixel coordinates and defaults to the
geometric image centre; an even-sized image has its centre between pixels, so
the default half-pixel coordinate is deliberate. ``mode="flat_field"`` applies
a supplied positive 2-D gain map of the same shape as the image, or a 2x2
per-CFA-site tile (which requires the CFA pattern so the caller states the
layout explicitly). The radial model is convenient for analytic studies and the
flat-field map is the accurate choice for a real part, because it captures
non-radial microlens and mechanical effects the polynomial cannot.

:func:`calibrate_lsc` turns a flat-field frame into a gain map by dividing the
brightest black-level-corrected sample by every pixel. Normalising to the
brightest sample keeps every gain at or above one and avoids the ill-defined
"centre" of an even-sized array. A pedestal left in the flat field would
compress the measured ratios and bias the map toward one, so a
:class:`~camsim.core.frames.RawFrame` is black-level corrected automatically;
an array input is assumed to be pedestal-free. Known defect sites and hot pixels
should be corrected before calibration because a single bright outlier can set
the reference.

Real-world corner falloff spans a few per cent for a long telephoto to more than
two stops for an ultra-wide phone lens; typical radial fits use ``c0`` near one
and ``c1`` between about 0.1 and 0.6, with higher-order terms for strong
falloff. All gains are dimensionless and the stage returns float32 linear
values; no gamma or tone curve is applied, preserving the linear-space
invariant.

Failure artefacts guarded here: applying a correction twice (over-bright
corners), calibrating on display-encoded data (the reciprocal of a non-linear
curve is not a gain), leaving the ADC pedestal in the flat field, a map whose
shape does not match the image (silent cropping or broadcasting), a
non-positive or non-finite gain (sign flips or infinities), using the wrong
radial centre or coefficient order (a tilt-shaped correction), and applying a
per-CFA tile without declaring the CFA layout.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.config.schema import LSCConfig
from camsim.core.frames import CFAPattern, RawFrame
from camsim.isp.blc import subtract_black_level

__all__ = ["LSCConfig", "apply_lsc", "calibrate_lsc"]

_MODES = ("none", "radial", "flat_field")


def apply_lsc(
    image: ArrayLike,
    config: LSCConfig,
    cfa: CFAPattern | str | None = None,
) -> NDArray[np.float32]:
    """Apply a lens-shading gain to a linear RAW image.

    Parameters
    ----------
    image:
        Linear RAW codes or intensities, a 2-D ``(H, W)`` array. Integer dtypes
        such as ``uint16`` and floating arrays from earlier stages are both
        accepted; negative values are allowed because a later linear stage can
        produce small negative excursions. Every value must be finite. The
        input is never modified.
    config:
        Validated :class:`~camsim.config.schema.LSCConfig`. ``enabled=False`` or
        ``mode="none"`` makes the stage an exact float32 pass-through.
        ``mode="radial"`` uses ``coefficients`` and optional ``center``;
        ``mode="flat_field"`` uses ``map``.
    cfa:
        Optional CFA pattern. It is required when ``map`` is a 2x2 per-site
        tile, so the caller explicitly declares the colour layout; the tile is
        positional (``map[0, 0]`` applies to pixel ``(0, 0)`` and every
        same-parity site), so the two green phases may carry different gains.
        For full-resolution maps ``cfa`` is validated but does not change the
        result, because those maps already carry any CFA-dependent shading.

    Returns
    -------
    A new float32 array of shape ``(H, W)`` equal to ``image * gain``. The gain
    is dimensionless, so scaling the input scales the output (linear-space
    invariant) and no clipping or gamma is applied.

    Raises
    ------
    ValueError
        If the image is not a non-empty 2-D real numeric array or contains
        non-finite values, if ``config`` is not an ``LSCConfig``, if a radial
        coefficient is non-finite or produces a non-positive gain, if a
        flat-field map has the wrong shape or is non-positive/non-finite, or if
        a 2x2 map is used without a valid CFA pattern.

    Notes
    -----
    The radial polynomial is evaluated in powers of ``r**2`` with ``r``
    normalised to the farthest corner, so the highest coefficient corresponds
    to the highest even power. A gain that reaches zero or changes sign anywhere
    is rejected: a "correction" that inverts a region is a calibration error,
    not a strong lens.
    """
    values = _as_float32_image(image, "image")
    if not isinstance(config, LSCConfig):
        raise ValueError(f"config must be an LSCConfig; got {type(config).__name__}")
    pattern = _coerce_cfa(cfa)
    if not config.enabled or config.mode == "none":
        return values
    if config.mode == "radial":
        gain = _radial_gain(values.shape, config)
    elif config.mode == "flat_field":
        gain = _flat_field_gain(values.shape, config, pattern)
    else:
        raise ValueError(f"mode must be one of {', '.join(_MODES)}; got {config.mode!r}")
    return values * gain


def calibrate_lsc(flat_field_raw: RawFrame | ArrayLike) -> NDArray[np.float32]:
    """Derive a lens-shading gain map from a flat-field capture.

    Parameters
    ----------
    flat_field_raw:
        A uniformly illuminated capture of the lens and sensor. A
        :class:`~camsim.core.frames.RawFrame` is black-level corrected
        automatically, because its ADC pedestal would otherwise compress the
        measured ratios. A plain 2-D array is assumed to be already
        pedestal-free and defect-corrected; all values must be finite and
        strictly positive.

    Returns
    -------
    A float32 ``(H, W)`` gain map ``max(flat) / flat`` whose largest entry is
    one at the brightest sample and whose remaining entries are at least one.
    It can be passed directly to :func:`apply_lsc` as
    ``LSCConfig(mode="flat_field", map=gain)``.

    Raises
    ------
    ValueError
        If the input is not a valid 2-D real array or ``RawFrame``, contains
        non-finite values, or contains a sample at or below zero after
        black-level subtraction. A zero would make the gain infinite and a
        negative sample would invert the correction, so both are calibration
        failures rather than values to clamp.

    Notes
    -----
    Normalising to the maximum rather than the centre avoids the ill-defined
    centre sample of an even-sized array and sidesteps a single noisy central
    pixel. It also guarantees gains at or above one. Known dead, hot, or stuck
    sites should still be corrected before calibration, because a hot pixel can
    be the maximum and set an unrealistically low gain everywhere else.
    """
    if isinstance(flat_field_raw, RawFrame):
        flat = subtract_black_level(
            flat_field_raw.data,
            flat_field_raw.black_level_dn,
            bit_depth=flat_field_raw.bit_depth,
        )
    else:
        flat = _as_float32_image(flat_field_raw, "flat_field_raw")
    flat64 = flat.astype(np.float64)
    peak = float(flat64.max())
    if peak <= 0.0:
        raise ValueError(
            "flat field must contain a positive sample; a dark frame cannot "
            "define a lens-shading gain"
        )
    if np.any(flat64 <= 0.0):
        raise ValueError(
            "flat field values must be strictly positive after black-level "
            "subtraction; a zero or negative sample makes the gain non-physical"
        )
    gain = peak / flat64
    if not np.all(np.isfinite(gain)):
        raise ValueError("flat field produced a non-finite lens-shading gain")
    return cast(NDArray[np.float32], gain.astype(np.float32))


def _as_float32_image(image: Any, field: str) -> NDArray[np.float32]:
    """Return ``image`` as a finite 2-D float32 array without aliasing it."""
    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a 2-D numeric array; {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"{field} must be a 2-D (H, W) array; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{field} must contain at least one pixel")
    if (
        array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{field} must contain real numeric values; got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} values must be finite; NaN or infinity is not a pixel")
    converted = np.ascontiguousarray(array, dtype=np.float32)
    if not np.all(np.isfinite(converted)):
        raise ValueError(
            f"{field} values must fit in float32; conversion produced a non-finite value"
        )
    return converted


def _coerce_cfa(cfa: CFAPattern | str | None) -> CFAPattern | None:
    """Return ``cfa`` as a :class:`CFAPattern`, or ``None`` when not supplied."""
    if cfa is None:
        return None
    if isinstance(cfa, CFAPattern):
        return cfa
    if isinstance(cfa, str):
        try:
            return CFAPattern(cfa)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in CFAPattern)
            raise ValueError(f"cfa must be one of {allowed}; got {cfa!r}") from exc
    raise ValueError(
        f"cfa must be a CFAPattern, its string value, or None; got {type(cfa).__name__}"
    )


def _radial_gain(shape: tuple[int, int], config: LSCConfig) -> NDArray[np.float32]:
    """Evaluate the radial polynomial gain over an ``(H, W)`` grid."""
    coefficients = config.coefficients
    if not coefficients:
        raise ValueError("LSC radial mode requires at least one coefficient")
    coeffs = [float(coefficient) for coefficient in coefficients]
    if not all(math.isfinite(coefficient) for coefficient in coeffs):
        raise ValueError("LSC radial coefficients must be finite")

    height, width = shape
    if config.center is None:
        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0
    else:
        try:
            center_y = float(config.center[0])
            center_x = float(config.center[1])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"LSC center must be a (row, column) pair; got {config.center!r}"
            ) from exc
        if not (math.isfinite(center_y) and math.isfinite(center_x)):
            raise ValueError(f"LSC center must be finite; got {config.center!r}")

    corner_distances = (
        center_y**2 + center_x**2,
        center_y**2 + (width - 1 - center_x) ** 2,
        (height - 1 - center_y) ** 2 + center_x**2,
        (height - 1 - center_y) ** 2 + (width - 1 - center_x) ** 2,
    )
    normalizer_squared = max(corner_distances)
    rows = np.arange(height, dtype=np.float64)[:, None]
    columns = np.arange(width, dtype=np.float64)[None, :]
    if normalizer_squared > 0.0:
        radius_squared = ((rows - center_y) ** 2 + (columns - center_x) ** 2) / normalizer_squared
    else:
        radius_squared = np.zeros((height, width), dtype=np.float64)

    # Horner evaluation keeps the coefficient order c0 + c1*r^2 + c2*r^4 + ...
    gain = np.full(radius_squared.shape, coeffs[-1], dtype=np.float64)
    for coefficient in reversed(coeffs[:-1]):
        gain = gain * radius_squared + coefficient
    if not np.all(np.isfinite(gain)):
        raise ValueError("LSC radial gain is non-finite for the supplied coefficients")
    if np.any(gain <= 0.0):
        raise ValueError(
            "LSC radial gain must be positive at every pixel; the supplied "
            "coefficients produce a zero or negative gain"
        )
    return cast(NDArray[np.float32], gain.astype(np.float32))


def _flat_field_gain(
    shape: tuple[int, int],
    config: LSCConfig,
    cfa: CFAPattern | None,
) -> NDArray[np.float32]:
    """Validate and return a supplied flat-field gain map for ``shape``."""
    if config.map is None:
        raise ValueError("LSC flat-field mode requires a map")
    try:
        gain = np.asarray(config.map, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LSC map must be a 2-D array of real numbers; {exc}") from exc
    if gain.ndim != 2 or gain.size == 0:
        raise ValueError(f"LSC map must be a non-empty 2-D array; got shape {gain.shape}")
    if not np.all(np.isfinite(gain)):
        raise ValueError("LSC map values must be finite")
    if np.any(gain <= 0.0):
        raise ValueError(
            f"LSC map values must be positive; got {float(gain.min()):g}. A "
            "non-positive gain would invert or blank part of the image."
        )
    if gain.shape == shape:
        return cast(NDArray[np.float32], gain.astype(np.float32))
    if gain.shape == (2, 2):
        if cfa is None:
            raise ValueError(
                "a 2x2 LSC map is a per-CFA-site gain tile and requires cfa so "
                "the colour layout is explicit"
            )
        tiled = np.empty(shape, dtype=np.float32)
        tiled[0::2, 0::2] = np.float32(gain[0, 0])
        tiled[0::2, 1::2] = np.float32(gain[0, 1])
        tiled[1::2, 0::2] = np.float32(gain[1, 0])
        tiled[1::2, 1::2] = np.float32(gain[1, 1])
        return tiled
    raise ValueError(
        f"LSC map shape {gain.shape} must match the image shape {shape} or be a 2x2 CFA tile"
    )
