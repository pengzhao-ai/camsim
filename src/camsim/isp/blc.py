"""Black-level correction: remove the ADC pedestal and clamp at zero signal.

An analog-to-digital converter represents zero input with a pedestal of
``black_level_dn`` codes rather than code zero. The pedestal exists so that the
symmetric read-noise distribution around zero volts is recorded instead of
clipped by an unsigned converter. Typical values are 64 DN at 10 bits and
256 DN at 12 bits, with a few codes of part-to-part variation and, on some
sensors, a slightly different level for each CFA phase. The ISP must subtract
this offset before exposure gain, white balance, demosaicing, or colour
correction, because every later operation treats the code value as proportional
to light. The transfer function implemented here is

``signal(y, x) = max(DN(y, x) - black_level(y, x), 0)``

where ``black_level`` is a scalar or a 2x2 tile repeated over the CFA lattice in
row-major order. The clamp acknowledges that a read-noise excursion can push a
dark pixel below the pedestal; without it the negative excursion would propagate
into white balance and colour matrices.

The stage is affine, not purely multiplicative: subtracting a constant offset
scales only the signal above the pedestal. The linear-space invariant therefore
means that, away from the clamp, scaling the signal and the pedestal together
scales the output (``blc(a * (x - b) + b) = a * blc(x)``); no gamma or tone
curve is applied here. The result is float32 codes ready for the remaining
linear ISP stages.

Real-world parameter ranges: black levels 0-1023 DN at 10 bits and 0-4095 DN at
12 bits, usually within a few DN of nominal across the array. A 2x2 pattern is
appropriate when the datasheet quotes separate pedestals per CFA colour.

Failure artefacts guarded here: subtracting the pedestal twice (crushed
shadows), leaving it in place (a grey cast that white balance turns into a
colour cast), applying gain before subtraction (an amplified pedestal), placing
the 2x2 pattern in the wrong CFA order (channel-dependent offsets), a negative
or non-finite level (a sign flip or poisoned arithmetic), and a level outside
the ADC code space (a configuration error that would offset every pixel).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.config.schema import BLCConfig

__all__ = ["BLCConfig", "subtract_black_level"]

_MIN_BIT_DEPTH = 8
_MAX_BIT_DEPTH = 16


def subtract_black_level(
    image: ArrayLike,
    black_level_dn: float,
    pattern: ArrayLike | None = None,
    *,
    bit_depth: int | None = None,
) -> NDArray[np.float32]:
    """Subtract a scalar or per-CFA black level and clamp at zero.

    Parameters
    ----------
    image:
        RAW codes, a 2-D array of shape ``(H, W)``. Integer dtypes such as
        ``uint16`` are accepted, as are the float32 arrays produced by earlier
        linear ISP stages. Every value must be finite and real; the input is
        never modified.
    black_level_dn:
        Scalar zero-signal code. It must be finite, non-negative, and (when
        ``bit_depth`` is supplied) smaller than ``2**bit_depth``.
    pattern:
        Optional 2x2 array of absolute per-site black levels, laid out in
        row-major order and repeated over the image. When supplied it defines
        the levels for every pixel, and ``black_level_dn`` is validated but not
        applied. Each entry must be finite, non-negative, and inside the ADC
        code space.
    bit_depth:
        Optional ADC resolution in bits (8-16). Supplying it validates the
        scalar and pattern against the actual code space; a 10-bit sensor with
        a 1024 DN pedestal is a configuration error, not a large offset.

    Returns
    -------
    A new float32 array of shape ``(H, W)`` containing ``max(code - level, 0)``.
    Values are linear-signal codes: they are proportional to light away from the
    clamp and have not been gamma encoded.

    Raises
    ------
    ValueError
        If the image is not a non-empty 2-D real numeric array, contains
        non-finite values, or the black level or pattern is invalid. These
        conditions otherwise produce a silently offset or sign-flipped image.

    Notes
    -----
    The pattern is positional, not colour-keyed: ``pattern[0, 0]`` applies to
    pixels ``(0, 0), (0, 2), ...`` and so on. A caller using per-CFA levels must
    therefore order the tile to match the sensor's CFA layout. Because
    subtraction is affine, scaling raw codes without also scaling the pedestal
    is not a linear operation; scale the signal above the pedestal instead.
    """
    values = _as_float32_image(image, "image")
    depth = _validate_bit_depth(bit_depth)
    level = _validate_level(black_level_dn, "black_level_dn", depth)
    if pattern is None:
        corrected = values - np.float32(level)
    else:
        tile = _validate_pattern(pattern, depth)
        corrected = values.copy()
        corrected[0::2, 0::2] -= np.float32(tile[0, 0])
        corrected[0::2, 1::2] -= np.float32(tile[0, 1])
        corrected[1::2, 0::2] -= np.float32(tile[1, 0])
        corrected[1::2, 1::2] -= np.float32(tile[1, 1])
    np.maximum(corrected, np.float32(0.0), out=corrected)
    return corrected


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
        raise ValueError(f"{field} values must be finite; NaN or infinity is not a code")
    converted = np.ascontiguousarray(array, dtype=np.float32)
    if not np.all(np.isfinite(converted)):
        raise ValueError(
            f"{field} values must fit in float32; conversion produced a non-finite value"
        )
    return converted


def _validate_bit_depth(bit_depth: int | None) -> int | None:
    """Return an ADC bit depth in ``[8, 16]``, or ``None`` when not supplied."""
    if bit_depth is None:
        return None
    if isinstance(bit_depth, bool) or not isinstance(bit_depth, (int, np.integer)):
        raise ValueError(f"bit_depth must be an integer between 8 and 16; got {bit_depth!r}")
    depth = int(bit_depth)
    if not _MIN_BIT_DEPTH <= depth <= _MAX_BIT_DEPTH:
        raise ValueError(f"bit_depth must be an integer between 8 and 16; got {depth}")
    return depth


def _validate_level(value: Any, field: str, bit_depth: int | None) -> float:
    """Return a finite, non-negative black level inside the ADC code space."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be a finite non-negative real number; got {value!r}")
    level = float(value)
    if not math.isfinite(level):
        raise ValueError(f"{field} must be finite; got {value!r}")
    if level < 0.0:
        raise ValueError(f"{field} must be non-negative; got {level:g}")
    if bit_depth is not None and level >= float(1 << bit_depth):
        raise ValueError(
            f"{field} ({level:g}) must be less than 2**bit_depth = {1 << bit_depth} "
            f"for bit_depth={bit_depth}"
        )
    return level


def _validate_pattern(pattern: Any, bit_depth: int | None) -> NDArray[np.float64]:
    """Return a finite, non-negative 2x2 black-level tile."""
    try:
        tile = np.asarray(pattern, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pattern must be a 2x2 array of real numbers; {exc}") from exc
    if tile.shape != (2, 2):
        raise ValueError(f"pattern must be a 2x2 array; got shape {tile.shape}")
    if not np.all(np.isfinite(tile)):
        raise ValueError("pattern values must be finite; NaN or infinity is not a code")
    if np.any(tile < 0.0):
        raise ValueError(f"pattern values must be non-negative; got {float(tile.min()):g}")
    if bit_depth is not None and np.any(tile >= float(1 << bit_depth)):
        raise ValueError(
            f"pattern values must be less than 2**bit_depth = {1 << bit_depth} "
            f"for bit_depth={bit_depth}; got {float(tile.max()):g}"
        )
    return tile
