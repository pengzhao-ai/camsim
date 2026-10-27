"""Linear sRGB and Display P3 primaries, matrices, and conversions.

A colour stimulus is described by the CIE 1931 tristimulus vector ``XYZ``, a
linear combination of the colour-matching functions against the stimulus
spectrum. An RGB space is a particular basis for that vector: three primaries
plus a white point, with the convention that RGB ``(1, 1, 1)`` reproduces the
white stimulus. A space is therefore fully specified by six chromaticities --
red, green, and blue primaries ``(x, y)`` and the white point -- and the
linear map to XYZ follows from them. For a primary at ``(x, y)``, the
un-normalised XYZ column is::

    [x / y, 1, (1 - x - y) / y]^T

which has unit luminance. Stacking the three columns gives a matrix ``M`` that
still maps ``(1, 1, 1)`` to an arbitrary mixture, so it is post-scaled by the
solution ``s`` of ``M s = W``, where ``W`` is the white point's XYZ vector::

    RGB_TO_XYZ = M @ diag(s)          s = M^-1 W

That normalisation is the whole point of the white point: without it the
space's neutral axis would not coincide with D65 and every grey in the image
would acquire a cast.

Both output spaces here are D65-referenced:

* **sRGB** (IEC 61966-2-1): primaries (0.640, 0.330), (0.300, 0.600),
  (0.150, 0.060); white D65 (0.3127, 0.3290).
* **Display P3** (SMPTE EG 432-1): the DCI-P3 cinema primaries (0.680, 0.320),
  (0.265, 0.690), (0.150, 0.060) with the D65 white point and the *sRGB*
  transfer function. It is a wider gamut than sRGB -- roughly 25% more
  reproducible colours -- which is why a linear sRGB value maps to a smaller,
  more saturated-looking P3 triple.

Because the white points are identical, the conversion between the spaces is
a pure 3x3 matrix with no chromatic adaptation::

    SRGB_TO_P3 = XYZ_TO_RGB_P3 @ RGB_TO_XYZ_SRGB
    P3_TO_SRGB = inverse(SRGB_TO_P3)

Bradford (or any other adaptation transform) is unnecessary and would be
wrong: it exists to correct for a change of white point, and there is none.
Both matrices preserve the neutral axis because every row sums to one, so
``(1, 1, 1)`` maps to ``(1, 1, 1)``.

The matrices are derived from the chromaticities in float64 at import time,
not copied from a table, so they carry full precision. Published tables
usually round to four decimals (the official sRGB matrix differs from the
derived one in the fourth decimal), which is invisible on an 8-bit display
but matters when the values are chained with other calibrations. The
constants are read-only so an accidental in-place edit cannot silently change
every conversion in the process.

Typical parameter ranges and usage: these spaces are the two output encodings
of the ISP, and the conversion is applied to *linear* float values before any
transfer function. Display P3 is the useful target for wide-gamut displays;
converting sRGB content into P3 does not create new colours, it only re-encodes
existing ones. When wide-gamut scene content is rendered into sRGB instead,
out-of-gamut colours acquire negative linear values or values above one. This
module never clips them -- the tone stage owns the final clamp -- because
clipping here would lose hue information before the display encoding.

Failure artefacts guarded or documented here: a wrong primary set (hue and
saturation error), a missing white-point normalisation (a cast on every grey),
an unnecessary chromatic-adaptation step between two D65 spaces, applying the
matrix to display-encoded instead of linear values (a hue shift), clipping
out-of-gamut linear values, assuming Display P3 has its own transfer curve
(it uses the sRGB one), and non-finite, boolean, complex, or wrongly shaped
input. All functions return new C-contiguous float32 arrays and never mutate
their inputs.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.isp.denoise import _as_checked_float32

__all__ = [
    "P3_TO_SRGB",
    "RGB_TO_XYZ_P3",
    "RGB_TO_XYZ_SRGB",
    "SRGB_TO_P3",
    "XYZ_TO_RGB_P3",
    "XYZ_TO_RGB_SRGB",
    "linear_p3_to_srgb",
    "linear_srgb_to_p3",
]

# CIE 1931 chromaticities. The white point is shared by both spaces, which is
# what makes the cross-space conversion adaptation-free.
_SRGB_PRIMARIES: Final[tuple[tuple[float, float], ...]] = (
    (0.640, 0.330),
    (0.300, 0.600),
    (0.150, 0.060),
)
_DISPLAY_P3_PRIMARIES: Final[tuple[tuple[float, float], ...]] = (
    (0.680, 0.320),
    (0.265, 0.690),
    (0.150, 0.060),
)
_D65_WHITE: Final[tuple[float, float]] = (0.3127, 0.3290)


def _rgb_to_xyz_matrix(
    primaries: tuple[tuple[float, float], ...],
    white: tuple[float, float],
) -> NDArray[np.float64]:
    """Derive the RGB-to-XYZ matrix from primaries and a white point.

    Each primary contributes the unit-luminance column
    ``[x / y, 1, (1 - x - y) / y]``; the columns are then scaled by
    ``M^-1 W`` so that RGB ``(1, 1, 1)`` produces the white stimulus ``W``.
    Deriving the matrix instead of tabulating it keeps the last digits exact
    and makes the white-point normalisation part of the definition.
    """
    xyz_columns = np.array(
        [[x / y, 1.0, (1.0 - x - y) / y] for x, y in primaries],
        dtype=np.float64,
    ).T
    x_white, y_white = white
    white_xyz = np.array(
        [x_white / y_white, 1.0, (1.0 - x_white - y_white) / y_white],
        dtype=np.float64,
    )
    scales = np.linalg.solve(xyz_columns, white_xyz)
    return np.ascontiguousarray(xyz_columns * scales, dtype=np.float64)


def _freeze(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mark a shared matrix constant read-only before it escapes the module."""
    matrix.setflags(write=False)
    return matrix


def _inverse(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the float64 inverse of a 3x3 matrix.

    ``np.linalg.inv`` is typed generically over the floating dtype, so the
    result is cast back to float64 explicitly; every input here is float64.
    """
    return np.asarray(np.linalg.inv(matrix), dtype=np.float64)


RGB_TO_XYZ_SRGB: Final[NDArray[np.float64]] = _freeze(
    _rgb_to_xyz_matrix(_SRGB_PRIMARIES, _D65_WHITE)
)
"""``(3, 3)`` float64 sRGB RGB-to-XYZ matrix under D65.

Columns are the XYZ of the red, green, and blue primaries; rows sum to the
D65 white stimulus because the matrix maps ``(1, 1, 1)`` to D65. Read-only.
"""

XYZ_TO_RGB_SRGB: Final[NDArray[np.float64]] = _freeze(_inverse(RGB_TO_XYZ_SRGB))
"""``(3, 3)`` float64 sRGB XYZ-to-RGB matrix, the inverse of the above. Read-only."""

RGB_TO_XYZ_P3: Final[NDArray[np.float64]] = _freeze(
    _rgb_to_xyz_matrix(_DISPLAY_P3_PRIMARIES, _D65_WHITE)
)
"""``(3, 3)`` float64 Display P3 RGB-to-XYZ matrix under D65. Read-only."""

XYZ_TO_RGB_P3: Final[NDArray[np.float64]] = _freeze(_inverse(RGB_TO_XYZ_P3))
"""``(3, 3)`` float64 Display P3 XYZ-to-RGB matrix, the inverse of the above. Read-only."""

SRGB_TO_P3: Final[NDArray[np.float64]] = _freeze(XYZ_TO_RGB_P3 @ RGB_TO_XYZ_SRGB)
"""``(3, 3)`` float64 linear sRGB-to-linear-P3 matrix: ``XYZ_TO_RGB_P3 @ RGB_TO_XYZ_SRGB``.

No chromatic adaptation is included because both spaces share D65. Rows sum
to one, so neutrals are preserved. Apply it to column vectors, or to an
``(..., 3)`` array as ``rgb @ SRGB_TO_P3.T``. Read-only.
"""

P3_TO_SRGB: Final[NDArray[np.float64]] = _freeze(_inverse(SRGB_TO_P3))
"""``(3, 3)`` float64 linear P3-to-linear-sRGB matrix, the inverse of ``SRGB_TO_P3``. Read-only."""


def linear_srgb_to_p3(rgb: ArrayLike) -> NDArray[np.float32]:
    """Convert linear sRGB values to linear Display P3.

    Parameters
    ----------
    rgb:
        Any real numeric array-like whose trailing axis is 3 in R, G, B order:
        a single ``(3,)`` colour, ``(N, 3)`` samples, an ``(H, W, 3)`` image,
        or any larger leading shape. Values must be finite; negatives and
        values above one are accepted because out-of-gamut colours are legal
        in a linear working space. The input is never modified.

    Returns
    -------
    A new C-contiguous float32 array with the same shape as ``rgb``, equal to
    ``rgb @ SRGB_TO_P3.T`` in float64 and rounded to float32 once. Nothing is
    clipped: an sRGB colour outside the P3 gamut keeps its negative linear
    components. This is a *linear* conversion; no transfer function or gamma
    is involved (Display P3 uses the sRGB curve, applied only at output).

    Raises
    ------
    ValueError
        If ``rgb`` does not have a trailing axis of 3, contains bool,
        complex, or non-finite values, or if the result cannot be represented
        in float32. Each condition would otherwise produce silently wrong or
        non-finite pixels.

    Notes
    -----
    The matrix has no chromatic-adaptation term: sRGB and Display P3 are both
    D65-referenced, and its rows sum to one so a neutral input stays neutral.
    The float64 computation from the original input and the single final
    rounding make the result the correctly rounded float32 value of the
    matrix product.
    """
    values = _as_float64_tristimulus(rgb, "rgb")
    return _as_checked_float32(
        values @ SRGB_TO_P3.T,
        "linear_srgb_to_p3 produced non-finite values; the input exceeds float32 range",
    )


def linear_p3_to_srgb(rgb: ArrayLike) -> NDArray[np.float32]:
    """Convert linear Display P3 values to linear sRGB.

    Parameters
    ----------
    rgb:
        Any real numeric array-like whose trailing axis is 3 in R, G, B order,
        with the same validation and shape rules as
        :func:`linear_srgb_to_p3`: finite values, negatives allowed, input
        never modified.

    Returns
    -------
    A new C-contiguous float32 array with the same shape as ``rgb``, equal to
    ``rgb @ P3_TO_SRGB.T`` in float64 and rounded to float32 once. Nothing is
    clipped; P3 colours outside the sRGB gamut return negative linear
    components, which the tone stage later maps to black rather than this
    module silently desaturating them.

    Raises
    ------
    ValueError
        If ``rgb`` does not have a trailing axis of 3, contains bool,
        complex, or non-finite values, or if the result cannot be represented
        in float32.

    Notes
    -----
    This is the exact inverse of :func:`linear_srgb_to_p3` up to the two
    float32 roundings, and it is likewise a pure linear map: no transfer
    function is applied. Neutrals are preserved because the rows sum to one.
    """
    values = _as_float64_tristimulus(rgb, "rgb")
    return _as_checked_float32(
        values @ P3_TO_SRGB.T,
        "linear_p3_to_srgb produced non-finite values; the input exceeds float32 range",
    )


def _as_float64_tristimulus(values: Any, field: str) -> NDArray[np.float64]:
    """Return ``values`` as a finite real float64 array with a trailing axis of 3.

    Any leading shape is accepted -- a single colour, a sample list, an image,
    or a batch of images. Arithmetic happens in float64 from the caller's
    precision and is rounded once on output, so a float32 caller gets the
    correctly rounded float32 result rather than the round-off of a float32
    intermediate.
    """
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a real numeric array with a trailing axis of 3 (R, G, B); {exc}"
        ) from exc
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(
            f"{field} must be an array with a trailing axis of 3 (R, G, B); got shape {array.shape}"
        )
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
    return np.asarray(array, dtype=np.float64)
