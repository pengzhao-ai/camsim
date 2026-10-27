"""Delta-E76 and delta-E2000 colour difference for display-encoded RGB patches.

A colour difference answers a simple question with a deceptively deep
measurement chain: how far apart are two colours *to a human observer*? The
CIE 1976 L*a*b* space is the foundation. It is a nonlinear remapping of CIE
XYZ designed so that equal numeric distances approximate equal perceived
differences: ``L*`` runs from 0 (black) to 100 (diffuse white) and encodes
lightness, while ``a*`` (green-red) and ``b*`` (blue-yellow) encode chromatic
opposition. The remapping divides each tristimulus by the reference white --
``X/Xn``, ``Y/Yn``, ``Z/Zn`` -- and applies the piecewise function

    f(t) = t**(1/3)                        for t > (6/29)**3
           (1/3) (29/6)**2 t + 4/29        otherwise

with ``L* = 116 f(Y/Yn) - 16``, ``a* = 500 (f(X/Xn) - f(Y/Yn))``, and
``b* = 200 (f(Y/Yn) - f(Z/Zn))``. The cube root approximates Stevens' power law
for lightness; the linear segment keeps the slope finite near black, where the
cube root's derivative diverges.

Delta-E76 is the plain Euclidean distance
``sqrt(dL^2 + da^2 + db^2)``. It is simple, symmetric, and still used in camera
tuning, but Lab is not perceptually uniform enough: a chroma difference in the
blue region looks smaller than the same Lab distance in a neutral region.
CIEDE2000 (the CIE 2000 recommendation, implemented here from the published
equation, not called from a library) corrects for that. It rescales the a*
axis by ``G`` so low-chroma pairs contribute less chroma, computes lightness,
chroma, and hue differences after that rescaling, weights them by ``S_L``,
``S_C``, and ``S_H``, and adds the rotation term

    R_T = -sin(2 dtheta) R_C,
    R_C = 2 sqrt(Cbar'^7 / (Cbar'^7 + 25^7)),

which compresses differences in the blue region around hue 275 degrees. The
hue-angle bookkeeping is the fragile part: hue differences wrap modulo 360, a
pair exactly 180 degrees apart must take the shorter, sign-preserving branch,
and a pair with zero chroma has no defined hue and is assigned hue 0. This
module follows the Sharma et al. (2005) formulation and test pairs.

Display RGB versus linear RGB
-----------------------------
The image this module measures is **display-encoded** sRGB or Display P3 in
``[0, 1]``. Display encoding is a nonlinear transfer function applied for
perceptual quantization, so display values must be decoded with the exact sRGB
EOTF before any colourimetry::

    linear = v / 12.92                      for v <= 0.04045
             ((v + 0.055) / 1.055)**2.4     otherwise

Display P3 shares the sRGB transfer curve and differs only in its primaries.
The linear signal then maps to CIE XYZ through the space's 3x3 matrix, whose
columns are the primary stimuli normalised so that ``(1, 1, 1)`` reproduces the
space's white point; both spaces here are D65-referenced. Averaging a patch
must happen in **linear** light, never on display values: the EOTF is convex,
so the mean of encoded values is lighter than the encoding of the mean, and the
bias grows with patch contrast.

Chromatic adaptation and the white point
----------------------------------------
sRGB and Display P3 are display spaces referenced to D65. A chart reference is
measured under the illuminant that lit it -- D65, a 3000 K incandescent lamp,
or a studio strobe -- and its Lab values are relative to that illuminant's
white point. Comparing D65-referenced Lab with 3000 K-referenced Lab
manufactures a large colour error that belongs to neither the camera nor the
chart. This module therefore takes the reference white point as a parameter,
adapts the measured XYZ from D65 to that white with the Bradford transform
(cone-response scaling: ``M_cat = M^-1 diag(M W_ref / M W_src) M``), and only
then converts to Lab with the destination white. For D65 the adaptation is
skipped exactly, so a D65 measurement is the direct matrix product. The
`white_point` argument is thus the **Lab/reference white, not the display
white**: it makes a rendered image comparable with a reference measured under
the same illuminant.

Metrics-layer duplication
-------------------------
The sensor's ISP has its own colour-space stage with the same primaries,
matrices, and sRGB transfer curve. This module deliberately duplicates those
small standard constants rather than importing them: :mod:`camsim.metrics` must
remain independent of :mod:`camsim.isp` (and the sensor, scene, and optics
packages) so the measurement harness cannot depend on the model it is meant to
judge. Every duplicated constant and formula is cross-checked against
colour-science in the tests.

Real-world ranges
-----------------
Camera colour accuracy on a ColorChecker is usually reported as a mean
delta-E00 of about 1-5, with good cameras below 2-3 and visible errors above 5;
a pair of adjacent neutral patches typically differs by less than 1. Lab
lightness spans 0-100 and a*/b* roughly -100 to 100 (a saturated red near
a* = 80, a deep blue near b* = -100). Display values span [0, 1] with 8-bit
quantization steps of about 0.0039; the sRGB EOTF knee sits at 0.04045, below
which the curve is linear.

Failure artefacts guarded here
------------------------------
* **Display values treated as linear, or linearized twice.** Both apply the
  transfer function wrongly and shift lightness and hue.
* **Averaging display values instead of linear ones.** A high-contrast patch
  measures lighter than the display mean suggests.
* **Comparing Lab under mismatched white points.** A D65 reference applied to a
  3000 K capture looks like a large camera error.
* **Clipping an out-of-range display patch.** Clipping hides a renderer or tone
  bug; values outside ``[0, 1]`` beyond ``1e-6`` are rejected instead.
* **A missing or mis-sized reference.** There is no implicit D65 fallback, and
  reference rows must match patches one to one; a silently truncated reference
  would compare the wrong colours.
* **Monochrome input.** Colour difference needs three channels; a 2-D image is
  rejected with a clear message rather than measured as if it were RGB.
* **In-sample CCM fitting.** A colour correction matrix fitted and evaluated on
  the same chart reports an optimistic delta-E; this module always compares
  against separate spectral ground truth supplied by the caller.

The module imports NumPy, the standard library, :mod:`camsim.core.color`, and
:mod:`camsim.metrics.chart` only -- never the sensor, ISP, scene, or optics
packages. Every function is deterministic and never modifies its input.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.core.color import ColorReference
from camsim.metrics.chart import ChartLayout, extract_patch

__all__ = ["ColorResult", "PatchColorResult", "delta_e_patches", "linear_rgb_to_lab"]

# CIE 1931 D65 chromaticity. Both output spaces below are D65-referenced, which
# is why the only adaptation in the pipeline is from D65 to the reference white.
_D65_WHITE_XY: Final[tuple[float, float]] = (0.3127, 0.3290)

# CIE 1931 primary chromaticities (x, y). sRGB is IEC 61966-2-1; Display P3 is
# SMPTE EG 432-1, which reuses the DCI-P3 primaries with the D65 white point
# and the sRGB transfer function.
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

_COLOR_SPACES: Final[tuple[str, str]] = ("sRGB", "DisplayP3")

# sRGB transfer-function constants from IEC 61966-2-1, duplicated from the ISP
# tone stage because the metrics layer must not import camsim.isp. The threshold
# is the encoded value where the linear and power segments meet; the two
# branches agree to display precision at that point.
_EOTF_THRESHOLD: Final[float] = 0.04045
_EOTF_SLOPE: Final[float] = 12.92
_EOTF_GAIN: Final[float] = 1.055
_EOTF_OFFSET: Final[float] = 0.055
_EOTF_GAMMA: Final[float] = 2.4

# Display-encoded samples may sit a hair outside [0, 1] after independent
# rounding; anything farther out is a renderer or tone bug, not quantisation,
# and is rejected rather than clipped into a plausible-looking colour.
_DISPLAY_RANGE_TOLERANCE: Final[float] = 1e-6

# CIE 1976 Lab constants. Epsilon is (6/29)**3 and kappa is (29/3)**3, the
# standard values where the cube-root segment meets the linear toe.
_LAB_EPSILON: Final[float] = 216.0 / 24389.0
_LAB_KAPPA: Final[float] = 24389.0 / 27.0


def _rgb_to_xyz_matrix(
    primaries: tuple[tuple[float, float], ...],
    white: tuple[float, float],
) -> NDArray[np.float64]:
    """Derive an RGB-to-XYZ matrix from primaries and a white point.

    Each primary contributes the unit-luminance column
    ``[x / y, 1, (1 - x - y) / y]``; the columns are then scaled by the
    solution of ``M s = W`` so that RGB ``(1, 1, 1)`` maps to the white
    stimulus ``W``. Deriving the matrix instead of tabulating a rounded
    published table keeps the last digits exact and makes the white-point
    normalisation part of the definition. The approach mirrors the ISP
    colour-space stage, duplicated here to keep the metrics layer independent.
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


_RGB_TO_XYZ_SRGB: Final[NDArray[np.float64]] = _freeze(
    _rgb_to_xyz_matrix(_SRGB_PRIMARIES, _D65_WHITE_XY)
)
_RGB_TO_XYZ_DISPLAY_P3: Final[NDArray[np.float64]] = _freeze(
    _rgb_to_xyz_matrix(_DISPLAY_P3_PRIMARIES, _D65_WHITE_XY)
)
_RGB_TO_XYZ_BY_SPACE: Final[dict[str, NDArray[np.float64]]] = {
    "sRGB": _RGB_TO_XYZ_SRGB,
    "DisplayP3": _RGB_TO_XYZ_DISPLAY_P3,
}

# Bradford cone-response matrix from the published chromatic-adaptation
# transform. The Von Kries adaptation is M^-1 diag(M W_ref / M W_src) M.
_BRADFORD: Final[NDArray[np.float64]] = _freeze(
    np.array(
        [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ],
        dtype=np.float64,
    )
)
_BRADFORD_INVERSE: Final[NDArray[np.float64]] = _freeze(
    np.asarray(np.linalg.inv(_BRADFORD), dtype=np.float64)
)


def _require_non_empty_text(value: object, field: str) -> str:
    """Return a stripped, non-empty string or raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string; got {value!r}")
    return value.strip()


def _require_finite_or_nan(value: object, field: str) -> float:
    """Return a real number as a float, allowing NaN but rejecting infinity."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be a real number or NaN; got {value!r}")
    number = float(value)
    if math.isinf(number):
        raise ValueError(f"{field} must be finite or NaN; got {value!r}")
    return number


def _require_bool(value: object, field: str) -> bool:
    """Return a Python bool, accepting NumPy booleans."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a bool; got {value!r}")
    return bool(value)


def _require_white_point(value: object) -> tuple[float, float]:
    """Return a physical ``(x, y)`` chromaticity or raise ``ValueError``.

    A white point must lie strictly inside the spectral locus: positive ``x``
    and ``y`` with ``x + y < 1``. The last condition keeps the implied ``z``
    chromaticity positive, so the white XYZ has no zero component and the Lab
    ratios are finite.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"white_point must be a 2-tuple of finite (x, y); got {value!r}")
    coordinates = list(value)
    if len(coordinates) != 2:
        raise ValueError(
            f"white_point must contain exactly two chromaticity coordinates; got {len(coordinates)}"
        )
    for coordinate in coordinates:
        if isinstance(coordinate, (bool, np.bool_)) or not isinstance(
            coordinate, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"white_point must contain real numbers; got {value!r}")
    x = float(coordinates[0])
    y = float(coordinates[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"white_point must be finite; got {value!r}")
    if not (0.0 < x < 1.0 and 0.0 < y < 1.0 and x + y < 1.0):
        raise ValueError(
            "white_point must be a physical (x, y) chromaticity with 0 < x < 1, "
            f"0 < y < 1, and x + y < 1; got {value!r}"
        )
    return (x, y)


def _require_color_space(value: object) -> str:
    """Return one of the exact, case-sensitive output space names."""
    if not isinstance(value, str) or value not in _COLOR_SPACES:
        raise ValueError(f"color_space must be one of {_COLOR_SPACES}; got {value!r}")
    return value


def _as_float64_tristimulus(values: ArrayLike, field: str) -> NDArray[np.float64]:
    """Return a finite, real float64 array whose trailing axis is 3.

    Any leading shape is accepted -- a single colour, a sample list, an image,
    or a batch of images. Boolean, complex, non-numeric, and non-finite values
    are rejected because each would silently corrupt a colour difference.
    """
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a real numeric array with a trailing axis of 3; {exc}"
        ) from exc
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(
            f"{field} must be an array with a trailing axis of 3 (R, G, B); got shape {array.shape}"
        )
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{field} must contain real numbers, not boolean values")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{field} must be real; complex values are not supported")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field} must contain real numbers; got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite values (no NaN or infinity)")
    return np.asarray(array, dtype=np.float64)


def _as_display_rgb(image: ArrayLike) -> NDArray[np.float64]:
    """Return a finite float64 ``(H, W, 3)`` display image in ``[0, 1]``.

    The range is enforced to within :data:`_DISPLAY_RANGE_TOLERANCE` and the
    values are passed through unclipped. Clipping would turn a renderer or tone
    bug into a plausible-looking colour and hide the evidence.
    """
    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"image must be a display-encoded RGB array-like; {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            "colour metrics require three-channel RGB; got shape "
            f"{array.shape}. Convert mono or non-RGB frames to (H, W, 3) RGB "
            "before measuring a colour difference."
        )
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError("image must contain real numbers, not boolean values")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("image must be real; complex values are not supported")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"image must contain real numbers; got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values (no NaN or infinity)")
    if not np.all(array >= -_DISPLAY_RANGE_TOLERANCE) or not np.all(
        array <= 1.0 + _DISPLAY_RANGE_TOLERANCE
    ):
        raise ValueError(
            "image values must be display-encoded RGB in [0, 1]; got values "
            f"outside [{-_DISPLAY_RANGE_TOLERANCE:g}, {1.0 + _DISPLAY_RANGE_TOLERANCE:g}]. "
            "A linear-light image must be encoded before a colour measurement, "
            "and an out-of-range rendered image is rejected rather than clipped."
        )
    return np.asarray(array, dtype=np.float64)


def _xy_to_xyz(white_point: tuple[float, float]) -> NDArray[np.float64]:
    """Return the white stimulus ``[x/y, 1, (1 - x - y)/y]`` for a chromaticity."""
    x, y = white_point
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def _srgb_eotf(values: ArrayLike) -> NDArray[np.float64]:
    """Decode display values with the exact sRGB EOTF.

    Display P3 uses the same transfer curve, so one decoder serves both output
    spaces. Values below the knee take the linear segment; the power segment is
    evaluated only conceptually for them (``np.where`` keeps the expression
    branchless), which is why the results stay finite for the tiny negative
    values the range tolerance admits.
    """
    array = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        encoded = ((array + _EOTF_OFFSET) / _EOTF_GAIN) ** _EOTF_GAMMA
    return np.asarray(
        np.where(array <= _EOTF_THRESHOLD, array / _EOTF_SLOPE, encoded),
        dtype=np.float64,
    )


def _bradford_adaptation(
    xyz: NDArray[np.float64],
    source_white_xyz: NDArray[np.float64],
    destination_white_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Adapt XYZ from one white point to another with the Bradford transform.

    The Von Kries hypothesis scales each cone response independently by the
    ratio of the destination to the source white. Bradford's sharpened cone
    matrix ``M`` is used for the response space:
    ``XYZ_a = M^-1 diag(M W_dst / M W_src) M XYZ``. The operation acts on the
    trailing axis, so a single colour, a sample list, and an image all work.
    """
    cone_response = xyz @ _BRADFORD.T
    gains = (_BRADFORD @ destination_white_xyz) / (_BRADFORD @ source_white_xyz)
    return np.asarray((cone_response * gains) @ _BRADFORD_INVERSE.T, dtype=np.float64)


def _xyz_to_lab(xyz: NDArray[np.float64], white_xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert XYZ to CIE 1976 L*a*b* relative to ``white_xyz``.

    The cube-root segment handles values above ``(6/29)**3``; below it a linear
    segment with the matching value and slope avoids the infinite derivative at
    black. ``np.cbrt`` is used so negative tristimulus components (legal in a
    linear working space) stay real instead of turning into NaN.
    """
    ratio = xyz / white_xyz
    transformed = np.where(
        ratio > _LAB_EPSILON,
        np.cbrt(ratio),
        (_LAB_KAPPA * ratio + 16.0) / 116.0,
    )
    lightness = 116.0 * transformed[..., 1] - 16.0
    a_star = 500.0 * (transformed[..., 0] - transformed[..., 1])
    b_star = 200.0 * (transformed[..., 1] - transformed[..., 2])
    return np.asarray(
        np.stack([lightness, a_star, b_star], axis=-1),
        dtype=np.float64,
    )


def linear_rgb_to_lab(
    image: ArrayLike,
    color_space: str,
    white_point: tuple[float, float],
) -> NDArray[np.float64]:
    """Convert **linear** RGB to CIE Lab relative to a reference white point.

    Parameters
    ----------
    image:
        Any real numeric array-like whose trailing axis is 3 in R, G, B order:
        a single ``(3,)`` colour, ``(N, 3)`` samples, an ``(H, W, 3)`` image, or
        any larger leading shape. The values are **linear light**, not display
        values. Negative and above-one components are accepted because
        out-of-gamut colours are legal in a linear working space; boolean,
        complex, non-finite, and wrongly shaped input is rejected.
    color_space:
        ``"sRGB"`` or ``"DisplayP3"``, exactly and case-sensitively. It selects
        the RGB-to-XYZ matrix and therefore the primaries. Both spaces are
        D65-referenced and share the sRGB transfer curve, but this function
        works in linear light and never applies a transfer function.
    white_point:
        Reference white ``(x, y)`` CIE 1931 chromaticity for the Lab
        conversion, for example ``(0.3127, 0.3290)`` for D65 or
        ``(0.4369, 0.4041)`` for a 3000 K blackbody. This is the Lab/reference
        white, **not** the display white: the linear RGB is converted to XYZ
        with its native D65 white and then chromatically adapted (Bradford) to
        this white point, so the result is comparable with a reference measured
        under that illuminant. For D65 the adaptation is the identity.

    Returns
    -------
    ndarray
        A new C-contiguous float64 array with the same shape as ``image`` and a
        trailing axis of 3 holding ``L*``, ``a*``, and ``b*``. A neutral colour
        is ``a* = b* = 0``; RGB ``(1, 1, 1)`` becomes ``L* = 100`` under any
        white point because the adaptation maps the D65 white to it.

    Raises
    ------
    ValueError
        If ``color_space`` is not an exact supported name, ``white_point`` is
        not a physical chromaticity, or ``image`` is not a finite real numeric
        array with a trailing axis of 3.

    Notes
    -----
    The conversion is deterministic, allocates its output, and never modifies
    the input. Display-encoded RGB must be decoded with the sRGB EOTF first
    (Display P3 uses the same curve); feeding display values here treats them
    as linear intensities and biases lightness and hue.
    """
    space = _require_color_space(color_space)
    white = _require_white_point(white_point)
    values = _as_float64_tristimulus(image, "image")

    xyz = values @ _RGB_TO_XYZ_BY_SPACE[space].T
    white_xyz = _xy_to_xyz(white)
    if white != _D65_WHITE_XY:
        # Adapting D65 to D65 is algebraically the identity but not bit-exact
        # after matrix inversion, so the exact D65 case skips the step.
        xyz = _bradford_adaptation(xyz, _xy_to_xyz(_D65_WHITE_XY), white_xyz)
    return np.ascontiguousarray(_xyz_to_lab(xyz, white_xyz), dtype=np.float64)


def _delta_e76(lab1: ArrayLike, lab2: ArrayLike) -> NDArray[np.float64]:
    """Return the CIE 1976 colour difference (Euclidean Lab distance).

    Both inputs may be ``(3,)``, ``(N, 3)``, or any leading shape whose last
    axis is 3, and their leading shapes broadcast. The result has the broadcast
    leading shape: a scalar-equivalent 0-d array for two single colours.
    """
    first = _as_float64_tristimulus(lab1, "lab1")
    second = _as_float64_tristimulus(lab2, "lab2")
    try:
        difference = first - second
    except ValueError as exc:
        raise ValueError(f"lab1 and lab2 leading shapes cannot be broadcast; {exc}") from exc
    return np.asarray(
        np.sqrt(np.sum(difference * difference, axis=-1)),
        dtype=np.float64,
    )


def _delta_e2000(lab1: ArrayLike, lab2: ArrayLike) -> NDArray[np.float64]:
    """Return the CIEDE2000 colour difference with ``kL = kC = kH = 1``.

    The full CIE 2000 recommendation is implemented from the published
    equations: the chroma rescaling ``G``, the prime hue angles, the wrapped
    hue difference, the weighting functions ``S_L``, ``S_C``, ``S_H``, and the
    blue-region rotation term ``R_T``. Inputs and broadcasting follow
    :func:`_delta_e76`. No library is called; the tests pin this
    implementation against colour-science and published CIEDE2000 test pairs.
    """
    first = _as_float64_tristimulus(lab1, "lab1")
    second = _as_float64_tristimulus(lab2, "lab2")
    try:
        np.broadcast_shapes(first.shape[:-1], second.shape[:-1])
    except ValueError as exc:
        raise ValueError(f"lab1 and lab2 leading shapes cannot be broadcast; {exc}") from exc

    l_1, a_1, b_1 = first[..., 0], first[..., 1], first[..., 2]
    l_2, a_2, b_2 = second[..., 0], second[..., 1], second[..., 2]

    chroma_1 = np.hypot(a_1, b_1)
    chroma_2 = np.hypot(a_2, b_2)
    mean_chroma = (chroma_1 + chroma_2) / 2.0
    mean_chroma_seventh = mean_chroma**7
    g_factor = 0.5 * (1.0 - np.sqrt(mean_chroma_seventh / (mean_chroma_seventh + 25.0**7)))

    a_1_prime = (1.0 + g_factor) * a_1
    a_2_prime = (1.0 + g_factor) * a_2
    chroma_1_prime = np.hypot(a_1_prime, b_1)
    chroma_2_prime = np.hypot(a_2_prime, b_2)

    # A neutral colour has no hue: the CIE assigns it hue 0 and the selection
    # below then reports zero hue difference and the sum of the two hues as the
    # mean hue, so a neutral pair contributes only lightness and chroma terms.
    hue_1_prime = np.where(
        (b_1 == 0.0) & (a_1_prime == 0.0),
        0.0,
        np.degrees(np.arctan2(b_1, a_1_prime)) % 360.0,
    )
    hue_2_prime = np.where(
        (b_2 == 0.0) & (a_2_prime == 0.0),
        0.0,
        np.degrees(np.arctan2(b_2, a_2_prime)) % 360.0,
    )

    delta_l_prime = l_2 - l_1
    delta_c_prime = chroma_2_prime - chroma_1_prime

    hue_difference = hue_2_prime - hue_1_prime
    chroma_product = chroma_1_prime * chroma_2_prime
    delta_h_prime = np.select(
        [
            chroma_product == 0.0,
            np.abs(hue_difference) <= 180.0,
            hue_difference > 180.0,
            hue_difference < -180.0,
        ],
        [
            0.0,
            hue_difference,
            hue_difference - 360.0,
            hue_difference + 360.0,
        ],
    )
    delta_h_big_prime = 2.0 * np.sqrt(chroma_product) * np.sin(np.radians(delta_h_prime / 2.0))

    mean_lightness = (l_1 + l_2) / 2.0
    mean_chroma_prime = (chroma_1_prime + chroma_2_prime) / 2.0

    hue_sum = hue_1_prime + hue_2_prime
    hue_separation = np.abs(hue_1_prime - hue_2_prime)
    mean_hue_prime = np.select(
        [
            chroma_product == 0.0,
            hue_separation <= 180.0,
            (hue_separation > 180.0) & (hue_sum < 360.0),
            (hue_separation > 180.0) & (hue_sum >= 360.0),
        ],
        [
            hue_sum,
            hue_sum / 2.0,
            (hue_sum + 360.0) / 2.0,
            (hue_sum - 360.0) / 2.0,
        ],
    )

    t_factor = (
        1.0
        - 0.17 * np.cos(np.radians(mean_hue_prime - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * mean_hue_prime))
        + 0.32 * np.cos(np.radians(3.0 * mean_hue_prime + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * mean_hue_prime - 63.0))
    )

    # The rotation term is largest in the blue region around hue 275 degrees.
    delta_theta = 30.0 * np.exp(-(((mean_hue_prime - 275.0) / 25.0) ** 2))
    mean_chroma_prime_seventh = mean_chroma_prime**7
    rotation_c = 2.0 * np.sqrt(mean_chroma_prime_seventh / (mean_chroma_prime_seventh + 25.0**7))
    rotation = -np.sin(np.radians(2.0 * delta_theta)) * rotation_c

    lightness_term = (mean_lightness - 50.0) ** 2
    s_l = 1.0 + (0.015 * lightness_term) / np.sqrt(20.0 + lightness_term)
    s_c = 1.0 + 0.045 * mean_chroma_prime
    s_h = 1.0 + 0.015 * mean_chroma_prime * t_factor

    scaled_lightness = delta_l_prime / s_l
    scaled_chroma = delta_c_prime / s_c
    scaled_hue = delta_h_big_prime / s_h
    return np.asarray(
        np.sqrt(
            scaled_lightness**2
            + scaled_chroma**2
            + scaled_hue**2
            + rotation * scaled_chroma * scaled_hue
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PatchColorResult:
    """One chart patch's colour difference from its reference.

    Parameters
    ----------
    name:
        Patch label copied from the layout, non-empty and stored stripped.
    delta_e76:
        CIE 1976 Euclidean Lab distance from the reference. Finite when
        ``valid`` is ``True``; ``NaN`` denotes an unmeasurable patch.
    delta_e2000:
        CIEDE2000 colour difference from the reference under the reference
        conditions (``kL = kC = kH = 1``). Same finiteness rule as above.
    valid:
        ``True`` only when both differences are finite. An invalid patch may
        carry ``NaN``; a valid one may not.

    The dataclass is frozen so a per-patch row can travel from a metric into a
    report without being edited. Camera colour differences typically run from
    below 1 for a near-perfect match to 5 or more for a visible error.
    """

    name: str
    delta_e76: float
    delta_e2000: float
    valid: bool

    def __post_init__(self) -> None:
        name = _require_non_empty_text(self.name, "patch name")
        delta_e76 = _require_finite_or_nan(self.delta_e76, "delta_e76")
        delta_e2000 = _require_finite_or_nan(self.delta_e2000, "delta_e2000")
        valid = _require_bool(self.valid, "valid")
        if valid and (math.isnan(delta_e76) or math.isnan(delta_e2000)):
            raise ValueError(
                "valid=True requires finite delta_e76 and delta_e2000 values; "
                "an unmeasurable patch must set valid=False"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "delta_e76", delta_e76)
        object.__setattr__(self, "delta_e2000", delta_e2000)
        object.__setattr__(self, "valid", valid)


def _mean_of(values: list[float]) -> float:
    """Return the arithmetic mean of ``values``, or NaN when there are none."""
    return float(np.mean(values)) if values else math.nan


def _require_mean_matches(value: float, expected: float, field: str) -> None:
    """Raise unless ``value`` is the mean of the valid per-patch differences."""
    if math.isnan(expected):
        if not math.isnan(value):
            raise ValueError(f"{field} must be NaN when no patch is valid; got {value!r}")
        return
    if math.isnan(value) or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"{field} must equal the mean over valid patches ({expected!r}); got {value!r}"
        )


@dataclass(frozen=True)
class ColorResult:
    """A chart-wide colour-accuracy measurement.

    Parameters
    ----------
    delta_e76, delta_e2000:
        Arithmetic means of the per-patch differences over the **valid**
        patches. ``NaN`` when no patch is valid. The constructor verifies that
        the supplied values really are those means, so a report cannot display
        a summary that disagrees with its rows.
    per_patch:
        Per-patch :class:`PatchColorResult` rows in layout order. A sequence is
        accepted and frozen into a tuple; every element must be a
        :class:`PatchColorResult`.
    valid:
        ``True`` only when there is at least one patch, every patch is valid,
        and both means are finite. An empty result has no defined mean and is
        therefore invalid.

    The dataclass is frozen. It carries the per-patch rows so a report can show
    which patches dominate an error instead of only a summary number; a mean of
    about 1-5 delta-E00 is typical camera colour, and individual patches can be
    several times that.
    """

    delta_e76: float
    delta_e2000: float
    per_patch: tuple[PatchColorResult, ...]
    valid: bool

    def __post_init__(self) -> None:
        try:
            per_patch = tuple(self.per_patch)
        except TypeError as exc:
            raise ValueError(f"per_patch must be an iterable of PatchColorResult; {exc}") from exc
        if not all(isinstance(patch, PatchColorResult) for patch in per_patch):
            raise ValueError("per_patch must contain only PatchColorResult instances")

        delta_e76 = _require_finite_or_nan(self.delta_e76, "delta_e76")
        delta_e2000 = _require_finite_or_nan(self.delta_e2000, "delta_e2000")
        valid = _require_bool(self.valid, "valid")

        valid_patches = [patch for patch in per_patch if patch.valid]
        expected_76 = _mean_of([patch.delta_e76 for patch in valid_patches])
        expected_2000 = _mean_of([patch.delta_e2000 for patch in valid_patches])
        _require_mean_matches(delta_e76, expected_76, "delta_e76")
        _require_mean_matches(delta_e2000, expected_2000, "delta_e2000")

        expected_valid = (
            bool(per_patch)
            and len(valid_patches) == len(per_patch)
            and math.isfinite(expected_76)
            and math.isfinite(expected_2000)
        )
        if valid != expected_valid:
            raise ValueError(
                "valid must be True only when every patch is valid and both means "
                f"are finite (expected {expected_valid}); got {valid}"
            )

        object.__setattr__(self, "per_patch", per_patch)
        object.__setattr__(self, "delta_e76", delta_e76)
        object.__setattr__(self, "delta_e2000", delta_e2000)
        object.__setattr__(self, "valid", valid)


def delta_e_patches(
    image: ArrayLike,
    layout: ChartLayout,
    reference: ColorReference | None = None,
    color_space: str = "sRGB",
) -> ColorResult:
    """Measure every chart patch's colour difference from its reference.

    Parameters
    ----------
    image:
        Display-encoded ``(H, W, 3)`` RGB array-like in ``[0, 1]`` (within
        ``1e-6`` at the ends). The values are **display** values; the function
        decodes them with the exact sRGB EOTF, applies the ``color_space``
        matrix, adapts to the reference white point, and computes Lab. A 2-D
        mono or non-three-channel image raises a clear error: colour difference
        has no meaning on a single plane.
    layout:
        A :class:`~camsim.metrics.chart.ChartLayout` whose patch ROIs select
        the regions to average. The ROI crop is delegated to
        :func:`~camsim.metrics.chart.extract_patch`, so a region that extends
        past the frame raises instead of being silently truncated. Patch means
        are taken in **linear** light.
    reference:
        The per-patch CIE Lab ground truth. When ``None``, ``layout.reference``
        is used. A missing reference is an error: the metrics layer has no
        implicit D65 fallback, because a reference that does not match the
        illuminant produces a systematic delta-E offset. An explicit reference
        overrides the layout's, which is useful for comparing one capture
        against several candidate references.
    color_space:
        ``"sRGB"`` (default) or ``"DisplayP3"``, exactly. It selects the
        primaries of the display encoding; Display P3 uses the same transfer
        curve.

    Returns
    -------
    ColorResult
        Per-patch rows in layout order plus the means over valid patches and a
        validity flag. Every patch is valid when its differences are finite,
        which is the case for a validated display image and a validated
        reference.

    Raises
    ------
    ValueError
        If ``layout`` is not a :class:`ChartLayout`, the image is not a finite
        real ``(H, W, 3)`` display image in range, ``color_space`` is invalid,
        no reference is available, the reference row count differs from the
        patch count, or a patch ROI lies outside the image.

    Notes
    -----
    The pipeline is: validate; decode display RGB to linear; crop each ROI and
    average it in linear light; ``linear_rgb_to_lab`` with the reference white
    point; then delta-E76 and delta-E2000 against the matching reference row.
    Because the reference Lab is illuminant-relative, a D65 reference applied
    to a warm-light capture is a white-point error rather than a camera error;
    the scene layer computes a matching reference spectrally and binds it to
    the layout. The result is deterministic and the input image is not
    modified. A colour correction matrix fitted on the same patches that are
    measured here would be an in-sample evaluation and is the caller's
    responsibility to avoid.
    """
    if not isinstance(layout, ChartLayout):
        raise ValueError(f"layout must be a ChartLayout; got {type(layout).__name__}")
    display = _as_display_rgb(image)
    space = _require_color_space(color_space)

    if reference is not None and not isinstance(reference, ColorReference):
        raise ValueError(
            f"reference must be a ColorReference or None; got {type(reference).__name__}"
        )
    resolved = layout.reference if reference is None else reference
    if resolved is None:
        raise ValueError(
            "a colour reference is required; pass reference= or attach a "
            "ColorReference to the chart layout (the metrics layer has no "
            "implicit D65 fallback)"
        )
    if len(resolved.lab) != len(layout.patches):
        raise ValueError(
            f"the colour reference has {len(resolved.lab)} rows but the layout has "
            f"{len(layout.patches)} patches; reference rows must match patches one to one"
        )

    linear = _srgb_eotf(display)
    per_patch: list[PatchColorResult] = []
    for index, patch in enumerate(layout.patches):
        crop = np.asarray(extract_patch(linear, patch), dtype=np.float64)
        # Average in linear light: the EOTF is convex, so a display-value mean
        # would report a lighter patch and a systematically different Lab.
        mean_rgb = crop.reshape(-1, 3).mean(axis=0)
        measured = linear_rgb_to_lab(mean_rgb, space, resolved.white_point_xy)
        difference_76 = float(_delta_e76(measured, resolved.lab[index]))
        difference_2000 = float(_delta_e2000(measured, resolved.lab[index]))
        patch_valid = math.isfinite(difference_76) and math.isfinite(difference_2000)
        per_patch.append(
            PatchColorResult(
                name=patch.name,
                delta_e76=difference_76,
                delta_e2000=difference_2000,
                valid=patch_valid,
            )
        )

    results = tuple(per_patch)
    valid_patches = [patch for patch in results if patch.valid]
    mean_76 = _mean_of([patch.delta_e76 for patch in valid_patches])
    mean_2000 = _mean_of([patch.delta_e2000 for patch in valid_patches])
    overall_valid = (
        bool(results)
        and len(valid_patches) == len(results)
        and math.isfinite(mean_76)
        and math.isfinite(mean_2000)
    )
    return ColorResult(
        delta_e76=mean_76,
        delta_e2000=mean_2000,
        per_patch=results,
        valid=overall_valid,
    )
