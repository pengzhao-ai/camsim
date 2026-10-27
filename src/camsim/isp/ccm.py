"""Colour correction: mix sensor RGB into a target colour space.

A colour sensor's filters are broad and overlapping, and the three RAW channels
are not a colorimetric basis. A camera's "red" channel also sees green light,
its "green" sees red and blue, and its "blue" leaks red, so captured colours are
desaturated and hue-shifted relative to what the eye sees. The ISP corrects this
with a 3x3 linear mixing matrix, calibrated once against reference patches
(typically a 24-patch ColorChecker) captured under a known illuminant and
compared with their reference values::

    target_rgb ~= camera_rgb @ M.T        (equivalently, t = M @ c)

The matrix is applied per pixel in linear light, after demosaicing and before
tone mapping. Diagonal entries are typically 0.5-1.5 and off-diagonal entries
roughly -0.5 to 0.5; larger off-diagonals mean stronger spectral overlap, and
therefore stronger noise amplification (see below). Rows usually sum to one,
which keeps the neutral axis fixed: a neutral patch must stay neutral, or every
grey in the scene acquires a cast.

Fitting row ``i`` of ``M`` is an independent ridged least-squares problem::

    m_i = argmin ||C m_i - t_i||^2 + ridge * ||m_i - e_i||^2

where ``C`` is the ``(N, 3)`` camera patch matrix, ``t_i`` is column ``i`` of the
target patches, and ``e_i`` is row ``i`` of the identity. The prior is the
identity, not zero: the fit is biased toward "do nothing", which is the safe
answer when the data do not support a strong correction. ``ridge`` therefore
controls a bias-variance trade-off. The noise gain of an unregularised fit is
governed by the singular values of ``C``. A patch set that weakly excites one
colour direction turns sensor or measurement noise into large coefficients
along that direction, and the resulting matrix amplifies chroma noise, most
visible in low-light shadows where signal-to-noise ratio is worst. Typical
``ridge`` values run from about 1e-6 for clean, well-exposed calibrations to
1e-2 for noisy low-light data or a small patch set; 1e-4 is a sensible default.
Too much ridge drags every row toward the identity and desaturates colours,
which is the opposite failure.

When ``row_sum`` is true the fit imposes ``sum_j M[i, j] = 1`` as an exact
equality constraint. The KKT system

    [[C^T C + ridge I, 1], [1^T, 0]] [m_i; nu] = [C^T t_i + ridge e_i; 1]

is solved directly, so row sums are one to machine precision rather than to the
accuracy of a penalty approximation. For rank-3 camera data ``C^T C`` is already
positive definite, so an unconstrained fit needs no ridge to be nonsingular; the
row-sum constraint additionally makes some rank-2 problems uniquely solvable by
determining the missing direction, but it cannot rescue a normal matrix whose
null space contains a direction orthogonal to the constraint vector (the
all-ones vector). A singular normal-equation system -- for example every patch
recorded as the same colour -- raises instead of returning huge or non-finite
coefficients, whether or not the constraint is enabled.

Application is a matrix product in linear light followed by a clamp at zero::

    out = max(rgb @ M.T, 0)

The clamp is the only non-linearity in the stage. It exists because a matrix
with negative off-diagonal entries drives saturated or out-of-gamut colours
below zero, and a negative radiance is not physical. Clipping does lose hue
information in those pixels, so a matrix whose outputs depend heavily on
clipping is a sign of an over-strong fit. Everything before the tone curve
remains linear; applying a CCM to gamma-encoded data is a failure mode, because
the transform then acts on a non-linear function of radiance.

**Bounded exception to the linear-space invariant.** The design says the CCM
operates in linear float32 light. It does, but its non-negativity clamp means
the stage is linear only on the in-gamut domain where every matrix output is
non-negative. That exception is bounded and explicit: on in-gamut data the
clamp is inactive and scaling the input scales the output exactly; on
out-of-gamut data the clamp trades hue accuracy for physical non-negative
radiance. The G3 linearity check uses an in-gamut frame and an explicitly
measured pre-clamp minimum, and this is the only pre-tone stage allowed such a
clamp.

Failure artefacts guarded or documented here: over-fitting a noisy or small
patch set (large coefficients, chroma noise), negative RGB after the transform
(clamped, at the cost of hue), a neutral shift when rows do not sum to one,
over-regularisation desaturating colours, a singular or rank-deficient fit, and
applying the matrix in the wrong domain or with the wrong convention.

The CIEDE2000 diagnostic in :func:`evaluate_ccm` interprets both camera RGB and
target RGB as **linear sRGB under D65**. Values are converted straight to XYZ
with no transfer function, and negatives are clipped for that calculation only.
Camera RGB is not actually colorimetric sRGB -- that is the very problem the
matrix corrects -- so absolute ΔE values are indicative, while the before/after
comparison is the usable signal.

Fitting accepts real numeric ``(N, 3)`` samples with ``N >= 3`` and computes in
float64. Application accepts any real numeric array whose trailing axis is 3 --
a single ``(3,)`` colour, ``(N, 3)`` samples, or an ``(H, W, 3)`` image -- and
returns a new C-contiguous float32 array. Neither function mutates its inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.config.schema import CCMConfig

__all__ = ["CCMConfig", "CCMResult", "apply_ccm", "evaluate_ccm", "fit_ccm"]

# The row-sum equality is imposed by the KKT solve, so any deviation beyond
# accumulated round-off means the problem was numerically singular even though
# the rank test passed. 1e-8 is far below any physically meaningful coefficient
# error (coefficients are order 0.1-1) yet far above the ~1e-16 a well-posed
# solve leaves behind.
_ROW_SUM_TOLERANCE: Final[float] = 1e-8

_SINGULAR_MESSAGE: Final[str] = (
    "the CCM normal equations are singular for this camera_rgb data; "
    "pass a positive ridge (for example ridge=1e-4) to regularise the fit"
)


@dataclass(frozen=True)
class CCMResult:
    """Fitted colour-correction matrix and its training-patch error.

    Parameters
    ----------
    matrix:
        The ``(3, 3)`` float64 matrix in the convention
        ``target_rgb ~= camera_rgb @ matrix.T``. It is a fresh array, not a view
        of caller data.
    residual:
        RMS per-sample Euclidean RGB error on the training patches,
        ``sqrt(mean(sum((C @ M.T - T)**2, axis=1)))``. It measures the linear
        model before the output clamp, so a matrix that relies on clipping can
        still report a small residual.
    delta_e_before:
        Mean CIEDE2000 between the un-corrected camera RGB and the target RGB,
        with both interpreted as linear sRGB under D65 and negatives clipped.
        Camera RGB is not genuinely colorimetric sRGB, so treat the absolute
        value as indicative.
    delta_e_after:
        The same statistic after applying ``matrix`` to the camera RGB. The
        before/after comparison, not the absolute value, is the useful signal.

    The dataclass is frozen, but ``matrix`` is an ndarray and therefore not
    deep-frozen; compare the fields rather than whole results.
    """

    matrix: NDArray[np.float64]
    residual: float
    delta_e_before: float
    delta_e_after: float


def fit_ccm(
    camera_rgb: ArrayLike,
    target_rgb: ArrayLike,
    ridge: float = 1e-4,
    row_sum: bool = True,
) -> NDArray[np.float64]:
    """Fit a ridge-regularised 3x3 colour-correction matrix.

    Parameters
    ----------
    camera_rgb:
        ``(N, 3)`` linear camera RGB samples, one row per reference patch or
        colour measurement. Any real numeric array-like is accepted, including
        negative values (earlier linear stages can leave small excursions below
        zero). At least three samples are required to determine three
        coefficients per row, and the samples must jointly span all three
        colour directions unless ``row_sum`` supplies the missing one.
    target_rgb:
        ``(N, 3)`` reference RGB for the same samples, in the same order and
        with the same dtype rules. Each row corresponds to ``camera_rgb`` row
        by row.
    ridge:
        Non-negative finite ridge strength ``lambda``. The prior is the
        identity, so larger values pull the matrix toward "no correction" and
        suppress noise amplification in weakly excited colour directions.
        ``1e-4`` is the reference default; use a few 1e-2 for noisy low-light
        calibrations. ``ridge=0`` gives ordinary least squares and is accepted
        whenever the resulting system is nonsingular; rank-deficient
        ``camera_rgb`` then raises and asks for a positive ridge.
    row_sum:
        When true (the default), impose ``sum_j M[i, j] = 1`` exactly via the
        KKT system, preserving neutrality. When false, fit the unconstrained
        ridged least squares (ordinary least squares when ``ridge=0``).

    Returns
    -------
    A new C-contiguous float64 ``(3, 3)`` matrix satisfying
    ``target_rgb ~= camera_rgb @ matrix.T``. When ``row_sum`` is true each row
    sums to one to machine precision.

    Raises
    ------
    ValueError
        If the samples are not 2-D ``(N, 3)`` real numeric finite arrays with
        the same shape and ``N >= 3``, if ``ridge`` is not a finite real
        number at least zero (booleans are rejected), if ``row_sum`` is not a
        bool, or if the normal-equation system is singular. A singular fit
        would otherwise return huge coefficients or NaNs, which silently
        destroy every colour downstream.

    Notes
    -----
    The normal equations are ``(C^T C + lambda I) m_i = C^T t_i + lambda e_i``;
    the identity prior is what makes the shrinkage direction "toward no
    correction". With the constraint, the KKT matrix is assembled and solved
    directly, so the row sums do not depend on a penalty weight. All arithmetic
    is float64 and no global random state is touched, so identical inputs give
    bit-identical outputs within one environment.
    """
    camera = _as_float64_samples(camera_rgb, "camera_rgb")
    target = _as_float64_samples(target_rgb, "target_rgb")
    _require_matching_samples(camera, target)
    ridge_value = _validate_ridge(ridge)
    constrained = _validate_row_sum(row_sum)

    with np.errstate(over="ignore", invalid="ignore"):
        gram = camera.T @ camera
        rhs = camera.T @ target
    if not (np.all(np.isfinite(gram)) and np.all(np.isfinite(rhs))):
        raise ValueError(
            "camera_rgb or target_rgb values are too large for the CCM normal equations; "
            "the products overflowed float64"
        )

    prior = np.eye(3, dtype=np.float64)
    normal = gram + ridge_value * prior
    rhs = rhs + ridge_value * prior

    rows = [_solve_row(normal, rhs[:, index], constrained) for index in range(3)]
    matrix = np.stack(rows, axis=0)
    if constrained:
        _check_row_sums(matrix)
    return np.ascontiguousarray(matrix, dtype=np.float64)


def apply_ccm(rgb: ArrayLike, matrix: ArrayLike) -> NDArray[np.float32]:
    """Apply a colour-correction matrix to linear RGB, clamping at zero.

    Parameters
    ----------
    rgb:
        Any real numeric array-like whose trailing axis is 3, in R, G, B order:
        a single ``(3,)`` colour, ``(N, 3)`` samples, or an ``(H, W, 3)``
        image, including larger leading shapes. Values must be finite, but
        negatives are accepted because earlier linear stages can leave small
        excursions below zero. The input is never modified.
    matrix:
        A finite real ``(3, 3)`` matrix in the convention
        ``target_rgb ~= rgb @ matrix.T``, as returned by :func:`fit_ccm`.

    Returns
    -------
    A new C-contiguous float32 array with the same shape as ``rgb``, equal to
    ``max(rgb @ matrix.T, 0)``. Clipping negatives is the stage's only
    non-linearity; positive values are preserved without a gamma curve or tone
    map. The input is taken to float32 first and then accumulated in float64,
    so the result is rounded to float32 once at the end.

    Raises
    ------
    ValueError
        If ``rgb`` has no trailing axis of 3, contains bool, complex, or
        non-finite values, or does not fit in float32; if ``matrix`` is not a
        finite real ``(3, 3)`` array; or if the product overflows float32.
        Each condition would otherwise produce silently wrong or non-finite
        pixels.

    Notes
    -----
    A matrix with negative off-diagonal entries drives saturated or
    out-of-gamut colours below zero; the clamp keeps the image physical but
    discards hue information in clipped pixels. Apply this stage only to
    linear-light data: on gamma-encoded values the matrix mixes a non-linear
    function of radiance and the correction is invalid.
    """
    values = _as_float32_rgb(rgb, "rgb")
    checked = _as_float64_matrix(matrix, "matrix")

    with np.errstate(over="ignore", invalid="ignore"):
        product = values.astype(np.float64) @ checked.T
    if not np.all(np.isfinite(product)):
        raise ValueError(
            "apply_ccm produced non-finite values; the matrix or input exceeds float32 range"
        )
    clipped = np.maximum(product, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.ascontiguousarray(clipped, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError(
            "apply_ccm produced non-finite values; the matrix or input exceeds float32 range"
        )
    return output


def evaluate_ccm(
    camera_rgb: ArrayLike,
    target_rgb: ArrayLike,
    matrix: ArrayLike,
) -> CCMResult:
    """Measure a colour-correction matrix on reference patches.

    Parameters
    ----------
    camera_rgb:
        ``(N, 3)`` linear camera RGB samples, with the same validation rules as
        :func:`fit_ccm`: real numeric, finite, and at least three samples.
        Negative values are allowed.
    target_rgb:
        ``(N, 3)`` reference RGB for the same samples.
    matrix:
        The finite real ``(3, 3)`` matrix to evaluate, typically the output of
        :func:`fit_ccm` or a manual calibration.

    Returns
    -------
    A :class:`CCMResult` holding a copy of ``matrix``, the RMS linear-model
    residual, and the mean CIEDE2000 before and after applying the matrix. The
    "after" values come from :func:`apply_ccm`, so they include the output
    clamp.

    Raises
    ------
    ValueError
        If the samples or matrix fail the same validation as :func:`fit_ccm`
        and :func:`apply_ccm`, or if the two sample arrays differ in shape.

    Notes
    -----
    ΔE is computed with ``colour-science`` by interpreting both RGB sets as
    linear sRGB under D65 (converted directly to XYZ, skipping the sRGB
    transfer function) and clipping negatives to zero for the diagnostic only.
    Camera RGB is not actually colorimetric sRGB, so absolute ΔE values are
    indicative; the before/after comparison is the usable signal. The residual
    is measured before the clamp, matching the fitted linear model.
    """
    camera = _as_float64_samples(camera_rgb, "camera_rgb")
    target = _as_float64_samples(target_rgb, "target_rgb")
    _require_matching_samples(camera, target)
    colour_matrix = _as_float64_matrix(matrix, "matrix")

    after = apply_ccm(camera, colour_matrix)
    with np.errstate(over="ignore", invalid="ignore"):
        linear = camera @ colour_matrix.T
    residual = float(np.sqrt(np.mean(np.sum((linear - target) ** 2, axis=1))))

    return CCMResult(
        matrix=colour_matrix,
        residual=residual,
        delta_e_before=_mean_delta_e(camera, target),
        delta_e_after=_mean_delta_e(np.asarray(after, dtype=np.float64), target),
    )


def _as_float64_samples(values: Any, field: str) -> NDArray[np.float64]:
    """Return ``values`` as a finite real ``(N, 3)`` float64 sample matrix."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a 2-D (N, 3) array of linear RGB samples; {exc}"
        ) from exc
    if array.ndim != 2:
        raise ValueError(f"{field} must be a 2-D (N, 3) array; got shape {array.shape}")
    if array.shape[1] != 3:
        raise ValueError(
            f"{field} must have exactly three channels in R, G, B order; got shape {array.shape}"
        )
    if array.shape[0] < 3:
        raise ValueError(
            f"{field} must contain at least three samples to determine a 3x3 "
            f"colour-correction matrix; got {array.shape[0]}"
        )
    if (
        array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{field} must contain real numeric values; got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{field} values must be finite; NaN or infinity cannot be fitted or evaluated"
        )
    return np.asarray(array, dtype=np.float64)


def _require_matching_samples(
    camera: NDArray[np.float64],
    target: NDArray[np.float64],
) -> None:
    """Reject camera and target patches whose shapes differ."""
    if camera.shape != target.shape:
        raise ValueError(
            "camera_rgb and target_rgb must have the same (N, 3) shape; "
            f"got {camera.shape} and {target.shape}"
        )


def _as_float32_rgb(values: Any, field: str) -> NDArray[np.float32]:
    """Return ``values`` as a finite float32 array whose trailing axis is 3."""
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
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.ascontiguousarray(array, dtype=np.float32)
    if not np.all(np.isfinite(converted)):
        raise ValueError(
            f"{field} values must fit in float32; conversion produced a non-finite value"
        )
    return converted


def _as_float64_matrix(matrix: Any, field: str) -> NDArray[np.float64]:
    """Return a new finite real ``(3, 3)`` float64 matrix.

    A copy is returned so a :class:`CCMResult` never aliases caller memory and
    a later mutation of the caller's array cannot change a stored result.
    """
    try:
        array = np.asarray(matrix)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite real 3x3 matrix; {exc}") from exc
    if array.shape != (3, 3):
        raise ValueError(f"{field} must be a 3x3 matrix; got shape {array.shape}")
    if (
        array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{field} must contain real numeric values; got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} entries must be finite; NaN or infinity is not a colour matrix")
    return np.array(array, dtype=np.float64, copy=True)


def _validate_ridge(ridge: Any) -> float:
    """Return the ridge strength as a finite float greater than or equal to zero."""
    if isinstance(ridge, (bool, np.bool_)) or not isinstance(
        ridge, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"ridge must be a finite real number greater than or equal to 0; got {ridge!r}"
        )
    value = float(ridge)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"ridge must be a finite real number greater than or equal to 0; got {ridge!r}"
        )
    return value


def _validate_row_sum(row_sum: Any) -> bool:
    """Return the row-sum flag, accepting only a bool-like value."""
    if not isinstance(row_sum, (bool, np.bool_)):
        raise ValueError(f"row_sum must be a bool; got {row_sum!r}")
    return bool(row_sum)


def _solve_row(
    normal: NDArray[np.float64],
    rhs: NDArray[np.float64],
    constrained: bool,
) -> NDArray[np.float64]:
    """Solve one row of the CCM fit, with or without the row-sum constraint.

    The constrained solve is the KKT system of the equality-constrained
    problem, not a quadratic penalty, so the constraint holds to machine
    precision regardless of the ridge. A rank-deficient system is reported as
    singular rather than solved, because a huge finite coefficient vector is
    just as destructive downstream as a NaN one.
    """
    if constrained:
        system = np.zeros((4, 4), dtype=np.float64)
        system[:3, :3] = normal
        system[:3, 3] = 1.0
        system[3, :3] = 1.0
        target = np.empty(4, dtype=np.float64)
        target[:3] = rhs
        target[3] = 1.0
    else:
        system = normal
        target = rhs

    try:
        if np.linalg.matrix_rank(system) < system.shape[0]:
            raise ValueError(_SINGULAR_MESSAGE)
        solution = np.linalg.solve(system, target)
    except np.linalg.LinAlgError as exc:
        raise ValueError(_SINGULAR_MESSAGE) from exc
    if not np.all(np.isfinite(solution)):
        raise ValueError(_SINGULAR_MESSAGE)
    # np.linalg.solve is typed generically over the floating dtype; the system
    # is float64, so the row is float64 as well.
    return np.asarray(solution[:3], dtype=np.float64)


def _check_row_sums(matrix: NDArray[np.float64]) -> None:
    """Reject a constrained fit whose rows do not sum to one.

    The KKT solve imposes the constraint exactly, so a material deviation means
    round-off from a numerically singular problem was amplified into the
    coefficients. Returning such a matrix would violate the documented
    neutral-preserving property.
    """
    deviations = np.abs(matrix.sum(axis=1) - 1.0)
    if np.any(deviations > _ROW_SUM_TOLERANCE):
        raise ValueError(
            "the CCM fit did not satisfy the row-sum constraint to machine precision; "
            "the problem is too ill-conditioned, so increase the ridge"
        )


def _mean_delta_e(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Return the mean CIEDE2000 between two ``(N, 3)`` linear sRGB sets.

    ``colour-science`` is imported here rather than at module scope because it
    is a sizeable dependency and most callers only fit or apply a matrix. Both
    inputs are interpreted as linear sRGB under D65: values go straight to XYZ
    with no transfer function, and negatives are clipped because the sRGB
    matrix is only defined for non-negative light. Camera RGB is not actually
    colorimetric sRGB, so the absolute value is indicative; the before/after
    difference is what a calibration report should quote.
    """
    from colour import CCS_ILLUMINANTS, RGB_to_XYZ, XYZ_to_Lab
    from colour.difference import delta_E_CIE2000
    from colour.models import RGB_COLOURSPACES

    illuminant = np.asarray(
        CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
        dtype=np.float64,
    )
    colourspace = RGB_COLOURSPACES["sRGB"]
    clipped_first = np.maximum(np.asarray(first, dtype=np.float64), 0.0)
    clipped_second = np.maximum(np.asarray(second, dtype=np.float64), 0.0)
    lab_first = XYZ_to_Lab(
        RGB_to_XYZ(clipped_first, colourspace, illuminant=illuminant),
        illuminant=illuminant,
    )
    lab_second = XYZ_to_Lab(
        RGB_to_XYZ(clipped_second, colourspace, illuminant=illuminant),
        illuminant=illuminant,
    )
    differences = np.asarray(delta_E_CIE2000(lab_first, lab_second), dtype=np.float64)
    return float(np.mean(differences))
