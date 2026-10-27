"""CFA demosaicing: reconstruct full-resolution linear RGB from a mosaic.

A colour image sensor places a colour filter over each photodiode, so every
pixel measures one spectral band and the other two channels must be inferred
from neighbouring samples. This module reconstructs those missing samples in
linear light, after black-level correction and white balance and before the
colour-correction matrix, denoise, and tone stages that expect three channels
per pixel. Demosaicing is spatial interpolation of radiance, not a colour
transform: an error here (a wrong tile origin, a swapped channel) propagates
through every later colour operation and cannot be undone.

The dominant consumer layout is the 2x2 Bayer tile, one red, two green, and one
blue sites named row-major from the top-left pixel (``RGGB``, ``BGGR``,
``GRBG``, ``GBRG``). ``RCCC`` is an automotive layout with one red and three
clear sites per tile; the clear channel is broadband and carries most of the
luminance, which is why it is used as a luma placeholder below.

Bilinear interpolation masks each colour plane, convolves it with a small
all-pass kernel, and reads the result at every pixel::

    y = H * x

For green it averages the four cross neighbours at red and blue sites; for red
and blue it averages the two same-colour neighbours along one axis at a green
site and the four diagonal same-colour neighbours at the opposite colour site.
It is cheap and exact for smooth fields, but the three channels are
interpolated independently, so an edge that falls on a red site but not on a
neighbouring blue site produces a colour fringe, a soft "zipper" along the
edge, and reduced sharpness.

Malvar, He, and Cutler (2004) replace those kernels with 5x5 gradient-corrected
filters. Each missing channel is estimated from a base interpolation plus a
correction proportional to the second derivative of the measured green plane,
so high-frequency detail measured at green sites refines red and blue instead
of being thrown away. The kernels are::

    G at R/B:      [[0, 0, -1, 0, 0],     R/B at G:    [[0, 0, 0.5, 0, 0],
                    [0, 0,  2, 0, 0],                  [0, -1, 0, -1, 0],
                    [-1, 2, 4, 2, -1],                 [-1, 4, 5, 4, -1],
                    [0, 0,  2, 0, 0],                  [0, -1, 0, -1, 0],
                    [0, 0, -1, 0, 0]] / 8               [0, 0, 0.5, 0, 0]] / 8

    R at B / B at R: [[0, 0, -1.5, 0, 0],
                      [0, 2, 0, 2, 0],
                      [-1.5, 0, 6, 0, -1.5],
                      [0, 2, 0, 2, 0],
                      [0, 0, -1.5, 0, 0]] / 8

with the transposed red-at-green kernel used for the opposite green phase. The
kernels sum to one, so a flat field is reproduced exactly, and the correction
term vanishes on a linear ramp. MHC costs roughly 2-3 times bilinear and
markedly reduces zippering and false colour at edges; residual artefacts remain
on fine chromatic textures, diagonal lines near the Nyquist limit can show a
"maze" pattern, and very noisy data benefit from denoising first.

Frame borders need a boundary rule. Bilinear and MHC need samples one or two
pixels outside the frame; this implementation extends each axis by symmetric
reflection (the ``"reflect"`` mode of ``scipy.ndimage``), the same rule the
``colour-demosaicing`` reference uses. Reflection double-counts same-colour
sites at the border (bilinear) and mixes colours there (MHC), so a constant
scene is not exactly constant in the outermost rows and columns even though
the interior is. That is a property of the reference boundary mode, not of the
scene: the validation tests compare the full frame against the reference and
check DC preservation in the interior. Production pipelines should crop or
mirror-fill a few border pixels before measuring image quality.

The RCCC baseline is deliberately simple and explicitly not colour accurate:
the sparse red plane is interpolated by weight-normalised bilinear interpolation
(so flat red fields are exact even at the border), the measured clear plane is
kept at its own sites, clear values at red sites are averaged from the cardinal
clear neighbours, and the result is returned as ``(R, C, C)``. The clear
channel acts as a luma placeholder for green and blue, so the output has the
right shape, linearity, and DC behaviour for pipeline smoke tests but no usable
chroma. Automotive RCCC demosaicing normally exploits the clear channel's
higher signal-to-noise ratio for luma and reconstructs chroma from the sparse
red samples plus a spectral model; that is deferred.

Typical sensor sizes span roughly 0.3 MP (VGA) to 200 MP, so images run from a
few hundred thousand to hundreds of millions of pixels; Bayer remains the
dominant consumer and industrial layout, RCCC appears in automotive forward
cameras, and monochrome sensors skip this stage entirely. Kernel support is
therefore fixed and small, but the convolution cost is proportional to pixel
count and matters at video rates.

All computation happens in linear float32 light. The stage does not apply a
gamma curve or tone map, does not clip, and preserves negative values, because
earlier linear stages (black-level correction, white balance) can leave small
excursions below zero and the tone stage owns the display range. The output is
a new C-contiguous float32 array in R, G, B order; the input is never mutated.
Mono input is a 2-D pass-through copy: the pipeline keeps it single-channel
and replicates only at the display boundary. Scaling the input scales the
output (``demosaic(a * x) == a * demosaic(x)``) because every operation is a
linear combination of samples.

Failure artefacts guarded here: interpolating across CFA colours (false colour
and zippering), a transposed or offset Bayer tile (red/blue swap), applying
demosaicing to gamma-encoded data (interpolation in the wrong domain),
treating an RCCC or mono mosaic as Bayer (silently wrong channels or shapes),
returning the input buffer or mutating it (aliasing a later stage's write),
and accepting non-2-D, empty, bool, complex, or non-finite data (poisoned
statistics downstream). Method/CFA mismatches raise rather than guessing a
layout.
"""

from __future__ import annotations

from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.core.frames import CFAPattern
from camsim.sensor.cfa import cfa_masks

__all__ = [
    "DemosaicMethod",
    "demosaic",
    "demosaic_bilinear",
    "demosaic_mhc",
    "demosaic_rccc_baseline",
]

DemosaicMethod: TypeAlias = Literal["bilinear", "mhc", "rccc_baseline"]
"""Methods accepted by :func:`demosaic`.

``"bilinear"`` and ``"mhc"`` reconstruct Bayer mosaics; ``"rccc_baseline"`` is
the documented shape/linearity placeholder for RCCC mosaics.
"""

_METHODS: Final[tuple[str, ...]] = ("bilinear", "mhc", "rccc_baseline")

# Bilinear green kernel: at a red or blue site the centre is zero and the four
# cross neighbours (all green) receive 1/4 each. At a green site the centre
# weight of one returns the measured sample unchanged.
_BILINEAR_GREEN: Final[NDArray[np.float32]] = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 4.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
) / np.float32(4.0)

# Bilinear red/blue kernel. On the masked plane the centre returns a measured
# sample; at a green site the two same-colour neighbours along one axis receive
# 2/4 each, and at the opposite colour site the four diagonal same-colour
# neighbours receive 1/4 each. The other taps land on zeroes of the mask.
_BILINEAR_RED_BLUE: Final[NDArray[np.float32]] = np.array(
    [
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 2.0],
        [1.0, 2.0, 1.0],
    ],
    dtype=np.float32,
) / np.float32(4.0)

# Malvar-He-Cutler (2004) kernels. They are applied to the raw mosaic, not to a
# masked plane, because every tap reads a measured sample of some colour.
_MHC_GREEN_AT_RED_BLUE: Final[NDArray[np.float32]] = np.array(
    [
        [0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0, 0.0],
        [-1.0, 2.0, 4.0, 2.0, -1.0],
        [0.0, 0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
) / np.float32(8.0)

_MHC_RED_BLUE_AT_GREEN: Final[NDArray[np.float32]] = np.array(
    [
        [0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, -1.0, 0.0, -1.0, 0.0],
        [-1.0, 4.0, 5.0, 4.0, -1.0],
        [0.0, -1.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0],
    ],
    dtype=np.float32,
) / np.float32(8.0)

# The transposed kernel serves the green sites on the opposite row phase: red
# and blue swap their roles when the tile orientation is transposed.
_MHC_RED_BLUE_AT_GREEN_TRANSPOSED: Final[NDArray[np.float32]] = np.ascontiguousarray(
    _MHC_RED_BLUE_AT_GREEN.T
)

_MHC_RED_AT_BLUE_OR_BLUE_AT_RED: Final[NDArray[np.float32]] = np.array(
    [
        [0.0, 0.0, -1.5, 0.0, 0.0],
        [0.0, 2.0, 0.0, 2.0, 0.0],
        [-1.5, 0.0, 6.0, 0.0, -1.5],
        [0.0, 2.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, -1.5, 0.0, 0.0],
    ],
    dtype=np.float32,
) / np.float32(8.0)


def demosaic_bilinear(
    raw: ArrayLike,
    cfa: CFAPattern | str,
) -> NDArray[np.float32]:
    """Interpolate a Bayer mosaic with 3x3 bilinear kernels.

    Parameters
    ----------
    raw:
        A 2-D ``(H, W)`` mosaic in linear sensor units, one CFA colour per
        site. Any real numeric array-like is accepted, including uint16 ADC
        codes and negative values; the input is converted to float32, never
        modified, and never clipped. ``MONO`` is returned as a 2-D copy.
    cfa:
        The CFA layout, as a :class:`~camsim.core.frames.CFAPattern` or its
        string value. Bayer patterns produce RGB; ``MONO`` bypasses; ``RCCC``
        is rejected because it has no bilinear colour model in this stage.

    Returns
    -------
    A new C-contiguous float32 array of shape ``(H, W, 3)`` in R, G, B order
    for a Bayer mosaic, or ``(H, W)`` for a mono pass-through. Values are
    linear and unclipped.

    Raises
    ------
    ValueError
        If ``raw`` is not a non-empty 2-D real numeric array with finite
        values, if ``cfa`` is not a known pattern, or if a Bayer pattern is not
        used. Each condition would otherwise produce silently wrong channels.

    Notes
    -----
    Missing samples are the convolution of each masked colour plane with
    :data:`_BILINEAR_GREEN` or :data:`_BILINEAR_RED_BLUE`. Borders use
    symmetric reflection, matching ``colour-demosaicing``; at the outermost
    rows and columns that reflection double-counts same-colour sites, so a flat
    scene is only exactly flat in the interior. The operation is linear in the
    mosaic values.
    """
    values = _as_float32_mosaic(raw)
    pattern = _coerce_pattern(cfa)
    if pattern is CFAPattern.MONO:
        # A mono sensor has no colour mosaic: keep the single channel and let
        # the display boundary decide whether to replicate it.
        return values.copy()
    if pattern is CFAPattern.RCCC:
        raise ValueError(
            "RCCC mosaics require the 'rccc_baseline' method; bilinear has no RCCC colour model"
        )

    masks = cfa_masks(pattern, values.shape)
    red = _convolve_reflect(values * masks["R"], _BILINEAR_RED_BLUE)
    green = _convolve_reflect(values * masks["G"], _BILINEAR_GREEN)
    blue = _convolve_reflect(values * masks["B"], _BILINEAR_RED_BLUE)
    return _stack_rgb(red, green, blue)


def demosaic_mhc(
    raw: ArrayLike,
    cfa: CFAPattern | str,
) -> NDArray[np.float32]:
    """Interpolate a Bayer mosaic with Malvar-He-Cutler 5x5 kernels.

    Parameters
    ----------
    raw:
        A 2-D ``(H, W)`` mosaic in linear sensor units; the same input contract
        as :func:`demosaic_bilinear`. ``MONO`` is returned as a 2-D copy.
    cfa:
        The CFA layout, as a :class:`~camsim.core.frames.CFAPattern` or its
        string value. Bayer patterns produce RGB; ``MONO`` bypasses; ``RCCC``
        is rejected because the 5x5 kernels assume a Bayer lattice.

    Returns
    -------
    A new C-contiguous float32 array of shape ``(H, W, 3)`` in R, G, B order
    for a Bayer mosaic, or ``(H, W)`` for a mono pass-through. Values are
    linear and unclipped.

    Raises
    ------
    ValueError
        If ``raw`` is not a non-empty 2-D real numeric array with finite
        values, if ``cfa`` is not a known pattern, or if a Bayer pattern is not
        used.

    Notes
    -----
    The gradient-corrected kernels of Malvar, He, and Cutler (2004) are applied
    directly to the mosaic; the measured sample at each site is kept and the
    two missing channels are estimated from the encoded green detail. Borders
    use the same symmetric reflection as the bilinear stage, so the reference's
    edge behaviour is reproduced exactly. The operation is linear in the
    mosaic values.
    """
    values = _as_float32_mosaic(raw)
    pattern = _coerce_pattern(cfa)
    if pattern is CFAPattern.MONO:
        return values.copy()
    if pattern is CFAPattern.RCCC:
        raise ValueError(
            "RCCC mosaics require the 'rccc_baseline' method; "
            "the MHC kernels assume a Bayer lattice"
        )

    masks = cfa_masks(pattern, values.shape)
    red_mask = masks["R"]
    green_mask = masks["G"]
    blue_mask = masks["B"]

    # Measured samples are carried through unchanged; only missing channels
    # take a convolved estimate.
    red = values * red_mask
    green = values * green_mask
    blue = values * blue_mask

    green = np.where(
        red_mask | blue_mask,
        _convolve_reflect(values, _MHC_GREEN_AT_RED_BLUE),
        green,
    )
    at_green_row = _convolve_reflect(values, _MHC_RED_BLUE_AT_GREEN)
    at_green_column = _convolve_reflect(values, _MHC_RED_BLUE_AT_GREEN_TRANSPOSED)
    at_opposite = _convolve_reflect(values, _MHC_RED_AT_BLUE_OR_BLUE_AT_RED)

    # A row (column) "is red" when it contains any red site: on a Bayer lattice
    # the opposite colour then occupies the other phase. These flags select the
    # kernel whose same-colour taps line up with the measured sites.
    red_rows = np.asarray(red_mask.any(axis=1), dtype=np.bool_)[:, np.newaxis]
    red_columns = np.asarray(red_mask.any(axis=0), dtype=np.bool_)[np.newaxis, :]
    blue_rows = np.asarray(blue_mask.any(axis=1), dtype=np.bool_)[:, np.newaxis]
    blue_columns = np.asarray(blue_mask.any(axis=0), dtype=np.bool_)[np.newaxis, :]

    red = np.where(red_rows & blue_columns, at_green_row, red)
    red = np.where(blue_rows & red_columns, at_green_column, red)
    blue = np.where(blue_rows & red_columns, at_green_row, blue)
    blue = np.where(red_rows & blue_columns, at_green_column, blue)
    red = np.where(blue_rows & blue_columns, at_opposite, red)
    blue = np.where(red_rows & red_columns, at_opposite, blue)
    return _stack_rgb(red, green, blue)


def demosaic_rccc_baseline(raw: ArrayLike) -> NDArray[np.float32]:
    """Reconstruct an RCCC mosaic with a documented luma-placeholder baseline.

    Parameters
    ----------
    raw:
        A 2-D ``(H, W)`` RCCC mosaic: red at even row/column sites (starting
        at ``(0, 0)``) and clear (``C``) elsewhere, as
        :class:`~camsim.core.frames.CFAPattern.RCCC` defines. The image must
        contain at least one clear sample, so a 1x1 crop is rejected.

    Returns
    -------
    A new C-contiguous float32 array of shape ``(H, W, 3)`` whose channels are
    ``(R, C, C)``. The red plane is weight-normalised bilinear interpolation of
    the sparse red sites; the clear plane is measured at clear sites and
    averaged from the available cardinal clear neighbours at red sites. Values
    are linear and unclipped.

    Raises
    ------
    ValueError
        If ``raw`` is not a non-empty 2-D real numeric array with finite
        values, or if it contains no clear sample.

    Notes
    -----
    This is a **baseline for shape, linearity, DC, and smoke tests only**. It
    is not colour accurate: clear, green, and blue are different spectral
    responses, and the ``(R, C, C)`` output cannot be used for colour-difference
    metrics. A production RCCC pipeline reconstructs chroma from the sparse red
    plane and a spectral model, and uses the clear channel's higher SNR for
    luminance; that work is deferred. Weight normalisation keeps a flat red
    field exact even at the border, because the numerator and denominator are
    extended with the same reflected sample weights.
    """
    values = _as_float32_mosaic(raw)
    masks = cfa_masks(CFAPattern.RCCC, values.shape)
    red_mask = masks["R"]
    clear_mask = masks["C"]
    if not clear_mask.any():
        raise ValueError(
            "RCCC baseline requires at least one clear (C) sample; "
            "a 1x1 crop is smaller than the RCCC tile"
        )

    red_plane = _interpolate_sparse_plane(values, red_mask, _BILINEAR_RED_BLUE)
    clear_plane = _fill_clear_at_red_sites(values, clear_mask)
    return _stack_rgb(red_plane, clear_plane, clear_plane)


def demosaic(
    raw: ArrayLike,
    cfa: CFAPattern | str,
    method: DemosaicMethod | str,
) -> NDArray[np.float32]:
    """Dispatch demosaicing for a Bayer, mono, or RCCC mosaic.

    Parameters
    ----------
    raw:
        A 2-D ``(H, W)`` linear mosaic in any real numeric array-like; the
        input is converted to float32, returned in a new array, and never
        modified or clipped.
    cfa:
        The CFA layout, as a :class:`~camsim.core.frames.CFAPattern` or its
        string value.
    method:
        ``"bilinear"`` or ``"mhc"`` for a Bayer mosaic, ``"rccc_baseline"``
        for RCCC. A mono mosaic bypasses reconstruction regardless of method.

    Returns
    -------
    A new C-contiguous float32 array: ``(H, W, 3)`` in R, G, B order for a
    Bayer mosaic and for the RCCC baseline, ``(H, W)`` for a mono pass-through.
    Values are linear and unclipped.

    Raises
    ------
    ValueError
        If ``raw`` or ``cfa`` is invalid, if ``method`` is not one of the three
        names, if an RCCC mosaic is paired with a Bayer method, or if a Bayer
        mosaic is paired with ``"rccc_baseline"``. Each mismatch would silently
        reconstruct the wrong channels.

    Notes
    -----
    The method is validated before the mono bypass, so a typo cannot hide
    behind a pass-through. The RCCC baseline is a shape/linearity placeholder
    and is not colour accurate; see :func:`demosaic_rccc_baseline`.
    """
    values = _as_float32_mosaic(raw)
    pattern = _coerce_pattern(cfa)
    resolved = _validate_method(method)

    if pattern is CFAPattern.MONO:
        return values.copy()
    if pattern is CFAPattern.RCCC:
        if resolved != "rccc_baseline":
            raise ValueError(f"RCCC mosaics require method 'rccc_baseline'; got {resolved!r}")
        return demosaic_rccc_baseline(values)
    if resolved == "rccc_baseline":
        raise ValueError(
            "method 'rccc_baseline' is only valid for RCCC mosaics; "
            f"got the Bayer pattern {pattern.value!r}"
        )
    if resolved == "bilinear":
        return demosaic_bilinear(values, pattern)
    return demosaic_mhc(values, pattern)


def _as_float32_mosaic(raw: Any, field: str = "raw") -> NDArray[np.float32]:
    """Return ``raw`` as a finite 2-D float32 mosaic without modifying it."""
    try:
        array = np.asarray(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a 2-D numeric CFA mosaic; {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"{field} must be a 2-D (H, W) CFA mosaic; got shape {array.shape}")
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
        converted = np.ascontiguousarray(array, dtype=np.float32)
    if not np.all(np.isfinite(converted)):
        raise ValueError(
            f"{field} values must fit in float32; conversion produced a non-finite value"
        )
    return converted


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


def _validate_method(method: Any) -> DemosaicMethod:
    """Return a supported demosaicing method name."""
    if not isinstance(method, str) or method not in _METHODS:
        allowed = ", ".join(_METHODS)
        raise ValueError(f"method must be one of {allowed}; got {method!r}")
    return cast(DemosaicMethod, method)


def _reflect_indices(size: int, pad: int) -> NDArray[np.intp]:
    """Return reflected indices for ``pad`` samples on each side of an axis.

    The mapping is symmetric reflection about the boundary (``... c b | a b c
    ...``), which repeats the edge sample. It matches the ``"reflect"`` mode of
    ``scipy.ndimage`` for any pad width, including images shorter than twice the
    pad, and degenerates to index zero for a length-one axis.
    """
    indices = np.arange(-pad, size + pad)
    period = 2 * size
    wrapped = np.mod(indices, period)
    reflected = np.where(wrapped >= size, period - 1 - wrapped, wrapped)
    return np.asarray(reflected, dtype=np.intp)


def _convolve_reflect(
    image: NDArray[np.float32],
    kernel: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Convolve a 2-D float32 image with symmetric boundary reflection.

    This reproduces ``scipy.ndimage.convolve(image, kernel, mode="reflect")``,
    including its kernel reversal and edge extension, using NumPy only. All
    production kernels are odd-sized and centred; the helper is private and
    assumes that invariant.
    """
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    row_pad = kernel_height // 2
    column_pad = kernel_width // 2

    rows = _reflect_indices(height, row_pad)
    columns = _reflect_indices(width, column_pad)
    padded = image[rows[:, np.newaxis], columns[np.newaxis, :]]

    # ndimage.convolve reverses the kernel relative to a correlation; flipping
    # once here keeps the tap order identical to the reference.
    flipped = kernel[::-1, ::-1]
    output = np.zeros_like(image, dtype=np.float32)
    for row in range(kernel_height):
        for column in range(kernel_width):
            coefficient = float(flipped[row, column])
            if coefficient == 0.0:
                continue
            output += (
                np.float32(coefficient)
                * padded[
                    row : row + height,
                    column : column + width,
                ]
            )
    return output


def _interpolate_sparse_plane(
    values: NDArray[np.float32],
    mask: NDArray[np.bool_],
    kernel: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Interpolate a plane from its masked samples with weight normalisation.

    Dividing the convolved samples by the convolved mask makes the estimate a
    weighted average of the same-colour samples that actually exist near each
    site. Without that division a flat field would be over-counted wherever
    reflection repeats a sample, which is exactly the border artefact the Bayer
    bilinear stage inherits from its reference.
    """
    numerator = _convolve_reflect(values * mask, kernel)
    weights = _convolve_reflect(mask.astype(np.float32), kernel)
    return numerator / weights


def _fill_clear_at_red_sites(
    values: NDArray[np.float32],
    clear_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Return a plane measured at clear sites and averaged at red sites.

    Each red site takes the mean of the cardinal clear neighbours that lie
    inside the frame. At an interior site all four exist; a border site uses the
    two or three that do. No reflected value can land on another red site here,
    because a single-axis reflection preserves row or column parity.
    """
    total = np.zeros_like(values)
    count = np.zeros_like(values)

    total[:-1, :] += values[1:, :] * clear_mask[1:, :]
    count[:-1, :] += clear_mask[1:, :]
    total[1:, :] += values[:-1, :] * clear_mask[:-1, :]
    count[1:, :] += clear_mask[:-1, :]
    total[:, :-1] += values[:, 1:] * clear_mask[:, 1:]
    count[:, :-1] += clear_mask[:, 1:]
    total[:, 1:] += values[:, :-1] * clear_mask[:, :-1]
    count[:, 1:] += clear_mask[:, :-1]

    output = values.copy()
    red_sites = ~clear_mask
    output[red_sites] = total[red_sites] / count[red_sites]
    return output


def _stack_rgb(
    red: NDArray[np.float32],
    green: NDArray[np.float32],
    blue: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Stack three planes into a new C-contiguous float32 RGB array."""
    return np.ascontiguousarray(
        np.stack((red, green, blue), axis=-1),
        dtype=np.float32,
    )
