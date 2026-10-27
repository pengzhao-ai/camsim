"""Pixel binning as charge aggregation or post-ADC code averaging.

Binning trades spatial resolution for signal-to-noise ratio, and the two
readout architectures that implement it are not interchangeable:

**Analog binning** connects several same-colour photodiodes to one sense node
and transfers their charge before sampling. The combined well holds
``factor**2`` times the single-pixel charge, one read event adds its noise to
the whole bucket, and shot noise still follows the summed charge. A 2x2 analog
bin therefore quadruples the signal and full well while leaving read noise at
one pixel's value, which is why it buys roughly one stop of low-light SNR.

**Digital binning** reads and quantizes every pixel and averages the resulting
codes in the digital domain. Signal stays at the single-pixel scale, read noise
is averaged down by ``sqrt(factor**2)``, and any full-well or ADC clipping has
already happened at native resolution. Digital binning can therefore not
recover highlights that saturate before the average; it only reduces temporal
noise after the fact.

Binning is a *per-CFA-plane* operation in camsim. A blind 2x2 sum of the mosaic
would add red, green, and blue charge into one value and destroy the CFA
pattern the ISP needs. Each colour plane is aggregated independently and the
planes are re-interleaved with the original tile, so a binned RGGB frame is
still RGGB at half resolution. For Bayer and RCCC layouts the factor must
divide both image dimensions; otherwise a partial block would silently mix
different pixel areas.

Real-world ranges: 1x1 (no binning) through 4x4 in consumer modes; 2x2 is the
usual low-light setting, and some quad-Bayer parts bin 2x2 in the charge domain
to emulate a larger pixel while retaining a full-resolution mode.

Failure artefacts guarded here: summing a mosaic across CFA colours (colour
loss), averaging in the charge domain but claiming analog gain (wrong noise
model), a factor that does not divide the shape (unequal pixel areas),
forgetting to scale the full well by ``factor**2`` (premature saturation),
scaling read noise by the same factor (over-penalising analog binning), and
leaving defect/PDAF masks at native resolution while data shrink (mask/shape
mismatch). Each condition raises with the offending quantity named.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.config.schema import BinningConfig

__all__ = [
    "BinningMode",
    "BinningPlan",
    "binning_plan",
    "mean_blocks",
    "reduce_mask",
    "sum_blocks",
]

BinningMode: TypeAlias = Literal["analog", "digital"]
"""The two supported binning domains."""


@dataclass(frozen=True)
class BinningPlan:
    """Resolved geometry for one binning configuration.

    Parameters
    ----------
    factor:
        Linear binning factor; one output pixel spans ``factor x factor``
        native pixels of the same CFA colour.
    mode:
        ``"analog"`` (charge-domain sum before sampling) or ``"digital"``
        (post-ADC code average).
    input_shape:
        Native ``(height, width)`` in pixels.
    output_shape:
        Binned ``(height // factor, width // factor)`` in pixels.

    ``pixel_pitch_scale`` is the factor by which the output pixel pitch grows;
    the sensor writes ``pixel_pitch_um * factor`` into the RAW frame.
    """

    factor: int
    mode: BinningMode
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]

    @property
    def pixel_pitch_scale(self) -> int:
        """Return the output pitch multiplier (equal to ``factor``)."""
        return self.factor


def binning_plan(config: BinningConfig, shape: Sequence[int]) -> BinningPlan:
    """Validate a binning configuration against a native sensor shape.

    Parameters
    ----------
    config:
        Validated :class:`~camsim.config.schema.BinningConfig`.
    shape:
        Native ``(height, width)`` in pixels.

    Returns
    -------
    A :class:`BinningPlan` whose output shape is the native shape divided by
    the factor.

    Raises
    ------
    ValueError
        If the configuration type is wrong, the shape is not two positive
        integers, the mode is unsupported, or the factor does not divide both
        dimensions. A partial block would aggregate unequal numbers of charge
        carriers and is rejected rather than padded.
    """
    if not isinstance(config, BinningConfig):
        raise ValueError(f"config must be a BinningConfig; got {type(config).__name__}")
    height, width = _validate_shape(shape)
    factor = _validate_factor(config.factor, (height, width))
    mode = config.mode
    if mode not in ("analog", "digital"):
        raise ValueError(f"binning mode must be 'analog' or 'digital'; got {mode!r}")
    return BinningPlan(
        factor=factor,
        mode=mode,
        input_shape=(height, width),
        output_shape=(height // factor, width // factor),
    )


def sum_blocks(values: ArrayLike, factor: int) -> NDArray[np.float64]:
    """Sum non-overlapping ``factor x factor`` blocks of a 2-D plane.

    This is the analog-binning arithmetic: charge means (or samples) from
    same-colour pixels are added before the read event. The output is float64
    even for integer input so a caller can add a fractional overflow term.
    """
    array = _validate_plane(values, "values")
    resolved = _validate_factor(factor, array.shape)
    height, width = array.shape
    blocks = array.reshape(
        height // resolved,
        resolved,
        width // resolved,
        resolved,
    )
    return np.asarray(blocks.sum(axis=(1, 3)), dtype=np.float64)


def mean_blocks(values: ArrayLike, factor: int) -> NDArray[np.float64]:
    """Average non-overlapping ``factor x factor`` blocks of a 2-D plane.

    This is the digital-binning arithmetic applied to post-ADC codes (or to
    any already-sampled plane). Averaging rather than summing keeps the signal
    at the single-pixel scale; read noise is reduced by ``sqrt(factor**2)``.
    """
    array = _validate_plane(values, "values")
    resolved = _validate_factor(factor, array.shape)
    height, width = array.shape
    blocks = array.reshape(
        height // resolved,
        resolved,
        width // resolved,
        resolved,
    )
    return np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float64)


def reduce_mask(mask: ArrayLike, factor: int) -> NDArray[np.bool_]:
    """Collapse a native boolean mask onto binned pixels by logical OR.

    A binned pixel is defective or a PDAF site if *any* native pixel in its
    block is flagged. Dropping a flag because the majority of the block is
    healthy would hide a dead pixel from defect correction.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be a 2-D array; got shape {array.shape}")
    if array.dtype != np.bool_:
        raise ValueError(f"mask must have dtype bool; got dtype {array.dtype}")
    if array.size == 0:
        raise ValueError("mask must contain at least one pixel")
    height, width = array.shape
    resolved = _validate_factor(factor, (height, width))
    blocks = array.reshape(
        height // resolved,
        resolved,
        width // resolved,
        resolved,
    )
    return cast(NDArray[np.bool_], blocks.any(axis=(1, 3)))


def _validate_shape(shape: Sequence[int]) -> tuple[int, int]:
    """Return a ``(height, width)`` pair of positive Python integers."""
    if isinstance(shape, (str, bytes)):
        raise ValueError(f"shape must be a (height, width) sequence; got {shape!r}")
    try:
        dimensions = tuple(shape)
    except TypeError as exc:
        raise ValueError(f"shape must be a (height, width) sequence; got {shape!r}") from exc
    if len(dimensions) != 2:
        raise ValueError(
            f"shape must contain exactly two dimensions (height, width); got {len(dimensions)}"
        )
    height = _positive_int(dimensions[0], "height")
    width = _positive_int(dimensions[1], "width")
    return height, width


def _validate_factor(factor: object, shape: tuple[int, int]) -> int:
    """Return a positive integer factor that divides both dimensions."""
    resolved = _positive_int(factor, "binning factor")
    height, width = shape
    if height % resolved != 0 or width % resolved != 0:
        raise ValueError(
            f"binning factor {resolved} must divide the sensor grid shape "
            f"({height}, {width}) evenly; a partial block would aggregate "
            "unequal pixel areas"
        )
    return resolved


def _positive_int(value: object, field: str) -> int:
    """Return a positive Python integer, rejecting bools and floats."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be a positive integer; got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer; got {result}")
    return result


def _validate_plane(values: ArrayLike, field: str) -> NDArray[np.float64]:
    """Return a finite 2-D float64 plane."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a real 2-D array; {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"{field} must be a 2-D (H, W) array; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{field} must contain at least one pixel")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be finite (no NaN or infinity)")
    return array
