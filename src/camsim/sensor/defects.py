"""Defective-pixel and phase-detection-autofocus pixel maps.

A production sensor array is not perfectly uniform. Three defect classes are
modelled here as independent per-pixel failure modes:

* **Dead** pixels have no photoresponse. They read near the black level
  regardless of illumination and appear as dark points. Dead rates for
  well-controlled consumer parts are roughly one in 10,000 to one in 1,000,000
  pixels. The sensor zeroes their expected charge before Poisson sampling, so
  enabling the rate changes the image, not only a mask.
* **Hot** pixels have anomalously high dark current. Their brightness grows
  with exposure time and roughly doubles every 5-10 K, so a pixel that is
  invisible in a short daylight frame can saturate a multi-second night frame.
  Hot rates span roughly 1e-5 to 1e-3. The sensor applies a dark-current
  multiplier and returns the site map separately as ``RawFrame.hot_mask``; the
  union ``RawFrame.defect_mask`` also contains hot sites so DPC replaces them.
* **Stuck** pixels are pinned to a fixed code (black or white) by a shorted or
  open readout node. Rates are similar to dead pixels, around 1e-6 to 1e-5.
  The simulator pins stuck sites to the ADC white code ``2**bits - 1`` after
  the ADC and digital gain. A real stuck site can be pinned black or white;
  the model chooses one deterministic convention so the output is
  reproducible.

The three masks are drawn independently, so a rare pixel can be in more than
one class. That is physically plausible and harmless: a defect-correction stage
takes the union, while the hot-pixel dark-current multiplier and the dead /
stuck signal behaviours are applied before correction. Recomputing any of
these maps per frame would turn a fixed, calibratable pattern into temporal
noise.

Phase-detection autofocus (PDAF) pixels are not defects. They use a partially
masked or split photodiode to sample the exit pupil and therefore collect only
a fraction ``qe_factor`` of the light of a normal pixel. Sparse PDAF layouts
place a small percentage of sites on one CFA colour, typically green because
the two green sites per Bayer tile give the densest sampling. The mask returned
by :func:`pdaf_mask` only flags those positions; the sensor applies
``qe_factor`` to their response, and defect correction must leave them alone
because their reduced value is intentional. Typical PDAF fractions are 0.5-5 %
of the eligible colour sites.

Real-world parameter ranges: dead 1e-6 to 1e-4, hot 1e-5 to 1e-3, stuck 1e-6
to 1e-5; PDAF fractions 0.005-0.05 with ``qe_factor`` 0.3-0.8.

Determinism. Both generators take an explicit non-negative seed and never touch
NumPy's global RNG. Each defect class and the PDAF layout use a named
``SeedSequence`` stream derived from that seed, so passing one sensor pattern
seed to every generator does not correlate their outputs. Stream identifiers
are part of the reproducibility contract.

Failure artefacts guarded here: a rate supplied in percent (``1`` instead of
``0.01``) flags every pixel; a PDAF pattern applied without consulting the CFA
places sites on the wrong colour or silently marks a mono sensor; treating PDAF
sites as defects destroys the autofocus signal; and sharing one random stream
between defects and PDAF makes every PDAF site coincide with a dead pixel at the
same rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from camsim.config.schema import DefectConfig, PDAFConfig
from camsim.core.frames import CFAPattern
from camsim.sensor.cfa import CFAColor, cfa_color_at

__all__ = ["DefectMasks", "defect_masks", "pdaf_mask"]

# Named seed streams are globally unique across the sensor package: noise.py
# owns 0-1 (DSNU, PRNU), defects.py owns 2-5. The identifiers are part of the
# reproducibility contract, so changing one changes the maps for existing seeds.
_DEAD_STREAM: Final[int] = 2
_HOT_STREAM: Final[int] = 3
_STUCK_STREAM: Final[int] = 4
_PDAF_STREAM: Final[int] = 5

# The reusable random thresholds are drawn in row blocks so that a 50 MP frame
# needs a few megabytes of temporary storage instead of hundreds.
_CHUNK_ELEMENTS: Final[int] = 1 << 20

_PDAF_COLOURS: Final[dict[str, CFAColor | None]] = {
    "sparse_green": "G",
    "sparse_red": "R",
    "sparse_blue": "B",
    "none": None,
}


@dataclass(eq=False)
class DefectMasks:
    """Boolean defect maps for one sensor.

    Parameters
    ----------
    dead:
        ``(H, W)`` mask of pixels with no photoresponse. The sensor zeroes
        their expected charge before sampling, and a defect-correction stage
        replaces the resulting near-black samples from same-CFA neighbours.
    hot:
        ``(H, W)`` mask of pixels with excess dark current. The sensor applies
        a dark-current multiplier and records this subset separately in
        ``RawFrame.hot_mask``; ``RawFrame.defect_mask`` contains the union.
    stuck:
        ``(H, W)`` mask of pixels pinned to a fixed output code. The simulator
        pins them to the ADC white code; corrected like dead pixels.

    The classes are drawn independently and may overlap. Equality is identity
    based because the bundle owns large arrays.
    """

    dead: NDArray[np.bool_]
    hot: NDArray[np.bool_]
    stuck: NDArray[np.bool_]


def defect_masks(
    config: DefectConfig,
    shape: Sequence[int],
    seed: int,
) -> DefectMasks:
    """Generate dead, hot, and stuck pixel masks from configured rates.

    Parameters
    ----------
    config:
        Validated :class:`~camsim.config.schema.DefectConfig` carrying the
        per-pixel probabilities. ``config.seed`` is the preset's suggested
        pattern seed; callers that manage sensor-wide reproducibility pass
        their pattern seed explicitly through ``seed`` so every map family
        derives from one seed.
    shape:
        ``(height, width)`` of the sensor in pixels; both positive integers.
    seed:
        Non-negative pattern seed. The same seed and rates reproduce identical
        masks; a different seed produces different placements.

    Returns
    -------
    :class:`DefectMasks` with three boolean arrays of ``shape``.

    Raises
    ------
    ValueError
        If the configuration type is wrong, the shape is not two positive
        integers, the seed is negative or not an integer, or a rate lies
        outside ``[0, 1]`` (for example a percentage such as ``1`` meaning
        0.01 %).

    Notes
    -----
    Each rate is an independent Bernoulli probability per pixel, so the
    realized count fluctuates around ``rate * height * width`` with binomial
    standard deviation. Defect maps are fixed for the life of the sensor; do
    not regenerate them per frame. Hot pixels receive the sensor's
    dark-current multiplier and remain identifiable through
    ``RawFrame.hot_mask``; dead and stuck pixels change the sampled or
    quantized signal, and all three classes join ``defect_mask`` so DPC
    replaces them.
    """
    if not isinstance(config, DefectConfig):
        raise ValueError(f"config must be a DefectConfig; got {type(config).__name__}")
    height, width = _validate_shape(shape)
    resolved_seed = _validate_seed(seed)
    map_shape = (height, width)
    dead_rate = _validate_rate(config.dead_rate, "dead_rate")
    hot_rate = _validate_rate(config.hot_rate, "hot_rate")
    stuck_rate = _validate_rate(config.stuck_rate, "stuck_rate")
    return DefectMasks(
        dead=_bernoulli_mask(dead_rate, map_shape, _map_rng(resolved_seed, _DEAD_STREAM)),
        hot=_bernoulli_mask(hot_rate, map_shape, _map_rng(resolved_seed, _HOT_STREAM)),
        stuck=_bernoulli_mask(stuck_rate, map_shape, _map_rng(resolved_seed, _STUCK_STREAM)),
    )


def pdaf_mask(
    config: PDAFConfig,
    shape: Sequence[int],
    seed: int,
    *,
    cfa: CFAPattern | str = CFAPattern.RGGB,
) -> NDArray[np.bool_]:
    """Flag sparse phase-detection autofocus sites for one CFA layout.

    Parameters
    ----------
    config:
        Validated :class:`~camsim.config.schema.PDAFConfig`. ``pattern``
        selects the eligible CFA colour (``sparse_green``, ``sparse_red``,
        ``sparse_blue``) or disables PDAF (``none``). ``fraction`` is the
        probability that an eligible site is used, and ``qe_factor`` is
        intentionally not applied here: the sensor scales the flagged sites'
        response so the mask can be reused across exposure and gain settings.
    shape:
        ``(height, width)`` of the sensor in pixels; both positive integers.
    seed:
        Non-negative pattern seed. Identical seeds reproduce identical
        placements.
    cfa:
        Colour-filter pattern whose tile defines the eligible sites. The
        default is ``RGGB``, the flagship preset layout; callers with ``BGGR``,
        ``GRBG``, ``GBRG``, ``RCCC``, or ``MONO`` must pass their actual
        pattern, otherwise sparse sites land on the wrong colour. A mono
        pattern has no red, green, or blue sites, so a sparse colour rule
        produces an empty mask.

    Returns
    -------
    A boolean array of ``shape``. True marks a PDAF pixel. At most
    ``fraction`` of the eligible colour sites are flagged, and no other pixel
    is ever flagged.

    Raises
    ------
    ValueError
        If the configuration type is wrong, the shape is invalid, the seed is
        invalid, ``fraction`` is outside ``[0, 1]``, or ``pattern`` is not a
        supported layout.

    Notes
    -----
    Eligible sites are identified by tiling the four positions of the 2x2 CFA
    tile returned by :func:`~camsim.sensor.cfa.cfa_color_at`, so the mask
    follows the same colour convention as the mosaic and spectral code. The
    requested fraction is a per-site probability; the realized count
    fluctuates binomially around ``fraction * eligible_count``. A fraction of
    exactly one selects all eligible sites, and zero selects none.
    """
    if not isinstance(config, PDAFConfig):
        raise ValueError(f"config must be a PDAFConfig; got {type(config).__name__}")
    height, width = _validate_shape(shape)
    resolved_seed = _validate_seed(seed)
    fraction = _validate_fraction(config.fraction, "fraction")
    try:
        colour = _PDAF_COLOURS[config.pattern]
    except (KeyError, TypeError) as exc:
        allowed = ", ".join(_PDAF_COLOURS)
        raise ValueError(f"pdaf pattern must be one of {allowed}; got {config.pattern!r}") from exc

    mask_shape = (height, width)
    mask = np.zeros(mask_shape, dtype=np.bool_)
    if colour is None or fraction == 0.0:
        return mask
    eligible = _colour_sites(cfa, colour, mask_shape)
    if not bool(eligible.any()):
        return mask
    if fraction >= 1.0:
        return eligible.copy()
    # Draw a threshold for every pixel and keep only eligible sites. This is
    # statistically identical to drawing only on eligible sites and keeps the
    # chunked implementation sharing code with the defect masks.
    return _bernoulli_mask(fraction, mask_shape, _map_rng(resolved_seed, _PDAF_STREAM)) & eligible


def _colour_sites(
    cfa: CFAPattern | str,
    colour: CFAColor,
    shape: tuple[int, int],
) -> NDArray[np.bool_]:
    """Return the sites whose CFA colour equals ``colour``.

    The 2x2 tile is repeated in both directions, so querying the four origin
    positions with :func:`cfa_color_at` and tiling each result reproduces the
    full mosaic without a per-pixel Python loop.
    """
    height, width = shape
    eligible = np.zeros((height, width), dtype=np.bool_)
    for row_offset in (0, 1):
        for column_offset in (0, 1):
            if cfa_color_at(cfa, row_offset, column_offset) == colour:
                eligible[row_offset::2, column_offset::2] = True
    return eligible


def _bernoulli_mask(
    rate: float,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """Draw a per-pixel Bernoulli mask in bounded-memory row blocks."""
    height, width = shape
    mask = np.zeros(shape, dtype=np.bool_)
    if rate <= 0.0:
        return mask
    if rate >= 1.0:
        mask.fill(True)
        return mask
    rows_per_chunk = max(1, _CHUNK_ELEMENTS // width)
    for start in range(0, height, rows_per_chunk):
        stop = min(start + rows_per_chunk, height)
        mask[start:stop] = rng.random((stop - start, width)) < rate
    return mask


def _map_rng(seed: int, stream: int) -> np.random.Generator:
    """Return a generator for one independent, named stream of a pattern seed."""
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def _validate_shape(shape: object) -> tuple[int, int]:
    """Return a ``(height, width)`` pair of positive Python integers."""
    try:
        dimensions = np.asarray(shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"shape must be a (height, width) pair of positive integers; got {shape!r}"
        ) from exc
    if dimensions.shape != (2,) or not np.issubdtype(dimensions.dtype, np.integer):
        raise ValueError(
            f"shape must be a (height, width) pair of positive integers; got {shape!r}"
        )
    height = int(dimensions[0])
    width = int(dimensions[1])
    if height <= 0 or width <= 0:
        raise ValueError(
            f"shape must be a (height, width) pair of positive integers; got {shape!r}"
        )
    return height, width


def _validate_seed(seed: object) -> int:
    """Return ``seed`` as a non-negative Python integer."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}")
    value = int(seed)
    if value < 0:
        raise ValueError(f"seed must be a non-negative integer; got {value}")
    return value


def _validate_rate(value: object, field: str) -> float:
    """Return a probability in ``[0, 1]``."""
    rate = _validate_finite_real(value, field)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(
            f"{field} must lie within [0, 1]; got {rate!r}. Rates are "
            "probabilities per pixel, so 0.01 percent is 0.0001, not 0.01."
        )
    return rate


def _validate_fraction(value: object, field: str) -> float:
    """Return a dimensionless fraction in ``[0, 1]``."""
    fraction = _validate_finite_real(value, field)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"{field} must lie within [0, 1]; got {fraction!r}. It is a "
            "fraction of eligible sites, so 1 percent is 0.01, not 1."
        )
    return fraction


def _validate_finite_real(value: object, field: str) -> float:
    """Return ``value`` as a finite Python float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be a real number; got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return result
