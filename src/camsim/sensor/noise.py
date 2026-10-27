"""Temporal and fixed-pattern noise sources in the electron domain.

A CMOS pixel converts photons to electrons, and every subsequent stage adds
noise that is best described in electrons because that is where the physics
lives. This module provides the stochastic primitives; conversion gain, analog
gain, and quantization belong to the next task.

**Shot noise.** Photon arrivals and the resulting photoelectrons are discrete
and independent, so a pixel whose mean signal is ``S`` electrons has variance
``S`` (Poisson). The signal-to-noise ratio is therefore ``sqrt(S)``: a
100-electron pixel carries 10 % noise, a 10,000-electron pixel 1 %, and a
twofold exposure buys only ``sqrt(2)`` in SNR. A Gaussian approximation is
tempting because it is cheaper, but at low counts it produces negative
electrons and underestimates the skew that dominates a dark image. Sampling
must stay Poisson until read noise, which *is* Gaussian, is added.

**Dark current.** Thermally generated carriers accumulate even with the shutter
closed. The rate follows the Arrhenius-like doubling law

``I_dark(T) = I_dark(T_ref) * 2 ** ((T - T_ref) / T_double)``

with silicon devices typically doubling every 5-10 K. A 1 e-/s pixel at 25 C
becomes about 8 e-/s at 46 C (three doublings), so a warm sensor in a long
exposure can be dominated by dark current rather than by the scene. The
electrons themselves are Poisson-distributed, so :func:`sample_dark_current`
uses the same sampler as shot noise.

**Fixed-pattern noise.** Two imperfections are spatial and constant for a given
part, unlike the temporal noise above. Dark-signal non-uniformity (DSNU) is a
per-pixel offset, modelled as a Gaussian with standard deviation
``dsnu_e`` electrons. Photo-response non-uniformity (PRNU) is a per-pixel
multiplicative responsivity error, modelled as the dimensionless deviation
``prnu`` such that the collected signal is ``S * (1 + prnu)``. DSNU and PRNU
are calibrated out with dark and flat fields respectively, but they are the
reason two nominally identical sensors produce different images and why a
single-frame noise measurement cannot separate fixed from temporal terms.

**Row noise.** Readout electronics share a reference per row, so a common-mode
offset appears on every column of a row and forms horizontal banding. It is
Gaussian in electrons, one value per row, and does not average down across the
row the way independent pixel noise does.

Real-world ranges: read noise 0.3-10 e- rms (sub-electron with correlated
double sampling and dual conversion gain); dark current 0.01-100 e-/s at 25 C,
doubling every 5-10 K; DSNU 0.1-10 e- rms; PRNU 0.1-2 % (``0.001-0.02`` as a
fraction); row noise 0.5-10 e- rms.

Determinism. Every stochastic helper accepts an explicit
``numpy.random.Generator`` (shot, dark-current sampling, row noise) or a
non-negative integer seed (the fixed maps). Nothing reads NumPy's global RNG.
Map families derive independent named streams from the same pattern seed, so a
defect map and a PDAF map generated from one sensor seed do not share random
draws. Stream identifiers are part of the reproducibility contract: changing
one changes every map produced from existing seeds.

Failure artefacts guarded here: a negative or non-finite mean electron count;
a dark-current model with the wrong reference temperature or doubling interval;
a PRNU value supplied in percent (``2`` instead of ``0.02``), which triples the
signal gain; zero-sigma stages that silently advance the RNG and perturb later
noise; and recomputing fixed-pattern maps per frame, which converts calibratable
spatial noise into temporal noise.

Wiring note: :class:`FPNConfig` is a local value object rather than a
``GainModeConfig`` field because DSNU, PRNU, and row noise describe the array,
not a readout mode. :class:`camsim.Sensor` composes it from ``SensorConfig``
and passes it to :func:`fixed_pattern_maps` and :func:`row_noise`; the module
remains usable on its own without changing the public configuration contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.config.schema import TemperatureConfig

__all__ = [
    "FPNConfig",
    "FPPMaps",
    "dark_current_e_per_s",
    "fixed_pattern_maps",
    "poisson_electrons",
    "row_noise",
    "sample_dark_current",
]

# Named seed streams keep map families independent when one pattern seed is
# shared across defects, PDAF, DSNU, and PRNU. defect.py owns streams 2-5.
# These integers are part of the reproducibility contract: changing a stream id
# changes every map generated from an existing seed.
_DSNU_STREAM: Final[int] = 0
_PRNU_STREAM: Final[int] = 1


@dataclass(frozen=True)
class FPNConfig:
    """Fixed-pattern and row-noise strengths for one sensor.

    Parameters
    ----------
    dsnu_e:
        Standard deviation of the per-pixel dark-signal offset, in electrons.
        Zero disables the map without perturbing the PRNU stream.
    prnu_fraction:
        Standard deviation of the per-pixel multiplicative responsivity
        deviation. It is a fraction, not a percentage: 2 % is ``0.02``. The
        value must be less than one so that a realised gain ``1 + prnu`` is
        positive in expectation.
    row_noise_e_rms:
        Common-mode row-noise sigma in electrons. It is carried here for the
        sensor pipeline; :func:`row_noise` accepts the sigma directly.

    Notes
    -----
    :class:`camsim.Sensor` composes this object from the ``SensorConfig`` noise
    block. Keeping the noise strengths in one array-level value object avoids
    attaching DSNU, PRNU, and row noise to a per-readout-mode configuration
    where they do not belong.
    """

    dsnu_e: float = 0.0
    prnu_fraction: float = 0.0
    row_noise_e_rms: float = 0.0

    def __post_init__(self) -> None:
        dsnu = _validate_non_negative_real(self.dsnu_e, "dsnu_e")
        prnu = _validate_non_negative_real(self.prnu_fraction, "prnu_fraction")
        if prnu >= 1.0:
            raise ValueError(
                f"prnu_fraction must be less than 1.0 (got {prnu:g}); it is a "
                "relative standard deviation, so 2 percent is 0.02, not 2"
            )
        row_noise = _validate_non_negative_real(self.row_noise_e_rms, "row_noise_e_rms")
        # A frozen dataclass cannot assign normally; normalise accepted integer
        # inputs to float so downstream arithmetic always sees one dtype.
        object.__setattr__(self, "dsnu_e", dsnu)
        object.__setattr__(self, "prnu_fraction", prnu)
        object.__setattr__(self, "row_noise_e_rms", row_noise)


@dataclass(eq=False)
class FPPMaps:
    """Fixed-pattern maps for one sensor, in linear electron units.

    Parameters
    ----------
    dsnu_e:
        ``(H, W)`` float64 additive offsets in electrons, zero mean and
        standard deviation ``FPNConfig.dsnu_e``.
    prnu_fraction:
        ``(H, W)`` float64 dimensionless multiplicative deviations. Apply as
        ``signal * (1 + prnu_fraction)``; the stored map is the deviation, not
        the gain, so its mean is near zero rather than near one.

    Equality is identity based because the bundle owns large arrays; compare
    fields explicitly when a test needs value equality.
    """

    dsnu_e: NDArray[np.float64]
    prnu_fraction: NDArray[np.float64]


def poisson_electrons(mean: ArrayLike, rng: np.random.Generator) -> NDArray[np.int64]:
    """Sample photoelectrons from a Poisson distribution.

    Parameters
    ----------
    mean:
        Mean electron count per pixel. A scalar or any array shape is
        accepted; every value must be finite and non-negative. The mean is the
        expected number of electrons for the exposure and pixel area, not a
        digital number and not a rate per second.
    rng:
        Explicit NumPy generator. Passing a generator rather than a global
        seed lets the caller keep independent streams for, for example,
        photoelectrons and read noise.

    Returns
    -------
    An int64 array with the same shape as ``mean``. A scalar input returns a
    zero-dimensional array. Values are non-negative by construction.

    Notes
    -----
    The Poisson variance equals the mean, so this sampler is the physical
    source of shot noise. Do not replace it with a rounded Gaussian: at a mean
    of one electron a Gaussian yields negative counts and hides the skew that
    makes very dark images differ from a symmetric blur.
    """
    return _poisson_samples(mean, rng, "mean")


def sample_dark_current(mean_e: ArrayLike, rng: np.random.Generator) -> NDArray[np.int64]:
    """Sample thermally generated dark electrons for one exposure.

    Parameters
    ----------
    mean_e:
        Mean dark electrons per pixel, already multiplied by exposure time and
        any hot-pixel multiplier. The value comes from
        :func:`dark_current_e_per_s` times the exposure in seconds. Must be
        finite and non-negative.
    rng:
        Explicit NumPy generator.

    Returns
    -------
    An int64 array with the same shape as ``mean_e``.

    Notes
    -----
    Dark current is a generation-recombination process and therefore Poisson,
    exactly like photoelectrons. It is separated from
    :func:`poisson_electrons` only to keep the physical origin of the mean
    explicit in the signal chain and in traces.
    """
    return _poisson_samples(mean_e, rng, "mean_e")


def dark_current_e_per_s(
    dark_current_ref_e_per_s: float,
    temperature_c: float,
    config: TemperatureConfig,
) -> float:
    """Scale a reference dark-current rate to the sensor temperature.

    The model is ``I_ref * 2 ** ((T - T_ref) / T_double)`` in electrons per
    second. Multiply the result by exposure time (and by a hot-pixel
    multiplier where applicable) before sampling Poisson electrons.

    Parameters
    ----------
    dark_current_ref_e_per_s:
        Dark-current rate at the configuration's reference temperature, in
        electrons per second. Must be finite and non-negative.
    temperature_c:
        Sensor temperature in degrees Celsius. Negative values are physical.
    config:
        Validated :class:`~camsim.config.schema.TemperatureConfig` supplying
        ``reference_c`` and ``dark_current_doubling_k``.

    Returns
    -------
    The dark-current rate in electrons per second at ``temperature_c``.

    Raises
    ------
    ValueError
        If an input is non-finite, the reference rate or doubling interval is
        non-positive, the configuration type is wrong, or the exponential
        scaling overflows. A silently infinite rate would poison every
        downstream frame.

    Notes
    -----
    A doubling interval of 7 K is representative of silicon. Because the
    exponent is relative to ``reference_c``, a preset whose dark current is
    quoted at 20 C must also set its reference to 20 C; mixing references
    under- or over-predicts the rate by the same factor the datasheet used.
    """
    if not isinstance(config, TemperatureConfig):
        raise ValueError(f"config must be a TemperatureConfig; got {type(config).__name__}")
    reference_rate = _validate_non_negative_real(
        dark_current_ref_e_per_s, "dark_current_ref_e_per_s"
    )
    temperature = _validate_finite_real(temperature_c, "temperature_c")
    reference_c = _validate_finite_real(config.reference_c, "config.reference_c")
    doubling_k = _validate_positive_real(
        config.dark_current_doubling_k, "config.dark_current_doubling_k"
    )
    try:
        scale = math.pow(2.0, (temperature - reference_c) / doubling_k)
    except OverflowError as exc:
        raise ValueError(
            f"dark current scaling overflows at {temperature:g} C with a "
            f"{doubling_k:g} K doubling interval"
        ) from exc
    rate = reference_rate * scale
    if not math.isfinite(rate):
        raise ValueError(
            f"dark current rate is not finite at {temperature:g} C; "
            "check the reference temperature and doubling interval"
        )
    return rate


def fixed_pattern_maps(
    config: FPNConfig,
    shape: Sequence[int],
    seed: int,
) -> FPPMaps:
    """Generate the DSNU offset and PRNU gain-deviation maps for one sensor.

    Parameters
    ----------
    config:
        Fixed-pattern strengths. A zero strength yields an all-zero map and
        does not consume that map's RNG stream, so enabling DSNU does not
        change the PRNU pattern for the same seed.
    shape:
        ``(height, width)`` of the sensor in pixels; both positive integers.
    seed:
        Non-negative pattern seed. Identical seeds and configuration produce
        bit-identical maps; a different seed produces different maps.

    Returns
    -------
    :class:`FPPMaps` with ``dsnu_e`` (additive, electrons) and
    ``prnu_fraction`` (multiplicative deviation) float64 arrays of ``shape``.

    Notes
    -----
    DSNU is drawn as ``N(0, dsnu_e)`` and PRNU as ``N(0, prnu_fraction)``. Both
    are fixed for the life of the sensor: regenerate them only when the pattern
    seed or the sensor configuration changes. Recomputing per frame would make
    the calibration-free fixed pattern indistinguishable from temporal noise.

    The maps use independent named seed streams (see the module docstring), so
    DSNU and PRNU are uncorrelated even though they share ``seed``, and the
    defect/PDAF maps in :mod:`camsim.sensor.defects` use disjoint streams.
    """
    if not isinstance(config, FPNConfig):
        raise ValueError(f"config must be an FPNConfig; got {type(config).__name__}")
    height, width = _validate_shape(shape)
    resolved_seed = _validate_seed(seed)
    map_shape = (height, width)
    return FPPMaps(
        dsnu_e=_gaussian_map(config.dsnu_e, map_shape, resolved_seed, _DSNU_STREAM),
        prnu_fraction=_gaussian_map(config.prnu_fraction, map_shape, resolved_seed, _PRNU_STREAM),
    )


def row_noise(
    row_sigma_e: float,
    height: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw one common-mode Gaussian offset per row.

    Parameters
    ----------
    row_sigma_e:
        Row-noise standard deviation in electrons, finite and non-negative.
    height:
        Number of rows; a positive integer.
    rng:
        Explicit NumPy generator.

    Returns
    -------
    A ``(height, 1)`` float64 array. Broadcast it across columns (for example
    ``electrons + row_noise(...)`` on an ``(H, W)`` frame) to add the same
    offset to every pixel in a row.

    Notes
    -----
    Row noise belongs before conversion gain because it is an electron-domain
    common-mode disturbance. A zero sigma returns zeros without consuming the
    generator, making the disabled stage a true identity for downstream noise
    draws. Drawing one value per pixel instead would model independent read
    noise and would not produce the horizontal banding caused by a shared row
    reference.
    """
    sigma = _validate_non_negative_real(row_sigma_e, "row_sigma_e")
    rows = _validate_positive_int(height, "height")
    generator = _require_generator(rng)
    if sigma == 0.0:
        return np.zeros((rows, 1), dtype=np.float64)
    return np.asarray(generator.normal(0.0, sigma, size=(rows, 1)), dtype=np.float64)


def _poisson_samples(
    mean: ArrayLike,
    rng: np.random.Generator,
    field: str,
) -> NDArray[np.int64]:
    """Validate an electron mean and draw Poisson samples with its shape."""
    values = _validate_electron_means(mean, field)
    generator = _require_generator(rng)
    return np.asarray(generator.poisson(values), dtype=np.int64)


def _validate_electron_means(mean: ArrayLike, field: str) -> NDArray[np.float64]:
    """Return finite, non-negative electron means as a float64 array."""
    try:
        values = np.asarray(mean, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a real scalar or array of electron counts; got {mean!r}"
        ) from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field} electron means must be finite (no NaN or infinity)")
    if np.any(values < 0.0):
        minimum = float(values.min())
        raise ValueError(f"{field} electron means must be non-negative; got minimum {minimum:g}")
    return values


def _gaussian_map(
    sigma: float,
    shape: tuple[int, int],
    seed: int,
    stream: int,
) -> NDArray[np.float64]:
    """Draw a zero-mean Gaussian map from one named seed stream."""
    if sigma == 0.0:
        return np.zeros(shape, dtype=np.float64)
    generator = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    return np.asarray(generator.normal(0.0, sigma, size=shape), dtype=np.float64)


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


def _validate_positive_int(value: object, field: str) -> int:
    """Return a positive Python integer, rejecting bools and floats."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be a positive integer; got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer; got {result}")
    return result


def _validate_finite_real(value: object, field: str) -> float:
    """Return ``value`` as a finite Python float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be a real number; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return result


def _validate_non_negative_real(value: object, field: str) -> float:
    """Return ``value`` as a finite, non-negative Python float."""
    result = _validate_finite_real(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative; got {result!r}")
    return result


def _validate_positive_real(value: object, field: str) -> float:
    """Return ``value`` as a finite, strictly positive Python float."""
    result = _validate_finite_real(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive; got {result!r}")
    return result


def _require_generator(rng: object) -> np.random.Generator:
    """Return ``rng`` after checking that it is a NumPy generator."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError(f"rng must be a numpy.random.Generator; got {type(rng).__name__}")
    return rng
