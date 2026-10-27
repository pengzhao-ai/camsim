"""ISO 12233-style slanted-edge e-SFR: edge-spread function to MTF.

A camera's resolution is described by its modulation transfer function (MTF),
the fraction of a sinusoidal pattern's contrast that survives the lens, the
pixel aperture, and any processing. Measuring it from a single image needs a
test structure whose spatial frequency content is known exactly. The slanted
edge is that structure. Unlike a vertical step, whose transition lands on one
column and therefore one pixel phase, an edge tilted a few degrees crosses every
sub-pixel phase as it runs down the frame, so the pixels carry far more
information about the transition than the pixel grid suggests (ISO 12233, the
standard slanted-edge method; this implementation follows its principles but is
**not certified** and makes no claim of conformance to the standard's chart,
windowing, or reporting details).

The measurement chain
---------------------
1. **Edge-spread function (ESF).** Project each pixel of the analysis region
   onto the edge normal, ``u = x cos(theta) - y sin(theta)``, and average the
   pixel values in half-open bins of width ``1 / oversampling`` pixel. Binning
   at four samples per pixel turns a 250-pixel-wide region into an ESF a
   thousand samples long, so a transition that occupies one or two pixels is
   sampled at a quarter-pixel spacing.
2. **Line-spread function (LSF).** The derivative of a step edge is the system's
   line-spread function: ``LSF(u) = dESF/du``. A central difference at spacing
   ``d = 1 / oversampling`` approximates it; the finite difference has its own
   frequency response, corrected below. The ESF is *not* renormalized to
   ``[0, 1]``: the LSF's DC term carries the edge contrast and cancels in the
   MTF normalization, so absolute image scaling and the RAW black-level pedestal
   do not matter.
3. **MTF.** The optical transfer function is the normalized Fourier transform of
   the LSF, and the MTF is its magnitude:
   ``MTF(f) = |FT{LSF}(f)| / |FT{LSF}(0)|``. The LSF is optionally multiplied by
   a Hamming window to suppress the discontinuity at the ends of the measured
   span; without a window the implicit rectangular window produces sidelobes
   that leak into the high-frequency estimate.
4. **Estimator correction.** A central difference does not differentiate
   perfectly: its transfer function is ``j sin(2 pi f d) / d``, whose normalized
   magnitude is ``|sinc(2 pi f d)|`` (with ``sinc(x) = sin(x) / x``). Dividing
   the magnitude by that factor removes the estimator's own roll-off. No other
   aperture correction is applied: the pixel aperture and the lens are part of
   the system being measured, so dividing them out would report an MTF the
   camera does not have.
5. **Band and levels.** The result is truncated to the native Nyquist limit,
   ``0 <= f <= 0.5`` cycles/pixel; the oversampled analysis grid extends to
   ``oversampling / 2`` but that region is dominated by the estimator, not the
   system. MTF50 and MTF20 are the first frequencies where the curve crosses one
   half and one fifth, interpolated linearly in frequency. A system that never
   falls that far inside the measured band (an ideal synthetic edge, or a lens
   sharper than Nyquist) has no finite MTF50 and reports ``nan`` rather than an
   invented number. Converting cycles/pixel to cycles/degree uses the angular
   pixel pitch: ``pixels_per_degree = 1 / degrees(atan(pitch / focal_length))``,
   the paraxial small-angle relation.

Real-world ranges
-----------------
Phone lenses measure MTF50 of roughly 0.1-0.4 cycles/pixel at the image centre;
diffraction-limited systems at f/8 fall near 0.1; a misfocused or heavily
denoised system can drop below 0.05. ISO 12233 recommends a 5 degree tilt
(printed charts use 2-7 degrees), and edge angles from 5 to 20 degrees are
common in practice. Analysis regions are typically 100-400 pixels tall; four
samples per pixel is the conventional oversampling factor, and eight or more is
used for diffraction-limited systems. Pixel pitches run from about 0.8 um in
dense phone arrays to 10 um in large machine-vision pixels, and focal lengths
from about 2 mm to 50 mm. The edge must stay inside the analysis region over its
full height, so near-horizontal edges are rejected.

Failure artefacts guarded here
------------------------------
* **Gamma-encoded input.** A display-referred image has already passed a
  non-linear transfer function; its edge profile is not the linear-light step
  the derivative assumes, so the reported MTF is biased (typically low at high
  frequencies, where the curve's shape is dominated by the tone curve). Feed the
  metric linear light.
* **Aliasing and noise near Nyquist.** Sensor noise adds a flat high-frequency
  floor and can push a noisy estimate above one; values outside ``[0, 1]`` are
  clipped, and the Hamming window suppresses the sidelobes that would otherwise
  leak into the band.
* **Derivative and window bias.** Oversampling of one sample per pixel puts the
  derivative response's zero exactly at Nyquist; the correction is capped there
  (a 5 % floor), which makes the last bin of an oversampling-1 curve unreliable.
  Oversampling of two or more keeps the derivative response above 0.64 across
  the band, so the correction never exceeds about 1.57x.
* **Edge-angle error.** A wrong tilt smears the projection and biases MTF low.
  The angle is estimated per image from a gradient-centroid line fit, not assumed.
* **Quincunx green sampling.** A Bayer green plane samples on a doubled pitch
  with the two green phases offset diagonally. Extracting one phase preserves a
  rectangular lattice (the right thing for MTF) but halves the Nyquist to 0.25
  cycles per original pixel; averaging the two phases is a hidden low-pass and
  must not be done.
* **Border contamination.** Convolution padding and vignetting create a fake
  ramp at the frame border. A margin of at least four pixels (about 2 % of the
  shorter side) is discarded before any analysis.
* **Missing or misplaced edges.** A flat region, a purely horizontal edge, or an
  edge that falls entirely inside the discarded margin yields no usable
  gradient and raises instead of returning a meaningless curve.

The module is part of the independent metrics layer: it imports NumPy and
:mod:`camsim.core.frames` only, never the sensor, ISP, scene, or optics
packages, so the measurement harness cannot accidentally depend on the model it
is meant to judge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.core.frames import CFAPattern, RawFrame

__all__ = ["MTFCurve", "MTFResult", "extract_raw_plane", "slanted_edge_mtf"]

# The native sampling limit of the image being measured. The oversampled ESF
# extends to `oversampling / 2`, but its top octave is dominated by the
# interpolation estimator rather than by the camera, so it is truncated.
_NYQUIST_CP: Final[float] = 0.5

# Border pixels are discarded because PSF convolution pads the frame and
# vignetting bends the illumination there; both create a fake ramp that the
# derivative would report as edge energy. Four pixels are the minimum for a
# compact kernel, and 2 % of the shorter side covers wide kernels on large
# frames.
_BORDER_MARGIN_MIN_PX: Final[int] = 4
_BORDER_MARGIN_FRACTION: Final[float] = 0.02

# Below this many pixels per side the projection produces too few oversampled
# bins for a meaningful derivative and frequency axis.
_MIN_ANALYSIS_PX: Final[int] = 16

# A per-row threshold at 10 % of that row's gradient peak rejects sensor noise
# and chart texture far from the edge while keeping the whole transition.
_EDGE_GRADIENT_THRESHOLD: Final[float] = 0.1
_MIN_EDGE_ROWS: Final[int] = 2

# The centroid fit assumes a near-vertical edge; beyond 45 degrees the edge has
# left the frame or the horizontal Sobel no longer sees it.
_MAX_EDGE_ANGLE_DEG: Final[float] = 45.0

# Standard Hamming window: 0.54 - 0.46 cos(2 pi n / (N - 1)).
_HAMMING_PEDESTAL: Final[float] = 0.54
_HAMMING_MODULATION: Final[float] = 0.46

# The central-difference response vanishes at f = 1 / (2 d). Dividing by it
# there would multiply noise by an arbitrary factor, so the correction is capped
# at 20x; oversampling of two or more never reaches the cap inside [0, 0.5].
_DERIVATIVE_SINC_FLOOR: Final[float] = 0.05

# A curve built by the metric normalizes DC exactly; the tolerance only lets
# callers round-trip through serialization without spurious failures.
_MTF_NORMALIZATION_TOLERANCE: Final[float] = 1e-6

_FREQUENCY_TOLERANCE: Final[float] = 1e-12

_WINDOW_NAMES: Final[tuple[str, ...]] = ("hamming", "none")

# A Bayer plane samples one colour every second pixel in both axes, so its
# sample spacing is two original pixel pitches.
_BAYER_PITCH_MULTIPLIER: Final[float] = 2.0
_MONO_PITCH_MULTIPLIER: Final[float] = 1.0
_MONO_CHANNEL: Final[str] = "M"
_BAYER_CHANNELS: Final[tuple[str, ...]] = ("R", "G", "B")


def _require_finite_number(value: object, name: str) -> float:
    """Return ``value`` as a finite float, rejecting booleans and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    return number


def _require_positive_number(value: object, name: str) -> float:
    """Return ``value`` as a finite, strictly positive float."""
    number = _require_finite_number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number; got {value!r}")
    return number


def _require_nan_or_finite(value: object, name: str) -> float:
    """Return ``value`` as a finite float or NaN, rejecting infinities."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number or NaN; got {value!r}")
    number = float(value)
    if math.isinf(number):
        raise ValueError(f"{name} must be finite or NaN; got {value!r}")
    return number


def _require_positive_integer(value: object, name: str) -> int:
    """Return ``value`` as an integer of at least one, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer of at least 1; got {value!r}")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be an integer of at least 1; got {value!r}")
    return integer


def _require_window(value: object) -> str:
    """Return the window name, one of ``"hamming"`` or ``"none"``."""
    if not isinstance(value, str) or value not in _WINDOW_NAMES:
        raise ValueError(f"window must be one of {_WINDOW_NAMES}; got {value!r}")
    return value


def _require_channel_name(value: object) -> str:
    """Return a one-letter RAW channel name in upper case."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"channel must be a non-empty string; got {value!r}")
    name = value.strip().upper()
    if len(name) != 1:
        raise ValueError(f"channel must be a single letter; got {value!r}")
    return name


def _as_real_vector(values: object, name: str) -> NDArray[np.float64]:
    """Return a copied, 1-D float64 array with a clear error on bad input."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 1-D array of real numbers; {exc}") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain real numbers, not boolean values")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be real; complex values are not supported")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain real numbers; got dtype {array.dtype}")
    return np.array(array, dtype=np.float64, copy=True)


def _as_2d_image(image: object) -> NDArray[np.float64]:
    """Return ``image`` as a finite float64 2-D array."""
    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"image must be a real 2-D array-like; {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"image must be 2-D (H, W); got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"image must contain at least one pixel; got shape {array.shape}")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError("image must contain real numbers, not boolean values")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("image must be real; complex values are not supported")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"image must contain real numbers; got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values (no NaN or infinity)")
    return np.asarray(array, dtype=np.float64)


@dataclass(frozen=True)
class MTFCurve:
    """A sampled modulation transfer function.

    Parameters
    ----------
    frequency_cp:
        One-dimensional spatial frequencies in cycles/pixel. Must be finite,
        non-negative, strictly increasing, and start at exactly zero. Zero
        frequency is the reference point at which the transfer function is one.
    mtf:
        Modulation transfer factors at the same frequencies. Finite and
        non-negative; ``mtf[0]`` must be one within ``1e-6`` of DC
        normalization. Values above one are accepted because a caller may hold
        an unclipped estimate, but a passive system cannot amplify modulation
        and :func:`slanted_edge_mtf` clips its own result to ``[0, 1]``.

    Notes
    -----
    Both inputs are converted to float64 and **copied**, so mutating the caller's
    array afterwards cannot change a curve that may be shared between a result,
    a report, and a plot. The dataclass is frozen; the arrays it holds are not
    marked read-only, so treat them as immutable.

    A real curve runs from 0 to 0.5 cycles/pixel (the native Nyquist limit); the
    class itself does not enforce an upper bound because a caller may keep the
    oversampled analysis band for diagnostics.
    """

    frequency_cp: NDArray[np.float64]
    mtf: NDArray[np.float64]

    def __post_init__(self) -> None:
        frequency = _as_real_vector(self.frequency_cp, "frequency_cp")
        response = _as_real_vector(self.mtf, "mtf")
        if frequency.size != response.size:
            raise ValueError(
                "frequency_cp and mtf must have equal length; "
                f"got {frequency.size} and {response.size}"
            )
        if not np.all(np.isfinite(frequency)):
            raise ValueError("frequency_cp must contain only finite values")
        if not np.all(np.isfinite(response)):
            raise ValueError("mtf must contain only finite values")
        if frequency[0] != 0.0:
            raise ValueError(f"frequency_cp must start at 0 cycles/pixel; got {frequency[0]!r}")
        if np.any(frequency < 0.0):
            raise ValueError("frequency_cp must be non-negative")
        if np.any(np.diff(frequency) <= 0.0):
            raise ValueError("frequency_cp must be strictly increasing")
        if np.any(response < 0.0):
            raise ValueError("mtf must be non-negative")
        if abs(float(response[0]) - 1.0) > _MTF_NORMALIZATION_TOLERANCE:
            raise ValueError(
                "mtf[0] must equal 1 within "
                f"{_MTF_NORMALIZATION_TOLERANCE:g} (DC normalization); got {response[0]!r}"
            )
        object.__setattr__(self, "frequency_cp", frequency)
        object.__setattr__(self, "mtf", response)


@dataclass(frozen=True)
class MTFResult:
    """A slanted-edge measurement: the curve, its crossings, and its geometry.

    Parameters
    ----------
    mtf50_cp, mtf20_cp:
        First frequencies, in cycles/pixel, where the curve falls to one half
        and one fifth. ``nan`` when the curve does not reach that level inside
        the measured band (``0 <= f <= nyquist_cp``), which is a legitimate
        "not reached within the measured band" answer for a very sharp system.
    mtf50_cyc_per_deg, mtf20_cyc_per_deg:
        The same crossings in cycles/degree, using the camera's angular pixel
        pitch. ``nan`` when either the crossing is undefined or no focal length
        was supplied; the frequency fields are never invented.
    curve:
        The sampled :class:`MTFCurve` the crossings were interpolated from.
    edge_angle_deg:
        Estimated tilt of the measured edge, positive when the edge moves
        rightwards as the row index increases. Finite.
    nyquist_cp:
        Upper edge of the measured band in cycles/pixel; 0.5 by default. The
        finite crossing fields must lie inside ``[0, nyquist_cp]``.

    Notes
    -----
    The fields are stored as given; the class does not recompute the crossings
    from the curve, so a caller who edits a curve before constructing the result
    is responsible for keeping the summary consistent. The dataclass is frozen;
    the curve's arrays are value data owned by the result.
    """

    mtf50_cp: float
    mtf20_cp: float
    mtf50_cyc_per_deg: float
    mtf20_cyc_per_deg: float
    curve: MTFCurve
    edge_angle_deg: float
    nyquist_cp: float = _NYQUIST_CP

    def __post_init__(self) -> None:
        if not isinstance(self.curve, MTFCurve):
            raise ValueError(f"curve must be an MTFCurve; got {type(self.curve).__name__}")
        nyquist = _require_positive_number(self.nyquist_cp, "nyquist_cp")
        object.__setattr__(self, "nyquist_cp", nyquist)
        for name in ("mtf50_cp", "mtf20_cp"):
            value = _require_nan_or_finite(getattr(self, name), name)
            if not math.isnan(value) and not 0.0 <= value <= nyquist:
                raise ValueError(
                    f"{name} must lie in [0, {nyquist:g}] cycles/pixel or be NaN; got {value!r}"
                )
            object.__setattr__(self, name, value)
        for name in ("mtf50_cyc_per_deg", "mtf20_cyc_per_deg"):
            value = _require_nan_or_finite(getattr(self, name), name)
            if not math.isnan(value) and value < 0.0:
                raise ValueError(f"{name} must be non-negative or NaN; got {value!r}")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "edge_angle_deg",
            _require_finite_number(self.edge_angle_deg, "edge_angle_deg"),
        )


def _horizontal_sobel(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the horizontal Sobel gradient with edge-replicated borders.

    The kernel ``[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]`` is the 3 x 3
    least-squares plane fit's horizontal derivative. Its absolute scale is
    irrelevant here because the angle estimator squares and normalizes the
    response; only the column of the gradient peak matters. Replicating the
    border keeps the first and last rows meaningful instead of injecting a
    zero-gradient frame.
    """
    padded = np.pad(image, 1, mode="edge")
    gradient: NDArray[np.float64] = (
        -padded[:-2, :-2]
        - 2.0 * padded[1:-1, :-2]
        - padded[2:, :-2]
        + padded[:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
    )
    return gradient


def _estimate_edge_angle(region: NDArray[np.float64]) -> float:
    """Estimate a near-vertical edge's tilt from horizontal gradients.

    Each row's horizontal Sobel response is squared into an energy profile.
    Columns below 10 % of that row's peak are discarded, which rejects noise
    and chart texture away from the edge; the remaining energy weights a column
    centroid. A weighted least-squares line through the row centroids gives
    ``dx / dy``, and the tilt is ``degrees(atan(dx / dy))``. Squares make the
    estimate independent of whether the edge is bright-to-dark or the reverse,
    and the same convention as
    :func:`camsim.scene.targets.estimate_edge_angle` makes a positive angle
    move rightwards as the row index increases.

    Raises
    ------
    ValueError
        If fewer than two rows carry a gradient, or the fitted angle is not
        finite or is too close to horizontal for a near-vertical analysis. The
        estimator deliberately does not wrap around to a horizontal orientation.
    """
    height, width = region.shape
    gradient = _horizontal_sobel(region)
    weights = gradient * gradient
    columns = np.arange(width, dtype=np.float64)
    rows: list[float] = []
    centroids: list[float] = []
    energies: list[float] = []
    for row in range(height):
        row_weights = weights[row]
        peak = float(row_weights.max())
        if peak <= 0.0:
            continue
        row_weights = np.where(row_weights >= _EDGE_GRADIENT_THRESHOLD * peak, row_weights, 0.0)
        total = float(row_weights.sum())
        if total <= 0.0:
            continue
        rows.append(float(row))
        centroids.append(float(row_weights @ columns) / total)
        energies.append(total)

    if len(rows) < _MIN_EDGE_ROWS:
        raise ValueError(
            "image has no usable horizontal gradient; a slanted edge must cross "
            "the analysis region after the border margin is removed"
        )
    coefficients = np.polyfit(
        np.asarray(rows, dtype=np.float64),
        np.asarray(centroids, dtype=np.float64),
        1,
        w=np.sqrt(np.asarray(energies, dtype=np.float64)),
    )
    angle_deg = math.degrees(math.atan(float(coefficients[0])))
    if not math.isfinite(angle_deg):
        raise ValueError("the slanted-edge line fit did not produce a finite angle")
    if abs(angle_deg) >= _MAX_EDGE_ANGLE_DEG:
        raise ValueError(
            f"estimated edge angle {angle_deg:.1f} degrees is too close to horizontal; "
            "the slanted-edge analysis assumes a near-vertical edge"
        )
    return angle_deg


def _oversampled_esf(
    region: NDArray[np.float64],
    angle_deg: float,
    oversampling: int,
) -> tuple[NDArray[np.float64], float]:
    """Return the oversampled edge-spread function and its sample spacing.

    Every pixel of ``region`` is projected onto the edge normal,
    ``u = x cos(theta) - y sin(theta)``, with ``x`` and ``y`` counted from the
    region's top-left pixel centre. The projection is binned into half-open
    intervals of width ``1 / oversampling`` pixel: bin ``k`` is centred on
    ``k * (1 / oversampling)`` and covers ``[centre - width/2, centre +
    width/2)``, so a pixel exactly on a boundary joins the upper bin. Bin means
    are weighted by occupancy; a bin that no pixel reaches (possible only at the
    ends of the projection near a corner) is filled by linear interpolation
    between the nearest populated bins so the derivative never sees a hole.

    The bin grid's absolute offset does not matter: shifting every ``u`` by a
    constant multiplies the Fourier transform by a linear phase but leaves the
    MTF magnitude unchanged.

    Returns
    -------
    esf:
        The binned edge profile, one sample per sub-pixel interval. It is not
        renormalized; its total rise becomes the LSF's DC term.
    spacing:
        ``1 / oversampling`` pixels, the spacing the LSF derivative and the FFT
        frequency axis must use.
    """
    height, width = region.shape
    spacing = 1.0 / oversampling
    theta = math.radians(angle_deg)
    row_positions = np.arange(height, dtype=np.float64)
    column_positions = np.arange(width, dtype=np.float64)
    u = np.cos(theta) * column_positions[None, :] - np.sin(theta) * row_positions[:, None]

    flat = u.ravel()
    bin_index = np.floor(flat / spacing + 0.5).astype(np.int64)
    first = int(bin_index.min())
    size = int(bin_index.max()) - first + 1
    offset = bin_index - first

    counts: NDArray[np.float64] = np.bincount(offset, minlength=size).astype(np.float64)
    sums: NDArray[np.float64] = np.asarray(
        np.bincount(offset, weights=region.ravel(), minlength=size), dtype=np.float64
    )
    occupied = counts > 0
    if not occupied.any():
        raise ValueError("the analysis region produced no edge-spread-function bins")

    esf = np.empty(size, dtype=np.float64)
    esf[occupied] = sums[occupied] / counts[occupied]
    if not occupied.all():
        indices = np.arange(size, dtype=np.float64)
        esf[~occupied] = np.interp(indices[~occupied], indices[occupied], esf[occupied])
    return esf, spacing


def _central_difference(samples: NDArray[np.float64], spacing: float) -> NDArray[np.float64]:
    """Differentiate at ``spacing`` with a central difference in the interior.

    The two end samples use one-sided differences because a central stencil
    would need neighbours outside the array. At the ends the ESF has already
    reached its flat plateaus, and the Hamming window tapers those samples, so
    the one-sided estimator there has negligible effect on the result.
    """
    count = samples.size
    derivative = np.empty(count, dtype=np.float64)
    derivative[1:-1] = (samples[2:] - samples[:-2]) / (2.0 * spacing)
    derivative[0] = (samples[1] - samples[0]) / spacing
    derivative[-1] = (samples[-1] - samples[-2]) / spacing
    return derivative


def _esf_to_mtf(
    esf: NDArray[np.float64],
    spacing: float,
    window: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Differentiate the ESF, transform the LSF, and correct the estimator.

    The pipeline is: central difference at ``spacing``; optional Hamming taper
    over the whole LSF; FFT magnitude; division by the DC magnitude; division by
    the normalized central-difference response ``|sinc(2 pi f spacing)|`` with a
    small-denominator floor; truncation to ``[0, 0.5]`` cycles/pixel; and a clip
    to ``[0, 1]``. The clip only removes window/noise excursions: a genuine
    zero, as in an Airy or box transfer function, is left at zero and the curve
    after it is allowed to rise again.

    Returns
    -------
    frequency_cp, mtf:
        Equal-length float64 arrays starting at exactly zero frequency with
        ``mtf[0] == 1``, suitable for :class:`MTFCurve`.
    """
    if esf.size < 3:
        raise ValueError(
            "the analysis region is too small to differentiate its edge-spread "
            f"function; got {esf.size} oversampled bins"
        )
    lsf = _central_difference(esf, spacing)
    if window == "hamming":
        taps = np.arange(lsf.size, dtype=np.float64)
        taper = _HAMMING_PEDESTAL - _HAMMING_MODULATION * np.cos(
            2.0 * math.pi * taps / (lsf.size - 1)
        )
        lsf = lsf * taper

    spectrum = np.abs(np.fft.rfft(lsf))
    frequency_cp = np.fft.rfftfreq(lsf.size, d=spacing)
    dc = float(spectrum[0])
    if not math.isfinite(dc) or dc <= 0.0:
        raise ValueError(
            "the edge-spread function has no measurable contrast; its derivative sums to zero"
        )
    magnitude = spectrum / dc

    # np.sinc(x) is sin(pi x) / (pi x), so np.sinc(2 f d) is
    # |sin(2 pi f d) / (2 pi f d)|, the normalized response of the central
    # difference used above. Dividing it out removes the estimator's own
    # roll-off without touching the system's pixel aperture.
    derivative_response = np.abs(np.sinc(2.0 * frequency_cp * spacing))
    magnitude = magnitude / np.maximum(derivative_response, _DERIVATIVE_SINC_FLOOR)

    within_band = frequency_cp <= _NYQUIST_CP + _FREQUENCY_TOLERANCE
    frequencies: NDArray[np.float64] = np.ascontiguousarray(frequency_cp[within_band])
    response: NDArray[np.float64] = np.asarray(
        np.clip(magnitude[within_band], 0.0, 1.0), dtype=np.float64
    )
    if frequencies.size < 2:
        raise ValueError(
            "the analysis region produced fewer than two frequencies inside the "
            "native Nyquist band; increase the region size or the oversampling"
        )
    frequencies[0] = 0.0
    response[0] = 1.0
    return frequencies, response


def _first_crossing(
    frequency_cp: NDArray[np.float64],
    mtf: NDArray[np.float64],
    level: float,
) -> float:
    """Return the first frequency where ``mtf`` falls below ``level``.

    The crossing is linearly interpolated between the two bracketing samples.
    ``nan`` is returned when no sample inside the measured band is below the
    level: the curve is then "not reached within the measured band", which is
    the honest answer for a system sharper than Nyquist, not an error.
    """
    below = np.flatnonzero(mtf < level)
    if below.size == 0:
        return math.nan
    index = int(below[0])
    if index == 0:
        return math.nan
    lower_mtf = float(mtf[index - 1])
    upper_mtf = float(mtf[index])
    lower_frequency = float(frequency_cp[index - 1])
    upper_frequency = float(frequency_cp[index])
    if upper_mtf == lower_mtf:
        return upper_frequency
    fraction = (level - lower_mtf) / (upper_mtf - lower_mtf)
    return lower_frequency + (upper_frequency - lower_frequency) * fraction


def _pixels_per_degree(pixel_pitch_um: float, focal_length_mm: float) -> float:
    """Return how many sensor pixels span one degree of field angle.

    The paraxial relation ``angle = pixel_pitch / focal_length`` is exact to
    better than 0.01 % for the sub-degree fields of consumer lenses; using the
    full arctangent keeps the conversion honest for long lenses without
    measurably changing the result. Both lengths are converted to the same unit
    (millimetres) before the ratio.
    """
    angular_pitch_deg = math.degrees(math.atan(pixel_pitch_um * 1e-3 / focal_length_mm))
    if not math.isfinite(angular_pitch_deg) or angular_pitch_deg <= 0.0:
        raise ValueError(
            "pixel_pitch_um and focal_length_mm do not produce a finite angular pixel pitch"
        )
    return 1.0 / angular_pitch_deg


def extract_raw_plane(
    raw: RawFrame,
    channel: str = "G",
) -> tuple[NDArray[np.float64], float]:
    """Return one colour phase of a RAW mosaic and its sampling pitch multiplier.

    Parameters
    ----------
    raw:
        A :class:`~camsim.core.frames.RawFrame`. Its data are ADC codes
        including the black-level pedestal; that constant offset does not affect
        an MTF measurement because the LSF is normalized at DC, but callers who
        want calibrated signal must subtract ``raw.black_level_dn`` themselves.
    channel:
        ``"G"`` (the default), ``"R"``, or ``"B"`` for a Bayer frame, and
        ``"M"`` for a monochrome frame. Matching is case-insensitive. The letter
        selects one phase of that colour: the **first occurrence in row-major
        order** of the 2 x 2 tile, never an average of the two green sites.

    Returns
    -------
    plane:
        The selected phase as a float64 **copy** of shape
        ``data[dy::2, dx::2]``. Mutating it cannot corrupt the RAW frame.
    pitch_multiplier:
        ``2.0`` for a Bayer phase and ``1.0`` for a mono frame. This is not a
        micrometre value: the caller passes ``raw.pixel_pitch_um *
        pitch_multiplier`` as ``pixel_pitch_um`` to :func:`slanted_edge_mtf`.

    Raises
    ------
    TypeError
        If ``raw`` is not a :class:`~camsim.core.frames.RawFrame`.
    ValueError
        For an unknown channel, a mono frame without ``channel="M"``, an RCCC
        frame (advanced automotive metrics are deferred), or a Bayer frame too
        small to contain the requested phase.

    Notes
    -----
    A Bayer colour plane samples a rectangular lattice whose sites are two
    original pixels apart in each axis, so its Nyquist falls to 0.25 cycles per
    original pixel. The green sites form a quincunx: the two green phases are
    offset by one pixel diagonally, and averaging them is a 2 x 2 box filter
    that silently low-passes the measurement. Using one phase keeps a
    rectangular lattice, which is what the edge projection assumes. With an odd
    number of rows or columns the last row or column belongs to only one phase,
    so the three colour planes can differ in shape by one; the returned shape is
    exactly the slice's and callers must not assume a common size.
    """
    if not isinstance(raw, RawFrame):
        raise TypeError(f"raw must be a RawFrame; got {type(raw).__name__}")
    name = _require_channel_name(channel)

    if raw.cfa == CFAPattern.MONO:
        if name != _MONO_CHANNEL:
            raise ValueError(
                f"MONO frames have one full-resolution plane; channel must be 'M'; got {name!r}"
            )
        return raw.data.astype(np.float64), _MONO_PITCH_MULTIPLIER

    if raw.cfa == CFAPattern.RCCC:
        raise ValueError(
            "RCCC frames are not supported by the slanted-edge MTF; "
            "advanced RCCC metrics are deferred"
        )

    tile = raw.cfa.value
    offsets = [
        (row_offset, column_offset)
        for row_offset in range(2)
        for column_offset in range(2)
        if tile[2 * row_offset + column_offset] == name
    ]
    if not offsets:
        available = sorted({letter for letter in tile if letter in _BAYER_CHANNELS})
        raise ValueError(f"channel must be one of {available} for a {tile} frame; got {name!r}")
    row_offset, column_offset = offsets[0]
    plane = raw.data[row_offset::2, column_offset::2].astype(np.float64)
    if plane.size == 0:
        raise ValueError(
            f"the {name!r} phase at tile offset {(row_offset, column_offset)} has no samples "
            f"in a {raw.data.shape} frame"
        )
    return plane, _BAYER_PITCH_MULTIPLIER


def slanted_edge_mtf(
    image_2d: ArrayLike,
    pixel_pitch_um: float,
    focal_length_mm: float | None = None,
    oversampling: int = 4,
    window: str = "hamming",
) -> MTFResult:
    """Measure the MTF of a linear-light image containing one slanted edge.

    Parameters
    ----------
    image_2d:
        A real, finite, 2-D ``(H, W)`` array. The input must be **linear light**:
        a display-referred (gamma-encoded) image has a non-linear edge profile
        and biases the result. There is exactly one edge; a region containing
        several edges, chart texture, or strong shading contaminates the
        projection. Multi-channel data must be reduced to one plane by the
        caller, for example with :func:`extract_raw_plane`.
    pixel_pitch_um:
        Sample pitch of ``image_2d`` in micrometres, finite and positive. For a
        Bayer phase this is ``raw.pixel_pitch_um * 2``; for a mono frame it is
        ``raw.pixel_pitch_um``.
    focal_length_mm:
        Actual lens focal length in millimetres, finite and positive, or
        ``None``. When supplied, crossings are also reported in cycles/degree;
        when ``None``, the cycles/degree fields are ``nan`` and the
        cycles/pixel fields are unaffected.
    oversampling:
        Analysis samples per pixel, an integer of at least one; four is the
        conventional value. Each pixel is projected to a position along the
        edge normal and binned at ``1 / oversampling`` pixel, so higher values
        give a finer ESF and frequency grid at the cost of more bins per
        transition (and, at one sample per pixel, an unusable derivative
        response at Nyquist).
    window:
        ``"hamming"`` (default) tapers the LSF to suppress the rectangular
        window's sidelobes; ``"none"`` leaves it rectangular. The Hamming taper
        widens the effective LSF slightly, which lowers high-frequency MTF by a
        few thousandths of a cycle.

    Returns
    -------
    MTFResult
        The curve from 0 to 0.5 cycles/pixel, the interpolated MTF50/MTF20
        crossings (NaN when not reached inside the band), the same crossings in
        cycles/degree when a focal length is supplied, the estimated edge angle,
        and the Nyquist limit.

    Raises
    ------
    ValueError
        If the image is not a finite real 2-D array, if a numeric parameter is
        invalid, if the border margin leaves less than a 16 x 16 analysis region,
        or if the remaining region has no usable near-vertical edge gradient.

    Notes
    -----
    This is an ISO 12233-style measurement, not a certified implementation. A
    margin of at least four pixels and about 2 % of the shorter side is removed
    before analysis, because convolution padding and vignetting make the frame
    border unrepresentative. The edge angle is estimated from the same region
    that supplies the ESF, so a chart whose edge is not straight across the whole
    frame should be cropped to a locally straight segment first.

    The curve is clipped to ``[0, 1]``. Values above one can only come from noise
    or window leakage, since a passive system cannot amplify modulation; a
    genuine zero (a box or Airy transfer function's first zero) is preserved and
    the curve is allowed to rise again after it. The DC term is exactly one by
    construction. The frequency grid resolution is ``oversampling / N``
    cycles/pixel for an ``N``-bin ESF, typically a few thousandths of a cycle
    per pixel, and the crossings are linearly interpolated on that grid.
    """
    image = _as_2d_image(image_2d)
    pitch = _require_positive_number(pixel_pitch_um, "pixel_pitch_um")
    focal = (
        None
        if focal_length_mm is None
        else _require_positive_number(focal_length_mm, "focal_length_mm")
    )
    factor = _require_positive_integer(oversampling, "oversampling")
    window_name = _require_window(window)

    height, width = image.shape
    margin = max(_BORDER_MARGIN_MIN_PX, math.ceil(_BORDER_MARGIN_FRACTION * min(height, width)))
    analysis_height = height - 2 * margin
    analysis_width = width - 2 * margin
    if analysis_height < _MIN_ANALYSIS_PX or analysis_width < _MIN_ANALYSIS_PX:
        raise ValueError(
            f"image shape {(height, width)} is too small for a slanted-edge analysis: "
            f"a {margin}-pixel border margin leaves {analysis_height}x{analysis_width}, "
            f"and at least {_MIN_ANALYSIS_PX}x{_MIN_ANALYSIS_PX} is required"
        )
    region = image[margin : height - margin, margin : width - margin]

    angle_deg = _estimate_edge_angle(region)
    esf, spacing = _oversampled_esf(region, angle_deg, factor)
    frequency_cp, mtf = _esf_to_mtf(esf, spacing, window_name)

    curve = MTFCurve(frequency_cp, mtf)
    mtf50_cp = _first_crossing(curve.frequency_cp, curve.mtf, 0.5)
    mtf20_cp = _first_crossing(curve.frequency_cp, curve.mtf, 0.2)
    if focal is None:
        mtf50_cyc_per_deg = math.nan
        mtf20_cyc_per_deg = math.nan
    else:
        pixels_per_degree = _pixels_per_degree(pitch, focal)
        mtf50_cyc_per_deg = mtf50_cp * pixels_per_degree
        mtf20_cyc_per_deg = mtf20_cp * pixels_per_degree

    return MTFResult(
        mtf50_cp=mtf50_cp,
        mtf20_cp=mtf20_cp,
        mtf50_cyc_per_deg=mtf50_cyc_per_deg,
        mtf20_cyc_per_deg=mtf20_cyc_per_deg,
        curve=curve,
        edge_angle_deg=angle_deg,
    )
