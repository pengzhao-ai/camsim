"""Flat-patch signal-to-noise ratio in linear light: spatial and temporal.

Signal-to-noise ratio measures how far a signal stands above its own random
fluctuations. For a flat patch measured in a linear signal quantity (electrons,
DN after black-level subtraction, or linear scene radiance) the ratio is
``SNR = mean / std``, reported in decibels as ``SNR_dB = 20 log10(mean / std)``.
The factor 20, rather than 10, follows because both terms are amplitudes of the
same quantity: a factor of two in signal is about 6.02 dB, whereas a power ratio
would use 10 log10. This module never converts to power and never uses 10 log10.

Two estimators
--------------
* **Spatial (single frame).** The sample standard deviation of every pixel in a
  rectangular ROI, with ``ddof=1`` (the unbiased variance estimator), measures
  the total pixel-to-pixel variation of one frame. That variation combines
  random noise frozen at the capture instant, fixed-pattern noise (per-pixel
  gain and offset differences, DSNU and PRNU), and any real scene or
  illumination texture inside the ROI. Spatial SNR therefore overstates the
  random noise of a sensor with strong fixed-pattern noise, and a patch that
  straddles chart texture reports that texture as noise.
* **Temporal (multi-frame).** For each pixel independently, take its mean and
  sample variance over ``F`` frames. The reported mean is the average of those
  per-pixel temporal means; the reported standard deviation is
  ``sqrt(mean(per-pixel temporal variances))``, the pooled random noise. A
  static per-pixel pattern shifts every frame equally, so it contributes to the
  means but cancels in the temporal variances; fixed-pattern variation is
  excluded by construction. Temporal SNR is the cleaner estimate of random
  noise (shot, read, and dark-current noise), but it needs a stable scene and
  illumination and at least two frames, and a drifting scene contaminates it
  with motion.

Luma
----
A colour patch also has a luminance, the Rec. 709 weighting of linear R, G, B:
``Y = 0.2126 R + 0.7152 G + 0.0722 B``. The weights are applied to **linear**
channel values, matching the opponent-space helper in
:mod:`camsim.isp.denoise` and Rec. 709's derivation from CIE 1931 luminance
under D65. Applying them to gamma-encoded values computes a different,
non-standard quantity.

Validity
--------
A measurement is valid only when every sample is finite, the mean is strictly
positive, the standard deviation is strictly positive, and at least two samples
exist. A flat patch - including a saturated all-ones patch - has ``std == 0``
and no SNR. An all-zero or negative-mean patch has no positive signal. When the
measurement is invalid the computed mean and std are still reported, because
they remain meaningful, and ``snr_db`` is ``NaN``.

Black level, clipping, and units
--------------------------------
The metric measures whatever quantity the caller supplies. A RAW black-level
pedestal is a constant offset: it raises the mean without changing the standard
deviation, so it inflates the reported SNR (a 64 DN pedestal on a 10 DN signal
with 1 DN of noise adds about 17 dB). Subtract the black level first. ADC
saturation and clipping diagnostics belong to the RAW metadata (defect and
saturation masks, DN histograms), not to this module: a saturated flat patch is
caught because its std is zero, never by comparing sample values against a
full-scale threshold, because the metric does not know the bit depth or scaling
of its input. Display-encoded (gamma) input is likewise out of scope: the tone
curve compresses fluctuations near the top of the range, so the measured
standard deviation is not the sensor's.

Real-world ranges
-----------------
Camera SNR on a mid-grey patch runs roughly 10-50 dB: about 10-20 dB in dim
indoor light, 35-45 dB at a well-exposed mid-grey in good light, and higher only
on bright low-noise patches. Flat patches are typically 16 x 16 to 128 x 128
pixels and are inset from the painted patch edge so that optical blur and
demosaic artefacts never enter the ROI. Temporal stacks are 4-32 frames; with
``F`` frames, a patch of ``N`` pixels estimates the pooled standard deviation
with roughly ``1 / sqrt(2 * F * N)`` relative uncertainty, so a 16 x 16 patch
over eight frames determines the noise to a few percent.

Failure artefacts guarded here
------------------------------
* **Display-encoded input.** Gamma compresses high-light noise and biases every
  SNR computed from it.
* **Clipped or saturated patches.** No dispersion remains; std is zero.
* **Zero or negative mean.** A black patch, or one whose black level was
  over-subtracted, has no positive signal and the ratio is meaningless.
* **Single-pixel or constant patches.** Fewer than two samples cannot estimate
  dispersion.
* **Fixed-pattern noise masquerading as temporal noise.** Pooling all pixels and
  frames into one sample, or taking the std of the per-pixel temporal means,
  makes a static pattern look like noise. This module pools per-pixel temporal
  variances instead.
* **Single-frame versus temporal confusion.** A single frame cannot separate
  random noise from fixed-pattern noise. The two functions report different
  quantities and are not interchangeable.

The module belongs to the independent metrics layer: it imports NumPy, the
standard library, and :mod:`camsim.metrics.chart` only - never the sensor, ISP,
scene, or optics packages - so the measurement harness cannot depend on the
model it is meant to judge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.metrics.chart import PatchROI, extract_patch

__all__ = ["SNRResult", "snr_channels", "snr_patch", "snr_temporal"]

# Rec. 709 luminance weights, applied to linear-light channel values. The three
# sum to one, so a neutral (R = G = B) patch keeps its level in luma.
_REC709_WEIGHTS: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)

# Channel indices are reported with the conventional colour names; the order is
# fixed so that `snr_channels` is deterministic and report-friendly.
_RGB_CHANNELS: Final[tuple[str, str, str]] = ("R", "G", "B")
_LUMA_CHANNEL: Final[str] = "luma"
_MONO_CHANNEL: Final[str] = "mono"
_MONO_ALIASES: Final[frozenset[str]] = frozenset({"mono", "m"})
_RGB_ALIASES: Final[dict[str, int]] = {"r": 0, "g": 1, "b": 2}

# A 3-D image is a colour image with three planes, or a single-plane array with
# a degenerate channel axis. The brief's contract tests measure (H, W, 1) data
# by index, so one channel is accepted; two, four, or more are ambiguous and
# rejected rather than silently reinterpreted.
_SUPPORTED_COLOUR_CHANNELS: Final[tuple[int, ...]] = (1, 3)

# Two frames are the minimum for a sample variance (one degree of freedom).
_MIN_FRAMES: Final[int] = 2

_DECIBEL_FACTOR: Final[float] = 20.0


def _require_finite_number(value: object, name: str) -> float:
    """Return ``value`` as a finite float, rejecting booleans and non-numbers."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    return number


def _require_non_negative_number(value: object, name: str) -> float:
    """Return ``value`` as a finite, non-negative float."""
    number = _require_finite_number(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value!r}")
    return number


def _require_nan_or_finite(value: object, name: str) -> float:
    """Return ``value`` as a finite float or NaN, rejecting infinities."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number or NaN; got {value!r}")
    number = float(value)
    if math.isinf(number):
        raise ValueError(f"{name} must be finite or NaN; got {value!r}")
    return number


def _require_bool(value: object, name: str) -> bool:
    """Return ``value`` as a Python bool, accepting NumPy booleans."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a bool; got {value!r}")
    return bool(value)


def _require_channel_text(value: object) -> str:
    """Return a stripped, non-empty channel label or raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"channel must be a non-empty string; got {value!r}")
    return value.strip()


@dataclass(frozen=True)
class SNRResult:
    """A flat-patch signal-to-noise measurement.

    Parameters
    ----------
    mean:
        Patch mean in the caller's linear signal units. Finite. Spatial
        measurements use the arithmetic mean of the ROI; temporal measurements
        use the mean of the per-pixel temporal means.
    std:
        Sample standard deviation in the same units. Finite and non-negative.
        Spatial measurements use ``ddof=1`` across the ROI; temporal
        measurements use ``sqrt(mean(per-pixel temporal variances))``, which
        excludes static fixed-pattern variation.
    snr_db:
        ``20 log10(mean / std)`` when the measurement is valid, ``NaN``
        otherwise. Infinity is rejected: an infinite ratio can only come from a
        zero or non-finite denominator. When ``valid`` is ``False`` the
        constructor overwrites the supplied value with ``NaN``, so an invalid
        result can never masquerade as a measured ratio.
    channel:
        ``"R"``, ``"G"``, ``"B"``, ``"luma"``, or ``"mono"``. Non-empty; stored
        stripped of surrounding whitespace.
    valid:
        ``True`` only when the estimator had finite samples, at least two of
        them, a strictly positive mean, and a strictly positive standard
        deviation. A valid result must also carry a finite ``snr_db`` and a
        positive mean and std, because those are exactly the conditions under
        which the ratio exists.

    Notes
    -----
    The dataclass is frozen. A result carries the statistics that produced it,
    so a report can show why a measurement was rejected - for example a flat
    patch with ``std == 0`` and its measured mean - rather than only a NaN.

    The value object validates internal consistency but does not recompute
    ``snr_db`` from ``mean`` and ``std``: a caller who builds one by hand is
    responsible for the arithmetic, while the metric functions always derive
    the three fields together.
    """

    mean: float
    std: float
    snr_db: float
    channel: str
    valid: bool

    def __post_init__(self) -> None:
        mean = _require_finite_number(self.mean, "mean")
        std = _require_non_negative_number(self.std, "std")
        channel = _require_channel_text(self.channel)
        valid = _require_bool(self.valid, "valid")
        snr_db = _require_nan_or_finite(self.snr_db, "snr_db")

        if valid:
            if math.isnan(snr_db):
                raise ValueError(
                    "snr_db must be finite when valid is True; an unmeasurable "
                    "patch must set valid=False and snr_db=NaN"
                )
            if mean <= 0.0 or std <= 0.0:
                raise ValueError(
                    "valid=True requires a positive mean and standard deviation; "
                    f"got mean={mean!r}, std={std!r}"
                )
        else:
            # An invalid measurement has no defined ratio, whatever the caller
            # supplied. Reporting NaN keeps the flag and the number consistent.
            snr_db = math.nan

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "snr_db", snr_db)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "valid", valid)


def _resolve_channel(channel: object, channels: int | None) -> tuple[str, int | None]:
    """Return the result label and colour-plane index for a requested channel.

    ``channels`` is ``None`` for a 2-D mono image and the plane count ``C`` for a
    3-D image. The returned index is ``None`` when the measurement uses the mono
    plane or the computed luma plane, and otherwise selects ``data[..., index]``.

    Channel resolution is deliberately strict: a wrong index silently measures
    the wrong colour, so an unknown label, an out-of-range index, or a colour
    request against a mono image raises :class:`ValueError` instead of guessing.
    Matching of string labels is case-insensitive and ignores surrounding
    whitespace. On a 3-D image, ``None`` means luma (Rec. 709 in linear light),
    indices ``0``/``1``/``2`` and ``"R"``/``"G"``/``"B"`` select one plane, and
    luma requires exactly three planes.
    """
    if channels is None:
        if channel is None:
            return _MONO_CHANNEL, None
        if isinstance(channel, str) and channel.strip().lower() in _MONO_ALIASES:
            return _MONO_CHANNEL, None
        raise ValueError(
            f"channel must be None, 'mono', or 'M' for a 2-D mono image; got {channel!r}"
        )

    if channels not in _SUPPORTED_COLOUR_CHANNELS:
        raise ValueError(f"a 3-D image must have 1 or 3 channels; got {channels} channels")

    label: str
    index: int | None
    if channel is None:
        label, index = _LUMA_CHANNEL, None
    elif isinstance(channel, str):
        name = channel.strip().lower()
        if name == _LUMA_CHANNEL:
            label, index = _LUMA_CHANNEL, None
        elif name in _RGB_ALIASES:
            index = _RGB_ALIASES[name]
            label = _RGB_CHANNELS[index]
        else:
            raise ValueError(
                "channel must be 0, 1, 2, 'R', 'G', 'B', or 'luma' for a 3-D image; "
                f"got {channel!r}"
            )
    elif isinstance(channel, (int, np.integer)) and not isinstance(channel, (bool, np.bool_)):
        candidate = int(channel)
        if not 0 <= candidate < len(_RGB_CHANNELS):
            raise ValueError(f"channel index must be 0, 1, or 2; got {channel!r}")
        index = candidate
        label = _RGB_CHANNELS[candidate]
    else:
        raise ValueError(
            f"channel must be 0, 1, 2, 'R', 'G', 'B', or 'luma' for a 3-D image; got {channel!r}"
        )

    if index is not None and index >= channels:
        raise ValueError(
            f"channel {channel!r} selects plane {index}, but the image has {channels} channel(s)"
        )
    if index is None and channels != 3:
        raise ValueError(f"the luma channel requires exactly 3 colour channels; got {channels}")
    return label, index


def _channel_resolution(frame: NDArray[np.float64], channel: object) -> tuple[str, int | None]:
    """Resolve ``channel`` against one spatial frame of shape ``(H, W)`` or ``(H, W, C)``."""
    channels = None if frame.ndim == 2 else int(frame.shape[2])
    return _resolve_channel(channel, channels)


def _luma_plane(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return Rec. 709 linear luma for a trailing-axis RGB array.

    ``rgb`` may be an image ``(H, W, 3)`` or a frame stack ``(F, H, W, 3)``; the
    weights act on the trailing axis in both cases. Working in float64 avoids
    the intermediate rounding a float32 accumulation would introduce.
    """
    luma = (
        _REC709_WEIGHTS[0] * rgb[..., 0]
        + _REC709_WEIGHTS[1] * rgb[..., 1]
        + _REC709_WEIGHTS[2] * rgb[..., 2]
    )
    return np.asarray(luma, dtype=np.float64)


def _channel_plane(data: NDArray[np.float64], label: str, index: int | None) -> NDArray[np.float64]:
    """Select a colour plane, compute luma, or pass the mono plane through."""
    if index is not None:
        return data[..., index]
    if label == _LUMA_CHANNEL:
        return _luma_plane(data)
    return data


def _snr_db(mean: float, std: float) -> float:
    """Return ``20 log10(mean / std)``, or NaN when the ratio is undefined."""
    if not (math.isfinite(mean) and math.isfinite(std)) or mean <= 0.0 or std <= 0.0:
        return math.nan
    return _DECIBEL_FACTOR * math.log10(mean / std)


def _spatial_result(plane: NDArray[np.float64], label: str) -> SNRResult:
    """Measure one flattened patch plane and package the statistics.

    ``extract_patch`` has already rejected non-finite values, so the finite
    condition is a defensive restatement of the estimator's contract rather than
    a reachable failure path for public callers. A one-pixel patch has no sample
    dispersion: its standard deviation is reported as zero, the sample-count
    rule makes the result invalid, and no degrees of freedom are invented.
    """
    flat = np.ravel(plane)
    mean = float(np.mean(flat))
    std = float(np.std(flat, ddof=1)) if flat.size >= 2 else 0.0
    valid = flat.size >= 2 and bool(np.all(np.isfinite(flat))) and mean > 0.0 and std > 0.0
    return SNRResult(
        mean=mean,
        std=std,
        snr_db=_snr_db(mean, std) if valid else math.nan,
        channel=label,
        valid=valid,
    )


def _temporal_result(plane: NDArray[np.float64], label: str) -> SNRResult:
    """Pool the per-pixel temporal variances of a ``(F, h, w)`` plane.

    The mean is the arithmetic mean of the per-pixel temporal means; the std is
    the square root of the mean per-pixel temporal variance. Averaging variances
    rather than concatenating all pixels and frames is what removes fixed-pattern
    variation: a static pixel offset inflates neither its own variance nor the
    pooled estimate, while adding it to the mean cancels over a zero-mean
    pattern. ``_as_frame_stack`` guarantees at least two frames, so every
    per-pixel variance has one degree of freedom.
    """
    temporal_means = plane.mean(axis=0)
    temporal_variances = plane.var(axis=0, ddof=1)
    mean = float(temporal_means.mean())
    std = float(math.sqrt(float(temporal_variances.mean())))
    valid = (
        plane.shape[0] >= _MIN_FRAMES
        and bool(np.all(np.isfinite(plane)))
        and mean > 0.0
        and std > 0.0
    )
    return SNRResult(
        mean=mean,
        std=std,
        snr_db=_snr_db(mean, std) if valid else math.nan,
        channel=label,
        valid=valid,
    )


def _as_frame_stack(frames: ArrayLike) -> NDArray[np.float64]:
    """Return a finite float64 ``(F, H, W)`` or ``(F, H, W, C)`` frame stack.

    A list of equal-shaped arrays is accepted and converted; ragged or
    non-numeric input is rejected. At least two frames are required, because a
    single frame has no temporal sample variance - that is a spatial
    measurement, not a temporal one.
    """
    try:
        array = np.asarray(frames)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frames must be a stack of real arrays; {exc}") from exc
    if array.ndim not in (3, 4):
        raise ValueError(
            f"frames must be a stack of shape (F, H, W) or (F, H, W, C); got shape {array.shape}"
        )
    frame_count = int(array.shape[0])
    if frame_count < _MIN_FRAMES:
        raise ValueError(f"frames must contain at least two frames; got {frame_count}")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError("frames must contain real numbers, not boolean values")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("frames must be real; complex values are not supported")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"frames must contain real numbers; got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError("frames must contain only finite values (no NaN or infinity)")
    return np.asarray(array, dtype=np.float64)


def snr_patch(
    image: ArrayLike,
    roi: PatchROI,
    channel: int | str | None = None,
) -> SNRResult:
    """Measure the spatial SNR of one flat patch in one linear-light frame.

    Parameters
    ----------
    image:
        Real, finite ``(H, W)`` mono or ``(H, W, C)`` colour array-like. The
        values must be **linear light** in any consistent unit - electrons,
        black-level-subtracted DN, or linear radiance. A display-encoded image
        has a non-linear noise statistic and must be linearized first. The
        black-level pedestal must already be subtracted: it raises the mean
        without changing the std and therefore inflates SNR.
    roi:
        A :class:`~camsim.metrics.chart.PatchROI` lying fully inside the image.
        Extraction and bounds checking are delegated to
        :func:`~camsim.metrics.chart.extract_patch`.
    channel:
        For a 2-D image, ``None``, ``"mono"``, or ``"M"`` (case-insensitive),
        reported as ``"mono"``. For a 3-D image, ``None`` (or ``"luma"``) for
        Rec. 709 linear luma, and ``0``/``1``/``2`` or ``"R"``/``"G"``/``"B"``
        for one plane, with string labels case-insensitive. A 3-D image with a
        single-plane channel axis accepts only its sole plane by index; an RGB
        request against a 2-D image, an out-of-range index, and an unknown label
        raise :class:`ValueError`.

    Returns
    -------
    SNRResult
        The patch mean, the sample standard deviation (``ddof=1``) over every
        pixel of the selected plane, ``snr_db = 20 log10(mean / std)``, the
        channel label, and the validity flag. When the patch is flat, black,
        negative-mean, or a single pixel, the mean and std are still reported and
        ``snr_db`` is ``NaN`` with ``valid=False``.

    Raises
    ------
    TypeError
        If ``roi`` is not a :class:`~camsim.metrics.chart.PatchROI`.
    ValueError
        If the image is not a finite real 2-D or 3-D array, the ROI extends past
        its boundary, the channel is invalid for the image, or a 3-D image has
        an unsupported number of channels.

    Notes
    -----
    A single frame cannot separate random noise from fixed-pattern noise: every
    static pixel-to-pixel difference enters this estimate. Use
    :func:`snr_temporal` with a stable scene when the random component is what
    matters. The estimator is deterministic and does not modify the input.
    """
    patch = np.asarray(extract_patch(image, roi), dtype=np.float64)
    label, index = _channel_resolution(patch, channel)
    return _spatial_result(_channel_plane(patch, label, index), label)


def snr_temporal(
    frames: ArrayLike,
    roi: PatchROI,
    channel: int | str | None = None,
) -> SNRResult:
    """Measure the temporal SNR of a flat patch over a stable frame stack.

    Parameters
    ----------
    frames:
        Real, finite stack of ``F >= 2`` frames with shape ``(F, H, W)`` mono or
        ``(F, H, W, C)`` colour, as an array or a list of equal-shaped arrays.
        The values are linear-light signal in a consistent unit. The scene and
        illumination must be static across the stack: motion, flicker, or a
        drifting exposure enters the temporal variance as if it were noise.
    roi:
        A :class:`~camsim.metrics.chart.PatchROI` lying fully inside every
        frame; each frame is cropped through
        :func:`~camsim.metrics.chart.extract_patch`.
    channel:
        Resolved exactly as in :func:`snr_patch`: mono aliases for a 2-D frame
        stack, and luma or one RGB plane for a colour stack.

    Returns
    -------
    SNRResult
        ``mean`` is the mean of the per-pixel temporal means, ``std`` is
        ``sqrt(mean(per-pixel temporal variances))`` with ``ddof=1`` per pixel,
        and ``snr_db = 20 log10(mean / std)``. Static fixed-pattern variation is
        excluded from ``std`` by construction, but it still enters ``mean``.
        A static stack (std zero), a non-positive mean, or fewer than two frames
        yields ``valid=False``; the frame-count error raises instead of
        returning an invalid result.

    Raises
    ------
    TypeError
        If ``roi`` is not a :class:`~camsim.metrics.chart.PatchROI`.
    ValueError
        If the stack is not a finite real ``(F, H, W)`` or ``(F, H, W, C)``
        array-like with ``F >= 2``, the ROI extends past a frame boundary, or
        the channel is invalid for the stack.

    Notes
    -----
    Pooling per-pixel variances is what makes this a random-noise estimate.
    Concatenating every pixel of every frame and taking one standard deviation
    would report fixed-pattern variation as noise; so would taking the standard
    deviation of the per-pixel temporal means. The estimator is deterministic
    and does not modify the input stack.
    """
    stack = _as_frame_stack(frames)
    first = np.asarray(extract_patch(stack[0], roi), dtype=np.float64)
    label, index = _channel_resolution(first, channel)
    crops = [extract_patch(stack[frame_index], roi) for frame_index in range(stack.shape[0])]
    plane = _channel_plane(np.asarray(np.stack(crops, axis=0), dtype=np.float64), label, index)
    return _temporal_result(plane, label)


def snr_channels(image: ArrayLike, roi: PatchROI) -> dict[str, SNRResult]:
    """Measure every colour plane and the linear luma of one flat patch.

    Parameters
    ----------
    image:
        Real, finite ``(H, W)`` mono or ``(H, W, 3)`` linear-light image.
    roi:
        The patch region, lying fully inside the image.

    Returns
    -------
    dict[str, SNRResult]
        For a 2-D image, ``{"mono": result}``. For a 3-D image the keys are
        ``"R"``, ``"G"``, ``"B"``, and ``"luma"``, in that order, so a report
        iterates deterministically. Luma uses Rec. 709 weights in linear light.
        A 3-D image that is not exactly three channels wide is rejected: the
        four documented measurements are not all defined for it, and silently
        dropping one would make reports incomparable.

    Raises
    ------
    TypeError
        If ``roi`` is not a :class:`~camsim.metrics.chart.PatchROI`.
    ValueError
        If the image is invalid, the ROI lies outside it, or it is a 3-D image
        without exactly three channels.

    Notes
    -----
    This is the per-channel helper a `measure_all`-style harness calls for a
    flat-patch chart. Each plane is measured with the same statistics as
    :func:`snr_patch`, including the validity rules; a saturated or black plane
    is reported as invalid rather than omitted.
    """
    patch = np.asarray(extract_patch(image, roi), dtype=np.float64)
    if patch.ndim == 2:
        return {_MONO_CHANNEL: _spatial_result(patch, _MONO_CHANNEL)}

    channels = int(patch.shape[2])
    if channels != 3:
        raise ValueError(
            "snr_channels requires a 2-D mono image or an (H, W, 3) colour image; "
            f"got {channels} channels"
        )
    return {
        "R": _spatial_result(patch[..., 0], "R"),
        "G": _spatial_result(patch[..., 1], "G"),
        "B": _spatial_result(patch[..., 2], "B"),
        _LUMA_CHANNEL: _spatial_result(_luma_plane(patch), _LUMA_CHANNEL),
    }
