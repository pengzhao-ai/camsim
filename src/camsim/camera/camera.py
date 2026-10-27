"""Camera orchestration from a rendered scene to a display image.

The camera is the only object that owns the whole chain, so it is also the
right place to decide how much light reaches the sensor. It composes four
layers that stay independent of each other: the lens geometry, the sensor
(spectral irradiance to a RAW frame), the ISP (RAW to a rendered image), and an
exposure policy. The default policy is a fixed exposure taken from the sensor
preset, so every experiment that does not opt in is deterministic and directly
comparable. Auto-exposure is an explicit request.

Exposure, gain, and metering
----------------------------
The sensor integrates irradiance for ``exposure_s`` seconds, so the charge in
each photodiode is proportional to exposure. The ADC then records

``DN = black_level_dn + round(V / full_scale_uv * ((2**b - 1) - black_level_dn))``

where ``V`` is the sense-node voltage after conversion gain and analog gain.
The meter used here is the centre-weighted mean of the black-corrected fraction

``meter = mean_w[(DN - black_level_dn) / ((2**b - 1) - black_level_dn)]``

over the native mosaic. Dividing by the usable span makes the value comparable
across bit depths and presets; subtracting the pedestal is mandatory, because a
metered pedestal would ask auto-exposure to expose for the black offset rather
than for the light. The weight is a normalized separable Gaussian centred on
the frame, so a bright sky or a dark border cannot dominate a portrait-style
scene. A photographic 18 % grey is the conventional target because it leaves
roughly two and a half stops of highlight headroom above it.

The sensitivity knobs multiply in this order:

1. **Exposure** multiplies the photon count, so it improves shot-noise-limited
   SNR. Auto-exposure spends it first.
2. **Analog gain** is applied before the ADC and raises signal and
   input-referred read noise together. It preserves input-referred SNR while
   reducing quantization loss, so it is the second knob. The preset documents a
   ``step_db`` grid; the policy always snaps requests down towards lower gain.
   A downward snap cannot overshoot an increase; for a reduction it
   over-attenuates slightly, and digital gain recovers the difference. This is
   the gain-quantization artefact: a coarse step leaves a residual that must go
   somewhere, and digital gain absorbs it without stalling convergence.
3. **Digital gain** multiplies already-quantized codes. It cannot recover
   information and amplifies quantization noise, so it is the last knob.
   Digital gain below one is a last resort for overexposure and is used only
   when exposure and analog gain are both at their minima.

Dual conversion gain
--------------------
HCG trades well capacity for lower input-referred read noise: the reference
preset has 300 uV/e, 2000 e- full well, and 0.7 e- rms read noise, versus
100 uV/e, 6000 e-, and 2.0 e- rms in LCG. ``Sensor.expose(mode="auto")`` picks
HCG below ``auto_hcg_threshold_e`` mean charge. An 18 % target corresponds to
roughly 480 e- in HCG and 1440 e- in LCG, both above the 200 e- threshold, so
re-resolving the mode at the target would flip an HCG-started ramp to LCG and
change sensitivity by the ratio of conversion gains. Auto-exposure therefore
resolves the mode on the first probe and locks it for the remaining probes. The
lock is what makes the returned ``mode`` reproducible; mode flapping is the
artefact this avoids.

Real-world parameter ranges
---------------------------
Phone main cameras expose from about 10 us to 100 ms; the reference preset uses
100 us to 100 ms with a 10 ms default. Analog gain spans roughly 0-24 dB in
1 dB steps, digital gain 1-8x, and the metering target is 18 % of usable full
scale. The flagship sensor quantizes to 10 bits with a 64 DN black level, so one
signal code is 1/959 (about 0.1 %) of the usable span and the meter has about
0.1 % resolution.

Failure artefacts guarded here
------------------------------
Metering without black-level correction (exposing for the pedestal), metering
only the green channel or the whole frame with equal weights (a dark border
silently raises the exposure), and clipping detection that looks only at the
ADC's largest code while the sensor full well saturates first (both limits are
real; the reference preset is full-well limited at unity gain, where the well
code is 783 of the 1023 possible codes). A clipped probe must not ask for more
light: when more than 1 % of pixels reach the earlier of the full-well and ADC
saturation codes, the correction is capped at a factor of 0.5, so the next
probe reduces instead of chasing a saturated statistic. Non-convergence is
reported rather than hidden, and a zero-valued probe is treated as maximally
underexposed instead of dividing by zero. Gain quantization can leave a
residual; digital gain carries it, but a digital gain floor stops the policy
from requesting an unreachable zero. No global random state is used:
auto-exposure probe ``i`` is seeded with ``i``, and capture without a seed draws
from the sensor's frame stream exactly like a direct
:meth:`~camsim.sensor.Sensor.expose` call, so reseeding the sensor replays a
capture bit-for-bit within one environment.

Public API
----------
:class:`Camera`, :class:`ExposureSettings`, and :class:`ExposureResult`. The
camera's default :meth:`Camera.capture` is the fixed-exposure baseline;
:meth:`Camera.auto_expose` is the opt-in policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from camsim.config.schema import LensConfig, SensorConfig
from camsim.core.frames import RawFrame, RenderedImage
from camsim.isp.pipeline import ISP
from camsim.scene.targets import SceneResult
from camsim.sensor import Sensor

__all__ = ["Camera", "ExposureResult", "ExposureSettings"]

ExposureMode: TypeAlias = Literal["LCG", "HCG"]
"""A concrete sensor conversion-gain readout mode."""

ExposurePolicy: TypeAlias = Literal["fixed", "auto"]
"""How a captured image chose its exposure."""

# The photographic 18 % grey target. It leaves about 2.5 stops of headroom.
_DEFAULT_TARGET_FRACTION: Final[float] = 0.18
_DEFAULT_MAX_ITERATIONS: Final[int] = 12
_DEFAULT_TOLERANCE: Final[float] = 0.02

# A damped multiplicative correction converges without oscillating around the
# target when the sensor response is close to linear. The 0.7 exponent makes a
# 4x error take about five probes instead of one risky full-step jump.
_DAMPING_EXPONENT: Final[float] = 0.7

# A probe with more than 1 % of pixels at the largest code is clipped. The
# correction is capped below unity so a clipped probe can never ask for more
# light, even if the (compressed) mean still reads below the target.
_CLIP_FRACTION: Final[float] = 0.01
_CLIP_FACTOR_CAP: Final[float] = 0.5

# A probe that rounds every pixel to the pedestal reports exactly zero and
# cannot be inverted. Treat it as maximally underexposed and jump one decade of
# light per probe; exposure and analog-gain limits bound the real increase and
# digital gain carries the rest.
_ZERO_METER_FACTOR: Final[float] = 8.0

# Digital gain is a display convenience, not a sensitivity control. The floor
# keeps the requested multiplier positive and finite even when a pathological
# scene would otherwise ask for zero; hitting it is reported as non-convergence.
_DIGITAL_GAIN_FLOOR: Final[float] = 1e-3

# The metering weight is a Gaussian whose sigma is one sixth of the longer
# frame side: at 64 x 64 the corner weight is about 0.016 % of the centre and
# the edge midpoint about 1.3 %, so the scene centre dominates without ignoring
# the periphery.
_WEIGHT_SIGMA_FRACTION: Final[float] = 1.0 / 6.0

# Weight maps are cached per shape. Eight entries cover the common grid sizes
# an experiment sweeps without letting an unbounded shape generator retain
# arrays forever.
_WEIGHT_CACHE_SIZE: Final[int] = 8

# Residual factors smaller than this are floating-point round-off, not a
# meaningful correction; treating them as unity avoids a spurious gain step.
_FACTOR_EPSILON: Final[float] = 1e-9


@dataclass(frozen=True)
class ExposureSettings:
    """A fixed exposure request for one capture.

    Parameters
    ----------
    exposure_s:
        Integration time in seconds, finite and strictly positive. It is
        validated against the sensor preset's exposure window by
        :meth:`Camera.capture`, not here, because this value type does not own
        a sensor.
    analog_gain_db:
        Analog programmable-gain-amplifier setting in decibels, finite and
        non-negative; 0 dB is unity gain. As with exposure, the preset range is
        enforced at capture time.
    mode:
        Concrete readout mode, ``"LCG"`` or ``"HCG"``. ``"auto"`` is
        deliberately not accepted: auto-selection belongs to the sensor, and a
        fixed policy must state the mode it actually used so two captures with
        the same settings are comparable.

    Notes
    -----
    Digital gain is not part of a fixed request; :meth:`Camera.capture` uses
    unity. The value object is frozen so a settings record cannot drift away
    from the image it produced.
    """

    exposure_s: float
    analog_gain_db: float
    mode: ExposureMode

    def __post_init__(self) -> None:
        object.__setattr__(self, "exposure_s", _positive_float(self.exposure_s, "exposure_s"))
        object.__setattr__(
            self,
            "analog_gain_db",
            _non_negative_float(self.analog_gain_db, "analog_gain_db"),
        )
        object.__setattr__(self, "mode", _validated_mode(self.mode))


@dataclass(frozen=True)
class ExposureResult:
    """The settings and evidence produced by one auto-exposure run.

    Parameters
    ----------
    exposure_s:
        Integration time of the final probe, in seconds, finite and positive.
    analog_gain_db:
        Analog gain of the final probe in decibels, finite and non-negative.
    digital_gain:
        Digital multiplier of the final probe, finite and strictly positive.
    mode:
        Readout mode of the final probe, ``"LCG"`` or ``"HCG"``. The first
        probe resolves ``"auto"`` and the rest of the ramp is locked to it to
        prevent conversion-gain flapping.
    converged:
        ``True`` when the final probe's metered value was within the requested
        tolerance of the target.
    iterations:
        Number of probe frames exposed. It is at least one and at most the
        requested ``max_iterations``.
    metered_value:
        Centre-weighted black-corrected signal fraction measured on the final
        probe, in ``[0, 1]`` for an unclipped mean. It is the number the
        convergence test compared with the target.

    Notes
    -----
    The result is a frozen value: passing it to :meth:`Camera.capture` replays
    the measured settings, including digital gain, without re-running the
    search.
    """

    exposure_s: float
    analog_gain_db: float
    digital_gain: float
    mode: ExposureMode
    converged: bool
    iterations: int
    metered_value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "exposure_s", _positive_float(self.exposure_s, "exposure_s"))
        object.__setattr__(
            self,
            "analog_gain_db",
            _non_negative_float(self.analog_gain_db, "analog_gain_db"),
        )
        object.__setattr__(self, "digital_gain", _positive_float(self.digital_gain, "digital_gain"))
        object.__setattr__(self, "mode", _validated_mode(self.mode))
        if not isinstance(self.converged, bool):
            raise ValueError(f"converged must be a bool; got {self.converged!r}")
        object.__setattr__(self, "iterations", _non_negative_int(self.iterations, "iterations"))
        object.__setattr__(
            self,
            "metered_value",
            _non_negative_float(self.metered_value, "metered_value"),
        )


class Camera:
    """A lens, a sensor, and an ISP with fixed and auto exposure policies.

    Parameters
    ----------
    lens:
        The :class:`~camsim.config.schema.LensConfig` whose focal length,
        f-number, and geometry describe the optics. It is recorded in captured
        metadata; scene rendering uses the same geometry when the caller
        passes it to :func:`camsim.scene.targets.render`.
    sensor:
        The :class:`~camsim.sensor.Sensor` that turns scene irradiance into RAW
        frames. The camera never replaces or resamples the scene: the cube must
        already be on the sensor's pixel grid.
    isp:
        The :class:`~camsim.isp.pipeline.ISP` that renders RAW frames. Its
        configuration hash is preserved in every capture's metadata.

    Raises
    ------
    ValueError
        If an argument is not an instance of its declared type.

    Attributes
    ----------
    lens, sensor, isp:
        The three layers, stored publicly so an experiment can reconfigure the
        sensor seed or inspect the ISP preset. The camera holds references, not
        copies; mutating a configuration is visible to later captures, exactly
        as with the individual layers.

    Notes
    -----
    A camera owns no random state. ``Sensor.expose(seed=...)`` controls
    reproducibility for both policies; auto-exposure uses deterministic probe
    seeds and a default capture draws from the sensor's frame stream.
    """

    lens: LensConfig
    sensor: Sensor
    isp: ISP

    def __init__(self, lens: LensConfig, sensor: Sensor, isp: ISP) -> None:
        if not isinstance(lens, LensConfig):
            raise ValueError(f"lens must be a LensConfig; got {type(lens).__name__}")
        if not isinstance(sensor, Sensor):
            raise ValueError(f"sensor must be a Sensor; got {type(sensor).__name__}")
        if not isinstance(isp, ISP):
            raise ValueError(f"isp must be an ISP; got {type(isp).__name__}")
        self.lens = lens
        self.sensor = sensor
        self.isp = isp

    def auto_expose(
        self,
        scene: SceneResult,
        target_fraction: float = _DEFAULT_TARGET_FRACTION,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> ExposureResult:
        """Find settings that place the scene near ``target_fraction``.

        The policy starts from the sensor preset's default exposure and analog
        gain, resolves the readout mode on the first probe, and then adjusts
        exposure, analog gain, and digital gain in that order. Each correction
        is damped by raising ``target / metered`` to the power 0.7, so a large
        error is approached geometrically rather than overshooting in one step.
        A probe with more than 1 % of pixels at the earlier of the full-well
        and ADC saturation codes caps the correction at 0.5: a clipped
        statistic must never request more light.

        Parameters
        ----------
        scene:
            A :class:`~camsim.scene.targets.SceneResult`. Its ``irradiance`` is
            exposed directly; the detector never resamples.
        target_fraction:
            Desired metered fraction of the usable code span, strictly between
            zero and one. 0.18 is photographic 18 % grey.
        max_iterations:
            Maximum number of probe frames, an integer of at least one. The
            probe budget bounds run time and makes non-convergence finite.
        tolerance:
            Absolute metering tolerance as a fraction of full scale, finite and
            strictly positive. Convergence requires
            ``abs(metered - target) <= tolerance``.

        Returns
        -------
        ExposureResult
            The settings of the final probe, whether it converged, the number
            of probes, and the final metered value.

        Raises
        ------
        ValueError
            If the scene is not a ``SceneResult``, a parameter is out of range,
            or the frame has no usable white span (black level leaves none).

        Notes
        -----
        Probe ``i`` is exposed with ``seed=i``. A per-call seed does not advance
        the sensor's frame stream, so calling ``auto_expose`` twice on the same
        camera and scene is bit-identical and does not perturb a later
        ``capture``. The returned settings are exactly the ones that produced
        ``metered_value``; the policy does not report an unprobed proposal.
        """
        _require_scene(scene)
        target = _unit_fraction(target_fraction, "target_fraction")
        budget = _positive_int(max_iterations, "max_iterations")
        goal = _positive_float(tolerance, "tolerance")

        config = self.sensor.config
        exposure_s = float(config.exposure.default_s)
        analog_gain_db = float(config.analog_gain.default_db)
        digital_gain = 1.0
        resolved_mode: ExposureMode | None = None
        metered_value = 0.0
        converged = False
        probes = 0

        for probe_index in range(budget):
            probes = probe_index + 1
            raw = self.sensor.expose(
                scene.irradiance,
                exposure_s,
                mode="auto" if resolved_mode is None else resolved_mode,
                analog_gain_db=analog_gain_db,
                digital_gain=digital_gain,
                seed=probe_index,
            )
            if resolved_mode is None:
                # Lock the first resolved mode; re-resolving every probe would
                # flip LCG/HCG around the auto threshold and change sensitivity
                # by the conversion-gain ratio mid-ramp.
                resolved_mode = _frame_mode(raw.mode)
            metered_value = _meter_frame(raw)
            if abs(metered_value - target) <= goal:
                converged = True
                break
            if probes == budget:
                # Report the settings of the final probe, not an unprobed next
                # proposal that would make metered_value inconsistent.
                break
            exposure_s, analog_gain_db, digital_gain = _allocate(
                exposure_s,
                analog_gain_db,
                digital_gain,
                _correction_factor(metered_value, target, _is_clipped(raw, config)),
                config,
            )

        if resolved_mode is None:  # pragma: no cover - budget >= 1 guarantees a probe
            raise RuntimeError("auto_expose completed without exposing a probe frame")
        return ExposureResult(
            exposure_s=exposure_s,
            analog_gain_db=analog_gain_db,
            digital_gain=digital_gain,
            mode=resolved_mode,
            converged=converged,
            iterations=probes,
            metered_value=metered_value,
        )

    def capture(
        self,
        scene: SceneResult,
        exposure: ExposureSettings | ExposureResult | None = None,
    ) -> RenderedImage:
        """Expose one scene and render it through the ISP.

        Parameters
        ----------
        scene:
            A :class:`~camsim.scene.targets.SceneResult` on the sensor's pixel
            grid.
        exposure:
            ``None`` uses the fixed baseline: the sensor preset's default
            exposure and analog gain, unity digital gain, and ``"LCG"``.
            Auto-exposure is opt-in, so the default never spends probes and never
            depends on scene brightness. An :class:`ExposureSettings` applies a
            fixed request with unity digital gain; an :class:`ExposureResult`
            replays an auto-exposure measurement, including its digital gain
            and mode.

        Returns
        -------
        RenderedImage
            A new image whose pixel buffers are the ISP's output and whose
            metadata is a copy of the ISP metadata plus the exposure policy,
            the applied exposure and gains, the readout mode, the lens focal
            length and f-number, and, for auto exposure, the result's metered
            value, iteration count, and convergence flag. ``digital_gain`` is
            the policy multiplier; the sensor composes it with the ADC preset's
            own digital gain, so the RAW frame can record a larger effective
            value.

        Raises
        ------
        ValueError
            If ``scene`` is not a ``SceneResult``, ``exposure`` is not ``None``
            or an accepted value type, or the requested exposure or analog gain
            lies outside the sensor preset's limits.

        Notes
        -----
        The exposure is drawn from the sensor's frame stream (``seed=None``),
        so consecutive captures are independent and the raw frame records its
        derived seed and frame index for later auditing. Reseed the sensor with
        :meth:`~camsim.sensor.Sensor.reseed` to replay a capture bit-for-bit in
        one environment. The input scene and the intermediate ISP result are
        never mutated; a fresh metadata dictionary is attached to the returned
        image.
        """
        _require_scene(scene)
        config = self.sensor.config
        settings, digital_gain, policy, result = _resolve_exposure(exposure, config)
        _validate_against_preset(settings, config)

        raw = self.sensor.expose(
            scene.irradiance,
            settings.exposure_s,
            mode=settings.mode,
            analog_gain_db=settings.analog_gain_db,
            digital_gain=digital_gain,
        )
        image = self.isp.process(raw)

        metadata = dict(image.metadata)
        metadata.update(
            {
                "exposure_policy": policy,
                "exposure_s": settings.exposure_s,
                "analog_gain_db": settings.analog_gain_db,
                "digital_gain": digital_gain,
                "camera_mode": _frame_mode(raw.mode),
                "focal_length_mm": float(self.lens.focal_length_mm),
                "f_number": float(self.lens.f_number),
            }
        )
        if result is not None:
            metadata.update(
                {
                    "metered_value": result.metered_value,
                    "iterations": result.iterations,
                    "converged": result.converged,
                }
            )
        return RenderedImage(
            data=image.data,
            color_space=image.color_space,
            channels=image.channels,
            linear_rgb=image.linear_rgb,
            stages=image.stages,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Metering and the exposure search
# ---------------------------------------------------------------------------


def _meter_frame(raw: RawFrame) -> float:
    """Return the centre-weighted black-corrected signal fraction.

    The pedestal is subtracted before the mean because it carries no scene
    information; dividing by the usable white span makes the statistic
    independent of bit depth. The 2-D weight map is applied as two 1-D
    contractions (``wy @ signal @ wx``), so metering never allocates a
    frame-sized weight array. The frame conversion itself is still one float64
    temporary; a tiled accumulator would bound that too at the cost of extra
    Python overhead, which is deferred until a 50 MP workload demands it.
    """
    span = float((1 << raw.bit_depth) - 1) - float(raw.black_level_dn)
    if span <= 0.0:
        raise ValueError(
            f"the {raw.bit_depth}-bit frame has no usable white span: black_level_dn="
            f"{raw.black_level_dn:g} reaches the largest code; auto-exposure cannot "
            "meter a signal range that does not exist"
        )
    signal = (np.asarray(raw.data, dtype=np.float64) - float(raw.black_level_dn)) / span
    weights_y, weights_x = _center_weights_1d(
        int(raw.data.shape[0]),
        int(raw.data.shape[1]),
    )
    return float(weights_y @ signal @ weights_x)


@lru_cache(maxsize=_WEIGHT_CACHE_SIZE)
def _center_weights_1d(
    height: int,
    width: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return normalized Gaussian row and column weights, cached by shape.

    The separable weight map is the outer product of the two arrays. The sigma
    is one sixth of the longer side, so at 64 x 64 the corner weight is about
    0.016 % of the centre and the edge midpoint about 1.3 %. Each axis is
    normalized independently, so their outer product sums to one and the
    statistic is a weighted mean independent of frame size.

    The arrays are cached and marked read-only; callers must treat them as
    shared values rather than scratch buffers.
    """
    sigma = max(height, width) * _WEIGHT_SIGMA_FRACTION
    rows = np.arange(height, dtype=np.float64) - (height - 1) / 2.0
    columns = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    weights_y = np.exp(-(rows**2) / (2.0 * sigma * sigma))
    weights_x = np.exp(-(columns**2) / (2.0 * sigma * sigma))
    weights_y /= float(weights_y.sum())
    weights_x /= float(weights_x.sum())
    weights_y.setflags(write=False)
    weights_x.setflags(write=False)
    return weights_y, weights_x


def _is_clipped(raw: RawFrame, config: SensorConfig) -> bool:
    """Return ``True`` when more than 1 % of pixels are saturated.

    Saturation can happen in either of two places. The charge-domain full well
    clips electrons before read noise, and the ADC clips the converted voltage
    at full scale. The earlier limit is the one that matters, so this function
    reconstructs both from the frame's recorded readout mode and gains (not
    from the settings the caller believes were used):

    ``well_code = black + full_well_e * conversion_gain_uv_per_e * gain_linear
    / adc.full_scale_uv * white_span``

    and compares pixels against ``min(2**bits - 1, floor(well_code))``. When
    the ADC clips first, the threshold is the largest code and only ADC
    saturation counts. When the well clips first, the threshold sits below the
    converter ceiling; for the reference preset at unity gain it is 783, not
    1023. Because the well clip happens before read noise, a marginal
    saturated pixel scatters around the well code by the read-noise interval;
    the 1 % fraction tolerates that statistical edge.
    """
    largest_code = (1 << raw.bit_depth) - 1
    mode = _frame_mode(raw.mode)
    mode_config = config.modes[mode]
    gain_linear = 10.0 ** (float(raw.analog_gain_db) / 20.0) * float(raw.digital_gain)
    full_well_uv = (
        float(mode_config.full_well_e) * float(mode_config.conversion_gain_uv_per_e) * gain_linear
    )
    white_span = float(largest_code) - float(raw.black_level_dn)
    well_code = (
        float(raw.black_level_dn) + (full_well_uv / float(config.adc.full_scale_uv)) * white_span
    )
    clip_code = min(largest_code, math.floor(well_code))
    clipped = int(np.count_nonzero(raw.data >= clip_code))
    return clipped / raw.data.size > _CLIP_FRACTION


def _correction_factor(metered_value: float, target: float, clipped: bool) -> float:
    """Return the damped multiplicative correction for the next probe.

    The sensor response is approximately linear in exposure times gain, so the
    ideal correction is ``target / metered``. The 0.7 exponent damps the step
    so a noisy or slightly nonlinear response cannot oscillate. A clipped probe
    can never request more light: the factor is capped at 0.5 because it was
    computed from a compressed statistic.
    """
    if metered_value <= 0.0:
        factor = _ZERO_METER_FACTOR
    else:
        factor = (target / metered_value) ** _DAMPING_EXPONENT
    if clipped:
        factor = min(factor, _CLIP_FACTOR_CAP)
    return factor


def _allocate(
    exposure_s: float,
    analog_gain_db: float,
    digital_gain: float,
    factor: float,
    config: SensorConfig,
) -> tuple[float, float, float]:
    """Distribute a correction across exposure, analog gain, and digital gain.

    Exposure absorbs as much of ``factor`` as its limits allow. Analog gain
    then absorbs the remainder, always snapped down (towards lower gain) to the
    preset's ``step_db`` grid: an increase cannot overshoot, while a reduction
    over-attenuates slightly and the residual is recovered by digital gain.
    Because the snap leaves the analog stage at or below the requested
    correction, the residual is at least one while exposure or gain still has
    room, so digital gain stays at or above its previous value and falls below
    unity only when exposure and analog gain are both at their minima.
    """
    minimum_exposure = float(config.exposure.min_s)
    maximum_exposure = float(config.exposure.max_s)
    new_exposure = min(max(exposure_s * factor, minimum_exposure), maximum_exposure)
    remaining = factor * exposure_s / new_exposure

    minimum_gain = float(config.analog_gain.min_db)
    maximum_gain = float(config.analog_gain.max_db)
    step = float(config.analog_gain.step_db)
    new_gain = analog_gain_db
    if remaining > 1.0 + _FACTOR_EPSILON:
        requested_db = min(analog_gain_db + 20.0 * math.log10(remaining), maximum_gain)
        snapped = _floor_to_step(requested_db, minimum_gain, step)
        new_gain = min(max(snapped, analog_gain_db), maximum_gain)
    elif remaining < 1.0 - _FACTOR_EPSILON:
        requested_db = max(analog_gain_db + 20.0 * math.log10(remaining), minimum_gain)
        snapped = _floor_to_step(requested_db, minimum_gain, step)
        new_gain = max(min(snapped, analog_gain_db), minimum_gain)

    remaining *= 10.0 ** ((analog_gain_db - new_gain) / 20.0)
    new_digital = max(digital_gain * remaining, _DIGITAL_GAIN_FLOOR)
    return new_exposure, new_gain, new_digital


def _floor_to_step(value: float, minimum: float, step: float) -> float:
    """Snap ``value`` down to the grid ``minimum + k * step``.

    The tiny positive offset absorbs floating-point round-off so a value that
    is mathematically on the grid (for example 5.0 with a 1 dB step) does not
    fall to the previous step.
    """
    steps = math.floor((value - minimum) / step + 1e-9)
    return minimum + steps * step


# ---------------------------------------------------------------------------
# Capture request resolution
# ---------------------------------------------------------------------------


def _resolve_exposure(
    exposure: ExposureSettings | ExposureResult | None,
    config: SensorConfig,
) -> tuple[ExposureSettings, float, ExposurePolicy, ExposureResult | None]:
    """Return settings, digital gain, policy label, and auto result if any."""
    if exposure is None:
        # The fixed baseline is deliberately LCG: opting into HCG is a policy
        # decision, and the default must not change with scene brightness.
        settings = ExposureSettings(
            exposure_s=float(config.exposure.default_s),
            analog_gain_db=float(config.analog_gain.default_db),
            mode="LCG",
        )
        return settings, 1.0, "fixed", None
    if isinstance(exposure, ExposureSettings):
        return exposure, 1.0, "fixed", None
    if isinstance(exposure, ExposureResult):
        settings = ExposureSettings(
            exposure_s=exposure.exposure_s,
            analog_gain_db=exposure.analog_gain_db,
            mode=exposure.mode,
        )
        return settings, exposure.digital_gain, "auto", exposure
    raise ValueError(
        f"exposure must be None, an ExposureSettings or an ExposureResult; "
        f"got {type(exposure).__name__}"
    )


def _validate_against_preset(settings: ExposureSettings, config: SensorConfig) -> None:
    """Reject exposure or gain outside the sensor preset's realisable range."""
    minimum_exposure = float(config.exposure.min_s)
    maximum_exposure = float(config.exposure.max_s)
    if not minimum_exposure <= settings.exposure_s <= maximum_exposure:
        raise ValueError(
            f"exposure_s ({settings.exposure_s:g} s) is outside the sensor preset's "
            f"exposure limits [{minimum_exposure:g}, {maximum_exposure:g}] s"
        )
    minimum_gain = float(config.analog_gain.min_db)
    maximum_gain = float(config.analog_gain.max_db)
    if not minimum_gain <= settings.analog_gain_db <= maximum_gain:
        raise ValueError(
            f"analog_gain_db ({settings.analog_gain_db:g} dB) is outside the sensor "
            f"preset's analog-gain limits [{minimum_gain:g}, {maximum_gain:g}] dB"
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_scene(scene: object) -> None:
    """Require a rendered scene rather than an irradiance cube or array."""
    if not isinstance(scene, SceneResult):
        raise ValueError(
            f"scene must be a SceneResult (the output of camsim.scene.targets.render); "
            f"got {type(scene).__name__}"
        )


def _frame_mode(mode: object) -> ExposureMode:
    """Return a concrete readout mode from a frame's ``mode`` field."""
    if mode in ("LCG", "HCG"):
        return mode
    raise ValueError(f"sensor readout mode must be 'LCG' or 'HCG'; got {mode!r}")


def _validated_mode(value: object) -> ExposureMode:
    """Return a concrete ``LCG``/``HCG`` mode, rejecting ``"auto"``."""
    if value not in ("LCG", "HCG"):
        raise ValueError(
            f"mode must be 'LCG' or 'HCG'; got {value!r}. Fixed exposure records the "
            "concrete readout mode; 'auto' is resolved by the sensor during "
            "auto-exposure and is not a fixed setting"
        )
    return value


def _finite_float(value: object, field: str) -> float:
    """Return ``value`` as a finite Python float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be a real number; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return result


def _positive_float(value: object, field: str) -> float:
    """Return ``value`` as a finite, strictly positive float."""
    result = _finite_float(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive; got {result!r}")
    return result


def _non_negative_float(value: object, field: str) -> float:
    """Return ``value`` as a finite, non-negative float."""
    result = _finite_float(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative; got {result!r}")
    return result


def _unit_fraction(value: object, field: str) -> float:
    """Return ``value`` as a finite fraction strictly inside ``(0, 1)``."""
    result = _finite_float(value, field)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{field} must lie strictly between 0 and 1; got {result!r}")
    return result


def _positive_int(value: object, field: str) -> int:
    """Return ``value`` as a Python integer of at least one."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer of at least 1; got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{field} must be at least 1; got {result}")
    return result


def _non_negative_int(value: object, field: str) -> int:
    """Return ``value`` as a non-negative Python integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be a non-negative integer; got {result}")
    return result
