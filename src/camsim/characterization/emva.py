"""EMVA 1288-style measurement of the simulated sensor.

EMVA Standard 1288 characterizes an image sensor from the mean and temporal
variance of flat-field and dark frames. This module implements the same
measurement chain against :class:`camsim.sensor.Sensor`, so a sensor configured
with known parameters can be measured back and the result compared with the
datasheet-style inputs.

Photon transfer curve
---------------------
For a pixel with mean signal ``mu_DN`` and temporal variance ``sigma_DN**2`` in
black-level-corrected digital numbers, Poisson shot noise plus input-referred
read noise gives

``sigma_DN**2 = K * mu_DN + sigma_read_DN**2``

where ``K`` is the overall system gain in DN per electron (EMVA 1288's ``K``).
The slope is therefore ``K`` itself, not ``1/K``: one electron produces ``K``
DN, so one electron of shot variance produces ``K**2`` DN**2, and
``K**2 * mu_e = K * mu_DN``. The intercept is the read-noise variance in DN**2.
Dividing by ``K`` converts the intercept to electrons. A fit over saturated
levels collapses the slope toward zero, so levels whose temporal variance
collapses at the full-well or ADC ceiling are excluded.

Two measurement details matter for a sub-LSB read noise (0.7 e- is only 0.25 DN
in the flagship preset):

* **Row noise.** The sensor adds one common-mode Gaussian offset per readout
  row. That offset is temporal noise, so a naive per-pixel variance includes it
  and overstates read noise. Each frame is row-demeaned per CFA plane before
  the variance is formed; the lost independent degrees of freedom are restored
  with the ``n / (n - 1)`` factor.
* **ADC quantization.** Rounding to integer codes is *not* a constant
  ``1/12`` DN**2 correction when the noise is sub-LSB: the quantization
  variance depends on the phase of the signal within a code. A 0.7 e- read
  noise is only 0.25 DN in HCG and 2 e- is only 0.24 DN in LCG, so the fixed
  correction biases both. The fit therefore evaluates the exact quantized
  forward model ``round(K * (Poisson(mu) + N(0, sigma_read)))`` with the charge
  clipped at zero, and solves for ``K`` and ``sigma_read`` from all points.
  The model is evaluated with a Poisson mixture at low signal and a Gaussian
  with the same mean and variance above 60 e-. Points below one DN are
  excluded because the ensemble of mostly-zero pixels is no longer described
  by a single Poisson mean; the sweep is extended downward until a point near
  that anchor exists.

Other measured quantities
-------------------------
* **Saturation capacity** is taken from the clipped high-level flat field: the
  median signal of pixels whose temporal variance has collapsed at the ceiling.
  The configured full well (or ADC full scale, whichever is lower) is recorded
  as a cross-check but is not used to compute the result.
* **Absolute sensitivity threshold** is the signal at which
  ``SNR(mu) = mu / sqrt(mu + sigma_read**2) = 1``, i.e.
  ``mu_min = (1 + sqrt(1 + 4 sigma_read**2)) / 2``. This is the EMVA 1288
  definition: shot noise and read noise are both included.
* **Dynamic range** is ``20 log10(saturation_capacity_e / read_noise_e)``.
* **DSNU** (dark-signal non-uniformity) is measured from dark frames as the
  spatial standard deviation of the per-pixel temporal mean, after removing the
  temporal variance of that mean. This sensor clamps the pre-Poisson mean
  charge at zero and its 10-bit ADC resolves only about 0.06 DN of dark offset
  in LCG, so the clamp and quantization compress the apparent dark pattern.
  The measured fixed-pattern spread is inverted through the exact forward model
  ``round(K * (Poisson(max(0, d)) + N(0, sigma_read)))`` with the read noise
  from the PTC. The reported ``dsnu_e`` is the **pre-clamp, input-referred**
  offset standard deviation (the configured DSNU parameter); the raw
  post-clamp fixed-pattern spread is recorded separately in provenance. Without
  the inversion, a configured 0.5 e- DSNU measures only about 0.25 e- (HCG) or
  0.15 e- (LCG).
* **PRNU** (photo-response non-uniformity) is measured on the unsaturated flat
  field nearest 70 % of saturation. The spatial variance of the averaged frame
  is corrected for temporal noise and DSNU, then divided by the squared mean and
  reported in percent. Reporting the fraction 0.005 instead of 0.5 % is a
  classic 100x error, which is why the report field is named ``prnu_percent``.
* **SNR curve** and **linearity residual** describe the range between the noise
  floor and saturation. Each SNR point stores both the row-demeaned
  pixel-temporal noise and the raw-frame total noise (which adds common-mode
  row noise), so a reader can use either convention. The residual is the
  maximum relative deviation from a straight-line fit over unsaturated levels;
  if the primary channel is partially clipped, a channel that still spans the
  linear range is used and recorded in provenance.

Levels, units, and levels that actually reach the sensor
--------------------------------------------------------
``irradiance_levels`` is the preferred input: flat spectral irradiance at the
sensor plane in W/m^2/nm. ``lux_levels`` is a convenience that normalises a D65
(or 3000 K / 4000 K) illuminant to the requested photopic illuminance and
applies the lens formula ``T(lambda) / (4 N**2)``; it assumes a
unity-reflectance Lambertian scene on axis, so it is not a substitute for a
rendered scene. When neither list is supplied, levels are auto-ranged from a
probe frame: target mean signals from about 0.2 % to 150 % of the nominal
saturation code, which gives the PTC both a deep low-signal anchor and a
clipping point.

The old plan snippet's 1-16 W/m^2/nm levels are far above this part's full-well
knee (a 10 ms exposure clips at roughly 5e-4 W/m^2/nm in HCG and 1.6e-3 in
LCG), so they cannot define a PTC. The module verifies the actual response and
auto-ranges when the caller does not supply levels; the tests use the
low-to-saturation sweep documented in ``tests/test_characterization.py``. If a
caller supplies only a high-level sweep, geometrically lower levels are
appended until the low-signal anchor exists, and the added levels are recorded
in provenance.

Assumptions and limitations
---------------------------
This is a **simulated, EMVA 1288-style characterization**. It is not a
certified measurement. It assumes spatially uniform illumination, a single
conversion-gain mode (a PTC has one ``K``), a fixed sensor temperature, no lens
shading, stationary DSNU/PRNU maps, Gaussian read noise, Poisson shot noise,
and an ideal linear ADC. ``mode="auto"`` is resolved once to a single mode,
because a photon transfer curve that mixes HCG and LCG frames has no single
slope; the resolved mode is recorded in the provenance. Dark current versus
temperature is **not** measured here; that sweep is scoped to Task 17. A small
residual row-noise/quantization coupling leaves a few percent uncertainty in
HCG read noise at the sub-LSB operating point; the exact-model fit is still
within the 5 % G2 criterion at the documented test seed, and provenance records
the model limitations.

Real-world ranges: system gain 0.01-10 DN/e-, read noise 0.3-10 e- rms,
full well 2-50 ke-, dark current 0.01-100 e-/s, DSNU 0.1-10 e-, PRNU 0.1-2 %,
dynamic range 40-90 dB.

Failure artefacts guarded here: subtracting the black level twice or not at all;
including saturated levels in the slope fit; mixing readout modes; treating PRNU
as a fraction when the report is in percent; measuring read noise without
removing row noise or ADC quantization; forgetting the zero-charge clipping knee
when interpreting dark frames; and silently accepting levels that never reach
saturation, which would understate the full well. Each condition is either
corrected with a documented model or reported in ``provenance`` instead of being
hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]
from scipy.stats import norm, poisson  # type: ignore[import-untyped]

from camsim.config.schema import GainMode, LensConfig, SensorConfig
from camsim.core.frames import RawFrame
from camsim.core.spectral import SpectralIrradiance, WavelengthGrid
from camsim.optics.lens import transmittance_at
from camsim.scene.illuminants import illuminant as illuminant_spd
from camsim.scene.illuminants import lux_to_spectral_irradiance
from camsim.sensor import Sensor
from camsim.sensor.cfa import cfa_masks

from .report import EMVAReport, SNRPoint

__all__ = ["characterize_sensor"]

Mode: TypeAlias = Literal["auto", "LCG", "HCG"]
"""Accepted readout-mode requests."""

# A 64x64 flat field gives 1024 red, 2048 green, and 1024 blue samples in a
# Bayer part: enough for a per-channel PTC without making every characterization
# allocate large cubes. The first exposure fixes the sensor grid, so a live
# sensor keeps its own shape instead.
_FLAT_WIDTH: Final[int] = 64
_FLAT_HEIGHT: Final[int] = 64

# The 1/12 DN^2 term is the asymptotic quantization variance for a signal whose
# noise spans many LSBs. It is used only to seed the exact-model fit; the fit
# itself evaluates the phase-dependent quantized Poisson/Gaussian variance.
_QUANTIZATION_VARIANCE_DN2: Final[float] = 1.0 / 12.0

# Below one DN the output is a nearly deterministic zero/one sequence; the
# single-mean Poisson model cannot represent the ensemble of sub-code pixels,
# and those points bias the read-noise intercept. The exact-model fit starts at
# one DN, and the sweep is extended downward until at least one level lands
# near that anchor.
_PTC_MIN_MEAN_DN: Final[float] = 1.0

# Above this signal the Poisson distribution is replaced by a Gaussian with the
# same mean and variance. The two agree to better than 0.1 % in the moments.
_POISSON_MODEL_MAX_E: Final[float] = 60.0

# If a caller supplies only a high-level sweep, geometrically lower levels are
# appended until the low-signal anchor is reached. The added levels are recorded
# in provenance.
_LOW_LEVEL_SCALE: Final[float] = 0.25
_MAX_LOW_SIGNAL_LEVELS: Final[int] = 8

# Pixels whose temporal variance collapses below this value are candidates for
# the clipped population. An unsaturated high-signal pixel has variance of
# order K * mean (tens of DN^2), so the threshold separates the two regimes.
_CLIPPED_VARIANCE_DN2: Final[float] = 1.0

# A pixel is called saturated only when its mean is in the upper half of the
# brightest channel at that level. This avoids counting the quantized dark
# population, whose variance is also below one DN^2.
_CLIPPED_MEAN_FRACTION: Final[float] = 0.5

# A level contributes to the PTC only while fewer than 1 % of its pixels clip.
# Above that, censoring biases both the mean and the variance.
_MAX_PTC_CLIPPED_FRACTION: Final[float] = 0.01

# A level is treated as a saturation measurement when at least half of the
# brightest channel is clipped; the median clipped code is then the ceiling.
_MIN_SATURATION_CLIPPED_FRACTION: Final[float] = 0.5

# A level must be at least this fraction of the brightest observed level before
# its clipping statistics are trusted. Low-signal levels have many zero-variance
# quantized pixels, and they are not saturation candidates.
_SATURATION_LEVEL_FRACTION: Final[float] = 0.5

# Default auto-ranging targets as fractions of the nominal saturation code.
_AUTO_TARGET_FRACTIONS: Final[tuple[float, ...]] = (
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.4,
    0.7,
    1.5,
)

# Probe frames use a seed namespace disjoint from base_seed + frame_index so
# auto-ranging and mode resolution cannot correlate with the measured frames.
_PROBE_SEED_OFFSET: Final[int] = 1_000_000

_MIN_FRAMES_PER_LEVEL: Final[int] = 2
_MIN_LEVELS: Final[int] = 3
_MIN_PTC_POINTS: Final[int] = 3
_MIN_CHANNEL_PIXELS: Final[int] = 16

# The reference lens mirrors the scene layer's reference phone optic (f/1.8,
# 6.4 mm, 11.46 mm diagonal, T = 0.9). It is only used for the lux convenience;
# radiometric callers pass sensor-plane irradiance directly.
_REFERENCE_LENS_F_NUMBER: Final[float] = 1.8
_REFERENCE_LENS_FOCAL_LENGTH_MM: Final[float] = 6.4
_REFERENCE_LENS_DIAGONAL_MM: Final[float] = 11.46
_REFERENCE_LENS_TRANSMITTANCE: Final[float] = 0.9

# DSNU clipping inversion grid. The forward model is monotonic; 20 e- covers
# the full real-world DSNU range (0.1-10 e-) and leaves headroom for stressed
# configurations. An observation above the model range raises rather than
# silently clamping.
_DSNU_MODEL_MAX_E: Final[float] = 20.0
_DSNU_MODEL_POINTS: Final[int] = 160
_DSNU_QUADRATURE_POINTS: Final[int] = 301


@dataclass(frozen=True)
class _LevelSpec:
    """One flat-field level before any frames are exposed."""

    x_value: float
    """Caller-facing level: W/m^2/nm for irradiance input, lux for lux input."""
    mean_w_per_m2_nm: float
    """Mean spectral irradiance over the sensor grid (W/m^2/nm)."""
    cube: SpectralIrradiance


@dataclass(frozen=True)
class _ChannelStats:
    """Per-CFA-channel statistics for one flat-field level."""

    mean_dn: float
    variance_dn2: float
    raw_variance_dn2: float
    mean_map_variance_dn2: float
    clipped_fraction: float
    clipped_mean_dn: float | None


@dataclass(frozen=True)
class _LevelStats:
    """Statistics for one flat-field level across frames and CFA channels."""

    spec: _LevelSpec
    channels: dict[str, _ChannelStats]
    max_mean_dn: float


def characterize_sensor(
    sensor: Sensor,
    irradiance_levels: list[float] | tuple[float, ...] | None = None,
    lux_levels: list[float] | tuple[float, ...] | None = None,
    frames_per_level: int = 32,
    exposure_s: float = 1e-3,
    mode: Mode = "auto",
    temperature_c: float = 25.0,
    illuminant: str = "D65",
    lens: LensConfig | None = None,
    base_seed: int = 0,
) -> EMVAReport:
    """Measure one readout mode of ``sensor`` with EMVA 1288-style methods.

    Parameters
    ----------
    sensor:
        A constructed :class:`camsim.sensor.Sensor`. Its first frame fixes the
        spatial grid; if it has already been exposed, that shape is reused,
        otherwise a 64x64 native grid is used.
    irradiance_levels:
        Flat spectral irradiance levels at the sensor plane in W/m^2/nm. This
        is the preferred, fully radiometric form. At least three positive
        levels are required so the PTC is over-determined.
    lux_levels:
        Photopic illuminance levels in lux. Mutually exclusive with
        ``irradiance_levels``. Each level is converted with
        :func:`camsim.scene.illuminants.lux_to_spectral_irradiance` and the lens
        formula ``T(lambda) / (4 N**2)`` for a unity-reflectance Lambertian
        scene. When ``lens`` is ``None`` the reference f/1.8 phone lens is used.
    frames_per_level:
        Frames per level, at least two. More frames reduce the uncertainty of
        every variance estimate; 16-32 is a practical range.
    exposure_s:
        Integration time in seconds for every frame. It must lie inside the
        sensor preset's configured exposure limits.
    mode:
        ``"HCG"`` or ``"LCG"`` to measure one conversion-gain mode, or
        ``"auto"`` to resolve the sensor's automatic choice once from the
        lowest level and then lock that mode. A single-mode lock is required:
        a PTC that mixes two conversion gains has no single system gain.
    temperature_c:
        Sensor temperature in degrees Celsius held constant for all frames.
    illuminant:
        ``"D65"``, ``"3000K"``, or ``"4000K"``. Only used when ``lux_levels``
        is supplied.
    lens:
        Optional :class:`camsim.config.schema.LensConfig` for the lux
        conversion. Ignored for irradiance input.
    base_seed:
        Frame ``i`` uses seed ``base_seed + i``, with one global index across
        levels so every frame is independent. Probe frames use a disjoint seed
        namespace.

    Returns
    -------
    EMVAReport
        Measured system gain, read noise, saturation capacity, absolute
        sensitivity threshold, dynamic range, DSNU, PRNU, SNR curve, linearity
        residual, and provenance.

    Raises
    ------
    ValueError
        If inputs are out of range, both level forms are supplied, fewer than
        three usable levels remain, no channel has enough valid pixels, or the
        PTC fit is degenerate (non-positive slope or intercept).
    TypeError
        If ``sensor`` is not a :class:`camsim.sensor.Sensor`.

    Notes
    -----
    The measurement is deterministic for fixed inputs and ``base_seed``. Fixed
    pattern maps come from the sensor's own defect seed, so DSNU and PRNU are
    stationary as EMVA assumes. The report explicitly records that this is a
    simulated characterization rather than a certified standard measurement.
    """
    if not isinstance(sensor, Sensor):
        raise TypeError(f"sensor must be a camsim.sensor.Sensor; got {type(sensor).__name__}")
    if irradiance_levels is not None and lux_levels is not None:
        raise ValueError("supply either irradiance_levels or lux_levels, not both")
    frames = _validated_frames(frames_per_level)
    exposure = _positive_float(exposure_s, "exposure_s")
    temperature = _finite_float(temperature_c, "temperature_c")
    seed = _non_negative_int(base_seed, "base_seed")
    requested_mode = _validated_mode(mode)
    resolved_lens = _validated_lens(lens)
    config = sensor.config
    native_shape = sensor.frame_shape or (_FLAT_HEIGHT, _FLAT_WIDTH)
    wavelengths_nm = _sensor_wavelengths(config)
    pitch_um = float(config.pixel.pitch_um)
    source = _FrameSource(
        sensor=sensor,
        native_shape=native_shape,
        exposure_s=exposure,
        temperature_c=temperature,
        base_seed=seed,
        mode=requested_mode,
    )

    levels_source: str
    resolved_mode: GainMode
    lens_info: dict[str, Any] | None = None
    lux_values: list[float] | None = None
    if irradiance_levels is not None:
        specs = [
            _flat_spec(level, native_shape, wavelengths_nm, pitch_um)
            for level in _validated_levels(irradiance_levels, "irradiance_levels")
        ]
        levels_source = "caller_supplied_irradiance"
        resolved_mode = (
            _resolve_auto_mode(source, specs[0]) if requested_mode == "auto" else requested_mode
        )
    elif lux_levels is not None:
        converted = _validated_levels(lux_levels, "lux_levels")
        specs, resolved_lens, lens_info = _lux_specs(
            converted,
            illuminant,
            resolved_lens,
            native_shape,
            wavelengths_nm,
            pitch_um,
        )
        levels_source = "caller_supplied_lux"
        lux_values = [float(value) for value in converted]
        resolved_mode = (
            _resolve_auto_mode(source, specs[0]) if requested_mode == "auto" else requested_mode
        )
    else:
        specs, resolved_mode = _auto_level_specs(
            source, requested_mode, native_shape, wavelengths_nm, pitch_um, config
        )
        levels_source = "auto_ranged"

    source.mode = resolved_mode

    nominal_gain = _nominal_system_gain(config, resolved_mode)
    nominal_sat_dn = _nominal_saturation_dn(config, resolved_mode)
    nominal_sat_e = _nominal_saturation_e(config, resolved_mode)

    level_stats = [source.measure(spec, frames) for spec in specs]
    added_low_levels = _extend_low_signal_sweep(source, level_stats, frames)
    primary = _primary_channel(level_stats)
    saturation = _find_saturation(level_stats, nominal_sat_dn)
    saturation_appended = False
    if saturation is None:
        # The caller's levels did not reach the ceiling. Add progressively
        # brighter levels so the saturation capacity is measured rather than
        # assumed; the fallback below is only for a sensor that still refuses
        # to clip.
        for _ in range(4):
            brightest = max(level_stats, key=lambda item: item.spec.mean_w_per_m2_nm)
            brightest_mean = max(stats.mean_dn for stats in brightest.channels.values())
            if brightest_mean <= 0.0:
                break
            factor = 1.2 * nominal_sat_dn / brightest_mean
            extra_spec = _scale_spec(brightest.spec, max(factor, 2.0))
            level_stats.append(source.measure(extra_spec, frames))
            saturation = _find_saturation(level_stats, nominal_sat_dn)
            if saturation is not None:
                saturation_appended = True
                break

    gain, read_noise_e, ptc_points = _fit_ptc(
        level_stats,
        nominal_sat_dn=nominal_sat_dn,
    )
    if saturation is None:
        saturation_dn = nominal_sat_dn
        saturation_source = "configured_saturation_fallback"
        saturation_clipped_fraction = 0.0
    else:
        _, saturation_dn, saturation_clipped_fraction = saturation
        saturation_source = "clipped_flat_field"
    saturation_capacity_e = saturation_dn / gain
    dynamic_range_db = 20.0 * math.log10(saturation_capacity_e / read_noise_e)
    absolute_sensitivity_threshold_e = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * read_noise_e**2))

    dsnu_e, dsnu_observed_e = _measure_dsnu(
        source=source,
        frames=frames,
        gain=gain,
        read_noise_e=read_noise_e,
        wavelengths_nm=wavelengths_nm,
        pitch_um=pitch_um,
    )
    prnu_percent, prnu_level, prnu_mean_dn = _measure_prnu(
        level_stats,
        primary=primary,
        saturation_dn=saturation_dn,
        frames=frames,
        dsnu_e=dsnu_e,
        gain=gain,
    )
    snr_curve = _measure_snr_curve(
        level_stats, primary=primary, gain=gain, nominal_sat_dn=nominal_sat_dn
    )
    linearity_residual, linearity_channel = _measure_linearity(
        level_stats, primary=primary, saturation_dn=saturation_dn
    )

    used_levels = [item.spec.mean_w_per_m2_nm for item in level_stats]
    used_x = [item.spec.x_value for item in level_stats]
    first_hash = source.config_hash or ""
    provenance: dict[str, Any] = {
        "characterization": (
            "simulated EMVA 1288-style characterization; not a certified EMVA 1288 measurement"
        ),
        "standard": "EMVA Standard 1288 (release 4.0) style",
        "sensor_config_name": config.name,
        "sensor_config_hash": first_hash,
        "cfa": config.cfa.value,
        "native_shape": [int(native_shape[0]), int(native_shape[1])],
        "frame_shape": [int(source.data_shape[0]), int(source.data_shape[1])],
        "exposure_s": exposure,
        "temperature_c": temperature,
        "analog_gain_db": float(config.analog_gain.default_db),
        "mode_requested": requested_mode,
        "mode_resolved": resolved_mode,
        "frames_per_level": frames,
        "frames_exposed": source.frame_count,
        "base_seed": seed,
        "seed_policy": (
            "every measured frame uses seed base_seed + global_frame_index; probe "
            "frames use base_seed + 1000000 + attempt and do not advance the sequence"
        ),
        "levels_source": levels_source,
        "level_x_unit": "lux" if lux_values is not None else "W/m^2/nm",
        "level_x_values": used_x,
        "irradiance_levels_w_per_m2_nm": used_levels,
        "lux_levels": lux_values,
        "illuminant": illuminant if lux_values is not None else None,
        "lens": lens_info,
        "low_levels_appended": bool(added_low_levels),
        "added_low_levels_w_per_m2_nm": added_low_levels,
        "saturation_level_appended": saturation_appended,
        "saturation_source": saturation_source,
        "saturation_clipped_fraction": saturation_clipped_fraction,
        "primary_channel": primary,
        "linearity_channel": linearity_channel,
        "ptc_points": ptc_points,
        "ptc_min_mean_dn": _PTC_MIN_MEAN_DN,
        "ptc_fit": (
            "nonlinear least squares against the exact quantized PTC model: "
            "Poisson(mu) + Gaussian read noise, zero-charge clip, integer ADC "
            "rounding; parameters K and read noise"
        ),
        "system_gain_method": (
            "joint photon-transfer fit across all CFA channels in DN per electron"
        ),
        "nominal_system_gain_dn_per_e": nominal_gain,
        "system_gain_relative_error": (gain - nominal_gain) / nominal_gain,
        "read_noise_definition": (
            "input-referred Gaussian read-noise sigma in the exact quantized PTC "
            "model, after per-row common-mode removal and n/(n-1) correction; "
            "zero-charge clipping is part of the forward model"
        ),
        "row_noise_handling": (
            "each frame row-demeaned per CFA plane; n/(n-1) restores independent "
            "shot and read-noise variance"
        ),
        "configured_full_well_e": float(config.modes[resolved_mode].full_well_e),
        "configured_read_noise_e": float(config.modes[resolved_mode].read_noise_e_rms),
        "configured_saturation_e": nominal_sat_e,
        "saturation_cross_check_ratio": saturation_capacity_e / nominal_sat_e,
        "dsnu_definition": (
            "pre-zero-clamp, input-referred dark-signal non-uniformity in electrons "
            "rms (the configured DSNU parameter); the raw post-clamp fixed-pattern "
            "spread is recorded separately"
        ),
        "dsnu_method": (
            "dark-frame fixed-pattern standard deviation after removing temporal "
            "variance of the mean, inverted through the exact pre-clamp forward "
            "model Poisson(max(0, d)) + Gaussian read noise + zero-charge clip + "
            "integer ADC rounding"
        ),
        "dsnu_observed_fixed_pattern_e": dsnu_observed_e,
        "dsnu_observed_fixed_pattern_definition": (
            "raw post-clamp, post-quantization fixed-pattern spread of the "
            "averaged dark frame in electrons (not deconvolved); this is what a "
            "naive dark-frame standard deviation reports"
        ),
        "configured_dsnu_e": float(config.noise.dsnu_e),
        "prnu_method": (
            "spatial standard deviation of the averaged flat field at the level "
            "nearest 70 % of saturation, corrected for temporal variance and DSNU, "
            "normalised by the mean and reported in percent"
        ),
        "prnu_measurement_level_w_per_m2_nm": prnu_level,
        "prnu_measurement_mean_dn": prnu_mean_dn,
        "configured_prnu_percent": 100.0 * float(config.noise.prnu_fraction),
        "absolute_sensitivity_threshold_definition": (
            "mu_min = (1 + sqrt(1 + 4*read_noise_e^2))/2, the signal where "
            "SNR(mu) = mu/sqrt(mu + read_noise_e^2) = 1"
        ),
        "linearity_residual_definition": (
            "maximum |mean_DN - linear fit| / fit over unsaturated levels between "
            "5 % and 95 % of the saturation code; dimensionless fraction"
        ),
        "snr_curve_definition": (
            "mean signal and measured temporal noise per unsaturated level. The "
            "primary noise fields are row-demeaned pixel-temporal noise (ADC "
            "quantization included, common-mode row noise removed); the *_total "
            "fields are raw-frame noise including row noise. snr_db = "
            "20 log10(mean_signal_e / noise_e)"
        ),
        "dark_current_characterization": (
            "not measured here; temperature-dependent dark current is scoped to Task 17"
        ),
        "emva_assumptions": [
            "spatially uniform flat-field illumination",
            "one conversion-gain mode per report",
            "constant sensor temperature",
            "no lens shading or camera shading",
            "stationary DSNU and PRNU maps",
            "Gaussian read noise and Poisson shot noise",
            "ideal linear ADC",
        ],
        "limitations": [
            "simulated sensor, not a physical camera or a certified measurement",
            "no dark-current-versus-temperature sweep (Task 17)",
            "no lens or illumination non-uniformity in the flat field",
            "residual row-noise/quantization coupling leaves a few percent "
            "uncertainty in HCG read noise at the sub-LSB operating point",
        ],
    }

    return EMVAReport(
        system_gain_dn_per_e=gain,
        read_noise_e=read_noise_e,
        saturation_capacity_e=saturation_capacity_e,
        absolute_sensitivity_threshold_e=absolute_sensitivity_threshold_e,
        dynamic_range_db=dynamic_range_db,
        dsnu_e=dsnu_e,
        prnu_percent=prnu_percent,
        snr_curve=snr_curve,
        linearity_residual=linearity_residual,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Frame source and level construction
# ---------------------------------------------------------------------------


class _FrameSource:
    """Expose a sensor on one fixed grid with deterministic, independent seeds."""

    def __init__(
        self,
        sensor: Sensor,
        native_shape: tuple[int, int],
        exposure_s: float,
        temperature_c: float,
        base_seed: int,
        mode: Mode,
    ) -> None:
        self.sensor = sensor
        self.native_shape = native_shape
        self.exposure_s = exposure_s
        self.temperature_c = temperature_c
        self.base_seed = base_seed
        self.mode = mode
        self.frame_index = 0
        self.frame_count = 0
        self.data_shape: tuple[int, int] = native_shape
        self.config_hash: str | None = None
        # A probe locks the sensor grid before any measured frame; remember the
        # output shape so masks and statistics use it even with binning.
        self._shape_locked = False

    def expose_spec(self, spec: _LevelSpec) -> RawFrame:
        """Expose the next measured frame at ``spec`` with ``base_seed + index``."""
        seed = self.base_seed + self.frame_index
        self.frame_index += 1
        frame = self.sensor.expose(
            self._on_native_grid(spec.cube),
            self.exposure_s,
            mode=self.mode,
            temperature_c=self.temperature_c,
            seed=seed,
        )
        self.frame_count += 1
        self._record(frame)
        return frame

    def probe(self, spec: _LevelSpec, mode: Mode, attempt: int) -> RawFrame:
        """Expose a probe frame without advancing the measured-frame sequence."""
        frame = self.sensor.expose(
            self._on_native_grid(spec.cube),
            self.exposure_s,
            mode=mode,
            temperature_c=self.temperature_c,
            seed=self.base_seed + _PROBE_SEED_OFFSET + attempt,
        )
        self._record(frame)
        return frame

    def measure(self, spec: _LevelSpec, frames: int) -> _LevelStats:
        """Expose ``frames`` frames at ``spec`` and summarise every CFA channel."""
        frames_data: list[NDArray[np.float64]] = []
        valid: NDArray[np.bool_] | None = None
        masks: dict[str, NDArray[np.bool_]] | None = None
        for _ in range(frames):
            frame = self.expose_spec(spec)
            if valid is None:
                valid = _valid_mask(frame)
                masks = cfa_masks(frame.cfa, frame.data.shape)
            frames_data.append(_signal_dn(frame))
        assert valid is not None and masks is not None
        data = np.stack(frames_data, axis=0)
        channels = _channel_statistics(data, valid, masks)
        if not channels:
            raise ValueError(
                "no CFA channel has enough valid (non-defective, non-PDAF) pixels for "
                "characterization; reduce the defect/PDAF rates or enlarge the frame"
            )
        return _LevelStats(
            spec=spec,
            channels=channels,
            max_mean_dn=max(stats.mean_dn for stats in channels.values()),
        )

    def _on_native_grid(self, cube: SpectralIrradiance) -> SpectralIrradiance:
        """Broadcast a (1, 1, L) convenience cube to the locked sensor grid."""
        if cube.data.shape[:2] == self.native_shape:
            return cube
        data = np.broadcast_to(cube.data, (*self.native_shape, cube.data.shape[2])).copy()
        return SpectralIrradiance(
            data,
            cube.wavelengths_nm,
            cube.unit,
            pixel_pitch_um=cube.pixel_pitch_um,
        )

    def _record(self, frame: RawFrame) -> None:
        """Remember the output shape and configuration hash from any frame."""
        if not self._shape_locked:
            self.data_shape = (int(frame.data.shape[0]), int(frame.data.shape[1]))
            self._shape_locked = True
        self.config_hash = frame.sensor_config_hash


def _flat_spec(
    level: float,
    native_shape: tuple[int, int],
    wavelengths_nm: NDArray[np.float64],
    pitch_um: float,
) -> _LevelSpec:
    """Return a spatially flat spectral cube at ``level`` W/m^2/nm."""
    data = np.full((*native_shape, wavelengths_nm.size), level, dtype=np.float64)
    cube = SpectralIrradiance.from_array(
        data,
        wavelengths_nm,
        "W/m^2/nm",
        pixel_pitch_um=pitch_um,
    )
    return _LevelSpec(x_value=float(level), mean_w_per_m2_nm=float(level), cube=cube)


def _scale_spec(spec: _LevelSpec, factor: float) -> _LevelSpec:
    """Return ``spec`` scaled by a positive factor, preserving its spectral shape."""
    data = np.asarray(spec.cube.data, dtype=np.float64) * factor
    cube = SpectralIrradiance.from_array(
        data,
        spec.cube.wavelengths_nm,
        spec.cube.unit,
        pixel_pitch_um=spec.cube.pixel_pitch_um,
    )
    return _LevelSpec(
        x_value=spec.x_value * factor,
        mean_w_per_m2_nm=spec.mean_w_per_m2_nm * factor,
        cube=cube,
    )


def _lux_specs(
    lux_levels: list[float],
    illuminant_name: str,
    lens: LensConfig,
    native_shape: tuple[int, int],
    wavelengths_nm: NDArray[np.float64],
    pitch_um: float,
) -> tuple[list[_LevelSpec], LensConfig, dict[str, Any]]:
    """Convert lux levels to sensor-plane flat cubes through the lens formula."""
    spd = illuminant_spd(illuminant_name)
    unit_scene = lux_to_spectral_irradiance(1.0, spd)
    transmittance = _transmittance_curve(lens, spd.wavelengths_nm.nm)
    aperture_factor = transmittance / (4.0 * float(lens.f_number) ** 2)
    spectral_shape = np.asarray(unit_scene.data[0, 0], dtype=np.float64) * aperture_factor
    target_grid = WavelengthGrid(np.asarray(wavelengths_nm, dtype=np.float64))
    specs: list[_LevelSpec] = []
    for lux in lux_levels:
        data = np.broadcast_to(lux * spectral_shape, (*native_shape, spectral_shape.size)).copy()
        cube = SpectralIrradiance.from_array(
            data,
            spd.wavelengths_nm,
            "W/m^2/nm",
            pixel_pitch_um=pitch_um,
        )
        if not np.allclose(
            cube.wavelengths_nm.nm,
            wavelengths_nm,
            rtol=0.0,
            atol=1e-6,
        ):
            cube = cube.resample_wavelengths(target_grid)
        specs.append(
            _LevelSpec(
                x_value=float(lux),
                mean_w_per_m2_nm=float(np.mean(cube.data)),
                cube=cube,
            )
        )
    info: dict[str, Any] = {
        "f_number": float(lens.f_number),
        "focal_length_mm": float(lens.focal_length_mm),
        "sensor_diagonal_mm": float(lens.sensor_diagonal_mm),
        "transmittance": (
            float(lens.transmittance)
            if not isinstance(lens.transmittance, str)
            else lens.transmittance
        ),
        "formula": "sensor_irradiance = scene_irradiance * T(lambda) / (4 N^2)",
    }
    return specs, lens, info


def _transmittance_curve(
    lens: LensConfig, wavelengths_nm: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the lens transmittance sampled on a wavelength grid."""
    if isinstance(lens.transmittance, str):
        return np.asarray(
            [transmittance_at(lens, float(wavelength)) for wavelength in wavelengths_nm],
            dtype=np.float64,
        )
    return np.full(wavelengths_nm.shape, float(lens.transmittance), dtype=np.float64)


def _resolve_auto_mode(source: _FrameSource, spec: _LevelSpec) -> GainMode:
    """Resolve ``mode="auto"`` once from the lowest measured level."""
    frame = source.probe(spec, "auto", 0)
    resolved = str(frame.mode)
    if resolved not in ("HCG", "LCG"):
        raise ValueError(f"sensor resolved mode='auto' to {resolved!r}; expected HCG or LCG")
    return cast(GainMode, resolved)


def _auto_level_specs(
    source: _FrameSource,
    requested_mode: Mode,
    native_shape: tuple[int, int],
    wavelengths_nm: NDArray[np.float64],
    pitch_um: float,
    config: SensorConfig,
) -> tuple[list[_LevelSpec], GainMode]:
    """Auto-range levels from a probe and resolve the readout mode."""
    reference = 1e-5
    resolved: GainMode = "HCG"
    mean_dn = 0.0
    for attempt in range(8):
        spec = _flat_spec(reference, native_shape, wavelengths_nm, pitch_um)
        frame = source.probe(spec, requested_mode, attempt)
        if requested_mode == "auto":
            mode_text = str(frame.mode)
            if mode_text not in ("HCG", "LCG"):
                raise ValueError(
                    f"sensor resolved mode='auto' to {mode_text!r}; expected HCG or LCG"
                )
            resolved = cast(GainMode, mode_text)
        else:
            resolved = requested_mode
        mean_dn = _frame_mean_dn(frame)
        nominal_sat_dn = _nominal_saturation_dn(config, resolved)
        if mean_dn > 0.6 * nominal_sat_dn and reference > 1e-12:
            reference /= 8.0
            continue
        if mean_dn < 0.5 and reference < 1.0:
            reference *= 10.0
            continue
        break
    if not math.isfinite(mean_dn) or mean_dn <= 0.0:
        raise ValueError(
            "cannot auto-range characterization levels: the probe frame has no signal; "
            "supply irradiance_levels explicitly"
        )
    nominal_sat_dn = _nominal_saturation_dn(config, resolved)
    levels = sorted(
        {
            max(fraction * nominal_sat_dn / mean_dn * reference, 1e-12)
            for fraction in _AUTO_TARGET_FRACTIONS
        }
    )
    if len(levels) < _MIN_LEVELS:
        raise ValueError("auto-ranging produced too few distinct levels")
    specs = [_flat_spec(level, native_shape, wavelengths_nm, pitch_um) for level in levels]
    return specs, resolved


# ---------------------------------------------------------------------------
# Per-level statistics
# ---------------------------------------------------------------------------


def _signal_dn(frame: RawFrame) -> NDArray[np.float64]:
    """Return black-level-corrected signal in DN as float64."""
    return np.asarray(frame.data, dtype=np.float64) - float(frame.black_level_dn)


def _valid_mask(frame: RawFrame) -> NDArray[np.bool_]:
    """Return pixels that are neither defective nor PDAF sites."""
    valid = np.ones(frame.data.shape, dtype=np.bool_)
    if frame.defect_mask is not None:
        valid &= ~frame.defect_mask
    if frame.pdaf_mask is not None:
        valid &= ~frame.pdaf_mask
    return valid


def _channel_statistics(
    data: NDArray[np.float64],
    valid: NDArray[np.bool_],
    masks: dict[str, NDArray[np.bool_]],
) -> dict[str, _ChannelStats]:
    """Summarise every CFA channel from an ``(F, H, W)`` frame stack."""
    frames = int(data.shape[0])
    if frames < _MIN_FRAMES_PER_LEVEL:
        raise ValueError("frames_per_level must be at least 2 for a variance estimate")
    mean_map = data.mean(axis=0)
    raw_var_map = data.var(axis=0, ddof=1)
    stats: dict[str, _ChannelStats] = {}
    for channel, mask in masks.items():
        selected = mask & valid
        count = int(selected.sum())
        if count < _MIN_CHANNEL_PIXELS:
            continue
        variance_dn2 = _row_demeaned_variance(data, selected)
        level_mean = float(mean_map[selected].mean())
        low_variance = raw_var_map[selected] < _CLIPPED_VARIANCE_DN2
        clipped = low_variance & (mean_map[selected] > _CLIPPED_MEAN_FRACTION * level_mean)
        clipped_fraction = float(np.mean(clipped))
        clipped_mean = (
            float(np.median(mean_map[selected][clipped])) if bool(clipped.any()) else None
        )
        stats[channel] = _ChannelStats(
            mean_dn=level_mean,
            variance_dn2=variance_dn2,
            raw_variance_dn2=float(raw_var_map[selected].mean()),
            mean_map_variance_dn2=float(np.var(mean_map[selected], ddof=1)),
            clipped_fraction=clipped_fraction,
            clipped_mean_dn=clipped_mean,
        )
    return stats


def _row_demeaned_variance(data: NDArray[np.float64], selected: NDArray[np.bool_]) -> float:
    """Return mean temporal variance after removing each frame's row means.

    Row noise is common to every pixel in a readout row, so subtracting the row
    mean removes it. The same subtraction also removes ``1/n`` of the
    independent shot and read noise; multiplying each pixel's variance by
    ``n/(n-1)`` restores those degrees of freedom.
    """
    residual = data.copy()
    height = int(data.shape[1])
    counts = selected.sum(axis=1)
    for row in range(height):
        columns = selected[row]
        count = int(counts[row])
        if count < 2:
            continue
        values = residual[:, row, columns]
        residual[:, row, columns] = values - values.mean(axis=1, keepdims=True)
    variance = residual.var(axis=0, ddof=1)
    factors = np.ones(data.shape[1:], dtype=np.float64)
    for row in range(height):
        count = int(counts[row])
        if count > 1:
            factors[row, selected[row]] = count / (count - 1.0)
    return float(np.mean(variance[selected] * factors[selected]))


def _find_saturation(
    level_stats: list[_LevelStats],
    nominal_sat_dn: float,
) -> tuple[_LevelStats, float, float] | None:
    """Return the highest level that clips and its saturation code in DN.

    The absolute floor of one quarter of the nominal saturation code keeps the
    sub-LSB population at low signal, which also has near-zero variance, from
    being mistaken for a clipped highlight when a sweep never reaches the
    ceiling.
    """
    global_max = max(level.max_mean_dn for level in level_stats)
    floor = max(_SATURATION_LEVEL_FRACTION * global_max, 0.25 * nominal_sat_dn)
    candidates: list[tuple[_LevelStats, float, float]] = []
    for level in level_stats:
        if level.max_mean_dn < floor:
            continue
        clipped: list[tuple[float, float]] = []
        for stats in level.channels.values():
            if (
                stats.clipped_fraction >= _MIN_SATURATION_CLIPPED_FRACTION
                and stats.clipped_mean_dn is not None
            ):
                clipped.append((stats.clipped_fraction, stats.clipped_mean_dn))
        if clipped:
            saturation_dn = max(value for _, value in clipped)
            fraction = max(value for value, _ in clipped)
            candidates.append((level, saturation_dn, fraction))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0].spec.mean_w_per_m2_nm)


def _extend_low_signal_sweep(
    source: _FrameSource,
    level_stats: list[_LevelStats],
    frames: int,
) -> list[float]:
    """Append geometrically lower levels until the PTC has a low anchor.

    Callers often supply a "bright" sweep copied from a plan whose levels are
    far above the full-well knee. The exact PTC model needs at least one level
    in the sub-ten-DN region to separate read noise from shot noise, so missing
    low levels are added at a quarter of the current lowest level. Added levels
    are recorded in provenance; the caller's original sweep is never removed.
    """
    added: list[float] = []
    for _ in range(_MAX_LOW_SIGNAL_LEVELS):
        primary = _primary_channel(level_stats)
        lowest = min(level_stats, key=lambda item: item.spec.mean_w_per_m2_nm)
        stats = lowest.channels.get(primary)
        if stats is None or stats.mean_dn <= _PTC_MIN_MEAN_DN:
            break
        if lowest.spec.mean_w_per_m2_nm * _LOW_LEVEL_SCALE < 1e-15:
            break
        new_spec = _scale_spec(lowest.spec, _LOW_LEVEL_SCALE)
        level_stats.append(source.measure(new_spec, frames))
        added.append(new_spec.mean_w_per_m2_nm)
    return added


# ---------------------------------------------------------------------------
# Photon transfer curve and derived quantities
# ---------------------------------------------------------------------------


def _fit_ptc(
    level_stats: list[_LevelStats],
    nominal_sat_dn: float,
) -> tuple[float, float, int]:
    """Fit the exact quantized photon-transfer model across usable points.

    The model for one channel is ``round(K * (Poisson(mu) + N(0, sigma)))``
    with the charge clipped at zero, where ``mu`` is the mean photoelectron
    count and ``sigma`` is the input-referred read noise. Both the mean and the
    variance predicted by that model are compared with the measured points:
    the mean fixes ``mu`` at each level and the variance fixes ``K`` and
    ``sigma``. This removes the phase-dependent bias of the asymptotic
    ``1/12 DN**2`` quantization correction, which is substantial when the read
    noise is only about a quarter of an LSB.

    Returns ``(K, read_noise_e, point_count)``. Saturated channels are excluded
    per channel so a partially clipped level can still contribute its
    unsaturated colours.
    """
    global_max = max(level.max_mean_dn for level in level_stats)
    high_floor = max(_SATURATION_LEVEL_FRACTION * global_max, 0.25 * nominal_sat_dn)
    xs: list[float] = []
    ys: list[float] = []
    for level in level_stats:
        high_level = level.max_mean_dn >= high_floor
        for stats in level.channels.values():
            if high_level and stats.clipped_fraction > _MAX_PTC_CLIPPED_FRACTION:
                continue
            if stats.mean_dn < _PTC_MIN_MEAN_DN or stats.variance_dn2 <= 0.0:
                continue
            xs.append(stats.mean_dn)
            ys.append(stats.variance_dn2)
    if len(xs) < _MIN_PTC_POINTS:
        raise ValueError(
            f"photon transfer curve needs at least {_MIN_PTC_POINTS} usable "
            f"channel-level points at or above {_PTC_MIN_MEAN_DN:g} DN; got {len(xs)}. "
            "Supply a sweep between the noise floor and saturation"
        )
    x_array = np.asarray(xs, dtype=np.float64)
    y_array = np.asarray(ys, dtype=np.float64)

    # Seed the nonlinear fit with the asymptotic linear PTC. That fit is only
    # approximate for a sub-LSB read noise, but it reliably lands in the right
    # basin for the exact model.
    approximate_x: list[float] = []
    approximate_y: list[float] = []
    for mean_dn, variance_dn2 in zip(xs, ys, strict=True):
        approximate = variance_dn2 - _QUANTIZATION_VARIANCE_DN2
        if approximate > 0.0:
            approximate_x.append(mean_dn)
            approximate_y.append(approximate)
    gain0 = 0.0
    sigma0 = 1.0
    if len(approximate_x) >= 2:
        coefficients = np.polyfit(
            np.asarray(approximate_x, dtype=np.float64),
            np.asarray(approximate_y, dtype=np.float64),
            1,
            w=1.0 / np.maximum(np.asarray(approximate_x, dtype=np.float64), 1e-6),
        )
        gain0 = float(coefficients[0])
        intercept0 = float(coefficients[1])
        if gain0 > 0.0 and math.isfinite(intercept0):
            sigma0 = math.sqrt(max(intercept0, 1e-6)) / gain0
    if gain0 <= 0.0 or not math.isfinite(gain0):
        high = [
            (mean_dn, variance_dn2)
            for mean_dn, variance_dn2 in zip(xs, ys, strict=True)
            if mean_dn > 5.0
        ]
        if high:
            gain0 = float(np.median([variance / mean for mean, variance in high]))
        else:
            gain0 = float(np.median(y_array / x_array))
        sigma0 = 1.0
    gain0 = min(max(gain0, 1e-4), 100.0)
    sigma0 = min(max(sigma0, 1e-3), 100.0)

    def residuals(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        gain, sigma = float(parameters[0]), float(parameters[1])
        if gain <= 0.0 or sigma <= 0.0:
            return np.full(x_array.shape, 1e6, dtype=np.float64)
        residual: list[float] = []
        for mean_dn, variance_dn2 in zip(xs, ys, strict=True):
            mean_e = _invert_model_mean(mean_dn, gain, sigma)
            _, model_variance = _quantized_moments(mean_e, gain, sigma)
            residual.append((model_variance - variance_dn2) / max(variance_dn2, 0.5))
        return np.asarray(residual, dtype=np.float64)

    result = least_squares(
        residuals,
        x0=np.asarray([gain0, sigma0], dtype=np.float64),
        bounds=([1e-6, 1e-6], [100.0, 100.0]),
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
        max_nfev=80,
    )
    if not result.success:
        raise ValueError(f"photon transfer fit did not converge: {result.message}")
    gain = float(result.x[0])
    read_noise = float(result.x[1])
    if not math.isfinite(gain) or not math.isfinite(read_noise):
        raise ValueError("photon transfer fit produced a non-finite gain or read noise")
    if gain <= 0.0 or read_noise <= 0.0:
        raise ValueError("photon transfer fit produced a non-positive gain or read noise")
    return gain, read_noise, len(xs)


def _quantized_moments(
    mean_e: float,
    gain: float,
    read_noise_e: float,
) -> tuple[float, float]:
    """Return ``(E[DN], Var[DN])`` for Poisson + read noise + ADC rounding.

    The charge is ``Poisson(max(0, mean_e))`` plus Gaussian read noise, clipped
    at zero charge, and rounded to an integer code with ``DN = round(K*charge)``.
    Above ``_POISSON_MODEL_MAX_E`` the Poisson distribution is replaced by a
    Gaussian with the same mean and variance; the difference in the first two
    moments is below 0.1 %.
    """
    mean_e = max(float(mean_e), 0.0)
    sigma = max(float(read_noise_e), 1e-9)
    if mean_e <= _POISSON_MODEL_MAX_E:
        nmax = int(mean_e + 10.0 * math.sqrt(mean_e) + 20.0)
        charges = np.arange(nmax + 1, dtype=np.float64)
        poisson_probabilities = np.asarray(poisson.pmf(charges, mean_e), dtype=np.float64)
        max_code = math.ceil(gain * (nmax + 8.0 * sigma)) + 2
        codes = np.arange(max_code + 1, dtype=np.float64)
        upper = (codes + 0.5) / gain
        lower = (codes - 0.5) / gain
        lower[0] = -np.inf
        conditional = norm.cdf((upper[None, :] - charges[:, None]) / sigma) - norm.cdf(
            (lower[None, :] - charges[:, None]) / sigma
        )
        probabilities = np.clip(poisson_probabilities @ conditional, 0.0, None)
    else:
        standard_deviation = math.sqrt(mean_e + sigma * sigma)
        max_code = math.ceil(gain * (mean_e + 8.0 * standard_deviation)) + 2
        codes = np.arange(max_code + 1, dtype=np.float64)
        upper = (codes + 0.5) / gain
        lower = (codes - 0.5) / gain
        lower[0] = -np.inf
        probabilities = np.clip(
            norm.cdf((upper - mean_e) / standard_deviation)
            - norm.cdf((lower - mean_e) / standard_deviation),
            0.0,
            None,
        )
    expected = float(probabilities @ codes)
    variance = float(((codes - expected) ** 2) @ probabilities)
    return expected, variance


def _invert_model_mean(mean_dn: float, gain: float, read_noise_e: float) -> float:
    """Return the photoelectron mean whose model expectation is ``mean_dn``."""
    low = 0.0
    high = max(2.0 * mean_dn / gain + 20.0, 50.0)
    for _ in range(24):
        middle = 0.5 * (low + high)
        model_mean, _ = _quantized_moments(middle, gain, read_noise_e)
        if model_mean < mean_dn:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _measure_dsnu(
    source: _FrameSource,
    frames: int,
    gain: float,
    read_noise_e: float,
    wavelengths_nm: NDArray[np.float64],
    pitch_um: float,
) -> tuple[float, float]:
    """Measure DSNU from dark frames, inverting the zero-charge clipping knee."""
    dark = _flat_spec(0.0, source.native_shape, wavelengths_nm, pitch_um)
    data: list[NDArray[np.float64]] = []
    valid: NDArray[np.bool_] | None = None
    masks: dict[str, NDArray[np.bool_]] | None = None
    for _ in range(frames):
        frame = source.expose_spec(dark)
        if valid is None:
            valid = _valid_mask(frame)
            masks = cfa_masks(frame.cfa, frame.data.shape)
        data.append(_signal_dn(frame))
    assert valid is not None and masks is not None
    stack = np.stack(data, axis=0)
    mean_map = stack.mean(axis=0)
    variance_map = stack.var(axis=0, ddof=1)
    centered: list[NDArray[np.float64]] = []
    for mask in masks.values():
        selected = mask & valid
        if int(selected.sum()) < _MIN_CHANNEL_PIXELS:
            continue
        values = mean_map[selected]
        centered.append(values - values.mean())
    if not centered:
        raise ValueError("no valid pixels available for the dark-frame DSNU measurement")
    spatial_variance = float(np.var(np.concatenate(centered), ddof=1))
    # The spatial variance of the averaged frame still contains the temporal
    # noise of each pixel mean; remove it to isolate the fixed pattern.
    temporal_variance = float(np.mean(variance_map[valid]))
    fixed_variance_dn2 = max(spatial_variance - temporal_variance / frames, 0.0)
    observed_dn = math.sqrt(fixed_variance_dn2)
    return _invert_clipped_dsnu(observed_dn, read_noise_e, gain), observed_dn / gain


def _invert_clipped_dsnu(observed_dn: float, read_noise_e: float, gain: float) -> float:
    """Invert the pre-clamp Poisson/read/quantization dark transfer.

    For a pixel with dark offset ``d`` the sensor forms ``max(0, d)``, samples
    Poisson photoelectrons, adds Gaussian read noise, clips the charge at zero,
    and rounds to an integer DN. Both the zero-charge clamp and the sub-LSB ADC
    quantization compress the observed spatial spread, and LCG is especially
    compressed (one DN is about eight electrons). The forward model computes
    the expected code for every offset, including all three effects; numeric
    inversion returns the pre-clamp DSNU that EMVA 1288 defines.
    """
    if observed_dn <= 0.0:
        return 0.0
    sigma = max(read_noise_e, 1e-9)
    grid = np.linspace(1e-4, _DSNU_MODEL_MAX_E, _DSNU_MODEL_POINTS)
    model = np.asarray([_observed_quantized_mean_std(value, sigma, gain) for value in grid])
    if not np.all(np.diff(model) >= -1e-12):
        raise ValueError(
            "DSNU inversion model is not monotonic in the offset standard deviation; "
            "the dark-frame statistics are too noisy for a unique inversion"
        )
    if observed_dn > float(model[-1]):
        raise ValueError(
            f"observed dark fixed-pattern spread ({observed_dn:g} DN) exceeds the "
            f"DSNU model range ({_DSNU_MODEL_MAX_E:g} e-); increase the model range "
            "rather than silently clamping the result"
        )
    return float(np.interp(observed_dn, model, grid))


def _observed_quantized_mean_std(dsnu_e: float, sigma_e: float, gain: float) -> float:
    """Return the spatial std in DN of the expected quantized dark code.

    The pre-Poisson clamp is explicit: a pixel with offset ``d`` samples
    ``Poisson(max(0, d))`` and then the usual read-noise, zero-charge clip, and
    rounding chain. The expectation over the Gaussian offset population uses a
    uniform grid with Gaussian weights rather than a low-order Hermite rule:
    the expected code has a sharp knee at ``d = 0``, which a small polynomial
    quadrature resolves poorly.
    """
    span = 6.0 * dsnu_e
    offsets = np.linspace(-span, span, _DSNU_QUADRATURE_POINTS)
    weights = norm.pdf(offsets / dsnu_e) / dsnu_e
    weights = weights / weights.sum()
    expected_codes = np.asarray(
        [_quantized_moments(max(float(offset), 0.0), gain, sigma_e)[0] for offset in offsets],
        dtype=np.float64,
    )
    average = float((weights * expected_codes).sum())
    variance = float((weights * (expected_codes - average) ** 2).sum())
    return math.sqrt(max(variance, 0.0))


def _measure_prnu(
    level_stats: list[_LevelStats],
    primary: str,
    saturation_dn: float,
    frames: int,
    dsnu_e: float,
    gain: float,
) -> tuple[float, float, float]:
    """Measure PRNU in percent from the flat level nearest 70 % of saturation."""
    target = 0.7 * saturation_dn
    candidates = [
        (level, level.channels[primary])
        for level in level_stats
        if primary in level.channels and level.channels[primary].mean_dn > 0.0
    ]
    if not candidates:
        raise ValueError(f"primary channel {primary!r} has no usable flat-field statistics")
    unsaturated = [
        (level, stats)
        for level, stats in candidates
        if stats.clipped_fraction <= _MAX_PTC_CLIPPED_FRACTION
    ]
    pool = unsaturated or candidates
    level, stats = min(pool, key=lambda item: abs(item[1].mean_dn - target))
    mean_dn = stats.mean_dn
    temporal = stats.raw_variance_dn2 / frames
    dsnu_dn2 = (dsnu_e * gain) ** 2
    prnu_variance = (stats.mean_map_variance_dn2 - temporal - dsnu_dn2) / (mean_dn * mean_dn)
    return 100.0 * math.sqrt(max(prnu_variance, 0.0)), level.spec.mean_w_per_m2_nm, mean_dn


def _measure_snr_curve(
    level_stats: list[_LevelStats],
    primary: str,
    gain: float,
    nominal_sat_dn: float,
) -> tuple[SNRPoint, ...]:
    """Build the pixel-temporal SNR curve from unsaturated primary-channel levels."""
    global_max = max(level.max_mean_dn for level in level_stats)
    high_floor = max(_SATURATION_LEVEL_FRACTION * global_max, 0.25 * nominal_sat_dn)
    points: list[SNRPoint] = []
    for level in level_stats:
        stats = level.channels.get(primary)
        if stats is None:
            continue
        # The clipping flag is only meaningful near the ceiling; sub-LSB low
        # signal also has near-zero variance and must stay on the curve.
        if level.max_mean_dn >= high_floor and stats.clipped_fraction > _MAX_PTC_CLIPPED_FRACTION:
            continue
        if stats.variance_dn2 <= 0.0 or stats.mean_dn <= 0.0:
            continue
        noise_dn = math.sqrt(stats.variance_dn2)
        noise_e = noise_dn / gain
        signal_e = stats.mean_dn / gain
        total_noise_dn = math.sqrt(max(stats.raw_variance_dn2, 0.0))
        total_noise_e = total_noise_dn / gain
        total_snr = 20.0 * math.log10(signal_e / total_noise_e) if total_noise_e > 0.0 else None
        points.append(
            SNRPoint(
                irradiance_w_per_m2_nm=level.spec.mean_w_per_m2_nm,
                mean_signal_dn=stats.mean_dn,
                mean_signal_e=signal_e,
                noise_dn=noise_dn,
                noise_e=noise_e,
                snr_db=20.0 * math.log10(signal_e / noise_e),
                noise_dn_total=total_noise_dn,
                noise_e_total=total_noise_e,
                snr_db_total=total_snr,
            )
        )
    if not points:
        raise ValueError(
            "no usable unsaturated levels remain for an SNR curve; supply a sweep "
            "between the noise floor and saturation"
        )
    points.sort(key=lambda point: point.irradiance_w_per_m2_nm)
    return tuple(points)


def _measure_linearity(
    level_stats: list[_LevelStats],
    primary: str,
    saturation_dn: float,
) -> tuple[float, str]:
    """Return the maximum relative deviation and the channel used.

    The primary channel is preferred, but a partially clipped Bayer sweep can
    leave it with only one unsaturated point while another colour still spans
    the linear range. Falling back to that channel keeps the residual
    measurable instead of failing an otherwise valid characterization.
    """
    available = sorted(level_stats[0].channels)
    order = [primary, *(channel for channel in available if channel != primary)]
    for channel in order:
        # Select by signal range rather than by the clipping flag: at low
        # signal, sub-LSB quantization gives many pixels a variance below the
        # clipping threshold, so that flag is only meaningful near the ceiling.
        candidates: list[tuple[float, float]] = []
        for level in level_stats:
            stats = level.channels.get(channel)
            if stats is None:
                continue
            mean_dn = stats.mean_dn
            if 0.05 * saturation_dn <= mean_dn <= 0.95 * saturation_dn:
                candidates.append((level.spec.mean_w_per_m2_nm, mean_dn))
        if len(candidates) < 2:
            candidates = [
                (level.spec.mean_w_per_m2_nm, level.channels[channel].mean_dn)
                for level in level_stats
                if channel in level.channels
                and 0.0 < level.channels[channel].mean_dn <= 0.95 * saturation_dn
            ]
        if len(candidates) < 2:
            continue
        x = np.asarray([item[0] for item in candidates], dtype=np.float64)
        y = np.asarray([item[1] for item in candidates], dtype=np.float64)
        coefficients = np.polyfit(x, y, 1)
        fitted = np.polyval(coefficients, x)
        if np.any(fitted <= 0.0):
            continue
        return float(np.max(np.abs(y - fitted) / fitted)), channel
    raise ValueError("not enough unsaturated levels to measure linearity")


# ---------------------------------------------------------------------------
# Configuration-derived nominal values
# ---------------------------------------------------------------------------


def _nominal_system_gain(config: SensorConfig, mode: GainMode) -> float:
    """Return the ADC system gain in DN/e- expected from the configuration."""
    usable_codes = float((1 << int(config.adc.bits)) - 1 - int(config.adc.black_level_dn))
    analog = 10.0 ** (float(config.analog_gain.default_db) / 20.0)
    digital = float(config.adc.digital_gain)
    conversion_gain = float(config.modes[mode].conversion_gain_uv_per_e)
    full_scale = float(config.adc.full_scale_uv)
    return float(conversion_gain * usable_codes / full_scale * analog * digital)


def _nominal_saturation_dn(config: SensorConfig, mode: GainMode) -> float:
    """Return the smaller of the full-well and ADC saturation codes in DN."""
    usable_codes = float((1 << int(config.adc.bits)) - 1 - int(config.adc.black_level_dn))
    full_well_dn = _nominal_system_gain(config, mode) * float(config.modes[mode].full_well_e)
    return min(full_well_dn, usable_codes)


def _nominal_saturation_e(config: SensorConfig, mode: GainMode) -> float:
    """Return the configured saturation capacity in electrons."""
    return _nominal_saturation_dn(config, mode) / _nominal_system_gain(config, mode)


# ---------------------------------------------------------------------------
# Small helpers and validation
# ---------------------------------------------------------------------------


def _sensor_wavelengths(config: SensorConfig) -> NDArray[np.float64]:
    """Reconstruct the sensor's uniform wavelength grid from its preset."""
    range_config = config.spectral.wavelength_nm
    start = float(range_config.start)
    stop = float(range_config.stop)
    step = float(range_config.step)
    steps = (stop - start) / step
    count = round(steps)
    if count < 1 or not math.isclose(steps, float(count), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"sensor wavelength grid {start:g}-{stop:g} nm with {step:g} nm steps "
            "does not contain an integer number of intervals"
        )
    return np.asarray(start + step * np.arange(count + 1, dtype=np.float64), dtype=np.float64)


def _primary_channel(level_stats: list[_LevelStats]) -> str:
    """Return the preferred channel for scalar read-noise and SNR measurements."""
    available = set(level_stats[0].channels)
    for candidate in ("G", "M"):
        if candidate in available:
            return candidate
    return sorted(available)[0]


def _frame_mean_dn(frame: RawFrame) -> float:
    """Return the mean black-level-corrected signal over valid pixels."""
    signal = _signal_dn(frame)
    valid = _valid_mask(frame)
    if not bool(valid.any()):
        return 0.0
    return float(signal[valid].mean())


def _validated_mode(value: object) -> Mode:
    """Return a validated readout-mode request."""
    if value not in ("auto", "LCG", "HCG"):
        raise ValueError(f"mode must be 'auto', 'LCG', or 'HCG'; got {value!r}")
    return value


def _validated_lens(value: object) -> LensConfig:
    """Return the supplied lens or the reference phone lens for lux conversion."""
    if value is None:
        return LensConfig(
            f_number=_REFERENCE_LENS_F_NUMBER,
            focal_length_mm=_REFERENCE_LENS_FOCAL_LENGTH_MM,
            sensor_diagonal_mm=_REFERENCE_LENS_DIAGONAL_MM,
            transmittance=_REFERENCE_LENS_TRANSMITTANCE,
        )
    if not isinstance(value, LensConfig):
        raise ValueError(f"lens must be a LensConfig or None; got {type(value).__name__}")
    return value


def _validated_levels(values: object, field: str) -> list[float]:
    """Return sorted, deduplicated, finite, positive levels with at least three."""
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f"{field} must be a sequence of positive numbers; got {values!r}")
    levels = sorted({_positive_float(value, field) for value in values})
    if len(levels) < _MIN_LEVELS:
        raise ValueError(
            f"{field} must contain at least {_MIN_LEVELS} distinct positive levels; "
            f"got {len(levels)}"
        )
    return levels


def _validated_frames(value: object) -> int:
    """Return a frame count of at least two."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"frames_per_level must be an integer; got {value!r}")
    count = int(value)
    if count < _MIN_FRAMES_PER_LEVEL:
        raise ValueError(f"frames_per_level must be at least {_MIN_FRAMES_PER_LEVEL}; got {count}")
    return count


def _finite_float(value: object, field: str) -> float:
    """Return ``value`` as a finite float."""
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


def _non_negative_int(value: object, field: str) -> int:
    """Return a non-negative Python integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be a non-negative integer; got {result}")
    return result
