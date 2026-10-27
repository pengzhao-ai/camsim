"""Validated, versioned configuration models for sensors, lenses, and ISPs.

Configuration is the numerical interface between a datasheet and the signal
chain. Every model here is therefore explicit about units (``_um``, ``_e``,
``_uv``, ``_db``, ``_s``, ``_c``, ``_k``), forbids unknown keys, and rejects
non-finite values, because a silently ignored typo in ``full_well_e`` or a
microvolt/millivolt mix-up produces a sweep that runs to completion and is
simply wrong. Provenance (source, URL, retrieval date, assumptions) is a
required part of every preset, so a number can always be traced back to a
datasheet or to an engineering estimate.

Physical foundations encoded as cross-field rules:

* **Full well versus conversion gain.** A pixel saturates either in the charge
  domain (``full_well_e`` electrons) or in the ADC (``full_scale_uv`` volts).
  At unity analog gain the voltage at full well is
  ``full_well_e * conversion_gain_uv_per_e``; if it exceeds ``full_scale_uv``
  the ADC clips first. That is legal but changes noise and highlight behaviour,
  so it is reported as a warning rather than rejected.
* **Dual conversion gain.** High conversion gain trades full well for lower
  read noise; ``HCG.full_well_e <= LCG.full_well_e`` keeps the two modes
  physically ordered.
* **Colour matrices.** Spectral crosstalk and colour-correction matrices are
  row-normalised (sum of each row is one) so a neutral input stays neutral;
  the tolerances below absorb decimal rounding in published matrices.
* **Geometry.** A lens of focal length ``f`` projects a sensor diagonal ``d``
  onto a field of view ``2 atan(d / 2f)``, and a pixel of pitch ``p`` subtends
  ``atan(p / f)``, which yields pixels per degree for cycles/degree MTF axes.

Real-world parameter ranges reflected in the presets: pixel pitch 0.6-10 um;
conversion gain 40-500 uV/e-; full well 2-50 ke-; read noise 0.3-10 e- rms;
dark current 0.01-100 e-/s at 25 C, doubling every 5-10 K; analog gain 0-30 dB;
ADC 8-16 bits with a 0.3-1.2 V reference; exposure 10 us-10 s; black level
64 DN at 10 bits and 256 DN at 12 bits. CSV paths (``qe_csv``, ``cfa_csv``,
``ir_cut_csv``, spectral lens transmittance) are deliberately kept as strings:
presets may be written before the datasets land, and the files are resolved at
sensor construction time, not at schema-validation time. The shipped preset
uses package-relative ``spectra/*.csv`` paths that resolve inside
``camsim.data`` after installation; an absolute or working-directory path still
overrides them. An optional ``spatial_crosstalk`` kernel is a normalised 2-D
stencil applied per CFA channel plane in the electron domain; ``None`` disables
charge diffusion.

Failure artefacts guarded here: negative or non-finite gains; an ADC black
level outside the code range (bright pedestal after black-level subtraction);
a default exposure outside its own limits (auto-exposure dead on arrival);
dual-gain modes whose full wells cross; unnormalised colour matrices (colour
casts that no white balance can remove); a zero pixel pitch (infinite
pixels/degree); and unknown JSON keys (a misspelled stage toggle silently
leaving a stage enabled). Each condition raises a validation error naming the
offending field.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Final, Literal, Self, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    PlainSerializer,
    PositiveFloat,
    StringConstraints,
    WithJsonSchema,
    model_validator,
)

from camsim.core.frames import CFAPattern, ColorSpace

SchemaVersion: TypeAlias = Literal["1.0"]
"""Configuration schema versions this package can load.

Bumping the version is a deliberate, reviewed act: incompatible presets must
fail loudly rather than be reinterpreted under new semantics.
"""

NonEmptyStr: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""A stripped, non-empty string (names, identifiers, CSV paths)."""

Fraction: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
"""A dimensionless quantity in the closed unit interval."""

PositiveFraction: TypeAlias = Annotated[float, Field(gt=0.0, le=1.0)]
"""A dimensionless quantity in ``(0, 1]`` such as an effective fill factor."""

NonNegative: TypeAlias = Annotated[float, Field(ge=0.0)]
"""A finite, non-negative physical quantity."""

GainMode: TypeAlias = Literal["LCG", "HCG"]
"""Dual-conversion-gain readout modes."""

_ROW_SUM_ABS_TOLERANCE: Final[float] = 1e-6


def _rows_sum_to_one(value: list[list[float]]) -> list[list[float]]:
    """Reject matrices whose rows do not sum to one.

    Row normalisation keeps the neutral axis fixed: a grey scene must stay grey
    after spectral crosstalk or a colour-correction matrix. ``math.fsum`` avoids
    the cancellation error of naive summation for matrices with large positive
    and negative off-diagonal entries.
    """
    for index, row in enumerate(value):
        total = math.fsum(row)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_ROW_SUM_ABS_TOLERANCE):
            raise ValueError(
                f"row {index} must sum to 1.0 (got {total:g}); "
                "row-normalised matrices preserve neutral-axis energy"
            )
    return value


Matrix3x3: TypeAlias = Annotated[
    list[Annotated[list[float], Field(min_length=3, max_length=3)]],
    Field(min_length=3, max_length=3),
    AfterValidator(_rows_sum_to_one),
]
"""A 3x3 row-normalised matrix stored as nested JSON-friendly lists."""


def _normalized_kernel(value: list[list[float]]) -> list[list[float]]:
    """Reject spatial-crosstalk kernels that are malformed or non-conservative.

    Optical and electrical charge diffusion moves signal between neighbouring
    pixels; it does not create or destroy charge. A kernel must therefore be a
    non-empty rectangular grid of finite, non-negative weights whose sum is one.
    ``math.fsum`` avoids the cancellation error of naive summation. Rescaling a
    kernel that misses unity is deliberately not attempted: a calibration that
    sums to 0.87 is a bug in the caller's data, and silently normalising it
    would hide the gain error.
    """
    if not value:
        raise ValueError("spatial crosstalk kernel must contain at least one row")
    widths = {len(row) for row in value}
    if len(widths) != 1:
        raise ValueError(
            f"spatial crosstalk kernel must be rectangular; row widths are {sorted(widths)}"
        )
    if 0 in widths:
        raise ValueError("spatial crosstalk kernel rows must contain at least one weight")
    for row_index, row in enumerate(value):
        for column_index, weight in enumerate(row):
            if not math.isfinite(weight):
                raise ValueError(
                    "spatial crosstalk kernel weights must be finite; "
                    f"got {weight!r} at row {row_index}, column {column_index}"
                )
            if weight < 0.0:
                raise ValueError(
                    "spatial crosstalk kernel weights must be non-negative; "
                    f"got {weight:g} at row {row_index}, column {column_index}"
                )
    total = math.fsum(math.fsum(row) for row in value)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_ROW_SUM_ABS_TOLERANCE):
        raise ValueError(
            f"spatial crosstalk kernel must sum to 1.0 (got {total:g}); "
            "a normalised kernel conserves total charge"
        )
    return value


Kernel2D: TypeAlias = Annotated[
    list[Annotated[list[float], Field(min_length=1)]],
    Field(min_length=1),
    AfterValidator(_normalized_kernel),
]
"""A non-empty, rectangular, non-negative 2-D kernel whose sum is one."""


def _as_gain_map(value: Any) -> NDArray[np.float64]:
    """Coerce a lens-shading gain map to a finite, positive 2-D float array.

    JSON supplies nested lists while callers such as the LSC calibration helper
    supply NumPy arrays; both are accepted. Flat-field gain maps are divisions
    of a reference level by a measured flat field, so a zero or negative entry
    means calibration failed and is rejected instead of producing infinities.

    Conversion errors are re-raised as ``ValueError`` because Pydantic wraps
    ``ValueError`` from a ``BeforeValidator`` into a ``ValidationError`` but
    lets ``TypeError`` escape. A JSON object such as ``{"a": 1}`` must become a
    validation report, not a traceback in the CLI.
    """
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"LSC map must be a rectangular 2-D array of numbers; got {type(value).__name__}"
        ) from exc
    if array.ndim != 2:
        raise ValueError(f"LSC map must be a 2-D array; got shape {array.shape}")
    if array.size == 0:
        raise ValueError("LSC map must contain at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError("LSC map values must be finite")
    if np.any(array <= 0.0):
        raise ValueError("LSC map values must be strictly positive")
    return array


def _gain_map_to_list(value: NDArray[np.float64]) -> list[list[float]]:
    """Serialise an LSC map back to nested lists for JSON round-trips."""
    return cast(list[list[float]], value.tolist())


Array2D: TypeAlias = Annotated[
    NDArray[np.float64],
    BeforeValidator(_as_gain_map),
    PlainSerializer(_gain_map_to_list, return_type=list),
    WithJsonSchema(
        {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Row-major 2-D array of finite, positive values.",
        }
    ),
]
"""A 2-D floating-point array accepted as nested JSON lists or NumPy input."""


class ConfigModel(BaseModel):
    """Base for every configuration model.

    ``extra="forbid"`` turns misspelled keys into validation errors, and
    ``allow_inf_nan=False`` rejects ``NaN``/``Infinity`` payloads that strict
    JSON does not carry but Python's ``json`` module accepts by default.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Provenance(ConfigModel):
    """Traceability metadata for a preset.

    Parameters
    ----------
    source:
        Where the numbers came from, for example a datasheet name or
        "representative engineering assumptions".
    url:
        Optional public reference for the source.
    retrieved:
        Optional ISO 8601 date (``YYYY-MM-DD``) on which the source was read.
    notes:
        Free-form caveats about the preset as a whole.
    assumptions:
        Explicit list of values that are estimates rather than measurements.
        Presets are representative, not claims about a specific part.
    """

    source: NonEmptyStr
    url: NonEmptyStr | None = None
    retrieved: NonEmptyStr | None = None
    notes: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class PixelConfig(ConfigModel):
    """Pixel geometry and light-collection efficiency.

    ``fill_factor`` is the photosensitive fraction of the pixel footprint and
    ``microlens_gain`` the extra collection from the microlens; their product,
    clipped at one, is the effective aperture fraction used by the spectral
    projection. A pitch of 1.12 um with fill 0.85 and microlens gain 1.15 is
    representative of a modern stacked phone sensor.
    """

    pitch_um: PositiveFloat
    fill_factor: PositiveFraction
    microlens_gain: PositiveFloat


class WavelengthRange(ConfigModel):
    """Uniform spectral sampling grid, in nanometres.

    Fields keep their short design names (``start``/``stop``/``step``) because
    the enclosing model is named ``wavelength_nm``; every value is in nm.
    The 360-830 nm, 5 nm grid covers the visible band with enough resolution
    for colorimetric integrals while staying cheap to store.
    """

    start: PositiveFloat
    stop: PositiveFloat
    step: PositiveFloat

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        span = self.stop - self.start
        if span <= 0.0:
            raise ValueError(
                f"stop ({self.stop:g} nm) must be greater than start ({self.start:g} nm)"
            )
        if self.step > span:
            raise ValueError(
                f"step ({self.step:g} nm) must not exceed the start-to-stop span ({span:g} nm)"
            )
        return self


class SpectralConfig(ConfigModel):
    """Spectral response files and channel crosstalk.

    The CSV paths are strings resolved when a sensor is constructed, so a
    preset can be validated before the data files exist. ``spectral_crosstalk``
    mixes neighbouring CFA channels in the charge domain; rows sum to one so a
    neutral input stays neutral (a doubly stochastic matrix also conserves the
    channel sum). The identity matrix models an ideal sensor with no spectral
    leakage.
    """

    wavelength_nm: WavelengthRange
    qe_csv: NonEmptyStr
    cfa_csv: NonEmptyStr
    ir_cut_csv: NonEmptyStr
    spectral_crosstalk: Matrix3x3


class GainModeConfig(ConfigModel):
    """One conversion-gain readout mode (LCG or HCG).

    ``conversion_gain_uv_per_e`` converts electrons to microvolts at the sense
    node; ``full_well_e`` is the charge-domain saturation; ``read_noise_e_rms``
    is input-referred; ``dark_current_e_per_s`` is specified at the sensor's
    reference temperature and scaled by the temperature model.
    """

    conversion_gain_uv_per_e: PositiveFloat
    full_well_e: PositiveFloat
    read_noise_e_rms: NonNegative
    dark_current_e_per_s: NonNegative


class AnalogGainConfig(ConfigModel):
    """Analog programmable-gain-amplifier limits in decibels.

    ``default_db`` must lie within ``[min_db, max_db]``; ``step_db`` documents
    the settable granularity so sweeps do not request unattainable gains.
    """

    min_db: NonNegative
    max_db: PositiveFloat
    step_db: PositiveFloat
    default_db: NonNegative

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.max_db < self.min_db:
            raise ValueError(f"max_db ({self.max_db:g} dB) must be >= min_db ({self.min_db:g} dB)")
        if not self.min_db <= self.default_db <= self.max_db:
            raise ValueError(
                f"default_db ({self.default_db:g} dB) must lie within "
                f"[min_db, max_db] = [{self.min_db:g}, {self.max_db:g}] dB"
            )
        return self


class ADCConfig(ConfigModel):
    """Analog-to-digital converter scale and black level.

    ``full_scale_uv`` is the input voltage that produces the largest code;
    ``black_level_dn`` is the integer code that represents zero signal and must
    fit inside the ``2**bits`` code space. ``digital_gain`` multiplies signal
    codes after conversion and before the black level is re-added.
    """

    bits: Annotated[int, Field(ge=8, le=16)]
    full_scale_uv: PositiveFloat
    black_level_dn: NonNegativeInt
    digital_gain: PositiveFloat

    @model_validator(mode="after")
    def _validate_black_level(self) -> Self:
        code_count = 1 << self.bits
        if self.black_level_dn >= code_count:
            raise ValueError(
                f"black_level_dn ({self.black_level_dn:g}) must be less than "
                f"2**bits = {code_count} for bits={self.bits}"
            )
        return self


class ExposureConfig(ConfigModel):
    """Integration-time limits in seconds.

    Phone main cameras commonly run 10 us to 100 ms; the default must be
    inside the limits so a fixed-exposure capture is always realisable.
    """

    min_s: PositiveFloat
    max_s: PositiveFloat
    default_s: PositiveFloat

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.max_s < self.min_s:
            raise ValueError(f"max_s ({self.max_s:g} s) must be >= min_s ({self.min_s:g} s)")
        if not self.min_s <= self.default_s <= self.max_s:
            raise ValueError(
                f"default_s ({self.default_s:g} s) must lie within "
                f"[min_s, max_s] = [{self.min_s:g}, {self.max_s:g}] s"
            )
        return self


class TemperatureConfig(ConfigModel):
    """Temperature reference and dark-current doubling behaviour.

    Dark current follows ``D(T) = D_ref * 2**((T - T_ref) / T_double)``. Silicon
    dark current roughly doubles every 5-10 K, so a doubling temperature of 7 K
    is representative.
    """

    reference_c: float
    default_c: float
    dark_current_doubling_k: PositiveFloat


class DefectConfig(ConfigModel):
    """Rates and seed for dead, hot, and stuck pixel maps.

    Rates are probabilities per pixel; 1e-5 is one defect in 100,000 pixels,
    typical of a well-controlled phone sensor. The seed makes the maps
    reproducible across runs.
    """

    dead_rate: Fraction
    hot_rate: Fraction
    stuck_rate: Fraction
    seed: NonNegativeInt


class NoiseConfig(ConfigModel):
    """Fixed-pattern and row-noise strengths in the electron domain.

    ``dsnu_e`` is the per-pixel dark-signal offset standard deviation;
    ``prnu_fraction`` is the dimensionless photo-response deviation (0.005 is
    0.5 %, not 5 %); ``row_noise_e_rms`` is the common-mode readout sigma per
    row; and ``hot_pixel_dark_multiplier`` scales dark current at sites flagged
    by :class:`DefectConfig`. These are the calibratable spatial and row
    imperfections that differ between nominally identical parts.
    """

    dsnu_e: NonNegative = 0.5
    prnu_fraction: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.005
    row_noise_e_rms: NonNegative = 0.5
    hot_pixel_dark_multiplier: Annotated[float, Field(ge=1.0)] = 100.0


class PDAFConfig(ConfigModel):
    """Phase-detection autofocus pixel layout.

    PDAF pixels sacrifice part of their aperture (``qe_factor`` < 1) and are
    flagged in the RAW frame so defect correction leaves them alone.
    ``fraction`` is the fraction of eligible sites used; ``pattern`` names the
    placement rule.
    """

    fraction: Fraction
    qe_factor: PositiveFraction
    pattern: Literal["sparse_green", "sparse_red", "sparse_blue", "none"]


class BinningConfig(ConfigModel):
    """Pixel binning factor and domain.

    Analog binning sums charge before the full-well limit (full well scales by
    ``factor**2`` and read noise adds in quadrature); digital binning averages
    codes after the ADC.
    """

    factor: Annotated[int, Field(ge=1, le=16)]
    mode: Literal["analog", "digital"]


class BloomingConfig(ConfigModel):
    """Charge-overflow spreading around saturated pixels.

    ``fraction`` of the excess charge is passed to neighbours each iteration;
    iterations bound the diffusion. Off by default because modern pinned
    photodiodes suppress blooming well.
    """

    enabled: bool
    fraction: Fraction
    iterations: NonNegativeInt


class SensorConfig(ConfigModel):
    """A complete sensor preset: geometry, spectra, gains, ADC, and defects.

    Cross-field rules live here because they span sub-models: the mode set must
    be exactly ``LCG``/``HCG``, HCG full well may not exceed LCG full well, and
    the ADC-limited saturation condition becomes a warning from
    :meth:`validation_warnings`. ``auto_hcg_threshold_e`` is the mean signal
    (electrons per pixel, before noise) below which ``mode="auto"`` selects the
    high-conversion-gain readout.

    ``spatial_crosstalk`` is an optional normalised charge-diffusion kernel
    applied to each CFA channel plane in the electron domain after spectral
    crosstalk and before read noise. ``None`` (the default) disables it. Unlike
    the 3x3 spectral-crosstalk matrix, which mixes colours at one pixel site,
    this kernel mixes neighbouring pixels of the same colour filter. A typical
    measured kernel is a small odd-sized stencil with most weight on the centre
    and a few percent on the four nearest neighbours.
    """

    schema_version: SchemaVersion
    name: NonEmptyStr
    provenance: Provenance
    pixel: PixelConfig
    cfa: CFAPattern
    spectral: SpectralConfig
    spatial_crosstalk: Kernel2D | None = None
    modes: dict[GainMode, GainModeConfig]
    analog_gain: AnalogGainConfig
    adc: ADCConfig
    exposure: ExposureConfig
    temperature: TemperatureConfig
    defects: DefectConfig
    pdaf: PDAFConfig
    binning: BinningConfig
    blooming: BloomingConfig
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    auto_hcg_threshold_e: NonNegative = 200.0

    @model_validator(mode="after")
    def _validate_modes(self) -> Self:
        expected = {"LCG", "HCG"}
        if set(self.modes) != expected:
            raise ValueError("modes must define exactly LCG and HCG")
        lcg = self.modes["LCG"]
        hcg = self.modes["HCG"]
        if hcg.full_well_e > lcg.full_well_e:
            raise ValueError(
                "HCG full_well_e must be <= LCG full_well_e; "
                f"got HCG {hcg.full_well_e:g} e- > LCG {lcg.full_well_e:g} e-"
            )
        return self

    def validation_warnings(self) -> list[str]:
        """Return non-fatal physical warnings for this configuration.

        The only warning implemented in schema version 1.0 is ADC-limited
        saturation. At the default analog gain, each mode's full-well voltage
        is compared with the ADC full scale; if the ADC clips first, highlight
        headroom and saturation SNR no longer follow from ``full_well_e``
        alone. This is legal (many sensors run ADC-limited in high-gain modes),
        so it is surfaced to users rather than rejected.
        """
        gain = 10.0 ** (self.analog_gain.default_db / 20.0)
        limited: list[str] = []
        for mode_name in ("LCG", "HCG"):
            mode = self.modes[mode_name]
            full_well_uv = mode.full_well_e * mode.conversion_gain_uv_per_e * gain
            if full_well_uv > self.adc.full_scale_uv:
                limited.append(f"{mode_name} full well reaches {full_well_uv:g} uV")
        if not limited:
            return []
        detail = "; ".join(limited)
        return [
            "At the default analog gain "
            f"({self.analog_gain.default_db:g} dB), the ADC full scale "
            f"({self.adc.full_scale_uv:g} uV) is reached before full well "
            f"({detail}); saturation is ADC-limited rather than full-well-limited."
        ]


class LensConfig(ConfigModel):
    """Reference lens geometry and transmission.

    Parameters
    ----------
    f_number:
        Relative aperture ``N``; real lenses span f/0.7 to f/22.
    focal_length_mm:
        Actual (not full-frame-equivalent) focal length.
    sensor_diagonal_mm:
        Active sensor diagonal used for the field-of-view calculation.
    transmittance:
        Scalar on-axis transmission in ``(0, 1]`` (0.9 is a good AR-coated
        lens) or a CSV path for a spectral transmission curve, resolved by the
        optics layer.
    vignetting:
        Optional strength in ``[0, 1]`` scaling the physical ``cos**4``
        falloff; ``None`` disables vignetting in the scene renderer.

    Derived quantities are properties rather than stored fields so they can
    never disagree with the geometry. The reference phone lens (f/1.8,
    6.4 mm actual focal length, 11.46 mm diagonal, T = 0.9) sees about 84
    degrees diagonally and about 100 pixels per degree at a 1.12 um pitch.
    """

    f_number: PositiveFloat
    focal_length_mm: PositiveFloat
    sensor_diagonal_mm: PositiveFloat
    transmittance: Annotated[float, Field(gt=0.0, le=1.0)] | NonEmptyStr
    vignetting: Fraction | None = None

    @property
    def field_of_view_deg(self) -> float:
        """Diagonal field of view in degrees, ``2 atan(d / (2 f))``."""
        half_angle = math.atan(self.sensor_diagonal_mm / (2.0 * self.focal_length_mm))
        return math.degrees(2.0 * half_angle)

    def pixels_per_degree(self, pixel_pitch_um: float) -> float:
        """Return how many pixels of ``pixel_pitch_um`` span one degree.

        A pixel subtends ``atan(pitch / f)`` radians at the lens centre. This
        small-angle conversion turns cycles/pixel MTF values into
        cycles/degree, the unit in which human acuity and object detection are
        usually discussed.
        """
        if not math.isfinite(pixel_pitch_um) or pixel_pitch_um <= 0.0:
            raise ValueError(f"pixel pitch must be finite and positive; got {pixel_pitch_um!r} um")
        pitch_mm = pixel_pitch_um * 1e-3
        angular_pitch_deg = math.degrees(math.atan(pitch_mm / self.focal_length_mm))
        return 1.0 / angular_pitch_deg


class DPCConfig(ConfigModel):
    """Defective-pixel correction: replace flagged pixels from same-CFA neighbours."""

    enabled: bool = True
    method: Literal["neighbor", "median"] = "neighbor"


class BLCConfig(ConfigModel):
    """Black-level correction: subtract the zero-signal pedestal and clamp."""

    enabled: bool = True


class LSCConfig(ConfigModel):
    """Lens-shading correction: analytic radial gain or a measured flat field.

    ``coefficients`` are radial polynomial coefficients for ``mode="radial"``;
    ``map`` is a positive 2-D gain map for ``mode="flat_field"`` (the reciprocal
    of a normalized flat field). The map may be supplied as nested JSON lists or
    as a NumPy array from the calibration helper.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, arbitrary_types_allowed=True)

    enabled: bool = True
    mode: Literal["none", "radial", "flat_field"] = "none"
    coefficients: list[float] | None = None
    map: Array2D | None = None

    @model_validator(mode="after")
    def _validate_mode_parameters(self) -> Self:
        if self.mode == "flat_field" and self.map is None:
            raise ValueError("LSC mode 'flat_field' requires a map")
        if self.mode == "radial" and not self.coefficients:
            raise ValueError("LSC mode 'radial' requires radial coefficients")
        return self


class AWBConfig(ConfigModel):
    """Auto-white-balance estimator settings.

    ``method`` selects gray-world, white-patch (brightest-pixel), or
    shades-of-gray. ``p`` is the Minkowski norm only for shades-of-gray and
    must be at least one. ``gains`` optionally pins per-channel gains (in
    R, G, B order) instead of estimating them.
    """

    enabled: bool = True
    method: Literal["gray_world", "white_patch", "shades_of_gray"] = "gray_world"
    p: Annotated[float, Field(ge=1.0)] = 6.0
    gains: tuple[PositiveFloat, PositiveFloat, PositiveFloat] | None = None


class DemosaicConfig(ConfigModel):
    """CFA demosaicing method.

    ``bilinear`` is the cheap baseline; ``mhc`` is Malvar-He-Cutler (2004),
    which adds gradient correction and markedly reduces zippering. Mono or RCCC
    inputs bypass this stage based on the sensor CFA, not on this toggle.
    """

    enabled: bool = True
    method: Literal["bilinear", "mhc"] = "mhc"


class CCMConfig(ConfigModel):
    """Colour-correction matrix and its least-squares fit settings.

    ``matrix`` is applied in linear RGB when supplied; it must be row-normalised
    so neutrals stay neutral. Otherwise the pipeline fits a matrix with ridge
    (L2) regularisation and, when ``row_sum`` is true, a row-sum constraint.
    Ridge suppresses noise amplification at low light; 1e-4 is a typical
    starting value.
    """

    enabled: bool = True
    matrix: Matrix3x3 | None = None
    ridge: NonNegative = 1e-4
    row_sum: bool = True


class DenoiseConfig(ConfigModel):
    """Luma/chroma denoise strengths in linear opponent space.

    Work happens after demosaicing and before tone mapping, so strengths are
    in linear-light units. Independent luma and chroma strengths let colour
    noise be suppressed more aggressively than detail; zero strengths are an
    identity operation.
    """

    enabled: bool = True
    method: Literal["gaussian", "bilateral"] = "gaussian"
    luma_strength: NonNegative = 0.0
    chroma_strength: NonNegative = 0.0
    radius: PositiveFloat = 1.0


class SharpenConfig(ConfigModel):
    """Thresholded unsharp-mask settings.

    ``amount`` scales the high-pass contribution, ``radius`` is the blur scale
    in pixels, and ``threshold`` suppresses low-contrast noise. ``luma_only``
    avoids chroma fringing by sharpening luminance alone.
    """

    enabled: bool = True
    amount: NonNegative = 0.0
    radius: PositiveFloat = 1.0
    threshold: NonNegative = 0.0
    luma_only: bool = True


class ToneConfig(ConfigModel):
    """Final non-linear tone mapping and output colour space.

    ``exposure_compensation_ev`` shifts exposure before the transfer function;
    ``gamma`` overrides the standard sRGB OETF with a plain power curve when
    supplied. Tone mapping is the first non-linear step in the pipeline.
    """

    enabled: bool = True
    color_space: ColorSpace = "sRGB"
    exposure_compensation_ev: float = 0.0
    gamma: PositiveFloat | None = None


class OutputConfig(ConfigModel):
    """Display encoding of the rendered image.

    ``bit_depth`` is the container depth used when writing images (8 for SDR
    displays, 10 or 16 for analysis); it does not change the in-memory
    ``RenderedImage`` float range.
    """

    color_space: ColorSpace = "sRGB"
    bit_depth: Annotated[int, Field(ge=8, le=16)] = 8


class ISPConfig(ConfigModel):
    """A complete image-signal-processor preset.

    Stages run in the fixed order ``dpc, blc, lsc, awb, demosaic, ccm,
    denoise, sharpen, tone, output``. Each stage config carries an ``enabled``
    toggle. The defaults form a conventional baseline, not a guarantee of
    identity: auto white balance estimates and applies per-channel gains, and
    colour correction either applies a supplied matrix or is fitted by the
    pipeline when ``matrix`` is ``None``. Callers that need a bit-exact
    pass-through must disable those stages explicitly. Optional enhancement
    stages whose numeric strengths default to zero (``denoise``, ``sharpen``)
    are identity operations until parameters are supplied.
    """

    schema_version: SchemaVersion
    name: NonEmptyStr
    provenance: Provenance
    dpc: DPCConfig = Field(default_factory=DPCConfig)
    blc: BLCConfig = Field(default_factory=BLCConfig)
    lsc: LSCConfig = Field(default_factory=LSCConfig)
    awb: AWBConfig = Field(default_factory=AWBConfig)
    demosaic: DemosaicConfig = Field(default_factory=DemosaicConfig)
    ccm: CCMConfig = Field(default_factory=CCMConfig)
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    sharpen: SharpenConfig = Field(default_factory=SharpenConfig)
    tone: ToneConfig = Field(default_factory=ToneConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
