"""End-to-end sensor pipeline from spectral irradiance to a RAW frame.

A camera sensor is a photon-to-code converter with several physical limits and
imperfections in series. This module assembles the pieces from the CFA,
noise, defect, gain, and ADC layers into one deterministic acquisition and
returns a self-describing :class:`~camsim.core.frames.RawFrame`.

Physical model
--------------
For a pixel with pitch ``p``, effective aperture fraction
``eta = min(1, fill_factor * microlens_gain)``, and area ``A = p**2 * eta``,
the mean electron rate under an external spectral irradiance ``E(lambda)`` is

``rate_c = integral P_c(lambda) * E(lambda) * A * dlambda / (h c / lambda)``

with ``P_c = QE * T_IR * T_CFA,c`` the dimensionless projection from
:mod:`camsim.sensor.cfa`. When ``E`` is supplied in
``"photons/s/m^2/nm"`` the photon-energy division is omitted. ``dlambda`` is
the uniform wavelength-sample spacing in nanometres. The projection matrix is
dimensionless and excludes area, exposure, bandwidth, and photon energy, so
the cacheable sensor response is separate from the scene radiometry.

The signal chain, in the order implemented, is::

    spectral integration -> PRNU -> dark current + DSNU
    -> Poisson sampling + full-well clip -> optional blooming
    -> spectral crosstalk -> optional spatial crosstalk -> read noise + clip
    -> row noise (electrons, before conversion)
    -> conversion gain -> analog gain -> ADC -> digital gain

Row noise is deliberately added in the electron domain, before conversion
gain. Adding it after digital gain would scale it by the conversion and analog
gains, producing the wrong banding amplitude at high gain. The full-well clip
is a charge-domain limit and is independent of the ADC code clip, so a sensor
can be full-well limited or ADC limited and the frame metadata records which
mode was used.

Dual conversion gain
--------------------
Each readout mode carries its own conversion gain, full well, read noise, and
dark-current reference rate. High conversion gain (HCG) trades well capacity
for lower input-referred read noise and is used below the configured
``auto_hcg_threshold_e`` mean signal; LCG is used above it.

Binning
-------
Analog binning sums charge means per CFA plane before Poisson sampling and
scales the full well by ``factor**2``; one read noise is added to the binned
bucket. Digital binning runs the native-resolution chain through the ADC and
averages codes afterwards, so read noise is averaged but clipping has already
happened. Both preserve the CFA pattern: the channel planes are aggregated
independently and re-interleaved, never summed across colours.

Non-idealities
--------------
* **PRNU** is a zero-mean multiplicative responsivity deviation:
  ``S * (1 + prnu_pixel)``.
* **DSNU** is a zero-mean additive dark offset in electrons.
* **Hot pixels** multiply dark current by ``hot_pixel_dark_multiplier``; the
  mask is otherwise metadata for the ISP's defect-correction stage.
* **PDAF** sites scale the signal by ``qe_factor`` and are flagged, not
  treated as defects.
* **Spectral crosstalk** is a row-normalised 3x3 channel mixing matrix applied
  to the Bayer channel planes before mosaicing. Mono has one channel and skips
  it. RCCC has two channels, so a non-identity 3x3 matrix is rejected as
  incompatible; the preset ships an identity matrix.
* **Spatial crosstalk** is an optional normalised charge-diffusion kernel
  applied to each CFA channel plane after spectral crosstalk and before read
  noise. ``SensorConfig.spatial_crosstalk = None`` disables it; a non-identity
  kernel mixes neighbouring pixels of the same colour without changing total
  interior charge.
* **Blooming** redistributes a fraction of the charge above full well to the
  four nearest neighbours for a bounded number of iterations. It is off by
  default because pinned photodiodes suppress it well.

Determinism contract
--------------------
``Sensor(config, seed)`` seeds the frame stream. A per-call ``seed`` creates a
fresh ``numpy.random.Generator`` for that frame and does not advance the frame
stream, so call order cannot change a seeded frame. Without a per-call seed the
frame stream advances and the derived seed and capture index are recorded.
Fixed-pattern maps (DSNU, PRNU, dead/hot/stuck pixels, PDAF) are generated once
from ``config.defects.seed`` and reused; regenerating them per frame would turn
calibratable spatial noise into temporal noise. Bit-identical output is
promised within one software environment, not across floating-point
implementations.

External irradiance contract
----------------------------
The cube must already be on the sensor's pixel grid: shape ``(H, W, L)`` with
``L`` matching the preset wavelength grid, an optional ``pixel_pitch_um`` that
matches the native pitch, and a spatial shape that does not change after the
first frame. The sensor never resamples. A mismatch raises a message naming
the offending quantity; resample explicitly with
:meth:`SpectralIrradiance.resample_spatial` or
:meth:`SpectralIrradiance.resample_wavelengths`.

Real-world parameter ranges: pixel pitch 0.6-10 um; QE 0.1-0.9; full well
2-50 ke-; read noise 0.3-10 e- rms; dark current 0.01-100 e-/s at 25 C;
exposure 10 us-10 s; analog gain 0-30 dB; ADC 8-16 bits; temperature
-20 to 70 C. Failure artefacts guarded here include a percent-valued PRNU,
dark current added without the temperature factor, crosstalk matrices whose
rows do not sum to one (colour casts that white balance cannot remove),
binning factors that do not divide the array, and implicit unit or grid
conversion that would produce plausible-looking but quantitatively wrong RAW
data.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import convolve2d  # type: ignore[import-untyped]

from camsim.config.schema import GainMode, GainModeConfig, SensorConfig
from camsim.core.frames import RawFrame
from camsim.core.spectral import SpectralIrradiance, WavelengthGrid
from camsim.core.units import photon_energy_j
from camsim.data import data_path
from camsim.sensor.adc import adc_quantize
from camsim.sensor.binning import BinningPlan, binning_plan, mean_blocks, reduce_mask, sum_blocks
from camsim.sensor.cfa import cfa_masks, mosaic_from_channels, spectral_projection
from camsim.sensor.defects import DefectMasks, defect_masks, pdaf_mask
from camsim.sensor.gain import apply_analog_gain, clamp_full_well, electrons_to_voltage
from camsim.sensor.noise import (
    FPNConfig,
    FPPMaps,
    dark_current_e_per_s,
    fixed_pattern_maps,
    poisson_electrons,
    row_noise,
)

__all__ = [
    "Sensor",
    "apply_blooming",
    "apply_spatial_crosstalk",
    "apply_spectral_crosstalk",
]

ModeRequest: TypeAlias = Literal["auto", "LCG", "HCG"]
"""Accepted values of the ``mode`` argument to :meth:`Sensor.expose`."""

# The sensor consumes these two per-nanometre density units. SpectralIrradiance
# validates the value, but naming the set here keeps the unit branch explicit.
_W_PER_AREA_NM: Final[str] = "W/m^2/nm"

# Canonical channel orders for the crosstalk matrix. A mapping is re-ordered to
# one of these sets so a caller cannot silently pair matrix row 0 with, say,
# blue by inserting the keys in another order.
_CHANNEL_ORDERS: Final[tuple[tuple[str, ...], ...]] = (
    ("R", "G", "B"),
    ("R", "C"),
    ("M",),
)

# Wavelength grids must agree to much better than one thousandth of a
# nanometre; the tolerance only absorbs float round-off from np.arange.
_WAVELENGTH_ATOL_NM: Final[float] = 1e-6

# One DN of matrix/row-sum tolerance absorbs decimal rounding in preset JSON.
_ROW_SUM_ATOL: Final[float] = 1e-6

# The repository root supports source-checkout runs whose preset still carries
# an older repo-relative CSV path. The packaged camsim.data directory is the
# authoritative installed location. No path is guessed: a candidate must exist
# before it is used.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


@dataclass(eq=False)
class _SensorPatterns:
    """Fixed-pattern maps generated once for one native sensor shape."""

    fpn: FPPMaps
    defects: DefectMasks
    pdaf: NDArray[np.bool_]
    defect_union: NDArray[np.bool_]


class Sensor:
    """A deterministic sensor that converts spectral irradiance into RAW frames.

    Parameters
    ----------
    config:
        Validated :class:`~camsim.config.schema.SensorConfig`. Spectral CSVs
        referenced by ``config.spectral`` are read once at construction. A
        relative path is resolved against the working directory, then the
        repository root (for source-checkout compatibility), and finally the
        bundled ``camsim.data`` package, which is where the shipped preset's
        ``spectra/*.csv`` files live after installation; no implicit
        resampling is performed.
    seed:
        Non-negative seed of the frame stream used when ``expose`` is called
        without an explicit per-call seed.

    Attributes
    ----------
    config:
        The configuration object supplied to the constructor.
    pattern_seed:
        The single seed used for DSNU/PRNU, defect, and PDAF maps, taken from
        ``config.defects.seed``. It is independent of the frame stream.

    The first successful exposure fixes the native sensor grid (the cube's
    spatial shape). A later cube with another shape is rejected, because a
    real part has one pixel array and implicit resampling would silently
    change the radiometry. Fixed-pattern maps are generated lazily for that
    shape on first use and cached for the life of the sensor.
    """

    def __init__(self, config: SensorConfig, seed: int = 0) -> None:
        if not isinstance(config, SensorConfig):
            raise ValueError(f"config must be a SensorConfig; got {type(config).__name__}")
        self.config = config
        self.pattern_seed = int(config.defects.seed)
        self._stream_seed = _non_negative_int(seed, "seed")
        self._frame_rng = np.random.default_rng(self._stream_seed)
        self._frame_index = 0
        self._native_shape: tuple[int, int] | None = None
        self._patterns: _SensorPatterns | None = None
        self._wavelength_grid = _sensor_wavelength_grid(config)
        self._projection = spectral_projection(
            _resolve_spectral_path(config.spectral.qe_csv),
            _resolve_spectral_path(config.spectral.cfa_csv),
            _resolve_spectral_path(config.spectral.ir_cut_csv),
            self._wavelength_grid,
            config.cfa,
        )
        self._channels = tuple(cfa_masks(config.cfa, (2, 2)))
        self._fpn_config = FPNConfig(
            dsnu_e=config.noise.dsnu_e,
            prnu_fraction=config.noise.prnu_fraction,
            row_noise_e_rms=config.noise.row_noise_e_rms,
        )
        self._config_hash = _sensor_config_hash(config)

    @property
    def frame_shape(self) -> tuple[int, int] | None:
        """Native ``(height, width)`` fixed by the first exposure, or ``None``."""
        return self._native_shape

    def reseed(self, seed: int) -> None:
        """Reset the frame stream and capture index.

        Fixed-pattern maps are unaffected because they derive from
        ``config.defects.seed``, not from the frame stream.
        """
        self._stream_seed = _non_negative_int(seed, "seed")
        self._frame_rng = np.random.default_rng(self._stream_seed)
        self._frame_index = 0

    def expose(
        self,
        irradiance: SpectralIrradiance,
        exposure_s: float,
        mode: str = "auto",
        analog_gain_db: float | None = None,
        digital_gain: float = 1.0,
        temperature_c: float | None = None,
        seed: int | None = None,
    ) -> RawFrame:
        """Integrate one exposure and return a RAW frame.

        Parameters
        ----------
        irradiance:
            Spectral irradiance at the sensor plane, before the sensor's own
            IR-cut and CFA filters, on the sensor's pixel grid. Both
            ``"W/m^2/nm"`` and ``"photons/s/m^2/nm"`` are accepted.
        exposure_s:
            Integration time in seconds, within the preset's exposure limits.
        mode:
            ``"auto"`` chooses HCG below ``auto_hcg_threshold_e`` and LCG
            otherwise; ``"LCG"`` and ``"HCG"`` force a readout mode.
        analog_gain_db:
            Programmable analog gain in decibels. ``None`` uses the preset
            default; the value must lie inside the configured limits.
        digital_gain:
            Additional digital multiplier composed with the ADC preset's
            ``digital_gain``. It is applied to the signal code before the
            black level is re-added, so it cannot lift the pedestal.
        temperature_c:
            Sensor temperature override in degrees Celsius. ``None`` uses the
            preset's default. Dark current follows the doubling model.
        seed:
            Per-call frame seed. Supplying it does not advance the sensor's
            frame stream, so the pixel data are independent of call order;
            ``frame_index`` still records the position in the capture
            sequence.

        Returns
        -------
        RawFrame
            ``uint16`` codes including the black-level pedestal, with the CFA
            pattern, gains, mode, temperature, frame seed/index, output pixel
            pitch, defect/PDAF masks, and a stable configuration hash.

        Raises
        ------
        ValueError
            If any input is out of range, the cube is not on the sensor grid
            (shape, pitch, or wavelength grid), the binning factor does not
            divide the shape, the crosstalk matrix is incompatible with the
            CFA, or a noise/gain parameter is non-finite.
        """
        if not isinstance(irradiance, SpectralIrradiance):
            raise ValueError(
                f"irradiance must be a SpectralIrradiance; got {type(irradiance).__name__}"
            )
        self._validate_external_cube(irradiance)
        exposure = _validated_exposure(exposure_s, self.config)
        requested_mode = _validated_mode(mode)
        gain_db = self._resolve_analog_gain(analog_gain_db)
        applied_digital_gain = self._resolve_digital_gain(digital_gain)
        temperature = self._resolve_temperature(temperature_c)

        height = int(irradiance.data.shape[0])
        width = int(irradiance.data.shape[1])
        shape = (height, width)
        plan = binning_plan(self.config.binning, shape)
        self._lock_shape(shape)
        patterns = self._patterns_for(shape)
        mean_electrons = self._mean_electrons(irradiance, exposure, patterns)
        selected_mode = self._resolve_mode(requested_mode, mean_electrons)
        mode_config = self.config.modes[selected_mode]

        full_well_e = float(mode_config.full_well_e)
        if plan.mode == "analog":
            full_well_e *= float(plan.factor**2)

        frame_seed, rng = self._next_frame_generator(seed)
        charges = self._sample_charges(
            mean_electrons,
            exposure,
            mode_config,
            full_well_e,
            temperature,
            patterns,
            rng,
            plan,
        )
        charges = self._apply_spectral_crosstalk(charges)
        charges = self._apply_spatial_crosstalk(charges)
        charges = self._add_read_and_row_noise(charges, full_well_e, mode_config, rng)
        dn_planes = self._quantize(charges, mode_config, gain_db, applied_digital_gain, plan)
        data = cast(
            NDArray[np.uint16],
            mosaic_from_channels(
                {channel: dn_planes[index] for index, channel in enumerate(self._channels)},
                self.config.cfa,
            ),
        )
        defect = patterns.defect_union
        pdaf = patterns.pdaf
        if plan.factor > 1:
            defect = reduce_mask(defect, plan.factor)
            pdaf = reduce_mask(pdaf, plan.factor)

        frame = RawFrame(
            data=data,
            cfa=self.config.cfa,
            bit_depth=int(self.config.adc.bits),
            black_level_dn=float(self.config.adc.black_level_dn),
            exposure_s=exposure,
            analog_gain_db=gain_db,
            digital_gain=applied_digital_gain,
            mode=selected_mode,
            temperature_c=temperature,
            seed=frame_seed,
            frame_index=self._frame_index,
            pixel_pitch_um=float(self.config.pixel.pitch_um) * plan.factor,
            defect_mask=defect,
            pdaf_mask=pdaf,
            sensor_config_hash=self._config_hash,
        )
        self._frame_index += 1
        return frame

    # ------------------------------------------------------------------
    # External cube validation and grid locking
    # ------------------------------------------------------------------

    def _validate_external_cube(self, irradiance: SpectralIrradiance) -> None:
        """Reject cubes that are not already on this sensor's pixel grid."""
        height, width, spectral_length = irradiance.data.shape
        if height <= 0 or width <= 0:
            raise ValueError(
                f"irradiance spatial shape {(height, width)} does not match the sensor grid; "
                "the cube must contain at least one pixel"
            )
        sensor_nm = self._wavelength_grid.nm
        cube_nm = irradiance.wavelengths_nm.nm
        if spectral_length != sensor_nm.size or cube_nm.shape != sensor_nm.shape:
            raise ValueError(
                f"irradiance wavelength grid has {cube_nm.size} samples but the sensor grid "
                f"expects {sensor_nm.size}; resample explicitly with "
                "SpectralIrradiance.resample_wavelengths"
            )
        if not np.allclose(cube_nm, sensor_nm, rtol=0.0, atol=_WAVELENGTH_ATOL_NM):
            raise ValueError(
                "irradiance wavelength grid does not match the sensor's wavelength grid "
                f"({cube_nm[0]:g}-{cube_nm[-1]:g} nm vs "
                f"{sensor_nm[0]:g}-{sensor_nm[-1]:g} nm); resample explicitly with "
                "SpectralIrradiance.resample_wavelengths"
            )
        pitch = irradiance.pixel_pitch_um
        expected_pitch = float(self.config.pixel.pitch_um)
        if pitch is not None and not math.isclose(
            pitch, expected_pitch, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(
                f"irradiance pixel_pitch_um ({pitch:g} um) does not match the sensor grid "
                f"pitch ({expected_pitch:g} um); resample explicitly with "
                "SpectralIrradiance.resample_spatial"
            )

    def _lock_shape(self, shape: tuple[int, int]) -> None:
        """Fix the native sensor grid on first use and enforce it afterwards."""
        if self._native_shape is None:
            self._native_shape = shape
            return
        if shape != self._native_shape:
            raise ValueError(
                f"irradiance spatial shape {shape} does not match the sensor grid "
                f"{self._native_shape}; resample explicitly with "
                "SpectralIrradiance.resample_spatial"
            )

    # ------------------------------------------------------------------
    # Fixed-pattern maps
    # ------------------------------------------------------------------

    def _patterns_for(self, shape: tuple[int, int]) -> _SensorPatterns:
        """Return the cached fixed-pattern maps for ``shape``."""
        if self._patterns is not None and self._native_shape == shape:
            return self._patterns
        fpn = fixed_pattern_maps(self._fpn_config, shape, self.pattern_seed)
        defects = defect_masks(self.config.defects, shape, seed=self.pattern_seed)
        pdaf = pdaf_mask(
            self.config.pdaf,
            shape,
            seed=self.pattern_seed,
            cfa=self.config.cfa,
        )
        self._patterns = _SensorPatterns(
            fpn=fpn,
            defects=defects,
            pdaf=pdaf,
            defect_union=defects.dead | defects.hot | defects.stuck,
        )
        return self._patterns

    # ------------------------------------------------------------------
    # Spectral integration and mode selection
    # ------------------------------------------------------------------

    def _mean_electrons(
        self,
        irradiance: SpectralIrradiance,
        exposure_s: float,
        patterns: _SensorPatterns,
    ) -> NDArray[np.float64]:
        """Integrate the cube through each CFA channel, returning ``(C, H, W)``.

        Each plane is the image that one CFA filter would see at every pixel;
        the mosaic step later selects the plane named by the pixel's tile
        position. Keeping the channel planes explicit is what makes spectral
        crosstalk and per-CFA-plane binning well defined. Selection uses the
        2x2 tile through :func:`mosaic_from_channels`, never a per-pixel Python
        loop, so a large frame costs a vectorised integration per channel.
        """
        cube = np.asarray(irradiance.data, dtype=np.float64)
        if irradiance.unit == _W_PER_AREA_NM:
            # E is W/m2/nm; dividing by photon energy gives photons/s/m2/nm.
            photons = cube / photon_energy_j(self._wavelength_grid.nm)
        elif irradiance.unit == "photons/s/m^2/nm":
            photons = cube
        else:  # pragma: no cover - SpectralIrradiance already rejects this
            raise ValueError(
                f"irradiance unit must be 'W/m^2/nm' or 'photons/s/m^2/nm'; got {irradiance.unit!r}"
            )
        integral = np.einsum("hwl,cl->chw", photons, self._projection)
        signal = cast(NDArray[np.float64], integral) * (
            self._aperture_area_m2() * self._wavelength_step_nm() * exposure_s
        )
        if bool(patterns.pdaf.any()):
            # PDAF sites lose part of their aperture; the mask is metadata for
            # the ISP and the qe_factor is applied only to the signal.
            pdaf_signal = signal[:, patterns.pdaf] * float(self.config.pdaf.qe_factor)
            signal[:, patterns.pdaf] = pdaf_signal
        return np.asarray(signal, dtype=np.float64)

    def _aperture_area_m2(self) -> float:
        """Effective light-collecting area of one native pixel in m^2."""
        pitch_m = float(self.config.pixel.pitch_um) * 1e-6
        eta = min(
            1.0,
            float(self.config.pixel.fill_factor) * float(self.config.pixel.microlens_gain),
        )
        return pitch_m**2 * eta

    def _wavelength_step_nm(self) -> float:
        """Uniform per-sample wavelength spacing in nanometres."""
        return float(self._wavelength_grid.nm[1] - self._wavelength_grid.nm[0])

    def _resolve_mode(
        self,
        requested: ModeRequest,
        mean_electrons: NDArray[np.float64],
    ) -> GainMode:
        """Resolve ``"auto"`` to HCG or LCG from the pre-noise mean signal.

        The statistic is the mean over the native mosaic, so it describes the
        signal one pixel would collect before binning. Binning increases the
        charge per output pixel but not the per-pixel read-noise regime that
        the HCG/LCG choice is meant to follow.
        """
        if requested != "auto":
            return requested
        mosaic = mosaic_from_channels(
            {channel: mean_electrons[index] for index, channel in enumerate(self._channels)},
            self.config.cfa,
        )
        mean_signal = float(np.mean(mosaic))
        return "HCG" if mean_signal < float(self.config.auto_hcg_threshold_e) else "LCG"

    # ------------------------------------------------------------------
    # Electron chain
    # ------------------------------------------------------------------

    def _sample_charges(
        self,
        mean_electrons: NDArray[np.float64],
        exposure_s: float,
        mode_config: GainModeConfig,
        full_well_e: float,
        temperature_c: float,
        patterns: _SensorPatterns,
        rng: np.random.Generator,
        plan: BinningPlan,
    ) -> NDArray[np.float64]:
        """Apply PRNU, dark current, DSNU, Poisson sampling, and blooming."""
        dark_rate = dark_current_e_per_s(
            mode_config.dark_current_e_per_s,
            temperature_c,
            self.config.temperature,
        )
        hot_multiplier = np.where(
            patterns.defects.hot,
            float(self.config.noise.hot_pixel_dark_multiplier),
            1.0,
        )
        dark_mean = dark_rate * exposure_s * hot_multiplier
        planes: list[NDArray[np.float64]] = []
        for channel_index in range(len(self._channels)):
            prnu = patterns.fpn.prnu_fraction
            mean_charge = mean_electrons[channel_index] * (1.0 + prnu)
            mean_charge = mean_charge + dark_mean + patterns.fpn.dsnu_e
            if plan.mode == "analog" and plan.factor > 1:
                # Charge is summed on the sense node before sampling. The
                # caller has already scaled full well by factor**2.
                mean_charge = sum_blocks(mean_charge, plan.factor)
            # DSNU is zero mean, so a dark pixel can have a slightly negative
            # expected charge; a Poisson mean cannot, so clamp before sampling.
            mean_charge = np.maximum(mean_charge, 0.0)
            sampled = poisson_electrons(mean_charge, rng).astype(np.float64)
            if self.config.blooming.enabled:
                charge = apply_blooming(
                    sampled,
                    full_well_e,
                    float(self.config.blooming.fraction),
                    int(self.config.blooming.iterations),
                )
            else:
                charge = clamp_full_well(sampled, full_well_e)
            planes.append(charge)
        return np.stack(planes, axis=0)

    def _apply_spectral_crosstalk(
        self,
        charges: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Mix per-channel charge with the configured spectral crosstalk."""
        channel_count = len(self._channels)
        if channel_count == 1:
            # Mono has no neighbouring colour filter to leak into.
            return charges
        matrix = np.asarray(self.config.spectral.spectral_crosstalk, dtype=np.float64)
        if np.allclose(matrix, np.eye(matrix.shape[0]), rtol=0.0, atol=_ROW_SUM_ATOL):
            # An identity matrix is no crosstalk regardless of its dimension, so
            # an RCCC preset can reuse the 3x3 identity used by Bayer presets.
            return charges
        if matrix.shape != (channel_count, channel_count):
            raise ValueError(
                f"spectral_crosstalk is a {matrix.shape[0]}x{matrix.shape[1]} matrix but "
                f"{self.config.cfa.value} has {channel_count} CFA channels; configure an "
                "identity matrix or a crosstalk model with one row per channel"
            )
        mixed = apply_spectral_crosstalk(
            {channel: charges[index] for index, channel in enumerate(self._channels)},
            matrix,
        )
        return np.stack([mixed[channel] for channel in self._channels], axis=0)

    def _apply_spatial_crosstalk(
        self,
        charges: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convolve each CFA channel plane with the configured diffusion kernel.

        The kernel mixes neighbouring pixels of the same colour, so it is
        applied to the explicit channel planes rather than to the mosaicked
        frame: convolving a Bayer mosaic directly would average different
        colour filters together and create a colour error. Applying it after
        spectral crosstalk but before read noise keeps both mixing processes in
        the charge domain, where physical crosstalk happens.
        """
        kernel = self.config.spatial_crosstalk
        if kernel is None:
            return charges
        stencil = np.asarray(kernel, dtype=np.float64)
        planes = [
            apply_spatial_crosstalk(charges[index], stencil) for index in range(charges.shape[0])
        ]
        return np.stack(planes, axis=0)

    def _add_read_and_row_noise(
        self,
        charges: NDArray[np.float64],
        full_well_e: float,
        mode_config: GainModeConfig,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Add input-referred read noise, clip, then add electron-domain row noise."""
        sigma = float(mode_config.read_noise_e_rms)
        if sigma > 0.0:
            charges = charges + rng.normal(0.0, sigma, size=charges.shape)
        charges = np.clip(charges, 0.0, full_well_e)
        row_sigma = float(self.config.noise.row_noise_e_rms)
        if row_sigma > 0.0:
            # One common-mode offset per readout row, shared by every channel.
            rows = row_noise(row_sigma, charges.shape[1], rng)
            charges = charges + rows
        return np.asarray(charges, dtype=np.float64)

    # ------------------------------------------------------------------
    # Conversion, ADC, and binning
    # ------------------------------------------------------------------

    def _quantize(
        self,
        charges: NDArray[np.float64],
        mode_config: GainModeConfig,
        gain_db: float,
        digital_gain: float,
        plan: BinningPlan,
    ) -> list[NDArray[np.uint16]]:
        """Convert charge to voltage, amplify, and quantize per channel plane."""
        configuration = self.config.adc
        planes: list[NDArray[np.uint16]] = []
        for channel_index in range(len(self._channels)):
            voltage = electrons_to_voltage(
                charges[channel_index], mode_config.conversion_gain_uv_per_e
            )
            voltage = apply_analog_gain(voltage, gain_db)
            codes = adc_quantize(
                voltage,
                bits=int(configuration.bits),
                full_scale_uv=float(configuration.full_scale_uv),
                black_level_dn=int(configuration.black_level_dn),
                digital_gain=digital_gain,
            )
            if plan.mode == "digital" and plan.factor > 1:
                # Average after quantization; read noise and ADC noise are
                # reduced by the average but clipping has already happened.
                averaged = mean_blocks(codes, plan.factor)
                codes = _round_codes(averaged, int(configuration.bits))
            planes.append(codes)
        return planes

    # ------------------------------------------------------------------
    # Gains, temperature, and frame stream
    # ------------------------------------------------------------------

    def _resolve_analog_gain(self, analog_gain_db: float | None) -> float:
        """Return a validated analog gain in dB."""
        default = float(self.config.analog_gain.default_db)
        gain = (
            default if analog_gain_db is None else _finite_float(analog_gain_db, "analog_gain_db")
        )
        minimum = float(self.config.analog_gain.min_db)
        maximum = float(self.config.analog_gain.max_db)
        if gain < minimum - 1e-9 or gain > maximum + 1e-9:
            raise ValueError(
                f"analog_gain_db ({gain:g} dB) must lie within "
                f"[{minimum:g}, {maximum:g}] dB from the sensor configuration"
            )
        return max(gain, 0.0)

    def _resolve_digital_gain(self, digital_gain: float) -> float:
        """Compose the per-call digital gain with the ADC preset's setting."""
        call_gain = _positive_float(digital_gain, "digital_gain")
        effective = call_gain * float(self.config.adc.digital_gain)
        if not math.isfinite(effective):
            raise ValueError(
                "digital_gain multiplied by adc.digital_gain is not finite; "
                "check the configured gains"
            )
        return effective

    def _resolve_temperature(self, temperature_c: float | None) -> float:
        """Return the per-call temperature override or the preset default."""
        if temperature_c is None:
            return float(self.config.temperature.default_c)
        return _finite_float(temperature_c, "temperature_c (temperature)")

    def _next_frame_generator(self, seed: int | None) -> tuple[int, np.random.Generator]:
        """Return the frame seed and a generator for this exposure.

        A per-call seed creates an independent generator and leaves the frame
        stream untouched. Without one, the stream advances so successive
        captures are independent.
        """
        if seed is None:
            derived = int(self._frame_rng.integers(0, np.iinfo(np.int64).max))
            return derived, np.random.default_rng(derived)
        resolved = _non_negative_int(seed, "seed")
        return resolved, np.random.default_rng(resolved)


# ---------------------------------------------------------------------------
# Crosstalk and blooming
# ---------------------------------------------------------------------------


@overload
def apply_spectral_crosstalk(
    channels: Mapping[str, ArrayLike],
    matrix: ArrayLike,
) -> dict[str, NDArray[np.float64]]: ...


@overload
def apply_spectral_crosstalk(
    channels: ArrayLike,
    matrix: ArrayLike,
) -> NDArray[np.float64]: ...


def apply_spectral_crosstalk(
    channels: Mapping[str, ArrayLike] | ArrayLike,
    matrix: ArrayLike,
) -> dict[str, NDArray[np.float64]] | NDArray[np.float64]:
    """Mix per-colour channel planes with a row-normalised crosstalk matrix.

    Parameters
    ----------
    channels:
        Either a mapping of CFA colour to a 2-D plane (``R,G,B`` for Bayer,
        ``R,C`` for RCCC, ``M`` for mono) or an array whose first axis is the
        channel count. Mapping keys may arrive in any order; the canonical
        order ``R,G,B`` (or ``R,C``) is restored so matrix row ``i`` always
        refers to channel ``i``.
    matrix:
        Square, finite, row-normalised mixing matrix. Each row sums to one so a
        neutral input stays neutral. Total channel charge is conserved only
        when the matrix is also column-normalised (doubly stochastic); a
        row-normalised matrix alone can change the channel sum. The diagonal
        is the same-channel fraction; off-diagonal entries model leakage from
        neighbouring filters.

    Returns
    -------
    A mapping (for mapping input) or an array (for array input) of float64
    planes with the same shape as the input. Output channel ``i`` is
    ``sum_j matrix[i, j] * input[j]``.

    Raises
    ------
    ValueError
        If the matrix is not square or finite, any row does not sum to one,
        the channel count does not match the matrix dimension, or a channel
        plane is missing, non-2-D, or non-finite.

    Notes
    -----
    Row normalisation is checked here as well as in the preset schema, because
    this function is public and a caller can construct a matrix directly. An
    unnormalised matrix would scale total charge and create a colour cast that
    no white balance can separate from the illuminant.
    """
    resolved_matrix = _validated_crosstalk_matrix(matrix)
    if isinstance(channels, Mapping):
        order = _canonical_channel_order(tuple(channels))
        planes: list[NDArray[np.float64]] = []
        shapes: set[tuple[int, int]] = set()
        for channel in order:
            plane = _finite_2d(channels[channel], f"channel {channel!r}")
            planes.append(plane)
            shapes.add(plane.shape)
        if len(shapes) != 1:
            raise ValueError(
                "crosstalk channel planes must have the same shape; got "
                + ", ".join(f"{channel!r}: {planes[i].shape}" for i, channel in enumerate(order))
            )
        if resolved_matrix.shape[0] != len(order):
            raise ValueError(
                f"crosstalk matrix has {resolved_matrix.shape[0]} channels but the "
                f"input has {len(order)} channel(s) ({', '.join(order)}); configure a "
                "compatible matrix"
            )
        stack = np.stack(planes, axis=0)
        mixed = np.einsum("ij,jhw->ihw", resolved_matrix, stack)
        return {
            channel: np.asarray(mixed[index], dtype=np.float64)
            for index, channel in enumerate(order)
        }
    array = np.asarray(channels, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("channels must be a channel-first array with at least one axis")
    if not np.all(np.isfinite(array)):
        raise ValueError("channel values must be finite (no NaN or infinity)")
    if array.shape[0] != resolved_matrix.shape[0]:
        raise ValueError(
            f"crosstalk matrix has {resolved_matrix.shape[0]} channels but the input "
            f"has {array.shape[0]} channel(s)"
        )
    return np.asarray(
        np.einsum("ij,j...->i...", resolved_matrix, array),
        dtype=np.float64,
    )


def apply_spatial_crosstalk(
    mosaic: ArrayLike,
    kernel: ArrayLike,
) -> NDArray[np.float64]:
    """Convolve a mosaic with a normalised spatial charge-diffusion kernel.

    Optical and electrical crosstalk moves charge between neighbouring pixels
    with the same CFA periodicity. The kernel should be non-negative and sum
    to one so total charge is conserved; a kernel that does not is rejected
    rather than rescaled, because silently normalising it would hide a
    calibration error in the caller's data.

    Parameters
    ----------
    mosaic:
        Finite 2-D array in linear sensor units (electrons or DN).
    kernel:
        Finite, non-negative 2-D kernel whose sum is one. Odd-sized, centred
        kernels are typical (for example a normalised 3x3 diffusion stencil).

    Returns
    -------
    A float64 array the same shape as ``mosaic``. Boundaries use mirror
    reflection, which keeps the result finite and avoids treating the frame
    edge as a black surround. Charge that a wide kernel would carry beyond the
    edge can still change the frame total, so a spatial-crosstalk calibration
    should use an interior region for gain checks.

    Notes
    -----
    :meth:`Sensor` calls this function once per CFA channel plane when
    ``SensorConfig.spatial_crosstalk`` is configured. It remains public because
    a caller may have a measured kernel outside the preset. Spectral crosstalk
    is a colour-mixing matrix and belongs before mosaicing; spatial crosstalk is
    a same-colour convolution and therefore also belongs on the channel planes,
    not on the mosaicked frame.
    """
    image = _finite_2d(mosaic, "mosaic")
    stencil = _finite_2d(kernel, "kernel")
    if np.any(stencil < 0.0):
        raise ValueError("spatial crosstalk kernel must be non-negative")
    total = float(stencil.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_ROW_SUM_ATOL):
        raise ValueError(
            f"spatial crosstalk kernel must sum to 1.0 (got {total:g}); "
            "an unnormalised kernel changes total charge"
        )
    return np.asarray(convolve2d(image, stencil, mode="same", boundary="symm"), dtype=np.float64)


def apply_blooming(
    electrons: ArrayLike,
    full_well_e: float,
    fraction: float,
    iterations: int,
) -> NDArray[np.float64]:
    """Clip charge to full well and redistribute a fraction of the overflow.

    Parameters
    ----------
    electrons:
        Finite 2-D charge in electrons, **before** the full-well clip. If the
        array has already been clipped, there is no overflow and the result is
        just a clip; this is why the sensor passes the raw Poisson sample.
    full_well_e:
        Charge-domain saturation in electrons, finite and non-negative.
    fraction:
        Fraction of the overflow passed to the four edge-sharing neighbours
        each iteration, in ``[0, 1]``. The remainder is assumed to drain away
        or recombine, which is why blooming is a lossy process.
    iterations:
        Non-negative number of diffusion passes. Zero disables spreading and
        returns only the full-well clip.

    Returns
    -------
    A float64 array in ``[0, full_well_e]`` with the same shape. Every
    overflowed pixel distributes ``fraction / 4`` of its excess to each
    orthogonal neighbour; diagonal neighbours and wrap-around are not used.
    Charge that would leave the array boundary is lost, and after the final
    iteration any remaining overflow is clipped. No NaN can be produced.

    Notes
    -----
    A pinned photodiode suppresses blooming well, so the preset ships it off.
    Enabling it matters for CCD-like or older parts with high exposure, where
    a saturated highlight bleeds into a cross-shaped halo. The function is
    deterministic: no random stream is involved.
    """
    charge = _finite_2d(electrons, "electrons")
    well = _non_negative_float(full_well_e, "full_well_e")
    share_fraction = _finite_float(fraction, "fraction")
    if not 0.0 <= share_fraction <= 1.0:
        raise ValueError(f"fraction must lie within [0, 1]; got {share_fraction:g}")
    passes = _non_negative_int(iterations, "iterations")

    result = np.clip(charge, 0.0, well)
    # The overflow must be captured before the clip; otherwise the first
    # redistribution pass would see only clipped values and spread nothing.
    excess = np.maximum(charge - well, 0.0)
    share = 0.25 * share_fraction
    for _ in range(passes):
        if not bool(np.any(excess > 0.0)):
            break
        spread = np.zeros_like(result)
        spread[1:, :] += excess[:-1, :] * share
        spread[:-1, :] += excess[1:, :] * share
        spread[:, 1:] += excess[:, :-1] * share
        spread[:, :-1] += excess[:, 1:] * share
        result = result + spread
        excess = np.maximum(result - well, 0.0)
        result = np.clip(result, 0.0, well)
    return np.asarray(np.clip(result, 0.0, well), dtype=np.float64)


# ---------------------------------------------------------------------------
# Validation and path helpers
# ---------------------------------------------------------------------------


def _sensor_wavelength_grid(config: SensorConfig) -> WavelengthGrid:
    """Build and validate the uniform wavelength grid from the preset."""
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
    wavelengths = start + step * np.arange(count + 1, dtype=np.float64)
    if not math.isclose(float(wavelengths[-1]), stop, rel_tol=0.0, abs_tol=_WAVELENGTH_ATOL_NM):
        raise ValueError(
            f"sensor wavelength grid ends at {wavelengths[-1]:g} nm but stop is {stop:g} nm"
        )
    return WavelengthGrid(wavelengths)


def _resolve_spectral_path(path: str) -> Path:
    """Resolve a preset CSV path against the working tree, then package data.

    Order matters for overrides: an absolute path or a file that exists relative
    to the current working directory wins, so a caller can replace a shipped
    curve without editing the preset. The repository root keeps source-checkout
    presets with older ``data/spectra`` paths working. Finally the path is
    looked up inside the installed ``camsim.data`` package, which is what makes
    a wheel self-contained. If none exists, the original candidate is returned
    so the reader raises a message naming the path the caller asked for.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    fallback = _REPO_ROOT / candidate
    if fallback.exists():
        return fallback
    try:
        return data_path(candidate)
    except FileNotFoundError:
        return candidate


def _sensor_config_hash(config: SensorConfig) -> str:
    """Return a stable SHA-256 of the canonical configuration JSON."""
    payload: dict[str, Any] = config.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validated_exposure(value: object, config: SensorConfig) -> float:
    """Return an exposure inside the preset's allowed interval."""
    exposure = _positive_float(value, "exposure_s (exposure)")
    minimum = float(config.exposure.min_s)
    maximum = float(config.exposure.max_s)
    if exposure < minimum - 1e-12 or exposure > maximum + 1e-12:
        raise ValueError(
            f"exposure_s ({exposure:g} s) must lie within the configured exposure "
            f"limits [{minimum:g}, {maximum:g}] s"
        )
    return exposure


def _validated_mode(value: object) -> ModeRequest:
    """Return ``value`` narrowed to an accepted readout-mode request."""
    if value not in ("auto", "LCG", "HCG"):
        raise ValueError(f"mode must be 'auto', 'LCG', or 'HCG'; got {value!r}")
    return value


def _validated_crosstalk_matrix(matrix: ArrayLike) -> NDArray[np.float64]:
    """Return a square, finite, row-normalised crosstalk matrix."""
    try:
        resolved = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"crosstalk matrix must be a 2-D numeric array; {exc}") from exc
    if resolved.ndim != 2 or resolved.shape[0] != resolved.shape[1]:
        raise ValueError(f"crosstalk matrix must be square; got shape {resolved.shape}")
    if resolved.shape[0] == 0:
        raise ValueError("crosstalk matrix must contain at least one channel")
    if not np.all(np.isfinite(resolved)):
        raise ValueError("crosstalk matrix values must be finite")
    row_sums = resolved.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=_ROW_SUM_ATOL):
        offending = int(np.argmax(np.abs(row_sums - 1.0)))
        raise ValueError(
            f"crosstalk matrix row {offending} sums to {row_sums[offending]:g}, not 1.0; "
            "row-normalised matrices preserve neutral-axis energy"
        )
    return resolved


def _canonical_channel_order(keys: tuple[str, ...]) -> tuple[str, ...]:
    """Return the canonical colour order matching ``keys`` or raise."""
    for order in _CHANNEL_ORDERS:
        if len(keys) == len(order) and set(keys) == set(order):
            return order
    allowed = "; ".join(", ".join(order) for order in _CHANNEL_ORDERS)
    raise ValueError(
        f"channel keys must be one of {allowed}; got {sorted(keys)}. The matrix row "
        "order must be unambiguous"
    )


def _round_codes(values: NDArray[np.float64], bits: int) -> NDArray[np.uint16]:
    """Round averaged codes and clamp them to the ADC's legal range."""
    maximum = (1 << bits) - 1
    return np.clip(np.rint(values), 0.0, float(maximum)).astype(np.uint16)


def _finite_2d(values: ArrayLike, field: str) -> NDArray[np.float64]:
    """Return a finite, non-empty 2-D float64 array."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a 2-D numeric array; {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"{field} must be a 2-D (H, W) array; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{field} must contain at least one pixel")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be finite (no NaN or infinity)")
    return array


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


def _non_negative_int(value: object, field: str) -> int:
    """Return ``value`` as a non-negative Python integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be a non-negative integer; got {result}")
    return result
