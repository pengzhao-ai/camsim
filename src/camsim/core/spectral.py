"""Validated spectral irradiance cubes on an explicit wavelength grid.

A :class:`SpectralIrradiance` is the radiometric boundary between the scene and
the sensor. It carries a spatial cube of shape ``(H, W, L)`` plus the ``L``
wavelength samples in nanometres and an explicit unit, either per-nanometre
radiant power (``"W/m^2/nm"``) or per-nanometre photon flux
(``"photons/s/m^2/nm"``). Values are densities per nanometre; turning a cube
into band energy or electrons requires an explicit wavelength integral in a
later stage.

The wavelength axis is a :class:`WavelengthGrid`: one-dimensional, strictly
increasing, uniformly spaced, and positive. Uniform spacing lets downstream
code precompute projection matrices and band widths once. Real spectral datasets
are published on grids from 0.1 nm (laboratory spectroradiometers) to 20 nm
(coarse colour-science tables); camsim's default sensor grid is 360-830 nm in
5 nm steps, matching common camera QE/CFA measurements.

Resampling is never implicit. :meth:`SpectralIrradiance.resample_spatial`
performs an area-conserving box average when the target dimensions divide the
input dimensions, which is how a 4x oversampled scene becomes sensor pixels.
:meth:`SpectralIrradiance.resample_wavelengths` performs overlap-weighted
integration between uniform grids, treating each sample as a piecewise-constant
per-nm density over its cell; interior cell edges are sample midpoints and the
outer edges are the first and last sample, so the represented domain is exactly
the sampled range and the cell integral over a covered target range is
preserved. Uncovered target ranges raise unless the caller explicitly opts into
zero fill.

Failure artefacts guarded here: a wavelength axis that does not match the data
length silently mis-colours a scene; a non-uniform grid breaks projection-matrix
assumptions; negative or NaN irradiance has no physical meaning and poisons
noise generation; and an implicit unit mix-up produces RAW values wrong by many
orders of magnitude. Each condition is rejected with a message naming the
offending field instead of being coerced.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.core.units import irradiance_w_to_photons, photons_to_irradiance_w

SpectralUnit: TypeAlias = Literal["W/m^2/nm", "photons/s/m^2/nm"]
"""The two supported per-nanometre spectral density units."""

_FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]

_SUPPORTED_UNITS: Final[tuple[SpectralUnit, ...]] = ("W/m^2/nm", "photons/s/m^2/nm")

# Grid uniformity is checked with a relative tolerance on the spacing. A 1e-9 nm
# absolute floor keeps grids near zero safe without accepting visibly irregular
# steps; float64 subtraction of nm-scale values is accurate to ~1e-13 nm.
_UNIFORM_RTOL: Final[float] = 1e-6
_UNIFORM_ATOL_NM: Final[float] = 1e-9

# Coverage is a physical containment check. A micrometre-nanometre allowance
# absorbs floating-point round-off in grids built with np.arange/np.linspace.
_COVERAGE_TOL_NM: Final[float] = 1e-6


def _validate_unit(unit: object) -> SpectralUnit:
    """Return ``unit`` narrowed to a supported literal or raise ValueError."""
    if unit not in _SUPPORTED_UNITS:
        raise ValueError(f"unit must be one of {_SUPPORTED_UNITS}; got {unit!r}")
    return unit


@dataclass(frozen=True, eq=False)
class WavelengthGrid:
    """A validated, immutable, uniform wavelength axis in nanometres.

    Parameters
    ----------
    nm:
        One-dimensional wavelength samples in nanometres. The array is copied
        and made read-only, so the grid cannot drift after construction.

    The grid must contain at least two wavelengths so that a spacing is
    defined. Non-uniform spacing, non-positive or non-finite wavelengths, and
    non-monotonic order are rejected because they break the projection-matrix
    and band-width assumptions shared by the optics, sensor, and metrics layers.
    """

    nm: NDArray[np.float64]

    def __post_init__(self) -> None:
        wavelengths = np.array(self.nm, dtype=np.float64, copy=True)
        if wavelengths.ndim != 1:
            raise ValueError("WavelengthGrid must be a 1-D array of wavelengths in nm")
        if wavelengths.size < 2:
            raise ValueError("WavelengthGrid requires at least two wavelengths")
        if not np.all(np.isfinite(wavelengths)):
            raise ValueError("WavelengthGrid wavelengths must be finite")
        if np.any(wavelengths <= 0.0):
            raise ValueError("WavelengthGrid wavelengths must be positive")
        spacing = np.diff(wavelengths)
        if np.any(spacing <= 0.0):
            raise ValueError("WavelengthGrid wavelengths must be strictly increasing")
        if not np.allclose(spacing, spacing[0], rtol=_UNIFORM_RTOL, atol=_UNIFORM_ATOL_NM):
            raise ValueError("WavelengthGrid spacing must be uniform")
        wavelengths.setflags(write=False)
        object.__setattr__(self, "nm", wavelengths)

    def __len__(self) -> int:
        """Return the number of wavelength samples on the grid."""
        return int(self.nm.size)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WavelengthGrid):
            return False
        return bool(np.array_equal(self.nm, other.nm))

    def __hash__(self) -> int:
        return hash((self.nm.shape, self.nm.tobytes()))


def _coerce_data(data: ArrayLike) -> _FloatArray:
    """Validate a spectral cube and normalise it to float32 or float64."""
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(
            f"SpectralIrradiance data must be a 3-D (H, W, L) array; got shape {array.shape}"
        )
    if array.dtype != np.float32 and array.dtype != np.float64:
        if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
            raise TypeError(f"SpectralIrradiance data must be real-valued; got dtype {array.dtype}")
        array = array.astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("SpectralIrradiance data must be finite (no NaN or infinity)")
    if np.any(array < 0.0):
        raise ValueError("SpectralIrradiance data must be non-negative")
    return array


def _cell_edges(wavelengths_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the L+1 piecewise-constant cell boundaries for a uniform grid.

    Interior boundaries sit at midpoints between samples; the outer boundaries
    coincide with the first and last sample. The represented domain is therefore
    exactly the sampled wavelength range, with half-width cells at the ends.
    This convention makes the cell integral equal to the trapezoidal integral of
    the sample values and lets grids with the same endpoints but different
    spacings be resampled without inventing data outside the measured range.
    """
    midpoints = 0.5 * (wavelengths_nm[:-1] + wavelengths_nm[1:])
    return np.asarray(
        np.concatenate((wavelengths_nm[:1], midpoints, wavelengths_nm[-1:])),
        dtype=np.float64,
    )


@dataclass(eq=False, init=False)
class SpectralIrradiance:
    """A validated spectral irradiance cube on a uniform wavelength grid.

    Parameters
    ----------
    data:
        Array-like of shape ``(H, W, L)``. Float32 and float64 are preserved;
        integer, float16, and other real inputs are promoted to float64. Values
        must be finite and non-negative.
    wavelengths_nm:
        A :class:`WavelengthGrid`, or an array-like of nanometres that is
        validated into one. Its length must equal ``data.shape[2]``.
    unit:
        ``"W/m^2/nm"`` or ``"photons/s/m^2/nm"``. The unit is metadata that
        conversions and later stages use; the class never guesses it.
    pixel_pitch_um:
        Optional square-pixel pitch in micrometres. Positive when present; used
        by the sensor to reject cubes rendered for a different pixel grid.

    The cube describes irradiance at the sensor plane *before* the sensor's own
    IR-cut and CFA filters, so those filters are applied exactly once. Values
    are per-nanometre densities; this class does not integrate over wavelength.
    """

    data: _FloatArray
    wavelengths_nm: WavelengthGrid
    unit: SpectralUnit
    pixel_pitch_um: float | None

    def __init__(
        self,
        data: ArrayLike,
        wavelengths_nm: WavelengthGrid | ArrayLike,
        unit: SpectralUnit,
        pixel_pitch_um: float | None = None,
    ) -> None:
        checked_data = _coerce_data(data)
        grid = (
            wavelengths_nm
            if isinstance(wavelengths_nm, WavelengthGrid)
            else WavelengthGrid(np.asarray(wavelengths_nm, dtype=np.float64))
        )
        checked_unit = _validate_unit(unit)
        if checked_data.shape[2] != grid.nm.size:
            raise ValueError(
                f"SpectralIrradiance data has {checked_data.shape[2]} wavelength samples "
                f"but the wavelength grid has {grid.nm.size}"
            )
        pitch: float | None
        if pixel_pitch_um is None:
            pitch = None
        else:
            pitch = float(pixel_pitch_um)
            if not np.isfinite(pitch) or pitch <= 0.0:
                raise ValueError("pixel_pitch_um must be a positive finite value")
        self.data = checked_data
        self.wavelengths_nm = grid
        self.unit = checked_unit
        self.pixel_pitch_um = pitch

    @classmethod
    def from_array(
        cls,
        data: ArrayLike,
        wavelengths_nm: WavelengthGrid | ArrayLike,
        unit: SpectralUnit,
        pixel_pitch_um: float | None = None,
    ) -> Self:
        """Construct from array-like data and wavelengths.

        This is the ergonomic entry point for external spectral cubes; it is
        equivalent to calling the constructor directly.
        """
        return cls(data, wavelengths_nm, unit, pixel_pitch_um)

    def to_unit(self, unit: SpectralUnit) -> SpectralIrradiance:
        """Return a new cube converted to ``unit``.

        The conversion is per-nanometre and pointwise: energy density is divided
        by ``h*c/lambda`` to obtain photon flux density, or multiplied by it in
        the reverse direction. Wavelength integration is intentionally not part
        of this operation. The returned cube is always a new object, even when
        the requested unit matches the current one.
        """
        target_unit = _validate_unit(unit)
        if target_unit == self.unit:
            return SpectralIrradiance(
                self.data.copy(),
                self.wavelengths_nm,
                target_unit,
                self.pixel_pitch_um,
            )
        if target_unit == "photons/s/m^2/nm":
            converted = irradiance_w_to_photons(self.data, self.wavelengths_nm.nm)
        else:
            converted = photons_to_irradiance_w(self.data, self.wavelengths_nm.nm)
        return SpectralIrradiance(
            converted,
            self.wavelengths_nm,
            target_unit,
            self.pixel_pitch_um,
        )

    def resample_spatial(
        self,
        target_shape: Sequence[int],
        method: str = "box",
    ) -> SpectralIrradiance:
        """Return an area-conserving box downsample of the spatial axes.

        Each output pixel is the mean of its ``(factor_y, factor_x)`` input
        block, preserving per-unit-area irradiance. Both target dimensions must
        divide the corresponding input dimension exactly; non-integer ratios
        would require storing fractional pixel areas and are rejected. The
        wavelength axis and unit are unchanged.

        When the input carries ``pixel_pitch_um`` and both axes are reduced by
        the same factor, the output pitch is scaled by that factor. For
        anisotropic factors the scalar pitch cannot describe rectangular output
        pixels, so the metadata is dropped rather than reported incorrectly.
        """
        if method != "box":
            raise ValueError(
                f"unsupported spatial resampling method {method!r}; only 'box' is supported"
            )
        if len(target_shape) != 2:
            raise ValueError("target_shape must contain exactly two dimensions (height, width)")
        target_height = int(target_shape[0])
        target_width = int(target_shape[1])
        if target_height <= 0 or target_width <= 0:
            raise ValueError("target_shape dimensions must be positive")
        height, width, spectral_length = self.data.shape
        if (
            target_height > height
            or target_width > width
            or height % target_height != 0
            or width % target_width != 0
        ):
            raise ValueError(
                f"target_shape {(target_height, target_width)} must divide the input "
                f"spatial shape {(height, width)} evenly"
            )
        factor_y = height // target_height
        factor_x = width // target_width
        averaged = self.data.reshape(
            target_height,
            factor_y,
            target_width,
            factor_x,
            spectral_length,
        ).mean(axis=(1, 3))
        pitch = self.pixel_pitch_um
        if pitch is not None:
            pitch = pitch * factor_y if factor_y == factor_x else None
        return SpectralIrradiance(averaged, self.wavelengths_nm, self.unit, pitch)

    def resample_wavelengths(
        self,
        target_grid: WavelengthGrid | ArrayLike,
        method: str = "integrate",
        allow_zero_fill: bool = False,
    ) -> SpectralIrradiance:
        """Return the cube integrated onto ``target_grid``.

        Samples are treated as piecewise-constant per-nm densities over cells
        whose interior edges are the midpoints between neighbouring samples and
        whose outer edges are the first and last sample. For each target cell,
        the overlapping source cells are weighted by their overlap length and
        divided by the target cell width. This preserves the cell integral over
        the covered target range and reproduces the input exactly when the grids
        are identical.

        The target wavelength range must lie inside the source range. Otherwise
        ``ValueError`` is raised; with ``allow_zero_fill=True`` uncovered regions
        contribute zero (followed by a :class:`UserWarning`) so partially covered
        edge cells are averaged against zero instead of inventing signal.
        """
        if method != "integrate":
            raise ValueError(
                "unsupported wavelength resampling method "
                f"{method!r}; only 'integrate' is supported"
            )
        target = (
            target_grid
            if isinstance(target_grid, WavelengthGrid)
            else WavelengthGrid(np.asarray(target_grid, dtype=np.float64))
        )
        source_nm = self.wavelengths_nm.nm
        target_nm = target.nm
        covered = (
            target_nm[0] >= source_nm[0] - _COVERAGE_TOL_NM
            and target_nm[-1] <= source_nm[-1] + _COVERAGE_TOL_NM
        )
        if not covered:
            if not allow_zero_fill:
                raise ValueError(
                    f"target wavelength range [{target_nm[0]:g}, {target_nm[-1]:g}] nm is "
                    f"not covered by the input range [{source_nm[0]:g}, {source_nm[-1]:g}] nm; "
                    "pass allow_zero_fill=True to zero-fill uncovered regions"
                )
            warnings.warn(
                "target wavelength range is not fully covered by the input range; "
                "zero-filling uncovered regions",
                UserWarning,
                stacklevel=2,
            )
        source_edges = _cell_edges(source_nm)
        target_edges = _cell_edges(target_nm)
        overlap = np.clip(
            np.minimum(target_edges[1:, None], source_edges[None, 1:])
            - np.maximum(target_edges[:-1, None], source_edges[None, :-1]),
            0.0,
            None,
        )
        target_widths = target_edges[1:] - target_edges[:-1]
        integrated = self.data.astype(np.float64, copy=False) @ overlap.T
        converted = integrated / target_widths
        return SpectralIrradiance(
            converted,
            target,
            self.unit,
            self.pixel_pitch_um,
        )
