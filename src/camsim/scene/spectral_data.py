"""Spectral reflectance data for the 24-patch ColorChecker chart.

The ColorChecker Classic is a physical chart whose patches are characterised by
*spectral reflectance factor* :math:`\\rho(\\lambda)`, a dimensionless ratio of
reflected to incident radiant flux under diffuse illumination. Reflectance is
not a colour and not a radiance: it is a material property in ``[0, 1]`` that
must be multiplied by an illuminant and integrated against colour-matching
functions before it becomes a tristimulus value. The patches span roughly
``rho = 0.02`` (black) to ``0.93`` (white); the six neutral patches step down in
roughly 0.23-density increments, which is what makes the chart useful for
tone and white-balance checks as well as colour error.

camsim uses the *BabelColor Average* spectra shipped with ``colour-science``:
the average of measurements of 30 ColorChecker Classic charts, tabulated on a
380-730 nm grid at 10 nm. The simulation works on a single common grid of
360-830 nm at 5 nm (95 samples) so that optics, sensor, and metrics can share
projection matrices. Resampling is explicit linear interpolation with constant
endpoint extrapolation, following the CIE recommendation also used by
``colour-science``: the 380 and 730 nm values are held constant over the small
unmeasured margins rather than dropping reflectance to zero, which would create
a spurious edge in the integrated colour.

Real-world parameter ranges: spectral reflectance is measured from 360-2500 nm
in laboratory spectrophotometers, but camera colour work almost always uses
360-830 nm; reflectance factors above 1.0 indicate measurement geometry or
fluorescence, not a passive surface. The chart's patch order is fixed by the
manufacturer (four rows of six patches, read left to right, top to bottom), and
reordering it silently mismatches every downstream comparison.

Failure artefacts guarded here: feeding display-encoded sRGB triplets where
reflectance spectra are expected produces plausible-looking but meaningless Lab
values; treating reflectance as radiance omits the illuminant and makes the
chart appear white under every light source; zero-padding the unmeasured
380-730 nm margins biases dark patches; a wavelength array in metres does not
match the nanometre grid; and a changed upstream patch order breaks chart
layouts without raising. The loader checks the upstream names and order, and
validates shape, range, and finiteness rather than coercing bad data.

Data are read from the installed ``colour-science`` package at runtime and are
not vendored as CSV files. Provenance, the BabelColor permission notice, and
the BSD-3-Clause packaging licence are recorded in ``THIRD_PARTY.md`` and in
each returned object's ``provenance`` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.core.spectral import WavelengthGrid

# 360-830 nm at 5 nm is the shared camera-colour grid used by the scene, optics,
# sensor, and metrics layers. It covers the CIE 1931 observer table exactly and
# matches common QE/CFA measurement spacing.
_COMMON_WAVELENGTHS_NM: Final[NDArray[np.float64]] = np.arange(360.0, 831.0, 5.0)

COMMON_WAVELENGTH_GRID: Final[WavelengthGrid] = WavelengthGrid(_COMMON_WAVELENGTHS_NM)
"""The 95-sample, 360-830 nm, 5 nm grid shared by all spectral scene data."""

# The dataset order is the physical chart layout: four rows of six patches.
# Comparing against this tuple turns an upstream reordering into an explicit
# error instead of silently swapping patch colours.
_COLORCHECKER_PATCH_ORDER: Final[tuple[str, ...]] = (
    "dark skin",
    "light skin",
    "blue sky",
    "foliage",
    "blue flower",
    "bluish green",
    "orange",
    "purplish blue",
    "moderate red",
    "purple",
    "yellow green",
    "orange yellow",
    "blue",
    "green",
    "red",
    "yellow",
    "magenta",
    "cyan",
    "white 9.5 (.05 D)",
    "neutral 8 (.23 D)",
    "neutral 6.5 (.44 D)",
    "neutral 5 (.70 D)",
    "neutral 3.5 (1.05 D)",
    "black 2 (1.5 D)",
)

_COLORCHECKER_SOURCE_URL: Final[str] = "http://www.babelcolor.com/main_level/ColorChecker.htm"
_COLORCHECKER_DATA_URL: Final[str] = (
    "http://www.babelcolor.com/download/ColorChecker_RGB_and_spectra.xls"
)
_COLORCHECKER_LICENCE: Final[str] = (
    "BSD-3-Clause (colour-science); BabelColor ColorChecker data Copyright (C) "
    "2004-2012 Danny Pascale (www.babelcolor.com); used by permission."
)


def _require_text(value: object, field_name: str) -> str:
    """Return a non-empty, non-whitespace string or raise ValueError."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string; got {value!r}")
    return value


def _copy_metadata(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    """Return a shallow copy of a provenance mapping, rejecting non-mappings."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping of metadata; got {type(value).__name__}")
    return dict(value)


def _interpolate_linear_columns(
    source_nm: ArrayLike,
    values: ArrayLike,
    target_nm: ArrayLike,
) -> NDArray[np.float64]:
    """Linearly interpolate one or more spectra onto ``target_nm``.

    ``values`` is either a 1-D array of samples or a 2-D ``(samples, columns)``
    array. Values outside the measured range are held at the nearest endpoint
    (constant extrapolation), the CIE recommendation for spectral data.
    """
    source = np.asarray(source_nm, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    target = np.asarray(target_nm, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("source_nm and target_nm must be 1-D wavelength arrays")
    if not (np.all(np.isfinite(source)) and np.all(np.isfinite(target))):
        raise ValueError("wavelengths must be finite")
    if source.size < 2 or np.any(np.diff(source) <= 0.0):
        raise ValueError("source_nm must contain at least two strictly increasing samples")
    if data.ndim == 1:
        if data.size != source.size:
            raise ValueError(
                f"values has {data.size} samples but source_nm has {source.size} wavelengths"
            )
        return np.asarray(np.interp(target, source, data), dtype=np.float64)
    if data.ndim == 2:
        if data.shape[0] != source.size:
            raise ValueError(
                f"values has {data.shape[0]} samples but source_nm has {source.size} wavelengths"
            )
        return np.column_stack(
            [np.interp(target, source, data[:, index]) for index in range(data.shape[1])]
        )
    raise ValueError(f"values must be 1-D or 2-D; got shape {data.shape}")


def _validate_spd_values(values: ArrayLike, grid: WavelengthGrid) -> NDArray[np.float64]:
    """Return a read-only, finite, non-negative 1-D SPD sampled on ``grid``."""
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SpectralPowerDistribution values must be real numbers; {exc}") from exc
    if array.ndim != 1:
        raise ValueError(
            f"SpectralPowerDistribution values must be a 1-D array; got shape {array.shape}"
        )
    if array.size != len(grid):
        raise ValueError(
            "SpectralPowerDistribution values must match the wavelength grid: "
            f"got {array.size} samples for {len(grid)} wavelengths"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("SpectralPowerDistribution values must be finite (no NaN or infinity)")
    if np.any(array < 0.0):
        raise ValueError("SpectralPowerDistribution values must be non-negative")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False, init=False)
class SpectralPowerDistribution:
    """A relative spectral power distribution on a validated wavelength grid.

    Parameters
    ----------
    name:
        Human-readable label such as ``"D65"`` or ``"3000K"``. Non-empty.
    wavelengths_nm:
        A :class:`~camsim.core.spectral.WavelengthGrid`, or an array-like of
        nanometres that is validated into one.
    values:
        One-dimensional, finite, non-negative spectral values. The absolute
        scale is arbitrary for relative SPDs; photometric normalisation happens
        in :func:`camsim.scene.illuminants.lux_to_spectral_irradiance`. The
        array is copied and made read-only.
    provenance:
        Optional mapping recording dataset, source URL, licence, and any
        processing (interpolation, normalisation). Copied to a plain dict.

    Notes
    -----
    An SPD is radiometric power per unit wavelength, not a colour, not a
    luminance, and not an irradiance at a particular scene location. Treating
    an SPD as already scaled in lux is the classic photometric/radiometric
    confusion; this type deliberately carries no photometric unit.
    """

    name: str
    wavelengths_nm: WavelengthGrid
    values: NDArray[np.float64]
    provenance: dict[str, Any]

    def __init__(
        self,
        name: str,
        wavelengths_nm: WavelengthGrid | ArrayLike,
        values: ArrayLike,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "name", _require_text(name, "name"))
        grid = (
            wavelengths_nm
            if isinstance(wavelengths_nm, WavelengthGrid)
            else WavelengthGrid(np.asarray(wavelengths_nm, dtype=np.float64))
        )
        object.__setattr__(self, "wavelengths_nm", grid)
        object.__setattr__(self, "values", _validate_spd_values(values, grid))
        object.__setattr__(self, "provenance", _copy_metadata(provenance, "provenance"))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpectralPowerDistribution):
            return NotImplemented
        return (
            self.name == other.name
            and self.wavelengths_nm == other.wavelengths_nm
            and np.array_equal(self.values, other.values)
            and self.provenance == other.provenance
        )


def _validate_patches(patches: Sequence[str]) -> tuple[str, ...]:
    """Return exactly 24 unique, non-empty ColorChecker patch names."""
    if isinstance(patches, (str, bytes)):
        raise ValueError("patches must be a sequence of names, not a single string")
    try:
        names = tuple(patches)
    except TypeError as exc:
        raise ValueError(f"patches must be a sequence of names; {exc}") from exc
    if len(names) != 24:
        raise ValueError(f"ColorCheckerData requires exactly 24 patches; got {len(names)}")
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"patch names must be non-empty strings; got {name!r}")
    if len(set(names)) != len(names):
        raise ValueError("patch names must be unique")
    return names


def _validate_spectra(
    spectra: ArrayLike,
    grid: WavelengthGrid,
    patch_count: int,
) -> NDArray[np.float64]:
    """Return read-only ``(patches, wavelengths)`` reflectance factors."""
    try:
        array = np.array(spectra, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ColorCheckerData spectra must be real numbers; {exc}") from exc
    if array.ndim != 2:
        raise ValueError(
            f"ColorCheckerData spectra must be a 2-D (patches, wavelengths) array; "
            f"got shape {array.shape}"
        )
    if array.shape != (patch_count, len(grid)):
        raise ValueError(
            f"ColorCheckerData spectra must match {patch_count} patches and "
            f"{len(grid)} wavelengths; got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("ColorCheckerData spectra must be finite (no NaN or infinity)")
    if np.any(array < 0.0):
        raise ValueError("ColorCheckerData spectra reflectance factors must be non-negative")
    if np.any(array > 1.0):
        raise ValueError("ColorCheckerData spectra reflectance factors must not exceed 1.0")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False, init=False)
class ColorCheckerData:
    """Spectral reflectance of the 24 ColorChecker Classic patches.

    Parameters
    ----------
    patches:
        Exactly 24 unique patch names in physical chart order (four rows of six
        patches, left to right, top to bottom).
    spectra:
        ``(24, L)`` reflectance factors in ``[0, 1]`` on ``wavelengths_nm``.
        Copied and made read-only.
    wavelengths_nm:
        A :class:`~camsim.core.spectral.WavelengthGrid` (or array-like that is
        validated into one) with ``L`` samples.
    provenance:
        Optional mapping recording the dataset, source URL, licence, patch
        order, and interpolation. Copied to a plain dict.

    Notes
    -----
    Reflectance is a material property, not a radiance. It must be multiplied
    by an illuminant SPD and integrated against colour-matching functions
    before it describes what a camera sees; see
    :func:`camsim.scene.illuminants.reference_lab`.
    """

    patches: tuple[str, ...]
    spectra: NDArray[np.float64]
    wavelengths_nm: WavelengthGrid
    provenance: dict[str, Any]

    def __init__(
        self,
        patches: Sequence[str],
        spectra: ArrayLike,
        wavelengths_nm: WavelengthGrid | ArrayLike,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        names = _validate_patches(patches)
        grid = (
            wavelengths_nm
            if isinstance(wavelengths_nm, WavelengthGrid)
            else WavelengthGrid(np.asarray(wavelengths_nm, dtype=np.float64))
        )
        object.__setattr__(self, "patches", names)
        object.__setattr__(self, "wavelengths_nm", grid)
        object.__setattr__(self, "spectra", _validate_spectra(spectra, grid, len(names)))
        object.__setattr__(self, "provenance", _copy_metadata(provenance, "provenance"))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColorCheckerData):
            return NotImplemented
        return (
            self.patches == other.patches
            and self.wavelengths_nm == other.wavelengths_nm
            and np.array_equal(self.spectra, other.spectra)
            and self.provenance == other.provenance
        )


def load_colorchecker() -> ColorCheckerData:
    """Load the 24 ColorChecker Classic reflectance spectra.

    Returns
    -------
    ColorCheckerData
        Spectra from the BabelColor average dataset distributed with
        ``colour-science``, interpolated onto
        :data:`COMMON_WAVELENGTH_GRID` with constant endpoint extrapolation.

    Raises
    ------
    RuntimeError
        If the installed dataset does not expose the expected 24 patches in
        the documented chart order. Failing loudly prevents a layout mismatch
        from silently swapping patch colours.

    Notes
    -----
    The dataset is read at runtime rather than vendored; no CSV copy is
    committed. See ``THIRD_PARTY.md`` for the BabelColor permission notice and
    the BSD-3-Clause packaging licence.
    """
    import colour
    from colour.characterisation.datasets.colour_checkers import SDS_COLOURCHECKERS

    dataset = SDS_COLOURCHECKERS["BabelColor Average"]
    available = tuple(dataset.keys())
    if available != _COLORCHECKER_PATCH_ORDER:
        raise RuntimeError(
            "the installed colour-science ColorChecker dataset no longer matches "
            "the documented 24-patch chart order; patch mapping would be unsafe"
        )

    grid = COMMON_WAVELENGTH_GRID
    spectra = np.vstack(
        [
            _interpolate_linear_columns(
                np.asarray(dataset[name].wavelengths, dtype=np.float64),
                np.asarray(dataset[name].values, dtype=np.float64),
                grid.nm,
            )
            for name in _COLORCHECKER_PATCH_ORDER
        ]
    )
    provenance = {
        "source": "BabelColor Average ColorChecker Classic spectra",
        "source_package": "colour-science",
        "source_version": colour.__version__,
        "source_module": "colour.characterisation.datasets.colour_checkers",
        "source_key": "BabelColor Average",
        "source_url": _COLORCHECKER_SOURCE_URL,
        "data_url": _COLORCHECKER_DATA_URL,
        "licence": _COLORCHECKER_LICENCE,
        "reference": "ISO 17321-1; ColorChecker Classic (GretagMacbeth, 2005)",
        "patch_order": "row-major ColorChecker Classic layout (4 rows x 6 patches)",
        "interpolation": "linear with constant endpoint extrapolation",
        "wavelength_range_nm": (float(grid.nm[0]), float(grid.nm[-1])),
        "wavelength_step_nm": float(grid.nm[1] - grid.nm[0]),
        "retrieval": "runtime from the installed colour-science package; no CSV vendored",
    }
    return ColorCheckerData(
        patches=_COLORCHECKER_PATCH_ORDER,
        spectra=spectra,
        wavelengths_nm=grid,
        provenance=provenance,
    )
