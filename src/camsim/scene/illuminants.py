"""Illuminant spectra, photometric normalisation, and CIE Lab ground truth.

Colour is not a property of a surface alone. A patch with spectral reflectance
:math:`\\rho(\\lambda)` looks different under daylight and under tungsten
because the illuminant's spectral power distribution :math:`S(\\lambda)`
weights the reflected light. camsim stores two kinds of illuminant:

* **D65** - the CIE standard daylight illuminant (correlated colour
  temperature about 6504 K), read from the ``colour-science`` CIE dataset and
  resampled onto the common 360-830 nm, 5 nm grid.
* **3000 K and 4000 K** - Planckian (blackbody) radiators generated from
  Planck's law with the CIE-recommended second radiation constant
  ``c2 = 1.4388e-2 m K``, then normalised to 100 at 560 nm so their relative
  scale matches the tabulated daylight spectra.

Photometry and radiometry are kept separate. Scene radiance needs an absolute
scale, but an SPD is usually published in arbitrary relative units, so a target
illuminance in lux must be converted explicitly::

    E_scene(lambda) = lux * S(lambda) / (K_m * integral(S(lambda) V(lambda) dlambda))

with ``K_m = 683 lm/W`` and ``V(lambda)`` the CIE 1924 photopic luminous
efficiency function. ``E_scene`` is a per-nanometre density in ``W/m^2/nm``,
so the integral in that formula runs over the nanometre grid. Converting the
grid to metres without also converting the density from per-nm to per-metre
scales the result by ``1e-9``; that is the single most common photometric unit
bug, and the zero-overlap check here raises instead of silently returning a
near-zero irradiance. A lux value of zero is valid and yields a dark scene.

The colour ground truth follows the same spectral chain. For each ColorChecker
patch::

    X = integral(rho(lambda) * S(lambda) * xbar(lambda) dlambda) / N
    Y = integral(rho(lambda) * S(lambda) * ybar(lambda) dlambda) / N
    Z = integral(rho(lambda) * S(lambda) * zbar(lambda) dlambda) / N
    N = integral(S(lambda) * ybar(lambda) dlambda)

so a perfect diffuser (``rho = 1``) has ``Y = 1``. The illuminant white point is
the same integral with ``rho = 1``, and CIE 1976 Lab is computed against that
white point. D65, 3000 K, and 4000 K therefore produce different Lab
references: applying the D65 reference to a warm-light capture creates a
systematic delta-E offset that looks like a sensor colour error but is really
a white-point mistake.

Real-world parameter ranges: illuminance spans roughly 0.1 lux (moonless
night) to 100,000 lux (direct sun); photographic scenes use 1-10,000 lux.
Correlated colour temperatures for common sources are about 2700-3000 K
(incandescent), 4000 K (neutral fluorescent/LED), and 5000-6500 K (daylight).
ColorChecker Lab values under daylight span ``L*`` about 20-97 and ``a*`` /
``b*`` about -60 to 70.

Failure artefacts guarded here: treating an SPD as already being in lux;
normalising an illuminant by ``integral(S dlambda)`` instead of
``integral(S ybar dlambda)``; using a fixed D65 white point for every
illuminant; reading wavelengths in metres as nanometres (all V values become
zero); and evaluating V outside its 360-830 nm table without deciding that the
result is zero.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.core.color import ColorReference
from camsim.core.spectral import SpectralIrradiance
from camsim.scene.spectral_data import (
    COMMON_WAVELENGTH_GRID,
    SpectralPowerDistribution,
    _interpolate_linear_columns,
    load_colorchecker,
)

K_M: Final[float] = 683.0
"""Maximum luminous efficacy of radiation at 555 nm [lm/W] (CIE 1924)."""

_OBSERVER: Final[str] = "CIE 1931 2 degree"
_CMF_KEY: Final[str] = "CIE 1931 2 Degree Standard Observer"
_PHOTOPIC_KEY: Final[str] = "CIE 1924 Photopic Standard Observer"
_D65_KEY: Final[str] = "D65"
_SUPPORTED_ILLUMINANTS: Final[tuple[str, ...]] = ("D65", "3000K", "4000K")

# CIE 1976 Lab constants, written as exact rationals to avoid a rounded
# epsilon causing a discontinuity near the linear/cubic crossover.
_LAB_EPSILON: Final[float] = 216.0 / 24389.0
_LAB_KAPPA: Final[float] = 24389.0 / 27.0

_COLOUR_SCIENCE_URL: Final[str] = "https://www.colour-science.org/"
_COLOUR_SCIENCE_LICENCE: Final[str] = (
    "BSD-3-Clause (colour-science package); CIE data distributed with colour-science."
)


def _photopic_efficiency_source() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the CIE 1924 photopic luminous efficiency table in nm."""
    from colour.colorimetry.datasets import SDS_LEFS_PHOTOPIC

    lef = SDS_LEFS_PHOTOPIC[_PHOTOPIC_KEY]
    return (
        np.asarray(lef.wavelengths, dtype=np.float64),
        np.asarray(lef.values, dtype=np.float64),
    )


def _cie_1931_2deg_cmf_source() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the CIE 1931 2-degree colour-matching functions in nm."""
    from colour.colorimetry import MSDS_CMFS

    cmfs = MSDS_CMFS[_CMF_KEY]
    return (
        np.asarray(cmfs.wavelengths, dtype=np.float64),
        np.asarray(cmfs.values, dtype=np.float64),
    )


def photopic_luminous_efficiency(wavelength_nm: ArrayLike) -> NDArray[np.float64]:
    """Return the CIE 1924 photopic luminous efficiency ``V(lambda)``.

    Parameters
    ----------
    wavelength_nm:
        Wavelengths in nanometres, any shape. Values are linearly interpolated
        through the CIE 1924 table (360-830 nm at 1 nm).

    Returns
    -------
    ndarray
        ``V(lambda)`` with the input shape, in ``[0, 1]``; ``V(555 nm) = 1``.
        Wavelengths outside the CIE table return zero, including the entire
        table when a caller accidentally supplies metres (for example
        ``5.55e-7``); callers that require coverage should check the result.

    Raises
    ------
    ValueError
        If any wavelength is NaN or infinite.
    """
    wavelengths = np.asarray(wavelength_nm, dtype=np.float64)
    if not np.all(np.isfinite(wavelengths)):
        raise ValueError("wavelength_nm must be finite")
    source_nm, source_values = _photopic_efficiency_source()
    interpolated = np.interp(
        wavelengths.ravel(),
        source_nm,
        source_values,
        left=0.0,
        right=0.0,
    )
    return np.asarray(interpolated.reshape(wavelengths.shape), dtype=np.float64)


def _cmf_on_common_grid() -> NDArray[np.float64]:
    """Return ``(95, 3)`` CIE 1931 2-degree XYZ colour-matching functions."""
    source_nm, values = _cie_1931_2deg_cmf_source()
    return _interpolate_linear_columns(source_nm, values, COMMON_WAVELENGTH_GRID.nm)


def _d65() -> SpectralPowerDistribution:
    """Build the CIE D65 SPD on the common grid."""
    from colour.colorimetry import SDS_ILLUMINANTS

    source = SDS_ILLUMINANTS[_D65_KEY]
    grid = COMMON_WAVELENGTH_GRID
    values = _interpolate_linear_columns(
        np.asarray(source.wavelengths, dtype=np.float64),
        np.asarray(source.values, dtype=np.float64),
        grid.nm,
    )
    provenance = {
        "source": "CIE standard illuminant D65 (relative spectral power distribution)",
        "source_package": "colour-science",
        "source_module": "colour.colorimetry.SDS_ILLUMINANTS",
        "source_key": _D65_KEY,
        "source_url": _COLOUR_SCIENCE_URL,
        "licence": _COLOUR_SCIENCE_LICENCE,
        "reference": "CIE 15:2004",
        "interpolation": "linear with constant endpoint extrapolation to 830 nm",
        "normalization": "CIE relative SPD (100 at 560 nm)",
        "wavelength_range_nm": (float(grid.nm[0]), float(grid.nm[-1])),
        "wavelength_step_nm": float(grid.nm[1] - grid.nm[0]),
        "retrieval": "runtime from the installed colour-science package; no CSV vendored",
    }
    return SpectralPowerDistribution(
        name=_D65_KEY,
        wavelengths_nm=grid,
        values=values,
        provenance=provenance,
    )


def _blackbody(temperature_k: float) -> SpectralPowerDistribution:
    """Build a relative Planckian radiator SPD on the common grid."""
    import colour
    from colour.colorimetry.blackbody import sd_blackbody

    grid = COMMON_WAVELENGTH_GRID
    step_nm = float(grid.nm[1] - grid.nm[0])
    source = sd_blackbody(
        temperature_k,
        shape=colour.SpectralShape(float(grid.nm[0]), float(grid.nm[-1]), step_nm),
    )
    source_nm = np.asarray(source.wavelengths, dtype=np.float64)
    source_values = np.asarray(source.values, dtype=np.float64)
    # Normalising at 560 nm matches the convention used by CIE relative SPDs
    # and keeps plots of D65 and tungsten on a comparable scale. The absolute
    # Planck scale is arbitrary for every downstream calculation.
    reference = float(np.interp(560.0, source_nm, source_values))
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("blackbody spectrum is not positive at the 560 nm normalisation point")
    values = _interpolate_linear_columns(source_nm, source_values, grid.nm) * (100.0 / reference)
    provenance = {
        "source": "Planckian (blackbody) radiator",
        "source_package": "colour-science",
        "source_module": "colour.colorimetry.blackbody.sd_blackbody",
        "source_key": "sd_blackbody",
        "source_url": _COLOUR_SCIENCE_URL,
        "licence": _COLOUR_SCIENCE_LICENCE,
        "reference": "CIE 15:2004",
        "temperature_k": float(temperature_k),
        "planck_constants": {
            "c1_w_m2": 3.741771e-16,
            "c2_m_k": 1.4388e-2,
            "n": 1.0,
        },
        "normalization": "relative SPD normalised to 100 at 560 nm",
        "interpolation": "generated on the common grid, linear interpolation guard",
        "wavelength_range_nm": (float(grid.nm[0]), float(grid.nm[-1])),
        "wavelength_step_nm": step_nm,
        "retrieval": "computed at runtime from colour-science; no CSV vendored",
    }
    return SpectralPowerDistribution(
        name=f"{round(temperature_k)}K",
        wavelengths_nm=grid,
        values=values,
        provenance=provenance,
    )


def illuminant(name: str) -> SpectralPowerDistribution:
    """Return a supported illuminant SPD on the common wavelength grid.

    Parameters
    ----------
    name:
        ``"D65"``, ``"3000K"``, or ``"4000K"``. Matching is case-insensitive
        and tolerates spaces (``"3000 K"``). No other illuminant is silently
        mapped to D65; unsupported names raise.

    Returns
    -------
    SpectralPowerDistribution
        Relative SPD on the 360-830 nm, 5 nm grid, with provenance recording
        whether the values came from the CIE D65 table or Planck's law.

    Raises
    ------
    TypeError
        If ``name`` is not a string.
    ValueError
        If ``name`` is not one of the three supported illuminants.
    """
    if not isinstance(name, str):
        raise TypeError(f"illuminant name must be a string; got {type(name).__name__}")
    key = "".join(name.split()).upper()
    if key == _D65_KEY:
        return _d65()
    if key in ("3000K", "4000K"):
        return _blackbody(float(key[:-1]))
    raise ValueError(f"unsupported illuminant {name!r}; expected one of {_SUPPORTED_ILLUMINANTS}")


def lux_to_spectral_irradiance(
    lux: float,
    spd: SpectralPowerDistribution,
) -> SpectralIrradiance:
    """Convert a target illuminance in lux to spectral irradiance.

    The returned cube has shape ``(1, 1, L)`` in ``"W/m^2/nm"`` and is
    normalised so that

    ``K_M * integral(E(lambda) * V(lambda) dlambda) = lux``

    with ``lambda`` in nanometres. The single spatial sample is a broadcastable
    reference spectrum; scene rendering multiplies it by spatially varying
    reflectance and lens transmittance.

    Parameters
    ----------
    lux:
        Target illuminance in lux. Must be finite and non-negative; zero is a
        valid dark scene.
    spd:
        Relative illuminant SPD. Its values provide the spectral shape; their
        absolute scale cancels in the normalisation.

    Raises
    ------
    TypeError
        If ``spd`` is not a :class:`SpectralPowerDistribution`.
    ValueError
        If ``lux`` is negative, NaN, or infinite, or if the SPD has no overlap
        with the photopic band (for example an infrared-only spectrum, or a
        grid expressed in metres).
    """
    if not isinstance(spd, SpectralPowerDistribution):
        raise TypeError(
            "lux_to_spectral_irradiance requires a SpectralPowerDistribution; "
            f"got {type(spd).__name__}"
        )
    try:
        illuminance = float(lux)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lux must be a finite, non-negative value; {exc}") from exc
    if not np.isfinite(illuminance) or illuminance < 0.0:
        raise ValueError(f"lux must be a finite, non-negative value; got {lux!r}")

    wavelengths_nm = spd.wavelengths_nm.nm
    efficiency = photopic_luminous_efficiency(wavelengths_nm)
    luminous_integral = float(np.trapezoid(spd.values * efficiency, wavelengths_nm))
    if not np.isfinite(luminous_integral) or luminous_integral <= 0.0:
        raise ValueError(
            "the illuminant has a zero photopic integral on its wavelength grid; "
            "check that the grid is in nanometres and overlaps 360-830 nm"
        )

    spectral = illuminance * spd.values / (K_M * luminous_integral)
    return SpectralIrradiance(
        spectral.reshape(1, 1, -1),
        spd.wavelengths_nm,
        "W/m^2/nm",
    )


def _lab_f(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the CIE Lab lightness/chroma transform to normalised ratios."""
    return np.asarray(
        np.where(t > _LAB_EPSILON, np.cbrt(t), (_LAB_KAPPA * t + 16.0) / 116.0),
        dtype=np.float64,
    )


def _xyz_to_lab(
    xyz: NDArray[np.float64],
    white_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert ``(n, 3)`` XYZ to CIE 1976 Lab against an explicit white point."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be an (n, 3) array; got shape {xyz.shape}")
    if white_xyz.shape != (3,):
        raise ValueError(f"white_xyz must have shape (3,); got {white_xyz.shape}")
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(white_xyz)):
        raise ValueError("XYZ values must be finite")
    if np.any(white_xyz <= 0.0):
        raise ValueError("the illuminant white point must have positive XYZ components")

    ratios = xyz / white_xyz
    fx = _lab_f(ratios[:, 0])
    fy = _lab_f(ratios[:, 1])
    fz = _lab_f(ratios[:, 2])
    return np.column_stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)))


def _xyz_to_xy(xyz: NDArray[np.float64]) -> tuple[float, float]:
    """Convert XYZ to CIE 1931 chromaticity ``(x, y)``."""
    total = float(np.sum(xyz))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("cannot convert a non-positive XYZ sum to chromaticity")
    return (float(xyz[0] / total), float(xyz[1] / total))


def reference_lab(illuminant_name: str) -> ColorReference:
    """Compute the ColorChecker Lab ground truth under one illuminant.

    The calculation is spectral, not a fixed table: patch reflectance is
    multiplied by the requested illuminant's SPD, integrated against the CIE
    1931 2-degree colour-matching functions, normalised by the illuminant's
    ``integral(S(lambda) ybar(lambda) dlambda)``, and converted to CIE 1976 Lab
    against that illuminant's own white point. A perfect diffuser therefore has
    ``Y = 1`` under every illuminant, and D65, 3000 K, and 4000 K give
    different references.

    Parameters
    ----------
    illuminant_name:
        ``"D65"``, ``"3000K"``, or ``"4000K"`` (see :func:`illuminant`).

    Returns
    -------
    ColorReference
        ``(24, 3)`` Lab patches in chart order, the illuminant's chromaticity
        white point, the observer label, and provenance linking the chart,
        illuminant, and integration method.

    Raises
    ------
    ValueError
        If the illuminant name is unsupported or the illuminant has no
        photopic/colorimetric overlap with the grid.
    """
    spd = illuminant(illuminant_name)
    chart = load_colorchecker()
    grid = COMMON_WAVELENGTH_GRID
    if not (
        np.array_equal(spd.wavelengths_nm.nm, grid.nm)
        and np.array_equal(chart.wavelengths_nm.nm, grid.nm)
    ):
        raise ValueError("chart and illuminant must share the common wavelength grid")

    cmfs = _cmf_on_common_grid()
    wavelengths_nm = grid.nm
    spd_values = spd.values

    norm = float(np.trapezoid(spd_values * cmfs[:, 1], wavelengths_nm))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("the illuminant has a zero colorimetric integral on the common grid")

    white_xyz = np.array(
        [
            float(np.trapezoid(spd_values * cmfs[:, index], wavelengths_nm)) / norm
            for index in range(3)
        ],
        dtype=np.float64,
    )
    patch_xyz = np.column_stack(
        [
            np.trapezoid(
                chart.spectra * spd_values[None, :] * cmfs[None, :, index],
                wavelengths_nm,
                axis=1,
            )
            / norm
            for index in range(3)
        ]
    )
    lab = _xyz_to_lab(np.asarray(patch_xyz, dtype=np.float64), white_xyz)
    white_point_xy = _xyz_to_xy(white_xyz)

    provenance: dict[str, Any] = {
        "source": "spectral integration of ColorChecker reflectance x illuminant x CMFs",
        "colorchecker": chart.provenance,
        "illuminant": spd.provenance,
        "cmf": _CMF_KEY,
        "observer": _OBSERVER,
        "integration": "trapezoidal on the common 360-830 nm, 5 nm grid",
        "normalization": (
            "divide by integral(S(lambda) * ybar(lambda) dlambda); a perfect diffuser has Y = 1"
        ),
        "white_point_xyz": [float(component) for component in white_xyz],
    }
    return ColorReference(
        lab=lab,
        white_point_xy=white_point_xy,
        illuminant_name=spd.name,
        observer=_OBSERVER,
        provenance=provenance,
    )
