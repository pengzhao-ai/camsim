"""Conversions between spectral radiant power and photon flux.

The simulation tracks light in two interchangeable currencies. Radiometry
measures power per unit area and wavelength, ``E(lambda)`` in W/m^2/nm.
Photon counting instead needs a rate per nanometre of bandwidth::

    Phi(lambda) = E(lambda) / E_photon(lambda)

with ``E_photon(lambda) = h * c / lambda`` in joules. Because both quantities
are densities *per nanometre*, these helpers never integrate across wavelength;
later stages multiply by wavelength cell widths when turning a spectral cube
into electrons, using the same trapezoidal weights as the resampler (half-width
cells at the first and last sample, full width between interior samples).
Keeping that integral explicit avoids double-counting bandwidth when grids are
resampled.

Real-world parameter ranges: the visible band spans roughly 360-830 nm. At
555 nm a photon carries 3.578 x 10^-19 J, corresponding to 2.79 x 10^18
photons/s per watt. Daylight scene irradiance is of order 0.1-10 W/m^2/nm, so
typical photon flux densities are 10^17-10^19 photons/s/m^2/nm.

Failure artefacts: supplying a wavelength already in metres to these nm-based
helpers applies the nm-to-metre factor twice, overestimating photon energy by
10^9 and underestimating photon flux by the same factor; supplying a total band
power instead of a per-nm density makes the result wrong by the bandwidth;
non-positive wavelengths are physically meaningless and produce infinities or
sign flips, so they are rejected rather than propagated.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.core.constants import C_LIGHT, H_PLANCK

# Wavelengths arrive in nanometres because that is how spectral data files and
# sensor datasheets are published; photon energy needs metres.
_NM_TO_M = 1e-9


def _wavelengths_to_metres(wavelength_nm: ArrayLike) -> NDArray[np.float64]:
    """Convert finite, strictly positive wavelengths from nm to metres."""
    wavelengths = np.asarray(wavelength_nm, dtype=np.float64)
    if not np.all(np.isfinite(wavelengths)):
        raise ValueError("wavelength_nm must be finite")
    if np.any(wavelengths <= 0.0):
        raise ValueError("wavelength_nm must be positive")
    return wavelengths * _NM_TO_M


def photon_energy_j(wavelength_nm: ArrayLike) -> NDArray[np.float64]:
    """Return the energy of one photon, in joules, at each wavelength in nm.

    ``E = h * c / lambda`` with ``lambda`` converted from nanometres to metres.
    The result has the broadcast shape of ``wavelength_nm`` (a 0-d array for a
    scalar input).
    """
    wavelengths_m = _wavelengths_to_metres(wavelength_nm)
    return np.asarray(H_PLANCK * C_LIGHT / wavelengths_m, dtype=np.float64)


def irradiance_w_to_photons(
    irradiance: ArrayLike,
    wavelength_nm: ArrayLike,
) -> NDArray[np.float64]:
    """Convert per-nm spectral irradiance in W/m^2/nm to photons/s/m^2/nm.

    ``irradiance`` and ``wavelength_nm`` follow NumPy broadcasting rules, which
    lets a ``(H, W, L)`` cube be converted with an ``(L,)`` wavelength array.
    """
    radiant = np.asarray(irradiance, dtype=np.float64)
    return np.asarray(radiant / photon_energy_j(wavelength_nm), dtype=np.float64)


def photons_to_irradiance_w(
    photons: ArrayLike,
    wavelength_nm: ArrayLike,
) -> NDArray[np.float64]:
    """Convert photons/s/m^2/nm to per-nm spectral irradiance in W/m^2/nm."""
    flux = np.asarray(photons, dtype=np.float64)
    return np.asarray(flux * photon_energy_j(wavelength_nm), dtype=np.float64)
