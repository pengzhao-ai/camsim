"""Exact SI physical constants used by the radiometric simulation.

camsim keeps every radiometric quantity in SI base units. Two constants bridge
the wave and photon pictures of light:

* ``H_PLANCK`` (J s) fixes the energy of a photon at frequency ``nu``:
  ``E = h * nu``.
* ``C_LIGHT`` (m/s) converts that frequency to a vacuum wavelength:
  ``nu = c / lambda``.

Together they give the relation used throughout the sensor model::

    E_photon(lambda) = h * c / lambda  [J]

with ``lambda`` in metres. Both values are exact under the 2019 SI redefinition,
so conversions are limited only by the floating-point representation of the
wavelength.

Real-world scale: a visible photon (360-830 nm) carries roughly 1.5-3.4 eV
(2.4-5.5 x 10^-19 J). A 10 ms exposure of a 1.12 um pixel under daylight collects
on the order of 10^3-10^5 photons, which is why shot noise matters.

Failure artefacts: passing a wavelength in nanometres directly to ``h * c /
lambda`` overestimates photon energy by 10^9, turning a plausible photon count
into an infinitesimal one; using a rounded ``c`` or ``h`` shifts absolute
radiometry by more than any sensor tolerance in this project permits.
"""

from __future__ import annotations

from typing import Final

# Exact SI values (2019 redefinition). Do not round: spectral projections and
# electron-count checks are validated against absolute radiometry.
H_PLANCK: Final[float] = 6.626_070_15e-34  # Planck constant [J s]
C_LIGHT: Final[float] = 2.997_924_58e8  # speed of light in vacuum [m/s]
