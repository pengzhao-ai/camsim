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

Failure artefacts: the nm/metre boundary is the classic unit trap. Passing a
555 nm wavelength as the bare number 555 to ``h * c / lambda`` expecting metres
underestimates photon energy by 10^9 (3.58 x 10^-28 J instead of 3.578 x 10^-19
J) and therefore inflates photon counts by the same factor. The opposite
mistake, passing a metre-valued wavelength such as 5.55 x 10^-7 to an API that
applies the nm-to-metre factor again, overestimates photon energy by 10^9 and
turns plausible photon counts into an infinitesimal number. Using a rounded
``c`` or ``h`` shifts absolute radiometry by more than any sensor tolerance in
this project permits.
"""

from __future__ import annotations

from typing import Final

# Exact SI values (2019 redefinition). Do not round: spectral projections and
# electron-count checks are validated against absolute radiometry.
H_PLANCK: Final[float] = 6.626_070_15e-34  # Planck constant [J s]
C_LIGHT: Final[float] = 2.997_924_58e8  # speed of light in vacuum [m/s]
