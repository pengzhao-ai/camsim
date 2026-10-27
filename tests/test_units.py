"""Unit tests for photon-energy and irradiance/photon-flux conversions.

The conversions are per nanometre of spectral bandwidth: a value expressed in
W/m^2/nm describes the radiant power density carried in one nanometre of
spectrum, so dividing by the photon energy at that nanometre gives the matching
photon flux density in photons/s/m^2/nm. Integrating across wavelength happens
later in the sensor projection, never inside these helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from camsim.core import units
from camsim.core.constants import C_LIGHT, H_PLANCK


def test_photon_energy_555nm() -> None:
    assert float(units.photon_energy_j(555.0)) == pytest.approx(3.578e-19, rel=1e-3)


def test_photon_energy_uses_metres_and_arrays() -> None:
    wavelengths_nm = np.array([450.0, 550.0, 650.0])

    energy_j = units.photon_energy_j(wavelengths_nm)

    assert energy_j.shape == wavelengths_nm.shape
    assert energy_j.dtype == np.float64
    np.testing.assert_allclose(
        energy_j,
        H_PLANCK * C_LIGHT / (wavelengths_nm * 1e-9),
        rtol=1e-14,
    )


def test_irradiance_photon_roundtrip() -> None:
    w = np.array([450.0, 550.0, 650.0])
    e = np.array([1.0, 2.0, 0.5])
    np.testing.assert_allclose(
        units.photons_to_irradiance_w(units.irradiance_w_to_photons(e, w), w),
        e,
        rtol=1e-12,
    )


def test_irradiance_conversion_matches_photon_energy() -> None:
    wavelengths_nm = np.array([450.0, 550.0, 650.0])
    irradiance = np.array([1.0, 2.0, 0.5])

    photons = units.irradiance_w_to_photons(irradiance, wavelengths_nm)

    np.testing.assert_allclose(
        photons,
        irradiance / units.photon_energy_j(wavelengths_nm),
        rtol=1e-15,
    )


def test_conversions_broadcast_over_spatial_axes() -> None:
    wavelengths_nm = np.array([450.0, 550.0, 650.0])
    irradiance = np.full((2, 4, 3), 2.0)

    photons = units.irradiance_w_to_photons(irradiance, wavelengths_nm)

    assert photons.shape == irradiance.shape
    np.testing.assert_allclose(
        units.photons_to_irradiance_w(photons, wavelengths_nm),
        irradiance,
        rtol=1e-12,
    )


@pytest.mark.parametrize("bad_wavelength", [0.0, -10.0, np.nan, np.inf])
def test_wavelength_validation_rejects_invalid_values(bad_wavelength: float) -> None:
    with pytest.raises(ValueError, match="wavelength_nm"):
        units.photon_energy_j(bad_wavelength)


# Hypothesis is declared as a development dependency because these invariants
# hold for every physically valid input, not only the hand-picked examples.
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@settings(max_examples=50, deadline=None, derandomize=True)
@given(
    irradiance=st.floats(min_value=1e-9, max_value=1e9, allow_nan=False, allow_infinity=False),
    wavelength_nm=st.floats(
        min_value=360.0, max_value=830.0, allow_nan=False, allow_infinity=False
    ),
)
def test_irradiance_photon_roundtrip_property(irradiance: float, wavelength_nm: float) -> None:
    photons = units.irradiance_w_to_photons(irradiance, wavelength_nm)
    recovered = units.photons_to_irradiance_w(photons, wavelength_nm)
    assert recovered == pytest.approx(irradiance, rel=1e-12, abs=0.0)


@settings(max_examples=50, deadline=None, derandomize=True)
@given(
    wavelength_nm=st.floats(min_value=360.0, max_value=830.0, allow_nan=False, allow_infinity=False)
)
def test_photon_energy_decreases_with_wavelength(wavelength_nm: float) -> None:
    longer = min(830.0, wavelength_nm + 1.0)
    if longer == wavelength_nm:
        return
    assert float(units.photon_energy_j(longer)) < float(units.photon_energy_j(wavelength_nm))
