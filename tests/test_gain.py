"""Tests for full-well clipping, conversion gain, and analog gain.

The charge-to-voltage chain has three separate physical limits that are easy to
conflate. The photodiode stores a finite charge, so the electron count saturates
at ``full_well_e``; the sense node converts charge to voltage with a conversion
gain measured in microvolts per electron; and a programmable gain amplifier
applies a linear factor that the datasheet quotes in decibels. These tests pin
each conversion independently, then verify that dual-conversion-gain modes
trade full well for output voltage exactly as the flagship preset describes.

Real-world values used here mirror the shipped sensor preset: LCG 100 uV/e with
a 6000 e- full well and HCG 300 uV/e with a 2000 e- full well. Both modes reach
600 mV at saturation, but HCG does so after one third of the charge.

Failure artefacts guarded here: clipping in the wrong domain (an ADC that
saturates before full well is a different design choice, not an error), a
conversion gain supplied in V/e instead of uV/e (a factor of one million), a
gain in dB treated as a linear multiplier, and negative or non-finite values
that would silently poison the downstream ADC.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from camsim.sensor.gain import (
    apply_analog_gain,
    clamp_full_well,
    electrons_to_voltage,
)


def test_full_well_clips() -> None:
    x = clamp_full_well(np.array([0.0, 5999.0, 6000.0, 7000.0]), 6000.0)
    np.testing.assert_array_equal(x, [0.0, 5999.0, 6000.0, 6000.0])


def test_full_well_clips_negative_charge_to_zero() -> None:
    """A clipped readout cannot return negative charge."""
    x = clamp_full_well(np.array([-25.0, -1e-12, 0.0]), 6000.0)

    np.testing.assert_array_equal(x, [0.0, 0.0, 0.0])


def test_full_well_preserves_shape_and_returns_float64() -> None:
    electrons = np.full((3, 7), 10.0)

    x = clamp_full_well(electrons, 6000.0)

    assert x.shape == electrons.shape
    assert x.dtype == np.float64


def test_full_well_accepts_integer_electrons_and_scalars() -> None:
    x = clamp_full_well(np.array([10, 7000], dtype=np.int64), 6000)
    scalar = clamp_full_well(7000, 6000.0)

    np.testing.assert_array_equal(x, [10.0, 6000.0])
    assert scalar.shape == ()
    assert float(scalar) == 6000.0


@pytest.mark.parametrize("bad_full_well", [-1.0, -1e-9, math.nan, math.inf, -math.inf])
def test_full_well_rejects_invalid_full_well(bad_full_well: float) -> None:
    with pytest.raises(ValueError, match="full_well_e"):
        clamp_full_well(np.array([1.0]), bad_full_well)


@pytest.mark.parametrize("bad_electrons", [math.nan, math.inf, -math.inf])
def test_full_well_rejects_non_finite_electrons(bad_electrons: float) -> None:
    with pytest.raises(ValueError, match="electrons"):
        clamp_full_well(np.array([10.0, bad_electrons]), 6000.0)


def test_electrons_to_voltage_is_linear() -> None:
    electrons = np.array([0.0, 1.0, 100.0, 6000.0])

    voltage = electrons_to_voltage(electrons, 100.0)

    np.testing.assert_allclose(voltage, [0.0, 100.0, 10_000.0, 600_000.0])


def test_electrons_to_voltage_uses_high_conversion_gain() -> None:
    """HCG converts each electron to three times the voltage of LCG."""
    electrons = np.array([1500.0])

    hcg = electrons_to_voltage(electrons, 300.0)

    np.testing.assert_allclose(hcg, [450_000.0])


@pytest.mark.parametrize("bad_gain", [0.0, -100.0, math.nan, math.inf, -math.inf])
def test_electrons_to_voltage_rejects_invalid_conversion_gain(bad_gain: float) -> None:
    with pytest.raises(ValueError, match="conversion_gain_uv_per_e"):
        electrons_to_voltage(np.array([1.0]), bad_gain)


@pytest.mark.parametrize("bad_electrons", [math.nan, math.inf, -math.inf])
def test_electrons_to_voltage_rejects_non_finite_electrons(bad_electrons: float) -> None:
    with pytest.raises(ValueError, match="electrons"):
        electrons_to_voltage(np.array([1.0, bad_electrons]), 100.0)


def test_electrons_to_voltage_rejects_overflow() -> None:
    """A finite input can still overflow; do not pass infinity to the ADC."""
    with pytest.raises(ValueError, match="finite"):
        electrons_to_voltage(np.array([np.finfo(np.float64).max]), 10.0)


def test_analog_gain_doubles_signal() -> None:
    v = np.array([100.0])
    np.testing.assert_allclose(apply_analog_gain(v, 6.0206), [200.0], rtol=1e-3)


def test_analog_gain_zero_db_is_identity() -> None:
    voltage = np.array([0.0, 1.0, 600_000.0])

    np.testing.assert_array_equal(apply_analog_gain(voltage, 0.0), voltage)


def test_analog_gain_negative_db_attenuates() -> None:
    """The model is a linear dB factor, so attenuation is legal."""
    np.testing.assert_allclose(apply_analog_gain(np.array([100.0]), -6.0206), [50.0], rtol=1e-3)


@pytest.mark.parametrize("bad_gain", [math.nan, math.inf, -math.inf])
def test_analog_gain_rejects_non_finite_gain(bad_gain: float) -> None:
    with pytest.raises(ValueError, match="gain_db"):
        apply_analog_gain(np.array([1.0]), bad_gain)


def test_analog_gain_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="gain_db"):
        apply_analog_gain(np.array([1.0]), 10_000.0)


def test_lcg_hcg_full_well_differs() -> None:
    """Dual conversion gain trades full well for output signal swing.

    At 1500 e- the HCG mode produces three times the voltage of LCG. At
    2500 e- the LCG mode is still linear while HCG has already clipped at its
    2000 e- full well, so both modes reach the same 600 mV saturation level
    but HCG saturates much earlier in the charge domain.
    """
    charge = np.array([1500.0])
    lcg_voltage = electrons_to_voltage(clamp_full_well(charge, 6000.0), 100.0)
    hcg_voltage = electrons_to_voltage(clamp_full_well(charge, 2000.0), 300.0)

    np.testing.assert_allclose(lcg_voltage, [150_000.0])
    np.testing.assert_allclose(hcg_voltage, [450_000.0])

    bright = np.array([2500.0])
    lcg_full = electrons_to_voltage(clamp_full_well(bright, 6000.0), 100.0)
    hcg_full = electrons_to_voltage(clamp_full_well(bright, 2000.0), 300.0)

    np.testing.assert_allclose(lcg_full, [250_000.0])
    np.testing.assert_allclose(hcg_full, [600_000.0])
