"""Tests for ADC quantization, black-level offset, and digital gain.

An ADC maps an analog voltage onto a finite code space. With ``b`` bits there
are ``2**b`` codes; the black level reserves a pedestal at the bottom so that
read noise around zero volts does not clip, leaving
``2**b - 1 - black_level_dn`` codes for signal. One signal code therefore
corresponds to ``full_scale_uv / (2**b - 1 - black_level_dn)`` microvolts, and a
voltage at or above full scale produces the largest code.

Digital gain multiplies the *signal* code after conversion and before the black
level is re-added. Applying it to the final code instead would amplify the
pedestal as if it were light, so a dark frame would brighten when the digital
gain is raised. The tests pin that ordering, the exact black-level and
saturation endpoints, the quantization step, and the clipping behaviour.

Failure artefacts guarded here: a black level counted twice (once in the
mapping and again in the final code), digital gain applied to the pedestal,
saturation off by one code, 8-/12-bit data written into a uint16 container with
out-of-range values, and NaN or infinity reaching an integer cast (undefined
behaviour in NumPy).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from camsim.sensor.adc import adc_quantize
from camsim.sensor.gain import clamp_full_well, electrons_to_voltage

BITS = 10
BLACK = 64
FULL_SCALE_UV = 800_000.0
USABLE_CODES = 2**BITS - 1 - BLACK


def test_black_level_and_saturation() -> None:
    bits, black, fs = 10, 64, 800_000.0
    dn = adc_quantize(np.array([0.0, fs]), bits, fs, black)
    assert dn[0] == black
    assert dn[1] == 2**bits - 1
    assert dn.dtype == np.uint16


def test_quantization_step() -> None:
    """Adjacent signal codes are one quantization step apart."""
    step_uv = FULL_SCALE_UV / USABLE_CODES

    dn = adc_quantize(
        step_uv * np.array([0.0, 0.49, 0.51, 1.49, 1.51]),
        BITS,
        FULL_SCALE_UV,
        BLACK,
    )

    np.testing.assert_array_equal(dn, [64, 64, 65, 65, 66])


def test_digital_gain_preserves_black_level() -> None:
    """Digital gain multiplies signal codes only, never the pedestal."""
    step_uv = FULL_SCALE_UV / USABLE_CODES
    signal_uv = np.array([0.0, 100.0 * step_uv])

    unity = adc_quantize(signal_uv, BITS, FULL_SCALE_UV, BLACK)
    doubled = adc_quantize(signal_uv, BITS, FULL_SCALE_UV, BLACK, digital_gain=2.0)

    assert unity[0] == BLACK
    assert doubled[0] == BLACK
    assert unity[1] == BLACK + 100
    assert doubled[1] == BLACK + 200


def test_adc_clips_at_max() -> None:
    """Full-scale and over-range voltages all saturate at the largest code."""
    dn = adc_quantize(
        np.array([FULL_SCALE_UV, 2.0 * FULL_SCALE_UV, 1e12]),
        BITS,
        FULL_SCALE_UV,
        BLACK,
        digital_gain=4.0,
    )

    np.testing.assert_array_equal(dn, [2**BITS - 1] * 3)


def test_negative_voltage_clips_to_black_level() -> None:
    """Voltage excursions below zero volts cannot produce codes below black."""
    dn = adc_quantize(np.array([-1.0, -FULL_SCALE_UV]), BITS, FULL_SCALE_UV, BLACK)

    np.testing.assert_array_equal(dn, [BLACK, BLACK])


def test_adc_returns_uint16_within_the_bit_depth() -> None:
    voltages = np.linspace(-1e6, 1e6, 1001)

    dn = adc_quantize(voltages, 12, 1e6, 256)

    assert dn.dtype == np.uint16
    assert int(dn.min()) == 256
    assert int(dn.max()) == 2**12 - 1


def test_eight_bit_mode_stays_within_uint8_range() -> None:
    dn = adc_quantize(np.array([0.0, 1e6]), 8, 1e6, 16)

    assert dn.dtype == np.uint16
    assert int(dn[0]) == 16
    assert int(dn[1]) == 255


def test_sixteen_bit_code_space_is_fully_usable() -> None:
    dn = adc_quantize(np.array([0.0, 1e9]), 16, 1e9, 4096)

    assert dn.dtype == np.uint16
    assert int(dn[0]) == 4096
    assert int(dn[1]) == 65535


def test_scalar_voltage_returns_a_zero_dimensional_array() -> None:
    dn = adc_quantize(0.0, BITS, FULL_SCALE_UV, BLACK)

    assert dn.shape == ()
    assert int(dn) == BLACK


def test_full_well_and_adc_clip_independently() -> None:
    """Charge clipping and code clipping are separate limits.

    At LCG's 100 uV/e and 6000 e- full well, the full-well voltage is 600 mV,
    which is only 75 % of the 800 mV ADC scale: the sensor is charge limited
    and lands at code 783. Shrinking the ADC scale to 300 mV makes the same
    illuminated pixel ADC limited instead, saturating at code 1023.
    """
    electrons = clamp_full_well(np.array([7000.0]), 6000.0)
    voltage = electrons_to_voltage(electrons, 100.0)

    charge_limited = adc_quantize(voltage, BITS, FULL_SCALE_UV, BLACK)
    adc_limited = adc_quantize(voltage, BITS, 300_000.0, BLACK)

    assert int(charge_limited[0]) == 783
    assert int(adc_limited[0]) == 2**BITS - 1


@pytest.mark.parametrize("bad_bits", [7, 17, 10.0, True, "10"])
def test_adc_rejects_invalid_bit_depths(bad_bits: object) -> None:
    with pytest.raises(ValueError, match="bits"):
        adc_quantize(np.array([0.0]), bad_bits, FULL_SCALE_UV, BLACK)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_black", [-1, 1024, 10.5, True])
def test_adc_rejects_invalid_black_levels(bad_black: object) -> None:
    with pytest.raises(ValueError, match="black_level_dn"):
        adc_quantize(np.array([0.0]), BITS, FULL_SCALE_UV, bad_black)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_full_scale", [0.0, -1.0, math.nan, math.inf])
def test_adc_rejects_invalid_full_scale(bad_full_scale: float) -> None:
    with pytest.raises(ValueError, match="full_scale_uv"):
        adc_quantize(np.array([0.0]), BITS, bad_full_scale, BLACK)


@pytest.mark.parametrize("bad_digital_gain", [0.0, -1.0, math.nan, math.inf])
def test_adc_rejects_invalid_digital_gain(bad_digital_gain: float) -> None:
    with pytest.raises(ValueError, match="digital_gain"):
        adc_quantize(np.array([0.0]), BITS, FULL_SCALE_UV, BLACK, digital_gain=bad_digital_gain)


@pytest.mark.parametrize("bad_voltage", [math.nan, math.inf, -math.inf])
def test_adc_rejects_non_finite_voltage(bad_voltage: float) -> None:
    with pytest.raises(ValueError, match="voltage_uv"):
        adc_quantize(np.array([0.0, bad_voltage]), BITS, FULL_SCALE_UV, BLACK)
