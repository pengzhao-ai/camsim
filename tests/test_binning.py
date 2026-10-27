"""Binning is charge aggregation, not image resizing.

A sensor has two physically different ways to trade resolution for signal. In
analog binning the charge from several same-colour pixels is summed on the
sense node *before* it is sampled, so one read noise is added to the combined
charge and the well capacity grows with the number of contributing pixels. In
digital binning each pixel is read and quantized independently and the codes
are averaged afterwards, so read noise is averaged down but quantization noise
and any clipping already happened at full resolution.

These tests pin the distinction, the shape and pitch bookkeeping, the
factor-divides-the-shape contract, and the mask reduction that keeps
defect/PDAF flags aligned with the binned pixels. An implementation that
blindly sums 2x2 mosaic blocks would fold red, green, and blue charge into one
value and lose the CFA pattern; the sensor-level test below would then fail.
"""

from __future__ import annotations

import numpy as np
import pytest

import camsim
from camsim.config.schema import BinningConfig, NoiseConfig, SensorConfig
from camsim.core.spectral import SpectralIrradiance
from camsim.sensor.binning import (
    BinningPlan,
    binning_plan,
    mean_blocks,
    reduce_mask,
    sum_blocks,
)

WAVELENGTHS_NM = np.arange(360.0, 831.0, 5.0)


def make_irradiance(level: float = 1e-5, shape: tuple[int, int] = (32, 32)) -> SpectralIrradiance:
    """A flat spectral cube on the flagship preset grid and pixel pitch."""
    data = np.full((*shape, WAVELENGTHS_NM.size), level, dtype=np.float64)
    return SpectralIrradiance.from_array(data, WAVELENGTHS_NM, "W/m^2/nm", pixel_pitch_um=1.12)


def quiet_config(config: SensorConfig, factor: int, mode: str) -> SensorConfig:
    """Return a preset with fixed-pattern noise disabled and a binning setting.

    Linearity-style comparisons need the fixed spatial pattern to be absent so
    that a sample mean measures charge, not the particular realisation of a
    PRNU/DSNU map.
    """
    noise = NoiseConfig(
        dsnu_e=0.0,
        prnu_fraction=0.0,
        row_noise_e_rms=0.0,
        hot_pixel_dark_multiplier=1.0,
    )
    return config.model_copy(
        update={
            "binning": BinningConfig(factor=factor, mode=mode),
            "noise": noise,
        }
    )


# ---------------------------------------------------------------------------
# Block arithmetic
# ---------------------------------------------------------------------------


def test_binning_plan_halves_shape_and_scales_pitch() -> None:
    plan = binning_plan(BinningConfig(factor=2, mode="analog"), (32, 24))
    assert isinstance(plan, BinningPlan)
    assert plan.factor == 2
    assert plan.mode == "analog"
    assert plan.input_shape == (32, 24)
    assert plan.output_shape == (16, 12)
    assert plan.pixel_pitch_scale == 2


def test_binning_plan_rejects_factor_that_does_not_divide_shape() -> None:
    with pytest.raises(ValueError, match="divide"):
        binning_plan(BinningConfig(factor=3, mode="digital"), (32, 32))


def test_sum_blocks_sums_each_charge_block() -> None:
    plane = np.arange(16, dtype=np.float64).reshape(4, 4)
    summed = sum_blocks(plane, 2)
    np.testing.assert_array_equal(summed, [[10.0, 18.0], [42.0, 50.0]])


def test_mean_blocks_averages_codes() -> None:
    plane = np.arange(16, dtype=np.float64).reshape(4, 4)
    averaged = mean_blocks(plane, 2)
    np.testing.assert_allclose(averaged, [[2.5, 4.5], [10.5, 12.5]])


def test_reduce_mask_ors_each_block() -> None:
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[0, 1] = True
    mask[3, 2] = True
    reduced = reduce_mask(mask, 2)
    np.testing.assert_array_equal(reduced, [[True, False], [False, True]])
    assert reduced.dtype == np.bool_


def test_block_functions_reject_bad_factor() -> None:
    plane = np.zeros((4, 4))
    with pytest.raises(ValueError, match="positive integer"):
        sum_blocks(plane, 0)
    with pytest.raises(ValueError, match="divide"):
        mean_blocks(plane, 3)


# ---------------------------------------------------------------------------
# Sensor-level binning
# ---------------------------------------------------------------------------


def test_binning_analog_vs_digital(sensor_config: SensorConfig) -> None:
    """Analog binning sums charge; digital binning averages post-ADC codes."""
    level = 1e-5
    exposure = 0.01
    analog = camsim.Sensor(quiet_config(sensor_config, 2, "analog"), seed=3).expose(
        make_irradiance(level), exposure, mode="HCG", seed=5
    )
    digital = camsim.Sensor(quiet_config(sensor_config, 2, "digital"), seed=3).expose(
        make_irradiance(level), exposure, mode="HCG", seed=5
    )

    assert analog.data.shape == (16, 16)
    assert digital.data.shape == (16, 16)
    assert analog.pixel_pitch_um == pytest.approx(2.24)
    assert digital.pixel_pitch_um == pytest.approx(2.24)
    assert analog.cfa is digital.cfa

    analog_signal = float(analog.data.mean()) - analog.black_level_dn
    digital_signal = float(digital.data.mean()) - digital.black_level_dn
    # Four native pixels contribute their charge to one analog output pixel,
    # while four digital codes are averaged. The ratio is the binning factor
    # squared, up to shot noise and quantization in the sample means.
    assert analog_signal == pytest.approx(4.0 * digital_signal, rel=0.08)
    # Nothing should saturate at this low signal level.
    assert int(analog.data.max()) < 2**analog.bit_depth - 1


def test_binning_preserves_cfa_tile(sensor_config: SensorConfig) -> None:
    analog = camsim.Sensor(quiet_config(sensor_config, 2, "analog"), seed=3).expose(
        make_irradiance(), 0.01, mode="HCG", seed=5
    )
    signal = analog.data.astype(np.float64) - analog.black_level_dn
    # With a flat spectrum the two green site classes still carry the same
    # channel response, while red and blue differ. A blind mosaic sum would
    # make all four sites equal and betray the CFA pattern. Compare plane means
    # rather than single pixels: PDAF attenuation is sparse and would otherwise
    # dominate a one-pixel difference.
    green_a = float(signal[0::2, 1::2].mean())
    green_b = float(signal[1::2, 0::2].mean())
    red = float(signal[0::2, 0::2].mean())
    blue = float(signal[1::2, 1::2].mean())
    assert abs(green_a - green_b) < 0.1 * green_a
    # The three CFA channels must remain distinguishable after binning; a
    # blind mosaic sum would collapse them to one value.
    spread = max(red, green_a, blue) - min(red, green_a, blue)
    assert spread > 0.05 * max(red, green_a, blue)


def test_binning_masks_follow_binned_pixels(sensor_config: SensorConfig) -> None:
    analog = camsim.Sensor(quiet_config(sensor_config, 2, "analog"), seed=3).expose(
        make_irradiance(), 0.01, mode="HCG", seed=5
    )
    assert analog.defect_mask is not None
    assert analog.pdaf_mask is not None
    assert analog.defect_mask.shape == analog.data.shape
    assert analog.pdaf_mask.shape == analog.data.shape
