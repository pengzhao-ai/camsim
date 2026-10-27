"""End-to-end tests for the sensor signal chain.

A sensor is the place where radiometry becomes data. The tests here follow the
chain in physical order and pin the properties that make the simulation
trustworthy rather than merely plausible:

* **Linearity.** Doubling irradiance or exposure must double the mean signal
  while no stage clips. Fixed-pattern noise is disabled for these checks so a
  spatial map is not mistaken for a gain error.
* **Photon statistics.** Across repeated frames, temporal variance must grow
  with the mean with slope one in electrons, which only holds if the sampler is
  Poisson and the DN-to-electron scale is right.
* **Determinism.** A per-call seed reproduces a frame bit for bit regardless of
  the calls around it, while the fixed-pattern maps stay tied to the preset's
  defect seed.
* **Units and external contract.** Irradiance in W/m2/nm and in photons is
  accepted; a cube on a different pitch, wavelength grid, or spatial grid is
  rejected with a precise message instead of being silently resampled.
* **Colour reconstruction.** Each CFA site must use its own channel response, so
  the RGGB mosaic shows the documented per-channel balance and a mono part
  collects the bare-silicon response.
* **Non-idealities.** Crosstalk matrices conserve channel sums, blooming
  spreads charge from saturated pixels without NaN, and no RAW code can be
  invalid even at absurd irradiance.

The flat-spectrum level used by default (1e-4 W/m2/nm) is a low-light
laboratory level: it produces a few hundred electrons per pixel in the
flagship preset. A value such as 100 W/m2/nm would saturate every pixel and
make a linearity check vacuous, which is why the low-signal helpers here are
explicit about their units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import camsim
from camsim.config.schema import DefectConfig, NoiseConfig, PDAFConfig, SensorConfig
from camsim.core.constants import C_LIGHT, H_PLANCK
from camsim.core.frames import CFAPattern, RawFrame
from camsim.core.spectral import SpectralIrradiance
from camsim.data import data_path
from camsim.sensor import Sensor
from camsim.sensor.cfa import cfa_masks, spectral_projection
from camsim.sensor.defects import defect_masks, pdaf_mask
from camsim.sensor.sensor import (
    apply_blooming,
    apply_spatial_crosstalk,
    apply_spectral_crosstalk,
)

WAVELENGTHS_NM = np.arange(360.0, 831.0, 5.0)
LOW_LEVEL = 1e-4  # W/m^2/nm: a few hundred electrons in a 10 ms exposure


def make_irr(
    level: float = LOW_LEVEL,
    shape: tuple[int, int] = (32, 32),
    unit: str = "W/m^2/nm",
    pixel_pitch_um: float | None = 1.12,
    wavelengths_nm: np.ndarray | None = None,
) -> SpectralIrradiance:
    """Return a flat spectral cube on the flagship grid."""
    grid = WAVELENGTHS_NM if wavelengths_nm is None else wavelengths_nm
    data = np.full((*shape, grid.size), level, dtype=np.float64)
    return SpectralIrradiance.from_array(data, grid, unit, pixel_pitch_um=pixel_pitch_um)


def quiet_sensor(config: SensorConfig, seed: int = 1) -> Sensor:
    """A sensor with fixed-pattern noise removed for linearity measurements."""
    noise = NoiseConfig(
        dsnu_e=0.0,
        prnu_fraction=0.0,
        row_noise_e_rms=0.0,
        hot_pixel_dark_multiplier=1.0,
    )
    return Sensor(config.model_copy(update={"noise": noise}), seed=seed)


def signal_dn(raw: RawFrame) -> np.ndarray:
    """Return black-level-subtracted codes as float64."""
    return raw.data.astype(np.float64) - raw.black_level_dn


def electrons_per_dn(config: SensorConfig, mode: str = "HCG", analog_gain_db: float = 0.0) -> float:
    """Convert one ADC code to input-referred electrons for a gain mode."""
    usable = (1 << config.adc.bits) - 1 - config.adc.black_level_dn
    uv_per_dn = config.adc.full_scale_uv / usable
    analog_factor = 10.0 ** (analog_gain_db / 20.0)
    return uv_per_dn / (config.modes[mode].conversion_gain_uv_per_e * analog_factor)  # type: ignore[index]


def expected_signal_e(
    config: SensorConfig,
    level: float,
    exposure_s: float,
) -> dict[str, float]:
    """Independently integrate a flat spectrum through the preset curves.

    This duplicates the physical formula on purpose: the assertion is that the
    sensor's mosaic carries the documented per-channel balance, not merely that
    its own arithmetic is self-consistent.
    """
    grid = np.arange(
        config.spectral.wavelength_nm.start,
        config.spectral.wavelength_nm.stop + 0.5 * config.spectral.wavelength_nm.step,
        config.spectral.wavelength_nm.step,
    )
    projection = spectral_projection(
        data_path(config.spectral.qe_csv),
        data_path(config.spectral.cfa_csv),
        data_path(config.spectral.ir_cut_csv),
        grid,
        config.cfa,
    )
    channels = list(cfa_masks(config.cfa, (2, 2)))
    area_m2 = (config.pixel.pitch_um * 1e-6) ** 2 * min(
        1.0, config.pixel.fill_factor * config.pixel.microlens_gain
    )
    photon_energy = H_PLANCK * C_LIGHT / (grid * 1e-9)
    rate = projection @ (level / photon_energy) * area_m2 * config.spectral.wavelength_nm.step
    return {channel: float(rate[i]) * exposure_s for i, channel in enumerate(channels)}


# ---------------------------------------------------------------------------
# Radiometric linearity and noise
# ---------------------------------------------------------------------------


def test_expose_linear_with_lux(sensor_config: SensorConfig) -> None:
    sensor = quiet_sensor(sensor_config)
    levels = (1e-5, 2e-5, 4e-5, 8e-5)
    means = [
        float(signal_dn(sensor.expose(make_irr(level), 0.01, mode="LCG", seed=1)).mean())
        for level in levels
    ]
    assert np.corrcoef(means, levels)[0, 1] > 0.999
    assert means[1] == pytest.approx(2.0 * means[0], rel=0.05)


def test_exposure_linearity(sensor_config: SensorConfig) -> None:
    sensor = quiet_sensor(sensor_config)
    exposures = (0.002, 0.004, 0.008)
    means = [
        float(signal_dn(sensor.expose(make_irr(1e-5), t, mode="LCG", seed=2)).mean())
        for t in exposures
    ]
    assert np.corrcoef(means, exposures)[0, 1] > 0.999
    assert means[2] == pytest.approx(4.0 * means[0], rel=0.05)


def test_shot_noise_variance_slope(sensor_config: SensorConfig) -> None:
    """Temporal variance must equal the mean in electrons in the shot regime."""
    sensor = quiet_sensor(sensor_config)
    frames = np.stack(
        [
            signal_dn(sensor.expose(make_irr(6.2e-4), 0.01, mode="HCG", seed=100 + index))
            for index in range(16)
        ]
    )
    e_per_dn = electrons_per_dn(sensor_config, "HCG")
    mean_e = float(frames.mean()) * e_per_dn
    variance_e = float(frames.var(axis=0).mean()) * e_per_dn**2
    # Read noise (0.7 e-) and quantization add a small offset, so the slope is
    # allowed a fifteen percent band around the Poisson ideal. The 6.2e-4
    # W/m^2/nm level places the mean near 1000 e- with the conventional Bayer
    # CFA; the narrower passbands collect less light than the superseded broad
    # curves, and 2.8e-4 no longer reached the intended shot-noise regime.
    assert 800.0 < mean_e < 1600.0, "test must stay below HCG full well"
    assert variance_e == pytest.approx(mean_e, rel=0.15)


def test_temperature_override_increases_dark(sensor_config: SensorConfig) -> None:
    sensor = quiet_sensor(sensor_config)
    cold = sensor.expose(make_irr(0.0), 0.1, mode="HCG", temperature_c=25.0, seed=1)
    hot = sensor.expose(make_irr(0.0), 0.1, mode="HCG", temperature_c=60.0, seed=1)
    assert hot.temperature_c == 60.0
    assert float(hot.data.mean()) > float(cold.data.mean()) + 0.5


def test_hot_pixels_multiply_dark_current(sensor_config: SensorConfig) -> None:
    defects = DefectConfig(
        dead_rate=0.0, hot_rate=0.2, stuck_rate=0.0, seed=sensor_config.defects.seed
    )
    noise = NoiseConfig(
        dsnu_e=0.0, prnu_fraction=0.0, row_noise_e_rms=0.0, hot_pixel_dark_multiplier=100.0
    )
    config = sensor_config.model_copy(update={"defects": defects, "noise": noise})
    raw = Sensor(config, seed=1).expose(make_irr(0.0), 0.1, mode="HCG", temperature_c=60.0, seed=3)
    hot = defect_masks(config.defects, (32, 32), seed=config.defects.seed).hot
    assert hot.any()
    signal = signal_dn(raw)
    assert float(signal[hot].mean()) > 20.0 * float(signal[~hot].mean())


# ---------------------------------------------------------------------------
# Determinism and RNG contract
# ---------------------------------------------------------------------------


def test_same_seed_bit_identical(sensor_config: SensorConfig) -> None:
    sensor = camsim.Sensor(sensor_config, seed=1)
    first = sensor.expose(make_irr(), 0.01, mode="HCG", seed=7)
    second = sensor.expose(make_irr(), 0.01, mode="HCG", seed=7)
    np.testing.assert_array_equal(first.data, second.data)


def test_per_call_seed_independent_of_call_order(sensor_config: SensorConfig) -> None:
    sensor = camsim.Sensor(sensor_config, seed=1)
    first = sensor.expose(make_irr(), 0.01, mode="HCG", seed=7).data
    sensor.expose(make_irr(), 0.01, mode="HCG", seed=8)
    third = sensor.expose(make_irr(), 0.01, mode="HCG", seed=7).data
    np.testing.assert_array_equal(first, third)


def test_seeded_calls_do_not_advance_frame_stream(sensor_config: SensorConfig) -> None:
    reference = Sensor(sensor_config, seed=5).expose(make_irr(), 0.01, mode="HCG", seed=None)
    interleaved = Sensor(sensor_config, seed=5)
    interleaved.expose(make_irr(), 0.01, mode="HCG", seed=99)
    after = interleaved.expose(make_irr(), 0.01, mode="HCG", seed=None)
    np.testing.assert_array_equal(reference.data, after.data)


def test_reseed_reproduces_frame_stream(sensor_config: SensorConfig) -> None:
    sensor = Sensor(sensor_config, seed=5)
    first_frame = sensor.expose(make_irr(), 0.01, mode="HCG", seed=None)
    second_frame = sensor.expose(make_irr(), 0.01, mode="HCG", seed=None)
    assert first_frame.frame_index == 0
    assert second_frame.frame_index == 1
    assert not np.array_equal(first_frame.data, second_frame.data)
    sensor.reseed(5)
    repeat_first = sensor.expose(make_irr(), 0.01, mode="HCG", seed=None)
    repeat_second = sensor.expose(make_irr(), 0.01, mode="HCG", seed=None)
    assert repeat_first.frame_index == 0
    assert repeat_second.frame_index == 1
    np.testing.assert_array_equal(first_frame.data, repeat_first.data)
    np.testing.assert_array_equal(second_frame.data, repeat_second.data)


def test_fixed_pattern_masks_stable_across_frames(sensor_config: SensorConfig) -> None:
    sensor = Sensor(sensor_config, seed=1)
    first = sensor.expose(make_irr(), 0.01, mode="HCG", seed=1)
    second = sensor.expose(make_irr(), 0.01, mode="HCG", seed=2)
    assert first.defect_mask is not None
    assert second.defect_mask is not None
    np.testing.assert_array_equal(first.defect_mask, second.defect_mask)
    np.testing.assert_array_equal(first.pdaf_mask, second.pdaf_mask)


def test_sensor_pattern_seed_is_the_preset_defect_seed(sensor_config: SensorConfig) -> None:
    sensor = camsim.Sensor(sensor_config, seed=1)
    assert sensor.pattern_seed == sensor_config.defects.seed


# ---------------------------------------------------------------------------
# External irradiance contract
# ---------------------------------------------------------------------------


def test_external_irradiance_wrong_shape_raises(sensor_config: SensorConfig) -> None:
    """A cube for a different spatial grid must not be resampled implicitly.

    The sensor grid is established by the first frame (a sensor has one fixed
    pixel array); a later cube with another shape is a caller error. The
    original plan snippet expected a fresh sensor to reject 8x8, but a fresh
    sensor has no fixed array yet, so the grid lock is exercised here.
    """
    sensor = camsim.Sensor(sensor_config, seed=1)
    sensor.expose(make_irr(shape=(32, 32)), 0.01, mode="LCG", seed=1)
    wrong = make_irr(shape=(8, 8))
    with pytest.raises(ValueError, match="sensor grid"):
        sensor.expose(wrong, 0.01)


def test_wrong_pixel_pitch_raises(sensor_config: SensorConfig) -> None:
    sensor = camsim.Sensor(sensor_config, seed=1)
    irr = make_irr(pixel_pitch_um=2.0)
    with pytest.raises(ValueError, match="sensor grid"):
        sensor.expose(irr, 0.01)


def test_wavelength_mismatch_raises(sensor_config: SensorConfig) -> None:
    sensor = camsim.Sensor(sensor_config, seed=1)
    irr = make_irr(wavelengths_nm=np.arange(400.0, 801.0, 5.0))
    with pytest.raises(ValueError, match="wavelength"):
        sensor.expose(irr, 0.01)


def test_missing_pixel_pitch_is_accepted(sensor_config: SensorConfig) -> None:
    sensor = camsim.Sensor(sensor_config, seed=1)
    irr = make_irr(pixel_pitch_um=None)
    raw = sensor.expose(irr, 0.01, mode="LCG", seed=1)
    assert raw.pixel_pitch_um == pytest.approx(sensor_config.pixel.pitch_um)


def test_photon_unit_matches_energy_equivalent(sensor_config: SensorConfig) -> None:
    electrons = quiet_sensor(sensor_config)
    photons = quiet_sensor(sensor_config)
    energy_raw = electrons.expose(make_irr(1e-5), 0.01, mode="HCG", seed=4)
    photon_cube = make_irr(1e-5, unit="photons/s/m^2/nm")
    # Convert the same photon density to W/m2/nm for the comparison frame.
    energy_equivalent = photon_cube.to_unit("W/m^2/nm")
    photon_raw = photons.expose(photon_cube, 0.01, mode="HCG", seed=4)
    equivalent_raw = electrons.expose(energy_equivalent, 0.01, mode="HCG", seed=4)
    np.testing.assert_array_equal(photon_raw.data, equivalent_raw.data)
    assert energy_raw.data.shape == photon_raw.data.shape


def test_no_nan_in_raw(sensor_config: SensorConfig) -> None:
    sensor = camsim.Sensor(sensor_config, seed=1)
    for level in (0.0, 1e6):
        raw = sensor.expose(make_irr(level), 0.1, mode="LCG", seed=1)
        assert raw.data.dtype == np.uint16
        assert np.all(raw.data >= 0)
        assert int(raw.data.max()) <= 2**raw.bit_depth - 1
        assert raw.defect_mask is not None and raw.pdaf_mask is not None


# ---------------------------------------------------------------------------
# CFA reconstruction
# ---------------------------------------------------------------------------


def test_rggb_mosaic_pattern(sensor_config: SensorConfig) -> None:
    raw = Sensor(sensor_config, seed=1).expose(make_irr(1e-5), 0.01, mode="HCG", seed=2)
    assert raw.cfa is CFAPattern.RGGB
    signal = signal_dn(raw)
    measured = {
        "R": float(signal[0::2, 0::2].mean()),
        "G": float(np.concatenate([signal[0::2, 1::2].ravel(), signal[1::2, 0::2].ravel()]).mean()),
        "B": float(signal[1::2, 1::2].mean()),
    }
    predicted_e = expected_signal_e(sensor_config, 1e-5, 0.01)
    scale = electrons_per_dn(sensor_config, "HCG")
    for channel, value in measured.items():
        assert value == pytest.approx(predicted_e[channel] / scale, rel=0.12), channel
    # The two green sites share one channel response.
    assert signal[0::2, 1::2].mean() == pytest.approx(signal[1::2, 0::2].mean(), rel=0.05)


def test_mono_output_single_channel(
    mono_sensor_config: SensorConfig, mono_raw_frame: RawFrame
) -> None:
    frame = mono_raw_frame
    assert frame.cfa is CFAPattern.MONO
    assert frame.data.ndim == 2
    assert frame.pdaf_mask is not None
    assert not frame.pdaf_mask.any()
    sensor = Sensor(mono_sensor_config, seed=1)
    raw = sensor.expose(make_irr(1e-5), 0.01, mode="HCG", seed=2)
    measured_e = float(signal_dn(raw).mean()) * electrons_per_dn(mono_sensor_config, "HCG")
    predicted_e = expected_signal_e(mono_sensor_config, 1e-5, 0.01)["M"]
    assert measured_e == pytest.approx(predicted_e, rel=0.12)


def test_mono_collects_more_than_bayer(sensor_config: SensorConfig) -> None:
    mono_config = sensor_config.model_copy(update={"cfa": CFAPattern.MONO})
    mono_e = float(
        signal_dn(
            Sensor(mono_config, seed=1).expose(make_irr(1e-5), 0.01, mode="HCG", seed=2)
        ).mean()
    ) * electrons_per_dn(mono_config, "HCG")
    bayer = expected_signal_e(sensor_config, 1e-5, 0.01)
    assert mono_e > max(bayer.values())


def test_auto_mode_selects_hcg_below_threshold(sensor_config: SensorConfig) -> None:
    sensor = quiet_sensor(sensor_config)
    low = sensor.expose(make_irr(1e-5), 0.01, mode="auto", seed=1)
    high = sensor.expose(make_irr(4e-4), 0.01, mode="auto", seed=1)
    assert low.mode == "HCG"
    assert high.mode == "LCG"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_raw_frame_metadata(sensor_config: SensorConfig) -> None:
    sensor = camsim.Sensor(sensor_config, seed=1)
    raw = sensor.expose(
        make_irr(1e-5),
        0.01,
        mode="HCG",
        analog_gain_db=6.0,
        digital_gain=2.0,
        temperature_c=30.0,
        seed=11,
    )
    assert raw.cfa is CFAPattern.RGGB
    assert raw.bit_depth == sensor_config.adc.bits
    assert raw.black_level_dn == pytest.approx(float(sensor_config.adc.black_level_dn))
    assert raw.exposure_s == pytest.approx(0.01)
    assert raw.analog_gain_db == pytest.approx(6.0)
    assert raw.digital_gain == pytest.approx(2.0)
    assert raw.mode == "HCG"
    assert raw.temperature_c == pytest.approx(30.0)
    assert raw.seed == 11
    assert raw.frame_index >= 0
    assert raw.pixel_pitch_um == pytest.approx(sensor_config.pixel.pitch_um)
    assert raw.defect_mask is not None
    assert raw.pdaf_mask is not None
    assert raw.defect_mask.shape == raw.data.shape
    assert len(raw.sensor_config_hash) == 64
    assert sensor.config is sensor_config


def test_sensor_config_hash_is_stable_and_sensitive(sensor_config: SensorConfig) -> None:
    first = camsim.Sensor(sensor_config, seed=1).expose(make_irr(), 0.01, mode="LCG", seed=1)
    second = camsim.Sensor(sensor_config, seed=2).expose(make_irr(), 0.01, mode="LCG", seed=1)
    assert first.sensor_config_hash == second.sensor_config_hash
    changed = sensor_config.model_copy(update={"auto_hcg_threshold_e": 123.0})
    third = camsim.Sensor(changed, seed=1).expose(make_irr(), 0.01, mode="LCG", seed=1)
    assert third.sensor_config_hash != first.sensor_config_hash


def test_defect_mask_is_union_of_defect_classes(sensor_config: SensorConfig) -> None:
    defects = DefectConfig(dead_rate=0.01, hot_rate=0.01, stuck_rate=0.01, seed=99)
    config = sensor_config.model_copy(update={"defects": defects})
    raw = Sensor(config, seed=1).expose(make_irr(), 0.01, mode="LCG", seed=1)
    masks = defect_masks(config.defects, (32, 32), seed=config.defects.seed)
    expected = masks.dead | masks.hot | masks.stuck
    assert raw.defect_mask is not None
    np.testing.assert_array_equal(raw.defect_mask, expected)


def test_pdaf_mask_uses_the_actual_cfa(sensor_config: SensorConfig) -> None:
    """A GRBG tile moves the green sites, so sparse-green flags must move too."""
    config = sensor_config.model_copy(update={"cfa": CFAPattern.GRBG})
    raw = Sensor(config, seed=1).expose(make_irr(), 0.01, mode="LCG", seed=1)
    expected = pdaf_mask(
        config.pdaf,
        (32, 32),
        seed=config.defects.seed,
        cfa=config.cfa,
    )
    rggb = pdaf_mask(
        config.pdaf,
        (32, 32),
        seed=config.defects.seed,
        cfa=CFAPattern.RGGB,
    )
    assert raw.pdaf_mask is not None
    np.testing.assert_array_equal(raw.pdaf_mask, expected)
    assert not np.array_equal(expected, rggb)


# ---------------------------------------------------------------------------
# Crosstalk, blooming, spatial mixing
# ---------------------------------------------------------------------------


def test_crosstalk_row_sums() -> None:
    rng = np.random.default_rng(0)
    channels = {
        "R": rng.random((6, 5)),
        "G": rng.random((6, 5)),
        "B": rng.random((6, 5)),
    }
    # Row-normalised matrices fix the neutral axis: equal channel values must
    # map to equal channel values even when the off-diagonal leakage is large.
    neutral = {"R": np.ones((4, 4)), "G": np.ones((4, 4)), "B": np.ones((4, 4))}
    neutral_matrix = [
        [0.90, 0.05, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.05, 0.90],
    ]
    neutral_out = apply_spectral_crosstalk(neutral, neutral_matrix)
    for plane in neutral_out.values():
        np.testing.assert_allclose(plane, 1.0, rtol=0.0, atol=1e-12)
    # A doubly stochastic matrix also conserves total channel charge.
    conservative = [
        [0.90, 0.05, 0.05],
        [0.10, 0.80, 0.10],
        [0.00, 0.15, 0.85],
    ]
    mixed = apply_spectral_crosstalk(channels, conservative)
    np.testing.assert_allclose(sum(mixed.values()), sum(channels.values()), rtol=1e-12, atol=1e-12)
    assert list(mixed) == ["R", "G", "B"]
    with pytest.raises(ValueError, match="row"):
        apply_spectral_crosstalk(channels, [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_crosstalk_requires_compatible_channel_count() -> None:
    with pytest.raises(ValueError, match="channel"):
        apply_spectral_crosstalk({"R": np.ones((2, 2)), "C": np.ones((2, 2))}, np.eye(3))


def test_rccc_accepts_identity_but_rejects_three_channel_crosstalk(
    sensor_config: SensorConfig, tmp_path: Path
) -> None:
    """RCCC has two channels; only the no-crosstalk identity is compatible."""
    cfa_path = tmp_path / "cfa_rccc.csv"
    cfa_path.write_text(
        "wavelength_nm,value\n360,0.80\n830,0.80\n",
        encoding="utf-8",
    )
    spectral = sensor_config.spectral.model_copy(update={"cfa_csv": str(cfa_path)})
    config = sensor_config.model_copy(update={"cfa": CFAPattern.RCCC, "spectral": spectral})
    raw = Sensor(config, seed=1).expose(make_irr(), 0.01, mode="LCG", seed=1)
    assert raw.cfa is CFAPattern.RCCC
    assert raw.data.shape == (32, 32)

    non_identity = sensor_config.spectral.model_copy(
        update={
            "spectral_crosstalk": [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.05, 0.05, 0.90],
            ]
        }
    )
    incompatible = sensor_config.model_copy(
        update={"cfa": CFAPattern.RCCC, "spectral": non_identity}
    )
    with pytest.raises(ValueError, match="channel"):
        Sensor(incompatible, seed=1).expose(make_irr(), 0.01, mode="LCG", seed=1)


def test_mono_skips_spectral_crosstalk(mono_sensor_config: SensorConfig) -> None:
    non_identity = mono_sensor_config.spectral.model_copy(
        update={
            "spectral_crosstalk": [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.05, 0.05, 0.90],
            ]
        }
    )
    config = mono_sensor_config.model_copy(update={"spectral": non_identity})
    raw = Sensor(config, seed=1).expose(make_irr(), 0.01, mode="LCG", seed=1)
    assert raw.cfa is CFAPattern.MONO


def test_spatial_crosstalk_preserves_charge_and_spreads() -> None:
    mosaic = np.zeros((7, 7), dtype=np.float64)
    mosaic[3, 3] = 1.0
    kernel = np.array([[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]], dtype=np.float64)
    mixed = apply_spatial_crosstalk(mosaic, kernel)
    assert mixed[3, 3] == pytest.approx(0.0)
    assert mixed[2, 3] == pytest.approx(0.25)
    assert mixed.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="sum"):
        apply_spatial_crosstalk(mosaic, kernel * 2.0)


def _quiet_config(
    config: SensorConfig,
    cfa: CFAPattern = CFAPattern.MONO,
) -> SensorConfig:
    """Return a preset with every stochastic and fixed-pattern source off.

    The spatial-crosstalk wiring tests isolate charge mixing: read noise, dark
    current, DSNU, PRNU, row noise, defects, and PDAF are disabled so every DN
    in the frame is attributable to the Poisson sample convolved with the
    kernel. The Poisson sampler itself is deterministic for a per-call seed.
    """
    modes = {
        name: mode.model_copy(update={"read_noise_e_rms": 0.0, "dark_current_e_per_s": 0.0})
        for name, mode in config.modes.items()
    }
    return config.model_copy(
        update={
            "cfa": cfa,
            "modes": modes,
            "noise": NoiseConfig(
                dsnu_e=0.0,
                prnu_fraction=0.0,
                row_noise_e_rms=0.0,
                hot_pixel_dark_multiplier=1.0,
            ),
            "defects": DefectConfig(dead_rate=0.0, hot_rate=0.0, stuck_rate=0.0, seed=1234),
            "pdaf": PDAFConfig(fraction=0.0, qe_factor=0.5, pattern="none"),
        }
    )


def _single_pixel_irradiance(
    level: float,
    shape: tuple[int, int] = (9, 9),
) -> SpectralIrradiance:
    """Return a dark cube with one broadband bright pixel at the array centre."""
    data = np.zeros((*shape, WAVELENGTHS_NM.size), dtype=np.float64)
    data[shape[0] // 2, shape[1] // 2] = level
    return SpectralIrradiance.from_array(
        data,
        WAVELENGTHS_NM,
        "W/m^2/nm",
        pixel_pitch_um=1.12,
    )


def _narrowband_irradiance(
    level: float,
    wavelength_nm: float = 450.0,
    shape: tuple[int, int] = (9, 9),
) -> SpectralIrradiance:
    """Return a single-pixel cube that excites only one wavelength sample."""
    data = np.zeros((*shape, WAVELENGTHS_NM.size), dtype=np.float64)
    index = int(np.flatnonzero(wavelength_nm == WAVELENGTHS_NM)[0])
    data[shape[0] // 2, shape[1] // 2, index] = level
    return SpectralIrradiance.from_array(
        data,
        WAVELENGTHS_NM,
        "W/m^2/nm",
        pixel_pitch_um=1.12,
    )


def test_spatial_crosstalk_kernel_is_validated(sensor_config: SensorConfig) -> None:
    """A kernel must be rectangular, non-negative, finite, and sum to one."""
    payload = sensor_config.model_dump(mode="json")
    for bad_kernel in (
        [],
        [[]],
        [[0.9, 0.2], [0.0, 0.0]],
        [[-0.1, 0.6, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.5, 0.5, 0.0], [0.0, 0.5]],
        [[float("nan"), 1.0]],
        [[float("inf"), 0.0]],
    ):
        with pytest.raises(ValidationError):
            SensorConfig.model_validate({**payload, "spatial_crosstalk": bad_kernel})


def test_sensor_applies_spatial_crosstalk_before_read_noise(
    sensor_config: SensorConfig,
) -> None:
    """A configured kernel mixes neighbouring pixels and preserves interior charge.

    The kernel is applied per CFA channel plane after spectral crosstalk and
    before read noise, so an interior single-pixel impulse is redistributed by
    the kernel while the total charge stays on the frame. The centre weight is
    0.6 and the four edge neighbours carry 0.1 each, summing to one.
    """
    kernel = [
        [0.0, 0.1, 0.0],
        [0.1, 0.6, 0.1],
        [0.0, 0.1, 0.0],
    ]
    quiet = _quiet_config(sensor_config)
    plain = quiet.model_copy(update={"spatial_crosstalk": None})
    spread = quiet.model_copy(update={"spatial_crosstalk": kernel})
    irradiance = _single_pixel_irradiance(1e-4)

    reference = signal_dn(Sensor(plain, seed=1).expose(irradiance, 0.01, mode="HCG", seed=7))
    mixed = signal_dn(Sensor(spread, seed=1).expose(irradiance, 0.01, mode="HCG", seed=7))

    centre = (4, 4)
    assert reference[centre] > 0.0
    assert float(reference.sum()) == pytest.approx(float(reference[centre]))
    # The centre keeps most of the charge; each edge neighbour receives a
    # quarter of the 0.4 total that leaves it.
    assert mixed[centre] == pytest.approx(0.6 * reference[centre], rel=0.03)
    for neighbour in ((3, 4), (5, 4), (4, 3), (4, 5)):
        assert mixed[neighbour] == pytest.approx(0.1 * reference[centre], rel=0.05)
    assert float(mixed.sum()) == pytest.approx(float(reference.sum()), rel=0.03)


def test_spatial_crosstalk_on_bayer_planes_does_not_mix_colours(
    sensor_config: SensorConfig,
) -> None:
    """A red impulse stays in the red plane and does not land on green/blue sites.

    The kernel is applied to each R/G/B plane independently, never to the
    mosaicked frame. A narrowband 690 nm impulse at the centre (a red site in
    RGGB) excites the red plane far more than green or blue. After a symmetric
    3x3 diffusion kernel, the centre keeps the 0.2 centre weight and the eight
    neighbouring sites would carry 0.1 each only if the mosaic were convolved
    directly. With per-plane application, a cardinal green site reads the green
    plane (about 1/20 of the red response at 690 nm) and a diagonal blue site
    reads the nearly dark blue plane, so both must stay below 5 % of the mixed
    red centre.
    """
    kernel = [
        [0.1, 0.1, 0.1],
        [0.1, 0.2, 0.1],
        [0.1, 0.1, 0.1],
    ]
    quiet = _quiet_config(sensor_config, cfa=CFAPattern.RGGB)
    plain = quiet.model_copy(update={"spatial_crosstalk": None})
    spread = quiet.model_copy(update={"spatial_crosstalk": kernel})
    # 690 nm is passed by the red CFA and strongly rejected by green/blue.
    irradiance = _narrowband_irradiance(5e-2, wavelength_nm=690.0)
    reference = signal_dn(Sensor(plain, seed=1).expose(irradiance, 0.01, mode="HCG", seed=7))
    signal = signal_dn(Sensor(spread, seed=1).expose(irradiance, 0.01, mode="HCG", seed=7))

    # (4, 4) is red in RGGB; the four diagonal neighbours are blue sites and
    # the four cardinal neighbours are green sites.
    centre = (4, 4)
    blue_sites = ((3, 3), (3, 5), (5, 3), (5, 5))
    green_sites = ((3, 4), (5, 4), (4, 3), (4, 5))
    assert reference[centre] > 0.0
    assert float(reference.sum()) == pytest.approx(float(reference[centre]))
    mixed_centre = float(signal[centre])
    assert mixed_centre == pytest.approx(0.2 * float(reference[centre]), rel=0.03)
    for sites in (green_sites, blue_sites):
        worst = max(float(signal[site]) for site in sites)
        assert worst < 0.05 * mixed_centre, (
            "spatial crosstalk leaked red charge into a non-red CFA site; "
            "the kernel must be applied per channel plane, not to the mosaic"
        )


def test_blooming_spreads_saturated_charge() -> None:
    full_well = 1000.0
    charge = np.zeros((5, 5), dtype=np.float64)
    charge[2, 2] = 2.0 * full_well
    bloomed = apply_blooming(charge, full_well, fraction=0.5, iterations=1)
    assert np.all(np.isfinite(bloomed))
    assert bloomed[2, 2] == pytest.approx(full_well)
    for y, x in ((1, 2), (3, 2), (2, 1), (2, 3)):
        assert bloomed[y, x] == pytest.approx(0.125 * full_well)
    assert bloomed.sum() == pytest.approx(1.5 * full_well)
    assert float(bloomed.max()) <= full_well


def test_blooming_disabled_is_a_clip() -> None:
    full_well = 100.0
    charge = np.full((3, 3), 250.0)
    clipped = apply_blooming(charge, full_well, fraction=0.0, iterations=0)
    np.testing.assert_allclose(clipped, full_well)


# ---------------------------------------------------------------------------
# Physical defect behaviour
# ---------------------------------------------------------------------------


def _defect_config(
    *,
    dead_rate: float = 0.0,
    hot_rate: float = 0.0,
    stuck_rate: float = 0.0,
) -> DefectConfig:
    """Return a defect configuration with high rates for an end-to-end check."""
    return DefectConfig(dead_rate=dead_rate, hot_rate=hot_rate, stuck_rate=stuck_rate, seed=1234)


def _defect_sensor(
    sensor_config: SensorConfig,
    defects: DefectConfig,
) -> Sensor:
    """Return a sensor with PDAF disabled and the requested defect rates."""
    pdaf = PDAFConfig(fraction=0.0, qe_factor=0.5, pattern="none")
    config = sensor_config.model_copy(update={"defects": defects, "pdaf": pdaf})
    return Sensor(config, seed=1)


def test_dead_pixels_read_near_black_not_normal_scene_signal(sensor_config: SensorConfig) -> None:
    defects = _defect_config(dead_rate=0.05)
    sensor = _defect_sensor(sensor_config, defects)
    raw = sensor.expose(make_irr(LOW_LEVEL), 0.01, mode="LCG", seed=7)
    masks = defect_masks(defects, raw.data.shape, seed=defects.seed)
    assert masks.dead.any()

    dead_codes = raw.data[masks.dead].astype(np.float64)
    healthy_codes = raw.data[~masks.dead].astype(np.float64)
    assert float(np.max(dead_codes)) <= raw.black_level_dn + 2.0
    assert float(np.median(healthy_codes)) > raw.black_level_dn + 10.0
    assert float(np.median(healthy_codes)) - float(np.median(dead_codes)) > 10.0
    assert raw.hot_mask is not None and not raw.hot_mask.any()
    assert raw.defect_mask is not None
    np.testing.assert_array_equal(raw.defect_mask, masks.dead)


def test_stuck_pixels_are_pinned_to_the_adc_white_code(sensor_config: SensorConfig) -> None:
    defects = _defect_config(stuck_rate=0.02)
    sensor = _defect_sensor(sensor_config, defects)
    raw = sensor.expose(make_irr(LOW_LEVEL), 0.01, mode="LCG", seed=7)
    masks = defect_masks(defects, raw.data.shape, seed=defects.seed)
    assert masks.stuck.any()

    maximum = 2**raw.bit_depth - 1
    np.testing.assert_array_equal(raw.data[masks.stuck], np.full(int(masks.stuck.sum()), maximum))
    assert raw.defect_mask is not None
    np.testing.assert_array_equal(raw.defect_mask, masks.stuck)
    assert raw.hot_mask is not None and not raw.hot_mask.any()


def test_hot_pixels_differ_from_healthy_pixels_under_dark_current(
    sensor_config: SensorConfig,
) -> None:
    defects = _defect_config(hot_rate=0.05)
    sensor = _defect_sensor(sensor_config, defects)
    raw = sensor.expose(
        make_irr(0.0),
        0.1,
        mode="LCG",
        temperature_c=70.0,
        seed=7,
    )
    masks = defect_masks(defects, raw.data.shape, seed=defects.seed)
    assert masks.hot.any()

    hot_codes = raw.data[masks.hot].astype(np.float64)
    healthy_codes = raw.data[~masks.hot].astype(np.float64)
    assert float(np.median(hot_codes)) > float(np.median(healthy_codes)) + 20.0
    assert raw.hot_mask is not None
    np.testing.assert_array_equal(raw.hot_mask, masks.hot)
    assert raw.defect_mask is not None
    np.testing.assert_array_equal(raw.defect_mask, masks.hot)


@pytest.mark.parametrize(
    "defects",
    [
        _defect_config(dead_rate=0.05),
        _defect_config(stuck_rate=0.02),
        _defect_config(hot_rate=0.05),
    ],
    ids=["dead", "stuck", "hot"],
)
def test_dpc_corrects_every_flagged_defect_class(
    sensor_config: SensorConfig,
    defects: DefectConfig,
) -> None:
    from camsim.config.schema import DPCConfig
    from camsim.isp.dpc import correct_defects

    sensor = _defect_sensor(sensor_config, defects)
    raw = sensor.expose(
        make_irr(LOW_LEVEL),
        0.01,
        mode="LCG",
        temperature_c=70.0 if defects.hot_rate else 25.0,
        seed=7,
    )
    assert raw.defect_mask is not None and raw.defect_mask.any()
    corrected = correct_defects(raw, DPCConfig(enabled=True, method="neighbor"))
    flagged = raw.defect_mask
    original = raw.data.astype(np.float32)
    changed = corrected[flagged] != original[flagged]
    assert float(np.mean(changed)) > 0.95
    np.testing.assert_array_equal(corrected[~flagged], original[~flagged])


def test_hot_mask_is_reduced_with_binning(sensor_config: SensorConfig) -> None:
    from camsim.config.schema import BinningConfig

    defects = _defect_config(hot_rate=0.05, dead_rate=0.01)
    pdaf = PDAFConfig(fraction=0.0, qe_factor=0.5, pattern="none")
    config = sensor_config.model_copy(
        update={"defects": defects, "pdaf": pdaf, "binning": BinningConfig(factor=2, mode="analog")}
    )
    raw = Sensor(config, seed=1).expose(make_irr(LOW_LEVEL), 0.01, mode="LCG", seed=7)
    assert raw.data.shape == (16, 16)
    assert raw.hot_mask is not None and raw.hot_mask.shape == raw.data.shape
    assert raw.defect_mask is not None and raw.defect_mask.shape == raw.data.shape
    assert bool(np.all(raw.defect_mask[raw.hot_mask]))


def test_sensor_memory_guard_rejects_an_oversized_working_set(
    sensor_config: SensorConfig,
) -> None:
    sensor = Sensor(sensor_config, seed=1)
    with pytest.raises(ValueError, match="peak memory"):
        sensor.expose(make_irr(1e-5), 0.01, mode="LCG", seed=1, max_memory_bytes=1)
    # The rejected call must not lock the native grid or advance the frame stream.
    raw = sensor.expose(make_irr(1e-5), 0.01, mode="LCG", seed=1)
    assert raw.data.shape == (32, 32)
