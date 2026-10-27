"""Tests for shot noise, dark current, fixed-pattern noise, and row noise.

Photon arrival and thermally generated charge are discrete Poisson processes,
so an electron count measured over one exposure has variance equal to its mean
in the shot-noise-limited regime. That square-root law is the reason a bright
pixel is proportionally quieter than a dark one and the reason SNR is usually
quoted in decibels. The tests here verify the sampler's first two moments on a
large synthetic population, then verify the non-ideal sensor terms that are
*not* temporal: DSNU is a fixed offset pattern, PRNU is a fixed multiplicative
responsivity pattern, and row noise is a per-row common-mode offset that
produces horizontal banding rather than per-pixel grain.

Determinism is part of the contract. Every stochastic helper accepts an
explicit ``numpy.random.Generator`` or a non-negative seed; none touches
NumPy's global RNG, and identical seeds must reproduce bit-identical maps.

Failure artefacts guarded here: a Gaussian approximation to shot noise would
produce negative electrons at low counts; a dark-current model with the wrong
reference temperature or doubling interval over- or under-predicts long
exposures by orders of magnitude; a PRNU value expressed in percent (2 instead
of 0.02) multiplies the signal by three; and a per-pixel implementation of row
noise removes the banding the term is meant to model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from camsim.config.schema import TemperatureConfig
from camsim.sensor.noise import (
    FPNConfig,
    FPPMaps,
    dark_current_e_per_s,
    fixed_pattern_maps,
    poisson_electrons,
    row_noise,
    sample_dark_current,
)

SHAPE_2D = (3, 7)


# --------------------------------------------------------------------------- #
# Shot noise
# --------------------------------------------------------------------------- #


def test_poisson_variance_matches_mean() -> None:
    """Shot noise is Poisson: variance equals the mean electron count."""
    rng = np.random.default_rng(0)

    x = poisson_electrons(np.full(200_000, 100.0), rng)

    assert x.mean() == pytest.approx(100.0, abs=0.5)
    assert x.var() == pytest.approx(100.0, abs=2.0)


def test_poisson_preserves_shape_and_returns_integers() -> None:
    x = poisson_electrons(np.full(SHAPE_2D, 12.0), np.random.default_rng(1))

    assert x.shape == SHAPE_2D
    assert np.issubdtype(x.dtype, np.integer)


def test_poisson_accepts_a_scalar_mean() -> None:
    x = poisson_electrons(12.0, np.random.default_rng(2))

    assert x.shape == ()
    assert int(x) >= 0


def test_poisson_zero_mean_is_all_zero() -> None:
    x = poisson_electrons(np.zeros(SHAPE_2D), np.random.default_rng(3))

    np.testing.assert_array_equal(x, np.zeros(SHAPE_2D, dtype=np.int64))


def test_poisson_is_reproducible_for_the_same_seed() -> None:
    a = poisson_electrons(np.full(SHAPE_2D, 9.0), np.random.default_rng(4))
    b = poisson_electrons(np.full(SHAPE_2D, 9.0), np.random.default_rng(4))

    np.testing.assert_array_equal(a, b)


def test_poisson_differs_for_a_different_seed() -> None:
    a = poisson_electrons(np.full(64, 9.0), np.random.default_rng(4))
    b = poisson_electrons(np.full(64, 9.0), np.random.default_rng(5))

    assert not np.array_equal(a, b)


@pytest.mark.parametrize("bad_mean", [-1.0, -1e-9, np.nan, np.inf, -np.inf])
def test_poisson_rejects_invalid_means(bad_mean: float) -> None:
    with pytest.raises(ValueError, match=r"finite|non-negative"):
        poisson_electrons(np.array([10.0, bad_mean]), np.random.default_rng(0))


def test_poisson_rejects_non_numeric_means() -> None:
    with pytest.raises(ValueError, match="real scalar or array"):
        poisson_electrons(np.array(["ten"]), np.random.default_rng(0))  # type: ignore[arg-type]


def test_poisson_rejects_a_non_generator_rng() -> None:
    with pytest.raises(ValueError, match="Generator"):
        poisson_electrons(1.0, 42)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Dark current
# --------------------------------------------------------------------------- #


@pytest.fixture
def temperature_config() -> TemperatureConfig:
    """The reference 25 C, 7 K doubling model used by the flagship preset."""
    return TemperatureConfig(reference_c=25.0, default_c=25.0, dark_current_doubling_k=7.0)


def test_dark_current_doubling(temperature_config: TemperatureConfig) -> None:
    """A 7 K rise above the reference doubles the dark-current rate."""
    assert dark_current_e_per_s(1.0, 32.0, temperature_config) == pytest.approx(2.0, rel=1e-3)


def test_dark_current_halves_below_the_reference(
    temperature_config: TemperatureConfig,
) -> None:
    assert dark_current_e_per_s(4.0, 18.0, temperature_config) == pytest.approx(2.0, rel=1e-3)


def test_dark_current_at_reference_temperature_is_the_reference_value(
    temperature_config: TemperatureConfig,
) -> None:
    assert dark_current_e_per_s(3.5, 25.0, temperature_config) == pytest.approx(3.5)


def test_zero_reference_dark_current_stays_zero(temperature_config: TemperatureConfig) -> None:
    assert dark_current_e_per_s(0.0, 60.0, temperature_config) == 0.0


def test_dark_current_rejects_a_negative_reference(
    temperature_config: TemperatureConfig,
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        dark_current_e_per_s(-1.0, 25.0, temperature_config)


@pytest.mark.parametrize("bad_temperature", [np.nan, np.inf, -np.inf])
def test_dark_current_rejects_non_finite_temperatures(
    temperature_config: TemperatureConfig,
    bad_temperature: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        dark_current_e_per_s(1.0, bad_temperature, temperature_config)


def test_dark_current_rejects_overflow_at_absurd_temperatures(
    temperature_config: TemperatureConfig,
) -> None:
    with pytest.raises(ValueError, match="overflow"):
        dark_current_e_per_s(1.0, 100_000.0, temperature_config)


def test_dark_current_rejects_a_non_real_reference(
    temperature_config: TemperatureConfig,
) -> None:
    with pytest.raises(ValueError, match="real number"):
        dark_current_e_per_s("1.0", 25.0, temperature_config)  # type: ignore[arg-type]


def test_dark_current_rejects_a_non_finite_result(
    temperature_config: TemperatureConfig,
) -> None:
    with pytest.raises(ValueError, match="not finite"):
        dark_current_e_per_s(1e308, 32.0, temperature_config)


def test_dark_current_rejects_an_invalid_doubling_interval() -> None:
    config = TemperatureConfig.model_construct(
        reference_c=25.0,
        default_c=25.0,
        dark_current_doubling_k=0.0,
    )

    with pytest.raises(ValueError, match="positive"):
        dark_current_e_per_s(1.0, 25.0, config)


def test_dark_current_rejects_the_wrong_config_type() -> None:
    with pytest.raises(ValueError, match="TemperatureConfig"):
        dark_current_e_per_s(1.0, 25.0, object())  # type: ignore[arg-type]


def test_dark_current_uses_the_configured_reference_temperature() -> None:
    config = TemperatureConfig(reference_c=20.0, default_c=25.0, dark_current_doubling_k=7.0)

    assert dark_current_e_per_s(1.0, 27.0, config) == pytest.approx(2.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# Dark-current sampling
# --------------------------------------------------------------------------- #


def test_sample_dark_current_is_poisson() -> None:
    rng = np.random.default_rng(6)

    x = sample_dark_current(np.full(200_000, 40.0), rng)

    assert x.mean() == pytest.approx(40.0, abs=0.5)
    assert x.var() == pytest.approx(40.0, abs=2.0)


def test_sample_dark_current_zero_mean_is_all_zero() -> None:
    x = sample_dark_current(np.zeros(SHAPE_2D), np.random.default_rng(7))

    np.testing.assert_array_equal(x, np.zeros(SHAPE_2D, dtype=np.int64))


def test_sample_dark_current_rejects_negative_means() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        sample_dark_current(np.array([1.0, -0.5]), np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# FPN configuration and maps
# --------------------------------------------------------------------------- #


def test_fpn_config_defaults_are_off() -> None:
    config = FPNConfig()

    assert config.dsnu_e == 0.0
    assert config.prnu_fraction == 0.0
    assert config.row_noise_e_rms == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dsnu_e", -1.0),
        ("dsnu_e", math.nan),
        ("prnu_fraction", -0.01),
        ("prnu_fraction", 1.0),
        ("row_noise_e_rms", -0.5),
        ("row_noise_e_rms", math.inf),
    ],
)
def test_fpn_config_rejects_invalid_values(field: str, value: float) -> None:
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        FPNConfig(**kwargs)  # type: ignore[arg-type]


def test_fixed_pattern_reproducible() -> None:
    config = FPNConfig(dsnu_e=2.0, prnu_fraction=0.01)

    a = fixed_pattern_maps(config, (32, 32), seed=11)
    b = fixed_pattern_maps(config, (32, 32), seed=11)

    assert isinstance(a, FPPMaps)
    np.testing.assert_array_equal(a.dsnu_e, b.dsnu_e)
    np.testing.assert_array_equal(a.prnu_fraction, b.prnu_fraction)


def test_fixed_pattern_different_seed_changes_both_maps() -> None:
    config = FPNConfig(dsnu_e=2.0, prnu_fraction=0.01)

    a = fixed_pattern_maps(config, (32, 32), seed=11)
    b = fixed_pattern_maps(config, (32, 32), seed=12)

    assert not np.array_equal(a.dsnu_e, b.dsnu_e)
    assert not np.array_equal(a.prnu_fraction, b.prnu_fraction)


def test_fixed_pattern_shape_and_dtype() -> None:
    maps = fixed_pattern_maps(FPNConfig(dsnu_e=1.0, prnu_fraction=0.01), (9, 5), seed=0)

    for array in (maps.dsnu_e, maps.prnu_fraction):
        assert array.shape == (9, 5)
        assert array.dtype == np.float64


def test_fixed_pattern_zero_config_returns_zero_maps() -> None:
    maps = fixed_pattern_maps(FPNConfig(), (16, 16), seed=3)

    np.testing.assert_array_equal(maps.dsnu_e, np.zeros((16, 16)))
    np.testing.assert_array_equal(maps.prnu_fraction, np.zeros((16, 16)))


def test_dsnu_standard_deviation_matches_the_config() -> None:
    maps = fixed_pattern_maps(FPNConfig(dsnu_e=3.0), (256, 256), seed=11)

    assert float(maps.dsnu_e.mean()) == pytest.approx(0.0, abs=0.1)
    assert float(maps.dsnu_e.std()) == pytest.approx(3.0, rel=0.05)


def test_prnu_is_a_dimensionless_multiplicative_deviation() -> None:
    maps = fixed_pattern_maps(FPNConfig(prnu_fraction=0.02), (256, 256), seed=11)

    assert float(maps.prnu_fraction.mean()) == pytest.approx(0.0, abs=1e-3)
    assert float(maps.prnu_fraction.std()) == pytest.approx(0.02, rel=0.05)


def test_prnu_multiplies_the_signal() -> None:
    """A flat signal acquires the configured relative standard deviation."""
    maps = fixed_pattern_maps(FPNConfig(prnu_fraction=0.02), (256, 256), seed=11)
    signal = np.full((256, 256), 1000.0)

    measured = signal * (1.0 + maps.prnu_fraction)

    assert float(measured.mean()) == pytest.approx(1000.0, rel=0.01)
    assert float(measured.std()) == pytest.approx(20.0, rel=0.10)


def test_fixed_pattern_maps_ignore_the_global_numpy_rng() -> None:
    config = FPNConfig(dsnu_e=2.0, prnu_fraction=0.01)
    np.random.seed(1)
    a = fixed_pattern_maps(config, (16, 16), seed=7)
    np.random.seed(2)
    b = fixed_pattern_maps(config, (16, 16), seed=7)

    np.testing.assert_array_equal(a.dsnu_e, b.dsnu_e)
    np.testing.assert_array_equal(a.prnu_fraction, b.prnu_fraction)


@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (-2, 4), (4,), (2, 3, 4), (4.0, 4.0)])
def test_fixed_pattern_rejects_invalid_shapes(shape: object) -> None:
    with pytest.raises(ValueError, match="shape"):
        fixed_pattern_maps(FPNConfig(), shape, seed=0)  # type: ignore[arg-type]


def test_fixed_pattern_rejects_a_ragged_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        fixed_pattern_maps(FPNConfig(), [[4, 4], [4]], seed=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_fixed_pattern_rejects_invalid_seeds(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        fixed_pattern_maps(FPNConfig(), (4, 4), seed=seed)  # type: ignore[arg-type]


def test_fixed_pattern_rejects_the_wrong_config_type() -> None:
    with pytest.raises(ValueError, match="FPNConfig"):
        fixed_pattern_maps(object(), (4, 4), seed=0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Row noise
# --------------------------------------------------------------------------- #


def test_row_noise_common_per_row() -> None:
    """One offset per row, identical across every column after broadcasting."""
    noise = row_noise(5.0, 1024, np.random.default_rng(0))

    assert noise.shape == (1024, 1)
    tiled = np.broadcast_to(noise, (1024, 64))
    for column in range(1, 64):
        np.testing.assert_array_equal(tiled[:, column], tiled[:, 0])
    assert float(noise.mean()) == pytest.approx(0.0, abs=1.0)
    assert float(noise.std()) == pytest.approx(5.0, rel=0.10)


def test_row_noise_zero_sigma_returns_zeros_without_consuming_rng() -> None:
    rng = np.random.default_rng(3)
    noise = row_noise(0.0, 8, rng)

    np.testing.assert_array_equal(noise, np.zeros((8, 1)))
    assert rng.random() == np.random.default_rng(3).random()


def test_row_noise_is_reproducible_and_seed_sensitive() -> None:
    a = row_noise(2.0, 64, np.random.default_rng(4))
    b = row_noise(2.0, 64, np.random.default_rng(4))
    c = row_noise(2.0, 64, np.random.default_rng(5))

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("sigma", [-0.1, np.nan, np.inf])
def test_row_noise_rejects_invalid_sigma(sigma: float) -> None:
    with pytest.raises(ValueError, match="row_sigma_e"):
        row_noise(sigma, 8, np.random.default_rng(0))


@pytest.mark.parametrize("height", [0, -1, 2.0, True])
def test_row_noise_rejects_invalid_height(height: object) -> None:
    with pytest.raises(ValueError, match="height"):
        row_noise(1.0, height, np.random.default_rng(0))  # type: ignore[arg-type]


def test_row_noise_rejects_a_non_generator_rng() -> None:
    with pytest.raises(ValueError, match="Generator"):
        row_noise(1.0, 4, 0)  # type: ignore[arg-type]
