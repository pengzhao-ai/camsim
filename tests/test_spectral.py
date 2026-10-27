"""Tests for wavelength grids and spectral irradiance containers.

The container contract is deliberately explicit so that a camera simulation can
never silently accept a spectral cube on the wrong grid: shapes are (H, W, L),
wavelengths are a validated uniform grid in nanometres, data are finite and
non-negative, and resampling is only ever performed by an explicit helper.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from camsim.core import units
from camsim.core.spectral import SpectralIrradiance, WavelengthGrid

GRID_NM = np.array([450.0, 550.0, 650.0])
UNIT_W = "W/m^2/nm"
UNIT_PHOTONS = "photons/s/m^2/nm"


def _cell_edges(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Cell boundaries implied by the documented resampling convention.

    Interior boundaries sit at midpoints between samples and the outer
    boundaries coincide with the first and last sample, so the represented
    domain is exactly the sampled wavelength range.
    """
    midpoints = 0.5 * (wavelengths_nm[:-1] + wavelengths_nm[1:])
    return np.concatenate(([wavelengths_nm[0]], midpoints, [wavelengths_nm[-1]]))


def test_wavelength_grid_rejects_nonuniform() -> None:
    with pytest.raises(ValueError, match="uniform"):
        WavelengthGrid(np.array([400.0, 410.0, 430.0]))


def test_wavelength_grid_validates_shape_order_positivity_and_size() -> None:
    with pytest.raises(ValueError, match="1-D"):
        WavelengthGrid(np.ones((2, 3)))
    with pytest.raises(ValueError, match="strictly increasing"):
        WavelengthGrid(np.array([450.0, 440.0, 430.0]))
    with pytest.raises(ValueError, match="positive"):
        WavelengthGrid(np.array([-20.0, -10.0, 0.0]))
    with pytest.raises(ValueError, match="two"):
        WavelengthGrid(np.array([550.0]))


def test_wavelength_grid_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        WavelengthGrid(np.array([450.0, np.nan, 650.0]))


def test_wavelength_grid_is_an_immutable_float64_value() -> None:
    source = np.array([450.0, 550.0, 650.0])
    grid = WavelengthGrid(source)
    source[0] = 999.0

    assert grid.nm.dtype == np.float64
    np.testing.assert_array_equal(grid.nm, GRID_NM)
    assert not grid.nm.flags.writeable
    with pytest.raises(ValueError):
        grid.nm[0] = 460.0
    assert grid == WavelengthGrid(GRID_NM.copy())
    assert grid != WavelengthGrid(np.array([440.0, 540.0, 640.0]))


def test_wavelength_grid_length_matches_sample_count() -> None:
    grid = WavelengthGrid(GRID_NM)

    assert len(grid) == GRID_NM.size


def test_spectral_irradiance_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SpectralIrradiance.from_array(
            -np.ones((2, 2, 3)),
            wavelengths_nm=np.array([450.0, 550.0, 650.0]),
            unit="W/m^2/nm",
        )


def test_nan_irradiance_raises() -> None:
    data = np.ones((2, 2, 3))
    data[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        SpectralIrradiance.from_array(data, GRID_NM, UNIT_W)


def test_spectral_irradiance_rejects_bad_shape_and_grid_length() -> None:
    with pytest.raises(ValueError, match="3-D"):
        SpectralIrradiance.from_array(np.ones((2, 3)), GRID_NM, UNIT_W)
    with pytest.raises(ValueError, match="wavelength"):
        SpectralIrradiance.from_array(np.ones((2, 2, 4)), GRID_NM, UNIT_W)


def test_spectral_irradiance_rejects_unknown_unit_and_pitch() -> None:
    with pytest.raises(ValueError, match="unit"):
        SpectralIrradiance.from_array(np.ones((2, 2, 3)), GRID_NM, "W/m^2/um")
    with pytest.raises(ValueError, match="pixel_pitch_um"):
        SpectralIrradiance.from_array(np.ones((2, 2, 3)), GRID_NM, UNIT_W, pixel_pitch_um=0.0)


def test_from_array_preserves_metadata_and_float32_data() -> None:
    irr = SpectralIrradiance.from_array(
        np.ones((2, 3, 3), dtype=np.float32),
        GRID_NM,
        UNIT_W,
        pixel_pitch_um=1.12,
    )

    assert irr.data.shape == (2, 3, 3)
    assert irr.data.dtype == np.float32
    assert irr.unit == UNIT_W
    assert irr.pixel_pitch_um == pytest.approx(1.12)
    np.testing.assert_array_equal(irr.wavelengths_nm.nm, GRID_NM)


def test_direct_constructor_accepts_a_wavelength_grid() -> None:
    grid = WavelengthGrid(GRID_NM)

    irr = SpectralIrradiance(np.ones((2, 2, 3)), grid, UNIT_W)

    assert irr.wavelengths_nm is grid


def test_unit_conversion_roundtrip() -> None:
    irr = SpectralIrradiance.from_array(np.full((2, 2, 3), 2.0), GRID_NM, UNIT_W)

    photons = irr.to_unit(UNIT_PHOTONS)

    assert photons.unit == UNIT_PHOTONS
    assert photons.data is not irr.data
    np.testing.assert_allclose(
        photons.data,
        irr.data / units.photon_energy_j(GRID_NM),
        rtol=1e-15,
    )

    back = photons.to_unit(UNIT_W)

    assert back.unit == UNIT_W
    np.testing.assert_allclose(back.data, irr.data, rtol=1e-12)


def test_unit_conversion_to_same_unit_returns_a_new_container() -> None:
    irr = SpectralIrradiance.from_array(np.ones((2, 2, 3)), GRID_NM, UNIT_W)

    same = irr.to_unit(UNIT_W)

    assert same is not irr
    assert same.data is not irr.data
    np.testing.assert_array_equal(same.data, irr.data)


def test_resample_spatial_conserves_energy() -> None:
    grid = np.array([450.0, 550.0, 650.0])
    cube = np.ones((8, 8, 3))
    out = SpectralIrradiance.from_array(cube, grid, "W/m^2/nm").resample_spatial((4, 4))
    assert out.data.shape == (4, 4, 3)
    np.testing.assert_allclose(out.data.mean(), 1.0)


def test_resample_spatial_averages_blocks_and_updates_pitch() -> None:
    cube = np.arange(4 * 4 * 3, dtype=np.float64).reshape(4, 4, 3)
    irr = SpectralIrradiance.from_array(cube, GRID_NM, UNIT_W, pixel_pitch_um=1.12)

    out = irr.resample_spatial((2, 2))
    expected = cube.reshape(2, 2, 2, 2, 3).mean(axis=(1, 3))

    np.testing.assert_allclose(out.data, expected)
    assert out.pixel_pitch_um == pytest.approx(2.24)


def test_resample_spatial_anisotropic_drops_pitch() -> None:
    irr = SpectralIrradiance.from_array(np.ones((4, 4, 3)), GRID_NM, UNIT_W, pixel_pitch_um=1.12)

    out = irr.resample_spatial((2, 1))

    assert out.data.shape == (2, 1, 3)
    assert out.pixel_pitch_um is None


def test_resample_spatial_rejects_nondivisible_target_and_unknown_method() -> None:
    irr = SpectralIrradiance.from_array(np.ones((8, 8, 3)), GRID_NM, UNIT_W)

    with pytest.raises(ValueError, match="divide"):
        irr.resample_spatial((3, 4))
    with pytest.raises(ValueError, match="method"):
        irr.resample_spatial((4, 4), method="nearest")


def test_resample_wavelengths_conserves_energy() -> None:
    source_nm = np.array([400.0, 450.0, 500.0, 550.0, 600.0])
    target_nm = np.array([400.0, 500.0, 600.0])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cube = np.broadcast_to(values, (2, 3, values.size))
    irr = SpectralIrradiance.from_array(cube, source_nm, UNIT_W)

    out = irr.resample_wavelengths(target_nm)

    assert out.data.shape == (2, 3, 3)
    np.testing.assert_array_equal(out.wavelengths_nm.nm, target_nm)
    np.testing.assert_allclose(out.data[0, 0], [1.5, 3.0, 4.5])

    source_energy = np.sum(values * np.diff(_cell_edges(source_nm)))
    target_energy = np.sum(out.data[0, 0] * np.diff(_cell_edges(target_nm)))
    assert target_energy == pytest.approx(source_energy, rel=1e-14)


def test_resample_wavelengths_identity_preserves_data() -> None:
    data = np.random.default_rng(0).random((2, 2, 5)) * 10.0
    irr = SpectralIrradiance.from_array(data, np.array([400.0, 450.0, 500.0, 550.0, 600.0]), UNIT_W)

    out = irr.resample_wavelengths(irr.wavelengths_nm)

    np.testing.assert_allclose(out.data, data, rtol=1e-14)


def test_resample_wavelengths_constant_field_stays_constant() -> None:
    irr = SpectralIrradiance.from_array(
        np.full((2, 2, 5), 7.0), np.array([400.0, 450.0, 500.0, 550.0, 600.0]), UNIT_W
    )

    out = irr.resample_wavelengths(np.array([400.0, 500.0, 600.0]))

    np.testing.assert_allclose(out.data, 7.0)


def test_resample_wavelengths_raises_on_uncovered_range() -> None:
    irr = SpectralIrradiance.from_array(
        np.ones((2, 2, 5)), np.array([400.0, 450.0, 500.0, 550.0, 600.0]), UNIT_W
    )

    with pytest.raises(ValueError, match="cover"):
        irr.resample_wavelengths(np.array([350.0, 450.0, 550.0]))
    with pytest.raises(ValueError, match="cover"):
        irr.resample_wavelengths(np.array([500.0, 600.0, 700.0]))


def test_resample_wavelengths_zero_fill_warns_and_fills() -> None:
    irr = SpectralIrradiance.from_array(
        np.full((2, 2, 5), 2.0), np.array([400.0, 450.0, 500.0, 550.0, 600.0]), UNIT_W
    )

    with pytest.warns(UserWarning, match="zero"):
        out = irr.resample_wavelengths(np.array([400.0, 500.0, 600.0, 700.0]), allow_zero_fill=True)

    assert out.unit == UNIT_W
    np.testing.assert_allclose(out.data[..., :2], 2.0)
    np.testing.assert_allclose(out.data[..., 2], 1.0)
    np.testing.assert_allclose(out.data[..., 3], 0.0)


def test_resample_wavelengths_rejects_unknown_method() -> None:
    irr = SpectralIrradiance.from_array(np.ones((2, 2, 3)), GRID_NM, UNIT_W)

    with pytest.raises(ValueError, match="method"):
        irr.resample_wavelengths(GRID_NM, method="linear")


def test_spectral_names_resolve_through_lazy_exports() -> None:
    code = "\n".join(
        [
            "import sys",
            "import camsim",
            "assert 'camsim.core.spectral' not in sys.modules",
            "assert 'WavelengthGrid' in camsim.__all__",
            "assert 'SpectralIrradiance' in camsim.__all__",
            "assert camsim.WavelengthGrid.__name__ == 'WavelengthGrid'",
            "assert camsim.SpectralIrradiance.__name__ == 'SpectralIrradiance'",
            "assert 'camsim.core.spectral' in sys.modules",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)
