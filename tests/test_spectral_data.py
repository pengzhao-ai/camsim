"""Tests for ColorChecker spectra, illuminants, and CIE Lab ground truth.

The contracts under test are spectral, not RGB: a 24-patch chart is a set of
reflectance factors on a validated nanometre grid, an illuminant is a relative
spectral power distribution, and the Lab reference is obtained by integrating
reflectance x illuminant x colour-matching functions. Absolute photometry is
checked by converting a lux target into a per-nanometre irradiance and
integrating it back with the CIE 1924 photopic efficiency function.

The integration axis is always the nanometre grid. ``SpectralIrradiance`` is a
per-nanometre density (W/m^2/nm), so the band integral is
``trapz(E * V, wavelength_nm)``. Multiplying that axis by 1e-9 without also
converting the density from per-nm to per-metre silently scales the result by
1e-9; that nm/metre trap is exactly what these tests are meant to catch.
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest

from camsim.core.color import ColorReference
from camsim.core.spectral import WavelengthGrid
from camsim.scene import illuminants, spectral_data
from camsim.scene.illuminants import (
    K_M,
    illuminant,
    lux_to_spectral_irradiance,
    photopic_luminous_efficiency,
    reference_lab,
)
from camsim.scene.spectral_data import (
    COMMON_WAVELENGTH_GRID,
    ColorCheckerData,
    SpectralPowerDistribution,
    load_colorchecker,
)

PATCH_NAMES = tuple(f"patch {index}" for index in range(24))


def test_colorchecker_has_24_patches() -> None:
    cc = spectral_data.load_colorchecker()

    assert len(cc.patches) == 24
    assert cc.spectra.shape == (24, len(cc.wavelengths_nm))
    assert len(cc.wavelengths_nm) == 95


def test_colorchecker_grid_and_patch_metadata() -> None:
    cc = load_colorchecker()

    assert isinstance(cc.wavelengths_nm, WavelengthGrid)
    np.testing.assert_allclose(cc.wavelengths_nm.nm, np.arange(360.0, 831.0, 5.0))
    assert len(set(cc.patches)) == 24
    assert all(name.strip() for name in cc.patches)
    assert cc.provenance["source_key"] == "BabelColor Average"
    assert "BabelColor" in cc.provenance["licence"]
    assert "BSD-3-Clause" in cc.provenance["licence"]


def test_colorchecker_reflectance_is_physical() -> None:
    spectra = load_colorchecker().spectra

    assert np.all(np.isfinite(spectra))
    assert np.all(spectra >= 0.0)
    assert np.all(spectra <= 1.0)
    assert spectra.max() > 0.8
    assert spectra.min() < 0.1


def test_colorchecker_data_rejects_bad_patch_metadata() -> None:
    grid = COMMON_WAVELENGTH_GRID
    spectra = np.full((24, grid.nm.size), 0.5)

    with pytest.raises(ValueError, match="24"):
        ColorCheckerData(patches=("only one",), spectra=spectra, wavelengths_nm=grid)
    with pytest.raises(ValueError, match="match"):
        ColorCheckerData(patches=PATCH_NAMES, spectra=spectra[:, :-1], wavelengths_nm=grid)
    with pytest.raises(ValueError, match="non-negative"):
        ColorCheckerData(patches=PATCH_NAMES, spectra=-spectra, wavelengths_nm=grid)
    with pytest.raises(ValueError, match=r"exceed 1\.0"):
        ColorCheckerData(patches=PATCH_NAMES, spectra=spectra + 1.0, wavelengths_nm=grid)


def test_spd_validates_values_grid_and_metadata() -> None:
    grid = COMMON_WAVELENGTH_GRID
    values = np.ones(grid.nm.size)

    with pytest.raises(ValueError, match="non-negative"):
        SpectralPowerDistribution(name="negative", wavelengths_nm=grid, values=-values)
    non_finite = values.copy()
    non_finite[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SpectralPowerDistribution(name="nan", wavelengths_nm=grid, values=non_finite)
    with pytest.raises(ValueError, match="match"):
        SpectralPowerDistribution(name="short", wavelengths_nm=grid, values=values[:-1])
    with pytest.raises(ValueError, match="1-D"):
        SpectralPowerDistribution(name="matrix", wavelengths_nm=grid, values=values[None, :])
    with pytest.raises(ValueError, match="name"):
        SpectralPowerDistribution(name="  ", wavelengths_nm=grid, values=values)
    with pytest.raises(ValueError, match="provenance"):
        SpectralPowerDistribution(
            name="provenance",
            wavelengths_nm=grid,
            values=values,
            provenance=["not", "a", "mapping"],  # type: ignore[arg-type]
        )


def test_illuminant_names_grids_and_provenance() -> None:
    for name in ("D65", "3000K", "4000K"):
        spd = illuminant(name)

        assert spd.name == name
        assert isinstance(spd.wavelengths_nm, WavelengthGrid)
        np.testing.assert_allclose(spd.wavelengths_nm.nm, COMMON_WAVELENGTH_GRID.nm)
        assert spd.values.shape == (95,)
        assert np.all(np.isfinite(spd.values))
        assert np.all(spd.values >= 0.0)
        assert spd.provenance["source"]
        assert "BSD-3-Clause" in spd.provenance["licence"]
        assert spd.provenance["reference"] == "CIE 15:2004"

    assert not np.array_equal(illuminant("D65").values, illuminant("3000K").values)


def test_warmer_illuminants_are_redder() -> None:
    wavelengths = COMMON_WAVELENGTH_GRID.nm
    red = (wavelengths >= 600.0) & (wavelengths <= 700.0)
    blue = (wavelengths >= 400.0) & (wavelengths <= 500.0)

    d65 = illuminant("D65").values
    warm = illuminant("3000K").values

    assert warm[red].mean() / warm[blue].mean() > d65[red].mean() / d65[blue].mean()


def test_illuminant_rejects_unknown_names_and_types() -> None:
    with pytest.raises(ValueError, match="illuminant"):
        illuminant("D50")
    with pytest.raises(TypeError, match="illuminant"):
        illuminant(6500)  # type: ignore[arg-type]


def test_photopic_peak_at_555nm() -> None:
    wavelengths = np.array([360.0, 500.0, 550.0, 555.0, 560.0, 600.0, 830.0])
    efficiency = photopic_luminous_efficiency(wavelengths)

    assert efficiency.shape == wavelengths.shape
    assert efficiency[3] == pytest.approx(1.0, abs=1e-12)
    assert efficiency[2] < 1.0
    assert efficiency[4] < 1.0
    assert np.all(efficiency >= 0.0)
    assert efficiency[0] < 1e-5
    assert efficiency[-1] < 1e-6
    assert float(photopic_luminous_efficiency(555.0)) == pytest.approx(1.0)


def test_photopic_outside_the_cie_table_is_zero() -> None:
    np.testing.assert_array_equal(
        photopic_luminous_efficiency(np.array([300.0, 900.0])),
        np.array([0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="finite"):
        photopic_luminous_efficiency(np.array([np.nan]))


def test_lux_normalization_d65() -> None:
    irradiance = lux_to_spectral_irradiance(1.0, illuminant("D65"))
    efficiency = photopic_luminous_efficiency(irradiance.wavelengths_nm.nm)

    # E is a per-nanometre density, so the photometric integral runs over nm.
    luminous = K_M * np.trapezoid(
        irradiance.data[0, 0] * efficiency,
        irradiance.wavelengths_nm.nm,
    )

    assert K_M == 683.0
    assert irradiance.data.shape == (1, 1, 95)
    assert irradiance.unit == "W/m^2/nm"
    assert luminous == pytest.approx(1.0, rel=1e-12)


def test_lux_scaling_and_zero() -> None:
    one = lux_to_spectral_irradiance(1.0, illuminant("D65")).data
    hundred = lux_to_spectral_irradiance(100.0, illuminant("D65")).data
    zero = lux_to_spectral_irradiance(0.0, illuminant("D65")).data

    np.testing.assert_allclose(hundred, 100.0 * one, rtol=1e-15)
    np.testing.assert_array_equal(zero, np.zeros_like(zero))


def test_lux_rejects_invalid_values_and_unusable_spd() -> None:
    spd = illuminant("D65")

    for lux in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="lux"):
            lux_to_spectral_irradiance(lux, spd)

    with pytest.raises(TypeError, match="SpectralPowerDistribution"):
        lux_to_spectral_irradiance(1.0, np.ones(95))  # type: ignore[arg-type]

    infrared = SpectralPowerDistribution(
        name="infrared",
        wavelengths_nm=np.array([850.0, 855.0, 860.0]),
        values=np.ones(3),
    )
    with pytest.raises(ValueError, match="photopic"):
        lux_to_spectral_irradiance(1.0, infrared)


def test_reference_lab_is_spectral_ground_truth() -> None:
    reference = reference_lab("D65")

    assert isinstance(reference, ColorReference)
    assert reference.lab.shape == (24, 3)
    assert reference.illuminant_name == "D65"
    assert reference.observer == "CIE 1931 2 degree"
    assert reference.white_point_xy == pytest.approx((0.31271, 0.32901), abs=2e-4)
    assert reference.provenance["integration"]
    assert reference.provenance["cmf"]
    assert reference.provenance["illuminant"]["source"]


def test_reference_lab_uses_each_illuminant_white_point() -> None:
    warm = reference_lab("3000K")
    neutral = reference_lab("4000K")

    assert warm.illuminant_name == "3000K"
    assert warm.white_point_xy == pytest.approx((0.4369, 0.4041), abs=5e-3)
    assert neutral.illuminant_name == "4000K"
    assert neutral.white_point_xy == pytest.approx((0.3804, 0.3767), abs=5e-3)
    assert neutral.white_point_xy != pytest.approx(warm.white_point_xy)


def test_reference_lab_rejects_unsupported_illuminant() -> None:
    with pytest.raises(ValueError, match="illuminant"):
        reference_lab("D50")


def test_reference_lab_changes_with_illuminant() -> None:
    d65 = illuminants.reference_lab("D65")
    warm = illuminants.reference_lab("3000K")

    assert np.max(np.abs(d65.lab - warm.lab)) > 1.0


def test_d65_reference_matches_colour_science() -> None:
    """Cross-check against colour-science's independent integration path.

    colour-science aligns the spectra to the 1 nm CIE 1931 observer grid with
    its own interpolation and uses a rectangular integration; camsim uses
    linear interpolation and trapezoidal integration on the 5 nm common grid.
    The residual measured for this dataset is about 0.3 Lab units, so a 0.5
    component tolerance and a 1.0 delta-E2000 tolerance are both conservative.
    """
    from colour import XYZ_to_Lab, XYZ_to_xy, sd_to_XYZ
    from colour.characterisation.datasets.colour_checkers import SDS_COLOURCHECKERS
    from colour.colorimetry import MSDS_CMFS, SDS_ILLUMINANTS
    from colour.difference import delta_E_CIE2000
    from colour.utilities import ColourRuntimeWarning

    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    d65 = SDS_ILLUMINANTS["D65"]
    # colour-science warns while aligning its native 10 nm reflectance and
    # 5 nm illuminant tables to the 1 nm observer grid; those warnings are
    # expected and do not indicate a problem with camsim's own resampling.
    # Restrict the suppression to those alignment messages so unrelated
    # deprecations or runtime warnings still surface.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=ColourRuntimeWarning,
            message=r".*(Aligning|Trimming|shape could not be honoured).*",
        )
        white = sd_to_XYZ(d65, cmfs)
        white_xy = XYZ_to_xy(white / white[1])
        expected = np.array(
            [
                XYZ_to_Lab(sd_to_XYZ(sd, cmfs, illuminant=d65) / 100.0, white_xy)
                for sd in SDS_COLOURCHECKERS["BabelColor Average"].values()
            ]
        )

    actual = reference_lab("D65").lab

    assert np.max(np.abs(actual - expected)) <= 0.5
    assert np.max(delta_E_CIE2000(actual, expected)) <= 1.0


def test_scene_modules_do_not_import_colour_science_eagerly() -> None:
    code = "\n".join(
        [
            "import sys",
            "import camsim.scene.spectral_data",
            "import camsim.scene.illuminants",
            "assert 'colour' not in sys.modules",
        ]
    )

    subprocess.run([sys.executable, "-c", code], check=True)
