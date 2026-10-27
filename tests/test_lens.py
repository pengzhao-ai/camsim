"""Tests for reference lens geometry and on-axis spectral transmission.

The lens model is the geometric and radiometric entry point of the pipeline.
Geometry fixes what the sensor can see: a focal length ``f`` images a sensor
diagonal ``d`` over the field of view ``2 atan(d / 2f)``, and a pixel of pitch
``p`` subtends ``atan(p / f)``, which is how cycles/pixel become cycles/degree.
For the reference phone lens (6.4 mm focal length, 11.46 mm diagonal) the
diagonal field is about 84 degrees, or about 100 pixels per degree at a
1.12 um pitch.

Transmission fixes how much light survives the glass. A modern AR-coated lens
transmits roughly 85-98 % on axis; the scalar form is a broadband average,
while an optical design or coating datasheet usually supplies a two-column
spectral curve (wavelength in nanometres, transmittance as a fraction).
Interpolating that curve linearly is adequate because coating curves are smooth
at the 5-10 nm scale of camera spectral datasets, and holding the endpoint
values outside the measured range avoids inventing data.

Failure artefacts guarded here: a non-positive f-number is unphysical and is
rejected by the configuration schema; a transmittance above one would create
energy; a curve with decreasing wavelengths mislabels every sample; a curve
value outside ``[0, 1]`` is a units error (percent versus fraction); and an
unknown CSV path must fail loudly rather than silently defaulting to full
transmission.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

import camsim
from camsim.config.schema import LensConfig as SchemaLensConfig
from camsim.optics.lens import LensConfig, transmittance_at


def _reference_lens(transmittance: float | str = 0.9) -> LensConfig:
    """Build the reference phone lens with an overridable transmittance."""
    return LensConfig(
        f_number=1.8,
        focal_length_mm=6.4,
        sensor_diagonal_mm=11.46,
        transmittance=transmittance,
    )


def test_lens_fov_for_reference_phone() -> None:
    lens = _reference_lens()

    assert lens.field_of_view_deg == pytest.approx(84.0, abs=1.0)
    assert lens.pixels_per_degree(1.12) == pytest.approx(99.73, rel=1e-3)


def test_lens_config_is_the_schema_definition() -> None:
    # optics.lens re-exports the one definition; there must be no duplicate
    # model whose validation could drift from the configuration schema.
    assert LensConfig is SchemaLensConfig
    assert camsim.LensConfig is SchemaLensConfig


@pytest.mark.parametrize("f_number", [0.0, -1.8, math.nan, math.inf])
def test_lens_rejects_invalid_f_number(f_number: float) -> None:
    with pytest.raises(ValidationError):
        LensConfig(
            f_number=f_number,
            focal_length_mm=6.4,
            sensor_diagonal_mm=11.46,
            transmittance=0.9,
        )


def test_lens_rejects_invalid_transmittance() -> None:
    with pytest.raises(ValidationError):
        _reference_lens(transmittance=1.5)


def test_lens_scalar_transmittance() -> None:
    lens = _reference_lens(transmittance=0.87)

    assert transmittance_at(lens, 550.0) == pytest.approx(0.87)
    assert transmittance_at(lens, 900.0) == pytest.approx(0.87)


def test_lens_spectral_transmittance(tmp_path: Path) -> None:
    curve_path = tmp_path / "transmittance.csv"
    curve_path.write_text(
        "wavelength_nm,transmittance\n400,0.80\n500,0.90\n600,0.85\n",
        encoding="utf-8",
    )
    lens = _reference_lens(transmittance=str(curve_path))

    assert transmittance_at(lens, 500.0) == pytest.approx(0.90)
    assert transmittance_at(lens, 450.0) == pytest.approx(0.85)
    # Endpoint clamping: the curve is held constant outside its measured range.
    assert transmittance_at(lens, 350.0) == pytest.approx(0.80)
    assert transmittance_at(lens, 700.0) == pytest.approx(0.85)


def test_lens_spectral_transmittance_without_header(tmp_path: Path) -> None:
    curve_path = tmp_path / "transmittance.csv"
    curve_path.write_text(
        "# measured with a spectrophotometer\n400,0.80\n600,0.90\n", encoding="utf-8"
    )
    lens = _reference_lens(transmittance=str(curve_path))

    assert transmittance_at(lens, 500.0) == pytest.approx(0.85)


@pytest.mark.parametrize(
    "contents",
    [
        "wavelength_nm,transmittance\n600,0.90\n500,0.85\n",
        "wavelength_nm,transmittance\n400,0.80\n500,1.50\n",
        "wavelength_nm,transmittance\n400,0.80\n",
        "wavelength_nm,transmittance\n400\n500,0.85\n",
    ],
)
def test_lens_rejects_malformed_transmittance_curve(tmp_path: Path, contents: str) -> None:
    curve_path = tmp_path / "bad_transmittance.csv"
    curve_path.write_text(contents, encoding="utf-8")
    lens = _reference_lens(transmittance=str(curve_path))

    with pytest.raises(ValueError, match="transmittance"):
        transmittance_at(lens, 500.0)


def test_lens_missing_transmittance_file_fails_loudly(tmp_path: Path) -> None:
    lens = _reference_lens(transmittance=str(tmp_path / "missing.csv"))

    with pytest.raises(ValueError, match="cannot read"):
        transmittance_at(lens, 550.0)


@pytest.mark.parametrize("wavelength_nm", [0.0, -550.0, math.nan, math.inf])
def test_lens_transmittance_rejects_invalid_wavelength(wavelength_nm: float) -> None:
    with pytest.raises(ValueError, match="wavelength"):
        transmittance_at(_reference_lens(), wavelength_nm)
