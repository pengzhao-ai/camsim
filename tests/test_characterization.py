"""Round-trip tests for EMVA 1288-style sensor characterization.

The characterization module configures a sensor with known read noise, full
well, DSNU, and PRNU, then measures them back from simulated flat fields and
dark frames. These tests pin the measurement chain end to end:

* the photon transfer curve recovers the ADC conversion scale;
* its intercept, corrected for row noise and ADC quantization, recovers the
  configured input-referred read noise;
* a clipped high-level flat field recovers the full-well saturation capacity;
* dynamic range follows from the two;
* the dark-frame fixed pattern recovers DSNU after accounting for the
  zero-charge clipping knee in the sensor model;
* the high-signal flat field recovers PRNU;
* the report survives a JSON round trip.

The irradiance levels start near 1e-6 W/m^2/nm and end above the full-well
knee. Higher "laboratory" levels such as 1e-2 W/m^2/nm saturate every frame in
the 1.12 um flagship part at a 10 ms exposure, so they cannot define a photon
transfer curve; the range used here spans roughly 1 to 2000 electrons per
pixel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import camsim
from camsim.characterization import EMVAReport, characterize_sensor
from camsim.config.io import load_sensor
from camsim.config.schema import SensorConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SENSOR_PRESET = (
    REPO_ROOT / "src" / "camsim" / "config" / "presets" / "sensors" / "flagship_phone_50mp.json"
)

# Full-well-limited (-) saturated signal code at 10 bits / black 64.
NOMINAL_HCG_GAIN_DN_PER_E = 300.0 * (1023 - 64) / 800_000.0

# Low-to-saturation sweep, W/m^2/nm. The top level clips every channel and is
# used only for the saturation measurement.
HCG_LEVELS = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 4e-4, 1e-3]
LCG_LEVELS = [
    1e-6,
    2e-6,
    5e-6,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    4e-4,
    8e-4,
    1.2e-3,
    3e-3,
]


@pytest.fixture(scope="module")
def flagship_config() -> SensorConfig:
    """The flagship sensor preset, loaded through the production loader."""
    return load_sensor(SENSOR_PRESET)


@pytest.fixture(scope="module")
def hcg_report(flagship_config: SensorConfig) -> EMVAReport:
    """One HCG characterization shared by the measurement tests."""
    sensor = camsim.Sensor(flagship_config, seed=1)
    return characterize_sensor(
        sensor,
        irradiance_levels=HCG_LEVELS,
        frames_per_level=16,
        mode="HCG",
        exposure_s=0.01,
        base_seed=3,
    )


@pytest.fixture(scope="module")
def lcg_report(flagship_config: SensorConfig) -> EMVAReport:
    """One LCG characterization shared by the dynamic-range test."""
    sensor = camsim.Sensor(flagship_config, seed=1)
    return characterize_sensor(
        sensor,
        irradiance_levels=LCG_LEVELS,
        frames_per_level=8,
        mode="LCG",
        exposure_s=0.01,
        base_seed=3,
    )


def test_characterization_recovers_read_noise(hcg_report: EMVAReport) -> None:
    """The PTC intercept recovers the configured 0.7 e- rms HCG read noise."""
    assert hcg_report.read_noise_e == pytest.approx(0.7, rel=0.10)


def test_characterization_recovers_saturation(hcg_report: EMVAReport) -> None:
    """A clipped flat field recovers the configured 2000 e- HCG full well."""
    assert hcg_report.saturation_capacity_e == pytest.approx(2000.0, rel=0.10)


def test_characterization_reports_dynamic_range(lcg_report: EMVAReport) -> None:
    """LCG combines a 6000 e- well with 2 e- read noise, far above 30 dB."""
    assert lcg_report.dynamic_range_db > 30.0


def test_ptc_slope_matches_system_gain(hcg_report: EMVAReport) -> None:
    """The fitted PTC slope is the ADC system gain in DN per electron."""
    assert hcg_report.system_gain_dn_per_e == pytest.approx(NOMINAL_HCG_GAIN_DN_PER_E, rel=0.02)


def test_dsnu_prnu_recovered(hcg_report: EMVAReport) -> None:
    """Dark fixed pattern and flat-field non-uniformity are recovered."""
    assert hcg_report.dsnu_e == pytest.approx(0.5, rel=0.25)
    assert hcg_report.prnu_percent == pytest.approx(0.5, rel=0.25)


def test_linearity_residual_small(hcg_report: EMVAReport) -> None:
    """The sensor is linear in irradiance below saturation."""
    assert 0.0 <= hcg_report.linearity_residual < 0.05


def test_report_json_roundtrip(hcg_report: EMVAReport) -> None:
    """The report serializes without losing a measurable quantity."""
    payload = hcg_report.to_dict()
    json.dumps(payload, allow_nan=False)
    assert EMVAReport.from_dict(payload) == hcg_report

    text = hcg_report.to_json()
    assert json.loads(text, parse_constant=_reject_json_constant)["read_noise_e"] == pytest.approx(
        hcg_report.read_noise_e
    )
    assert EMVAReport.from_json(text) == hcg_report


def test_report_to_json_writes_file(hcg_report: EMVAReport, tmp_path: Path) -> None:
    """``to_json`` writes the same strict JSON it returns."""
    path = tmp_path / "emva" / "flagship.json"
    text = hcg_report.to_json(path)
    written = path.read_text(encoding="utf-8")
    assert written.rstrip("\n") == text
    assert EMVAReport.from_json(written) == hcg_report


def test_default_auto_characterization(flagship_config: SensorConfig) -> None:
    """The default auto-level, auto-mode call produces a usable report."""
    sensor = camsim.Sensor(flagship_config, seed=1)
    report = characterize_sensor(sensor, frames_per_level=8, exposure_s=0.01, base_seed=11)
    assert report.system_gain_dn_per_e > 0.0
    assert report.read_noise_e > 0.0
    assert report.saturation_capacity_e > 0.0
    assert report.dynamic_range_db > 0.0
    assert report.provenance["mode_resolved"] in {"HCG", "LCG"}
    assert report.provenance["mode_requested"] == "auto"
    assert "simulated" in str(report.provenance["characterization"]).lower()


def test_lux_levels_record_lens_and_illuminant(flagship_config: SensorConfig) -> None:
    """Lux input travels through the illuminant and reference-lens conversion."""
    sensor = camsim.Sensor(flagship_config, seed=1)
    report = characterize_sensor(
        sensor,
        lux_levels=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        frames_per_level=4,
        mode="HCG",
        exposure_s=0.01,
        illuminant="D65",
        base_seed=5,
    )
    assert report.provenance["illuminant"] == "D65"
    assert report.provenance["lux_levels"] == [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    lens = report.provenance["lens"]
    assert isinstance(lens, dict)
    assert lens["f_number"] == pytest.approx(1.8)
    assert lens["transmittance"] == pytest.approx(0.9)
    assert report.read_noise_e > 0.0
    assert report.saturation_capacity_e > 0.0


def test_conflicting_level_units_raise(flagship_config: SensorConfig) -> None:
    """A caller cannot supply both irradiance and lux level lists."""
    sensor = camsim.Sensor(flagship_config, seed=1)
    with pytest.raises(ValueError, match="not both"):
        characterize_sensor(
            sensor,
            irradiance_levels=[1e-5, 1e-4, 1e-3],
            lux_levels=[1.0, 10.0, 100.0],
        )


def _reject_json_constant(value: str) -> None:
    """Fail the test if the JSON encoder emitted NaN or infinity."""
    raise AssertionError(f"non-finite JSON constant {value!r}")
