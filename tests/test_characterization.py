"""Round-trip tests for EMVA 1288-style sensor characterization.

The characterization module configures a sensor with known read noise, full
well, DSNU, and PRNU, then measures them back from simulated flat fields and
dark frames. The dedicated G2 fixtures use the 32-frame characterization
default and assert read noise within 5 % and DSNU/PRNU within 10 %. The coarse
high-level-sweep robustness tests use a 10 % envelope, and the 8-frame Task 17
sweep uses a documented 20 % DSNU waiver because its dark average is too noisy
for the G2 gate. These tests pin the measurement chain end to end:

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
import math
from pathlib import Path

import pytest

import camsim
from camsim.characterization import EMVAReport, characterize_sensor
from camsim.characterization.emva import (
    _invert_clipped_dsnu,
    _observed_quantized_mean_std,
)
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
    # 32 frames is the characterization default and averages the residual
    # row-noise/quantization coupling enough to meet the 5 % read-noise G2 gate
    # at the sub-LSB operating point. Sixteen frames still passes at most
    # seeds but leaves a few percent seed-to-seed scatter.
    return characterize_sensor(
        sensor,
        irradiance_levels=HCG_LEVELS,
        frames_per_level=32,
        mode="HCG",
        exposure_s=0.01,
        base_seed=3,
    )


@pytest.fixture(scope="module")
def lcg_report(flagship_config: SensorConfig) -> EMVAReport:
    """One LCG characterization at the 32-frame default.

    LCG's dark offset is only about 0.06 DN, so an 8-frame dark average has a
    seed-dependent DSNU spread that can exceed the 10 % G2 gate. The production
    default of 32 frames is stable across seeds; the 8-frame Task 17-style
    sweep is covered separately with a documented waiver.
    """
    sensor = camsim.Sensor(flagship_config, seed=1)
    return characterize_sensor(
        sensor,
        irradiance_levels=LCG_LEVELS,
        frames_per_level=32,
        mode="LCG",
        exposure_s=0.01,
        base_seed=3,
    )


def test_characterization_recovers_read_noise(hcg_report: EMVAReport) -> None:
    """The exact quantized PTC recovers 0.7 e- rms HCG read noise at G2 (5 %)."""
    assert hcg_report.read_noise_e == pytest.approx(0.7, rel=0.05)


def test_lcg_read_noise_recovered(lcg_report: EMVAReport) -> None:
    """The same model recovers the configured 2.0 e- rms LCG read noise."""
    assert lcg_report.read_noise_e == pytest.approx(2.0, rel=0.05)


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
    """Dark fixed pattern and flat-field non-uniformity are recovered at G2 (10 %)."""
    assert hcg_report.dsnu_e == pytest.approx(0.5, rel=0.10)
    assert hcg_report.prnu_percent == pytest.approx(0.5, rel=0.10)


def test_lcg_dsnu_prnu_recovered(lcg_report: EMVAReport) -> None:
    """LCG's sub-LSB dark signal still recovers DSNU and PRNU at G2 (10 %)."""
    assert lcg_report.dsnu_e == pytest.approx(0.5, rel=0.10)
    assert lcg_report.prnu_percent == pytest.approx(0.5, rel=0.10)


def test_dsnu_recovers_configured_strengths(flagship_config: SensorConfig) -> None:
    """The clipped/quantized inversion is calibrated across known DSNU values."""
    for configured in (0.25, 1.0):
        noise = flagship_config.noise.model_copy(update={"dsnu_e": configured})
        config = flagship_config.model_copy(update={"noise": noise})
        sensor = camsim.Sensor(config, seed=1)
        report = characterize_sensor(
            sensor,
            irradiance_levels=HCG_LEVELS,
            frames_per_level=32,
            mode="HCG",
            exposure_s=0.01,
            base_seed=3,
        )
        assert report.dsnu_e == pytest.approx(configured, rel=0.10)


def test_dsnu_inversion_preserves_high_values() -> None:
    """The inversion range covers stressed DSNU and raises beyond it.

    The production DSNU range is 0.1-10 e-, but a 10 e- model ceiling silently
    clamped 12-15 e- configurations. The range is now 20 e- and an observation
    beyond it raises instead of returning the ceiling.
    """
    gain = NOMINAL_HCG_GAIN_DN_PER_E
    sigma = 0.7
    for configured in (12.0, 15.0):
        observed = _observed_quantized_mean_std(configured, sigma, gain)
        inferred = _invert_clipped_dsnu(observed, sigma, gain)
        assert inferred == pytest.approx(configured, rel=0.02)
    with pytest.raises(ValueError, match="DSNU model range"):
        too_large = _observed_quantized_mean_std(25.0, sigma, gain)
        _invert_clipped_dsnu(too_large, sigma, gain)


def test_high_level_sweep_is_extended_downward(flagship_config: SensorConfig) -> None:
    """A too-bright caller sweep gets low levels instead of a fit failure."""
    sensor = camsim.Sensor(flagship_config, seed=1)
    report = characterize_sensor(
        sensor,
        irradiance_levels=[2e-4, 5e-4, 1e-3, 2e-3],
        frames_per_level=8,
        mode="HCG",
        exposure_s=0.01,
        base_seed=3,
    )
    assert report.provenance["low_levels_appended"] is True
    assert len(report.provenance["added_low_levels_w_per_m2_nm"]) > 0
    # This coarse, auto-extended sweep is a robustness check, not the G2
    # fixture; it is allowed the wider 10 % envelope.
    assert report.read_noise_e == pytest.approx(0.7, rel=0.10)
    assert report.saturation_capacity_e == pytest.approx(2000.0, rel=0.10)


def test_task17_lcg_high_sweep_runs_with_8_frame_waiver(
    flagship_config: SensorConfig,
) -> None:
    """The Task 17 LCG sweep runs at its 8-frame/seed-9 configuration.

    Its DSNU is a noisy dark-frame statistic at 8 frames and is allowed a 20 %
    envelope here; the G2 DSNU assertion uses the 32-frame default. Read noise
    and saturation remain inside 10 %.
    """
    sensor = camsim.Sensor(flagship_config, seed=1)
    report = characterize_sensor(
        sensor,
        irradiance_levels=[2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 4e-2, 8e-2],
        frames_per_level=8,
        mode="LCG",
        exposure_s=0.01,
        base_seed=9,
    )
    assert report.provenance["low_levels_appended"] is True
    assert report.read_noise_e == pytest.approx(2.0, rel=0.10)
    assert report.saturation_capacity_e == pytest.approx(6000.0, rel=0.10)
    assert report.dynamic_range_db > 30.0
    # Documented 8-frame waiver: the controller will move the Task 17 G2 sweep
    # to 32 frames, where LCG DSNU is stable inside 10 %.
    assert report.dsnu_e == pytest.approx(0.5, rel=0.20)


def test_snr_curve_exposes_total_noise(hcg_report: EMVAReport) -> None:
    """Each SNR point carries both row-removed and total-noise variants."""
    assert hcg_report.snr_curve
    for point in hcg_report.snr_curve:
        assert point.noise_dn_total is not None
        assert point.noise_e_total is not None
        assert point.snr_db_total is not None
        assert point.noise_dn_total > 0.0
        assert point.noise_e_total == pytest.approx(
            point.noise_dn_total / hcg_report.system_gain_dn_per_e
        )
        assert point.snr_db_total == pytest.approx(
            20.0 * math.log10(point.mean_signal_e / point.noise_e_total)
        )


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
