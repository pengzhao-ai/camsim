"""Tests for the versioned sensor, lens, and ISP configuration schema.

The configuration layer is the contract between datasheet-derived presets and
every later pipeline stage, so these tests pin four properties:

* the shipped presets load through the production loader and carry the
  representative flagship values (pixel pitch, ADC depth, LCG/HCG split);
* misspelled or unknown keys are rejected instead of silently ignored;
* cross-field physics rules fail loudly (HCG cannot out-resolve LCG full well,
  black level must fit the ADC code range, gain/exposure defaults must lie
  inside their limits) while ADC-limited saturation is a warning, not an error;
* the exported JSON Schema names every unit-bearing field, and the lens model
  derives field of view and pixels-per-degree from geometry.

Configuration files are code: a typo in ``full_well_e`` should stop a sweep
before it produces a plausible-looking but physically wrong dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from camsim.config.io import (
    ValidationReport,
    isp_schema,
    load_isp,
    load_sensor,
    sensor_schema,
    validate_config,
)
from camsim.config.schema import (
    CCMConfig,
    ISPConfig,
    LensConfig,
    LSCConfig,
    SensorConfig,
    WavelengthRange,
)
from camsim.core.frames import CFAPattern

REPO_ROOT = Path(__file__).resolve().parents[1]
SENSOR_PRESET = (
    REPO_ROOT / "src" / "camsim" / "config" / "presets" / "sensors" / "flagship_phone_50mp.json"
)
ISP_PRESET = REPO_ROOT / "src" / "camsim" / "config" / "presets" / "isp" / "default.json"


def _preset_payload(path: Path) -> dict[str, Any]:
    """Return a preset as mutable JSON data so a test can corrupt one field."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(tmp_path: Path, payload: dict[str, Any], name: str = "config.json") -> Path:
    """Write ``payload`` to a temporary JSON file and return its path."""
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------


def test_valid_sensor_preset_loads() -> None:
    cfg = load_sensor(SENSOR_PRESET)
    assert cfg.pixel.pitch_um == pytest.approx(1.12)
    assert cfg.adc.bits == 10
    assert cfg.modes["HCG"].full_well_e == 2000
    assert cfg.cfa is CFAPattern.RGGB
    assert cfg.name == "flagship_phone_50mp"
    assert cfg.provenance.source
    assert cfg.provenance.assumptions


def test_valid_isp_preset_loads() -> None:
    cfg = load_isp(ISP_PRESET)
    assert cfg.dpc.enabled
    assert cfg.blc.enabled
    assert cfg.awb.enabled
    assert cfg.demosaic.enabled
    assert cfg.demosaic.method == "mhc"
    assert cfg.ccm.enabled
    assert cfg.ccm.matrix is not None
    for row in cfg.ccm.matrix:
        assert sum(row) == pytest.approx(1.0)
    assert not cfg.lsc.enabled
    assert not cfg.denoise.enabled
    assert not cfg.sharpen.enabled
    assert cfg.tone.enabled
    assert cfg.tone.color_space == "sRGB"
    assert cfg.output.color_space == "sRGB"


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_config(tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError):
        load_sensor(tmp_path / "absent.json")


# ---------------------------------------------------------------------------
# Strictness
# ---------------------------------------------------------------------------


def test_unknown_field_rejected(tmp_path: Path) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data["unknown_field"] = 1
    path = _write_json(tmp_path, data, "bad.json")
    with pytest.raises(ValidationError):
        load_sensor(path)


def test_unknown_isp_field_rejected(tmp_path: Path) -> None:
    data = _preset_payload(ISP_PRESET)
    data["sharpness"] = 1.0
    path = _write_json(tmp_path, data, "bad_isp.json")
    with pytest.raises(ValidationError):
        load_isp(path)


def test_provenance_is_required(tmp_path: Path) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data.pop("provenance")
    with pytest.raises(ValidationError, match="provenance"):
        load_sensor(_write_json(tmp_path, data))


# ---------------------------------------------------------------------------
# Cross-field physics rules
# ---------------------------------------------------------------------------


def test_hcg_full_well_above_lcg_rejected(tmp_path: Path) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data["modes"]["HCG"]["full_well_e"] = 7000
    with pytest.raises(ValidationError, match="HCG full_well_e"):
        load_sensor(_write_json(tmp_path, data))


def test_modes_must_define_both_gain_modes(tmp_path: Path) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data["modes"].pop("HCG")
    with pytest.raises(ValidationError, match="exactly LCG and HCG"):
        load_sensor(_write_json(tmp_path, data))


def test_analog_gain_default_outside_range_rejected(tmp_path: Path) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data["analog_gain"]["default_db"] = 30.0
    with pytest.raises(ValidationError, match="default_db"):
        load_sensor(_write_json(tmp_path, data))


def test_exposure_default_outside_range_rejected(tmp_path: Path) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data["exposure"]["default_s"] = 1.0
    with pytest.raises(ValidationError, match="default_s"):
        load_sensor(_write_json(tmp_path, data))


def test_adc_black_level_out_of_code_range_rejected(tmp_path: Path) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data["adc"]["black_level_dn"] = 1024
    with pytest.raises(ValidationError, match="black_level_dn"):
        load_sensor(_write_json(tmp_path, data))


def test_ccm_row_sum_rejected(tmp_path: Path) -> None:
    data = _preset_payload(ISP_PRESET)
    data["ccm"]["matrix"] = [[1.0, 0.1, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValidationError, match="sum to 1"):
        load_isp(_write_json(tmp_path, data, "bad_ccm.json"))


def test_ccm_row_sum_rejected_when_constructed_directly() -> None:
    with pytest.raises(ValidationError, match="row 0"):
        CCMConfig(matrix=[[0.5, 0.4, 0.4], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_spectral_crosstalk_row_sum_rejected(tmp_path: Path) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data["spectral"]["spectral_crosstalk"] = [[0.9, 0.1, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValidationError, match="sum to 1"):
        load_sensor(_write_json(tmp_path, data, "bad_crosstalk.json"))


def test_wavelength_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError, match="greater than start"):
        WavelengthRange(start=830.0, stop=360.0, step=5.0)
    with pytest.raises(ValidationError, match="must not exceed"):
        WavelengthRange(start=360.0, stop=400.0, step=50.0)


def test_full_well_vs_adc_warning(tmp_path: Path) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data["adc"]["full_scale_uv"] = 100.0
    path = _write_json(tmp_path, data, "adc_limited.json")
    report = validate_config(path)
    assert report.errors == []
    assert any("ADC" in warning and "full well" in warning for warning in report.warnings)


def test_shipped_preset_validates_without_errors_or_warnings() -> None:
    report = validate_config(SENSOR_PRESET)
    assert report.valid
    assert report.errors == []
    assert report.warnings == []
    assert validate_config(ISP_PRESET).valid


# ---------------------------------------------------------------------------
# Validation reports
# ---------------------------------------------------------------------------


def test_validation_report_defaults_to_valid() -> None:
    report = ValidationReport()
    assert report.valid
    assert report.errors == []
    assert report.warnings == []


def test_validate_config_reports_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    report = validate_config(path)
    assert not report.valid
    assert any("JSON" in error for error in report.errors)


def test_validate_config_reports_schema_errors_without_raising(tmp_path: Path) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data["unknown_field"] = 1
    report = validate_config(_write_json(tmp_path, data, "bad.json"))
    assert not report.valid
    assert any("unknown_field" in error for error in report.errors)


def test_validate_config_rejects_unrecognised_payload(tmp_path: Path) -> None:
    report = validate_config(_write_json(tmp_path, {"hello": "world"}, "unknown.json"))
    assert not report.valid
    assert any("sensor" in error for error in report.errors)


def test_validate_config_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    report = validate_config(path)
    assert not report.valid
    assert any("object" in error for error in report.errors)


def test_validate_config_rejects_mixed_sensor_and_isp_fields(tmp_path: Path) -> None:
    data = _preset_payload(SENSOR_PRESET)
    data["demosaic"] = {"enabled": True, "method": "mhc"}
    report = validate_config(_write_json(tmp_path, data, "mixed.json"))
    assert not report.valid
    assert any("mixes sensor and ISP" in error for error in report.errors)


# ---------------------------------------------------------------------------
# Schema export
# ---------------------------------------------------------------------------


def test_schema_export_contains_units() -> None:
    sensor_text = json.dumps(sensor_schema())
    for field_name in (
        "pitch_um",
        "fill_factor",
        "microlens_gain",
        "conversion_gain_uv_per_e",
        "full_well_e",
        "read_noise_e_rms",
        "dark_current_e_per_s",
        "min_db",
        "max_db",
        "step_db",
        "default_db",
        "full_scale_uv",
        "black_level_dn",
        "digital_gain",
        "min_s",
        "max_s",
        "default_s",
        "reference_c",
        "default_c",
        "dark_current_doubling_k",
        "dead_rate",
        "hot_rate",
        "stuck_rate",
        "qe_factor",
        "factor",
    ):
        assert field_name in sensor_text, field_name

    isp_text = json.dumps(isp_schema())
    for field_name in (
        "luma_strength",
        "chroma_strength",
        "radius",
        "threshold",
        "luma_only",
        "exposure_compensation_ev",
        "gamma",
        "bit_depth",
        "color_space",
    ):
        assert field_name in isp_text, field_name


def test_schema_export_titles() -> None:
    assert sensor_schema()["title"] == "SensorConfig"
    assert isp_schema()["title"] == "ISPConfig"


# ---------------------------------------------------------------------------
# Lens geometry
# ---------------------------------------------------------------------------


def test_lens_field_of_view_and_pixels_per_degree() -> None:
    lens = LensConfig(
        f_number=1.8,
        focal_length_mm=6.4,
        sensor_diagonal_mm=11.46,
        transmittance=0.9,
    )
    assert lens.field_of_view_deg == pytest.approx(84.0, abs=1.0)
    assert lens.pixels_per_degree(1.12) == pytest.approx(99.73, rel=1e-3)
    with pytest.raises(ValueError, match="pixel pitch"):
        lens.pixels_per_degree(0.0)


def test_lens_rejects_invalid_optics() -> None:
    with pytest.raises(ValidationError):
        LensConfig(
            f_number=0.0,
            focal_length_mm=6.4,
            sensor_diagonal_mm=11.46,
            transmittance=0.9,
        )
    with pytest.raises(ValidationError):
        LensConfig(
            f_number=1.8,
            focal_length_mm=6.4,
            sensor_diagonal_mm=11.46,
            transmittance=1.5,
        )


def test_lens_accepts_spectral_transmittance_path() -> None:
    lens = LensConfig(
        f_number=1.8,
        focal_length_mm=6.4,
        sensor_diagonal_mm=11.46,
        transmittance="data/spectra/transmittance.csv",
    )
    assert lens.transmittance == "data/spectra/transmittance.csv"


# ---------------------------------------------------------------------------
# Model interoperability
# ---------------------------------------------------------------------------


def test_sensor_config_json_round_trip() -> None:
    cfg = load_sensor(SENSOR_PRESET)
    restored = SensorConfig.model_validate(cfg.model_dump(mode="json"))
    assert restored == cfg


def test_isp_config_json_round_trip() -> None:
    cfg = load_isp(ISP_PRESET)
    restored = ISPConfig.model_validate(cfg.model_dump(mode="json"))
    assert restored == cfg


def test_mono_sensor_config_fixture(mono_sensor_config: SensorConfig) -> None:
    assert mono_sensor_config.cfa is CFAPattern.MONO
    assert mono_sensor_config.modes["LCG"].full_well_e == 6000
    assert mono_sensor_config.pixel.pitch_um == pytest.approx(1.12)


def test_lsc_map_accepts_arrays_and_json_lists() -> None:
    config = LSCConfig(mode="flat_field", map=np.full((4, 4), 2.0))
    assert isinstance(config.map, np.ndarray)
    assert config.map.shape == (4, 4)

    restored = LSCConfig.model_validate(config.model_dump(mode="json"))
    np.testing.assert_allclose(restored.map, config.map)


def test_lsc_flat_field_requires_a_map() -> None:
    with pytest.raises(ValidationError, match="flat_field"):
        LSCConfig(mode="flat_field")


def test_lsc_radial_requires_coefficients() -> None:
    with pytest.raises(ValidationError, match="radial"):
        LSCConfig(mode="radial")
    assert LSCConfig(mode="radial", coefficients=[1.0, -0.2]).coefficients == [1.0, -0.2]
