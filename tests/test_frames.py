"""Tests for RAW frames, stage traces, and rendered-image contracts.

The frame contract is the hand-off between simulation and ISP: RAW values are
unsigned integer codes offset by a black level, while rendered images are
display-referred floats in ``[0, 1]``. These tests pin down validation,
provenance round-trips, and the mono/RGB shape distinction so later stages
cannot silently confuse the two representations.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from camsim.core.frames import CFAPattern, RawFrame, RenderedImage, StageOutput


def make_raw_frame(**overrides: Any) -> RawFrame:
    """Build a valid 10-bit RGGB frame, overriding individual metadata fields."""
    values: dict[str, Any] = {
        "data": np.zeros((4, 4), dtype=np.uint16),
        "cfa": CFAPattern.RGGB,
        "bit_depth": 10,
        "black_level_dn": 64,
        "exposure_s": 1e-3,
        "analog_gain_db": 0.0,
        "digital_gain": 1.0,
        "mode": "LCG",
        "temperature_c": 25.0,
        "seed": 1,
        "frame_index": 0,
        "pixel_pitch_um": 1.12,
        "sensor_config_hash": "test",
    }
    values.update(overrides)
    return RawFrame(**values)


def test_raw_frame_validates_bit_depth_and_black_level() -> None:
    raw = RawFrame(
        data=np.zeros((4, 4), dtype=np.uint16),
        cfa=CFAPattern.RGGB,
        bit_depth=10,
        black_level_dn=64,
        exposure_s=1e-3,
        analog_gain_db=0.0,
        digital_gain=1.0,
        mode="LCG",
        temperature_c=25.0,
        seed=1,
        frame_index=0,
        pixel_pitch_um=1.12,
        sensor_config_hash="test",
    )
    assert raw.black_level_dn == 64
    with pytest.raises(ValueError, match="black level"):
        RawFrame(**{**raw.__dict__, "black_level_dn": 2000})


def test_raw_frame_validates_data_container_and_bit_depth() -> None:
    with pytest.raises(ValueError, match="data"):
        make_raw_frame(data=np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="data"):
        make_raw_frame(data=np.zeros((4, 4, 1), dtype=np.uint16))
    with pytest.raises(ValueError, match="data"):
        make_raw_frame(data=np.full((4, 4), 1024, dtype=np.uint16))
    with pytest.raises(ValueError, match="bit depth"):
        make_raw_frame(bit_depth=7)
    with pytest.raises(ValueError, match="bit depth"):
        make_raw_frame(bit_depth=17)
    with pytest.raises(ValueError, match="bit depth"):
        make_raw_frame(bit_depth=10.0)


def test_raw_frame_validates_acquisition_metadata() -> None:
    with pytest.raises(ValueError, match="exposure"):
        make_raw_frame(exposure_s=0.0)
    with pytest.raises(ValueError, match="exposure"):
        make_raw_frame(exposure_s=np.inf)
    with pytest.raises(ValueError, match="analog gain"):
        make_raw_frame(analog_gain_db=-0.5)
    with pytest.raises(ValueError, match="analog gain"):
        make_raw_frame(analog_gain_db=np.nan)
    with pytest.raises(ValueError, match="digital gain"):
        make_raw_frame(digital_gain=0.0)
    with pytest.raises(ValueError, match="digital gain"):
        make_raw_frame(digital_gain=np.inf)
    with pytest.raises(ValueError, match="temperature"):
        make_raw_frame(temperature_c=np.nan)
    with pytest.raises(ValueError, match="pixel pitch"):
        make_raw_frame(pixel_pitch_um=0.0)
    with pytest.raises(ValueError, match="seed"):
        make_raw_frame(seed=-1)
    with pytest.raises(ValueError, match="frame_index"):
        make_raw_frame(frame_index=-1)
    with pytest.raises(ValueError, match="frame_index"):
        make_raw_frame(frame_index=1.0)
    with pytest.raises(ValueError, match="sensor_config_hash"):
        make_raw_frame(sensor_config_hash="")


def test_raw_frame_validates_cfa_and_mode() -> None:
    with pytest.raises(ValueError, match="cfa"):
        make_raw_frame(cfa="NOPE")
    assert make_raw_frame(cfa="RGGB").cfa is CFAPattern.RGGB
    with pytest.raises(ValueError, match="mode"):
        make_raw_frame(mode="")


def test_raw_frame_masks_are_optional_boolean_and_shape_matched() -> None:
    defect = np.zeros((4, 4), dtype=bool)
    pdaf = np.eye(4, dtype=bool)

    raw = make_raw_frame(defect_mask=defect, pdaf_mask=pdaf)

    assert raw.defect_mask is defect
    assert raw.pdaf_mask is pdaf
    with pytest.raises(ValueError, match="defect_mask"):
        make_raw_frame(defect_mask=np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="pdaf_mask"):
        make_raw_frame(pdaf_mask=np.zeros((2, 4), dtype=bool))


def test_raw_frame_with_data_preserves_metadata() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    raw = make_raw_frame(defect_mask=mask)
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)

    updated = raw.with_data(data)

    assert updated is not raw
    np.testing.assert_array_equal(updated.data, data)
    assert updated.cfa is raw.cfa
    assert updated.bit_depth == raw.bit_depth
    assert updated.black_level_dn == raw.black_level_dn
    assert updated.exposure_s == raw.exposure_s
    assert updated.sensor_config_hash == raw.sensor_config_hash
    assert updated.defect_mask is mask
    with pytest.raises(ValueError, match="data"):
        raw.with_data(np.zeros((4, 4), dtype=np.float32))


def test_raw_frame_metadata_json_roundtrip() -> None:
    raw = make_raw_frame(
        data=np.arange(16, dtype=np.uint16).reshape(4, 4),
        defect_mask=np.array([[True, False, False, True]] * 4),
        pdaf_mask=None,
    )

    payload = json.loads(json.dumps(raw.to_dict()))

    assert isinstance(payload["data"], list)
    assert isinstance(payload["defect_mask"], list)
    assert payload["pdaf_mask"] is None
    restored = RawFrame.from_dict(payload)
    np.testing.assert_array_equal(restored.data, raw.data)
    assert restored.data.dtype == np.uint16
    np.testing.assert_array_equal(restored.defect_mask, raw.defect_mask)
    assert restored.pdaf_mask is None
    for field_name in (
        "cfa",
        "bit_depth",
        "black_level_dn",
        "exposure_s",
        "analog_gain_db",
        "digital_gain",
        "mode",
        "temperature_c",
        "seed",
        "frame_index",
        "pixel_pitch_um",
        "sensor_config_hash",
    ):
        assert getattr(restored, field_name) == getattr(raw, field_name)


def test_cfa_pattern_members_and_string_values() -> None:
    assert [member.value for member in CFAPattern] == [
        "RGGB",
        "BGGR",
        "GRBG",
        "GBRG",
        "MONO",
        "RCCC",
    ]
    assert CFAPattern("RGGB") is CFAPattern.RGGB
    assert isinstance(CFAPattern.RCCC, str)


def test_rendered_image_mono_and_rgb_shapes() -> None:
    rgb = RenderedImage(data=np.zeros((4, 4, 3), np.float32), color_space="sRGB", channels="RGB")
    mono = RenderedImage(data=np.zeros((4, 4, 1), np.float32), color_space="sRGB", channels="MONO")
    assert rgb.channels == "RGB"
    assert mono.channels == "MONO"
    with pytest.raises(ValueError, match="channels"):
        RenderedImage(data=np.zeros((4, 4, 3), np.float32), color_space="sRGB", channels="MONO")


def test_rendered_image_display_range() -> None:
    with pytest.raises(ValueError, match="range"):
        RenderedImage(
            data=np.full((4, 4, 3), -0.01, np.float32), color_space="sRGB", channels="RGB"
        )
    with pytest.raises(ValueError, match="range"):
        RenderedImage(data=np.full((4, 4, 3), 1.01, np.float32), color_space="sRGB", channels="RGB")
    with pytest.raises(ValueError, match="range"):
        RenderedImage(
            data=np.full((4, 4, 3), np.nan, np.float32), color_space="sRGB", channels="RGB"
        )
    near_one = RenderedImage(
        data=np.full((4, 4, 3), 1.0 + 1e-7, np.float32), color_space="sRGB", channels="RGB"
    )
    assert near_one.data.max() == pytest.approx(1.0, abs=1e-6)


def test_rendered_image_validates_channel_value_dtype_and_color_space() -> None:
    with pytest.raises(ValueError, match="channels"):
        RenderedImage(data=np.zeros((4, 4, 3), np.float32), color_space="sRGB", channels="RGBY")
    with pytest.raises(ValueError, match="color_space"):
        RenderedImage(data=np.zeros((4, 4, 3), np.float32), color_space="AdobeRGB", channels="RGB")
    with pytest.raises(ValueError, match="data"):
        RenderedImage(data=np.zeros((4, 4, 3), np.uint8), color_space="sRGB", channels="RGB")


def test_rendered_image_linear_rgb_shape_must_match_data() -> None:
    linear = np.zeros((4, 4, 3), np.float32)
    image = RenderedImage(
        data=np.zeros((4, 4, 3), np.float32),
        color_space="sRGB",
        channels="RGB",
        linear_rgb=linear,
    )

    assert image.linear_rgb is linear
    with pytest.raises(ValueError, match="linear_rgb"):
        RenderedImage(
            data=np.zeros((4, 4, 3), np.float32),
            color_space="sRGB",
            channels="RGB",
            linear_rgb=np.zeros((4, 4), np.float32),
        )


def test_stage_output_metadata_defaults_are_independent() -> None:
    first = StageOutput(data=np.zeros((1, 1), np.float32), description="blc")
    second = StageOutput(data=np.zeros((1, 1), np.float32), description="tone")

    assert first.metadata == {}
    assert first.metadata is not second.metadata


def test_rendered_image_json_roundtrip() -> None:
    stages = {
        "tone": StageOutput(
            data=np.full((4, 4, 3), 0.5, np.float32),
            description="sRGB OETF applied",
            metadata={"gamma": 2.2},
        )
    }
    image = RenderedImage(
        data=np.full((4, 4, 3), 0.25, np.float32),
        color_space="sRGB",
        channels="RGB",
        linear_rgb=np.full((4, 4, 3), 0.1, np.float32),
        stages=stages,
        metadata={"isp_config_hash": "abc"},
    )

    payload = json.loads(json.dumps(image.to_dict()))
    restored = RenderedImage.from_dict(payload)

    np.testing.assert_allclose(restored.data, image.data)
    np.testing.assert_allclose(restored.linear_rgb, image.linear_rgb)
    assert restored.color_space == "sRGB"
    assert restored.channels == "RGB"
    assert restored.metadata == {"isp_config_hash": "abc"}
    assert restored.stages is not None
    assert set(restored.stages) == {"tone"}
    np.testing.assert_allclose(restored.stages["tone"].data, stages["tone"].data)
    assert restored.stages["tone"].description == "sRGB OETF applied"
    assert restored.stages["tone"].metadata == {"gamma": 2.2}


def test_rendered_image_from_dict_without_optional_fields() -> None:
    image = RenderedImage(
        data=np.zeros((2, 2, 1), np.float32), color_space="DisplayP3", channels="MONO"
    )

    restored = RenderedImage.from_dict(image.to_dict())

    assert restored.linear_rgb is None
    assert restored.stages is None
    assert restored.metadata == {}
    assert restored.color_space == "DisplayP3"
