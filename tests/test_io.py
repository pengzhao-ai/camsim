"""Tests for RAW-frame and rendered-image file input/output.

The IO layer is where an in-memory array becomes a file on disk, and the two
formats carry very different risks:

* A **RAW frame** is integer ADC code. Saving it as a plain ``.npy`` without
  its sidecar loses the colour-filter pattern, black level, bit depth, gains,
  readout mode, seed, and masks, and a later ISP call would either reject the
  frame or silently mis-scale it. The sidecar is therefore mandatory on load,
  and masks travel in a compressed ``.npz`` because a megapixel boolean array
  does not belong in JSON text.
* A **rendered image** is display referred and already tone encoded. PNG and
  TIFF store 8-bit codes (16-bit grayscale TIFF is the one precision upgrade
  Pillow can write), so a round trip is quantized, and the sidecar records the
  colour space and channel layout that the pixel values alone cannot express.
  Lossy formats are rejected outright: JPEG ringing would change every metric
  computed from the file.

The tests exercise exact RAW round trips, mask preservation, deterministic
filenames, sidecar corruption and missing-file failures, 8-bit PNG and 16-bit
mono TIFF quantization, JSON-safe metadata, colour-space and channel
inference, and the prohibition on lossy or unknown formats.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from camsim.core.frames import CFAPattern, RenderedImage
from camsim.io.images import load_image, save_image
from camsim.io.raw import load_raw, save_raw

# ---------------------------------------------------------------------------
# Brief: the two verbatim contracts
# ---------------------------------------------------------------------------


def test_raw_roundtrip(tmp_path, raw_frame):
    path = save_raw(raw_frame, tmp_path)
    loaded = load_raw(path)
    np.testing.assert_array_equal(loaded.data, raw_frame.data)
    assert loaded.cfa == raw_frame.cfa
    assert loaded.black_level_dn == raw_frame.black_level_dn


def test_image_roundtrip(tmp_path):
    from camsim.core.frames import RenderedImage

    img = RenderedImage(
        data=np.full((16, 16, 3), 0.5, np.float32), color_space="sRGB", channels="RGB"
    )
    path = save_image(img, tmp_path / "out.png")
    loaded = load_image(path)
    assert loaded.channels == "RGB"
    np.testing.assert_allclose(loaded.data, img.data, atol=1.0 / 255.0)


# ---------------------------------------------------------------------------
# RAW frames
# ---------------------------------------------------------------------------


def _read_sidecar(path: Path) -> dict[str, object]:
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def test_raw_sidecar_records_every_scalar_field(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    payload = _read_sidecar(data_path)

    assert payload["kind"] == "camsim.raw_frame"
    assert payload["schema_version"] == "1.0"
    data = payload["data"]
    assert isinstance(data, dict)
    assert data["file"] == data_path.name
    assert data["shape"] == list(raw_frame.data.shape)
    assert data["dtype"] == "uint16"

    frame = payload["frame"]
    assert isinstance(frame, dict)
    assert frame["cfa"] == raw_frame.cfa.value
    assert frame["bit_depth"] == raw_frame.bit_depth
    assert frame["black_level_dn"] == raw_frame.black_level_dn
    assert frame["exposure_s"] == raw_frame.exposure_s
    assert frame["analog_gain_db"] == raw_frame.analog_gain_db
    assert frame["digital_gain"] == raw_frame.digital_gain
    assert frame["mode"] == raw_frame.mode
    assert frame["temperature_c"] == raw_frame.temperature_c
    assert frame["seed"] == raw_frame.seed
    assert frame["frame_index"] == raw_frame.frame_index
    assert frame["pixel_pitch_um"] == raw_frame.pixel_pitch_um
    assert frame["sensor_config_hash"] == raw_frame.sensor_config_hash

    masks = payload["masks"]
    assert isinstance(masks, dict)
    assert masks["file"] == f"{data_path.stem}_masks.npz"
    assert set(masks["arrays"]) == {"defect_mask", "pdaf_mask"}


def test_load_raw_preserves_masks(tmp_path: Path, raw_frame) -> None:
    """Masks round-trip exactly, including pixels that differ from the default."""
    defect = np.zeros(raw_frame.data.shape, dtype=bool)
    defect[0, 0] = True
    defect[-1, -1] = True
    pdaf = np.ones(raw_frame.data.shape, dtype=bool)
    pdaf[1, 1] = False
    framed = replace(raw_frame, defect_mask=defect, pdaf_mask=pdaf)

    data_path = save_raw(framed, tmp_path)
    loaded = load_raw(data_path)

    assert loaded.defect_mask is not None
    assert loaded.pdaf_mask is not None
    np.testing.assert_array_equal(loaded.defect_mask, defect)
    np.testing.assert_array_equal(loaded.pdaf_mask, pdaf)
    assert loaded.defect_mask[0, 0]
    assert not loaded.pdaf_mask[1, 1]


def test_load_raw_accepts_the_sidecar_path(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    loaded = load_raw(data_path.with_suffix(".json"))
    np.testing.assert_array_equal(loaded.data, raw_frame.data)
    assert loaded.seed == raw_frame.seed


def test_save_raw_is_deterministic(tmp_path: Path, raw_frame) -> None:
    first = save_raw(raw_frame, tmp_path / "a")
    second = save_raw(raw_frame, tmp_path / "b")

    assert first.name == f"raw_{raw_frame.frame_index:06d}_seed{raw_frame.seed}.npy"
    assert first.name == second.name
    assert first.read_bytes() == second.read_bytes()
    assert first.with_suffix(".json").read_bytes() == second.with_suffix(".json").read_bytes()


def test_save_raw_omits_the_mask_archive_without_masks(tmp_path: Path, raw_frame) -> None:
    bare = replace(raw_frame, defect_mask=None, pdaf_mask=None)
    data_path = save_raw(bare, tmp_path)
    payload = _read_sidecar(data_path)
    assert payload["masks"] is None
    assert not data_path.with_name(f"{data_path.stem}_masks.npz").exists()
    loaded = load_raw(data_path)
    assert loaded.defect_mask is None
    assert loaded.pdaf_mask is None


def test_load_raw_rejects_a_missing_sidecar(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    data_path.with_suffix(".json").unlink()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_raw(data_path)


def test_load_raw_rejects_a_missing_data_file(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    sidecar = data_path.with_suffix(".json")
    data_path.unlink()
    with pytest.raises(FileNotFoundError, match="data file"):
        load_raw(sidecar)


def test_load_raw_rejects_missing_masks(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    data_path.with_name(f"{data_path.stem}_masks.npz").unlink()
    with pytest.raises(FileNotFoundError, match="mask"):
        load_raw(data_path)


def test_load_raw_rejects_unknown_schema_versions(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    sidecar = data_path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["schema_version"] = "9.9"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_raw(data_path)


def test_load_raw_rejects_a_kind_mismatch(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    sidecar = data_path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["kind"] = "camsim.rendered_image"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="kind"):
        load_raw(data_path)


def test_load_raw_rejects_a_wrong_dtype(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    np.save(data_path, raw_frame.data.astype(np.float32))
    with pytest.raises(ValueError, match="uint16"):
        load_raw(data_path)


def test_load_raw_rejects_a_shape_mismatch(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    np.save(data_path, np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(ValueError, match="shape"):
        load_raw(data_path)


def test_load_raw_rejects_codes_above_the_bit_depth(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    sidecar = data_path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    frame = payload["frame"]
    frame["bit_depth"] = 8
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    np.save(data_path, np.full(raw_frame.data.shape, 300, dtype=np.uint16))
    with pytest.raises(ValueError, match="bit depth"):
        load_raw(data_path)


def test_load_raw_rejects_a_mask_shape_mismatch(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    np.savez_compressed(
        data_path.with_name(f"{data_path.stem}_masks.npz"),
        defect_mask=np.zeros((4, 4), dtype=bool),
        pdaf_mask=np.zeros(raw_frame.data.shape, dtype=bool),
    )
    with pytest.raises(ValueError, match="shape"):
        load_raw(data_path)


def test_load_raw_rejects_unknown_extensions(tmp_path: Path, raw_frame) -> None:
    with pytest.raises(ValueError, match=r"\.npy"):
        load_raw(tmp_path / "frame.tiff")


def test_save_raw_rejects_a_non_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="RawFrame"):
        save_raw(np.zeros((4, 4), dtype=np.uint16), tmp_path)  # type: ignore[arg-type]


def _corrupt_sidecar(data_path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    sidecar = data_path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    mutate(payload)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")


def test_load_raw_rejects_malformed_json(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    data_path.with_suffix(".json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_raw(data_path)


def test_load_raw_rejects_a_non_object_sidecar(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    data_path.with_suffix(".json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_raw(data_path)


def test_load_raw_rejects_a_missing_frame_field(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        frame = payload["frame"]
        assert isinstance(frame, dict)
        del frame["seed"]

    _corrupt_sidecar(data_path, mutate)
    with pytest.raises(ValueError, match="seed"):
        load_raw(data_path)


def test_load_raw_rejects_an_unsafe_data_file_name(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        data = payload["data"]
        assert isinstance(data, dict)
        data["file"] = "../escaped.npy"

    _corrupt_sidecar(data_path, mutate)
    with pytest.raises(ValueError, match="plain file name"):
        load_raw(data_path)


@pytest.mark.parametrize("shape_meta", ["32x32", [0, 32], [32], [True, 32]])
def test_load_raw_rejects_malformed_shape_metadata(
    tmp_path: Path, raw_frame, shape_meta: object
) -> None:
    data_path = save_raw(raw_frame, tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        data = payload["data"]
        assert isinstance(data, dict)
        data["shape"] = shape_meta

    _corrupt_sidecar(data_path, mutate)
    with pytest.raises(ValueError, match="shape"):
        load_raw(data_path)


def test_load_raw_rejects_a_corrupt_npy(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    data_path.write_bytes(b"not a numpy file")
    with pytest.raises(ValueError, match="not a valid"):
        load_raw(data_path)


def test_load_raw_rejects_an_extra_array_dimension(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    np.save(data_path, np.zeros((*raw_frame.data.shape, 1), dtype=np.uint16))
    with pytest.raises(ValueError, match="2-D"):
        load_raw(data_path)


def test_load_raw_rejects_a_mismatched_sidecar_pair(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    renamed = tmp_path / "other.npy"
    data_path.rename(renamed)
    shutil.copy(data_path.with_suffix(".json"), renamed.with_suffix(".json"))
    with pytest.raises(ValueError, match="does not match"):
        load_raw(renamed)


def test_load_raw_detects_a_missing_data_file_via_the_npy_path(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    data_path.unlink()
    with pytest.raises(FileNotFoundError, match="data file"):
        load_raw(data_path)


def test_load_raw_rejects_a_bad_mask_block(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload["masks"] = "not-a-mapping"

    _corrupt_sidecar(data_path, mutate)
    with pytest.raises(ValueError, match="JSON object"):
        load_raw(data_path)


@pytest.mark.parametrize(
    "arrays",
    [[], ["defect_mask", "other_mask"], ["defect_mask", "defect_mask"]],
)
def test_load_raw_rejects_bad_mask_lists(tmp_path: Path, raw_frame, arrays: list[str]) -> None:
    data_path = save_raw(raw_frame, tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        masks = payload["masks"]
        assert isinstance(masks, dict)
        masks["arrays"] = arrays

    _corrupt_sidecar(data_path, mutate)
    with pytest.raises(ValueError, match="mask"):
        load_raw(data_path)


def test_load_raw_rejects_a_mask_missing_from_the_archive(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    np.savez_compressed(
        data_path.with_name(f"{data_path.stem}_masks.npz"),
        defect_mask=np.zeros(raw_frame.data.shape, dtype=bool),
    )
    with pytest.raises(ValueError, match="missing from the archive"):
        load_raw(data_path)


def test_load_raw_rejects_a_non_boolean_mask(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    np.savez_compressed(
        data_path.with_name(f"{data_path.stem}_masks.npz"),
        defect_mask=np.zeros(raw_frame.data.shape, dtype=np.uint8),
        pdaf_mask=np.zeros(raw_frame.data.shape, dtype=bool),
    )
    with pytest.raises(ValueError, match="dtype bool"):
        load_raw(data_path)


def test_load_raw_rejects_a_corrupt_mask_archive(tmp_path: Path, raw_frame) -> None:
    data_path = save_raw(raw_frame, tmp_path)
    data_path.with_name(f"{data_path.stem}_masks.npz").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="mask archive"):
        load_raw(data_path)


# ---------------------------------------------------------------------------
# Rendered images
# ---------------------------------------------------------------------------


def _rgb_image(value: float = 0.5, size: tuple[int, int] = (16, 16)) -> RenderedImage:
    return RenderedImage(
        data=np.full((*size, 3), value, dtype=np.float32),
        color_space="sRGB",
        channels="RGB",
    )


def _mono_image(size: tuple[int, int] = (16, 16)) -> RenderedImage:
    ramp = np.linspace(0.0, 1.0, size[0] * size[1], dtype=np.float32).reshape(*size, 1)
    return RenderedImage(data=ramp, color_space="sRGB", channels="MONO")


def test_png_roundtrip_preserves_colour_space_and_metadata(tmp_path: Path) -> None:
    image = RenderedImage(
        data=np.full((8, 8, 3), 0.25, dtype=np.float32),
        color_space="DisplayP3",
        channels="RGB",
        metadata={
            "numpy_int": np.int64(7),
            "numpy_float": np.float32(0.5),
            "vector": np.array([1, 2, 3], dtype=np.int16),
            "pair": (1.0, 2.0),
            "nested": {"gain": np.float64(1.5), "name": "x"},
            "undefined": np.float64("nan"),
        },
    )
    path = save_image(image, tmp_path / "display.png")
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))

    assert payload["kind"] == "camsim.rendered_image"
    assert payload["schema_version"] == "1.0"
    assert payload["format"] == "PNG"
    assert payload["bit_depth"] == 8
    assert payload["color_space"] == "DisplayP3"
    assert payload["channels"] == "RGB"
    assert "linear_rgb" not in payload
    assert "stages" not in payload
    assert payload["metadata"] == {
        "numpy_int": 7,
        "numpy_float": 0.5,
        "vector": [1, 2, 3],
        "pair": [1.0, 2.0],
        "nested": {"gain": 1.5, "name": "x"},
        "undefined": None,
    }

    loaded = load_image(path)
    assert loaded.color_space == "DisplayP3"
    assert loaded.channels == "RGB"
    assert loaded.metadata == payload["metadata"]
    np.testing.assert_allclose(loaded.data, image.data, atol=1.0 / 255.0)


def test_mono_tiff_roundtrip_is_16_bit(tmp_path: Path) -> None:
    image = _mono_image()
    path = save_image(image, tmp_path / "mono.tiff")
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))

    assert payload["format"] == "TIFF"
    assert payload["bit_depth"] == 16
    assert payload["channels"] == "MONO"

    loaded = load_image(path)
    assert loaded.channels == "MONO"
    assert loaded.data.shape == image.data.shape
    np.testing.assert_allclose(loaded.data, image.data, atol=1.0 / 65535.0)


def test_rgb_tiff_is_written_at_8_bit(tmp_path: Path) -> None:
    image = _rgb_image(0.75)
    path = save_image(image, tmp_path / "rgb.tif")
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["format"] == "TIFF"
    assert payload["bit_depth"] == 8
    loaded = load_image(path)
    np.testing.assert_allclose(loaded.data, image.data, atol=1.0 / 255.0)


def test_mono_png_is_written_at_8_bit(tmp_path: Path) -> None:
    path = save_image(_mono_image(), tmp_path / "mono.png")
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["bit_depth"] == 8
    assert load_image(path).channels == "MONO"


def test_save_image_appends_a_png_suffix(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "capture")
    assert path == tmp_path / "capture.png"
    assert path.is_file()
    assert (tmp_path / "capture.json").is_file()


@pytest.mark.parametrize("suffix", [".jpg", ".jpeg", ".webp", ".bmp", ".gif"])
def test_save_image_rejects_lossy_and_unknown_formats(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(ValueError, match=r"PNG|TIFF|lossy|supported"):
        save_image(_rgb_image(), tmp_path / f"capture{suffix}")


def test_load_image_rejects_lossy_and_unknown_formats(tmp_path: Path) -> None:
    path = tmp_path / "capture.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match=r"PNG|TIFF|lossy|supported"):
        load_image(path)


def test_load_image_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="image"):
        load_image(tmp_path / "missing.png")


def test_load_image_without_a_sidecar_infers_srgb_and_channels(tmp_path: Path) -> None:
    rgb_path = save_image(_rgb_image(), tmp_path / "rgb.png")
    rgb_path.with_suffix(".json").unlink()
    rgb = load_image(rgb_path)
    assert rgb.color_space == "sRGB"
    assert rgb.channels == "RGB"
    assert rgb.metadata == {}

    mono_path = save_image(_mono_image(), tmp_path / "mono.png")
    mono_path.with_suffix(".json").unlink()
    mono = load_image(mono_path)
    assert mono.color_space == "sRGB"
    assert mono.channels == "MONO"


def test_load_image_rejects_a_channel_mismatch(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["channels"] = "MONO"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="channels"):
        load_image(path)


def test_load_image_rejects_an_unknown_schema_version(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["schema_version"] = "2.0"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_image(path)


def test_load_image_rejects_a_kind_mismatch(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["kind"] = "camsim.raw_frame"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="kind"):
        load_image(path)


def test_load_image_rejects_a_renamed_sidecar(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["file"] = "other.png"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="file"):
        load_image(path)


def test_load_image_rejects_unsupported_pillow_modes(tmp_path: Path) -> None:
    path = tmp_path / "palette.png"
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(source).convert("P").save(path)
    with pytest.raises(ValueError, match=r"mode|PNG|TIFF"):
        load_image(path)


def test_load_image_rejects_a_corrupt_png(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not a png")
    with pytest.raises(OSError):
        load_image(path)


def test_save_image_rejects_a_non_image_object(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="RenderedImage"):
        save_image(np.zeros((4, 4, 3), dtype=np.float32), tmp_path / "out.png")  # type: ignore[arg-type]


def test_save_image_serializes_enum_path_and_bytes_metadata(tmp_path: Path) -> None:
    image = RenderedImage(
        data=np.full((4, 4, 3), 0.5, dtype=np.float32),
        color_space="sRGB",
        channels="RGB",
        metadata={
            "cfa": CFAPattern.RGGB,
            "path": Path("results/captures"),
            "blob": b"\x01\x02",
        },
    )
    path = save_image(image, tmp_path / "out.png")
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["metadata"] == {
        "cfa": "RGGB",
        "path": "results/captures",
        "blob": "0102",
    }
    assert load_image(path).metadata["cfa"] == "RGGB"


def test_save_image_rejects_unserialisable_metadata(tmp_path: Path) -> None:
    image = RenderedImage(
        data=np.full((4, 4, 3), 0.5, dtype=np.float32),
        color_space="sRGB",
        channels="RGB",
        metadata={"object": object()},
    )
    with pytest.raises(ValueError, match="JSON"):
        save_image(image, tmp_path / "out.png")


def _corrupt_image_sidecar(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    mutate(payload)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")


def test_load_image_rejects_malformed_json(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")
    path.with_suffix(".json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_image(path)


def test_load_image_rejects_a_non_object_sidecar(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")
    path.with_suffix(".json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_image(path)


def test_load_image_rejects_an_unsafe_file_name(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")

    def mutate(payload: dict[str, object]) -> None:
        payload["file"] = "../escaped.png"

    _corrupt_image_sidecar(path, mutate)
    with pytest.raises(ValueError, match="plain file name"):
        load_image(path)


def test_load_image_rejects_a_renamed_lossy_file(tmp_path: Path) -> None:
    """A JPEG renamed to .png is detected from its actual container format."""
    path = tmp_path / "photo.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path, format="JPEG")
    with pytest.raises(ValueError, match="not PNG or TIFF"):
        load_image(path)


def test_load_image_rejects_a_format_mismatch(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")

    def mutate(payload: dict[str, object]) -> None:
        payload["format"] = "TIFF"

    _corrupt_image_sidecar(path, mutate)
    with pytest.raises(ValueError, match="format"):
        load_image(path)


@pytest.mark.parametrize("declared", [12, True, "8", 16])
def test_load_image_rejects_bad_bit_depth_metadata(tmp_path: Path, declared: object) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")

    def mutate(payload: dict[str, object]) -> None:
        payload["bit_depth"] = declared

    _corrupt_image_sidecar(path, mutate)
    with pytest.raises(ValueError, match="bit"):
        load_image(path)


def test_load_image_rejects_unknown_channels_metadata(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")

    def mutate(payload: dict[str, object]) -> None:
        payload["channels"] = "GRAY"

    _corrupt_image_sidecar(path, mutate)
    with pytest.raises(ValueError, match="channels"):
        load_image(path)


def test_load_image_rejects_unknown_colour_space_metadata(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")

    def mutate(payload: dict[str, object]) -> None:
        payload["color_space"] = "AdobeRGB"

    _corrupt_image_sidecar(path, mutate)
    with pytest.raises(ValueError, match="color_space"):
        load_image(path)


def test_load_image_rejects_non_mapping_metadata(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")

    def mutate(payload: dict[str, object]) -> None:
        payload["metadata"] = [1, 2, 3]

    _corrupt_image_sidecar(path, mutate)
    with pytest.raises(ValueError, match="metadata"):
        load_image(path)


def test_load_image_accepts_null_metadata(tmp_path: Path) -> None:
    path = save_image(_rgb_image(), tmp_path / "rgb.png")

    def mutate(payload: dict[str, object]) -> None:
        payload["metadata"] = None

    _corrupt_image_sidecar(path, mutate)
    assert load_image(path).metadata == {}
