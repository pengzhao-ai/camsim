"""Tests for chart layouts, ROI extraction, and colour-reference binding.

A chart layout is the bridge between a rendered image and the metrics that
measure it: it says *where* each patch and edge lives, and *what colour* each
patch should be. Two properties make or break that bridge.

* **Coordinates are sensor pixels.** The scene renderer paints a target on an
  oversampled grid and box-averages down to the sensor grid; only the sensor
  grid is ever measured. A ROI that leaks onto the chart surround mixes two
  reflectances into one patch mean, so the ColorChecker ROIs are derived from
  the rectangles actually painted, inset from their edges, and validated for
  overlap and for containment in the frame.
* **Colour truth is illuminant-bound.** A ColorChecker's reflectance spectra
  are not its colour. The same chart under 3000 K and D65 has different CIE
  Lab values, so a layout carries the spectrally computed
  :class:`~camsim.core.color.ColorReference` for the illuminant actually used.
  Copying a D65 reference onto a warm-light capture produces a systematic
  delta-E offset that looks like a sensor colour error.

ROIs are half-open: a region ``(x, y, width, height)`` covers columns
``x .. x + width - 1`` and rows ``y .. y + height - 1``. That matches NumPy
slicing, so ``image[y : y + h, x : x + w]`` is exactly the region and two
rectangles that merely touch share no pixel. ``extract_patch`` returns a view,
not a copy; callers must not mutate it, because the same buffer backs the
source image.

Real-world parameter ranges: ColorChecker patches are about 40 mm square on a
215 x 280 mm chart, which at a 1.12 um pixel pitch and 6.4 mm focal length
covers tens of pixels per patch; analysis ROIs are usually inset 10-20 % from
the patch edge. Slanted-edge regions are hundreds of rows tall so the edge
angle fit averages out noise. Camera metadata on a layout uses micrometres for
pixel pitch (0.8-10 um) and millimetres for focal length (2-50 mm for phone to
large-format lenses).

Failure artefacts guarded here: transposing x and y silently measures the
wrong patch; an ROI touching or crossing a patch boundary reports a colour
between two patches; overlapping ROIs double-count pixels in a statistics
sweep; a layout with no patches and no edge gives ``measure_all`` nothing to
do; attaching a D65 reference to a 3000 K capture injects a white-point error;
and accepting NaN/infinite image samples poisons every downstream mean and
standard deviation.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from camsim.config.schema import LensConfig
from camsim.metrics.chart import ChartLayout, PatchROI, extract_patch
from camsim.scene.charts import colorchecker_layout, flat_patch_layout, slanted_edge_layout
from camsim.scene.illuminants import reference_lab
from camsim.scene.spectral_data import load_colorchecker
from camsim.scene.targets import render

# --------------------------------------------------------------------------- #
# Brief contract tests (verbatim from the task brief)
# --------------------------------------------------------------------------- #


def test_colorchecker_layout_has_24_patches_and_reference():
    layout = colorchecker_layout(illuminant="D65")
    assert len(layout.patches) == 24
    assert layout.reference is not None
    assert layout.reference.illuminant_name == "D65"


def test_extract_patch_returns_pixels():
    img = np.arange(100, dtype=np.float32).reshape(10, 10, 1)
    roi = PatchROI("p", 2, 2, 3, 3)
    assert extract_patch(img, roi).shape == (3, 3, 1)


def test_layout_json_roundtrip():
    layout = colorchecker_layout(illuminant="D65")
    restored = ChartLayout.from_dict(layout.to_dict())
    assert restored.name == layout.name
    assert len(restored.patches) == 24


# --------------------------------------------------------------------------- #
# Scene-owned convenience layouts
# --------------------------------------------------------------------------- #


def test_flat_patch_layout() -> None:
    layout = flat_patch_layout((48, 64))

    assert layout.name == "flat_patch"
    assert layout.patches == (PatchROI("flat", 0, 0, 64, 48),)
    assert layout.edge_roi is None
    assert layout.reference is None
    assert layout.pixel_pitch_um == pytest.approx(1.12)
    assert layout.focal_length_mm is None


def test_default_flat_patch_layout_is_full_frame() -> None:
    layout = flat_patch_layout()

    # The default shape is the renderer default, so the ROI covers every pixel.
    assert layout.patches == (PatchROI("flat", 0, 0, 256, 256),)


def test_slanted_edge_roi() -> None:
    layout = slanted_edge_layout((64, 96))

    assert layout.name == "slanted_edge"
    assert layout.patches == ()
    assert layout.edge_roi == PatchROI("edge", 24, 16, 48, 32)
    assert layout.reference is None
    assert layout.pixel_pitch_um == pytest.approx(1.12)
    assert layout.focal_length_mm is None


def test_default_slanted_edge_layout_uses_a_centred_half_size_roi() -> None:
    layout = slanted_edge_layout()

    assert layout.edge_roi == PatchROI("edge", 64, 64, 128, 128)


@pytest.mark.parametrize("target", ["flat_patch", "slanted_edge"])
def test_single_region_layouts_match_render(target: str) -> None:
    shape = (48, 64)
    layout = flat_patch_layout(shape) if target == "flat_patch" else slanted_edge_layout(shape)
    scene_layout = render(target, shape=shape).layout

    assert layout.name == scene_layout.name
    assert len(layout.patches) == len(scene_layout.patches)
    for metrics_patch, scene_patch in zip(layout.patches, scene_layout.patches, strict=True):
        assert (metrics_patch.name, metrics_patch.x, metrics_patch.y) == (
            scene_patch.name,
            scene_patch.x,
            scene_patch.y,
        )
        assert (metrics_patch.width, metrics_patch.height) == (
            scene_patch.width,
            scene_patch.height,
        )
    if scene_layout.edge_roi is None:
        assert layout.edge_roi is None
    else:
        assert layout.edge_roi is not None
        assert (layout.edge_roi.name, layout.edge_roi.x, layout.edge_roi.y) == (
            scene_layout.edge_roi.name,
            scene_layout.edge_roi.x,
            scene_layout.edge_roi.y,
        )
        assert (layout.edge_roi.width, layout.edge_roi.height) == (
            scene_layout.edge_roi.width,
            scene_layout.edge_roi.height,
        )


@pytest.mark.parametrize("illuminant", ["3000K", "4000K"])
def test_reference_required_for_non_d65(illuminant: str) -> None:
    layout = colorchecker_layout(illuminant=illuminant, shape=(64, 64), oversampling=1)

    assert layout.reference is not None
    assert layout.reference.illuminant_name == illuminant
    np.testing.assert_allclose(layout.reference.lab, reference_lab(illuminant).lab)

    daylight = colorchecker_layout(illuminant="D65", shape=(64, 64), oversampling=1)
    assert not np.allclose(layout.reference.lab, daylight.reference.lab)

    # Unknown illuminants must fail through reference_lab, never fall back to D65.
    with pytest.raises(ValueError, match="unsupported illuminant"):
        colorchecker_layout(illuminant="D50")


def test_colorchecker_layout_records_the_render_reference() -> None:
    result = render("colorchecker", shape=(64, 96), oversampling=2)

    layout = colorchecker_layout(shape=(64, 96), oversampling=2)

    assert layout.reference == result.reference


def test_colorchecker_patch_names_follow_chart_order() -> None:
    layout = colorchecker_layout(shape=(64, 96), oversampling=2)

    names = tuple(patch.name for patch in layout.patches)
    assert names == load_colorchecker().patches
    assert len(set(names)) == 24


@pytest.mark.parametrize(
    ("shape", "oversampling"),
    [((64, 96), 2), ((80, 80), 3)],
)
def test_colorchecker_layout_matches_render(
    shape: tuple[int, int],
    oversampling: int,
) -> None:
    layout = colorchecker_layout(shape=shape, oversampling=oversampling)
    scene_layout = render("colorchecker", shape=shape, oversampling=oversampling).layout

    assert layout.name == scene_layout.name
    assert tuple(patch.name for patch in layout.patches) == tuple(
        patch.name for patch in scene_layout.patches
    )
    for metrics_patch, scene_patch in zip(layout.patches, scene_layout.patches, strict=True):
        assert (
            metrics_patch.x,
            metrics_patch.y,
            metrics_patch.width,
            metrics_patch.height,
        ) == (scene_patch.x, scene_patch.y, scene_patch.width, scene_patch.height)


def test_colorchecker_layout_records_lens_focal_length() -> None:
    lens = LensConfig(
        f_number=1.8,
        focal_length_mm=6.4,
        sensor_diagonal_mm=11.46,
        transmittance=0.9,
    )

    layout = colorchecker_layout(lens=lens, shape=(64, 64), oversampling=1)
    without_lens = colorchecker_layout(shape=(64, 64), oversampling=1)

    assert layout.focal_length_mm == pytest.approx(6.4)
    assert layout.pixel_pitch_um == pytest.approx(1.12)
    assert without_lens.focal_length_mm is None


def test_colorchecker_layout_rejects_a_non_lens_object() -> None:
    with pytest.raises(ValueError, match="LensConfig"):
        colorchecker_layout(lens="f/1.8")  # type: ignore[arg-type]


def test_layout_builders_record_a_custom_camera_metadata() -> None:
    layouts = (
        flat_patch_layout((16, 16), pixel_pitch_um=2.4, focal_length_mm=4.2),
        slanted_edge_layout((16, 16), pixel_pitch_um=2.4, focal_length_mm=4.2),
    )

    for layout in layouts:
        assert layout.pixel_pitch_um == pytest.approx(2.4)
        assert layout.focal_length_mm == pytest.approx(4.2)


@pytest.mark.parametrize("shape", [(0, 10), (10, 0), (10,), (10, 10, 10), "10x10"])
def test_layout_builders_reject_invalid_shapes(shape: object) -> None:
    with pytest.raises(ValueError, match="shape"):
        flat_patch_layout(shape)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        slanted_edge_layout(shape)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        colorchecker_layout(shape=shape)  # type: ignore[arg-type]


@pytest.mark.parametrize("oversampling", [0, -1, 2.5, True])
def test_colorchecker_layout_rejects_invalid_oversampling(oversampling: object) -> None:
    with pytest.raises(ValueError, match="oversampling"):
        colorchecker_layout(
            shape=(64, 64),
            oversampling=oversampling,  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# PatchROI validation and serialization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "arguments",
    [
        ("", 0, 0, 2, 2),
        ("   ", 0, 0, 2, 2),
        ("p", -1, 0, 2, 2),
        ("p", 0, -1, 2, 2),
        ("p", 0, 0, 0, 2),
        ("p", 0, 0, 2, 0),
        ("p", 0, 0, -2, 2),
        ("p", 0.0, 0, 2, 2),
        ("p", 0, 0, 2.5, 2),
        ("p", True, 0, 2, 2),
    ],
)
def test_patch_roi_rejects_invalid_geometry(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        PatchROI(*arguments)  # type: ignore[arg-type]


def test_patch_roi_rejects_non_string_names() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        PatchROI(7, 0, 0, 2, 2)  # type: ignore[arg-type]


def test_patch_roi_serializes_to_a_flat_mapping() -> None:
    roi = PatchROI("dark skin", 3, 4, 5, 6)

    assert roi.to_dict() == {
        "name": "dark skin",
        "x": 3,
        "y": 4,
        "width": 5,
        "height": 6,
    }
    assert PatchROI.from_dict(roi.to_dict()) == roi
    assert PatchROI.from_dict(json.loads(json.dumps(roi.to_dict()))) == roi


def test_patch_roi_from_dict_requires_every_field() -> None:
    with pytest.raises(ValueError, match="missing"):
        PatchROI.from_dict({"name": "p", "x": 0, "y": 0, "width": 2})


@pytest.mark.parametrize("payload", [None, [], "p"])
def test_patch_roi_from_dict_rejects_non_mappings(payload: object) -> None:
    with pytest.raises(ValueError, match="mapping"):
        PatchROI.from_dict(payload)  # type: ignore[arg-type]


def test_patch_rois_are_frozen_value_objects() -> None:
    roi = PatchROI("p", 0, 0, 2, 2)

    with pytest.raises(FrozenInstanceError):
        roi.x = 1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# ChartLayout validation and serialization
# --------------------------------------------------------------------------- #


def test_layout_rejects_overlapping_patches() -> None:
    with pytest.raises(ValueError, match="overlap"):
        ChartLayout("chart", (PatchROI("a", 0, 0, 4, 4), PatchROI("b", 2, 2, 4, 4)))

    with pytest.raises(ValueError, match="overlap"):
        ChartLayout(
            "chart",
            (PatchROI("a", 0, 0, 4, 4),),
            edge_roi=PatchROI("edge", 3, 3, 4, 4),
        )

    # Rectangles that merely touch share no positive-area pixel.
    layout = ChartLayout("chart", (PatchROI("a", 0, 0, 4, 4), PatchROI("b", 4, 0, 4, 4)))
    assert len(layout.patches) == 2


def test_layout_rejects_duplicate_patch_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        ChartLayout("chart", (PatchROI("p", 0, 0, 2, 2), PatchROI("p", 4, 0, 2, 2)))


def test_layout_requires_patches_or_an_edge_roi() -> None:
    with pytest.raises(ValueError, match="edge"):
        ChartLayout("empty")

    edge_only = ChartLayout("edge", edge_roi=PatchROI("edge", 0, 0, 4, 4))
    assert edge_only.patches == ()


def test_layout_rejects_non_roi_patches() -> None:
    with pytest.raises(ValueError, match="PatchROI"):
        ChartLayout("chart", ("p",))  # type: ignore[arg-type]


def test_layout_rejects_non_roi_edge_region() -> None:
    with pytest.raises(ValueError, match="edge_roi"):
        ChartLayout(
            "chart",
            (PatchROI("p", 0, 0, 2, 2),),
            edge_roi=(0, 0, 2, 2),  # type: ignore[arg-type]
        )


def test_layout_rejects_invalid_names_and_references() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        ChartLayout("", (PatchROI("p", 0, 0, 2, 2),))

    with pytest.raises(ValueError, match="ColorReference"):
        ChartLayout(
            "chart",
            (PatchROI("p", 0, 0, 2, 2),),
            reference="D65",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("pixel_pitch_um", 0.0),
        ("pixel_pitch_um", -1.12),
        ("pixel_pitch_um", math.inf),
        ("pixel_pitch_um", math.nan),
        ("pixel_pitch_um", "1.12"),
        ("pixel_pitch_um", True),
        ("focal_length_mm", 0.0),
        ("focal_length_mm", -6.4),
        ("focal_length_mm", math.inf),
        ("focal_length_mm", math.nan),
        ("focal_length_mm", "6.4"),
    ],
)
def test_layout_rejects_invalid_camera_metadata(field_name: str, value: object) -> None:
    with pytest.raises(ValueError, match=field_name):
        ChartLayout(
            "chart",
            (PatchROI("p", 0, 0, 2, 2),),
            **{field_name: value},
        )


def test_layout_normalizes_patches_to_a_tuple() -> None:
    layout = ChartLayout("chart", [PatchROI("a", 0, 0, 2, 2)])  # type: ignore[arg-type]

    assert isinstance(layout.patches, tuple)
    assert layout.patches == (PatchROI("a", 0, 0, 2, 2),)


def test_layouts_are_frozen_value_objects() -> None:
    layout = ChartLayout("chart", (PatchROI("p", 0, 0, 2, 2),))

    with pytest.raises(FrozenInstanceError):
        layout.name = "other"  # type: ignore[misc]


def test_layout_to_dict_schema() -> None:
    roi = PatchROI("p", 0, 0, 2, 2)
    layout = ChartLayout(
        "chart",
        (roi,),
        edge_roi=PatchROI("edge", 4, 4, 2, 2),
        pixel_pitch_um=1.12,
        focal_length_mm=6.4,
    )

    assert layout.to_dict() == {
        "name": "chart",
        "patches": [roi.to_dict()],
        "edge_roi": {"name": "edge", "x": 4, "y": 4, "width": 2, "height": 2},
        "reference": None,
        "pixel_pitch_um": 1.12,
        "focal_length_mm": 6.4,
    }


def test_layout_roundtrip_preserves_every_field() -> None:
    lens = LensConfig(
        f_number=1.8,
        focal_length_mm=6.4,
        sensor_diagonal_mm=11.46,
        transmittance=0.9,
    )
    layout = colorchecker_layout(
        illuminant="4000K",
        lens=lens,
        shape=(64, 96),
        oversampling=2,
        pixel_pitch_um=1.4,
    )

    restored = ChartLayout.from_dict(layout.to_dict())

    assert restored == layout
    assert restored.reference is not None
    assert restored.reference.observer == "CIE 1931 2 degree"
    assert restored.pixel_pitch_um == pytest.approx(1.4)
    assert restored.focal_length_mm == pytest.approx(6.4)


@pytest.mark.parametrize(
    "layout",
    [
        flat_patch_layout((16, 24), pixel_pitch_um=1.4, focal_length_mm=6.4),
        slanted_edge_layout((16, 24), pixel_pitch_um=1.4, focal_length_mm=6.4),
    ],
)
def test_layout_roundtrip_covers_edge_and_patch_only_layouts(layout: ChartLayout) -> None:
    assert ChartLayout.from_dict(layout.to_dict()) == layout


def test_layout_roundtrip_preserves_reference_provenance() -> None:
    layout = colorchecker_layout(illuminant="3000K", shape=(64, 64), oversampling=1)
    assert layout.reference is not None

    restored = ChartLayout.from_dict(layout.to_dict())

    assert restored.reference is not None
    assert restored.reference == layout.reference
    assert restored.reference.provenance == layout.reference.provenance
    assert restored.reference.provenance["illuminant"]["temperature_k"] == pytest.approx(3000.0)


def test_layout_payload_is_json_serializable() -> None:
    layout = colorchecker_layout(illuminant="3000K", shape=(64, 64), oversampling=1)

    parsed = json.loads(json.dumps(layout.to_dict()))
    restored = ChartLayout.from_dict(parsed)

    assert restored.name == layout.name
    assert restored.reference is not None
    assert restored.reference.illuminant_name == "3000K"
    np.testing.assert_allclose(restored.reference.lab, layout.reference.lab)
    assert tuple(patch.name for patch in restored.patches) == tuple(
        patch.name for patch in layout.patches
    )


@pytest.mark.parametrize("payload", [None, [], "chart"])
def test_layout_from_dict_rejects_non_mappings(payload: object) -> None:
    with pytest.raises(ValueError, match="mapping"):
        ChartLayout.from_dict(payload)  # type: ignore[arg-type]


def test_layout_from_dict_requires_a_name() -> None:
    with pytest.raises(ValueError, match="name"):
        ChartLayout.from_dict({"patches": []})


def test_layout_from_dict_rejects_a_bad_reference_payload() -> None:
    with pytest.raises(ValueError, match="mapping"):
        ChartLayout.from_dict(
            {
                "name": "chart",
                "patches": [{"name": "p", "x": 0, "y": 0, "width": 2, "height": 2}],
                "reference": "D65",
            }
        )


# --------------------------------------------------------------------------- #
# extract_patch
# --------------------------------------------------------------------------- #


def test_extract_patch_2d_returns_a_view_of_the_requested_region() -> None:
    image = np.arange(30, dtype=np.int16).reshape(5, 6)

    patch = extract_patch(image, PatchROI("p", 1, 2, 3, 2))

    assert patch.shape == (2, 3)
    assert patch.dtype == np.int16
    np.testing.assert_array_equal(patch, image[2:4, 1:4])
    assert np.shares_memory(patch, image)


def test_extract_patch_preserves_trailing_axes() -> None:
    image = np.arange(4 * 5 * 3, dtype=np.float64).reshape(4, 5, 3)

    patch = extract_patch(image, PatchROI("p", 1, 1, 2, 2))

    assert patch.shape == (2, 2, 3)
    assert patch.dtype == np.float64
    np.testing.assert_array_equal(patch, image[1:3, 1:3, :])


def test_extract_patch_returns_a_view_without_mutating_the_source() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    original = image.copy()

    patch = extract_patch(image, PatchROI("p", 1, 1, 2, 2))

    np.testing.assert_array_equal(image, original)
    assert np.shares_memory(patch, image)


@pytest.mark.parametrize(
    "roi",
    [
        PatchROI("p", 8, 0, 3, 3),
        PatchROI("p", 0, 8, 3, 3),
        PatchROI("p", 0, 0, 11, 2),
        PatchROI("p", 0, 0, 2, 11),
    ],
)
def test_extract_patch_rejects_out_of_bounds_rois(roi: PatchROI) -> None:
    with pytest.raises(ValueError, match="outside"):
        extract_patch(np.zeros((10, 10)), roi)


@pytest.mark.parametrize("roi", [(0, 0, 2, 2), None, "p"])
def test_extract_patch_rejects_non_patch_rois(roi: object) -> None:
    with pytest.raises(TypeError, match="PatchROI"):
        extract_patch(np.zeros((4, 4)), roi)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((4, 4), dtype=bool), "bool"),
        (np.zeros((4, 4), dtype=np.complex64), "complex"),
        (np.full((4, 4), np.nan), "finite"),
        (np.full((4, 4), np.inf), "finite"),
        (np.zeros(4), "2-D"),
        (np.zeros((2, 2, 2, 2)), "2-D"),
        (np.array([["a", "b"], ["c", "d"]]), "real numbers"),
    ],
)
def test_extract_patch_rejects_invalid_images(image: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        extract_patch(image, PatchROI("p", 0, 0, 1, 1))  # type: ignore[arg-type]


def test_extract_patch_accepts_nested_lists() -> None:
    patch = extract_patch([[1.0, 2.0], [3.0, 4.0]], PatchROI("p", 1, 0, 1, 2))

    assert patch.shape == (2, 1)
    np.testing.assert_array_equal(patch, np.array([[2.0], [4.0]]))


# --------------------------------------------------------------------------- #
# Package surface and import direction
# --------------------------------------------------------------------------- #


def test_metrics_package_exports_the_chart_types() -> None:
    import camsim.metrics as metrics

    assert metrics.ChartLayout is ChartLayout
    assert metrics.PatchROI is PatchROI
    assert metrics.extract_patch is extract_patch
    assert set(metrics.__all__) == {"ChartLayout", "PatchROI", "extract_patch"}


def test_metrics_chart_does_not_import_simulator_modules() -> None:
    code = "\n".join(
        [
            "import sys",
            "import camsim.metrics.chart",
            "from camsim.metrics.chart import ChartLayout, PatchROI, extract_patch",
            "for name in (",
            "    'camsim.sensor', 'camsim.isp', 'camsim.scene', 'camsim.optics',",
            "):",
            "    assert name not in sys.modules, name",
        ]
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
