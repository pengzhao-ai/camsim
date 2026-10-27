"""Tests for edge acutance and the unified ``measure_all`` metrics harness.

Acutance is a scalar sharpness proxy built from the gradient energy of a
contrast-normalized edge profile. An edge that rises from black to white in one
pixel has a gradient concentrated in a single transition; a blurred edge spreads
the same rise over many pixels, so the root-mean-square gradient falls. Dividing
out the profile's own ``max - min`` makes the number insensitive to exposure and
contrast, which is what allows it to be correlated with rendering parameters
without confounding sharpness with brightness. It is not a calibrated physical
quantity: a display-encoded image has already had a tone curve applied, so the
same optics measured through a different OETF returns a different acutance, and
a noisy patch inflates the gradient energy. A flat ROI contains no edge at all
and reports exactly zero rather than an arbitrary number.

``measure_all`` is the measurement plan's executor. A ``ChartLayout`` says which
groups can be measured: an edge ROI enables slanted-edge MTF and acutance, patch
ROIs enable SNR, and patches plus a colour reference enable delta-E. Groups that
the layout or image cannot support are skipped and recorded in the report's
metadata instead of silently producing fabricated numbers. Colour is the one
group that can refuse to be skipped: a chart whose name says "colorchecker" and
that carries no reference raises, because the metrics layer cannot infer an
illuminant and a silent D65 fallback would report a white-point error as a
sensor colour error.

The tests exercise the Monte-Carlo-free physics: a hand-computable step edge for
the acutance formula, a rendered slanted edge for the end-to-end path, a
two-noises image to prove SNR measures the linear buffer rather than display
values, and a JSON round trip to prove non-finite crossings serialize as
``null`` rather than the non-standard ``NaN`` token.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest
from scipy.ndimage import uniform_filter

from camsim.core.color import ColorReference
from camsim.core.frames import RenderedImage
from camsim.metrics.acutance import edge_acutance
from camsim.metrics.chart import ChartLayout, PatchROI
from camsim.metrics.color import _srgb_eotf, delta_e_patches
from camsim.metrics.mtf import MTFCurve, MTFResult
from camsim.metrics.snr import SNRResult, snr_channels
from camsim.metrics.summary import MetricReport, measure_all
from camsim.scene.targets import slanted_edge_target

_D65 = (0.3127, 0.3290)
_WARM = (0.4369, 0.4041)

# --------------------------------------------------------------------------- #
# Brief contract tests (verbatim from the task brief)
# --------------------------------------------------------------------------- #


def test_measure_all_complete_report():
    ref = ColorReference(lab=np.zeros((1, 3)), white_point_xy=(0.3127, 0.3290), illuminant_name="D65")  # fmt: skip  # noqa: E501
    layout = ChartLayout(name="t", patches=(PatchROI("p", 0, 0, 16, 16),), reference=ref)
    img = RenderedImage(data=np.full((32, 32, 3), 0.5, np.float32), color_space="sRGB", channels="RGB")  # fmt: skip  # noqa: E501
    report = measure_all(img, layout)
    assert report.color is not None
    assert report.to_dict()["layout_id"] == "t"


def test_measure_all_skips_missing_groups():
    layout = ChartLayout(name="flat", patches=(PatchROI("p", 0, 0, 8, 8),))
    img = RenderedImage(data=np.full((16, 16, 3), 0.5, np.float32), color_space="sRGB", channels="RGB")  # fmt: skip  # noqa: E501
    report = measure_all(img, layout)
    assert report.mtf is None and report.color is None


def test_measure_all_mono_skips_color():
    layout = ChartLayout(name="flat", patches=(PatchROI("p", 0, 0, 8, 8),))
    img = RenderedImage(data=np.full((16, 16, 1), 0.5, np.float32), color_space="sRGB", channels="MONO")  # fmt: skip  # noqa: E501
    report = measure_all(img, layout)
    assert report.color is None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _reference(
    lab: Any,
    white_point: tuple[float, float] = _D65,
    name: str = "D65",
) -> ColorReference:
    """Return a colour reference carrying ``lab`` under ``white_point``."""
    return ColorReference(
        lab=np.asarray(lab, dtype=np.float64),
        white_point_xy=white_point,
        illuminant_name=name,
    )


def _srgb_oetf(linear: Any) -> np.ndarray:
    """Encode linear values with the sRGB OETF (the exact inverse of the EOTF)."""
    values = np.asarray(linear, dtype=np.float64)
    encoded = np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(np.maximum(values, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.asarray(encoded, dtype=np.float64)


def _rendered_rgb(linear_rgb: Any, *, data: Any = None) -> RenderedImage:
    """Build an RGB ``RenderedImage`` from a linear buffer, encoding if needed."""
    linear = np.asarray(linear_rgb, dtype=np.float32)
    display = _srgb_oetf(linear) if data is None else np.asarray(data, dtype=np.float32)
    return RenderedImage(
        data=np.asarray(display, dtype=np.float32),
        color_space="sRGB",
        channels="RGB",
        linear_rgb=linear,
    )


def _rendered_mono(linear: Any, *, data: Any = None) -> RenderedImage:
    """Build a mono ``RenderedImage`` from a 2-D linear plane."""
    plane = np.asarray(linear, dtype=np.float32)
    display = _srgb_oetf(plane) if data is None else np.asarray(data, dtype=np.float32)
    return RenderedImage(
        data=np.asarray(display, dtype=np.float32)[..., None],
        color_space="sRGB",
        channels="MONO",
        linear_rgb=plane[..., None],
    )


def _edge_plane(shape: tuple[int, int] = (96, 96), blur: int = 0) -> np.ndarray:
    """Return a synthetic slanted-edge reflectance plane, optionally box-blurred."""
    edge = slanted_edge_target(shape, angle_deg=5.0)
    if blur:
        edge = np.asarray(uniform_filter(edge, size=blur, mode="nearest"), dtype=np.float64)
    return edge


def _edge_rgb_image(blur: int = 0) -> RenderedImage:
    """Return a three-channel rendered image containing one slanted edge."""
    edge = _edge_plane((96, 96), blur)
    return _rendered_rgb(np.repeat(edge[..., None], 3, axis=2))


def _edge_layout(*, pitch: float | None = 1.12, focal: float | None = 6.4) -> ChartLayout:
    """Return an edge-only layout covering the 96 x 96 slanted-edge frame."""
    return ChartLayout(
        name="slanted_edge",
        edge_roi=PatchROI("edge", 0, 0, 96, 96),
        pixel_pitch_um=pitch,
        focal_length_mm=focal,
    )


def _complete_layout() -> ChartLayout:
    """Return a layout with two constant patches and a separate edge region."""
    return ChartLayout(
        name="mixed_chart",
        patches=(PatchROI("p1", 0, 0, 8, 8), PatchROI("p2", 8, 0, 8, 8)),
        edge_roi=PatchROI("edge", 32, 0, 64, 64),
        reference=_reference([[50.0, 0.0, 0.0], [60.0, 2.0, -2.0]]),
        pixel_pitch_um=1.12,
        focal_length_mm=6.4,
    )


def _complete_image() -> RenderedImage:
    """Return a rendered frame with flat patch regions and a slanted edge."""
    linear = np.full((96, 96, 3), 0.5, dtype=np.float64)
    linear[0:64, 32:96] = np.repeat(_edge_plane((64, 64))[..., None], 3, axis=2)
    return _rendered_rgb(linear, data=_srgb_oetf(linear))


# --------------------------------------------------------------------------- #
# Acutance
# --------------------------------------------------------------------------- #


def test_acutance_known_edge() -> None:
    """A one-pixel vertical step has a hand-computable gradient energy.

    ``np.gradient`` places the transition in two adjacent columns, each with a
    central difference of 0.5. Every row contributes ``2 * 0.25 = 0.5`` to the
    sum of squared gradients, so the mean over the ROI is ``1 / (2 W)``.
    """
    width = 32
    image = np.zeros((16, width), dtype=np.float64)
    image[:, width // 2 :] = 1.0

    acutance = edge_acutance(image, PatchROI("step", 0, 0, width, 16))

    assert acutance == pytest.approx(1.0 / math.sqrt(2.0 * width), rel=1e-12)
    # Rec. 709 luma weights sum to one, so identical channels give the same value.
    rgb = np.repeat(image[..., None], 3, axis=2)
    assert edge_acutance(rgb, PatchROI("step", 0, 0, width, 16)) == pytest.approx(acutance)


def test_acutance_is_contrast_and_exposure_invariant() -> None:
    """Normalizing by the ROI's own range removes exposure and contrast scaling."""
    width = 32
    image = np.zeros((16, width), dtype=np.float64)
    image[:, width // 2 :] = 1.0
    roi = PatchROI("step", 0, 0, width, 16)

    reference = edge_acutance(image, roi)
    scaled = edge_acutance(image * 100.0 - 3.0, roi)
    compressed = edge_acutance(0.1 + image * 0.2, roi)

    assert scaled == pytest.approx(reference, rel=1e-12)
    assert compressed == pytest.approx(reference, rel=1e-12)


def test_acutance_sharp_edge_exceeds_blurred_edge() -> None:
    """Optical blur lowers the normalized gradient energy of the same edge."""
    roi = PatchROI("edge", 0, 0, 96, 96)
    sharp = edge_acutance(_edge_plane((96, 96)), roi)
    blurred = edge_acutance(np.asarray(uniform_filter(_edge_plane((96, 96)), size=5)), roi)

    assert sharp > blurred > 0.0


def test_flat_roi_has_zero_acutance() -> None:
    """A constant ROI contains no edge; the documented answer is exactly zero."""
    roi = PatchROI("flat", 0, 0, 8, 8)
    assert edge_acutance(np.full((8, 8), 0.3), roi) == 0.0
    assert edge_acutance(np.zeros((8, 8, 3)), roi) == 0.0


def test_acutance_handles_a_single_row_roi() -> None:
    """A one-row ROI has no vertical variation; the horizontal ramp still counts."""
    image = np.tile(np.linspace(0.0, 1.0, 8), (4, 1))
    roi = PatchROI("row", 0, 1, 8, 1)
    # Gradients along the ramp: 1/7 in the six interior columns, one-sided
    # 1/7 at each end, so every column contributes exactly (1/7)^2.
    linear_ramp = edge_acutance(image, roi)
    assert linear_ramp == pytest.approx(1.0 / 7.0, rel=1e-12)
    flat_row = edge_acutance(np.full((4, 8), 0.5), roi)
    assert flat_row == 0.0


def test_acutance_rejects_invalid_input() -> None:
    """Malformed images, channel counts, and ROIs raise clear errors."""
    roi = PatchROI("p", 0, 0, 4, 4)
    with pytest.raises(ValueError):
        edge_acutance(np.full((8, 8), np.nan), roi)
    with pytest.raises(ValueError):
        edge_acutance(np.full((8, 8), np.inf), roi)
    with pytest.raises(ValueError):
        edge_acutance(np.zeros((8, 8), dtype=bool), roi)
    with pytest.raises(ValueError):
        edge_acutance(np.zeros((8, 8), dtype=np.complex128), roi)
    with pytest.raises(ValueError):
        edge_acutance(np.zeros((8, 8, 1)), roi)
    with pytest.raises(ValueError):
        edge_acutance(np.zeros((8, 8, 4)), roi)
    with pytest.raises(ValueError):
        edge_acutance(np.zeros((2, 8, 8, 1)), roi)
    with pytest.raises(ValueError, match="lies outside image shape"):
        edge_acutance(np.zeros((4, 4)), PatchROI("p", 2, 2, 4, 4))
    with pytest.raises(TypeError):
        edge_acutance(np.zeros((8, 8)), object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Measurement harness orchestration
# --------------------------------------------------------------------------- #


def test_measure_all_requires_a_rendered_image_and_chart_layout() -> None:
    """Wrong argument types are rejected with a type error, never coerced."""
    layout = ChartLayout(name="flat", patches=(PatchROI("p", 0, 0, 4, 4),))
    image = _rendered_rgb(np.full((8, 8, 3), 0.5))
    with pytest.raises(TypeError):
        measure_all(np.zeros((8, 8, 3), dtype=np.float32), layout)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        measure_all(image, "flat")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        measure_all(image, layout, reference="D65")  # type: ignore[arg-type]


def test_measure_all_rgb_and_mono() -> None:
    """RGB patches report R/G/B/luma; mono patches report a single plane."""
    reference = _reference([[50.0, 0.0, 0.0], [60.0, 2.0, -2.0]])
    layout = ChartLayout(
        name="flat",
        patches=(PatchROI("p1", 0, 0, 8, 8), PatchROI("p2", 8, 0, 8, 8)),
        reference=reference,
    )
    rgb = np.full((8, 16, 3), 0.5, dtype=np.float32)
    rgb[:, 8:] = 0.75
    rgb_report = measure_all(_rendered_rgb(_srgb_eotf(rgb), data=rgb), layout)

    assert tuple(rgb_report.snr) == ("p1", "p2")
    assert tuple(rgb_report.snr["p1"]) == ("R", "G", "B", "luma")
    assert rgb_report.snr["p1"]["R"].valid is False  # constant patch: no dispersion
    assert rgb_report.color is not None and rgb_report.color.valid
    assert rgb_report.mtf is None and rgb_report.acutance is None
    assert rgb_report.metadata["channels"] == "RGB"

    mono = np.full((8, 16), 0.5, dtype=np.float32)
    mono[:, 8:] = 0.75
    mono_report = measure_all(_rendered_mono(mono), layout)

    assert tuple(mono_report.snr) == ("p1", "p2")
    assert tuple(mono_report.snr["p1"]) == ("mono",)
    assert mono_report.color is None
    assert mono_report.metadata["channels"] == "MONO"


def test_measure_all_edge_layout_produces_mtf_and_acutance() -> None:
    """An edge ROI enables both the slanted-edge MTF and the acutance proxy."""
    image = _edge_rgb_image(blur=5)
    report = measure_all(image, _edge_layout())

    assert isinstance(report.mtf, MTFResult)
    assert math.isfinite(report.mtf.mtf50_cp)
    assert report.mtf.curve.mtf[0] == pytest.approx(1.0, abs=1e-6)
    assert report.acutance is not None and report.acutance > 0.0
    assert report.snr is None and report.color is None
    assert any(item.startswith("snr:") for item in report.metadata["skipped"])
    assert any(item.startswith("color:") for item in report.metadata["skipped"])


def test_measure_all_mono_edge_uses_the_linear_plane() -> None:
    """A mono image feeds its single plane to MTF and acutance unchanged."""
    edge = _edge_plane((96, 96), blur=5)
    rgb_report = measure_all(_edge_rgb_image(blur=5), _edge_layout())
    mono_report = measure_all(_rendered_mono(edge), _edge_layout())

    assert isinstance(mono_report.mtf, MTFResult)
    assert mono_report.mtf.mtf50_cp == pytest.approx(rgb_report.mtf.mtf50_cp, abs=1e-6)
    assert mono_report.acutance == pytest.approx(rgb_report.acutance, rel=1e-6)


def test_measure_all_mtf_requires_a_pixel_pitch() -> None:
    """An edge ROI without a pixel pitch cannot produce cycles/pixel: it raises."""
    layout = _edge_layout(pitch=None)
    with pytest.raises(ValueError, match="pixel_pitch_um"):
        measure_all(_edge_rgb_image(blur=5), layout)


def test_measure_all_without_focal_length_reports_nan_angular_crossings() -> None:
    """Without a focal length the cycles/pixel crossings stand; angular ones are NaN."""
    report = measure_all(_edge_rgb_image(blur=5), _edge_layout(focal=None))
    assert math.isfinite(report.mtf.mtf50_cp)
    assert math.isnan(report.mtf.mtf50_cyc_per_deg)
    assert math.isnan(report.mtf.mtf20_cyc_per_deg)


def test_measure_all_skips_colour_for_non_chart_layouts_without_reference() -> None:
    """A non-colorchecker layout without a reference skips colour, never guesses D65."""
    layout = ChartLayout(name="flat", patches=(PatchROI("p", 0, 0, 8, 8),))
    report = measure_all(_rendered_rgb(np.full((8, 8, 3), 0.5)), layout)

    assert report.color is None
    assert report.metadata["reference_illuminant"] is None
    assert any(item.startswith("color:") for item in report.metadata["skipped"])


def test_missing_non_d65_reference_raises() -> None:
    """A colour-checker layout without a reference raises rather than assuming D65."""
    layout = ChartLayout(name="ColorChecker_3000K", patches=(PatchROI("p", 0, 0, 8, 8),))
    image = _rendered_rgb(np.full((8, 8, 3), 0.5))
    with pytest.raises(ValueError, match="colour chart requires a reference"):
        measure_all(image, layout)
    # An explicit reference resolves the same layout.
    report = measure_all(image, layout, reference=_reference([[50.0, 0.0, 0.0]]))
    assert report.color is not None


def test_explicit_reference_overrides_the_layout_reference() -> None:
    """The explicit reference wins, and the report records its illuminant."""
    data = np.full((8, 8, 3), 0.5, dtype=np.float32)
    layout_reference = _reference([[100.0, 0.0, 0.0]])
    explicit_reference = _reference([[80.0, 5.0, 5.0]], white_point=_WARM, name="3000K")
    layout = ChartLayout(
        name="flat",
        patches=(PatchROI("p", 0, 0, 8, 8),),
        reference=layout_reference,
    )
    explicit = measure_all(
        _rendered_rgb(_srgb_eotf(data), data=data), layout, reference=explicit_reference
    )
    default = measure_all(_rendered_rgb(_srgb_eotf(data), data=data), layout)

    expected = delta_e_patches(data, layout, reference=explicit_reference)
    assert explicit.color is not None and default.color is not None
    assert explicit.color.delta_e76 == pytest.approx(expected.delta_e76, rel=1e-12)
    assert explicit.color.delta_e76 != pytest.approx(default.color.delta_e76)
    assert explicit.metadata["reference_illuminant"] == "3000K"
    assert default.metadata["reference_illuminant"] == "D65"


def test_measure_all_snr_uses_the_linear_buffer() -> None:
    """SNR measures linear light, never the display-encoded buffer."""
    rng = np.random.default_rng(7)
    display = np.clip(0.5 + rng.normal(0.0, 0.02, (16, 16, 3)), 0.0, 1.0).astype(np.float32)
    linear = np.clip(0.2 + rng.normal(0.0, 0.01, (16, 16, 3)), 0.0, 1.0).astype(np.float32)
    layout = ChartLayout(name="flat", patches=(PatchROI("p", 0, 0, 16, 16),))
    report = measure_all(_rendered_rgb(linear, data=display), layout)

    expected = snr_channels(np.asarray(linear, dtype=np.float64), layout.patches[0])
    display_only = snr_channels(np.asarray(display, dtype=np.float64), layout.patches[0])
    for channel in ("R", "G", "B", "luma"):
        assert report.snr["p"][channel].mean == pytest.approx(expected[channel].mean, rel=1e-12)
        assert report.snr["p"][channel].mean != pytest.approx(display_only[channel].mean, rel=1e-6)

    # With no linear buffer the display data is decoded with the exact sRGB EOTF.
    decoded_image = RenderedImage(data=display, color_space="sRGB", channels="RGB")
    decoded_report = measure_all(decoded_image, layout)
    decoded_expected = snr_channels(_srgb_eotf(display), layout.patches[0])
    assert decoded_report.snr["p"]["G"].mean == pytest.approx(decoded_expected["G"].mean, rel=1e-12)


def test_measure_all_metadata_content() -> None:
    """Metadata records the image, layout, input, and every skipped group."""
    report = measure_all(_edge_rgb_image(blur=5), _edge_layout())

    assert report.metadata == {
        "color_space": "sRGB",
        "channels": "RGB",
        "pixel_pitch_um": 1.12,
        "focal_length_mm": 6.4,
        "reference_illuminant": None,
        "mtf_input": "linear_luma",
        "skipped": ["snr: layout has no patches", "color: layout has no patches"],
    }


def test_measure_all_records_monochrome_and_missing_groups() -> None:
    """The skipped list explains why each inapplicable group did not run."""
    layout = ChartLayout(
        name="flat",
        patches=(PatchROI("p", 0, 0, 8, 8),),
        reference=_reference([[50.0, 0.0, 0.0]]),
    )
    report = measure_all(_rendered_mono(np.full((8, 8), 0.5)), layout)

    skipped = report.metadata["skipped"]
    assert any(item.startswith("mtf:") for item in skipped)
    assert any(item.startswith("acutance:") for item in skipped)
    assert any("monochrome" in item for item in skipped)
    assert "snr" not in " ".join(item for item in skipped if "monochrome" not in item)


def test_measure_all_does_not_mutate_its_inputs() -> None:
    """The image buffers and the layout are untouched by a measurement."""
    image = _complete_image()
    layout = _complete_layout()
    data_before = image.data.copy()
    linear_before = image.linear_rgb.copy()
    layout_before = layout.to_dict()

    measure_all(image, layout)

    np.testing.assert_array_equal(image.data, data_before)
    np.testing.assert_array_equal(image.linear_rgb, linear_before)
    assert image.data.dtype == np.float32
    assert image.linear_rgb.dtype == np.float32
    assert layout.to_dict() == layout_before


def test_measure_all_is_deterministic() -> None:
    """Repeated measurement of the same image and layout is bit-identical."""
    image = _complete_image()
    layout = _complete_layout()

    first = measure_all(image, layout)
    second = measure_all(image, layout)

    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()


# --------------------------------------------------------------------------- #
# JSON serialization
# --------------------------------------------------------------------------- #


def test_measure_all_json_roundtrip() -> None:
    """A complete report serializes to standards-compliant JSON and back."""
    report = measure_all(_complete_image(), _complete_layout())

    assert report.mtf is not None
    assert report.snr is not None
    assert report.color is not None
    assert report.acutance is not None

    payload = report.to_dict()
    text = report.to_json()

    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == payload
    assert json.dumps(payload, allow_nan=False) == text


def test_json_serialization_maps_non_finite_values_to_null() -> None:
    """A sharp edge never crosses MTF50 inside the band: NaN becomes JSON null."""
    report = measure_all(_edge_rgb_image(blur=0), _edge_layout())
    assert report.mtf is not None
    assert math.isnan(report.mtf.mtf50_cp)
    assert math.isnan(report.mtf.mtf20_cp)

    payload = report.to_dict()
    assert payload["mtf"]["mtf50_cp"] is None
    assert payload["mtf"]["mtf20_cp"] is None
    assert payload["mtf"]["mtf50_cyc_per_deg"] is None
    assert payload["mtf"]["curve"]["frequency_cp"]
    assert len(payload["mtf"]["curve"]["frequency_cp"]) == len(payload["mtf"]["curve"]["mtf"])
    json.loads(report.to_json())

    # The same rule covers an invalid (flat-patch) SNR, whose snr_db is NaN.
    layout = ChartLayout(name="flat", patches=(PatchROI("p", 0, 0, 8, 8),))
    flat = measure_all(_rendered_rgb(np.full((8, 8, 3), 0.5)), layout)
    assert math.isnan(flat.snr["p"]["R"].snr_db)
    assert flat.to_dict()["snr"]["p"]["R"]["snr_db"] is None


def test_measure_all_to_dict_includes_the_full_mtf_curve() -> None:
    """The serialized report carries the sampled curve, not just its crossings."""
    payload = measure_all(_edge_rgb_image(blur=5), _edge_layout()).to_dict()

    assert set(payload) == {"layout_id", "mtf", "snr", "color", "acutance", "metadata"}
    assert set(payload["mtf"]) == {
        "mtf50_cp",
        "mtf20_cp",
        "mtf50_cyc_per_deg",
        "mtf20_cyc_per_deg",
        "curve",
        "edge_angle_deg",
        "nyquist_cp",
    }
    assert set(payload["mtf"]["curve"]) == {"frequency_cp", "mtf"}
    assert payload["mtf"]["curve"]["frequency_cp"][0] == 0.0
    assert payload["mtf"]["curve"]["mtf"][0] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# MetricReport value object
# --------------------------------------------------------------------------- #


def _mtf_result() -> MTFResult:
    """Return a small valid MTF result for value-object tests."""
    return MTFResult(
        mtf50_cp=0.2,
        mtf20_cp=0.4,
        mtf50_cyc_per_deg=math.nan,
        mtf20_cyc_per_deg=math.nan,
        curve=MTFCurve([0.0, 0.1, 0.2], [1.0, 0.8, 0.5]),
        edge_angle_deg=5.0,
    )


def test_metric_report_validates_its_fields() -> None:
    """Wrong types and invalid values are rejected at construction."""
    snr = {"p": {"R": SNRResult(mean=1.0, std=0.1, snr_db=20.0, channel="R", valid=True)}}
    report = MetricReport(
        mtf=_mtf_result(),
        snr=snr,
        acutance=0.1,
        layout_id=" t ",
        metadata={"source": "test"},
    )
    assert report.layout_id == "t"
    assert report.acutance == 0.1
    assert report.metadata == {"source": "test"}

    with pytest.raises(ValueError):
        MetricReport(mtf=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MetricReport(color=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MetricReport(snr={"p": {"R": object()}})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        MetricReport(snr=[("p", {})])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MetricReport(snr={"": {"R": snr["p"]["R"]}})
    with pytest.raises(ValueError):
        MetricReport(snr={"p": {"": snr["p"]["R"]}})
    with pytest.raises(ValueError):
        MetricReport(acutance=math.nan)
    with pytest.raises(ValueError):
        MetricReport(acutance=math.inf)
    with pytest.raises(ValueError):
        MetricReport(acutance=-0.5)
    with pytest.raises(ValueError):
        MetricReport(acutance=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MetricReport(layout_id="")
    with pytest.raises(ValueError):
        MetricReport(layout_id=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MetricReport(metadata=[1, 2])  # type: ignore[arg-type]


def test_metric_report_defaults_and_copies() -> None:
    """Metadata defaults to an empty dict and is copied from the caller."""
    empty = MetricReport()
    assert empty.metadata == {}
    assert empty.mtf is None and empty.snr is None and empty.color is None
    assert empty.acutance is None and empty.layout_id is None

    source = {"a": 1}
    report = MetricReport(metadata=source)
    source["b"] = 2
    assert report.metadata == {"a": 1}


def test_metric_report_preserves_patch_and_channel_order() -> None:
    """Nested SNR mappings keep the caller's (layout) order."""
    first = SNRResult(mean=1.0, std=0.1, snr_db=20.0, channel="R", valid=True)
    second = SNRResult(mean=2.0, std=0.1, snr_db=26.0, channel="G", valid=True)
    report = MetricReport(snr={"p2": {"G": second}, "p1": {"R": first}})

    assert list(report.snr) == ["p2", "p1"]
    assert list(report.snr["p2"]) == ["G"]


def test_metric_report_is_frozen() -> None:
    """A report cannot be edited after construction."""
    report = MetricReport(acutance=0.1)
    with pytest.raises(FrozenInstanceError):
        report.acutance = 0.2  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Package API and architecture
# --------------------------------------------------------------------------- #


def test_metrics_package_exports_the_documented_names() -> None:
    """The metrics package exposes the complete public API in ``__all__``."""
    import camsim.metrics as metrics

    expected = {
        "ChartLayout",
        "PatchROI",
        "extract_patch",
        "MTFCurve",
        "MTFResult",
        "slanted_edge_mtf",
        "extract_raw_plane",
        "SNRResult",
        "snr_patch",
        "snr_temporal",
        "snr_channels",
        "PatchColorResult",
        "ColorResult",
        "delta_e_patches",
        "linear_rgb_to_lab",
        "edge_acutance",
        "MetricReport",
        "measure_all",
    }
    assert set(metrics.__all__) == expected
    for name in expected:
        assert hasattr(metrics, name), name


def test_metrics_import_does_not_pull_in_simulator_modules() -> None:
    """Importing the metrics package keeps the simulator packages out of memory."""
    code = "\n".join(
        [
            "import sys",
            "import camsim.metrics",
            "from camsim.metrics import MetricReport, edge_acutance, measure_all",
            "for name in (",
            "    'camsim.sensor', 'camsim.isp', 'camsim.scene', 'camsim.optics',",
            "):",
            "    assert name not in sys.modules, name",
        ]
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
