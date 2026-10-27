"""Tests for the ISO 12233-style slanted-edge e-SFR and its RAW-plane helper.

The slanted edge is the standard way to measure a camera's resolution from a
single capture. A vertical edge tilted a few degrees crosses every pixel phase
along the image, so projecting the pixels onto the edge normal and binning at
sub-pixel spacing reconstructs the edge-spread function (ESF) far more finely
than the pixel grid. Differentiating the ESF gives the line-spread function
(LSF); the magnitude of its Fourier transform, normalized to one at zero
frequency, is the modulation transfer function (MTF), and the frequencies where
it falls to 50 % and 20 % are the familiar MTF50 and MTF20 numbers.

The references used here come from an independent code path
(:mod:`camsim.optics.psf`): a Gaussian, box, or Airy kernel's discrete transfer
function is evaluated analytically at arbitrary frequencies, while the metric
estimates the same curve from a rendered edge. Agreement within a few thousandths
of a cycle per pixel is the evidence that the projection, the derivative
response correction, and the DC normalization are all correct. The tests may
import the optics and scene layers to build references; the metric module itself
must not.

Real-world values: phone lenses measure MTF50 of roughly 0.1-0.4 cycles/pixel,
diffraction-limited systems at f/8 fall near 0.1, ISO 12233 recommends a 5 degree
tilt (2-7 degrees in printed charts), analysis regions are 100-400 pixels tall,
and four-fold analysis oversampling is conventional. A sharp synthetic edge
without optical blur does not cross 0.5 inside the measured band at all, so its
MTF50 is undefined and reported as NaN rather than invented.

Failure artefacts guarded here: gamma-encoded input biases every frequency,
noise or aliasing near Nyquist fakes a high-frequency tail, an over-corrected
derivative response inflates MTF near Nyquist, border pixels contaminated by
convolution padding masquerade as edge energy, an incorrect edge angle smears
the ESF, and averaging the two green phases of a Bayer quincunx silently
low-passes the measurement.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.typing import NDArray

from camsim.core.frames import CFAPattern, RawFrame
from camsim.metrics.mtf import MTFCurve, MTFResult, extract_raw_plane, slanted_edge_mtf
from camsim.optics.propagation import apply_psf
from camsim.optics.psf import airy_psf, box_psf, gaussian_psf, psf_mtf, psf_mtf50
from camsim.scene.targets import slanted_edge_target

# --------------------------------------------------------------------------- #
# Brief contract tests (verbatim from the task brief)
# --------------------------------------------------------------------------- #


def test_gaussian_mtf50_analytic():
    from camsim.optics.psf import gaussian_psf  # noqa: I001 - kept verbatim from the brief
    from camsim.optics.propagation import apply_psf
    from camsim.scene.targets import slanted_edge_target

    edge = slanted_edge_target((256, 256), angle_deg=5.0)
    blurred = apply_psf(edge[..., None], gaussian_psf(1.0, size=31))[..., 0]
    res = slanted_edge_mtf(blurred, pixel_pitch_um=1.12, focal_length_mm=6.4)
    assert res.mtf50_cp == pytest.approx(0.1874, abs=0.01)


def test_mtf_curve_is_normalized_and_monotonic():
    from camsim.scene.targets import slanted_edge_target

    edge = slanted_edge_target((256, 256), angle_deg=5.0)
    res = slanted_edge_mtf(edge, pixel_pitch_um=1.12)
    assert res.curve.mtf[0] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(res.curve.mtf) <= 1e-6)


def test_edge_angle_robustness():
    from camsim.scene.targets import slanted_edge_target

    for angle in (5, 10, 15, 20):
        img = slanted_edge_target((256, 256), angle_deg=angle)
        res = slanted_edge_mtf(img, pixel_pitch_um=1.12)
        assert res.edge_angle_deg == pytest.approx(angle, abs=1.0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# A one-sigma Gaussian at the reference 1.12 um pitch is a moderately soft phone
# lens; its discrete MTF50 is 0.1874 cycles/pixel (see psf_mtf50).
_GAUSSIAN_MTF50_REFERENCE = 0.1874


def _blurred_edge(
    sigma_px: float = 1.0,
    angle_deg: float = 5.0,
    shape: tuple[int, int] = (256, 256),
) -> NDArray[np.float64]:
    """Return a binary slanted edge blurred by a Gaussian kernel."""
    edge = slanted_edge_target(shape, angle_deg=angle_deg)
    return apply_psf(edge[..., None], gaussian_psf(sigma_px, size=31))[..., 0]


def _raw_frame(data: NDArray[np.uint16], cfa: CFAPattern) -> RawFrame:
    """Build a minimally valid RAW frame around a prepared mosaic."""
    return RawFrame(
        data=data,
        cfa=cfa,
        bit_depth=10,
        black_level_dn=64.0,
        exposure_s=0.01,
        analog_gain_db=0.0,
        digital_gain=1.0,
        mode="LCG",
        temperature_c=25.0,
        seed=0,
        frame_index=0,
        pixel_pitch_um=1.12,
        sensor_config_hash="mtf-test",
    )


def _valid_curve() -> MTFCurve:
    """Return a small, valid, strictly decreasing curve for value-object tests."""
    return MTFCurve([0.0, 0.1, 0.2], [1.0, 0.8, 0.5])


# --------------------------------------------------------------------------- #
# Physics references: the measured curve must agree with the PSF transfer
# --------------------------------------------------------------------------- #


def test_mtf_cycles_per_degree() -> None:
    blurred = _blurred_edge()
    assert slanted_edge_mtf(blurred, 1.12).mtf50_cp == pytest.approx(
        _GAUSSIAN_MTF50_REFERENCE, abs=0.01
    )
    with_focal = slanted_edge_mtf(blurred, pixel_pitch_um=1.12, focal_length_mm=6.4)
    without_focal = slanted_edge_mtf(blurred, pixel_pitch_um=1.12)

    assert math.isfinite(with_focal.mtf50_cyc_per_deg)
    assert math.isfinite(with_focal.mtf20_cyc_per_deg)
    assert math.isnan(without_focal.mtf50_cyc_per_deg)
    assert math.isnan(without_focal.mtf20_cyc_per_deg)
    assert without_focal.mtf50_cp == pytest.approx(with_focal.mtf50_cp)

    pixels_per_degree = 1.0 / math.degrees(math.atan(1.12e-3 / 6.4))
    assert pixels_per_degree == pytest.approx(99.73, abs=0.05)
    assert with_focal.mtf50_cyc_per_deg == pytest.approx(
        with_focal.mtf50_cp * pixels_per_degree, rel=1e-12
    )
    assert with_focal.mtf20_cyc_per_deg == pytest.approx(
        with_focal.mtf20_cp * pixels_per_degree, rel=1e-12
    )
    # A soft 1.12 um phone lens lands near 19 cycles/degree at 6.4 mm.
    assert 15.0 < with_focal.mtf50_cyc_per_deg < 25.0
    assert with_focal.mtf50_cp < with_focal.mtf20_cp < with_focal.nyquist_cp


def test_gaussian_mtf_curve_matches_the_discrete_psf() -> None:
    kernel = gaussian_psf(1.0, size=31)
    edge = slanted_edge_target((256, 256), angle_deg=5.0)
    result = slanted_edge_mtf(apply_psf(edge[..., None], kernel)[..., 0], 1.12)
    reference = psf_mtf(kernel, result.curve.frequency_cp)
    # The estimate differs from the ideal discrete transfer function only by the
    # binning and window discretisation, measured at a few thousandths.
    assert np.allclose(result.curve.mtf, reference, atol=0.01)


def test_mtf_noise_robustness() -> None:
    clean = _blurred_edge()
    clean_mtf50 = slanted_edge_mtf(clean, 1.12).mtf50_cp

    rng = np.random.default_rng(20261026)
    noisy = clean + rng.normal(0.0, 0.02, clean.shape)
    result = slanted_edge_mtf(noisy, 1.12)

    # 0.01 cycles/pixel covers the measured drift (about 0.001) with margin.
    assert result.mtf50_cp == pytest.approx(clean_mtf50, abs=0.01)
    assert result.edge_angle_deg == pytest.approx(5.0, abs=0.5)
    assert result.curve.mtf[0] == pytest.approx(1.0, abs=1e-6)
    assert np.all(result.curve.mtf >= 0.0)
    assert np.all(result.curve.mtf <= 1.0)


def test_box_psf_first_zero() -> None:
    width_px = 4.0
    kernel = box_psf(width_px)
    edge = slanted_edge_target((256, 256), angle_deg=5.0)
    result = slanted_edge_mtf(apply_psf(edge[..., None], kernel)[..., 0], 1.12)

    frequency = result.curve.frequency_cp
    response = result.curve.mtf
    slopes = np.diff(response)
    minima = np.flatnonzero((slopes[:-1] < 0.0) & (slopes[1:] > 0.0)) + 1
    assert minima.size > 0, "the box MTF must dip to a zero and recover"
    first_minimum = int(minima[0])
    assert frequency[first_minimum] == pytest.approx(1.0 / width_px, abs=0.01)
    assert response[first_minimum] < 0.05
    # The response recovers after the zero: the curve is not forced monotonic.
    assert response[first_minimum:].max() > 0.1


def test_airy_mtf() -> None:
    kernel = airy_psf(f_number=4.0, wavelength_nm=550.0, pixel_pitch_um=1.12)
    edge = slanted_edge_target((256, 256), angle_deg=5.0)
    result = slanted_edge_mtf(apply_psf(edge[..., None], kernel)[..., 0], 1.12)
    # 0.01 cycles/pixel covers the edge-projection discretisation; the Airy
    # response itself oscillates and is not smoothed into a monotone envelope.
    assert result.mtf50_cp == pytest.approx(psf_mtf50(kernel), abs=0.01)


def test_sharp_edge_has_no_crossing_within_the_measured_band() -> None:
    result = slanted_edge_mtf(slanted_edge_target((256, 256), angle_deg=5.0), 1.12)
    assert result.curve.mtf.min() > 0.5
    assert math.isnan(result.mtf50_cp)
    assert math.isnan(result.mtf20_cp)
    assert math.isnan(result.mtf50_cyc_per_deg)
    assert math.isnan(result.mtf20_cyc_per_deg)


def test_window_and_oversampling_options() -> None:
    blurred = _blurred_edge()
    for window in ("hamming", "none"):
        result = slanted_edge_mtf(blurred, 1.12, window=window)
        assert result.curve.mtf[0] == pytest.approx(1.0, abs=1e-6)
        assert result.mtf50_cp == pytest.approx(_GAUSSIAN_MTF50_REFERENCE, abs=0.01)

    estimates = [
        slanted_edge_mtf(blurred, 1.12, oversampling=factor).mtf50_cp for factor in (1, 2, 3, 4, 8)
    ]
    assert all(
        estimate == pytest.approx(_GAUSSIAN_MTF50_REFERENCE, abs=0.01) for estimate in estimates
    )
    # More oversampling converges towards the discrete reference.
    assert abs(estimates[-1] - _GAUSSIAN_MTF50_REFERENCE) < abs(
        estimates[0] - _GAUSSIAN_MTF50_REFERENCE
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("oversampling", [0, -1, 1.5, True, "4", None])
def test_rejects_invalid_oversampling(oversampling: object) -> None:
    with pytest.raises(ValueError, match="oversampling"):
        slanted_edge_mtf(_blurred_edge(), 1.12, oversampling=oversampling)


@pytest.mark.parametrize("window", ["blackman", "", "HAMMING", None, 3])
def test_rejects_invalid_window(window: object) -> None:
    with pytest.raises(ValueError, match="window"):
        slanted_edge_mtf(_blurred_edge(), 1.12, window=window)


@pytest.mark.parametrize("pitch", [0.0, -1.0, math.nan, math.inf, True, "1.12", None])
def test_rejects_invalid_pixel_pitch(pitch: object) -> None:
    with pytest.raises(ValueError, match="pixel_pitch_um"):
        slanted_edge_mtf(_blurred_edge(), pitch)


@pytest.mark.parametrize("focal", [0.0, -1.0, math.nan, math.inf, True, "6.4"])
def test_rejects_invalid_focal_length(focal: object) -> None:
    with pytest.raises(ValueError, match="focal_length_mm"):
        slanted_edge_mtf(_blurred_edge(), 1.12, focal_length_mm=focal)


def test_accepts_integer_and_numpy_scalar_parameters() -> None:
    result = slanted_edge_mtf(
        _blurred_edge(),
        np.float64(1.12),
        focal_length_mm=np.float64(6.4),
        oversampling=np.int64(4),
    )
    assert math.isfinite(result.mtf50_cp)


_INVALID_IMAGES: list[tuple[str, object]] = [
    ("one-dimensional", np.zeros(16)),
    ("three-dimensional", np.zeros((16, 16, 1))),
    ("empty", np.zeros((0, 16))),
    ("zero-width", np.zeros((16, 0))),
    ("boolean", np.zeros((16, 16), dtype=bool)),
    ("complex", np.zeros((16, 16), dtype=complex)),
    ("strings", np.full((16, 16), "x")),
    ("object", np.zeros((16, 16), dtype=object)),
    ("nan", np.full((16, 16), math.nan)),
    ("infinity", np.full((16, 16), math.inf)),
    ("negative-infinity", np.full((16, 16), -math.inf)),
    ("scalar", 1.0),
    ("none", None),
]


@pytest.mark.parametrize(
    ("label", "image"), _INVALID_IMAGES, ids=[label for label, _ in _INVALID_IMAGES]
)
def test_rejects_invalid_images(label: str, image: object) -> None:
    with pytest.raises(ValueError, match="image"):
        slanted_edge_mtf(image, 1.12)


@pytest.mark.parametrize("shape", [(8, 8), (16, 16), (20, 20), (40, 4)])
def test_rejects_images_too_small_for_an_analysis_region(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="too small"):
        slanted_edge_mtf(np.zeros(shape, dtype=np.float64), 1.12)


def test_rejects_regions_without_a_usable_gradient() -> None:
    with pytest.raises(ValueError, match="gradient"):
        slanted_edge_mtf(np.full((64, 64), 0.5), 1.12)

    # A horizontal step has no horizontal Sobel gradient.
    horizontal = np.zeros((64, 64), dtype=np.float64)
    horizontal[32:] = 1.0
    with pytest.raises(ValueError, match="gradient"):
        slanted_edge_mtf(horizontal, 1.12)

    # A near-horizontal edge is outside the near-vertical estimator's domain.
    rows, columns = np.mgrid[0:128, 0:128]
    angle = math.radians(70.0)
    steep = np.asarray(
        (columns - 63.5) * math.cos(angle) - (rows - 63.5) * math.sin(angle) >= 0,
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="horizontal"):
        slanted_edge_mtf(steep, 1.12)


def test_border_margin_excludes_contaminated_pixels() -> None:
    edge = slanted_edge_target((256, 256), angle_deg=5.0)
    baseline = slanted_edge_mtf(edge, 1.12)

    # Columns 0-2 and 253-255 lie inside the 6-pixel margin; changing them must
    # not reach the estimator at all.
    corrupted = edge.copy()
    corrupted[:, :3] = 1.0
    corrupted[:, -3:] = 0.0
    np.testing.assert_array_equal(slanted_edge_mtf(corrupted, 1.12).curve.mtf, baseline.curve.mtf)

    # An edge that only exists inside the margin disappears from the analysis.
    border_only = np.zeros((64, 64), dtype=np.float64)
    border_only[:, :2] = 1.0
    with pytest.raises(ValueError, match="gradient"):
        slanted_edge_mtf(border_only, 1.12)


def test_border_margin_scales_with_the_short_side() -> None:
    edge = slanted_edge_target((400, 400), angle_deg=5.0)
    baseline = slanted_edge_mtf(edge, 1.12)

    corrupted = edge.copy()
    corrupted[:, :8] = 1.0
    corrupted[:, -8:] = 0.0
    np.testing.assert_array_equal(slanted_edge_mtf(corrupted, 1.12).curve.mtf, baseline.curve.mtf)


def test_small_region_still_produces_a_curve() -> None:
    result = slanted_edge_mtf(slanted_edge_target((24, 24), angle_deg=5.0), 1.12)
    assert result.curve.mtf.size >= 2
    assert result.curve.mtf[0] == pytest.approx(1.0, abs=1e-6)
    assert result.curve.frequency_cp[-1] <= 0.5


def test_mtf_is_deterministic_and_leaves_the_input_untouched() -> None:
    blurred = _blurred_edge()
    before = blurred.copy()

    first = slanted_edge_mtf(blurred, 1.12, focal_length_mm=6.4)
    second = slanted_edge_mtf(blurred, 1.12, focal_length_mm=6.4)

    np.testing.assert_array_equal(blurred, before)
    np.testing.assert_array_equal(first.curve.frequency_cp, second.curve.frequency_cp)
    np.testing.assert_array_equal(first.curve.mtf, second.curve.mtf)
    assert first.mtf50_cp == second.mtf50_cp
    assert first.mtf50_cyc_per_deg == second.mtf50_cyc_per_deg
    assert first.edge_angle_deg == second.edge_angle_deg


# --------------------------------------------------------------------------- #
# MTFCurve and MTFResult value objects
# --------------------------------------------------------------------------- #


def test_mtf_curve_accepts_sequences_and_stores_float64_copies() -> None:
    frequency = [0.0, 0.1, 0.2]
    response = [1.0, 0.8, 0.5]
    curve = MTFCurve(frequency, response)
    assert curve.frequency_cp.dtype == np.float64
    assert curve.mtf.dtype == np.float64
    np.testing.assert_array_equal(curve.frequency_cp, np.asarray(frequency))
    np.testing.assert_array_equal(curve.mtf, np.asarray(response))

    frequency_array = np.array([0.0, 0.25], dtype=np.float32)
    response_array = np.array([1.0, 0.75], dtype=np.float32)
    copied = MTFCurve(frequency_array, response_array)
    frequency_array[1] = 0.5
    response_array[1] = 0.25
    assert copied.frequency_cp[1] == 0.25
    assert copied.mtf[1] == 0.75


_INVALID_CURVES: list[tuple[str, object, object]] = [
    ("frequency-not-starting-at-zero", [0.1, 0.2], [1.0, 0.8]),
    ("repeated-frequency", [0.0, 0.0], [1.0, 0.8]),
    ("decreasing-frequency", [0.0, 0.2, 0.1], [1.0, 0.8, 0.6]),
    ("un-normalized", [0.0, 0.1], [0.9, 0.8]),
    ("negative-mtf", [0.0, 0.1], [1.0, -0.1]),
    ("non-finite-mtf", [0.0, 0.1], [1.0, math.nan]),
    ("infinite-mtf", [0.0, 0.1], [1.0, math.inf]),
    ("non-finite-frequency", [0.0, math.nan], [1.0, 0.5]),
    ("unequal-lengths", [0.0], [1.0, 0.5]),
    ("empty", [], []),
    ("two-dimensional", [[0.0, 0.1]], [[1.0, 0.5]]),
    ("scalar", 0.0, 1.0),
    ("boolean", np.array([False, True]), np.array([True, False])),
    ("complex", np.array([0.0, 0.1j]), np.array([1.0, 0.5])),
    ("strings", np.array(["a", "b"]), np.array(["c", "d"])),
    ("none", None, None),
]


@pytest.mark.parametrize(
    ("label", "frequency", "response"),
    _INVALID_CURVES,
    ids=[label for label, _, _ in _INVALID_CURVES],
)
def test_mtf_curve_rejects_invalid_arrays(label: str, frequency: object, response: object) -> None:
    with pytest.raises(ValueError):
        MTFCurve(frequency, response)


def test_mtf_curve_is_frozen() -> None:
    curve = _valid_curve()
    with pytest.raises(FrozenInstanceError):
        curve.mtf = np.ones(3)  # type: ignore[misc]


def test_mtf_result_holds_crossings_and_defaults() -> None:
    curve = _valid_curve()
    result = MTFResult(
        mtf50_cp=0.2,
        mtf20_cp=0.3,
        mtf50_cyc_per_deg=19.9,
        mtf20_cyc_per_deg=29.9,
        curve=curve,
        edge_angle_deg=5.0,
    )
    assert result.curve is curve
    assert result.nyquist_cp == 0.5
    assert result.mtf50_cp == 0.2


def test_mtf_result_allows_nan_crossings() -> None:
    result = MTFResult(
        mtf50_cp=math.nan,
        mtf20_cp=math.nan,
        mtf50_cyc_per_deg=math.nan,
        mtf20_cyc_per_deg=math.nan,
        curve=_valid_curve(),
        edge_angle_deg=5.0,
    )
    assert math.isnan(result.mtf50_cp)
    assert math.isnan(result.mtf20_cp)


_INVALID_RESULTS: list[tuple[str, dict[str, object]]] = [
    (
        "infinite-mtf50",
        {"mtf50_cp": math.inf, "mtf20_cp": 0.3, "mtf50_cyc_per_deg": 1.0, "mtf20_cyc_per_deg": 2.0},
    ),
    (
        "negative-mtf50",
        {"mtf50_cp": -0.1, "mtf20_cp": 0.3, "mtf50_cyc_per_deg": 1.0, "mtf20_cyc_per_deg": 2.0},
    ),
    (
        "mtf50-beyond-nyquist",
        {"mtf50_cp": 0.6, "mtf20_cp": 0.3, "mtf50_cyc_per_deg": 1.0, "mtf20_cyc_per_deg": 2.0},
    ),
    (
        "negative-degrees",
        {
            "mtf50_cp": 0.2,
            "mtf20_cp": 0.3,
            "mtf50_cyc_per_deg": -1.0,
            "mtf20_cyc_per_deg": 2.0,
        },
    ),
    (
        "not-a-curve",
        {"mtf50_cp": 0.2, "mtf20_cp": 0.3, "mtf50_cyc_per_deg": 1.0, "mtf20_cyc_per_deg": 2.0},
    ),
    (
        "non-finite-angle",
        {
            "mtf50_cp": 0.2,
            "mtf20_cp": 0.3,
            "mtf50_cyc_per_deg": 1.0,
            "mtf20_cyc_per_deg": 2.0,
            "edge_angle_deg": math.nan,
        },
    ),
]


@pytest.mark.parametrize(
    ("label", "fields"), _INVALID_RESULTS, ids=[label for label, _ in _INVALID_RESULTS]
)
def test_mtf_result_rejects_invalid_fields(label: str, fields: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "mtf50_cp": 0.2,
        "mtf20_cp": 0.3,
        "mtf50_cyc_per_deg": 1.0,
        "mtf20_cyc_per_deg": 2.0,
        "curve": _valid_curve(),
        "edge_angle_deg": 5.0,
    }
    payload.update(fields)
    if label == "not-a-curve":
        payload["curve"] = "not a curve"
    with pytest.raises(ValueError):
        MTFResult(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize("nyquist", [0.0, -0.5, math.inf, math.nan, True])
def test_mtf_result_rejects_invalid_nyquist(nyquist: object) -> None:
    with pytest.raises(ValueError, match="nyquist_cp"):
        MTFResult(
            mtf50_cp=0.2,
            mtf20_cp=0.3,
            mtf50_cyc_per_deg=1.0,
            mtf20_cyc_per_deg=2.0,
            curve=_valid_curve(),
            edge_angle_deg=5.0,
            nyquist_cp=nyquist,  # type: ignore[arg-type]
        )


def test_mtf_result_is_frozen() -> None:
    result = MTFResult(
        mtf50_cp=0.2,
        mtf20_cp=0.3,
        mtf50_cyc_per_deg=1.0,
        mtf20_cyc_per_deg=2.0,
        curve=_valid_curve(),
        edge_angle_deg=5.0,
    )
    with pytest.raises(FrozenInstanceError):
        result.mtf50_cp = 0.1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# RAW-plane extraction
# --------------------------------------------------------------------------- #


def test_raw_green_plane_sampling_note(raw_frame: RawFrame, mono_raw_frame: RawFrame) -> None:
    plane, spacing = extract_raw_plane(raw_frame, channel="G")
    assert spacing == 2.0
    assert plane.dtype == np.float64
    assert plane.shape == (raw_frame.data.shape[0] // 2, raw_frame.data.shape[1] // 2)
    # The first green phase in the RGGB tile is the top-right site (row 0, col 1).
    np.testing.assert_array_equal(plane, raw_frame.data[0::2, 1::2].astype(np.float64))
    # The caller must scale the pitch by the returned spacing.
    assert raw_frame.pixel_pitch_um * spacing == pytest.approx(2.24)

    mono, mono_spacing = extract_raw_plane(mono_raw_frame, channel="M")
    assert mono_spacing == 1.0
    assert mono.shape == mono_raw_frame.data.shape
    assert mono.dtype == np.float64
    np.testing.assert_array_equal(mono, mono_raw_frame.data.astype(np.float64))

    # Defaulting to "G" is a Bayer convention; a mono frame has no green site.
    with pytest.raises(ValueError, match="MONO"):
        extract_raw_plane(mono_raw_frame)

    rccc = _raw_frame(np.zeros((8, 8), dtype=np.uint16), CFAPattern.RCCC)
    with pytest.raises(ValueError, match="RCCC"):
        extract_raw_plane(rccc, channel="R")


_BAYER_PHASES: list[tuple[CFAPattern, dict[str, tuple[int, int]]]] = [
    (CFAPattern.RGGB, {"R": (0, 0), "G": (0, 1), "B": (1, 1)}),
    (CFAPattern.BGGR, {"R": (1, 1), "G": (0, 1), "B": (0, 0)}),
    (CFAPattern.GRBG, {"R": (0, 1), "G": (0, 0), "B": (1, 0)}),
    (CFAPattern.GBRG, {"R": (1, 0), "G": (0, 0), "B": (0, 1)}),
]


@pytest.mark.parametrize(
    ("cfa", "offsets"),
    _BAYER_PHASES,
    ids=[cfa.value for cfa, _ in _BAYER_PHASES],
)
def test_raw_plane_selects_the_first_phase_of_each_colour(
    cfa: CFAPattern, offsets: dict[str, tuple[int, int]]
) -> None:
    data = np.arange(36, dtype=np.uint16).reshape(6, 6)
    frame = _raw_frame(data, cfa)
    for channel, (dy, dx) in offsets.items():
        plane, spacing = extract_raw_plane(frame, channel=channel)
        assert spacing == 2.0
        np.testing.assert_array_equal(plane, data[dy::2, dx::2].astype(np.float64))


def test_raw_plane_crops_odd_shapes_per_phase() -> None:
    data = np.arange(25, dtype=np.uint16).reshape(5, 5)
    frame = _raw_frame(data, CFAPattern.RGGB)
    red, _ = extract_raw_plane(frame, "R")
    green, _ = extract_raw_plane(frame, "G")
    blue, _ = extract_raw_plane(frame, "B")
    assert red.shape == (3, 3)
    assert green.shape == (3, 2)
    assert blue.shape == (2, 2)


def test_raw_plane_dtype_and_immutability(raw_frame: RawFrame) -> None:
    original = raw_frame.data.copy()
    plane, _ = extract_raw_plane(raw_frame, "G")
    plane[0, 0] = -12345.0
    np.testing.assert_array_equal(raw_frame.data, original)


@pytest.mark.parametrize("channel", ["M", "X", "", 1, None, 3.0])
def test_raw_plane_rejects_invalid_channels(raw_frame: RawFrame, channel: object) -> None:
    with pytest.raises(ValueError, match="channel"):
        extract_raw_plane(raw_frame, channel=channel)  # type: ignore[arg-type]


def test_raw_plane_rejects_non_frames() -> None:
    with pytest.raises(TypeError, match="RawFrame"):
        extract_raw_plane(np.zeros((4, 4), dtype=np.uint16), "G")  # type: ignore[arg-type]


def test_raw_plane_rejects_a_frame_without_the_requested_phase() -> None:
    frame = _raw_frame(np.zeros((1, 1), dtype=np.uint16), CFAPattern.RGGB)
    with pytest.raises(ValueError, match="phase"):
        extract_raw_plane(frame, channel="G")


# --------------------------------------------------------------------------- #
# Package surface and import direction
# --------------------------------------------------------------------------- #


def test_metrics_mtf_exports_the_documented_names() -> None:
    import camsim.metrics.mtf as module

    assert set(module.__all__) == {
        "MTFCurve",
        "MTFResult",
        "extract_raw_plane",
        "slanted_edge_mtf",
    }


def test_metrics_mtf_does_not_import_simulator_modules() -> None:
    code = "\n".join(
        [
            "import sys",
            "import camsim.metrics.mtf",
            "from camsim.metrics.mtf import MTFCurve, MTFResult, extract_raw_plane, "
            "slanted_edge_mtf",
            "for name in (",
            "    'camsim.sensor', 'camsim.isp', 'camsim.scene', 'camsim.optics',",
            "):",
            "    assert name not in sys.modules, name",
        ]
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
