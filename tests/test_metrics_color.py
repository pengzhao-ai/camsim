"""Tests for the delta-E76 and delta-E2000 colour metrics.

Colour difference is measured in CIE 1976 L*a*b*, a space built so that equal
numeric distances approximate equal perceptual differences. L* runs from 0
(black) to 100 (diffuse white), while a* and b* encode green-red and blue-yellow
opposition; a 24-patch ColorChecker spans roughly L* 20-95 and a*/b* about -60
to 100. Delta-E76 is the plain Euclidean distance in that space. Delta-E2000
adds the corrections the CIE recommended after years of visual experiments: it
rescales chroma with the ``G`` factor, weights lightness, chroma, and hue by
``S_L``, ``S_C``, and ``S_H``, and adds a rotation term ``R_T`` for the blue
region where hue differences are perceptually compressed. For camera colour,
delta-E00 of about 1-5 is a good result and above about 5 is visible.

The measurement compares a display-encoded RGB patch with a reference measured
under a specific illuminant, so three details are load-bearing. The display
values must be linearized with the exact sRGB EOTF before any colourimetry
(Display P3 shares that curve); a metric that treats gamma-encoded values as
linear reports a systematic lightness and hue error. The reference Lab is
relative to its own white point, so the measured XYZ must be chromatically
adapted from D65 (the native white of both sRGB and Display P3) to that white
point with the Bradford transform before the Lab conversion; comparing Lab
values computed under different white points manufactures a colour error that
belongs to neither the camera nor the chart. And the patch mean must be taken
in linear light: the mean of gamma-encoded values is lighter than the mean of
the linear values, and the difference grows with patch contrast.

Every formula here is implemented from scratch because the metrics layer must
not import the simulator it judges. The cross-checks against colour-science in
this file are the independent anchors: the RGB-to-XYZ matrices, the Bradford
adaptation, the sRGB EOTF, the XYZ-to-Lab conversion, and a set of published
CIEDE2000 pairs. Known values from the Sharma et al. (2005) CIEDE2000 test data
pin the hue-angle and rotation branches, which are the parts most easily got
wrong when the formula is transcribed.

Failure artefacts guarded here: linearizing display values twice, averaging
display values instead of linear ones, comparing Lab under mismatched white
points, clipping an out-of-range display patch instead of rejecting it, using a
chart layout without a colour reference, a reference whose row count does not
match the patch count, a monochrome image where three channels are required, and
taking a CCM's in-sample fit as an independent colour measurement (the metrics
layer never does: it compares against spectral ground truth).
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import FrozenInstanceError
from typing import Any

import colour
import numpy as np
import pytest
from colour.adaptation import CHROMATIC_ADAPTATION_TRANSFORMS, chromatic_adaptation_VonKries
from colour.models import RGB_COLOURSPACES

from camsim.core.color import ColorReference
from camsim.metrics.chart import ChartLayout, PatchROI
from camsim.metrics.color import (
    _BRADFORD,
    _RGB_TO_XYZ_DISPLAY_P3,
    _RGB_TO_XYZ_SRGB,
    ColorResult,
    PatchColorResult,
    _bradford_adaptation,
    _delta_e76,
    _delta_e2000,
    _srgb_eotf,
    delta_e_patches,
    linear_rgb_to_lab,
)

_D65 = (0.3127, 0.3290)
_WARM = (0.4369, 0.4041)

# --------------------------------------------------------------------------- #
# Brief contract tests (verbatim from the task brief)
# --------------------------------------------------------------------------- #


def test_delta_e_identity():
    from camsim.metrics.color import _delta_e2000

    lab = np.array([50.0, 10.0, -10.0])
    assert _delta_e2000(lab, lab) == pytest.approx(0.0, abs=1e-9)


def test_delta_e_patches_uses_layout_reference():
    ref = ColorReference(
        lab=np.array([[50.0, 10.0, -10.0]]),
        white_point_xy=(0.3127, 0.3290),
        illuminant_name="D65",
    )
    layout = ChartLayout(name="t", patches=(PatchROI("p", 0, 0, 8, 8),), reference=ref)
    img = np.full((8, 8, 3), 0.5, np.float32)
    res = delta_e_patches(img, layout)
    assert res.per_patch[0].delta_e2000 > 0.0


def test_delta_e_matches_colour_science():
    lab1 = np.array([50.0, 10.0, -10.0])
    lab2 = np.array([45.0, 12.0, -8.0])
    expected = colour.delta_E(lab1, lab2, method="CIE 2000")
    # Use a synthetic ColorReference and a precomputed Lab image to exercise the formula.
    # (The implementation exposes `_delta_e2000` for this exact comparison.)
    from camsim.metrics.color import _delta_e2000

    assert _delta_e2000(lab1, lab2) == pytest.approx(expected, rel=1e-9)


def test_non_d65_reference_is_used():
    ref_d65 = ColorReference(
        lab=np.array([[50.0, 0.0, 0.0]]),
        white_point_xy=(0.3127, 0.3290),
        illuminant_name="D65",
    )
    ref_warm = ColorReference(
        lab=np.array([[50.0, 5.0, 10.0]]),
        white_point_xy=(0.4369, 0.4041),
        illuminant_name="3000K",
    )
    layout = ChartLayout(name="t", patches=(PatchROI("p", 0, 0, 8, 8),), reference=ref_warm)
    img = np.full((8, 8, 3), 0.5, np.float32)
    res = delta_e_patches(img, layout, reference=ref_warm)
    assert res.per_patch[0].delta_e2000 != pytest.approx(0.0, abs=1e-3)

    # The same image against the D65 reference gives a different difference:
    # the warm reference is actually used rather than silently replaced.
    d65_layout = ChartLayout(name="t", patches=(PatchROI("p", 0, 0, 8, 8),), reference=ref_d65)
    d65_res = delta_e_patches(img, d65_layout)
    assert d65_res.per_patch[0].delta_e2000 != pytest.approx(res.per_patch[0].delta_e2000, abs=1e-3)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _reference(
    lab: Any,
    white_point: tuple[float, float] = _D65,
    name: str = "D65",
) -> ColorReference:
    """Return a reference carrying ``lab`` under ``white_point``."""
    return ColorReference(
        lab=np.asarray(lab, dtype=np.float64),
        white_point_xy=white_point,
        illuminant_name=name,
    )


def _layout(
    patches: tuple[PatchROI, ...],
    reference: ColorReference | None = None,
) -> ChartLayout:
    """Return a chart layout with the requested patches and reference."""
    return ChartLayout(name="test", patches=patches, reference=reference)


def _single_patch(reference: ColorReference | None = None) -> ChartLayout:
    """Return a one-patch 8 x 8 layout covering the top-left corner."""
    return _layout((PatchROI("p", 0, 0, 8, 8),), reference)


def _colour_science_lab(
    rgb: Any,
    colourspace: str,
    white_point: tuple[float, float],
) -> np.ndarray:
    """Return the reference Lab pipeline as colour-science computes it.

    The RGB-to-XYZ matrix is derived from the chromaticities (not the rounded
    published table), the XYZ is adapted from D65 to the requested white point
    with Bradford, and the Lab conversion uses that same white point.
    """
    space = RGB_COLOURSPACES[colourspace].copy()
    space.use_derived_matrix_RGB_to_XYZ = True
    space.use_derived_matrix_XYZ_to_RGB = True
    xyz = np.asarray(rgb, dtype=np.float64) @ space.matrix_RGB_to_XYZ.T
    adapted = chromatic_adaptation_VonKries(
        xyz,
        colour.xy_to_XYZ(np.asarray(_D65)),
        colour.xy_to_XYZ(np.asarray(white_point)),
        transform="Bradford",
    )
    return np.asarray(
        colour.XYZ_to_Lab(adapted, illuminant=np.asarray(white_point)),
        dtype=np.float64,
    )


def _measured_lab(image: np.ndarray, reference: ColorReference, roi: PatchROI) -> np.ndarray:
    """Return the linear patch mean's Lab for a display-encoded RGB image."""
    linear = _srgb_eotf(np.asarray(image, dtype=np.float64))
    patch = linear[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    mean_rgb = patch.reshape(-1, 3).mean(axis=0)
    return linear_rgb_to_lab(mean_rgb, "sRGB", reference.white_point_xy)


# --------------------------------------------------------------------------- #
# Delta-E76 and CIEDE2000 formulas
# --------------------------------------------------------------------------- #


def test_delta_e76_formula() -> None:
    """Delta-E76 is the Euclidean Lab distance, hand-computed and cross-checked."""
    lab1 = np.array([50.0, 10.0, -10.0])
    lab2 = np.array([45.0, 12.0, -8.0])
    # (50 - 45)^2 + (10 - 12)^2 + (-10 + 8)^2 = 25 + 4 + 4 = 33.
    assert _delta_e76(lab1, lab2) == pytest.approx(math.sqrt(33.0), rel=1e-15)
    assert _delta_e76(lab1, lab1) == pytest.approx(0.0, abs=0.0)
    assert _delta_e76(lab1, lab2) == pytest.approx(
        float(colour.delta_E(lab1, lab2, method="CIE 1976")), rel=1e-12
    )
    # The metric is a distance: it is symmetric.
    assert _delta_e76(lab1, lab2) == pytest.approx(float(_delta_e76(lab2, lab1)), rel=1e-15)


_PUBLISHED_CIEDE2000_PAIRS = (
    pytest.param(
        (50.0, 2.6772, -79.7751),
        (50.0, 0.0, -82.7485),
        2.0425,
        id="blue-hue-rotation",
    ),
    pytest.param(
        (50.0, 0.0, 0.0),
        (50.0, -1.0, 2.0),
        2.3669,
        id="neutral-to-low-chroma",
    ),
    pytest.param(
        (50.0, 2.49, -0.001),
        (50.0, -2.49, 0.0009),
        7.1792,
        id="hue-flip-across-180",
    ),
    pytest.param(
        (50.0, 2.5, 0.0),
        (73.0, 25.0, -18.0),
        27.1492,
        id="large-difference",
    ),
)


@pytest.mark.parametrize(("lab1", "lab2", "published"), _PUBLISHED_CIEDE2000_PAIRS)
def test_delta_e2000_matches_published_sharma_pairs(
    lab1: tuple[float, float, float],
    lab2: tuple[float, float, float],
    published: float,
) -> None:
    """CIEDE2000 reproduces the published Sharma et al. test pairs."""
    first = np.asarray(lab1)
    second = np.asarray(lab2)
    expected = float(colour.delta_E(first, second, method="CIE 2000"))

    assert float(_delta_e2000(first, second)) == pytest.approx(expected, rel=1e-12)
    assert float(_delta_e2000(first, second)) == pytest.approx(published, abs=1e-4)
    # And the reverse pair, which exercises the complementary hue branch.
    assert float(_delta_e2000(second, first)) == pytest.approx(expected, rel=1e-12)


def test_delta_e2000_lightness_pair_matches_closed_form() -> None:
    """A neutral pair with only a lightness difference reduces to |dL| / S_L."""
    lab1 = np.array([50.0, 0.0, 0.0])
    lab2 = np.array([60.0, 0.0, 0.0])
    # L_bar = 55, S_L = 1 + 0.015 * (L_bar - 50)^2 / sqrt(20 + (L_bar - 50)^2).
    lightness_weight = 1.0 + (0.015 * 25.0) / math.sqrt(45.0)
    assert float(_delta_e2000(lab1, lab2)) == pytest.approx(10.0 / lightness_weight, rel=1e-14)
    assert float(_delta_e2000(lab1, lab2)) == pytest.approx(
        float(colour.delta_E(lab1, lab2, method="CIE 2000")), rel=1e-12
    )


def test_delta_e2000_differs_from_delta_e76_on_a_hue_pair() -> None:
    """The hue and rotation terms make CIEDE2000 differ from the Euclidean distance."""
    lab1 = np.array([50.0, 2.49, -0.001])
    lab2 = np.array([50.0, -2.49, 0.0009])
    difference_76 = float(_delta_e76(lab1, lab2))
    difference_00 = float(_delta_e2000(lab1, lab2))
    assert difference_00 == pytest.approx(7.1792, abs=1e-4)
    assert abs(difference_00 - difference_76) > 0.5


def test_delta_e_formulas_are_vectorized_and_broadcast() -> None:
    """Both formulas accept (3,) or (N, 3) and broadcast a single colour."""
    rng = np.random.default_rng(1)
    samples = np.column_stack(
        (
            rng.uniform(0.0, 100.0, 12),
            rng.uniform(-80.0, 80.0, 12),
            rng.uniform(-80.0, 80.0, 12),
        )
    )
    target = np.array([50.0, 10.0, -10.0])

    packed_76 = _delta_e76(samples, target)
    packed_00 = _delta_e2000(samples, target)
    assert packed_76.shape == (12,)
    assert packed_00.shape == (12,)
    np.testing.assert_allclose(
        packed_76, colour.delta_E(samples, target, method="CIE 1976"), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        packed_00, colour.delta_E(samples, target, method="CIE 2000"), rtol=0.0, atol=1e-12
    )

    # A single pair produces a scalar-equivalent result; the identity is zero.
    assert np.ndim(_delta_e2000(target, target)) == 0
    assert float(_delta_e2000(target, target)) == pytest.approx(0.0, abs=1e-12)

    # Two equally shaped batches compare row by row.
    shifted = samples + 3.0
    np.testing.assert_allclose(
        _delta_e2000(samples, shifted),
        colour.delta_E(samples, shifted, method="CIE 2000"),
        rtol=0.0,
        atol=1e-12,
    )


def test_delta_e_rejects_malformed_lab() -> None:
    """Boolean, complex, non-finite, and mis-shaped Lab arrays are rejected."""
    good = np.array([50.0, 10.0, -10.0])
    with pytest.raises(ValueError):
        _delta_e2000(np.array([True, False, True]), good)
    with pytest.raises(ValueError):
        _delta_e76(np.array([1.0 + 1.0j, 0.0, 0.0]), good)
    with pytest.raises(ValueError):
        _delta_e76(np.array([50.0, math.nan, -10.0]), good)
    with pytest.raises(ValueError):
        _delta_e2000(np.array([50.0, 10.0]), good)
    with pytest.raises(ValueError):
        _delta_e2000(np.zeros((2, 2)), good)


# --------------------------------------------------------------------------- #
# Local constants: matrices, Bradford adaptation, and the sRGB EOTF
# --------------------------------------------------------------------------- #


def _derived_colourspace(name: str) -> colour.RGB_Colourspace:
    """Return a colour-science space whose matrices are derived from chromaticities."""
    space = RGB_COLOURSPACES[name].copy()
    space.use_derived_matrix_RGB_to_XYZ = True
    space.use_derived_matrix_XYZ_to_RGB = True
    return space


def test_rgb_to_xyz_matrices_match_colour_science() -> None:
    """The local matrices reproduce colour-science's chromaticity derivation."""
    np.testing.assert_allclose(
        _RGB_TO_XYZ_SRGB,
        _derived_colourspace("sRGB").matrix_RGB_to_XYZ,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _RGB_TO_XYZ_DISPLAY_P3,
        _derived_colourspace("Display P3").matrix_RGB_to_XYZ,
        rtol=0.0,
        atol=1e-12,
    )
    # Both spaces are D65-referenced, so (1, 1, 1) maps to the D65 white.
    d65_xyz = colour.xy_to_XYZ(np.asarray(_D65))
    np.testing.assert_allclose(_RGB_TO_XYZ_SRGB @ np.ones(3), d65_xyz, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_RGB_TO_XYZ_DISPLAY_P3 @ np.ones(3), d65_xyz, rtol=0.0, atol=1e-12)
    # The published sRGB table is rounded to four decimals, so it differs from
    # the chromaticity-derived matrix by up to about 4e-5 in Z; the stored
    # Display P3 matrix is already full precision.
    np.testing.assert_allclose(
        _RGB_TO_XYZ_SRGB,
        RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ,
        rtol=0.0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        _RGB_TO_XYZ_DISPLAY_P3,
        RGB_COLOURSPACES["Display P3"].matrix_RGB_to_XYZ,
        rtol=0.0,
        atol=1e-12,
    )


def test_bradford_adaptation_matches_colour_science() -> None:
    """The local Bradford matrix and Von Kries adaptation match colour-science."""
    np.testing.assert_allclose(
        _BRADFORD, CHROMATIC_ADAPTATION_TRANSFORMS["Bradford"], rtol=0.0, atol=0.0
    )
    rng = np.random.default_rng(2)
    xyz = np.column_stack(
        (
            rng.uniform(0.0, 1.0, 16),
            rng.uniform(0.0, 1.0, 16),
            rng.uniform(0.0, 1.2, 16),
        )
    )
    source = colour.xy_to_XYZ(np.asarray(_D65))
    destination = colour.xy_to_XYZ(np.asarray(_WARM))
    expected = chromatic_adaptation_VonKries(xyz, source, destination, transform="Bradford")
    np.testing.assert_allclose(
        _bradford_adaptation(xyz, source, destination), expected, rtol=0.0, atol=1e-12
    )
    # Adapting from a white point to itself is the identity.
    np.testing.assert_allclose(_bradford_adaptation(xyz, source, source), xyz, rtol=0.0, atol=1e-15)


def test_srgb_eotf_matches_colour_science() -> None:
    """The exact sRGB decoding curve matches colour-science, including the knee."""
    values = np.array([0.0, 1e-9, 0.04045, 0.04045 + 1e-12, 0.2, 0.5, 1.0, 1.0 + 1e-7, -1e-7])
    expected = colour.cctf_decoding(values, function="sRGB")
    # The IEC standard rounds the knee to 0.04045, while colour-science uses
    # the exact segment intersection 12.92 * 0.0031308 = 0.040449936. The two
    # branch conventions differ by 2.3e-9 at the rounded threshold, far below
    # one 8-bit LSB; everywhere else the curves agree to float rounding.
    np.testing.assert_allclose(_srgb_eotf(values), expected, rtol=0.0, atol=1e-8)
    assert float(_srgb_eotf(np.array(0.04045))) == 0.04045 / 12.92
    above_knee = float(np.nextafter(0.04045, 1.0))
    assert float(_srgb_eotf(np.array(above_knee))) == pytest.approx(
        ((above_knee + 0.055) / 1.055) ** 2.4, rel=1e-15
    )
    assert float(_srgb_eotf(np.array(0.0))) == pytest.approx(0.0, abs=0.0)
    assert float(_srgb_eotf(np.array(1.0))) == pytest.approx(1.0, rel=1e-15)


# --------------------------------------------------------------------------- #
# linear_rgb_to_lab
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("colourspace", "white_point"),
    (
        pytest.param("sRGB", _D65, id="srgb-d65"),
        pytest.param("DisplayP3", _D65, id="display-p3-d65"),
        pytest.param("sRGB", _WARM, id="srgb-3000k"),
        pytest.param("DisplayP3", _WARM, id="display-p3-3000k"),
    ),
)
def test_linear_rgb_to_lab_matches_colour_science(
    colourspace: str, white_point: tuple[float, float]
) -> None:
    """RGB-to-Lab matches the colour-science pipeline, including adaptation."""
    rng = np.random.default_rng(3)
    rgb = rng.uniform(0.0, 1.0, (4, 5, 3))
    expected = _colour_science_lab(
        rgb,
        "Display P3" if colourspace == "DisplayP3" else "sRGB",
        white_point,
    )
    np.testing.assert_allclose(
        linear_rgb_to_lab(rgb, colourspace, white_point), expected, rtol=0.0, atol=1e-10
    )


@pytest.mark.parametrize("colourspace", ("sRGB", "DisplayP3"))
@pytest.mark.parametrize("white_point", (_D65, _WARM), ids=("d65", "3000k"))
def test_linear_rgb_to_lab_maps_white_to_the_reference_white(
    colourspace: str, white_point: tuple[float, float]
) -> None:
    """Linear RGB (1, 1, 1) becomes L* = 100, a* = b* = 0 under any white point."""
    lab = linear_rgb_to_lab(np.ones(3), colourspace, white_point)
    np.testing.assert_allclose(lab, np.array([100.0, 0.0, 0.0]), rtol=0.0, atol=1e-9)
    black = linear_rgb_to_lab(np.zeros(3), colourspace, white_point)
    np.testing.assert_allclose(black, np.zeros(3), rtol=0.0, atol=0.0)


def test_linear_rgb_to_lab_differs_between_srgb_and_display_p3() -> None:
    """Wider P3 primaries give a more saturated Lab for the same linear RGB."""
    neutral = np.array([0.2, 0.2, 0.2])
    np.testing.assert_allclose(
        linear_rgb_to_lab(neutral, "sRGB", _D65),
        linear_rgb_to_lab(neutral, "DisplayP3", _D65),
        rtol=0.0,
        atol=1e-12,
    )

    saturated = np.array([1.0, 0.0, 0.0])
    srgb_lab = linear_rgb_to_lab(saturated, "sRGB", _D65)
    p3_lab = linear_rgb_to_lab(saturated, "DisplayP3", _D65)
    srgb_chroma = math.hypot(srgb_lab[1], srgb_lab[2])
    p3_chroma = math.hypot(p3_lab[1], p3_lab[2])
    assert p3_chroma > srgb_chroma + 1.0


def test_linear_rgb_to_lab_preserves_shape_and_returns_float64() -> None:
    """Any leading shape is preserved and the result is float64."""
    reference = linear_rgb_to_lab(np.array([0.1, 0.2, 0.3]), "sRGB", _D65)
    assert reference.shape == (3,)
    assert reference.dtype == np.float64

    samples = linear_rgb_to_lab(np.full((7, 3), 0.5), "sRGB", _D65)
    assert samples.shape == (7, 3)

    images = linear_rgb_to_lab(np.full((2, 4, 5, 3), 0.5), "DisplayP3", _D65)
    assert images.shape == (2, 4, 5, 3)
    assert images.dtype == np.float64
    assert images.flags.c_contiguous


def test_linear_rgb_to_lab_does_not_modify_its_input() -> None:
    """The conversion is a pure function of a float32 input."""
    rgb = np.full((4, 4, 3), 0.5, dtype=np.float32)
    before = rgb.copy()
    linear_rgb_to_lab(rgb, "sRGB", _WARM)
    np.testing.assert_array_equal(rgb, before)
    assert rgb.dtype == np.float32


def test_linear_rgb_to_lab_rejects_malformed_arrays() -> None:
    """Boolean, complex, non-finite, and mis-shaped arrays raise ValueError."""
    with pytest.raises(ValueError):
        linear_rgb_to_lab(np.zeros((4, 2)), "sRGB", _D65)
    with pytest.raises(ValueError):
        linear_rgb_to_lab(np.float64(0.5), "sRGB", _D65)
    with pytest.raises(ValueError):
        linear_rgb_to_lab(np.array([True, False, True]), "sRGB", _D65)
    with pytest.raises(ValueError):
        linear_rgb_to_lab(np.array([1.0 + 1.0j, 0.0, 0.0]), "sRGB", _D65)
    with pytest.raises(ValueError):
        linear_rgb_to_lab(np.array([0.1, math.nan, 0.3]), "sRGB", _D65)


def test_linear_rgb_to_lab_rejects_unknown_colour_spaces() -> None:
    """Only the exact, case-sensitive names sRGB and DisplayP3 are accepted."""
    for colourspace in ("srgb", "SRGB", "displayp3", "Display P3", "P3", ""):
        with pytest.raises(ValueError):
            linear_rgb_to_lab(np.zeros(3), colourspace, _D65)


def test_linear_rgb_to_lab_rejects_invalid_white_points() -> None:
    """A white point must be a physical (x, y) chromaticity."""
    for white_point in (
        (0.3127,),
        (0.3127, 0.3290, 0.1),
        (math.nan, 0.3290),
        (0.3127, math.inf),
        (0.3127, 0.0),
        (0.0, 0.3290),
        (-0.1, 0.3290),
        (0.7, 0.5),
        (0.5, 0.5),
    ):
        with pytest.raises(ValueError):
            linear_rgb_to_lab(np.zeros(3), "sRGB", white_point)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        linear_rgb_to_lab(np.zeros(3), "sRGB", "D65")  # type: ignore[arg-type]


def test_d65_conversion_skips_the_adaptation() -> None:
    """For D65 the adapted XYZ equals the unadapted matrix product."""
    rng = np.random.default_rng(4)
    rgb = rng.uniform(0.0, 1.0, (6, 3))
    adapted = linear_rgb_to_lab(rgb, "sRGB", _D65)
    plain_xyz = rgb @ _RGB_TO_XYZ_SRGB.T
    expected = colour.XYZ_to_Lab(plain_xyz, illuminant=np.asarray(_D65))
    np.testing.assert_allclose(adapted, expected, rtol=0.0, atol=1e-11)


# --------------------------------------------------------------------------- #
# delta_e_patches
# --------------------------------------------------------------------------- #


def test_white_patch_reference() -> None:
    """A display white patch matches a reference Lab of (100, 0, 0)."""
    reference = _reference([[100.0, 0.0, 0.0]])
    layout = _single_patch(reference)
    result = delta_e_patches(np.ones((8, 8, 3), dtype=np.float32), layout)
    assert result.valid
    assert result.per_patch[0].name == "p"
    assert result.per_patch[0].delta_e76 == pytest.approx(0.0, abs=1e-9)
    assert result.per_patch[0].delta_e2000 == pytest.approx(0.0, abs=1e-9)
    assert result.delta_e76 == pytest.approx(0.0, abs=1e-9)
    assert result.delta_e2000 == pytest.approx(0.0, abs=1e-9)


def test_delta_e_patches_reports_each_patch_in_layout_order() -> None:
    """Per-patch results follow the layout patch order and carry their names."""
    reference = _reference(
        [
            [53.4, 0.0, 0.0],
            [53.4, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    patches = (
        PatchROI("dark", 0, 0, 4, 4),
        PatchROI("mid", 4, 0, 4, 4),
        PatchROI("white", 0, 4, 4, 4),
    )
    layout = _layout(patches, reference)
    image = np.full((8, 8, 3), 0.5, dtype=np.float64)
    image[0:4, 0:4] = 0.2
    image[4:8, 0:4] = 1.0

    result = delta_e_patches(image, layout)

    assert tuple(patch.name for patch in result.per_patch) == ("dark", "mid", "white")
    assert result.valid
    assert len(result.per_patch) == 3
    # The mid patch matches its reference level; the dark patch does not.
    assert result.per_patch[1].delta_e76 < result.per_patch[0].delta_e76
    assert result.per_patch[2].delta_e76 == pytest.approx(0.0, abs=1e-9)


def test_delta_e_patches_uses_the_linear_patch_mean() -> None:
    """The patch mean is taken after linearization, not on display values."""
    # Opposite display values average to display 0.5, but the sRGB EOTF is
    # convex, so the linear mean is below EOTF(0.5).
    image = np.empty((1, 2, 3), dtype=np.float64)
    image[0, 0] = (0.1, 0.9, 0.2)
    image[0, 1] = (0.9, 0.1, 0.8)
    reference = _reference([[50.0, 0.0, 0.0]])
    layout = _layout((PatchROI("p", 0, 0, 2, 1),), reference)

    result = delta_e_patches(image, layout)

    linear_patch = _srgb_eotf(image)[0]
    linear_mean = linear_patch.mean(axis=0)
    expected_lab = linear_rgb_to_lab(linear_mean, "sRGB", reference.white_point_xy)
    expected = float(_delta_e76(expected_lab, reference.lab[0]))
    assert result.per_patch[0].delta_e76 == pytest.approx(expected, rel=1e-12)

    # Averaging the display values instead would give a visibly different Lab.
    display_mean_lab = linear_rgb_to_lab(image[0].mean(axis=0), "sRGB", reference.white_point_xy)
    display_mean_delta = float(_delta_e76(display_mean_lab, reference.lab[0]))
    assert abs(display_mean_delta - result.per_patch[0].delta_e76) > 0.5


def test_delta_e_patches_means_are_over_every_valid_patch() -> None:
    """The top-level fields are the arithmetic means of the per-patch values."""
    reference = _reference([[50.0, 0.0, 0.0], [55.0, 2.0, -3.0]])
    layout = _layout(
        (PatchROI("a", 0, 0, 4, 4), PatchROI("b", 4, 0, 4, 4)),
        reference,
    )
    image = np.full((4, 8, 3), 0.5, dtype=np.float64)
    image[:, 4:] = 0.25

    result = delta_e_patches(image, layout)

    values_76 = [patch.delta_e76 for patch in result.per_patch]
    values_00 = [patch.delta_e2000 for patch in result.per_patch]
    assert result.delta_e76 == pytest.approx(float(np.mean(values_76)), rel=1e-15)
    assert result.delta_e2000 == pytest.approx(float(np.mean(values_00)), rel=1e-15)


def test_explicit_reference_overrides_the_layout_reference() -> None:
    """The explicit reference wins even when the layout carries its own."""
    layout_reference = _reference([[50.0, 10.0, -10.0]])
    explicit_reference = _reference([[30.0, 0.0, 0.0]])
    layout = _single_patch(layout_reference)
    image = np.full((8, 8, 3), 0.5, dtype=np.float64)

    from_explicit = delta_e_patches(image, layout, reference=explicit_reference)
    from_layout = delta_e_patches(image, layout)

    measured = _measured_lab(image, explicit_reference, layout.patches[0])
    expected_76 = float(_delta_e76(measured, explicit_reference.lab[0]))
    expected_00 = float(_delta_e2000(measured, explicit_reference.lab[0]))
    assert from_explicit.per_patch[0].delta_e76 == pytest.approx(expected_76, rel=1e-12)
    assert from_explicit.per_patch[0].delta_e2000 == pytest.approx(expected_00, rel=1e-12)
    assert from_explicit.per_patch[0].delta_e76 != pytest.approx(from_layout.per_patch[0].delta_e76)


def test_reference_row_count_must_match_the_patch_count() -> None:
    """A reference with the wrong number of rows is rejected, never truncated."""
    reference = _reference([[50.0, 0.0, 0.0], [60.0, 0.0, 0.0]])
    layout = _single_patch(reference)
    with pytest.raises(ValueError, match="reference"):
        delta_e_patches(np.full((8, 8, 3), 0.5), layout)


def test_missing_reference_non_d65_raises() -> None:
    """There is no implicit D65 fallback: colour metrics need an explicit reference."""
    layout = _single_patch(None)
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="reference"):
        delta_e_patches(image, layout)
    with pytest.raises(ValueError, match="reference"):
        delta_e_patches(image, layout, reference=None)


def test_mono_raises_clear_error() -> None:
    """A monochrome image is not a colour measurement."""
    layout = _single_patch(_reference([[50.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="three-channel RGB"):
        delta_e_patches(np.full((8, 8), 0.5), layout)


def test_wrong_channel_counts_raise_clear_errors() -> None:
    """One, two, four, or higher-dimensional planes are rejected."""
    layout = _single_patch(_reference([[50.0, 0.0, 0.0]]))
    for shape in ((8, 8, 1), (8, 8, 2), (8, 8, 4), (8, 8, 3, 1)):
        with pytest.raises(ValueError, match="three-channel RGB"):
            delta_e_patches(np.full(shape, 0.5), layout)


def test_out_of_range_display_values_raise_without_clipping() -> None:
    """Display values outside [0, 1] (beyond 1e-6) are rejected, not clipped."""
    layout = _single_patch(_reference([[50.0, 0.0, 0.0]]))
    for value in (1.5, -0.1, 1.0 + 2e-6, -2e-6):
        image = np.full((8, 8, 3), 0.5)
        image[0, 0, 0] = value
        with pytest.raises(ValueError):
            delta_e_patches(image, layout)

    # Values inside the tolerance are accepted as-is and never clipped.
    accepted = np.full((8, 8, 3), 0.5)
    accepted[0, 0, 0] = 1.0 + 5e-7
    accepted[0, 1, 1] = -5e-7
    assert delta_e_patches(accepted, layout).valid


def test_delta_e_patches_rejects_malformed_images() -> None:
    """Non-finite and non-numeric display images are rejected."""
    layout = _single_patch(_reference([[50.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        delta_e_patches(np.full((8, 8, 3), math.nan), layout)
    with pytest.raises(ValueError):
        delta_e_patches(np.full((8, 8, 3), math.inf), layout)
    with pytest.raises(ValueError):
        delta_e_patches(np.full((8, 8, 3), True), layout)
    with pytest.raises(ValueError):
        delta_e_patches(np.full((8, 8, 3), 0.5 + 0.0j), layout)


def test_roi_out_of_bounds_raises_from_extract_patch() -> None:
    """A patch outside the image is a clear error, not a truncated region."""
    layout = _layout((PatchROI("p", 4, 4, 8, 8),), _reference([[50.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="lies outside image shape"):
        delta_e_patches(np.full((8, 8, 3), 0.5), layout)


def test_delta_e_patches_is_deterministic_and_does_not_modify_the_image() -> None:
    """Repeated measurement is bit-identical and leaves the input untouched."""
    rng = np.random.default_rng(5)
    image = rng.uniform(0.05, 0.95, (8, 8, 3)).astype(np.float32)
    before = image.copy()
    layout = _single_patch(_reference([[50.0, 10.0, -10.0]]))

    first = delta_e_patches(image, layout)
    second = delta_e_patches(image, layout)

    np.testing.assert_array_equal(image, before)
    assert image.dtype == np.float32
    assert first == second
    assert first.per_patch[0].delta_e76 == second.per_patch[0].delta_e76


def test_delta_e_patches_accepts_display_p3() -> None:
    """The DisplayP3 output space changes the measured Lab and therefore delta-E."""
    reference = _reference([[50.0, 0.0, 0.0]])
    layout = _single_patch(reference)
    image = np.zeros((8, 8, 3), dtype=np.float64)
    image[..., 0] = 1.0
    image[..., 1] = 0.2
    image[..., 2] = 0.1

    srgb = delta_e_patches(image, layout, color_space="sRGB")
    p3 = delta_e_patches(image, layout, color_space="DisplayP3")

    assert srgb.per_patch[0].delta_e76 != pytest.approx(p3.per_patch[0].delta_e76)
    with pytest.raises(ValueError):
        delta_e_patches(image, layout, color_space="displayp3")


# --------------------------------------------------------------------------- #
# Result value objects
# --------------------------------------------------------------------------- #


def test_patch_colour_result_validates_its_fields() -> None:
    """Names, finite-or-NaN deltas, and the valid flag are checked."""
    assert PatchColorResult(" p ", 1.0, 2.0, True).name == "p"
    with pytest.raises(ValueError):
        PatchColorResult("", 1.0, 2.0, True)
    with pytest.raises(ValueError):
        PatchColorResult("p", math.inf, 2.0, True)
    with pytest.raises(ValueError):
        PatchColorResult("p", 1.0, -math.inf, False)
    with pytest.raises(ValueError):
        PatchColorResult("p", math.nan, math.nan, True)
    with pytest.raises(ValueError):
        PatchColorResult("p", 1.0, math.nan, True)
    # An invalid patch may carry NaN deltas.
    invalid = PatchColorResult("p", math.nan, math.nan, False)
    assert not invalid.valid


def test_colour_result_validates_consistency_with_its_patches() -> None:
    """Top-level means and the valid flag must agree with the per-patch rows."""
    patch = PatchColorResult("p", 1.0, 3.0, True)
    result = ColorResult(1.0, 3.0, (patch,), True)
    assert result.per_patch == (patch,)
    assert result.valid

    with pytest.raises(ValueError, match="mean"):
        ColorResult(2.0, 3.0, (patch,), True)
    with pytest.raises(ValueError, match="valid"):
        ColorResult(1.0, 3.0, (patch,), False)
    with pytest.raises(ValueError, match="valid"):
        ColorResult(math.nan, math.nan, (patch,), True)
    with pytest.raises(ValueError):
        ColorResult(1.0, 3.0, (object(),), True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ColorResult(1.0, 3.0, 5, True)  # type: ignore[arg-type]

    # All-invalid and empty tuples both have no defined mean.
    invalid = PatchColorResult("p", math.nan, math.nan, False)
    empty = ColorResult(math.nan, math.nan, (invalid,), False)
    assert not empty.valid
    assert math.isnan(empty.delta_e76)
    no_patches = ColorResult(math.nan, math.nan, (), False)
    assert not no_patches.valid


def test_colour_result_coerces_a_patch_sequence_to_a_tuple() -> None:
    """A list of patch results is accepted and frozen into a tuple."""
    patch = PatchColorResult("p", 1.0, 2.0, True)
    result = ColorResult(1.0, 2.0, [patch], True)  # type: ignore[arg-type]
    assert isinstance(result.per_patch, tuple)


def test_colour_results_are_frozen() -> None:
    """Result value objects cannot be edited after construction."""
    patch = PatchColorResult("p", 1.0, 2.0, True)
    result = ColorResult(1.0, 2.0, (patch,), True)
    with pytest.raises(FrozenInstanceError):
        patch.delta_e76 = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.valid = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #


def test_metrics_color_exports_the_documented_names() -> None:
    """Only the public metric API is exported."""
    import camsim.metrics.color as module

    assert set(module.__all__) == {
        "ColorResult",
        "PatchColorResult",
        "delta_e_patches",
        "linear_rgb_to_lab",
    }


def test_metrics_color_does_not_import_simulator_modules() -> None:
    """The metrics layer stays independent of the simulator packages."""
    code = "\n".join(
        [
            "import sys",
            "import camsim.metrics.color",
            "from camsim.metrics.color import (",
            "    ColorResult, PatchColorResult, delta_e_patches, linear_rgb_to_lab,",
            ")",
            "for name in (",
            "    'camsim.sensor', 'camsim.isp', 'camsim.scene', 'camsim.optics',",
            "):",
            "    assert name not in sys.modules, name",
        ]
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
