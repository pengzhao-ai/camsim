"""Tests for the sRGB and Display P3 primaries and linear conversions.

The output colour space fixes what the three encoded channels mean. sRGB and
Display P3 share the D65 white point and the sRGB transfer function but use
different primaries: P3 is the wider digital-cinema gamut, with a red primary
at (0.680, 0.320) instead of (0.640, 0.330) and a green at (0.265, 0.690)
instead of (0.300, 0.600). Because both spaces are D65-referenced, converting
between them is a pure 3x3 matrix with no chromatic adaptation::

    XYZ   = RGB_TO_XYZ_SRGB @ rgb_srgb
    rgb_p3 = XYZ_TO_RGB_P3 @ XYZ
    SRGB_TO_P3 = XYZ_TO_RGB_P3 @ RGB_TO_XYZ_SRGB

Each RGB-to-XYZ matrix is derived from the chromaticities alone by scaling the
primary XYZ columns so that RGB (1, 1, 1) maps to the D65 stimulus. That
normalisation is what keeps the neutral axis neutral; a matrix without it
tints every grey.

The tests pin the derivation against colour-science (configured to derive its
matrices from the same chromaticities), the white-point and row-sum
invariants, the pure-linear round trip, shape/dtype/contiguity/immutability
contracts, and validation of malformed input. Failure artefacts guarded here:
a wrong primary set, a missing white-point normalisation (a colour cast on
neutrals), an unnecessary chromatic-adaptation step between two D65 spaces,
accidental clipping of out-of-gamut linear values, and silently accepting
NaNs, booleans, complex values, or a mis-shaped channel axis.
"""

from __future__ import annotations

import colour
import numpy as np
import pytest
from colour.models import RGB_COLOURSPACES
from numpy.typing import NDArray

from camsim.isp.colorspace import (
    P3_TO_SRGB,
    RGB_TO_XYZ_P3,
    RGB_TO_XYZ_SRGB,
    SRGB_TO_P3,
    XYZ_TO_RGB_P3,
    XYZ_TO_RGB_SRGB,
    linear_p3_to_srgb,
    linear_srgb_to_p3,
)

_MATRICES = (
    RGB_TO_XYZ_SRGB,
    XYZ_TO_RGB_SRGB,
    RGB_TO_XYZ_P3,
    XYZ_TO_RGB_P3,
    SRGB_TO_P3,
    P3_TO_SRGB,
)


def _derived_space(name: str) -> colour.RGB_Colourspace:
    """Return a colour-science space with matrices derived from chromaticities.

    colour-science ships the official rounded 4-decimal sRGB matrix, which
    differs from the chromaticity-derived matrix in the fourth decimal. The
    derivation here follows the chromaticities exactly, so the reference is
    colour's own derivation rather than its stored sRGB table.
    """
    space = RGB_COLOURSPACES[name].copy()
    space.use_derived_matrix_RGB_to_XYZ = True
    space.use_derived_matrix_XYZ_to_RGB = True
    return space


def _linear_image(seed: int, shape: tuple[int, int] = (8, 8)) -> NDArray[np.float64]:
    """Return a reproducible linear image spanning in- and out-of-gamut values."""
    return np.random.default_rng(seed).uniform(-0.25, 1.25, (*shape, 3))


def test_p3_matrices() -> None:
    """Matrices and conversions match colour-science's derived matrices."""
    srgb = _derived_space("sRGB")
    p3 = _derived_space("Display P3")

    np.testing.assert_allclose(RGB_TO_XYZ_SRGB, srgb.matrix_RGB_to_XYZ, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(XYZ_TO_RGB_SRGB, srgb.matrix_XYZ_to_RGB, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(RGB_TO_XYZ_P3, p3.matrix_RGB_to_XYZ, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(XYZ_TO_RGB_P3, p3.matrix_XYZ_to_RGB, rtol=0.0, atol=1e-12)

    # The cross-space matrices are derived, not independently tabulated.
    np.testing.assert_allclose(SRGB_TO_P3, XYZ_TO_RGB_P3 @ RGB_TO_XYZ_SRGB, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(P3_TO_SRGB, np.linalg.inv(SRGB_TO_P3), rtol=0.0, atol=1e-15)

    rgb = _linear_image(0, shape=(8, 8)).reshape(-1, 3)
    reference = colour.RGB_to_RGB(rgb, srgb, p3, chromatic_adaptation_transform=None)
    np.testing.assert_allclose(rgb @ SRGB_TO_P3.T, reference, rtol=0.0, atol=1e-12)
    reference_back = colour.RGB_to_RGB(reference, p3, srgb, chromatic_adaptation_transform=None)
    np.testing.assert_allclose(reference @ P3_TO_SRGB.T, reference_back, rtol=0.0, atol=1e-12)


def test_matrices_agree_with_the_official_srgb_table() -> None:
    """The derivation agrees with colour's stored (rounded) official matrices.

    colour-science stores sRGB's published 4-decimal matrix while Display P3's
    stored matrix is already full precision, so the cross-space conversion
    differs by about 1e-4 -- far below one 8-bit display LSB and expected,
    because our matrices are derived from chromaticities as the design
    requires.
    """
    rgb = _linear_image(1).reshape(-1, 3)
    expected = colour.RGB_to_RGB(
        rgb,
        RGB_COLOURSPACES["sRGB"],
        RGB_COLOURSPACES["Display P3"],
        chromatic_adaptation_transform=None,
    )
    np.testing.assert_allclose(rgb @ SRGB_TO_P3.T, expected, rtol=0.0, atol=5e-4)


def test_matrices_map_white_to_d65_and_preserve_neutrals() -> None:
    """Both matrices map RGB (1, 1, 1) to the shared D65 white stimulus."""
    d65 = colour.xy_to_XYZ(RGB_COLOURSPACES["sRGB"].whitepoint)
    np.testing.assert_allclose(d65, colour.xy_to_XYZ(RGB_COLOURSPACES["Display P3"].whitepoint))
    np.testing.assert_allclose(RGB_TO_XYZ_SRGB @ np.ones(3), d65, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(RGB_TO_XYZ_P3 @ np.ones(3), d65, rtol=0.0, atol=1e-12)

    # A neutral input therefore stays neutral through either cross-space
    # matrix: every row sums to one.
    np.testing.assert_allclose(SRGB_TO_P3.sum(axis=1), np.ones(3), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(P3_TO_SRGB.sum(axis=1), np.ones(3), rtol=0.0, atol=1e-15)


def test_matrices_reproduce_the_primaries() -> None:
    """Column ``i`` of RGB-to-XYZ is the XYZ of the space's ``i``-th primary."""
    for matrix, name in ((RGB_TO_XYZ_SRGB, "sRGB"), (RGB_TO_XYZ_P3, "Display P3")):
        space = RGB_COLOURSPACES[name]
        for index, primary in enumerate(space.primaries):
            np.testing.assert_allclose(
                colour.XYZ_to_xy(matrix[:, index]), primary, rtol=0.0, atol=1e-12
            )


def test_linear_conversions_match_colour_science() -> None:
    """Both public conversions match colour's linear (no CAT) RGB-to-RGB."""
    srgb = _derived_space("sRGB")
    p3 = _derived_space("Display P3")
    rgb = _linear_image(2)

    expected = colour.RGB_to_RGB(rgb, srgb, p3, chromatic_adaptation_transform=None)
    np.testing.assert_allclose(linear_srgb_to_p3(rgb), expected, rtol=0.0, atol=1e-6)

    expected_back = colour.RGB_to_RGB(rgb, p3, srgb, chromatic_adaptation_transform=None)
    np.testing.assert_allclose(linear_p3_to_srgb(rgb), expected_back, rtol=0.0, atol=1e-6)


def test_linear_conversions_round_trip() -> None:
    """sRGB -> P3 -> sRGB recovers the input to float32 round-off."""
    rgb = _linear_image(3)
    converted = linear_srgb_to_p3(rgb)
    np.testing.assert_allclose(linear_p3_to_srgb(converted), rgb, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(linear_p3_to_srgb(linear_srgb_to_p3(rgb)), rgb, atol=1e-6)


def test_linear_conversions_are_linear_and_unclipped() -> None:
    """The conversions are linear maps and never clamp out-of-gamut values."""
    rng = np.random.default_rng(4)
    first = rng.normal(0.0, 0.5, (4, 4, 3))
    second = rng.normal(0.0, 0.5, (4, 4, 3))
    alpha, beta = 0.3, -0.7
    combined = linear_srgb_to_p3(alpha * first + beta * second)
    expected = alpha * linear_srgb_to_p3(first).astype(np.float64) + beta * linear_srgb_to_p3(
        second
    ).astype(np.float64)
    np.testing.assert_allclose(combined, expected, rtol=0.0, atol=1e-6)

    out_of_gamut = linear_srgb_to_p3(np.array([[[2.0, -0.5, 0.25]]]))
    assert out_of_gamut.max() > 1.0
    assert out_of_gamut.min() < 0.0


def test_linear_conversions_are_scale_equivariant() -> None:
    """A linear gain in sRGB is the same gain in linear P3."""
    rgb = _linear_image(5, shape=(4, 4))
    np.testing.assert_allclose(
        linear_srgb_to_p3(3.0 * rgb), 3.0 * linear_srgb_to_p3(rgb).astype(np.float64), atol=1e-6
    )


@pytest.mark.parametrize(
    ("function", "shape"),
    [
        (linear_srgb_to_p3, (3,)),
        (linear_srgb_to_p3, (5, 3)),
        (linear_srgb_to_p3, (4, 6, 3)),
        (linear_srgb_to_p3, (2, 3, 4, 3)),
        (linear_p3_to_srgb, (3,)),
        (linear_p3_to_srgb, (5, 3)),
        (linear_p3_to_srgb, (4, 6, 3)),
        (linear_p3_to_srgb, (2, 3, 4, 3)),
    ],
)
def test_conversions_preserve_shape_and_dtype(function, shape: tuple[int, ...]) -> None:
    """Any leading shape with a trailing axis of 3 is accepted and preserved."""
    values = np.random.default_rng(6).random(shape)
    output = function(values)
    assert output.shape == shape
    assert output.dtype == np.float32
    assert output.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(output, values)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 2)),
        np.zeros((2, 2, 4)),
        np.zeros((2, 0)),
        np.array(0.5),
    ],
)
def test_conversions_reject_wrong_shapes(bad: NDArray[np.float64]) -> None:
    """A missing or non-three trailing axis is rejected, not guessed at."""
    for function in (linear_srgb_to_p3, linear_p3_to_srgb):
        with pytest.raises(ValueError, match="trailing axis of 3"):
            function(bad)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_conversions_reject_non_finite_values(value: float) -> None:
    """Non-finite input cannot be a linear intensity and is rejected."""
    image = np.zeros((2, 2, 3))
    image[0, 0, 1] = value
    for function in (linear_srgb_to_p3, linear_p3_to_srgb):
        with pytest.raises(ValueError, match="finite"):
            function(image)


@pytest.mark.parametrize("dtype", [np.bool_, np.complex128])
def test_conversions_reject_bool_and_complex(dtype: np.dtype) -> None:
    """Booleans and complex values are not linear RGB."""
    image = np.zeros((2, 2, 3), dtype=dtype)
    for function in (linear_srgb_to_p3, linear_p3_to_srgb):
        with pytest.raises(ValueError, match="real numeric"):
            function(image)


def test_conversions_reject_values_that_overflow_float32() -> None:
    """A finite float64 value that cannot round to float32 raises."""
    image = np.full((1, 1, 3), 1e300)
    for function in (linear_srgb_to_p3, linear_p3_to_srgb):
        with pytest.raises(ValueError, match="float32"):
            function(image)


def test_conversions_do_not_mutate_input() -> None:
    """The input buffer is untouched and never aliased by the output."""
    image = _linear_image(7)
    before = image.copy()
    for function in (linear_srgb_to_p3, linear_p3_to_srgb):
        output = function(image)
        np.testing.assert_array_equal(image, before)
        assert not np.shares_memory(output, image)


def test_conversions_are_deterministic() -> None:
    """Identical input gives bit-identical output (no hidden state)."""
    image = _linear_image(8)
    np.testing.assert_array_equal(linear_srgb_to_p3(image), linear_srgb_to_p3(image))
    np.testing.assert_array_equal(linear_p3_to_srgb(image), linear_p3_to_srgb(image))


def test_matrices_are_read_only_float64_constants() -> None:
    """Shared matrix constants are float64 and cannot be mutated in place."""
    for matrix in _MATRICES:
        assert matrix.dtype == np.float64
        assert not matrix.flags["WRITEABLE"]
        with pytest.raises(ValueError):
            matrix[0, 0] = 0.0
