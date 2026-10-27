"""Tests for black-level correction of RAW ADC codes.

An ADC reserves a pedestal of ``black_level_dn`` codes for zero signal so that
read-noise excursions around zero volts are not clipped. The ISP must therefore
subtract that pedestal before any gain, white balance, or colour operation;
otherwise the offset is amplified into a grey cast that white balance turns
into a colour cast. Some sensors also carry a slightly different pedestal on
each CFA phase, which is why the stage accepts a 2x2 tile as well as a scalar.

The stage is linear in the physically meaningful sense: away from the zero
clamp, scaling the *signal above the pedestal* scales the corrected output.
Scaling the raw codes without also scaling the pedestal is not linear, because
a constant offset is affine rather than multiplicative. Tests pin that
invariant, the clamp, the per-CFA pattern order, the float32 output contract,
and validation of the pedestal against the ADC code space.

Failure artefacts guarded here: subtracting the pedestal twice (crushed
shadows), leaving it in place (a grey pedestal after gain), applying gain
before subtraction (an amplified pedestal), putting the 2x2 pattern in the
wrong CFA order (channel-dependent offsets), and a pedestal outside the ADC
code space (a configuration error that would otherwise produce a constant
offset everywhere).
"""

from __future__ import annotations

import numpy as np
import pytest

from camsim.isp.blc import subtract_black_level


def test_blc_subtracts_and_clamps() -> None:
    x = np.array([[64, 100, 1000]], dtype=np.uint16)
    y = subtract_black_level(x, 64)
    np.testing.assert_array_equal(y, [[0, 36, 936]])
    assert y.dtype == np.float32


def test_blc_clamps_values_below_the_pedestal() -> None:
    x = np.array([[0, 63, 64, 65]], dtype=np.uint16)
    y = subtract_black_level(x, 64)
    np.testing.assert_array_equal(y, [[0, 0, 0, 1]])


def test_blc_pattern_uses_the_2x2_cfa_tile() -> None:
    x = np.array(
        [
            [10, 20, 30, 40],
            [30, 40, 50, 60],
            [50, 60, 70, 80],
            [70, 80, 90, 100],
        ],
        dtype=np.uint16,
    )
    # Make one pixel fall below its own per-site pedestal to exercise the clamp.
    x[1, 1] = 2
    pattern = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = subtract_black_level(x, 0, pattern=pattern)
    expected = np.array(
        [
            [9, 18, 29, 38],
            [27, 0, 47, 56],
            [49, 58, 69, 78],
            [67, 76, 87, 96],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(y, expected)


def test_blc_pattern_defines_absolute_levels_and_overrides_the_scalar() -> None:
    x = np.full((4, 4), 100, dtype=np.uint16)
    pattern = np.array([[10, 20], [30, 40]], dtype=np.float64)
    from_zero = subtract_black_level(x, 0, pattern=pattern)
    from_scalar = subtract_black_level(x, 64, pattern=pattern)
    np.testing.assert_array_equal(from_zero, from_scalar)
    np.testing.assert_array_equal(
        from_zero,
        np.array(
            [
                [90, 80, 90, 80],
                [70, 60, 70, 60],
                [90, 80, 90, 80],
                [70, 60, 70, 60],
            ],
            dtype=np.float32,
        ),
    )


def test_blc_is_affine_about_the_pedestal() -> None:
    x = np.array([[100.0, 200.0, 1000.0]], dtype=np.float32)
    pedestal = 64.0
    scale = 2.5
    # Scale the signal above the pedestal and rebuild the raw codes, which is
    # the physically correct linearity statement for an affine offset stage.
    scaled_raw = (scale * (x - pedestal) + pedestal).astype(np.float32)
    np.testing.assert_allclose(
        subtract_black_level(scaled_raw, pedestal),
        scale * subtract_black_level(x, pedestal),
        rtol=1e-6,
    )


def test_blc_with_zero_pedestal_is_purely_linear() -> None:
    x = np.array([[1.0, 7.0, 1000.0]], dtype=np.float32)
    np.testing.assert_allclose(
        subtract_black_level(3.0 * x, 0.0),
        3.0 * subtract_black_level(x, 0.0),
        rtol=1e-6,
    )


def test_blc_zero_pedestal_is_identity() -> None:
    x = np.arange(16, dtype=np.uint16).reshape(4, 4)
    y = subtract_black_level(x, 0)
    np.testing.assert_array_equal(y, x.astype(np.float32))
    assert y.dtype == np.float32


def test_blc_returns_float32_for_float64_input() -> None:
    y = subtract_black_level(np.array([[10.0]], dtype=np.float64), 4.0)
    assert y.dtype == np.float32
    np.testing.assert_array_equal(y, [[6.0]])


def test_blc_does_not_mutate_the_input() -> None:
    x = np.array([[64, 100]], dtype=np.uint16)
    original = x.copy()
    subtract_black_level(x, 64)
    np.testing.assert_array_equal(x, original)


def test_blc_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        subtract_black_level(np.zeros((2, 2, 2), dtype=np.uint16), 64)


def test_blc_rejects_boolean_input() -> None:
    with pytest.raises(ValueError, match="numeric"):
        subtract_black_level(np.ones((2, 2), dtype=bool), 0)


def test_blc_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        subtract_black_level(np.array([[np.nan, 1.0]]), 0)
    with pytest.raises(ValueError, match="finite"):
        subtract_black_level(np.array([[np.inf, 1.0]]), 0)


def test_blc_rejects_invalid_black_levels() -> None:
    x = np.zeros((2, 2), dtype=np.uint16)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="black_level_dn"):
            subtract_black_level(x, bad)
    with pytest.raises(ValueError, match="black_level_dn"):
        subtract_black_level(x, True)  # type: ignore[arg-type]


def test_blc_rejects_invalid_patterns() -> None:
    x = np.zeros((2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="2x2"):
        subtract_black_level(x, 0, pattern=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        subtract_black_level(x, 0, pattern=[[0.0, np.nan], [0.0, 0.0]])
    with pytest.raises(ValueError, match="non-negative"):
        subtract_black_level(x, 0, pattern=[[0.0, -1.0], [0.0, 0.0]])


def test_blc_validates_the_pedestal_against_the_bit_depth() -> None:
    x = np.zeros((2, 2), dtype=np.uint16)
    subtract_black_level(x, 1023, bit_depth=10)
    with pytest.raises(ValueError, match=r"bit_depth|bit depth"):
        subtract_black_level(x, 1024, bit_depth=10)
    with pytest.raises(ValueError, match=r"bit_depth|bit depth"):
        subtract_black_level(x, 64, bit_depth=7)
    with pytest.raises(ValueError, match=r"bit_depth|bit depth"):
        subtract_black_level(x, 0, pattern=[[0.0, 0.0], [0.0, 1024.0]], bit_depth=10)
