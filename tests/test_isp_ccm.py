"""Tests for the colour-correction-matrix stage.

A sensor's colour filters overlap spectrally, so RAW R, G, B are not a
colorimetric basis: each channel sees some light from the other primaries and
saturation is low. The ISP corrects this with a 3x3 linear mixing matrix fitted
on reference patches::

    target_rgb ~= camera_rgb @ M.T

Each row of ``M`` is an independent ridge-regularised least-squares problem with
the identity as the prior. That prior is not cosmetic: when a patch set weakly
excites one colour direction, sensor noise in that direction is amplified into
large fitted coefficients, which appear as chroma noise at low light. A positive
ridge shrinks those weak directions toward "do nothing" at the cost of a small
bias on the calibration patches. When ``row_sum`` is on, the fit also imposes
``sum_j M[i, j] == 1`` exactly, so a neutral patch stays neutral.

The tests pin the exact reference solves, the row-sum constraint at machine
precision, unconstrained recovery, ridge shrinkage toward identity, the
single-sample and image-shaped application paths, float32 output and
contiguity, the output clamp at zero (the stage's only non-linearity), the
CIEDE2000 diagnostic, and the validation of malformed samples, matrices, ridge,
and constraints. Failure artefacts guarded here: applying the matrix with the
wrong convention (``M @ c`` instead of ``c @ M.T``), a non-row-summed matrix
shifting neutrals, an unregularised singular fit returning huge coefficients,
silently regularising a degenerate problem, and returning or mutating the input
buffer.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.typing import NDArray

from camsim.config.schema import CCMConfig as SchemaCCMConfig
from camsim.isp.ccm import (
    CCMConfig,
    CCMResult,
    apply_ccm,
    evaluate_ccm,
    fit_ccm,
)


def _target_patches(seed: int, count: int = 24) -> NDArray[np.float64]:
    """Return reproducible target RGB patches in ``[0, 1)``."""
    return np.random.default_rng(seed).random((count, 3))


def test_ccm_recovers_known_matrix():
    rng = np.random.default_rng(0)
    target = rng.random((24, 3))
    true = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    camera = target @ np.linalg.inv(true).T
    fitted = fit_ccm(camera, target, ridge=0.0)
    np.testing.assert_allclose(fitted, true, atol=1e-6)


def test_ccm_rows_sum_to_one():
    rng = np.random.default_rng(1)
    target = rng.random((24, 3))
    camera = target @ np.array([[1.1, -0.2, 0.1], [-0.1, 1.2, -0.1], [0.0, -0.1, 1.1]]).T
    fitted = fit_ccm(camera, target, ridge=1e-4, row_sum=True)
    np.testing.assert_allclose(fitted.sum(axis=1), 1.0, atol=1e-6)


def test_ccm_config_is_the_schema_model() -> None:
    assert CCMConfig is SchemaCCMConfig


def test_ccm_rows_sum_to_one_at_machine_precision() -> None:
    rng = np.random.default_rng(2)
    target = rng.random((32, 3))
    camera = target @ np.array([[1.05, -0.1, 0.05], [-0.2, 1.3, -0.1], [0.1, -0.25, 1.15]]).T
    for ridge in (0.0, 1e-4, 1e-2):
        fitted = fit_ccm(camera, target, ridge=ridge, row_sum=True)
        # The KKT system imposes the constraint exactly, not as a penalty; a
        # 1e-12 tolerance is far tighter than any penalty approximation reaches.
        np.testing.assert_allclose(fitted.sum(axis=1), np.ones(3), rtol=0.0, atol=1e-12)


def test_ccm_recovers_non_identity_matrix() -> None:
    rng = np.random.default_rng(5)
    target = rng.random((48, 3))
    true = np.array([[0.7, 0.25, 0.05], [-0.15, 0.95, 0.2], [0.05, -0.1, 1.05]])
    camera = target @ np.linalg.inv(true).T
    fitted = fit_ccm(camera, target, ridge=0.0)
    np.testing.assert_allclose(fitted, true, rtol=0.0, atol=1e-9)


def test_ccm_accepts_negative_camera_samples() -> None:
    # Pre-CCM linear data can dip below zero after black-level subtraction, so
    # the fit must not reject (or silently clip) negative camera samples. A
    # strong mixing matrix with row sums of one drives the camera values of
    # some patches below zero; its inverse is the matrix to recover.
    rng = np.random.default_rng(6)
    target = rng.random((48, 3))
    mixing = np.array([[1.6, -0.9, 0.3], [-0.4, 1.5, -0.1], [0.2, -0.7, 1.5]])
    true = np.linalg.inv(mixing)
    camera = target @ mixing.T
    assert camera.min() < 0.0
    fitted = fit_ccm(camera, target, ridge=0.0)
    np.testing.assert_allclose(fitted, true, rtol=0.0, atol=1e-9)


def test_ccm_without_row_sum_recovers_unconstrained_matrix() -> None:
    rng = np.random.default_rng(3)
    target = rng.random((32, 3))
    # Rows sum to 1.3, 1.05, and 1.1, so an unconstrained fit must not force them
    # to one.
    true = np.array([[0.8, 0.2, 0.3], [0.1, 1.1, -0.15], [0.05, 0.15, 0.9]])
    camera = target @ np.linalg.inv(true).T
    fitted = fit_ccm(camera, target, ridge=0.0, row_sum=False)
    np.testing.assert_allclose(fitted, true, rtol=0.0, atol=1e-9)
    assert not np.allclose(fitted.sum(axis=1), 1.0)


def test_ridge_reduces_noise_amplification() -> None:
    # A patch set whose blue direction is almost a blend of red and green is
    # ill-conditioned: with no ridge, sensor noise is fitted into large
    # coefficients far from the identity. The ridge prior pulls the solution
    # back toward "do nothing" while the row sums stay exact.
    rng = np.random.default_rng(0)
    target = rng.random((96, 3))
    target[:, 2] = 0.5 * target[:, 0] + 0.5 * target[:, 1] + 0.002 * rng.standard_normal(96)
    camera = target + 0.005 * rng.standard_normal((96, 3))
    identity = np.eye(3)

    unregularized = fit_ccm(camera, target, ridge=0.0, row_sum=True)
    regularized = fit_ccm(camera, target, ridge=1e-2, row_sum=True)

    unregularized_error = np.linalg.norm(unregularized - identity)
    regularized_error = np.linalg.norm(regularized - identity)
    assert regularized_error < 0.5 * unregularized_error
    np.testing.assert_allclose(unregularized.sum(axis=1), np.ones(3), atol=1e-12)
    np.testing.assert_allclose(regularized.sum(axis=1), np.ones(3), atol=1e-12)


def test_fit_ccm_is_deterministic() -> None:
    rng = np.random.default_rng(8)
    target = rng.random((40, 3))
    camera = target @ np.array([[1.1, -0.2, 0.1], [-0.1, 1.2, -0.1], [0.0, -0.1, 1.1]]).T
    first = fit_ccm(camera, target)
    second = fit_ccm(camera, target)
    np.testing.assert_array_equal(first, second)


def test_fit_ccm_returns_new_float64_array_without_mutating_inputs() -> None:
    rng = np.random.default_rng(9)
    target = rng.random((24, 3))
    camera = target @ np.array([[1.1, -0.2, 0.1], [-0.1, 1.2, -0.1], [0.0, -0.1, 1.1]]).T
    camera_before = camera.copy()
    target_before = target.copy()

    fitted = fit_ccm(camera, target)

    assert fitted.shape == (3, 3)
    assert fitted.dtype == np.float64
    assert fitted.flags.c_contiguous
    assert not np.shares_memory(fitted, camera)
    np.testing.assert_array_equal(camera, camera_before)
    np.testing.assert_array_equal(target, target_before)


def test_fit_ccm_rejects_wrong_sample_shapes() -> None:
    good = _target_patches(10)
    for bad, message in (
        (np.zeros(3), "2-D"),
        (np.zeros((24, 4)), "three channels"),
        (np.zeros((24, 3, 1)), "2-D"),
    ):
        with pytest.raises(ValueError, match=message):
            fit_ccm(bad, good)
        with pytest.raises(ValueError, match=message):
            fit_ccm(good, bad)


def test_fit_ccm_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least three samples"):
        fit_ccm(np.zeros((2, 3)), np.zeros((2, 3)))


def test_fit_ccm_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same"):
        fit_ccm(np.zeros((24, 3)), np.zeros((25, 3)))


def test_fit_ccm_rejects_non_finite_samples() -> None:
    good = _target_patches(12)
    for bad_value in (np.nan, np.inf):
        bad = good.copy()
        bad[0, 1] = bad_value
        with pytest.raises(ValueError, match="finite"):
            fit_ccm(bad, good)
        with pytest.raises(ValueError, match="finite"):
            fit_ccm(good, bad)


def test_fit_ccm_rejects_bool_and_complex_samples() -> None:
    good = _target_patches(13)
    with pytest.raises(ValueError, match="real numeric"):
        fit_ccm(np.ones((24, 3), dtype=bool), good)
    with pytest.raises(ValueError, match="real numeric"):
        fit_ccm(good, np.ones((24, 3), dtype=np.complex128))


def test_fit_ccm_rejects_malformed_array_likes() -> None:
    with pytest.raises(ValueError, match="2-D"):
        fit_ccm([[0.0, 0.0, 0.0], [1.0, 1.0]], np.zeros((2, 3)))


def test_fit_ccm_rejects_normal_equation_overflow() -> None:
    # Finite samples can still square beyond float64 range; the fit reports
    # that rather than returning infinite coefficients.
    huge = np.full((4, 3), 1e200)
    with pytest.raises(ValueError, match="overflow"):
        fit_ccm(huge, huge)


def test_fit_ccm_rejects_invalid_ridge() -> None:
    good = _target_patches(14)
    for bad_ridge in (-1e-4, np.nan, np.inf, True):
        with pytest.raises(ValueError, match="ridge"):
            fit_ccm(good, good, ridge=bad_ridge)


def test_fit_ccm_rejects_non_bool_row_sum() -> None:
    good = _target_patches(15)
    for bad_row_sum in (1, 0, 1.0, "yes", None):
        with pytest.raises(ValueError, match="row_sum"):
            fit_ccm(good, good, row_sum=bad_row_sum)


def test_fit_ccm_zero_ridge_without_row_sum_is_ordinary_least_squares() -> None:
    # No ridge and no constraint is plain OLS: it is well-posed whenever the
    # camera samples span three directions, and it recovers an exact known
    # matrix without any regularisation bias.
    rng = np.random.default_rng(16)
    target = rng.random((24, 3))
    true = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    camera = target @ np.linalg.inv(true).T
    fitted = fit_ccm(camera, target, ridge=0.0, row_sum=False)
    np.testing.assert_allclose(fitted, true, rtol=0.0, atol=1e-9)


def test_fit_ccm_rejects_singular_unconstrained_system() -> None:
    # Every camera sample is the same colour, so the unconstrained normal
    # equations have rank one; the caller is told to use a ridge instead of
    # receiving NaNs or silently regularised coefficients.
    camera = np.ones((24, 3))
    target = _target_patches(16)
    with pytest.raises(ValueError, match="singular"):
        fit_ccm(camera, target, ridge=0.0, row_sum=False)


def test_fit_ccm_rejects_singular_system_without_ridge() -> None:
    # Every camera sample is the same colour, so the normal equations have rank
    # one and cannot determine three coefficients; the caller is told to use a
    # ridge instead of receiving NaNs or silently regularised coefficients.
    camera = np.ones((24, 3))
    target = _target_patches(17)
    with pytest.raises(ValueError, match="singular"):
        fit_ccm(camera, target, ridge=0.0)


def test_apply_ccm_identity() -> None:
    rng = np.random.default_rng(18)
    rgb = rng.random((16, 3)).astype(np.float32)
    out = apply_ccm(rgb, np.eye(3))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, rgb)


def test_apply_ccm_matches_matrix_product() -> None:
    rng = np.random.default_rng(19)
    matrix = np.array([[0.9, 0.15, -0.05], [0.0, 1.1, -0.1], [0.05, -0.2, 1.15]])
    rgb = rng.random((10, 3)).astype(np.float32)
    expected = np.maximum(rgb.astype(np.float64) @ matrix.T, 0.0).astype(np.float32)
    np.testing.assert_allclose(apply_ccm(rgb, matrix), expected, rtol=0.0, atol=1e-6)


def test_apply_ccm_shapes() -> None:
    rng = np.random.default_rng(20)
    matrix = np.array([[0.9, 0.15, -0.05], [0.0, 1.1, -0.1], [0.05, -0.2, 1.15]])
    for shape in ((3,), (7, 3), (4, 5, 3)):
        rgb = rng.random(shape).astype(np.float32)
        out = apply_ccm(rgb, matrix)
        assert out.shape == shape
        assert out.dtype == np.float32
        assert out.flags.c_contiguous
        expected = np.maximum(rgb.astype(np.float64) @ matrix.T, 0.0).astype(np.float32)
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-6)


def test_apply_ccm_returns_contiguous_copy_for_non_contiguous_input() -> None:
    rng = np.random.default_rng(21)
    base = rng.random((8, 6, 3)).astype(np.float32)
    view = base[::2, ::2, :]
    assert not view.flags.c_contiguous
    out = apply_ccm(view, np.eye(3))
    assert out.flags.c_contiguous
    assert not np.shares_memory(out, view)
    np.testing.assert_array_equal(out, view)


def test_negative_values_clamped() -> None:
    rgb = np.array([[1.0, 1.0, 1.0], [0.5, 0.25, 0.75]], dtype=np.float32)
    matrix = np.array([[1.0, -2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = apply_ccm(rgb, matrix)
    np.testing.assert_allclose(
        out,
        np.array([[0.0, 1.0, 1.0], [0.0, 0.25, 0.75]], dtype=np.float32),
    )
    assert np.all(out >= 0.0)


def test_apply_ccm_accepts_negative_input() -> None:
    # Negative camera values are legal before the matrix; only the output is
    # clamped to zero.
    rgb = np.array([[-0.2, 0.5, -1.0]], dtype=np.float32)
    out = apply_ccm(rgb, np.eye(3))
    np.testing.assert_array_equal(out, np.array([[0.0, 0.5, 0.0]], dtype=np.float32))


def test_apply_ccm_does_not_mutate_inputs() -> None:
    rng = np.random.default_rng(22)
    rgb = (rng.random((6, 3)) - 0.25).astype(np.float32)
    matrix = np.array([[0.9, 0.15, -0.05], [0.0, 1.1, -0.1], [0.05, -0.2, 1.15]])
    rgb_before = rgb.copy()
    matrix_before = matrix.copy()
    out = apply_ccm(rgb, matrix)
    assert not np.shares_memory(out, rgb)
    assert not np.shares_memory(out, matrix)
    np.testing.assert_array_equal(rgb, rgb_before)
    np.testing.assert_array_equal(matrix, matrix_before)


def test_apply_ccm_rejects_wrong_shapes() -> None:
    matrix = np.eye(3)
    for bad in (np.zeros(4), np.zeros((2, 2)), np.float64(1.0), np.zeros((2, 3, 4))):
        with pytest.raises(ValueError, match="trailing axis of 3"):
            apply_ccm(bad, matrix)


def test_apply_ccm_rejects_non_finite_values() -> None:
    matrix = np.eye(3)
    for bad_value in (np.nan, np.inf):
        rgb = np.ones((2, 3), dtype=np.float32)
        rgb[0, 0] = bad_value
        with pytest.raises(ValueError, match="finite"):
            apply_ccm(rgb, matrix)


def test_apply_ccm_rejects_bool_and_complex_values() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        apply_ccm(np.ones((2, 3), dtype=bool), np.eye(3))
    with pytest.raises(ValueError, match="real numeric"):
        apply_ccm(np.ones((2, 3), dtype=np.complex128), np.eye(3))


def test_apply_ccm_rejects_float32_overflow() -> None:
    with pytest.raises(ValueError, match="float32"):
        apply_ccm(np.full((2, 3), 1e300), np.eye(3))
    # Finite operands can still overflow the float32 output; that is rejected
    # rather than returning infinity.
    with pytest.raises(ValueError, match="non-finite"):
        apply_ccm(np.full((2, 3), 1e10, dtype=np.float32), np.full((3, 3), 1e30))
    # A huge but finite matrix can overflow the float64 accumulation first.
    with pytest.raises(ValueError, match="non-finite"):
        apply_ccm(np.full((2, 3), 1e38, dtype=np.float32), np.full((3, 3), 1e308))


def test_apply_ccm_rejects_malformed_array_likes() -> None:
    with pytest.raises(ValueError, match="trailing axis of 3"):
        apply_ccm([[0.0, 0.0, 0.0], [1.0, 1.0]], np.eye(3))
    with pytest.raises(ValueError, match="3x3"):
        apply_ccm(np.ones((2, 3), dtype=np.float32), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0]])


def test_apply_ccm_rejects_bad_matrix() -> None:
    rgb = np.ones((2, 3), dtype=np.float32)
    for bad, message in (
        (np.zeros((2, 3)), "3x3"),
        (np.zeros(3), "3x3"),
        (np.ones((3, 3), dtype=bool), "real numeric"),
        (np.ones((3, 3), dtype=np.complex128), "real numeric"),
    ):
        with pytest.raises(ValueError, match=message):
            apply_ccm(rgb, bad)
    bad = np.eye(3)
    bad[1, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        apply_ccm(rgb, bad)


def test_evaluate_ccm_identity_reports_zero_error() -> None:
    rgb = np.random.default_rng(23).random((12, 3)).astype(np.float32)
    result = evaluate_ccm(rgb, rgb, np.eye(3))
    assert isinstance(result, CCMResult)
    assert result.residual == pytest.approx(0.0, abs=1e-12)
    assert result.delta_e_before == pytest.approx(0.0, abs=1e-12)
    assert result.delta_e_after == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_array_equal(result.matrix, np.eye(3))


def test_evaluate_ccm_residual_matches_documented_formula() -> None:
    rng = np.random.default_rng(24)
    camera = rng.random((16, 3))
    target = rng.random((16, 3))
    matrix = np.array([[0.9, 0.05, 0.05], [-0.1, 1.05, 0.05], [0.0, -0.1, 1.1]])
    result = evaluate_ccm(camera, target, matrix)
    expected = np.sqrt(np.mean(np.sum((camera @ matrix.T - target) ** 2, axis=1)))
    assert result.residual == pytest.approx(float(expected), rel=1e-12, abs=1e-15)


def test_ccm_improves_delta_e() -> None:
    rng = np.random.default_rng(7)
    target = rng.random((24, 3))
    true = np.array([[1.2, -0.25, 0.05], [-0.1, 1.25, -0.15], [0.0, -0.2, 1.2]])
    camera = target @ np.linalg.inv(true).T
    fitted = fit_ccm(camera, target, ridge=1e-6)
    result = evaluate_ccm(camera, target, fitted)
    assert result.delta_e_after < result.delta_e_before
    assert result.delta_e_after < 0.01 * result.delta_e_before
    assert result.residual < 1e-5
    np.testing.assert_allclose(result.matrix, fitted)


def test_evaluate_ccm_handles_negative_samples_and_owns_its_matrix() -> None:
    rng = np.random.default_rng(25)
    camera = rng.random((12, 3)) - 0.2
    assert camera.min() < 0.0
    target = rng.random((12, 3))
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = evaluate_ccm(camera, target, matrix)
    # The diagnostic clips negatives for both ΔE values instead of feeding the
    # sRGB transform out-of-range linear values.
    assert np.isfinite(result.delta_e_before)
    assert np.isfinite(result.delta_e_after)
    assert result.delta_e_before >= 0.0
    assert result.delta_e_after >= 0.0
    matrix[0, 0] = 99.0
    assert result.matrix[0, 0] == pytest.approx(1.0)


def test_ccm_result_is_frozen() -> None:
    rgb = _target_patches(26, count=8)
    result = evaluate_ccm(rgb, rgb, np.eye(3))
    with pytest.raises(FrozenInstanceError):
        result.residual = 0.0


def test_evaluate_ccm_rejects_malformed_inputs() -> None:
    good = _target_patches(27)
    matrix = np.eye(3)
    with pytest.raises(ValueError, match="same"):
        evaluate_ccm(np.zeros((24, 3)), np.zeros((25, 3)), matrix)
    with pytest.raises(ValueError, match="at least three samples"):
        evaluate_ccm(np.zeros((2, 3)), np.zeros((2, 3)), matrix)
    with pytest.raises(ValueError, match="3x3"):
        evaluate_ccm(good, good, np.zeros((2, 3)))
