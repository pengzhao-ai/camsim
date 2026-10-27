"""Tests for auto white balance on linear RAW mosaics and RGB images.

White balance is a diagonal (von Kries) per-channel gain applied in the linear
domain so that a neutral scene under an arbitrary illuminant renders neutral.
The three estimators tested here make different scene assumptions:

* gray-world: the average scene reflectance is achromatic, so per-channel means
  should be equal;
* white-patch: the brightest sample in each channel came from a white surface;
* shades-of-gray: the Minkowski ``p``-norm of each channel should be equal,
  interpolating between gray-world (``p=1``) and white-patch (``p -> inf``).

For a Bayer mosaic each sample belongs to one spectral band, so statistics must
be collected per CFA colour; mixing R, G, and B sites would bias every estimate.
Gains are normalised to green (``gains[1] == 1``) so white balance cannot change
overall exposure, and the stage never clips because later linear stages expect
scene radiance, not display values.

The tests pin the estimator arithmetic, per-CFA statistics for all four Bayer
layouts, gain normalisation, scale invariance and linearity, the documented
confidence range, mono/disabled no-ops, manual gains, and the validation of
malformed images, CFA patterns, configs, and gains. Failure artefacts guarded
here include mixing CFA colours, applying gains to a demosaiced image as though
it were a mosaic, zero channel statistics (infinite gains), gamma-encoded input,
and clipping large values after gain.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from camsim.config.schema import AWBConfig as SchemaAWBConfig
from camsim.core.frames import CFAPattern
from camsim.isp.awb import AWBConfig, AWBResult, apply_awb, estimate_awb
from camsim.sensor.cfa import mosaic_from_channels


def _rgb(
    red: float,
    green: float,
    blue: float,
    shape: tuple[int, int] = (8, 8),
) -> NDArray[np.float32]:
    """Return a constant per-channel RGB image."""
    image = np.empty((*shape, 3), dtype=np.float32)
    image[..., 0] = red
    image[..., 1] = green
    image[..., 2] = blue
    return image


def _bayer(
    red: float,
    green: float,
    blue: float,
    pattern: CFAPattern,
    shape: tuple[int, int] = (8, 8),
) -> NDArray[np.float32]:
    """Return a constant-plane Bayer mosaic for ``pattern``."""
    planes = {
        "R": np.full(shape, red, dtype=np.float32),
        "G": np.full(shape, green, dtype=np.float32),
        "B": np.full(shape, blue, dtype=np.float32),
    }
    return mosaic_from_channels(planes, pattern)


def test_awb_config_is_the_schema_model() -> None:
    assert AWBConfig is SchemaAWBConfig


def test_gray_world_equalizes_channel_means() -> None:
    img = np.zeros((16, 16, 3), np.float32)
    img[..., 0] = 2.0
    img[..., 1] = 1.0
    img[..., 2] = 0.5
    res = estimate_awb(img, cfa=None, config=AWBConfig(method="gray_world"))
    out = apply_awb(img, res.gains, cfa=None)
    means = out.reshape(-1, 3).mean(axis=0)
    np.testing.assert_allclose(means, means.mean(), rtol=1e-2)
    np.testing.assert_allclose(res.gains, (0.5, 1.0, 2.0), rtol=1e-6)
    assert out.dtype == np.float32


def test_white_patch_uses_brightest_pixel() -> None:
    img = np.ones((8, 8, 3), np.float32)
    img[0, 0] = [4.0, 2.0, 1.0]
    res = estimate_awb(img, cfa=None, config=AWBConfig(method="white_patch"))
    assert res.gains[0] < res.gains[1] < res.gains[2]
    np.testing.assert_allclose(res.gains, (0.5, 1.0, 2.0), rtol=1e-6)


def test_white_patch_uses_per_channel_maxima() -> None:
    # Each channel's maximum occurs at a different pixel, so a single brightest
    # pixel rule would return a different answer than per-channel maxima.
    img = np.full((4, 4, 3), 0.5, np.float32)
    img[0, 0] = [4.0, 0.5, 0.5]
    img[1, 1] = [0.5, 2.0, 0.5]
    img[2, 2] = [0.5, 0.5, 1.0]
    res = estimate_awb(img, cfa=None, config=AWBConfig(method="white_patch"))
    np.testing.assert_allclose(res.gains, (0.5, 1.0, 2.0), rtol=1e-6)


def test_shades_of_gray_p6() -> None:
    rng = np.random.default_rng(1234)
    img = rng.random((8, 8, 3), dtype=np.float32) + 0.05
    img[..., 0] *= 2.0
    img[..., 2] *= 0.5
    res = estimate_awb(img, cfa=None, config=AWBConfig(method="shades_of_gray", p=6.0))
    data = img.astype(np.float64)
    stats = [float(np.mean(data[..., channel] ** 6.0) ** (1.0 / 6.0)) for channel in range(3)]
    expected = (stats[1] / stats[0], 1.0, stats[1] / stats[2])
    np.testing.assert_allclose(res.gains, expected, rtol=1e-6)


def test_shades_of_gray_p1_matches_gray_world() -> None:
    rng = np.random.default_rng(99)
    img = rng.random((8, 8, 3), dtype=np.float32) + 0.1
    gray_world = estimate_awb(img, cfa=None, config=AWBConfig(method="gray_world"))
    shades = estimate_awb(img, cfa=None, config=AWBConfig(method="shades_of_gray", p=1.0))
    np.testing.assert_allclose(shades.gains, gray_world.gains, rtol=1e-9)
    assert shades.illuminant_estimate == pytest.approx(gray_world.illuminant_estimate)


@pytest.mark.parametrize("method", ["gray_world", "white_patch", "shades_of_gray"])
def test_gains_are_normalized_to_green(method: str) -> None:
    img = _rgb(4.0, 1.0, 0.25)
    config = AWBConfig(
        method=method,  # type: ignore[arg-type]
        p=6.0,
    )
    res = estimate_awb(img, cfa=None, config=config)
    assert res.gains[1] == 1.0
    assert res.method == method
    assert isinstance(res, AWBResult)
    assert isinstance(res.gains, tuple)
    assert all(isinstance(value, float) for value in res.gains)


@pytest.mark.parametrize(
    "pattern",
    [CFAPattern.RGGB, CFAPattern.BGGR, CFAPattern.GRBG, CFAPattern.GBRG],
)
def test_bayer_gray_world_uses_per_cfa_statistics(pattern: CFAPattern) -> None:
    raw = _bayer(2.0, 1.0, 0.5, pattern)
    res = estimate_awb(raw, cfa=pattern, config=AWBConfig(method="gray_world"))
    np.testing.assert_allclose(res.gains, (0.5, 1.0, 2.0), rtol=1e-6)
    out = apply_awb(raw, res.gains, cfa=pattern)
    np.testing.assert_allclose(out, np.ones_like(raw), rtol=1e-6)
    assert out.dtype == np.float32


@pytest.mark.parametrize("method", ["gray_world", "white_patch", "shades_of_gray"])
def test_bayer_estimators_never_mix_cfa_colours(method: str) -> None:
    raw = _bayer(2.0, 1.0, 0.5, CFAPattern.RGGB)
    config = AWBConfig(method=method)  # type: ignore[arg-type]
    np.testing.assert_allclose(
        estimate_awb(raw, cfa=CFAPattern.RGGB, config=config).gains,
        (0.5, 1.0, 2.0),
        rtol=1e-6,
    )


def test_awb_accepts_integer_raw_codes() -> None:
    raw = mosaic_from_channels(
        {
            "R": np.full((4, 4), 200, dtype=np.uint16),
            "G": np.full((4, 4), 100, dtype=np.uint16),
            "B": np.full((4, 4), 50, dtype=np.uint16),
        },
        CFAPattern.RGGB,
    )
    res = estimate_awb(raw, cfa=CFAPattern.RGGB, config=AWBConfig())
    np.testing.assert_allclose(res.gains, (0.5, 1.0, 2.0), rtol=1e-6)
    out = apply_awb(raw, res.gains, cfa=CFAPattern.RGGB)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.full((4, 4), 100.0), rtol=1e-6)


def test_mono_awb_is_noop() -> None:
    mono = np.arange(16, dtype=np.float32).reshape(4, 4)
    res = estimate_awb(mono, cfa=CFAPattern.MONO, config=AWBConfig())
    assert res.gains == (1.0, 1.0, 1.0)
    assert res.illuminant_estimate == (1.0, 1.0, 1.0)
    assert res.method == "none"
    assert res.confidence == 0.0
    out = apply_awb(mono, res.gains, cfa=CFAPattern.MONO)
    np.testing.assert_array_equal(out, mono)
    assert out.dtype == np.float32
    assert out is not mono

    # Manual gains cannot colour a sensor that has no colour filters.
    manual = estimate_awb(mono, cfa=CFAPattern.MONO, config=AWBConfig(gains=(2.0, 1.0, 4.0)))
    assert manual.gains == (1.0, 1.0, 1.0)
    assert manual.method == "none"


def test_disabled_awb_is_identity() -> None:
    img = _rgb(4.0, 1.0, 0.25)
    res = estimate_awb(img, cfa=None, config=AWBConfig(enabled=False))
    assert res == AWBResult(
        gains=(1.0, 1.0, 1.0),
        illuminant_estimate=(1.0, 1.0, 1.0),
        method="none",
        confidence=0.0,
    )
    # Disabled takes precedence over manually supplied gains.
    manual = estimate_awb(img, cfa=None, config=AWBConfig(enabled=False, gains=(2.0, 1.0, 4.0)))
    assert manual.gains == (1.0, 1.0, 1.0)


def test_manual_gains_are_used_and_normalized_to_green() -> None:
    img = _rgb(4.0, 2.0, 1.0)
    res = estimate_awb(img, cfa=None, config=AWBConfig(gains=(2.0, 4.0, 8.0)))
    assert res.method == "manual"
    np.testing.assert_allclose(res.gains, (0.5, 1.0, 2.0), rtol=1e-12)
    assert 0.0 <= res.confidence <= 1.0
    out = apply_awb(img, res.gains, cfa=None)
    np.testing.assert_allclose(out, np.full_like(img, 2.0), rtol=1e-6)
    # The implied illuminant is the reciprocal of the gains, mean-normalised.
    reciprocal = 1.0 / np.asarray(res.gains)
    np.testing.assert_allclose(res.illuminant_estimate, reciprocal / reciprocal.mean(), rtol=1e-12)


def test_illuminant_estimate_is_normalized_channel_statistics() -> None:
    img = _rgb(2.0, 1.0, 0.5)
    res = estimate_awb(img, cfa=None, config=AWBConfig(method="gray_world"))
    expected = np.array([2.0, 1.0, 0.5])
    expected /= expected.mean()
    np.testing.assert_allclose(res.illuminant_estimate, expected, rtol=1e-6)
    assert float(np.mean(res.illuminant_estimate)) == pytest.approx(1.0)
    # Gains and the illuminant estimate are reciprocal up to one common scale.
    product = np.asarray(res.gains) * np.asarray(res.illuminant_estimate)
    np.testing.assert_allclose(product, np.full(3, product[0]), rtol=1e-6)


@pytest.mark.parametrize("method", ["gray_world", "white_patch", "shades_of_gray"])
def test_awb_confidence_range(method: str) -> None:
    rng = np.random.default_rng(5)
    img = (rng.random((8, 8, 3), dtype=np.float32) + 0.1) * np.array(
        [3.0, 1.0, 0.3], dtype=np.float32
    )
    res = estimate_awb(
        img,
        cfa=None,
        config=AWBConfig(method=method),  # type: ignore[arg-type]
    )
    assert isinstance(res.confidence, float)
    assert 0.0 <= res.confidence <= 1.0

    balanced = _rgb(2.0, 2.0, 2.0)
    assert estimate_awb(balanced, None, AWBConfig()).confidence == pytest.approx(1.0)
    strong_cast = _rgb(4.0, 1.0, 0.25)
    assert estimate_awb(strong_cast, None, AWBConfig()).confidence < 1.0


def test_awb_preserves_linearity() -> None:
    rng = np.random.default_rng(7)
    rgb = rng.random((8, 8, 3), dtype=np.float32) + 0.1
    gains = (0.5, 1.0, 2.0)
    np.testing.assert_allclose(
        apply_awb(3.0 * rgb, gains, cfa=None),
        3.0 * apply_awb(rgb, gains, cfa=None),
        rtol=1e-6,
    )

    raw = _bayer(2.0, 1.0, 0.5, CFAPattern.RGGB)
    raw[0, 0] += 0.25
    np.testing.assert_allclose(
        apply_awb(3.0 * raw, gains, cfa=CFAPattern.RGGB),
        3.0 * apply_awb(raw, gains, cfa=CFAPattern.RGGB),
        rtol=1e-6,
    )


@pytest.mark.parametrize("method", ["gray_world", "white_patch", "shades_of_gray"])
def test_estimated_gains_are_scale_invariant(method: str) -> None:
    rng = np.random.default_rng(11)
    img = rng.random((8, 8, 3), dtype=np.float32) + 0.1
    config = AWBConfig(method=method)  # type: ignore[arg-type]
    np.testing.assert_allclose(
        estimate_awb(3.0 * img, None, config).gains,
        estimate_awb(img, None, config).gains,
        rtol=1e-6,
    )


def test_apply_awb_does_not_clip() -> None:
    img = np.full((2, 2, 3), 10.0, np.float32)
    out = apply_awb(img, (2.0, 1.0, 1.0), cfa=None)
    assert float(out[..., 0].max()) == pytest.approx(20.0)
    assert float(out[..., 1].max()) == pytest.approx(10.0)


def test_apply_awb_returns_a_new_float32_array() -> None:
    img = _rgb(1.0, 2.0, 3.0).astype(np.float64)
    original = img.copy()
    out = apply_awb(img, (2.0, 1.0, 0.5), cfa=None)
    assert out.dtype == np.float32
    assert out is not img
    np.testing.assert_array_equal(img, original)
    np.testing.assert_allclose(out[..., 0], 2.0)
    np.testing.assert_allclose(out[..., 2], 1.5)


def test_estimate_awb_is_deterministic() -> None:
    rng = np.random.default_rng(21)
    img = rng.random((8, 8, 3), dtype=np.float32) + 0.1
    config = AWBConfig(method="shades_of_gray", p=6.0)
    assert estimate_awb(img, None, config) == estimate_awb(img, None, config)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(8, dtype=np.float32),
        np.zeros((4, 4, 4), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.float32),
    ],
)
def test_estimate_rejects_malformed_shapes(bad: NDArray[np.float32]) -> None:
    with pytest.raises(ValueError, match=r"2-D|three channels|at least one pixel"):
        estimate_awb(bad, cfa=None, config=AWBConfig())


def test_estimate_rejects_boolean_and_complex_input() -> None:
    with pytest.raises(ValueError, match="numeric"):
        estimate_awb(np.ones((4, 4, 3), dtype=bool), cfa=None, config=AWBConfig())
    with pytest.raises(ValueError, match="numeric"):
        estimate_awb(np.ones((4, 4, 3), dtype=np.complex64), cfa=None, config=AWBConfig())


def test_estimate_rejects_non_finite_input() -> None:
    for bad_value in (np.nan, np.inf):
        img = _rgb(1.0, 1.0, 1.0)
        img[0, 0, 0] = bad_value
        with pytest.raises(ValueError, match="finite"):
            estimate_awb(img, cfa=None, config=AWBConfig())


def test_estimate_rejects_negative_input() -> None:
    img = _rgb(1.0, 1.0, 1.0)
    img[0, 0, 0] = -0.5
    with pytest.raises(ValueError, match="non-negative"):
        estimate_awb(img, cfa=None, config=AWBConfig())


@pytest.mark.parametrize("method", ["gray_world", "white_patch", "shades_of_gray"])
def test_estimate_rejects_zero_channel_statistics(method: str) -> None:
    img = _rgb(1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match=r"channel G.*non-positive"):
        estimate_awb(img, cfa=None, config=AWBConfig(method=method))  # type: ignore[arg-type]


def test_estimate_requires_cfa_for_a_2d_mosaic() -> None:
    raw = _bayer(2.0, 1.0, 0.5, CFAPattern.RGGB)
    with pytest.raises(ValueError, match="cfa"):
        estimate_awb(raw, cfa=None, config=AWBConfig())


def test_estimate_rejects_a_mosaic_smaller_than_its_cfa_tile() -> None:
    raw = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="no samples"):
        estimate_awb(raw, cfa=CFAPattern.RGGB, config=AWBConfig())


def test_estimate_rejects_cfa_for_a_3d_rgb_image() -> None:
    with pytest.raises(ValueError, match="cfa must be None"):
        estimate_awb(_rgb(1.0, 1.0, 1.0), cfa=CFAPattern.RGGB, config=AWBConfig())


def test_estimate_rejects_unknown_cfa() -> None:
    raw = _bayer(2.0, 1.0, 0.5, CFAPattern.RGGB)
    with pytest.raises(ValueError, match="cfa pattern"):
        estimate_awb(raw, cfa="XYZ", config=AWBConfig())


def test_estimate_rejects_rccc() -> None:
    raw = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="RCCC"):
        estimate_awb(raw, cfa=CFAPattern.RCCC, config=AWBConfig())


def test_estimate_rejects_wrong_config_type() -> None:
    with pytest.raises(ValueError, match="AWBConfig"):
        estimate_awb(_rgb(1.0, 1.0, 1.0), cfa=None, config="gray_world")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_gains",
    [
        (1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, float("nan"), 1.0),
        (1.0, float("inf"), 1.0),
        (1e40, 1.0, 1.0),
        (1e-40, 1.0, 1.0),
        ("1", "1", "1"),
    ],
)
def test_estimate_rejects_invalid_manual_gains(bad_gains: object) -> None:
    # model_construct bypasses pydantic so the stage's own guards are exercised,
    # as they would be if a caller mutated a validated config.
    config = AWBConfig.model_construct(
        enabled=True,
        method="gray_world",
        p=6.0,
        gains=bad_gains,
    )
    with pytest.raises(ValueError, match="gains"):
        estimate_awb(_rgb(1.0, 1.0, 1.0), cfa=None, config=config)


@pytest.mark.parametrize("bad_p", [0.5, float("nan"), float("inf"), "six"])
def test_estimate_rejects_invalid_p(bad_p: object) -> None:
    config = AWBConfig.model_construct(
        enabled=True,
        method="shades_of_gray",
        p=bad_p,
        gains=None,
    )
    with pytest.raises(ValueError, match="p must be"):
        estimate_awb(_rgb(1.0, 1.0, 1.0), cfa=None, config=config)


def test_estimate_rejects_unknown_mutated_method() -> None:
    config = AWBConfig.model_construct(
        enabled=True,
        method="magic",
        p=6.0,
        gains=None,
    )
    with pytest.raises(ValueError, match="method"):
        estimate_awb(_rgb(1.0, 1.0, 1.0), cfa=None, config=config)


@pytest.mark.parametrize(
    "bad_gains",
    [
        (1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, float("nan"), 1.0),
        (1.0, float("inf"), 1.0),
        (1e40, 1.0, 1.0),
        ("1", "1", "1"),
    ],
)
def test_apply_rejects_invalid_gains(bad_gains: object) -> None:
    with pytest.raises(ValueError, match="gains"):
        apply_awb(_rgb(1.0, 1.0, 1.0), bad_gains, cfa=None)  # type: ignore[arg-type]


def test_apply_validates_the_cfa_layout() -> None:
    raw = _bayer(2.0, 1.0, 0.5, CFAPattern.RGGB)
    with pytest.raises(ValueError, match="cfa"):
        apply_awb(raw, (1.0, 1.0, 1.0), cfa=None)
    with pytest.raises(ValueError, match="cfa must be None"):
        apply_awb(_rgb(1.0, 1.0, 1.0), (1.0, 1.0, 1.0), cfa=CFAPattern.RGGB)
    with pytest.raises(ValueError, match="RCCC"):
        apply_awb(raw, (1.0, 1.0, 1.0), cfa=CFAPattern.RCCC)


def test_apply_rejects_malformed_images() -> None:
    with pytest.raises(ValueError, match=r"2-D|three channels"):
        apply_awb(np.zeros(8, dtype=np.float32), (1.0, 1.0, 1.0), cfa=None)
    with pytest.raises(ValueError, match="numeric"):
        apply_awb(np.ones((4, 4, 3), dtype=bool), (1.0, 1.0, 1.0), cfa=None)
    bad = _rgb(1.0, 1.0, 1.0)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        apply_awb(bad, (1.0, 1.0, 1.0), cfa=None)


def test_apply_rejects_float32_overflow() -> None:
    img = _rgb(1e38, 1.0, 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        apply_awb(img, (4.0, 1.0, 1.0), cfa=None)
