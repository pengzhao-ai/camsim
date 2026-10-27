"""Tests for the thresholded unsharp-mask sharpening stage.

Lenses, diffraction, the pixel aperture, the anti-aliasing filter, and
demosaicing all attenuate high spatial frequencies, so the rendered image is
softer than the scene. Unsharp masking restores apparent contrast by adding a
scaled high-pass residual::

    highpass = x - gaussian_blur(x, radius)
    gated    = highpass if |highpass| > threshold else 0
    out      = x + amount * gated

The stage runs in linear light, uses a separable Gaussian (sigma = radius,
truncated at 4 sigma, reflected borders) shared with the denoise stage, and
clips nothing: undershoot below zero and overshoot above one are preserved for
the tone stage to handle. A hard threshold suppresses low-contrast noise
instead of amplifying it.

With ``luma_only`` (the default) the high-pass is taken from BT.709 luma and
the same detail is added to R, G, and B. That construction is exact because the
luma weights sum to one: raising Y by ``d`` while holding Cb and Cr fixed adds
``d`` to every RGB channel, so chroma is untouched and colour fringing from
independently sharpened channels is avoided. Mono (2-D) input is sharpened
directly and ``luma_only`` is ignored.

The tests pin the identity contracts, edge-contrast gain, the exact threshold
gate and its boundary, suppression of sub-threshold noise, chroma preservation
under ``luma_only`` versus per-channel sharpening, linearity in ``amount``,
radius behaviour, the float32 C-contiguous output contract, input immutability,
determinism, and validation of images and (possibly mutated) configs. Failure
artefacts guarded here: clipping the overshoot, sharpening noise below the
threshold, chroma fringing, halos from an oversized radius/amount, and silently
accepting malformed data.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from camsim.config.schema import DenoiseConfig
from camsim.config.schema import SharpenConfig as SchemaSharpenConfig
from camsim.isp.denoise import gaussian_blur, rgb_to_opponent
from camsim.isp.sharpen import SharpenConfig, unsharp_mask


def _rgb_image(seed: int, shape: tuple[int, int] = (16, 16)) -> NDArray[np.float32]:
    """Return a reproducible linear RGB image in ``[0, 1)``."""
    return np.random.default_rng(seed).random((*shape, 3)).astype(np.float32)


def _step_image(size: int = 64, low: float = 0.0, high: float = 1.0) -> NDArray[np.float32]:
    """Return an RGB image with a vertical step between two plateaus."""
    image = np.full((size, size, 3), low, np.float32)
    image[:, size // 2 :] = high
    return image


def _step_image_2d(size: int = 64, low: float = 0.0, high: float = 1.0) -> NDArray[np.float32]:
    """Return a single-channel image with a vertical step."""
    image = np.full((size, size), low, np.float32)
    image[:, size // 2 :] = high
    return image


def test_sharpen_identity_when_amount_zero():
    x = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    np.testing.assert_allclose(unsharp_mask(x, SharpenConfig(amount=0.0)), x)


def test_sharpen_increases_edge_contrast():
    x = np.zeros((64, 64, 3), np.float32)
    x[:, 32:] = 1.0
    y = unsharp_mask(x, SharpenConfig(amount=1.0, radius=1.0, threshold=0.0))
    assert np.abs(np.diff(y[32, :, 0])).max() > np.abs(np.diff(x[32, :, 0])).max()


def test_sharpen_config_is_the_schema_model() -> None:
    assert SharpenConfig is SchemaSharpenConfig


def test_sharpen_disabled_is_an_exact_copy() -> None:
    image = _rgb_image(seed=1)
    output = unsharp_mask(image, SharpenConfig(enabled=False, amount=2.0, luma_only=False))
    np.testing.assert_array_equal(output, image)
    assert output.dtype == np.float32
    assert output.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(output, image)


def test_sharpen_mono_ignores_luma_only() -> None:
    image = _step_image_2d(size=32)
    luma = unsharp_mask(image, SharpenConfig(amount=1.0, radius=1.0, threshold=0.0, luma_only=True))
    per_channel = unsharp_mask(
        image,
        SharpenConfig(amount=1.0, radius=1.0, threshold=0.0, luma_only=False),
    )
    np.testing.assert_array_equal(luma, per_channel)
    assert luma.shape == image.shape
    assert np.abs(np.diff(luma[16, :])).max() > np.abs(np.diff(image[16, :])).max()


def test_luma_only_preserves_chroma() -> None:
    image = _rgb_image(seed=2, shape=(24, 24))
    before = rgb_to_opponent(image)
    luma = unsharp_mask(image, SharpenConfig(amount=1.5, radius=1.0, threshold=0.0))
    after = rgb_to_opponent(luma)
    # Chroma is untouched by construction; the residual difference is only the
    # float32 round-off of recomputing the opponent transform.
    np.testing.assert_allclose(after[..., 1:], before[..., 1:], rtol=0.0, atol=1e-6)
    assert np.abs(after[..., 0] - before[..., 0]).max() > 1e-2

    per_channel = unsharp_mask(
        image,
        SharpenConfig(amount=1.5, radius=1.0, threshold=0.0, luma_only=False),
    )
    shifted = rgb_to_opponent(per_channel)
    assert np.abs(shifted[..., 1:] - before[..., 1:]).max() > 1e-2


def test_sharpen_per_channel_applies_the_full_highpass() -> None:
    image = _rgb_image(seed=3, shape=(24, 24))
    output = unsharp_mask(
        image,
        SharpenConfig(amount=1.0, radius=1.0, threshold=0.0, luma_only=False),
    )
    expected = image - gaussian_blur(image, 1.0)
    np.testing.assert_allclose(output - image, expected, rtol=0.0, atol=1e-6)


def test_sharpen_threshold_suppresses_noise() -> None:
    rng = np.random.default_rng(4)
    image = (0.5 + rng.normal(0.0, 0.005, (32, 32, 3))).astype(np.float32)
    output = unsharp_mask(image, SharpenConfig(amount=1.0, radius=1.0, threshold=0.05))
    np.testing.assert_allclose(output, image, rtol=0.0, atol=1e-6)
    unthresholded = unsharp_mask(image, SharpenConfig(amount=1.0, radius=1.0, threshold=0.0))
    assert np.abs(unthresholded - image).max() > 1e-3


def test_sharpen_threshold_gate_is_inclusive_at_the_boundary() -> None:
    image = np.zeros((21, 21), np.float32)
    image[10, 10] = 1.0
    highpass = image - gaussian_blur(image, 1.0)
    boundary = float(np.abs(highpass[10, 10]))

    at_boundary = unsharp_mask(image, SharpenConfig(amount=1.0, radius=1.0, threshold=boundary))
    np.testing.assert_array_equal(at_boundary, image)

    just_below = np.nextafter(np.float32(boundary), np.float32(0.0))
    above = unsharp_mask(image, SharpenConfig(amount=1.0, radius=1.0, threshold=float(just_below)))
    assert above[10, 10] > image[10, 10]


def test_sharpen_amount_increases_edge_contrast() -> None:
    image = _step_image(size=32)
    contrasts = []
    for amount in (0.5, 1.0, 1.5):
        config = SharpenConfig(amount=amount, radius=1.0, threshold=0.0)
        contrasts.append(np.abs(np.diff(unsharp_mask(image, config)[16, :, 0])).max())
    assert contrasts[0] < contrasts[1] < contrasts[2]


def test_sharpen_detail_scales_linearly_with_amount() -> None:
    image = _rgb_image(seed=5, shape=(24, 24))
    single = unsharp_mask(image, SharpenConfig(amount=1.0, radius=1.0, threshold=0.0)) - image
    double = unsharp_mask(image, SharpenConfig(amount=2.0, radius=1.0, threshold=0.0)) - image
    np.testing.assert_allclose(double, 2.0 * single, rtol=1e-6, atol=1e-6)


def test_sharpen_preserves_overshoot_without_clipping() -> None:
    image = _step_image(size=32)
    output = unsharp_mask(image, SharpenConfig(amount=1.0, radius=1.0, threshold=0.0))
    assert output.min() < 0.0
    assert output.max() > 1.0


def test_sharpen_flat_image_with_threshold_is_an_exact_copy() -> None:
    image = np.full((16, 16, 3), 0.42, np.float32)
    output = unsharp_mask(image, SharpenConfig(amount=1.0, radius=1.0, threshold=0.01))
    np.testing.assert_array_equal(output, image)


def test_sharpen_larger_radius_widens_highpass_support() -> None:
    image = np.zeros((41, 41), np.float32)
    image[20, 20] = 1.0
    narrow = unsharp_mask(image, SharpenConfig(amount=1.0, radius=0.5, threshold=0.0))
    wide = unsharp_mask(image, SharpenConfig(amount=1.0, radius=2.0, threshold=0.0))
    assert np.count_nonzero(wide - image) > np.count_nonzero(narrow - image)


def test_sharpen_accepts_negative_linear_values() -> None:
    image = _rgb_image(seed=6) - 0.5
    output = unsharp_mask(image, SharpenConfig(amount=1.0, radius=1.0, threshold=0.0))
    assert output.dtype == np.float32
    assert output.shape == image.shape
    assert np.all(np.isfinite(output))


def test_sharpen_output_contract() -> None:
    codes = np.arange(16 * 16 * 3, dtype=np.int16).reshape(16, 16, 3) % 1024
    output = unsharp_mask(codes, SharpenConfig(amount=1.0, radius=1.0))
    assert output.dtype == np.float32
    assert output.shape == codes.shape
    assert output.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(output, codes)

    fortran = np.asfortranarray(_rgb_image(seed=7, shape=(12, 12)).astype(np.float64))
    converted = unsharp_mask(fortran, SharpenConfig(amount=1.0, radius=1.0))
    assert converted.dtype == np.float32
    assert converted.shape == fortran.shape
    assert converted.flags["C_CONTIGUOUS"]


def test_sharpen_does_not_mutate_input() -> None:
    image = _rgb_image(seed=8)
    reference = image.copy()
    output = unsharp_mask(image, SharpenConfig(amount=1.0, radius=1.0, threshold=0.0))
    assert not np.shares_memory(output, image)
    np.testing.assert_array_equal(image, reference)


def test_sharpen_is_deterministic() -> None:
    image = _rgb_image(seed=9, shape=(24, 24))
    config = SharpenConfig(amount=1.2, radius=1.3, threshold=0.005)
    np.testing.assert_array_equal(unsharp_mask(image, config), unsharp_mask(image, config))


@pytest.mark.parametrize("config", [None, {}, DenoiseConfig(), "sharpen", 1])
def test_sharpen_requires_a_sharpen_config(config: object) -> None:
    with pytest.raises(ValueError):
        unsharp_mask(_rgb_image(seed=10), config)


@pytest.mark.parametrize(
    "update",
    [
        {"radius": 0.0},
        {"radius": -1.0},
        {"radius": float("nan")},
        {"amount": -1.0},
        {"amount": float("inf")},
        {"amount": True},
        {"threshold": -0.1},
        {"threshold": float("nan")},
        {"threshold": "none"},
        {"radius": "wide"},
        {"luma_only": "yes"},
    ],
)
def test_sharpen_rejects_mutated_config(update: dict[str, object]) -> None:
    config = SharpenConfig(amount=1.0).model_copy(update=update)
    with pytest.raises(ValueError):
        unsharp_mask(_rgb_image(seed=11), config)


@pytest.mark.parametrize(
    "image",
    [
        np.float32(1.0),
        np.zeros(8, np.float32),
        np.zeros((2, 2, 2, 2), np.float32),
        np.zeros((8, 8, 2), np.float32),
        np.zeros((8, 8, 4), np.float32),
        np.zeros((0, 8, 3), np.float32),
        np.zeros((8, 8, 3), np.bool_),
        np.zeros((8, 8, 3), np.complex64),
        np.full((8, 8, 3), np.nan, np.float32),
        np.full((8, 8, 3), np.inf, np.float32),
        np.full((8, 8, 3), 1e300, np.float64),
        np.zeros((8, 8, 3), dtype="U3"),
    ],
)
def test_sharpen_rejects_invalid_images(image: NDArray[np.generic]) -> None:
    with pytest.raises(ValueError):
        unsharp_mask(image, SharpenConfig(amount=1.0))


def test_sharpen_disabled_still_validates_input() -> None:
    with pytest.raises(ValueError):
        unsharp_mask(np.zeros((4, 4, 2), np.float32), SharpenConfig(enabled=False, amount=1.0))
