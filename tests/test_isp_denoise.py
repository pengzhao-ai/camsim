"""Tests for the luma/chroma denoise stage.

Sensor noise is additive in linear light: photon shot noise grows as the square
root of the signal, read noise is roughly constant, and demosaicing plus colour
correction spread it across channels and correlate it spatially. Denoising
therefore runs after demosaicing (and the CCM) and before tone mapping, in
linear light, because smoothing gamma-encoded values would make the effective
strength depend on the signal level.

Human vision resolves luminance detail far better than chroma detail, and most
demosaicing error is chromatic, so the stage works in the reversible BT.709
opponent space::

    Y  = 0.2126 R + 0.7152 G + 0.0722 B
    Cb = (B - Y) / 1.8556
    Cr = (R - Y) / 1.5748

Both filters are blended toward the original with an independent strength per
channel, ``out = (1 - s) * x + s * filter(x)``. ``s = 1`` replaces the channel
with the filtered version; strengths above one are allowed and extrapolate past
it (documented overshoot), and zero strengths are an exact identity. The
Gaussian method uses a separable sigma=radius blur; the bilateral method uses
an edge-preserving filter whose range sigma follows the channel's 99th
percentile, making it scale equivariant.

The tests pin the identity contracts, opponent round-trip and matrix, noise
reduction and the direction of the strength/radius knobs, bilateral edge
preservation, linearity/scale equivariance of both methods, mono (2-D) paths,
the float32 C-contiguous output contract, input immutability, determinism, and
the validation of images and (possibly mutated) configs. Failure artefacts
guarded here: mixing channels during the blur, smoothing across edges, chroma
bleeding from a shared strength, over-smoothing (plastic look), filtering the
wrong domain, and silently accepting malformed data.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from camsim.config.schema import DenoiseConfig as SchemaDenoiseConfig
from camsim.config.schema import SharpenConfig
from camsim.isp.denoise import (
    DenoiseConfig,
    denoise,
    gaussian_blur,
    opponent_to_rgb,
    rgb_to_opponent,
)


def _rgb_image(seed: int, shape: tuple[int, int] = (16, 16)) -> NDArray[np.float32]:
    """Return a reproducible linear RGB image in ``[0, 1)``."""
    return np.random.default_rng(seed).random((*shape, 3)).astype(np.float32)


def _noisy_flat(
    seed: int,
    shape: tuple[int, int] = (48, 48),
    level: float = 0.5,
    sigma: float = 0.02,
) -> NDArray[np.float32]:
    """Return a flat grey patch with reproducible additive linear noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, (*shape, 3)).astype(np.float32)
    return np.full((*shape, 3), level, np.float32) + noise


def _noisy_step(
    seed: int,
    size: int = 64,
    low: float = 0.2,
    high: float = 0.8,
    sigma: float = 0.01,
) -> NDArray[np.float32]:
    """Return a noisy vertical step edge between two linear plateaus."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size), low, np.float32)
    image[:, size // 2 :] = high
    return image + rng.normal(0.0, sigma, image.shape).astype(np.float32)


def test_denoise_identity_when_disabled():
    x = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    np.testing.assert_allclose(denoise(x, DenoiseConfig(luma_strength=0, chroma_strength=0)), x)


def test_denoise_reduces_flat_patch_noise():
    rng = np.random.default_rng(0)
    x = np.full((64, 64, 3), 0.5, np.float32) + rng.normal(0, 0.02, (64, 64, 3)).astype(np.float32)
    y = denoise(
        x,
        DenoiseConfig(method="gaussian", luma_strength=1.0, chroma_strength=1.0, radius=1.0),
    )
    assert y.std() < x.std()


def test_denoise_config_is_the_schema_model() -> None:
    assert DenoiseConfig is SchemaDenoiseConfig


def test_denoise_disabled_is_an_exact_copy() -> None:
    image = _rgb_image(seed=1)
    output = denoise(
        image,
        DenoiseConfig(enabled=False, method="bilateral", luma_strength=1.0, chroma_strength=1.0),
    )
    np.testing.assert_array_equal(output, image)
    assert output.dtype == np.float32
    assert output.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(output, image)


def test_opponent_matrix_uses_bt709_luma_and_chroma_scales() -> None:
    colours = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    opponent = rgb_to_opponent(colours[np.newaxis, ...])[0]
    np.testing.assert_allclose(
        opponent[:, 0],
        np.array([0.2126, 0.7152, 0.0722, 0.5]),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        opponent[:, 1],
        np.array([-0.2126 / 1.8556, -0.7152 / 1.8556, (1.0 - 0.0722) / 1.8556, 0.0]),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        opponent[:, 2],
        np.array([(1.0 - 0.2126) / 1.5748, -0.7152 / 1.5748, -0.0722 / 1.5748, 0.0]),
        rtol=0.0,
        atol=1e-6,
    )


def test_opponent_round_trip() -> None:
    # Negative linear values are legal here: earlier stages can leave
    # excursions below zero, and the transform is a linear bijection.
    rng = np.random.default_rng(2)
    image = (rng.random((12, 10, 3)) - 0.5).astype(np.float32)
    restored = opponent_to_rgb(rgb_to_opponent(image))
    np.testing.assert_allclose(restored, image, rtol=0.0, atol=1e-6)
    assert restored.dtype == np.float32
    assert restored.shape == image.shape
    assert restored.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 2), (8, 8, 4)])
def test_opponent_helpers_reject_non_rgb_images(shape: tuple[int, ...]) -> None:
    image = np.zeros(shape, np.float32)
    with pytest.raises(ValueError):
        rgb_to_opponent(image)
    with pytest.raises(ValueError):
        opponent_to_rgb(image)


def test_gaussian_blur_does_not_mix_channels() -> None:
    image = _rgb_image(seed=3, shape=(20, 18))
    blurred = gaussian_blur(image, 1.3)
    for channel in range(3):
        np.testing.assert_array_equal(
            blurred[..., channel],
            gaussian_blur(image[..., channel], 1.3),
        )


def test_gaussian_blur_preserves_constants_and_contract() -> None:
    image = np.full((12, 13, 3), 0.37, np.float32)
    blurred = gaussian_blur(image, 2.0)
    np.testing.assert_allclose(blurred, image, rtol=0.0, atol=1e-6)
    assert blurred.dtype == np.float32
    assert blurred.shape == image.shape
    assert blurred.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(blurred, image)


def test_gaussian_blur_zero_sigma_is_an_exact_copy() -> None:
    image = _rgb_image(seed=4)
    np.testing.assert_array_equal(gaussian_blur(image, 0.0), image)


@pytest.mark.parametrize("sigma", [-1.0, float("nan"), float("inf")])
def test_gaussian_blur_rejects_invalid_sigma(sigma: float) -> None:
    with pytest.raises(ValueError):
        gaussian_blur(_rgb_image(seed=5), sigma)


def test_denoise_mono_ignores_chroma_strength() -> None:
    rng = np.random.default_rng(6)
    image = (0.5 + rng.normal(0.0, 0.02, (40, 40))).astype(np.float32)
    quiet = denoise(image, DenoiseConfig(luma_strength=1.0, chroma_strength=0.0, radius=1.0))
    loud = denoise(image, DenoiseConfig(luma_strength=1.0, chroma_strength=5.0, radius=1.0))
    np.testing.assert_array_equal(quiet, loud)
    assert quiet.shape == image.shape
    assert quiet.std() < image.std()


def test_denoise_mono_zero_luma_is_an_exact_copy() -> None:
    image = np.random.default_rng(7).random((16, 16)).astype(np.float32)
    np.testing.assert_array_equal(denoise(image, DenoiseConfig(chroma_strength=3.0)), image)


def test_denoise_luma_strength_reduces_luma_noise() -> None:
    image = _noisy_flat(seed=8, sigma=0.05)
    before = rgb_to_opponent(image)
    output = denoise(image, DenoiseConfig(luma_strength=1.0, chroma_strength=0.0, radius=1.0))
    after = rgb_to_opponent(output)
    assert after[..., 0].std() < before[..., 0].std()


def test_denoise_chroma_strength_reduces_chroma_noise() -> None:
    image = _noisy_flat(seed=9, sigma=0.05)
    before = rgb_to_opponent(image)
    output = denoise(image, DenoiseConfig(luma_strength=0.0, chroma_strength=1.0, radius=1.0))
    after = rgb_to_opponent(output)
    assert after[..., 1].std() < before[..., 1].std()
    assert after[..., 2].std() < before[..., 2].std()
    # A zero luma strength leaves luminance untouched apart from float32
    # round-off in the opponent round trip.
    np.testing.assert_allclose(after[..., 0], before[..., 0], rtol=0.0, atol=1e-6)


def test_denoise_stronger_strengths_smooth_more() -> None:
    image = _noisy_flat(seed=10)
    luma_stds = []
    for strength in (0.25, 0.5, 1.0):
        output = denoise(
            image,
            DenoiseConfig(luma_strength=strength, chroma_strength=strength, radius=1.0),
        )
        luma_stds.append(rgb_to_opponent(output)[..., 0].std())
    assert luma_stds[0] > luma_stds[1] > luma_stds[2]


def test_denoise_larger_radius_smooths_more() -> None:
    image = _noisy_flat(seed=11)
    narrow = denoise(image, DenoiseConfig(luma_strength=1.0, chroma_strength=1.0, radius=0.5))
    wide = denoise(image, DenoiseConfig(luma_strength=1.0, chroma_strength=1.0, radius=2.0))
    assert wide.std() < narrow.std()


@pytest.mark.parametrize("method", ["gaussian", "bilateral"])
def test_denoise_linearity(method: str) -> None:
    rng = np.random.default_rng(12)
    image = (0.1 + 0.8 * rng.random((24, 24, 3))).astype(np.float32)
    config = DenoiseConfig(method=method, luma_strength=0.7, chroma_strength=0.4, radius=1.2)
    scale = 2.0
    np.testing.assert_allclose(
        denoise(scale * image, config),
        scale * denoise(image, config),
        rtol=1e-6,
        atol=1e-6,
    )


def test_bilateral_edge_preservation() -> None:
    image = _noisy_step(seed=13)
    bilateral = denoise(image, DenoiseConfig(method="bilateral", luma_strength=1.0, radius=1.0))
    gaussian = denoise(image, DenoiseConfig(method="gaussian", luma_strength=1.0, radius=1.0))

    rows = slice(8, -8)
    plateau = slice(2, 24)
    assert bilateral[rows, plateau].std() < image[rows, plateau].std()
    assert gaussian[rows, plateau].std() < image[rows, plateau].std()

    # A narrower transition shows up as a larger single-pixel step in the row
    # profile: bilateral filtering averages within each side of the edge,
    # while the Gaussian necessarily mixes across it.
    bilateral_step = np.abs(np.diff(bilateral[rows].mean(axis=0))).max()
    gaussian_step = np.abs(np.diff(gaussian[rows].mean(axis=0))).max()
    assert bilateral_step > gaussian_step


def test_denoise_bilateral_reduces_flat_patch_noise() -> None:
    image = _noisy_flat(seed=14)
    output = denoise(
        image,
        DenoiseConfig(method="bilateral", luma_strength=1.0, chroma_strength=1.0, radius=1.0),
    )
    assert output.std() < image.std()


def test_denoise_strength_above_one_is_accepted() -> None:
    # Strengths are blend fractions, not caps; the config schema permits any
    # non-negative value and the stage documents the resulting overshoot.
    image = _noisy_flat(seed=15, sigma=0.05)
    single = denoise(image, DenoiseConfig(luma_strength=1.0, chroma_strength=1.0, radius=1.0))
    doubled = denoise(image, DenoiseConfig(luma_strength=2.0, chroma_strength=2.0, radius=1.0))
    assert np.all(np.isfinite(doubled))
    assert not np.allclose(single, doubled)


@pytest.mark.parametrize("method", ["gaussian", "bilateral"])
def test_denoise_accepts_negative_linear_values(method: str) -> None:
    image = _rgb_image(seed=16) - 0.5
    output = denoise(image, DenoiseConfig(method=method, luma_strength=1.0, chroma_strength=1.0))
    assert output.dtype == np.float32
    assert output.shape == image.shape
    assert np.all(np.isfinite(output))


def test_denoise_output_contract() -> None:
    codes = np.arange(16 * 16 * 3, dtype=np.int16).reshape(16, 16, 3) % 1024
    output = denoise(codes, DenoiseConfig(luma_strength=0.5, chroma_strength=0.5))
    assert output.dtype == np.float32
    assert output.shape == codes.shape
    assert output.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(output, codes)

    fortran = np.asfortranarray(_rgb_image(seed=17, shape=(12, 12)).astype(np.float64))
    converted = denoise(fortran, DenoiseConfig(luma_strength=0.5, chroma_strength=0.5))
    assert converted.dtype == np.float32
    assert converted.shape == fortran.shape
    assert converted.flags["C_CONTIGUOUS"]


def test_denoise_does_not_mutate_input() -> None:
    image = _rgb_image(seed=18)
    reference = image.copy()
    for method in ("gaussian", "bilateral"):
        output = denoise(
            image,
            DenoiseConfig(method=method, luma_strength=1.0, chroma_strength=1.0),
        )
        assert not np.shares_memory(output, image)
    np.testing.assert_array_equal(image, reference)


@pytest.mark.parametrize("method", ["gaussian", "bilateral"])
def test_denoise_is_deterministic(method: str) -> None:
    image = _rgb_image(seed=19, shape=(24, 24))
    config = DenoiseConfig(method=method, luma_strength=0.8, chroma_strength=0.5, radius=1.1)
    np.testing.assert_array_equal(denoise(image, config), denoise(image, config))


@pytest.mark.parametrize("config", [None, {}, SharpenConfig(), "gaussian", 1])
def test_denoise_requires_a_denoise_config(config: object) -> None:
    with pytest.raises(ValueError):
        denoise(_rgb_image(seed=20), config)


@pytest.mark.parametrize(
    "update",
    [
        {"radius": 0.0},
        {"radius": -1.0},
        {"radius": float("nan")},
        {"radius": float("inf")},
        {"luma_strength": -0.1},
        {"luma_strength": True},
        {"chroma_strength": float("inf")},
        {"chroma_strength": "high"},
        {"radius": "wide"},
        {"method": "median"},
    ],
)
def test_denoise_rejects_mutated_config(update: dict[str, object]) -> None:
    config = DenoiseConfig(luma_strength=1.0).model_copy(update=update)
    with pytest.raises(ValueError):
        denoise(_rgb_image(seed=21), config)


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
def test_denoise_rejects_invalid_images(image: NDArray[np.generic]) -> None:
    with pytest.raises(ValueError):
        denoise(image, DenoiseConfig(luma_strength=1.0))


def test_denoise_disabled_still_validates_input() -> None:
    with pytest.raises(ValueError):
        denoise(np.zeros((4, 4, 2), np.float32), DenoiseConfig(enabled=False))
