"""Tests for flat-patch SNR metrics: per-channel, luma, spatial, and temporal.

A signal-to-noise ratio measures how far a signal sits above its own
fluctuations, ``SNR_dB = 20 log10(mean / std)``. Spatial SNR reads the sample
standard deviation across the pixels of one flat patch in one frame. Temporal
SNR reads the per-pixel fluctuation across a stack of frames and pools the
per-pixel sample variances, so a static fixed-pattern variation (which is
signal, not noise, in the temporal sense) cancels while random,
frame-independent noise survives. The distinction matters: a sensor with strong
fixed-pattern noise but low temporal noise can look bad in a single frame and
excellent in a temporal measurement, and the two numbers must not be confused.

Every measurement here is on linear-light data. A display-encoded (gamma) image
compresses its fluctuations near the top of the range, so its standard
deviation is not the sensor's and the reported SNR is biased. A black-level
pedestal is a constant offset: it raises the mean without changing the std,
inflating SNR unless the caller subtracts it first. A clipped or saturated
patch is flat, so it is caught by ``std == 0`` rather than by inspecting sample
values; clipping diagnostics belong to the RAW metadata.

Known-noise references are generated with ``np.random.default_rng`` and a fixed
seed, so a failure is reproducible. Real-world ranges: SNR 10-50 dB, flat
patches 16 x 16 to 128 x 128 pixels, and temporal stacks of 4-32 frames. The
luma plane is Rec. 709 luminance in linear light,
``0.2126 R + 0.7152 G + 0.0722 B``.

Failure artefacts guarded here: gamma-encoded input biases SNR, a saturated or
black patch has no noise statistic at all, a negative mean (an over-subtracted
black level) makes the ratio meaningless, spatial fixed-pattern noise inflates
a temporal estimate when variances are pooled incorrectly, and a single frame
is not a temporal measurement.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from camsim.metrics.chart import PatchROI
from camsim.metrics.snr import SNRResult, snr_channels, snr_patch, snr_temporal

# --------------------------------------------------------------------------- #
# Brief contract tests (verbatim from the task brief)
# --------------------------------------------------------------------------- #


def test_snr_known_gaussian_noise():
    rng = np.random.default_rng(0)
    img = 10.0 + rng.normal(0, 1.0, (64, 64, 1)).astype(np.float32)
    res = snr_patch(img, PatchROI("p", 0, 0, 64, 64), channel=0)
    assert res.snr_db == pytest.approx(20.0, abs=0.5)


def test_snr_clipped_patch_invalid():
    img = np.ones((8, 8, 1), np.float32)
    res = snr_patch(img, PatchROI("p", 0, 0, 8, 8), channel=0)
    assert not res.valid


def test_snr_luma():
    img = np.full((16, 16, 3), 0.5, np.float32)
    res = snr_patch(img, PatchROI("p", 0, 0, 16, 16), channel="luma")
    assert res.channel == "luma"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_REC709 = (0.2126, 0.7152, 0.0722)


def _roi(x: int = 0, y: int = 0, width: int = 16, height: int = 16) -> PatchROI:
    """Return a patch ROI with a fixed name and the requested geometry."""
    return PatchROI("patch", x, y, width, height)


def _noisy_rgb(seed: int = 0, shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    """Return an RGB image whose channels have distinct level and noise."""
    rng = np.random.default_rng(seed)
    image = np.empty((*shape, 3), dtype=np.float64)
    image[..., 0] = 10.0 + rng.normal(0.0, 1.0, shape)
    image[..., 1] = 20.0 + rng.normal(0.0, 2.0, shape)
    image[..., 2] = 40.0 + rng.normal(0.0, 4.0, shape)
    return image


# --------------------------------------------------------------------------- #
# Spatial statistics
# --------------------------------------------------------------------------- #


def test_snr_spatial_uses_sample_std_on_the_flattened_roi() -> None:
    image = np.arange(64.0).reshape(8, 8)
    roi = PatchROI("region", 2, 1, 3, 2)  # rows 1-2, columns 2-4
    values = image[1:3, 2:5].ravel()

    result = snr_patch(image, roi)

    assert result.channel == "mono"
    assert result.mean == pytest.approx(float(values.mean()), rel=1e-12)
    assert result.std == pytest.approx(float(values.std(ddof=1)), rel=1e-12)
    assert result.snr_db == pytest.approx(
        20.0 * math.log10(float(values.mean()) / float(values.std(ddof=1))), rel=1e-12
    )
    assert result.valid
    # The ROI is a strict subset, so it cannot report the whole-image mean.
    assert result.mean != pytest.approx(float(image.mean()))


def test_snr_spatial_reports_python_floats_from_integer_input() -> None:
    image = np.array([[10, 12], [11, 13]], dtype=np.uint16)
    result = snr_patch(image, PatchROI("p", 0, 0, 2, 2))
    assert isinstance(result.mean, float)
    assert isinstance(result.std, float)
    assert isinstance(result.snr_db, float)
    assert isinstance(result.valid, bool)
    assert image.dtype == np.uint16


def test_snr_accepts_nested_lists_as_array_like() -> None:
    image = [[10.0, 11.0], [9.0, 10.5]]
    result = snr_patch(image, PatchROI("p", 0, 0, 2, 2))
    values = np.asarray(image, dtype=np.float64).ravel()
    assert result.mean == pytest.approx(float(values.mean()), rel=1e-12)
    assert result.std == pytest.approx(float(values.std(ddof=1)), rel=1e-12)
    assert result.channel == "mono"


# --------------------------------------------------------------------------- #
# Per-channel and luma resolution
# --------------------------------------------------------------------------- #


def test_snr_per_channel() -> None:
    image = _noisy_rgb(seed=1, shape=(32, 32))
    roi = PatchROI("p", 0, 0, 32, 32)

    results = {}
    for index, label, level, sigma in (
        (0, "R", 10.0, 1.0),
        (1, "G", 20.0, 2.0),
        (2, "B", 40.0, 4.0),
    ):
        by_index = snr_patch(image, roi, channel=index)
        by_label = snr_patch(image, roi, channel=label.lower())
        assert by_index.channel == label
        assert by_label.channel == label
        assert by_index.mean == pytest.approx(level, abs=5.0 * sigma / 32.0)
        assert by_index.std == pytest.approx(sigma, rel=0.1)
        assert by_index.snr_db == pytest.approx(20.0 * math.log10(level / sigma), abs=0.5)
        assert by_index.valid
        assert by_label.mean == by_index.mean
        assert by_label.std == by_index.std
        results[label] = by_index

    # Distinct levels and noise make the three channels distinguishable.
    assert results["R"].std < results["G"].std < results["B"].std
    assert results["R"].mean < results["G"].mean < results["B"].mean


def test_snr_luma_uses_rec709_weights_in_linear_light() -> None:
    rng = np.random.default_rng(2)
    shape = (24, 20)
    image = np.stack(
        [
            0.6 + rng.normal(0.0, 0.01, shape),
            0.4 + rng.normal(0.0, 0.01, shape),
            0.2 + rng.normal(0.0, 0.01, shape),
        ],
        axis=-1,
    )
    roi = PatchROI("p", 2, 3, 16, 18)

    result = snr_patch(image, roi, channel="luma")

    patch = image[3:21, 2:18]
    red, green, blue = _REC709
    luma = red * patch[..., 0] + green * patch[..., 1] + blue * patch[..., 2]
    expected_mean = float(luma.mean())
    expected_std = float(luma.std(ddof=1))
    assert result.channel == "luma"
    assert result.mean == pytest.approx(expected_mean, rel=1e-12)
    assert result.std == pytest.approx(expected_std, rel=1e-12)
    assert result.snr_db == pytest.approx(
        20.0 * math.log10(expected_mean / expected_std), rel=1e-12
    )
    assert result.valid


def test_snr_luma_hand_computed_triple() -> None:
    image = np.zeros((4, 4, 3), dtype=np.float64)
    image[..., 0] = 0.5
    image[..., 1] = 0.25
    image[..., 2] = 0.1

    result = snr_patch(image, PatchROI("p", 0, 0, 4, 4), channel="luma")

    # 0.2126*0.5 + 0.7152*0.25 + 0.0722*0.1 = 0.29232
    assert result.mean == pytest.approx(0.29232, rel=1e-12)
    assert result.std == 0.0
    assert not result.valid
    assert math.isnan(result.snr_db)


def test_snr_default_channel_for_a_colour_image_is_luma() -> None:
    image = _noisy_rgb(seed=3)
    roi = PatchROI("p", 0, 0, 16, 16)
    default = snr_patch(image, roi)
    explicit = snr_patch(image, roi, channel="luma")
    assert default.channel == "luma"
    assert default.mean == pytest.approx(explicit.mean, rel=1e-12)
    assert default.std == pytest.approx(explicit.std, rel=1e-12)
    assert default.snr_db == pytest.approx(explicit.snr_db, rel=1e-12)


def test_snr_mono() -> None:
    rng = np.random.default_rng(4)
    image = 10.0 + rng.normal(0.0, 1.0, (16, 24))
    roi = PatchROI("p", 0, 0, 24, 16)
    expected_mean = float(image.mean())
    expected_std = float(image.std(ddof=1))

    for channel in (None, "mono", "M", "mOnO"):
        result = snr_patch(image, roi, channel=channel)
        assert result.channel == "mono"
        assert result.mean == pytest.approx(expected_mean, rel=1e-12)
        assert result.std == pytest.approx(expected_std, rel=1e-12)
        assert result.snr_db == pytest.approx(
            20.0 * math.log10(expected_mean / expected_std), rel=1e-12
        )
        assert result.valid

    channels = snr_channels(image, roi)
    assert list(channels) == ["mono"]
    assert channels["mono"].mean == pytest.approx(expected_mean, rel=1e-12)
    assert channels["mono"].std == pytest.approx(expected_std, rel=1e-12)


@pytest.mark.parametrize(
    ("channel", "label"),
    [
        (0, "R"),
        (1, "G"),
        (2, "B"),
        ("R", "R"),
        ("r", "R"),
        ("G", "G"),
        ("g", "G"),
        ("B", "B"),
        ("b", "B"),
        ("LUMA", "luma"),
        ("Luma", "luma"),
        (None, "luma"),
    ],
)
def test_snr_resolves_channel_labels_and_indices(channel: object, label: str) -> None:
    image = _noisy_rgb(seed=5)
    result = snr_patch(image, _roi(width=16, height=16), channel=channel)  # type: ignore[arg-type]
    assert result.channel == label


@pytest.mark.parametrize(
    "channel",
    [0, 1, 2, "R", "G", "B", "luma", "LUMA", "x", "", 1.0, True, np.int64(0)],
)
def test_snr_two_dimensional_image_rejects_colour_channels(channel: object) -> None:
    image = 10.0 + np.arange(64.0).reshape(8, 8) * 0.01
    with pytest.raises(ValueError, match="channel"):
        snr_patch(image, PatchROI("p", 0, 0, 8, 8), channel=channel)  # type: ignore[arg-type]


@pytest.mark.parametrize("channel", [3, -1, "X", "mono", "M", "", 1.0, True])
def test_snr_colour_image_rejects_invalid_channels(channel: object) -> None:
    image = _noisy_rgb(seed=6, shape=(8, 8))
    with pytest.raises(ValueError, match="channel"):
        snr_patch(image, PatchROI("p", 0, 0, 8, 8), channel=channel)  # type: ignore[arg-type]


def test_snr_luma_requires_exactly_three_channels() -> None:
    single = np.full((8, 8, 1), 0.5)
    with pytest.raises(ValueError, match="channel"):
        snr_patch(single, PatchROI("p", 0, 0, 8, 8), channel="luma")
    # The default for a 3-D image is luma, so a one-channel stack also fails.
    with pytest.raises(ValueError, match="channel"):
        snr_patch(single, PatchROI("p", 0, 0, 8, 8))

    # A one-channel 3-D image still exposes its sole channel by index and label.
    noisy_single = single + np.arange(64.0).reshape(8, 8)[..., None] * 0.01
    assert snr_patch(noisy_single, PatchROI("p", 0, 0, 8, 8), channel=0).channel == "R"
    with pytest.raises(ValueError, match="channel"):
        snr_patch(noisy_single, PatchROI("p", 0, 0, 8, 8), channel=1)


@pytest.mark.parametrize("channels", [2, 4, 5])
def test_snr_three_dimensional_images_must_have_one_or_three_channels(channels: int) -> None:
    image = np.full((8, 8, channels), 0.5)
    with pytest.raises(ValueError, match="channel"):
        snr_patch(image, PatchROI("p", 0, 0, 8, 8), channel=0)
    with pytest.raises(ValueError, match="channel"):
        snr_patch(image, PatchROI("p", 0, 0, 8, 8))


# --------------------------------------------------------------------------- #
# Invalid patches and images
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "image", "expected_mean"),
    [
        ("flat-half", np.full((8, 8), 0.5), 0.5),
        ("saturated-ones", np.ones((8, 8)), 1.0),
        ("black", np.zeros((8, 8)), 0.0),
        ("negative-flat", np.full((8, 8), -0.5), -0.5),
    ],
)
def test_snr_flat_or_nonpositive_patches_are_invalid(
    label: str, image: np.ndarray, expected_mean: float
) -> None:
    result = snr_patch(image, PatchROI("p", 0, 0, 8, 8))
    assert not result.valid
    assert math.isnan(result.snr_db)
    assert result.mean == pytest.approx(expected_mean, rel=1e-12)
    assert result.std == 0.0


def test_snr_negative_mean_with_noise_is_invalid() -> None:
    rng = np.random.default_rng(7)
    image = -5.0 + rng.normal(0.0, 1.0, (16, 16))
    result = snr_patch(image, _roi(width=16, height=16))
    assert result.mean < 0.0
    assert result.std > 0.0
    assert not result.valid
    assert math.isnan(result.snr_db)


def test_snr_single_pixel_patch_is_invalid() -> None:
    image = np.full((4, 4), 2.0)
    image[0, 0] = 3.0
    result = snr_patch(image, PatchROI("p", 0, 0, 1, 1))
    assert result.mean == pytest.approx(3.0)
    assert result.std == 0.0
    assert not result.valid
    assert math.isnan(result.snr_db)


def test_snr_black_level_handling() -> None:
    rng = np.random.default_rng(8)
    signal = 10.0 + rng.normal(0.0, 1.0, (32, 32))
    roi = PatchROI("p", 0, 0, 32, 32)

    without_pedestal = snr_patch(signal, roi)
    with_pedestal = snr_patch(signal + 64.0, roi)

    # The noise is unchanged, but an unsubtracted pedestal raises the mean and
    # therefore inflates the reported SNR.
    assert without_pedestal.snr_db == pytest.approx(20.0, abs=0.5)
    assert with_pedestal.mean - without_pedestal.mean == pytest.approx(64.0, abs=1e-9)
    assert with_pedestal.std == pytest.approx(without_pedestal.std, rel=1e-9)
    assert with_pedestal.snr_db == pytest.approx(20.0 * math.log10(74.0 / 1.0), abs=0.5)

    # A patch sitting exactly on the black level has no measurable signal and
    # is rejected by std == 0, not by inspecting the code values.
    black = snr_patch(np.full((16, 16), 64.0), _roi(width=16, height=16))
    assert black.mean == pytest.approx(64.0)
    assert black.std == 0.0
    assert not black.valid
    assert math.isnan(black.snr_db)


@pytest.mark.parametrize(
    ("label", "image"),
    [
        ("one-dimensional", np.zeros(8)),
        ("three-dimensional-empty", np.zeros((8, 8, 0))),
        ("four-dimensional", np.zeros((2, 8, 8, 1))),
        ("boolean", np.zeros((8, 8), dtype=bool)),
        ("complex", np.zeros((8, 8), dtype=complex)),
        ("strings", np.full((8, 8), "x")),
        ("object", np.zeros((8, 8), dtype=object)),
        ("nan", np.full((8, 8), math.nan)),
        ("infinity", np.full((8, 8), math.inf)),
        ("negative-infinity", np.full((8, 8), -math.inf)),
        ("scalar", 1.0),
        ("none", None),
    ],
)
def test_snr_rejects_invalid_images(label: str, image: object) -> None:
    with pytest.raises(ValueError):
        snr_patch(image, PatchROI("p", 0, 0, 4, 4))  # type: ignore[arg-type]


def test_snr_rejects_an_roi_outside_the_image() -> None:
    image = np.full((8, 8), 10.0)
    with pytest.raises(ValueError, match="outside"):
        snr_patch(image, PatchROI("p", 6, 0, 4, 4))
    with pytest.raises(ValueError, match="outside"):
        snr_patch(image, PatchROI("p", 0, 6, 4, 4))


def test_snr_rejects_a_non_roi_argument() -> None:
    with pytest.raises(TypeError, match="PatchROI"):
        snr_patch(np.zeros((8, 8)), (0, 0, 8, 8))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Temporal statistics
# --------------------------------------------------------------------------- #


def test_snr_temporal() -> None:
    rng = np.random.default_rng(9)
    frames = 5.0 + rng.normal(0.0, 0.5, (24, 16, 20))
    roi = PatchROI("p", 2, 3, 12, 10)

    result = snr_temporal(frames, roi)

    crop = frames[:, 3:13, 2:14]
    per_pixel_means = crop.mean(axis=0)
    per_pixel_variances = crop.var(axis=0, ddof=1)
    expected_mean = float(per_pixel_means.mean())
    expected_std = float(np.sqrt(per_pixel_variances.mean()))

    assert result.channel == "mono"
    assert result.mean == pytest.approx(expected_mean, rel=1e-12)
    assert result.std == pytest.approx(expected_std, rel=1e-12)
    assert result.std == pytest.approx(0.5, rel=0.1)
    assert result.snr_db == pytest.approx(20.0 * math.log10(5.0 / 0.5), abs=0.5)
    assert result.snr_db == pytest.approx(
        20.0 * math.log10(expected_mean / expected_std), rel=1e-12
    )
    assert result.valid


def test_snr_temporal_per_channel_and_luma() -> None:
    rng = np.random.default_rng(10)
    shape = (16, 12, 12)
    frames = np.empty((*shape, 3), dtype=np.float64)
    frames[..., 0] = 10.0 + rng.normal(0.0, 1.0, shape)
    frames[..., 1] = 20.0 + rng.normal(0.0, 1.0, shape)
    frames[..., 2] = 5.0 + rng.normal(0.0, 1.0, shape)
    roi = PatchROI("p", 0, 0, 12, 12)

    red = snr_temporal(frames, roi, channel="R")
    assert red.channel == "R"
    assert red.snr_db == pytest.approx(20.0, abs=0.5)
    assert snr_temporal(frames, roi, channel=1).channel == "G"
    assert snr_temporal(frames, roi, channel="b").channel == "B"

    luma = snr_temporal(frames, roi, channel="luma")
    red, green, blue = _REC709
    luma_frames = red * frames[..., 0] + green * frames[..., 1] + blue * frames[..., 2]
    expected = snr_temporal(luma_frames, roi)
    assert luma.channel == "luma"
    assert luma.mean == pytest.approx(expected.mean, rel=1e-12)
    assert luma.std == pytest.approx(expected.std, rel=1e-12)
    assert luma.snr_db == pytest.approx(expected.snr_db, rel=1e-12)
    assert luma.valid


def test_snr_temporal_pools_noise_and_rejects_fixed_pattern() -> None:
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.5, (16, 32, 32))
    pattern = np.empty((32, 32), dtype=np.float64)
    pattern[::2] = -5.0
    pattern[1::2] = 5.0
    roi = PatchROI("p", 0, 0, 32, 32)

    temporal_probe = snr_temporal(10.0 + noise, roi)
    with_fixed_pattern = snr_temporal(10.0 + pattern + noise, roi)

    # Pooling per-pixel temporal variances excludes the static pattern, so the
    # temporal SNR is the noise-only value.
    assert with_fixed_pattern.std == pytest.approx(temporal_probe.std, rel=0.1)
    assert with_fixed_pattern.snr_db == pytest.approx(temporal_probe.snr_db, abs=0.2)
    assert with_fixed_pattern.mean == pytest.approx(10.0, abs=0.1)
    assert with_fixed_pattern.valid

    # A single frame confuses the fixed pattern with noise and reports far less.
    spatial = snr_patch(10.0 + pattern + noise[0], roi)
    assert spatial.snr_db < temporal_probe.snr_db - 10.0


def test_snr_temporal_flat_zero_and_negative_stacks_are_invalid() -> None:
    roi = PatchROI("p", 0, 0, 8, 8)

    flat = snr_temporal(np.full((8, 8, 8), 0.5), roi)
    assert flat.mean == pytest.approx(0.5)
    assert flat.std == 0.0
    assert not flat.valid
    assert math.isnan(flat.snr_db)

    assert not snr_temporal(np.zeros((4, 8, 8)), roi).valid

    rng = np.random.default_rng(12)
    negative = -1.0 + rng.normal(0.0, 0.1, (4, 8, 8))
    assert not snr_temporal(negative, roi).valid


def test_snr_temporal_requires_at_least_two_frames() -> None:
    roi = PatchROI("p", 0, 0, 8, 8)
    with pytest.raises(ValueError, match="two frames"):
        snr_temporal(np.ones((1, 8, 8)), roi)
    with pytest.raises(ValueError, match="two frames"):
        snr_temporal(np.empty((0, 8, 8)), roi)


@pytest.mark.parametrize(
    ("label", "frames"),
    [
        ("one-dimensional", np.zeros(8)),
        ("two-dimensional", np.zeros((2, 8))),
        ("five-dimensional", np.zeros((2, 8, 8, 1, 1))),
        ("boolean", np.zeros((2, 8, 8), dtype=bool)),
        ("complex", np.zeros((2, 8, 8), dtype=complex)),
        ("strings", np.full((2, 8, 8), "x")),
        ("nan", np.full((2, 8, 8), math.nan)),
        ("infinity", np.full((2, 8, 8), math.inf)),
        ("list-of-frames", [[1.0, 2.0], [3.0, 4.0]]),
        ("none", None),
    ],
)
def test_snr_temporal_rejects_invalid_stacks(label: str, frames: object) -> None:
    with pytest.raises(ValueError):
        snr_temporal(frames, PatchROI("p", 0, 0, 2, 2))  # type: ignore[arg-type]


def test_snr_temporal_rejects_ragged_stacks() -> None:
    ragged = [np.zeros((8, 8)), np.zeros((9, 9))]
    with pytest.raises(ValueError, match="frames"):
        snr_temporal(ragged, PatchROI("p", 0, 0, 4, 4))


def test_snr_temporal_rejects_an_roi_outside_the_frames() -> None:
    frames = np.ones((4, 8, 8))
    with pytest.raises(ValueError, match="outside"):
        snr_temporal(frames, PatchROI("p", 6, 0, 4, 4))
    with pytest.raises(TypeError, match="PatchROI"):
        snr_temporal(frames, (0, 0, 8, 8))  # type: ignore[arg-type]


def test_snr_temporal_accepts_float32_and_leaves_the_stack_untouched() -> None:
    rng = np.random.default_rng(13)
    frames = (5.0 + rng.normal(0.0, 0.5, (8, 8, 8))).astype(np.float32)
    before = frames.copy()

    first = snr_temporal(frames, PatchROI("p", 0, 0, 8, 8))
    second = snr_temporal(frames, PatchROI("p", 0, 0, 8, 8))

    np.testing.assert_array_equal(frames, before)
    assert frames.dtype == np.float32
    assert first.mean == second.mean
    assert first.std == second.std
    assert first.snr_db == second.snr_db


def test_snr_temporal_accepts_a_list_of_frames() -> None:
    rng = np.random.default_rng(14)
    frames = [5.0 + rng.normal(0.0, 0.5, (8, 8)) for _ in range(4)]
    result = snr_temporal(frames, PatchROI("p", 0, 0, 8, 8))
    stack = np.stack(frames, axis=0)
    assert result.mean == pytest.approx(float(stack.mean(axis=(1, 2)).mean()), rel=1e-12)
    assert result.valid


# --------------------------------------------------------------------------- #
# Per-channel helper
# --------------------------------------------------------------------------- #


def test_snr_channels_returns_every_colour_and_luma_in_order() -> None:
    image = _noisy_rgb(seed=15)
    roi = PatchROI("p", 0, 0, 16, 16)

    results = snr_channels(image, roi)

    assert list(results) == ["R", "G", "B", "luma"]
    assert all(isinstance(result, SNRResult) for result in results.values())
    for name in ("R", "G", "B"):
        direct = snr_patch(image, roi, channel=name)
        assert results[name].channel == name
        assert results[name].mean == pytest.approx(direct.mean, rel=1e-12)
        assert results[name].std == pytest.approx(direct.std, rel=1e-12)
        assert results[name].snr_db == pytest.approx(direct.snr_db, rel=1e-12)
    luma = snr_patch(image, roi, channel="luma")
    assert results["luma"].mean == pytest.approx(luma.mean, rel=1e-12)
    assert results["luma"].std == pytest.approx(luma.std, rel=1e-12)
    assert results["luma"].channel == "luma"
    assert all(result.valid for result in results.values())
    # The three channels carry distinct noise levels in this fixture.
    assert results["R"].std < results["G"].std < results["B"].std


def test_snr_channels_rejects_images_that_are_not_mono_or_rgb() -> None:
    roi = PatchROI("p", 0, 0, 8, 8)
    with pytest.raises(ValueError, match="3"):
        snr_channels(np.full((8, 8, 1), 0.5), roi)
    with pytest.raises(ValueError, match="3"):
        snr_channels(np.full((8, 8, 4), 0.5), roi)
    # A mono image is reported under one key, not four.
    mono = 10.0 + np.arange(64.0).reshape(8, 8) * 0.01
    assert list(snr_channels(mono, roi)) == ["mono"]


# --------------------------------------------------------------------------- #
# SNRResult value object
# --------------------------------------------------------------------------- #


def _result(**overrides: object) -> SNRResult:
    """Return a valid SNRResult with selected fields replaced."""
    fields: dict[str, object] = {
        "mean": 10.0,
        "std": 1.0,
        "snr_db": 20.0,
        "channel": "R",
        "valid": True,
    }
    fields.update(overrides)
    return SNRResult(**fields)  # type: ignore[arg-type]


def test_snr_result_is_frozen() -> None:
    result = _result()
    with pytest.raises(FrozenInstanceError):
        result.mean = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean", math.nan),
        ("mean", math.inf),
        ("mean", -math.inf),
        ("std", math.nan),
        ("std", math.inf),
        ("std", -1.0),
        ("channel", ""),
        ("channel", "   "),
        ("channel", 3),
        ("valid", 1),
        ("valid", "yes"),
        ("valid", None),
        ("snr_db", math.inf),
        ("snr_db", -math.inf),
    ],
)
def test_snr_result_rejects_invalid_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _result(**{field: value})


def test_snr_result_invalid_measurement_reports_nan_snr() -> None:
    result = SNRResult(mean=0.5, std=0.0, snr_db=20.0, channel="mono", valid=False)
    assert result.mean == 0.5
    assert result.std == 0.0
    assert math.isnan(result.snr_db)
    assert result.channel == "mono"
    assert result.valid is False


def test_snr_result_valid_measurement_requires_a_finite_positive_ratio() -> None:
    with pytest.raises(ValueError):
        SNRResult(mean=10.0, std=1.0, snr_db=math.nan, channel="R", valid=True)
    with pytest.raises(ValueError):
        SNRResult(mean=0.0, std=1.0, snr_db=0.0, channel="R", valid=True)
    with pytest.raises(ValueError):
        SNRResult(mean=10.0, std=0.0, snr_db=0.0, channel="R", valid=True)


def test_snr_result_normalizes_numpy_scalars_and_strips_the_channel() -> None:
    result = SNRResult(
        mean=np.float64(10.0),
        std=np.float32(1.0),
        snr_db=np.float64(20.0),
        channel=" R ",
        valid=np.bool_(True),
    )
    assert isinstance(result.mean, float)
    assert isinstance(result.std, float)
    assert isinstance(result.snr_db, float)
    assert result.channel == "R"
    assert result.valid is True


# --------------------------------------------------------------------------- #
# Determinism and architecture
# --------------------------------------------------------------------------- #


def test_snr_patch_is_deterministic_and_leaves_the_input_untouched() -> None:
    rng = np.random.default_rng(16)
    image = (10.0 + rng.normal(0.0, 1.0, (16, 20, 3))).astype(np.float32)
    before = image.copy()
    roi = PatchROI("p", 1, 2, 12, 10)

    first = snr_patch(image, roi, channel="G")
    second = snr_patch(image, roi, channel="G")

    np.testing.assert_array_equal(image, before)
    assert image.dtype == np.float32
    assert first == second
    assert first.channel == "G"


def test_metrics_snr_exports_the_documented_names() -> None:
    import camsim.metrics.snr as module

    assert set(module.__all__) == {"SNRResult", "snr_channels", "snr_patch", "snr_temporal"}


def test_metrics_snr_does_not_import_simulator_modules() -> None:
    code = "\n".join(
        [
            "import sys",
            "import camsim.metrics.snr",
            "from camsim.metrics.snr import SNRResult, snr_channels, snr_patch, snr_temporal",
            "for name in (",
            "    'camsim.sensor', 'camsim.isp', 'camsim.scene', 'camsim.optics',",
            "):",
            "    assert name not in sys.modules, name",
        ]
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
