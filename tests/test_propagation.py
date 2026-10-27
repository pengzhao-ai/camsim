"""Tests for PSF propagation and block-mean downsampling.

Propagation is where a scene cube becomes a sensor-plane image. A point-spread
function redistributes irradiance without creating or destroying it: after
convolution with a normalized kernel the image average must be unchanged, which
is the physical statement that a lens does not invent light. The discretization
must not violate that statement either. Zero padding silently absorbs energy at
the frame border, so the implementation extends the image by reflection; a
constant field therefore stays exactly constant right up to the edge.

Oversampling is the other half of the sampled-imaging chain. Rendering at N x N
samples per sensor pixel and averaging each block (a box mean) models the pixel
aperture's spatial integration and suppresses aliasing. The block mean must
preserve per-unit-area irradiance, and the block factor must divide the array
exactly because a partial block would represent a fractional pixel area.

Failure artefacts guarded here: an unnormalized kernel changes brightness; a
non-reflecting boundary darkens or brightens borders; a mean that sums rather
than averages scales the image by the block area; non-divisible shapes hide an
incorrect pixel-pitch conversion; and a non-finite or negative kernel is not a
radiometric PSF. The slanted-edge angle test additionally guards the geometric
convention shared by the target generator and the edge estimator: a mirrored
or transposed convention estimates the wrong sign.
"""

from __future__ import annotations

import numpy as np
import pytest

from camsim.optics.propagation import apply_psf, box_downsample
from camsim.optics.psf import gaussian_psf


def test_box_downsample_conserves_mean() -> None:
    x = np.ones((8, 8, 3))

    assert box_downsample(x, 2).mean() == pytest.approx(1.0)


def test_energy_conservation_through_psf_and_downsample() -> None:
    x = np.zeros((64, 64, 3))
    x[30:34, 30:34, :] = 1.0

    y = box_downsample(apply_psf(x, gaussian_psf(1.0, size=11)), 4)

    assert y.sum() * 16 == pytest.approx(x.sum(), rel=1e-2)


def test_slanted_edge_angle_is_estimated() -> None:
    from camsim.scene.targets import estimate_edge_angle, slanted_edge_target

    img = slanted_edge_target((256, 256), angle_deg=5.0)

    assert estimate_edge_angle(img) == pytest.approx(5.0, abs=0.5)


def test_box_downsample_is_a_block_mean() -> None:
    x = np.arange(16.0).reshape(4, 4)

    np.testing.assert_allclose(box_downsample(x, 2), [[2.5, 4.5], [10.5, 12.5]])


def test_box_downsample_preserves_channel_axis() -> None:
    x = np.zeros((8, 8, 2))
    x[..., 0] = 1.0
    x[..., 1] = 3.0

    y = box_downsample(x, 2)

    assert y.shape == (4, 4, 2)
    np.testing.assert_allclose(y[..., 0], 1.0)
    np.testing.assert_allclose(y[..., 1], 3.0)


def test_box_downsample_requires_exact_divisibility() -> None:
    with pytest.raises(ValueError, match="divide"):
        box_downsample(np.ones((6, 8)), 4)


@pytest.mark.parametrize("factor", [0, -2, 1.5, True])
def test_box_downsample_rejects_invalid_factor(factor: object) -> None:
    with pytest.raises(ValueError, match="factor"):
        box_downsample(np.ones((8, 8)), factor)  # type: ignore[arg-type]


@pytest.mark.parametrize("size", [8, 9])
def test_apply_psf_preserves_a_constant_field_to_the_border(size: int) -> None:
    image = np.full((32, 24, 2), 3.0)

    blurred = apply_psf(image, gaussian_psf(1.0, size=size))

    # Reflection padding is the point: a constant field remains constant even
    # at the frame border, so no energy is absorbed by the boundary. Even
    # kernels exercise the half-pixel origin path as well.
    np.testing.assert_allclose(blurred, 3.0, rtol=1e-12)


def test_apply_psf_preserves_2d_input_shape() -> None:
    image = np.zeros((16, 16))
    image[8, 8] = 1.0

    blurred = apply_psf(image, gaussian_psf(1.0, size=5))

    assert blurred.shape == (16, 16)
    assert blurred[8, 8] > 0.0


def test_apply_psf_normalizes_the_kernel() -> None:
    rng = np.random.default_rng(4)
    image = rng.random((24, 24, 3))
    kernel = gaussian_psf(1.0, size=7)

    unit = apply_psf(image, kernel)
    scaled = apply_psf(image, 5.0 * kernel)

    np.testing.assert_allclose(scaled, unit, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("size", [4, 5, 6, 7])
def test_apply_psf_with_a_delta_kernel_is_the_identity(size: int) -> None:
    rng = np.random.default_rng(5)
    image = rng.random((12, 12, 2))
    delta = np.zeros((size, size))
    # The documented origin is index size // 2 in each axis. This must hold for
    # even sizes too: an off-by-one padding convention shifts even kernels by
    # one pixel down and right.
    delta[size // 2, size // 2] = 1.0

    np.testing.assert_allclose(apply_psf(image, delta), image, atol=1e-12)


def test_apply_psf_conserves_border_energy_within_one_percent_for_asymmetric_kernels() -> None:
    rng = np.random.default_rng(6)
    image = rng.random((16, 16))
    kernel = rng.random((7, 7))
    kernel /= kernel.sum()

    blurred = apply_psf(image, kernel)

    # Reflection is exact for symmetric kernels. An asymmetric kernel wraps
    # some mass across the border, so the guarantee is the documented ~1 %
    # energy budget rather than floating-point precision.
    assert blurred.sum() == pytest.approx(image.sum(), rel=1e-2)


def test_apply_psf_keeps_a_non_negative_cube_non_negative() -> None:
    image = np.zeros((32, 32, 3))
    image[16, 16, 1] = 1.0

    blurred = apply_psf(image, gaussian_psf(1.0, size=9))

    # FFT convolution can leave ~1e-17 negative round-off; a radiometric cube
    # must remain non-negative so SpectralIrradiance validation succeeds.
    assert blurred.min() >= 0.0


def test_apply_psf_accepts_a_kernel_wider_than_the_image() -> None:
    image = np.zeros((4, 5))
    image[2, 2] = 1.0

    blurred = apply_psf(image, gaussian_psf(3.0, size=21))

    assert blurred.shape == image.shape
    # Reflection is repeated as needed, so even this extreme case keeps unit
    # energy rather than losing the kernel mass outside the frame.
    assert blurred.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (np.ones(5), "2-D"),
        (np.array([[-1.0, 1.0], [0.0, 0.0]]), "non-negative"),
        (np.zeros((3, 3)), "positive"),
        (np.full((3, 3), np.nan), "finite"),
    ],
)
def test_apply_psf_rejects_invalid_kernels(kernel: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        apply_psf(np.ones((8, 8)), kernel)


def test_apply_psf_rejects_non_finite_cube() -> None:
    cube = np.ones((8, 8, 2))
    cube[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        apply_psf(cube, gaussian_psf(1.0, size=3))


def test_apply_psf_rejects_other_than_2_or_3_dimensions() -> None:
    with pytest.raises(ValueError, match="2-D or 3-D"):
        apply_psf(np.ones(8), gaussian_psf(1.0, size=3))
