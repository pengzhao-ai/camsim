"""Tests for lens-shading correction (LSC) in the RAW linear domain.

Lens shading is the radial falloff of image-plane illuminance: natural
``cos^4`` vignetting from pupil foreshortening, plus pupil aberration and
microlens crosstalk, which make real parts fall off faster than the thin-lens
law. LSC multiplies each RAW sample by a position-dependent gain

``g(y, x) = F_ref / F(y, x)``

where ``F`` is a flat-field response measured through the same lens and sensor.
The gain is therefore a reciprocal illumination profile, not a colour-balance
operation, and it must be applied before AWB, demosaicing, and the colour
matrix so those later stages see a uniformly illuminated mosaic.

Two gain representations are tested: an analytic radial polynomial
``g(r) = c0 + c1 r^2 + c2 r^4 + ...`` with ``r`` normalised to one at the
farthest image corner, and a supplied 2-D map (full resolution, or a 2x2
per-CFA-site tile). ``calibrate_lsc`` derives a map from a flat field by
normalising to the brightest sample, which keeps every gain at or above one.
Black-level subtraction is performed when a :class:`RawFrame` is passed, because
a pedestal would otherwise compress the measured ratios and bias the map.

Failure artefacts guarded here: applying a map twice (over-corrected corners),
calibrating on display-encoded data (the reciprocal of a non-linear curve is
not a gain), leaving the ADC pedestal in the flat field (a map biased towards
one), a map with a different shape from the image (silent cropping or
broadcasting), a non-positive or non-finite gain (sign flips or infinities),
using the wrong radial centre or coefficient order (a tilt-shaped correction),
and treating a 2x2 per-CFA map without declaring the CFA layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from camsim.config.schema import LSCConfig
from camsim.core.frames import CFAPattern, RawFrame
from camsim.isp.blc import subtract_black_level
from camsim.isp.lsc import apply_lsc, calibrate_lsc


def _vignetted_flat(size: int = 64, depth: float = 0.3) -> np.ndarray:
    """Return ``1 - depth * r^2`` with ``r`` normalised to the farthest corner."""
    yy, xx = np.mgrid[0:size, 0:size]
    centre = (size - 1) / 2.0
    r2 = ((xx - centre) ** 2 + (yy - centre) ** 2) / (2.0 * centre**2)
    return (1.0 - depth * r2).astype(np.float32)


def _raw_frame(data: np.ndarray, black_level_dn: float = 64.0) -> RawFrame:
    """Build a minimal valid frame for the flat-field calibration path."""
    return RawFrame(
        data=data,
        cfa=CFAPattern.RGGB,
        bit_depth=10,
        black_level_dn=black_level_dn,
        exposure_s=0.01,
        analog_gain_db=0.0,
        digital_gain=1.0,
        mode="LCG",
        temperature_c=25.0,
        seed=1,
        frame_index=0,
        pixel_pitch_um=1.12,
        sensor_config_hash="lsc-test",
    )


def test_lsc_flat_field_corrects_vignetting() -> None:
    yy, xx = np.mgrid[0:64, 0:64]
    r2 = ((xx - 31.5) ** 2 + (yy - 31.5) ** 2) / (31.5**2 + 31.5**2)
    img = (1.0 - 0.3 * r2).astype(np.float32)
    out = apply_lsc(img, LSCConfig(mode="flat_field", map=1.0 / img), cfa=None)
    assert out.dtype == np.float32
    assert out.max() / out.min() < 1.02


def test_lsc_disabled_identity() -> None:
    img = _vignetted_flat(16)
    config = LSCConfig(enabled=False, mode="flat_field", map=1.0 / img)
    out = apply_lsc(img, config, cfa=None)
    np.testing.assert_array_equal(out, img)
    assert out.dtype == np.float32


def test_lsc_none_mode_identity() -> None:
    img = _vignetted_flat(16)
    out = apply_lsc(img, LSCConfig(mode="none"), cfa=None)
    np.testing.assert_array_equal(out, img)


def test_lsc_preserves_linearity() -> None:
    img = np.linspace(0.1, 1.0, 25, dtype=np.float32).reshape(5, 5)
    radial = LSCConfig(mode="radial", coefficients=[1.0, 0.4])
    np.testing.assert_allclose(
        apply_lsc(3.0 * img, radial),
        3.0 * apply_lsc(img, radial),
        rtol=1e-6,
    )
    flat_field = LSCConfig(mode="flat_field", map=np.full((5, 5), 1.5))
    np.testing.assert_allclose(
        apply_lsc(3.0 * img, flat_field),
        3.0 * apply_lsc(img, flat_field),
        rtol=1e-6,
    )


def test_lsc_radial_uses_the_center_and_coefficients() -> None:
    img = np.ones((5, 5), dtype=np.float32)
    out = apply_lsc(img, LSCConfig(mode="radial", coefficients=[1.0, 0.5]))
    assert out[2, 2] == pytest.approx(1.0)
    assert out[0, 0] == pytest.approx(1.5)  # r^2 = 1 at the farthest corner
    assert out[0, 2] == pytest.approx(1.25)  # r^2 = 4 / 8

    # center is (row, column): putting it at the top-right corner makes that
    # corner the unit-gain point and the opposite corner the farthest one.
    shifted = apply_lsc(
        img,
        LSCConfig(mode="radial", coefficients=[1.0, 0.5], center=(0.0, 4.0)),
    )
    assert shifted[0, 4] == pytest.approx(1.0)
    assert shifted[4, 0] == pytest.approx(1.5)


def test_lsc_radial_coefficients_are_ascending_powers_of_r_squared() -> None:
    img = np.ones((5, 5), dtype=np.float32)
    out = apply_lsc(img, LSCConfig(mode="radial", coefficients=[1.0, 0.0, 0.5]))
    assert out[2, 2] == pytest.approx(1.0)
    assert out[0, 2] == pytest.approx(1.0 + 0.5 * 0.25)  # r^2 = 0.5, r^4 = 0.25
    assert out[0, 0] == pytest.approx(1.5)  # r^2 = r^4 = 1


def test_lsc_radial_rejects_non_positive_gain() -> None:
    img = np.ones((5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        apply_lsc(img, LSCConfig(mode="radial", coefficients=[1.0, -2.0]))


def test_lsc_radial_validates_coefficients_after_mutation() -> None:
    img = np.ones((5, 5), dtype=np.float32)
    config = LSCConfig(mode="radial", coefficients=[1.0])
    config.coefficients = []
    with pytest.raises(ValueError, match="coefficient"):
        apply_lsc(img, config)

    config.coefficients = [1.0, float("nan")]
    with pytest.raises(ValueError, match="finite"):
        apply_lsc(img, config)


def test_lsc_flat_field_requires_a_matching_shape() -> None:
    img = np.ones((5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        apply_lsc(img, LSCConfig(mode="flat_field", map=np.ones((3, 3))))


def test_lsc_flat_field_rejects_non_positive_or_non_finite_maps() -> None:
    img = np.ones((5, 5), dtype=np.float32)
    config = LSCConfig(mode="flat_field", map=np.ones((5, 5)))
    config.map[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive"):
        apply_lsc(img, config)

    config.map[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        apply_lsc(img, config)


def test_lsc_flat_field_accepts_a_2x2_per_cfa_tile() -> None:
    img = np.ones((4, 4), dtype=np.float32)
    config = LSCConfig(mode="flat_field", map=np.array([[2.0, 1.0], [1.0, 1.0]]))
    out = apply_lsc(img, config, cfa=CFAPattern.RGGB)
    assert out[0, 0] == pytest.approx(2.0)
    assert out[0, 1] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(1.0)
    assert out[1, 1] == pytest.approx(1.0)
    assert out[2, 2] == pytest.approx(2.0)

    with pytest.raises(ValueError, match="cfa"):
        apply_lsc(img, config, cfa=None)
    with pytest.raises(ValueError, match="cfa"):
        apply_lsc(img, config, cfa="XYZ")


def test_lsc_validates_argument_types_and_images() -> None:
    img = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="LSCConfig"):
        apply_lsc(img, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2-D"):
        apply_lsc(np.ones((4, 4, 1), dtype=np.float32), LSCConfig(mode="none"))
    with pytest.raises(ValueError, match="finite"):
        apply_lsc(np.array([[np.nan, 1.0]]), LSCConfig(mode="none"))


def test_lsc_accepts_integer_images_and_returns_float32() -> None:
    img = np.full((4, 4), 100, dtype=np.uint16)
    out = apply_lsc(img, LSCConfig(mode="flat_field", map=np.full((4, 4), 2.0)))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.full((4, 4), 200.0, dtype=np.float32))


def test_calibrate_lsc_normalizes_a_flat_field() -> None:
    flat = _vignetted_flat(64)
    gain = calibrate_lsc(flat)
    assert gain.dtype == np.float32
    assert gain.shape == flat.shape
    assert gain.min() == pytest.approx(1.0)
    assert gain.max() > 1.0

    corrected = apply_lsc(flat, LSCConfig(mode="flat_field", map=gain))
    assert corrected.max() / corrected.min() < 1.02


def test_calibrate_lsc_accepts_a_raw_frame_with_a_black_pedestal() -> None:
    flat = _vignetted_flat(32)
    data = (64 + np.rint(800.0 * flat)).astype(np.uint16)
    raw = _raw_frame(data)
    gain = calibrate_lsc(raw)
    corrected = apply_lsc(
        subtract_black_level(raw.data, raw.black_level_dn, bit_depth=raw.bit_depth),
        LSCConfig(mode="flat_field", map=gain),
    )
    assert corrected.max() / corrected.min() < 1.02


def test_calibrate_lsc_rejects_invalid_flat_fields() -> None:
    with pytest.raises(ValueError, match="2-D"):
        calibrate_lsc(np.ones(4, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        calibrate_lsc(np.array([[1.0, np.nan], [1.0, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        calibrate_lsc(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        calibrate_lsc(np.full((4, 4), -1.0, dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        calibrate_lsc(_raw_frame(np.full((4, 4), 64, dtype=np.uint16)))
