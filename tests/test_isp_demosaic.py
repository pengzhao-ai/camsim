"""Tests for CFA demosaicing: bilinear, Malvar-He-Cutler, and the RCCC baseline.

A colour-filter-array sensor measures one spectral band per photosite, so a
Bayer mosaic carries only one of R, G, or B at each pixel and the ISP must
interpolate the two missing channels. Demosaicing therefore happens in linear
light, after black-level correction and white balance, and before every stage
that expects full-resolution linear RGB.

Bilinear interpolation is the cheap reference: each masked colour plane is
convolved with a 3x3 all-pass kernel, so a missing sample is the average of its
same-colour neighbours. It is exact for smooth fields but zippers along edges
and fringes colour where the independently interpolated channels disagree.
Malvar, He, and Cutler (2004) replace those kernels with 5x5
gradient-corrected filters that borrow high-frequency detail from the measured
green plane, which markedly reduces zippering and false colour at about twice
the arithmetic. Both are validated against ``colour-demosaicing`` across all
four Bayer orientations, odd frame sizes, uint16 input, and negative values.

The full-frame reference comparison deliberately includes the frame border.
``colour-demosaicing`` extends masked planes with symmetric reflection, which
double-counts same-colour samples at the edge; the implementations here mirror
that boundary mode exactly, so the frame-border artefacts of the reference are
reproduced rather than hidden. Constant-scene (DC) tests therefore check the
interior, where the kernels have full support, and the reference comparison
pins the border behaviour itself.

The RCCC baseline is a documented shape/linearity/smoke reconstruction only:
it bilinearly interpolates the sparse red plane, keeps the measured clear (C)
plane, fills C at red sites from the four neighbouring clear samples, and
returns ``(R, C, C)``. It is not colour accurate and is not intended for image
quality work.

Failure artefacts guarded here: mixing CFA colours, treating a mono or RCCC
mosaic as Bayer, returning the input buffer (aliasing a later stage's write),
mutating the input, losing the float32 linear contract, clipping negative
linear excursions, and accepting NaN, complex, bool, empty, or non-2-D data.
All tests are deterministic: fixed generator seeds, no global RNG.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from camsim.core.frames import CFAPattern
from camsim.isp.demosaic import (
    demosaic,
    demosaic_bilinear,
    demosaic_mhc,
    demosaic_rccc_baseline,
)
from camsim.sensor.cfa import cfa_masks, mosaic_from_channels

BAYER_PATTERNS = (CFAPattern.RGGB, CFAPattern.BGGR, CFAPattern.GRBG, CFAPattern.GBRG)


def _mosaic_from_rgb(rgb, cfa=CFAPattern.RGGB):
    from camsim.sensor.cfa import mosaic_from_channels

    return mosaic_from_channels({"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}, cfa)


def test_bilinear_matches_reference_on_smooth_image():
    from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

    rgb = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    raw = _mosaic_from_rgb(rgb)
    ours = demosaic_bilinear(raw, CFAPattern.RGGB)
    ref = demosaicing_CFA_Bayer_bilinear(raw, "RGGB")
    np.testing.assert_allclose(ours, ref, atol=1e-5)


def test_mhc_matches_reference_on_smooth_image():
    from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004

    rgb = np.random.default_rng(1).random((32, 32, 3)).astype(np.float32)
    raw = _mosaic_from_rgb(rgb)
    ours = demosaic_mhc(raw, CFAPattern.RGGB)
    ref = demosaicing_CFA_Bayer_Malvar2004(raw, "RGGB")
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def _random_rgb(
    seed: int,
    shape: tuple[int, int] = (24, 26),
) -> NDArray[np.float32]:
    """Return a deterministic smooth-ish random RGB image in ``[0, 1)``."""
    return np.random.default_rng(seed).random((*shape, 3)).astype(np.float32)


def _rccc_mosaic(
    red: NDArray[np.float32],
    clear: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Interleave red and clear planes into an RCCC mosaic."""
    return mosaic_from_channels({"R": red, "C": clear}, CFAPattern.RCCC)


@pytest.mark.parametrize("pattern", BAYER_PATTERNS)
def test_bilinear_matches_reference_for_every_bayer_orientation(
    pattern: CFAPattern,
) -> None:
    from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

    rgb = _random_rgb(10)
    raw = _mosaic_from_rgb(rgb, pattern)
    # Dispatch through the public entry point as well as the direct helper.
    ours = demosaic(raw, pattern, "bilinear")
    ref = demosaicing_CFA_Bayer_bilinear(raw, pattern.value)
    np.testing.assert_allclose(ours, ref, atol=1e-5)
    np.testing.assert_array_equal(demosaic_bilinear(raw, pattern), ours)


@pytest.mark.parametrize("pattern", BAYER_PATTERNS)
def test_mhc_matches_reference_for_every_bayer_orientation(
    pattern: CFAPattern,
) -> None:
    from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004

    rgb = _random_rgb(11)
    raw = _mosaic_from_rgb(rgb, pattern)
    ours = demosaic(raw, pattern, "mhc")
    ref = demosaicing_CFA_Bayer_Malvar2004(raw, pattern.value)
    np.testing.assert_allclose(ours, ref, atol=1e-4)
    np.testing.assert_array_equal(demosaic_mhc(raw, pattern), ours)


def test_reference_comparison_covers_odd_frame_sizes() -> None:
    from colour_demosaicing import (
        demosaicing_CFA_Bayer_bilinear,
        demosaicing_CFA_Bayer_Malvar2004,
    )

    # Odd height and width crop the 2x2 tile, exercising every boundary edge.
    rgb = _random_rgb(12, shape=(23, 31))
    raw = _mosaic_from_rgb(rgb, CFAPattern.GBRG)
    np.testing.assert_allclose(
        demosaic_bilinear(raw, CFAPattern.GBRG),
        demosaicing_CFA_Bayer_bilinear(raw, "GBRG"),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        demosaic_mhc(raw, CFAPattern.GBRG),
        demosaicing_CFA_Bayer_Malvar2004(raw, "GBRG"),
        atol=1e-4,
    )


def test_uint16_input_is_accepted() -> None:
    from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

    rgb = _random_rgb(13)
    codes = np.rint(_mosaic_from_rgb(rgb) * 1023.0).astype(np.uint16)
    ours = demosaic_bilinear(codes, CFAPattern.RGGB)
    ref = demosaicing_CFA_Bayer_bilinear(codes, "RGGB")
    assert ours.dtype == np.float32
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-3)


def test_negative_input_is_preserved_without_clipping() -> None:
    from colour_demosaicing import (
        demosaicing_CFA_Bayer_bilinear,
        demosaicing_CFA_Bayer_Malvar2004,
    )

    rgb = _random_rgb(14) * 2.0 - 1.0
    raw = _mosaic_from_rgb(rgb, CFAPattern.BGGR)
    bilinear = demosaic_bilinear(raw, CFAPattern.BGGR)
    mhc = demosaic_mhc(raw, CFAPattern.BGGR)
    np.testing.assert_allclose(bilinear, demosaicing_CFA_Bayer_bilinear(raw, "BGGR"), atol=1e-5)
    np.testing.assert_allclose(mhc, demosaicing_CFA_Bayer_Malvar2004(raw, "BGGR"), atol=1e-4)
    # A linear stage must not clip: negative excursions survive.
    assert bilinear.min() < 0.0
    assert mhc.min() < 0.0


@pytest.mark.parametrize("method", ["bilinear", "mhc"])
def test_output_is_float32_and_c_contiguous(method: str) -> None:
    raw = _mosaic_from_rgb(_random_rgb(15))
    # Fortran order is a valid array-like input but not a C-contiguous buffer,
    # so a second buffer must be allocated for the result.
    fortran_input = np.asfortranarray(raw)
    assert not fortran_input.flags["C_CONTIGUOUS"]
    out = demosaic(fortran_input, CFAPattern.RGGB, method)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert out.shape == (*raw.shape, 3)
    np.testing.assert_array_equal(out, demosaic(raw, CFAPattern.RGGB, method))


def test_rccc_output_is_float32_and_c_contiguous() -> None:
    rng = np.random.default_rng(16)
    raw = _rccc_mosaic(
        rng.random((10, 12), dtype=np.float32),
        rng.random((10, 12), dtype=np.float32),
    )
    out = demosaic(raw, CFAPattern.RCCC, "rccc_baseline")
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert out.shape == (10, 12, 3)


def test_input_is_not_mutated() -> None:
    rgb = _random_rgb(17)
    raw = _mosaic_from_rgb(rgb)
    raw_original = raw.copy()
    demosaic_bilinear(raw, CFAPattern.RGGB)
    demosaic_mhc(raw, CFAPattern.RGGB)
    demosaic(raw, CFAPattern.RGGB, "mhc")
    np.testing.assert_array_equal(raw, raw_original)

    rng = np.random.default_rng(18)
    rccc_raw = _rccc_mosaic(
        rng.random((9, 11), dtype=np.float32),
        rng.random((9, 11), dtype=np.float32),
    )
    rccc_original = rccc_raw.copy()
    demosaic_rccc_baseline(rccc_raw)
    np.testing.assert_array_equal(rccc_raw, rccc_original)


def test_mono_bypass() -> None:
    raw = np.random.default_rng(19).random((7, 9), dtype=np.float32)
    for method in ("bilinear", "mhc", "rccc_baseline"):
        out = demosaic(raw, CFAPattern.MONO, method)
        # Mono is a single 2-D channel, not an RGB replication at this stage.
        assert out.shape == raw.shape
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]
        assert out is not raw
        np.testing.assert_array_equal(out, raw)

    for helper in (demosaic_bilinear, demosaic_mhc):
        out = helper(raw, CFAPattern.MONO)
        assert out.shape == raw.shape
        assert out is not raw
        np.testing.assert_array_equal(out, raw)


@pytest.mark.parametrize("pattern", BAYER_PATTERNS)
@pytest.mark.parametrize("method", ["bilinear", "mhc"])
def test_demosaic_preserves_dc_in_the_interior(
    pattern: CFAPattern,
    method: str,
) -> None:
    # A constant scene must reconstruct as the same constant colour. The
    # reference's symmetric boundary extension double-counts same-colour sites
    # at the frame edge (and MHC mixes colours there), so the meaningful DC
    # check is the interior where every kernel has full support.
    rgb = np.empty((16, 18, 3), dtype=np.float32)
    rgb[...] = np.array([0.2, 0.45, 0.7], dtype=np.float32)
    raw = _mosaic_from_rgb(rgb, pattern)
    out = demosaic(raw, pattern, method)
    np.testing.assert_allclose(out[2:-2, 2:-2], rgb[2:-2, 2:-2], atol=1e-6)


def test_rccc_baseline_preserves_dc_everywhere() -> None:
    raw = _rccc_mosaic(
        np.full((8, 10), 0.25, dtype=np.float32),
        np.full((8, 10), 0.6, dtype=np.float32),
    )
    out = demosaic_rccc_baseline(raw)
    np.testing.assert_allclose(out[..., 0], 0.25, atol=1e-6)
    np.testing.assert_allclose(out[..., 1], 0.6, atol=1e-6)
    np.testing.assert_allclose(out[..., 2], 0.6, atol=1e-6)


def test_rccc_baseline_shape_and_measured_sites() -> None:
    rng = np.random.default_rng(20)
    red = rng.random((10, 12), dtype=np.float32)
    clear = rng.random((10, 12), dtype=np.float32)
    raw = _rccc_mosaic(red, clear)
    out = demosaic_rccc_baseline(raw)
    masks = cfa_masks(CFAPattern.RCCC, raw.shape)

    assert out.shape == (*raw.shape, 3)
    # C is the documented luma placeholder for both G and B.
    np.testing.assert_array_equal(out[..., 1], out[..., 2])
    # Measured samples are preserved; only the missing sites are interpolated.
    np.testing.assert_allclose(out[..., 0][masks["R"]], raw[masks["R"]], atol=1e-6)
    np.testing.assert_array_equal(out[..., 1][masks["C"]], raw[masks["C"]])
    # The dispatcher and the direct helper agree bit for bit.
    np.testing.assert_array_equal(demosaic(raw, CFAPattern.RCCC, "rccc_baseline"), out)


def test_rccc_baseline_fills_clear_from_four_neighbours() -> None:
    rng = np.random.default_rng(21)
    red = rng.random((10, 12), dtype=np.float32)
    clear = rng.random((10, 12), dtype=np.float32)
    raw = _rccc_mosaic(red, clear)
    out = demosaic_rccc_baseline(raw)

    # (2, 2) is a red site; its four cardinal neighbours are clear samples.
    expected = np.mean(raw[[1, 3, 2, 2], [2, 2, 1, 3]])
    np.testing.assert_allclose(out[2, 2, 1], expected, rtol=1e-6)
    np.testing.assert_allclose(out[2, 2, 2], expected, rtol=1e-6)


@pytest.mark.parametrize("method", ["bilinear", "mhc"])
def test_bayer_demosaic_is_linear(method: str) -> None:
    raw = _mosaic_from_rgb(_random_rgb(22), CFAPattern.GRBG)
    scale = np.float32(0.3)
    np.testing.assert_allclose(
        demosaic(scale * raw, CFAPattern.GRBG, method),
        scale * demosaic(raw, CFAPattern.GRBG, method),
        rtol=1e-6,
        atol=1e-7,
    )


def test_rccc_baseline_is_linear() -> None:
    rng = np.random.default_rng(23)
    raw = _rccc_mosaic(
        rng.random((8, 10), dtype=np.float32),
        rng.random((8, 10), dtype=np.float32),
    )
    scale = np.float32(0.3)
    np.testing.assert_allclose(
        demosaic_rccc_baseline(scale * raw),
        scale * demosaic_rccc_baseline(raw),
        rtol=1e-6,
        atol=1e-7,
    )


def test_mhc_differs_from_bilinear_at_edges() -> None:
    # A hard vertical edge is where MHC's gradient correction must matter; an
    # accidental alias of bilinear would make this test fail.
    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    rgb[:, 8:] = 1.0
    raw = _mosaic_from_rgb(rgb)
    bilinear = demosaic_bilinear(raw, CFAPattern.RGGB)
    mhc = demosaic_mhc(raw, CFAPattern.RGGB)
    assert np.max(np.abs(mhc - bilinear)) > 1e-3


def test_string_cfa_value_is_accepted() -> None:
    raw = _mosaic_from_rgb(_random_rgb(24), CFAPattern.GBRG)
    np.testing.assert_array_equal(
        demosaic(raw, "GBRG", "mhc"),
        demosaic(raw, CFAPattern.GBRG, "mhc"),
    )


_INVALID_MOSAICS = (
    pytest.param(np.zeros((4, 4, 1), dtype=np.float32), id="3d"),
    pytest.param(np.zeros(4, dtype=np.float32), id="1d"),
    pytest.param(np.zeros((0, 0), dtype=np.float32), id="empty"),
    pytest.param(np.zeros((4, 4), dtype=np.bool_), id="bool"),
    pytest.param(np.zeros((4, 4), dtype=np.complex64), id="complex"),
    pytest.param(np.full((4, 4), np.nan, dtype=np.float32), id="nan"),
    pytest.param(np.full((4, 4), np.inf, dtype=np.float32), id="inf"),
    pytest.param(np.full((4, 4), 1e300, dtype=np.float64), id="float32-overflow"),
)


@pytest.mark.parametrize("invalid", _INVALID_MOSAICS)
def test_invalid_mosaic_raises(invalid: NDArray[np.generic]) -> None:
    with pytest.raises(ValueError):
        demosaic_bilinear(invalid, CFAPattern.RGGB)
    with pytest.raises(ValueError):
        demosaic_mhc(invalid, CFAPattern.RGGB)
    with pytest.raises(ValueError):
        demosaic(invalid, CFAPattern.RGGB, "mhc")
    with pytest.raises(ValueError):
        demosaic_rccc_baseline(invalid)


def test_ragged_input_raises() -> None:
    with pytest.raises(ValueError):
        demosaic_bilinear([[0.0, 1.0], [2.0]], CFAPattern.RGGB)


def test_plain_array_like_input_is_accepted() -> None:
    out = demosaic_bilinear([[0.2, 0.4], [0.4, 0.6]], CFAPattern.RGGB)
    assert out.shape == (2, 2, 3)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("cfa", ["RGGBB", "rggb", "", 3, None])
def test_invalid_cfa_raises(cfa: object) -> None:
    raw = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        demosaic_bilinear(raw, cfa)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        demosaic_mhc(raw, cfa)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        demosaic(raw, cfa, "bilinear")  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["nearest", "", "BILINEAR", None, 1])
def test_invalid_method_raises(method: object) -> None:
    raw = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        demosaic(raw, CFAPattern.RGGB, method)  # type: ignore[arg-type]


def test_method_is_validated_before_mono_bypass() -> None:
    raw = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        demosaic(raw, CFAPattern.MONO, "nearest")


def test_rccc_requires_the_rccc_baseline_method() -> None:
    rng = np.random.default_rng(25)
    raw = _rccc_mosaic(
        rng.random((6, 8), dtype=np.float32),
        rng.random((6, 8), dtype=np.float32),
    )
    with pytest.raises(ValueError):
        demosaic(raw, CFAPattern.RCCC, "bilinear")
    with pytest.raises(ValueError):
        demosaic(raw, CFAPattern.RCCC, "mhc")
    with pytest.raises(ValueError):
        demosaic_bilinear(raw, CFAPattern.RCCC)
    with pytest.raises(ValueError):
        demosaic_mhc(raw, CFAPattern.RCCC)


def test_rccc_baseline_rejects_bayer_input() -> None:
    raw = _mosaic_from_rgb(_random_rgb(26))
    with pytest.raises(ValueError):
        demosaic(raw, CFAPattern.RGGB, "rccc_baseline")


def test_rccc_baseline_rejects_a_missing_clear_sample() -> None:
    # A 1x1 RCCC crop contains only the red site, so no clear sample exists.
    with pytest.raises(ValueError):
        demosaic_rccc_baseline(np.zeros((1, 1), dtype=np.float32))


def test_demosaic_is_deterministic() -> None:
    raw = _mosaic_from_rgb(_random_rgb(27))
    for method in ("bilinear", "mhc"):
        np.testing.assert_array_equal(
            demosaic(raw, CFAPattern.RGGB, method),
            demosaic(raw, CFAPattern.RGGB, method),
        )
