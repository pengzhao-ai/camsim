"""Tests for mask-guided defective-pixel correction in the RAW domain.

Dead, hot, and stuck pixels are flagged in ``RawFrame.defect_mask``. Their
values carry no useful scene information, so the ISP replaces them from
neighbouring pixels that saw the same scene through the same colour filter.
Interpolating across CFA colours would mix spectral bands (for example red into
green), producing colour artefacts and a white-balance bias that later stages
cannot undo. Phase-detection autofocus (PDAF) pixels are not defects: their
signal is deliberately attenuated, so the corrector must leave them in place
and must not use them as donors.

The tests pin four properties: same-CFA-colour donor selection, exact
preservation of every unaffected sample, the mean-versus-median behaviour of
the two supported methods, and the rule that PDAF sites are neither corrected
nor used as references. They also cover edges and defect clusters, where the
available donor set is smaller or empty, and the float32 linear-space output
contract (codes are passed to black-level correction without gain or gamma).

Failure artefacts guarded here: interpolating across CFA colours (colour
fringes and cast), using a defective or PDAF neighbour as a donor (the
correction reproduces a bad value), correcting a PDAF site (destroying the
autofocus signal), changing pixels that were not flagged (violating the
deterministic pass-through contract), and accepting a mutated mask whose shape
or dtype no longer matches the frame.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from camsim.config.schema import DPCConfig
from camsim.core.frames import CFAPattern, RawFrame
from camsim.isp.dpc import correct_defects
from camsim.sensor.cfa import cfa_masks


def _raw_frame(
    data: NDArray[np.uint16],
    *,
    cfa: CFAPattern = CFAPattern.RGGB,
    defect_mask: NDArray[np.bool_] | None = None,
    pdaf_mask: NDArray[np.bool_] | None = None,
    bit_depth: int = 10,
    black_level_dn: float = 64.0,
) -> RawFrame:
    """Build a minimal valid frame carrying the masks under test."""
    return RawFrame(
        data=data,
        cfa=cfa,
        bit_depth=bit_depth,
        black_level_dn=black_level_dn,
        exposure_s=0.01,
        analog_gain_db=0.0,
        digital_gain=1.0,
        mode="LCG",
        temperature_c=25.0,
        seed=1,
        frame_index=0,
        pixel_pitch_um=1.12,
        defect_mask=defect_mask,
        pdaf_mask=pdaf_mask,
        sensor_config_hash="dpc-test",
    )


def _mosaic(
    values: dict[str, int],
    shape: tuple[int, int] = (6, 6),
    cfa: CFAPattern = CFAPattern.RGGB,
) -> NDArray[np.uint16]:
    """Fill each CFA colour plane with a distinct constant code."""
    masks = cfa_masks(cfa, shape)
    data = np.zeros(shape, dtype=np.uint16)
    for colour, value in values.items():
        data[masks[colour]] = value
    return data


def _mask_like(data: NDArray[np.uint16], **sites: bool) -> NDArray[np.bool_]:
    """Return an all-false mask with the named sites set."""
    mask = np.zeros(data.shape, dtype=np.bool_)
    for site, flagged in sites.items():
        row, column = (int(part) for part in site.split("_"))
        mask[row, column] = flagged
    return mask


def test_dpc_interpolates_defects() -> None:
    data = np.full((6, 6), 100, dtype=np.uint16)
    data[2, 2] = 0
    defect = _mask_like(data, **{"2_2": True})
    out = correct_defects(_raw_frame(data, defect_mask=defect), DPCConfig())
    assert out.dtype == np.float32
    assert out[2, 2] == pytest.approx(100.0)
    expected = data.astype(np.float32)
    expected[2, 2] = 100.0
    np.testing.assert_array_equal(out, expected)


def test_dpc_never_mixes_cfa_colours() -> None:
    data = _mosaic({"R": 100, "G": 200, "B": 300})
    data[2, 2] = 0  # red site
    data[2, 3] = 0  # green site
    data[3, 3] = 0  # blue site
    defect = _mask_like(data, **{"2_2": True, "2_3": True, "3_3": True})
    out = correct_defects(_raw_frame(data, defect_mask=defect), DPCConfig())
    assert out[2, 2] == pytest.approx(100.0)
    assert out[2, 3] == pytest.approx(200.0)
    assert out[3, 3] == pytest.approx(300.0)


@pytest.mark.parametrize(
    "cfa",
    [CFAPattern.RGGB, CFAPattern.BGGR, CFAPattern.GRBG, CFAPattern.GBRG],
)
def test_dpc_respects_every_bayer_tile_orientation(cfa: CFAPattern) -> None:
    values = {"R": 100, "G": 200, "B": 300}
    data = _mosaic(values, shape=(6, 6), cfa=cfa)
    masks = cfa_masks(cfa, (6, 6))
    defect = np.zeros(data.shape, dtype=np.bool_)
    expected = data.astype(np.float32)
    for colour, value in values.items():
        sites = np.argwhere(masks[colour])
        row, column = (int(coordinate) for coordinate in sites[len(sites) // 2])
        data[row, column] = 0
        defect[row, column] = True
        expected[row, column] = value
    out = correct_defects(_raw_frame(data, cfa=cfa, defect_mask=defect), DPCConfig())
    np.testing.assert_array_equal(out, expected)


def test_dpc_preserves_unaffected_pixels_exactly() -> None:
    rng = np.random.default_rng(1234)
    data = rng.integers(1, 1024, size=(12, 12), dtype=np.uint16)
    data[5, 7] = 0
    defect = np.zeros(data.shape, dtype=np.bool_)
    defect[5, 7] = True
    out = correct_defects(_raw_frame(data, defect_mask=defect), DPCConfig())
    keep = ~defect
    np.testing.assert_array_equal(out[keep], data[keep].astype(np.float32))
    assert out[5, 7] > 0.0


def test_dpc_neighbor_uses_the_mean_and_median_uses_the_median() -> None:
    masks = cfa_masks(CFAPattern.RGGB, (6, 6))
    data = np.zeros((6, 6), dtype=np.uint16)
    data[masks["R"]] = 10
    data[0, 4] = 1000  # same-colour outlier among the red donors
    data[2, 2] = 0
    defect = _mask_like(data, **{"2_2": True})
    raw = _raw_frame(data, defect_mask=defect)

    neighbor = correct_defects(raw, DPCConfig(method="neighbor"))
    median = correct_defects(raw, DPCConfig(method="median"))

    # Eight red donors: seven at 10 DN and one at 1000 DN.
    assert neighbor[2, 2] == pytest.approx((7 * 10 + 1000) / 8)
    assert median[2, 2] == pytest.approx(10.0)


def test_dpc_does_not_use_pdaf_pixels_as_donors() -> None:
    data = _mosaic({"R": 100, "G": 200, "B": 300})
    data[0, 2] = 500  # PDAF red site inside the defect's donor window
    data[2, 2] = 0
    defect = _mask_like(data, **{"2_2": True})
    pdaf = _mask_like(data, **{"0_2": True})
    out = correct_defects(_raw_frame(data, defect_mask=defect, pdaf_mask=pdaf), DPCConfig())
    # Including the attenuated PDAF value would raise the mean to 150 DN.
    assert out[2, 2] == pytest.approx(100.0)


def test_dpc_leaves_pdaf_pixels_alone_even_when_flagged_as_defects() -> None:
    data = _mosaic({"R": 100, "G": 200, "B": 300})
    data[0, 2] = 500
    defect = _mask_like(data, **{"0_2": True})
    pdaf = _mask_like(data, **{"0_2": True})
    out = correct_defects(_raw_frame(data, defect_mask=defect, pdaf_mask=pdaf), DPCConfig())
    assert out[0, 2] == pytest.approx(500.0)


def test_dpc_handles_corners_and_defect_clusters() -> None:
    data = np.full((6, 6), 100, dtype=np.uint16)
    data[0, 0] = 0
    defect = _mask_like(data, **{"0_0": True})
    out = correct_defects(_raw_frame(data, defect_mask=defect), DPCConfig())
    assert out[0, 0] == pytest.approx(100.0)

    # A 2x2 RGGB frame has only one red site, so a red defect there has no
    # same-colour donor and must be left untouched rather than crash.
    tiny = np.array([[77, 88], [99, 111]], dtype=np.uint16)
    tiny_defect = _mask_like(tiny, **{"0_0": True})
    tiny_out = correct_defects(_raw_frame(tiny, defect_mask=tiny_defect), DPCConfig())
    assert tiny_out[0, 0] == pytest.approx(77.0)


def test_dpc_corrects_mono_pixels_from_mono_neighbours() -> None:
    data = np.full((4, 4), 50, dtype=np.uint16)
    data[1, 1] = 0
    defect = _mask_like(data, **{"1_1": True})
    out = correct_defects(
        _raw_frame(data, cfa=CFAPattern.MONO, defect_mask=defect),
        DPCConfig(),
    )
    assert out[1, 1] == pytest.approx(50.0)


def test_dpc_disabled_or_unmasked_is_a_float32_pass_through() -> None:
    data = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    np.testing.assert_array_equal(
        correct_defects(_raw_frame(data), DPCConfig(enabled=False)),
        data.astype(np.float32),
    )
    np.testing.assert_array_equal(
        correct_defects(_raw_frame(data), DPCConfig()),
        data.astype(np.float32),
    )


def test_dpc_empty_mask_is_an_exact_pass_through() -> None:
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    mask = np.zeros(data.shape, dtype=np.bool_)
    out = correct_defects(_raw_frame(data, defect_mask=mask), DPCConfig())
    np.testing.assert_array_equal(out, data.astype(np.float32))


def test_dpc_does_not_mutate_inputs() -> None:
    data = np.full((4, 4), 100, dtype=np.uint16)
    data[1, 1] = 0
    defect = _mask_like(data, **{"1_1": True})
    original_data = data.copy()
    original_mask = defect.copy()
    correct_defects(_raw_frame(data, defect_mask=defect), DPCConfig())
    np.testing.assert_array_equal(data, original_data)
    np.testing.assert_array_equal(defect, original_mask)


def test_dpc_validates_argument_types() -> None:
    raw = _raw_frame(np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(ValueError, match="DPCConfig"):
        correct_defects(raw, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="RawFrame"):
        correct_defects(None, DPCConfig())  # type: ignore[arg-type]


def test_dpc_validates_mutated_mask_shape_and_dtype() -> None:
    data = np.zeros((4, 4), dtype=np.uint16)
    raw = _raw_frame(data, defect_mask=np.zeros(data.shape, dtype=np.bool_))
    raw.defect_mask = np.zeros((3, 3), dtype=np.bool_)
    with pytest.raises(ValueError, match="shape"):
        correct_defects(raw, DPCConfig())

    raw.defect_mask = np.zeros((4, 4), dtype=np.uint8)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="dtype"):
        correct_defects(raw, DPCConfig())

    raw.defect_mask = None
    raw.pdaf_mask = np.zeros((3, 3), dtype=np.bool_)
    with pytest.raises(ValueError, match="shape"):
        correct_defects(raw, DPCConfig())

    raw.pdaf_mask = np.zeros((4, 4), dtype=np.int32)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="dtype"):
        correct_defects(raw, DPCConfig())


def test_dpc_rejects_an_unknown_method_after_mutation() -> None:
    data = np.full((4, 4), 100, dtype=np.uint16)
    data[1, 1] = 0
    raw = _raw_frame(data, defect_mask=_mask_like(data, **{"1_1": True}))
    config = DPCConfig()
    config.method = "bilinear"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="method"):
        correct_defects(raw, config)


def test_dpc_is_deterministic() -> None:
    data = _mosaic({"R": 100, "G": 200, "B": 300})
    data[2, 2] = 0
    raw = _raw_frame(data, defect_mask=_mask_like(data, **{"2_2": True}))
    first = correct_defects(raw, DPCConfig(method="median"))
    second = correct_defects(raw, DPCConfig(method="median"))
    np.testing.assert_array_equal(first, second)
