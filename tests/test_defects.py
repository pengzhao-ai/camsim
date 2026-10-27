"""Tests for defective-pixel and PDAF masks.

A real sensor array is not perfect. Dead pixels have no photoresponse, hot
pixels collect excess dark current and brighten with exposure and temperature,
and stuck pixels are pinned to a fixed code regardless of illumination. Each
failure mode is a per-pixel probability, typically between one in ten thousand
and one in a million for a well-controlled consumer part, so the tests here
check that the realized rates are statistically consistent with the configured
values and that the maps are reproducible from a seed. Hot pixels are flagged
rather than corrected here: the sensor applies their dark-current multiplier
and a defect-correction stage later replaces dead and stuck sites.

Phase-detection autofocus (PDAF) pixels are also flagged but are *not*
defects. They sacrifice part of the photodiode aperture to sample the exit
pupil, so they collect less light (``qe_factor`` < 1) and must be excluded from
defect correction while the sensor applies their reduced quantum efficiency.
Sparse PDAF layouts place a small fraction of sites on one CFA colour, usually
green. This module tests that the requested colour, fraction, and pattern are
respected, including the ``none`` layout and mono sensors with no colour sites.

Failure artefacts guarded here: rates expressed as percentages (1 instead of
0.01) flag every pixel; recomputing maps per frame turns fixed-pattern defects
into temporal noise; and sharing one RNG sequence between defect and PDAF maps
would make the two layouts artificially correlated.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from camsim.config.schema import DefectConfig, PDAFConfig
from camsim.core.frames import CFAPattern
from camsim.sensor.cfa import cfa_masks
from camsim.sensor.defects import DefectMasks, defect_masks, pdaf_mask


def _defect_config(
    *,
    dead_rate: float = 1e-3,
    hot_rate: float = 2e-3,
    stuck_rate: float = 5e-4,
) -> DefectConfig:
    """Return a validated defect configuration with the requested rates."""
    return DefectConfig(
        dead_rate=dead_rate,
        hot_rate=hot_rate,
        stuck_rate=stuck_rate,
        seed=1234,
    )


def _pdaf_config(
    *,
    fraction: float = 0.05,
    qe_factor: float = 0.5,
    pattern: str = "sparse_green",
) -> PDAFConfig:
    """Return a validated PDAF configuration."""
    return PDAFConfig(fraction=fraction, qe_factor=qe_factor, pattern=pattern)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Defect masks
# --------------------------------------------------------------------------- #


def test_defect_masks_have_the_requested_shape_and_dtype() -> None:
    masks = defect_masks(_defect_config(), (17, 9), seed=0)

    assert isinstance(masks, DefectMasks)
    for array in (masks.dead, masks.hot, masks.stuck):
        assert array.shape == (17, 9)
        assert array.dtype == np.bool_


def test_defect_rates_plausible() -> None:
    """Realized defect counts track the configured per-pixel probabilities."""
    shape = (1024, 1024)
    pixels = shape[0] * shape[1]

    masks = defect_masks(_defect_config(), shape, seed=3)

    assert int(masks.dead.sum()) == pytest.approx(1e-3 * pixels, rel=0.25)
    assert int(masks.hot.sum()) == pytest.approx(2e-3 * pixels, rel=0.25)
    assert int(masks.stuck.sum()) == pytest.approx(5e-4 * pixels, rel=0.25)


def test_defect_masks_are_reproducible_from_a_seed() -> None:
    config = _defect_config(dead_rate=0.01, hot_rate=0.02, stuck_rate=0.005)

    a = defect_masks(config, (128, 128), seed=7)
    b = defect_masks(config, (128, 128), seed=7)

    np.testing.assert_array_equal(a.dead, b.dead)
    np.testing.assert_array_equal(a.hot, b.hot)
    np.testing.assert_array_equal(a.stuck, b.stuck)


def test_defect_masks_change_with_the_seed() -> None:
    config = _defect_config(dead_rate=0.01, hot_rate=0.02, stuck_rate=0.005)

    a = defect_masks(config, (128, 128), seed=7)
    b = defect_masks(config, (128, 128), seed=8)

    assert not np.array_equal(a.dead, b.dead)
    assert not np.array_equal(a.hot, b.hot)
    assert not np.array_equal(a.stuck, b.stuck)


def test_zero_rates_produce_empty_masks() -> None:
    masks = defect_masks(
        _defect_config(dead_rate=0.0, hot_rate=0.0, stuck_rate=0.0),
        (64, 64),
        seed=1,
    )

    assert not masks.dead.any()
    assert not masks.hot.any()
    assert not masks.stuck.any()


def test_unit_rates_flag_every_pixel() -> None:
    masks = defect_masks(
        _defect_config(dead_rate=1.0, hot_rate=1.0, stuck_rate=1.0),
        (16, 16),
        seed=1,
    )

    assert masks.dead.all()
    assert masks.hot.all()
    assert masks.stuck.all()


def test_defect_masks_ignore_the_global_numpy_rng() -> None:
    config = _defect_config(dead_rate=0.01, hot_rate=0.01, stuck_rate=0.01)
    np.random.seed(1)
    a = defect_masks(config, (64, 64), seed=5)
    np.random.seed(2)
    b = defect_masks(config, (64, 64), seed=5)

    np.testing.assert_array_equal(a.dead, b.dead)
    np.testing.assert_array_equal(a.hot, b.hot)
    np.testing.assert_array_equal(a.stuck, b.stuck)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dead_rate", -0.1),
        ("dead_rate", 1.5),
        ("dead_rate", "0.1"),
        ("hot_rate", math.nan),
        ("stuck_rate", math.inf),
    ],
)
def test_defect_masks_validate_bypassed_rates(field: str, value: object) -> None:
    """Runtime validation catches rates that pydantic's ``model_construct`` skipped."""
    payload: dict[str, object] = {
        "dead_rate": 0.0,
        "hot_rate": 0.0,
        "stuck_rate": 0.0,
        "seed": 0,
    }
    payload[field] = value
    config = DefectConfig.model_construct(**payload)

    with pytest.raises(ValueError, match=field):
        defect_masks(config, (8, 8), seed=0)


@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (-1, 4), (4,), (2, 3, 4), (4.0, 4.0)])
def test_defect_masks_reject_invalid_shapes(shape: object) -> None:
    with pytest.raises(ValueError, match="shape"):
        defect_masks(_defect_config(), shape, seed=0)  # type: ignore[arg-type]


def test_defect_masks_reject_a_ragged_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        defect_masks(_defect_config(), [[4, 4], [4]], seed=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 2.5, True])
def test_defect_masks_reject_invalid_seeds(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        defect_masks(_defect_config(), (4, 4), seed=seed)  # type: ignore[arg-type]


def test_defect_masks_reject_the_wrong_config_type() -> None:
    with pytest.raises(ValueError, match="DefectConfig"):
        defect_masks(object(), (4, 4), seed=0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# PDAF masks
# --------------------------------------------------------------------------- #


def test_pdaf_mask_fraction() -> None:
    """The mask covers roughly ``fraction`` of the eligible colour sites."""
    shape = (512, 512)
    config = _pdaf_config(fraction=0.05)

    mask = pdaf_mask(config, shape, seed=7)

    eligible = cfa_masks(CFAPattern.RGGB, shape)["G"]
    assert mask.shape == shape
    assert mask.dtype == np.bool_
    assert not np.any(mask & ~eligible)
    assert int(mask.sum()) == pytest.approx(0.05 * int(eligible.sum()), rel=0.10)


@pytest.mark.parametrize(
    ("pattern", "cfa", "colour"),
    [
        ("sparse_green", CFAPattern.RGGB, "G"),
        ("sparse_green", CFAPattern.GBRG, "G"),
        ("sparse_red", CFAPattern.RGGB, "R"),
        ("sparse_blue", CFAPattern.RGGB, "B"),
        ("sparse_red", CFAPattern.BGGR, "R"),
        ("sparse_blue", CFAPattern.BGGR, "B"),
    ],
)
def test_pdaf_mask_selects_only_the_requested_colour(
    pattern: str,
    cfa: CFAPattern,
    colour: str,
) -> None:
    shape = (64, 64)
    config = _pdaf_config(fraction=0.25, pattern=pattern)

    mask = pdaf_mask(config, shape, seed=2, cfa=cfa)

    eligible = cfa_masks(cfa, shape)[colour]
    assert mask.any()
    assert not np.any(mask & ~eligible)


def test_pdaf_mask_selection_is_exact_at_full_fraction() -> None:
    shape = (32, 32)
    config = _pdaf_config(fraction=1.0, pattern="sparse_red")

    mask = pdaf_mask(config, shape, seed=1)

    np.testing.assert_array_equal(mask, cfa_masks(CFAPattern.RGGB, shape)["R"])


def test_pdaf_mask_respects_the_cfa_argument() -> None:
    """The same sparse-red rule lands on different sites for RGGB and BGGR."""
    shape = (32, 32)
    config = _pdaf_config(fraction=1.0, pattern="sparse_red")

    rggb = pdaf_mask(config, shape, seed=1)
    bggr = pdaf_mask(config, shape, seed=1, cfa=CFAPattern.BGGR)

    np.testing.assert_array_equal(rggb, cfa_masks(CFAPattern.RGGB, shape)["R"])
    np.testing.assert_array_equal(bggr, cfa_masks(CFAPattern.BGGR, shape)["R"])
    assert not np.array_equal(rggb, bggr)


def test_pdaf_mask_defaults_to_the_rggb_tile() -> None:
    config = _pdaf_config(fraction=0.1)

    default = pdaf_mask(config, (64, 64), seed=4)
    explicit = pdaf_mask(config, (64, 64), seed=4, cfa=CFAPattern.RGGB)

    np.testing.assert_array_equal(default, explicit)


def test_pdaf_mask_zero_fraction_flags_nothing() -> None:
    mask = pdaf_mask(_pdaf_config(fraction=0.0), (64, 64), seed=0)

    assert not mask.any()


def test_pdaf_mask_none_pattern_flags_nothing() -> None:
    mask = pdaf_mask(_pdaf_config(fraction=1.0, pattern="none"), (64, 64), seed=0)

    assert not mask.any()


def test_pdaf_mask_mono_has_no_eligible_colour_sites() -> None:
    mask = pdaf_mask(_pdaf_config(fraction=1.0), (64, 64), seed=0, cfa=CFAPattern.MONO)

    assert not mask.any()


def test_pdaf_mask_is_reproducible_and_seed_sensitive() -> None:
    config = _pdaf_config(fraction=0.1)

    a = pdaf_mask(config, (128, 128), seed=11)
    b = pdaf_mask(config, (128, 128), seed=11)
    c = pdaf_mask(config, (128, 128), seed=12)

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_pdaf_mask_ignores_qe_factor() -> None:
    """The mask only flags sites; the sensor applies the reduced QE."""
    shape = (64, 64)
    a = pdaf_mask(_pdaf_config(fraction=0.1, qe_factor=0.5), shape, seed=9)
    b = pdaf_mask(_pdaf_config(fraction=0.1, qe_factor=0.9), shape, seed=9)

    np.testing.assert_array_equal(a, b)


def test_pdaf_and_defect_masks_use_independent_seed_streams() -> None:
    """A shared seed must not make every PDAF site coincide with a dead pixel."""
    shape = (256, 256)
    defects = defect_masks(
        _defect_config(dead_rate=1e-3, hot_rate=0.0, stuck_rate=0.0),
        shape,
        seed=42,
    )

    pdaf = pdaf_mask(_pdaf_config(fraction=0.05), shape, seed=42)

    assert np.any(pdaf & ~defects.dead)


def test_pdaf_mask_ignores_the_global_numpy_rng() -> None:
    config = _pdaf_config(fraction=0.1)
    np.random.seed(1)
    a = pdaf_mask(config, (64, 64), seed=5)
    np.random.seed(2)
    b = pdaf_mask(config, (64, 64), seed=5)

    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("fraction", [-0.1, 1.5, math.nan, math.inf])
def test_pdaf_mask_validates_bypassed_fractions(fraction: float) -> None:
    config = PDAFConfig.model_construct(
        fraction=fraction,
        qe_factor=0.5,
        pattern="sparse_green",
    )

    with pytest.raises(ValueError, match="fraction"):
        pdaf_mask(config, (8, 8), seed=0)


def test_pdaf_mask_rejects_an_unknown_pattern() -> None:
    config = PDAFConfig.model_construct(
        fraction=0.1,
        qe_factor=0.5,
        pattern="sparse_rainbow",
    )

    with pytest.raises(ValueError, match="pattern"):
        pdaf_mask(config, (8, 8), seed=0)


@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (-1, 4), (4,), (2, 3, 4), (4.0, 4.0)])
def test_pdaf_mask_rejects_invalid_shapes(shape: object) -> None:
    with pytest.raises(ValueError, match="shape"):
        pdaf_mask(_pdaf_config(), shape, seed=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_pdaf_mask_rejects_invalid_seeds(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        pdaf_mask(_pdaf_config(), (4, 4), seed=seed)  # type: ignore[arg-type]


def test_pdaf_mask_rejects_the_wrong_config_type() -> None:
    with pytest.raises(ValueError, match="PDAFConfig"):
        pdaf_mask(object(), (4, 4), seed=0)  # type: ignore[arg-type]
