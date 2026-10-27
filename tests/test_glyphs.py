"""Tests for deterministic spectral glyph rendering and dataset generation.

A glyph scene is an image-plane reflectance target: dark ink on light paper,
both described by spectral reflectance factors on the common 360-830 nm,
5 nm grid. A grayscale coverage mask :math:`m` from FreeType rasterisation maps
the two materials with the convex combination

``rho(lambda) = paper(lambda) + (ink(lambda) - paper(lambda)) * m``.

The mask is rendered on an oversampled grid and materialised into a spectral
cube only on demand, so a dataset stores ``n`` masks rather than ``n`` full
cubes. Determinism is tested at both levels: repeated renders with the same
seed must be bit-identical, and train/test datasets must draw from disjoint
seed streams so no sample can leak between splits.

Failure artefacts guarded here: rasterising at the sensor grid aliases thin
strokes; disabling antialiasing turns edge pixels into a binary step; a font
size or rotation that exceeds the frame clips the glyph; reflectance spectra
off the common wavelength grid cannot be combined with the illuminant; and a
train/test seed collision silently leaks samples across the protocol boundary.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from camsim.scene.glyphs import (
    DEFAULT_INK_PATCH,
    DEFAULT_PAPER_PATCH,
    GLYPH_CLASSES,
    MAX_ROTATION_DEG,
    GlyphDataset,
    glyph_dataset,
    render_glyph,
    render_glyph_mask,
)
from camsim.scene.spectral_data import COMMON_WAVELENGTH_GRID, load_colorchecker

SPECTRUM_LENGTH = len(COMMON_WAVELENGTH_GRID.nm)


def _constant_spectrum(value: float) -> np.ndarray:
    """Return a spectrally neutral reflectance on the common grid."""
    return np.full(SPECTRUM_LENGTH, value, dtype=np.float64)


def test_glyph_classes_are_36() -> None:
    assert len(GLYPH_CLASSES) == 36
    assert GLYPH_CLASSES[:10] == tuple("0123456789")
    assert GLYPH_CLASSES[10:] == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert "0" in GLYPH_CLASSES and "A" in GLYPH_CLASSES
    assert len(set(GLYPH_CLASSES)) == 36


def test_glyph_render_is_deterministic() -> None:
    a = render_glyph("A", (64, 64), seed=7)
    b = render_glyph("A", (64, 64), seed=7)
    c = render_glyph("A", (64, 64), seed=8)

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_glyph_rendered_at_4x() -> None:
    shape = (16, 24)
    mask = render_glyph_mask("7", shape, seed=11)

    assert mask.shape == (shape[0] * 4, shape[1] * 4)
    assert mask.dtype == np.float64
    assert np.all(np.isfinite(mask))
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    # Antialiased TrueType edges produce fractional coverage; a nearest-neighbour
    # or binary raster would leave only exact zeros and ones.
    assert np.any((mask > 0.05) & (mask < 0.95))

    assert render_glyph("A", shape, seed=3).shape == (shape[0] * 4, shape[1] * 4, SPECTRUM_LENGTH)
    assert render_glyph_mask("A", shape, seed=3, oversampling=2).shape == (32, 48)
    assert render_glyph("A", shape, seed=3, oversampling=2).shape == (32, 48, SPECTRUM_LENGTH)


def test_glyph_ink_and_paper_spectra() -> None:
    shape = (32, 32)
    mask = render_glyph_mask("A", shape, seed=5)
    cube = render_glyph("A", shape, seed=5)
    chart = load_colorchecker()
    paper = chart.spectra[chart.patches.index(DEFAULT_PAPER_PATCH)]
    ink = chart.spectra[chart.patches.index(DEFAULT_INK_PATCH)]

    np.testing.assert_allclose(cube, paper + (ink - paper) * mask[:, :, None], atol=1e-12)
    assert np.any(mask == 0.0) and np.any(mask == 1.0)
    paper_rows = np.broadcast_to(paper, (int(np.sum(mask == 0.0)), SPECTRUM_LENGTH))
    ink_rows = np.broadcast_to(ink, (int(np.sum(mask == 1.0)), SPECTRUM_LENGTH))
    np.testing.assert_allclose(cube[mask == 0.0], paper_rows)
    np.testing.assert_allclose(cube[mask == 1.0], ink_rows)

    # Custom spectra follow the same convex mapping and stay within their bounds.
    dark_paper = _constant_spectrum(0.80)
    light_ink = _constant_spectrum(0.10)
    custom_mask = render_glyph_mask("Q", shape, seed=5)
    custom = render_glyph(
        "Q",
        shape,
        seed=5,
        paper_spectrum=dark_paper,
        ink_spectrum=light_ink,
    )
    assert custom.min() >= 0.10 - 1e-12
    assert custom.max() <= 0.80 + 1e-12
    expected_custom = np.broadcast_to(
        (0.80 + (0.10 - 0.80) * custom_mask)[:, :, None],
        custom.shape,
    )
    np.testing.assert_allclose(custom, expected_custom, atol=1e-12)


def test_glyph_rotation_bounds() -> None:
    dataset = glyph_dataset(12, seed=4, shape=(24, 24))

    assert np.all(np.abs(dataset.rotations) <= MAX_ROTATION_DEG)
    assert np.any(dataset.rotations != 0.0)
    assert render_glyph_mask("A", (32, 32), seed=2, rotation_deg=0.0).any()
    assert not np.array_equal(
        render_glyph_mask("A", (32, 32), seed=2, rotation_deg=0.0),
        render_glyph_mask("A", (32, 32), seed=2, rotation_deg=5.0),
    )

    with pytest.raises(ValueError, match="rotation"):
        render_glyph("A", (16, 16), seed=0, rotation_deg=MAX_ROTATION_DEG + 0.5)
    with pytest.raises(ValueError, match="rotation"):
        render_glyph_mask("A", (16, 16), seed=0, rotation_deg=float("nan"))


def test_glyph_dataset_split_has_no_overlap() -> None:
    train = glyph_dataset(20, seed=1, shape=(64, 64), split="train")
    test = glyph_dataset(20, seed=2, shape=(64, 64), split="test")

    assert isinstance(train, GlyphDataset)
    assert train.classes.shape == (20,) and test.classes.shape == (20,)
    assert set(train.seeds.tolist()).isdisjoint(test.seeds.tolist())
    # The split stream, not only the caller's base seed, must separate the sets.
    same_seed_train = glyph_dataset(20, seed=1, shape=(64, 64), split="train")
    same_seed_test = glyph_dataset(20, seed=1, shape=(64, 64), split="test")
    assert set(same_seed_train.seeds.tolist()).isdisjoint(same_seed_test.seeds.tolist())
    assert not np.array_equal(same_seed_train.masks, same_seed_test.masks)


def test_glyph_dataset_is_deterministic() -> None:
    first = glyph_dataset(8, seed=3, shape=(16, 16), split="train")
    second = glyph_dataset(8, seed=3, shape=(16, 16), split="train")
    different = glyph_dataset(8, seed=3, shape=(16, 16), split="test")

    np.testing.assert_array_equal(first.masks, second.masks)
    np.testing.assert_array_equal(first.classes, second.classes)
    np.testing.assert_array_equal(first.seeds, second.seeds)
    np.testing.assert_array_equal(first.rotations, second.rotations)
    np.testing.assert_array_equal(first.paper_spectra, second.paper_spectra)
    np.testing.assert_array_equal(first.ink_spectra, second.ink_spectra)
    assert first.chars == second.chars
    assert not np.array_equal(first.seeds, different.seeds)


def test_glyph_dataset_stores_masks_and_materialises_cubes_on_demand() -> None:
    dataset = glyph_dataset(20, seed=1, shape=(64, 64), split="train")

    assert dataset.masks.shape == (20, 256, 256)
    assert dataset.masks.dtype == np.float16
    assert dataset.masks.nbytes == 20 * 256 * 256 * 2
    assert dataset.masks.nbytes < 4 * 1024 * 1024
    assert dataset.classes.shape == (20,)
    assert dataset.chars and len(dataset.chars) == 20
    assert dataset.seeds.shape == (20,)
    assert dataset.rotations.shape == (20,)
    assert dataset.paper_spectra.shape == (20, SPECTRUM_LENGTH)
    assert dataset.ink_spectra.shape == (20, SPECTRUM_LENGTH)
    assert dataset.shape == (64, 64)
    assert dataset.oversampling == 4
    assert dataset.split == "train"
    assert len(dataset) == 20

    cube = dataset.materialize(0)
    expected = (
        dataset.paper_spectra[0]
        + (dataset.ink_spectra[0] - dataset.paper_spectra[0])
        * dataset.masks[0].astype(np.float64)[:, :, None]
    )
    assert cube.shape == (256, 256, SPECTRUM_LENGTH)
    assert cube.dtype == np.float64
    np.testing.assert_allclose(cube, expected, atol=1e-12)

    # The stored mask and seed must be the same render the public function makes.
    direct = render_glyph(
        dataset.chars[0],
        dataset.shape,
        seed=int(dataset.seeds[0]),
        rotation_deg=float(dataset.rotations[0]),
    )
    np.testing.assert_allclose(cube, direct, atol=2e-3)


def test_glyph_dataset_classes_are_balanced_and_match_chars() -> None:
    dataset = glyph_dataset(36, seed=9, shape=(16, 16))

    assert sorted(dataset.classes.tolist()) == list(range(36))
    assert set(dataset.chars) == set(GLYPH_CLASSES)
    for class_index, char in zip(dataset.classes, dataset.chars, strict=True):
        assert GLYPH_CLASSES[int(class_index)] == char


def test_glyph_dataset_materialize_rejects_bad_indices() -> None:
    dataset = glyph_dataset(2, seed=0, shape=(8, 8))

    for index in (-1, 2, 1.5, True):
        with pytest.raises(ValueError, match="index"):
            dataset.materialize(index)  # type: ignore[arg-type]


def test_glyph_dataset_rejects_inconsistent_stores() -> None:
    dataset = glyph_dataset(2, seed=0, shape=(8, 8))
    cases: list[tuple[dict[str, object], str]] = [
        ({"masks": dataset.masks[:, :, :-1]}, "masks"),
        ({"masks": dataset.masks.astype(np.float32)}, "masks"),
        ({"masks": dataset.masks * np.float16(2.0)}, "masks"),
        ({"classes": dataset.classes[:1]}, "classes"),
        ({"chars": ("a", "A")}, "chars"),
        ({"seeds": np.full(2, -1, dtype=np.int64)}, "seeds"),
        ({"rotations": np.full(2, MAX_ROTATION_DEG + 1.0)}, "rotations"),
        ({"paper_spectra": dataset.paper_spectra[:, :-1]}, "paper_spectra"),
        ({"provenance": []}, "provenance"),
    ]

    for overrides, message in cases:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(dataset, **overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"n": 0}, "n"),
        ({"n": -3}, "n"),
        ({"n": 1.5}, "n"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"shape": (0, 16)}, "shape"),
        ({"shape": (16, 16, 3)}, "shape"),
        ({"split": "validation"}, "split"),
    ],
)
def test_glyph_dataset_rejects_invalid_arguments(
    arguments: dict[str, object], message: str
) -> None:
    options: dict[str, object] = {"n": 4, "seed": 0, "shape": (8, 8), "split": "train"}
    options.update(arguments)
    with pytest.raises(ValueError, match=message):
        glyph_dataset(**options)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"char": "a"}, "char"),
        ({"char": "!"}, "char"),
        ({"char": ""}, "char"),
        ({"char": 7}, "char"),
        ({"shape": (0, 8)}, "shape"),
        ({"shape": (8, 8, 8)}, "shape"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"seed": True}, "seed"),
        ({"oversampling": 0}, "oversampling"),
        ({"oversampling": 2.5}, "oversampling"),
    ],
)
def test_render_glyph_rejects_invalid_arguments(kwargs: dict[str, object], message: str) -> None:
    options: dict[str, object] = {"char": "A", "shape": (8, 8), "seed": 0}
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        render_glyph(**options)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["paper_spectrum", "ink_spectrum"])
@pytest.mark.parametrize(
    "spectrum",
    [
        np.ones(SPECTRUM_LENGTH + 1),
        np.ones((SPECTRUM_LENGTH, 2)),
        np.full(SPECTRUM_LENGTH, np.nan),
        np.full(SPECTRUM_LENGTH, -0.1),
        np.full(SPECTRUM_LENGTH, 1.1),
    ],
)
def test_render_glyph_rejects_spectra_off_the_grid(field_name: str, spectrum: np.ndarray) -> None:
    kwargs: dict[str, object] = {"char": "A", "shape": (8, 8), "seed": 0, field_name: spectrum}
    with pytest.raises(ValueError, match=field_name):
        render_glyph(**kwargs)  # type: ignore[arg-type]
