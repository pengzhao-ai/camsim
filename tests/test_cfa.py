"""Tests for CFA tiling, channel mosaics, and spectral projection matrices.

A colour-filter array turns one scalar photodiode signal per pixel into a
mosaic of colour samples. The 2x2 Bayer tile is the dominant consumer layout;
the four Bayer names encode which corner carries red. MONO has no filters at
all, and RCCC is an automotive layout with one red and three clear sites per
tile. The spatial masks tested here must partition every pixel exactly once,
because a missing or double-counted site produces a colour cast that no later
ISP stage can recover.

The spectral projection matrix is the radiometric heart of the sensor:

``P_c(lambda) = QE(lambda) * T_IR(lambda) * T_CFA,c(lambda)``

Each row is the dimensionless spectral response of one CFA channel; the sensor
multiplies it by scene irradiance, photon energy, pixel area, and exposure to
obtain electrons. This module deliberately excludes those factors so the
matrix can be cached and audited independently. Tests use synthetic CSVs that
cover the common 360-830 nm grid, keeping the suite independent of the
flagship preset data that arrives later.

Failure artefacts guarded here: a transposed or offset Bayer tile swaps red
and blue; a projection matrix with a negative or dead channel silently biases
white balance; and extrapolating a measured QE curve beyond its data invents
near-infrared sensitivity that the real part does not have.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from camsim.core.frames import CFAPattern
from camsim.core.spectral import WavelengthGrid
from camsim.sensor.cfa import (
    cfa_color_at,
    cfa_masks,
    mosaic_from_channels,
    spectral_projection,
)

GRID_NM = np.arange(360.0, 831.0, 5.0)
GRID = WavelengthGrid(GRID_NM)


# --------------------------------------------------------------------------- #
# Synthetic spectral curves
# --------------------------------------------------------------------------- #


def _qe_curve() -> np.ndarray:
    """A smooth, strictly positive QE curve peaking in the green."""
    return 0.35 + 0.50 * np.exp(-0.5 * ((GRID_NM - 560.0) / 120.0) ** 2)


def _ir_cut_curve() -> np.ndarray:
    """A visible-pass, near-infrared-blocking IR-cut filter."""
    return np.where(
        GRID_NM <= 650.0,
        1.0,
        np.clip(1.0 - (GRID_NM - 650.0) / 60.0, 0.02, 1.0),
    )


def _cfa_curves() -> dict[str, np.ndarray]:
    """Three overlapping, strictly positive Bayer passbands."""
    return {
        "R": 0.02 + 0.88 * np.exp(-0.5 * ((GRID_NM - 600.0) / 45.0) ** 2),
        "G": 0.02 + 0.88 * np.exp(-0.5 * ((GRID_NM - 540.0) / 45.0) ** 2),
        "B": 0.02 + 0.88 * np.exp(-0.5 * ((GRID_NM - 460.0) / 45.0) ** 2),
    }


def _write_curve(
    path: Path,
    values: np.ndarray,
    *,
    wavelengths_nm: np.ndarray = GRID_NM,
    header: str | None = "wavelength_nm,value",
) -> None:
    """Write a two-column spectral curve with a comment and optional header."""
    lines = ["# synthetic curve written by the CFA test suite"]
    if header is not None:
        lines.append(header)
    lines.extend(
        f"{wavelength:.17g},{value:.17g}"
        for wavelength, value in zip(wavelengths_nm, values, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cfa(
    path: Path,
    curves: dict[str, np.ndarray],
    *,
    header: bool = True,
    wavelengths_nm: np.ndarray = GRID_NM,
) -> None:
    """Write one column per CFA channel, optionally preceded by a header."""
    lines: list[str] = []
    if header:
        lines.append(",".join(("wavelength_nm", *curves)))
    for index, wavelength in enumerate(wavelengths_nm):
        values = ",".join(f"{curves[name][index]:.17g}" for name in curves)
        lines.append(f"{wavelength:.17g},{values}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _projection_files(
    tmp_path: Path,
    *,
    cfa_header: bool = True,
    qe: np.ndarray | None = None,
    ir: np.ndarray | None = None,
    cfa: dict[str, np.ndarray] | None = None,
) -> tuple[Path, Path, Path, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Write a complete set of synthetic QE, CFA, and IR-cut CSVs."""
    qe_values = _qe_curve() if qe is None else qe
    ir_values = _ir_cut_curve() if ir is None else ir
    cfa_values = _cfa_curves() if cfa is None else cfa
    qe_path = tmp_path / "qe.csv"
    cfa_path = tmp_path / "cfa.csv"
    ir_path = tmp_path / "ir_cut.csv"
    _write_curve(qe_path, qe_values)
    _write_cfa(cfa_path, cfa_values, header=cfa_header)
    _write_curve(ir_path, ir_values)
    return qe_path, cfa_path, ir_path, qe_values, ir_values, cfa_values


# --------------------------------------------------------------------------- #
# CFA tiling and masks
# --------------------------------------------------------------------------- #


def test_rggb_layout() -> None:
    assert cfa_color_at(CFAPattern.RGGB, 0, 0) == "R"
    assert cfa_color_at(CFAPattern.RGGB, 0, 1) == "G"
    assert cfa_color_at(CFAPattern.RGGB, 1, 0) == "G"
    assert cfa_color_at(CFAPattern.RGGB, 1, 1) == "B"


@pytest.mark.parametrize(
    ("pattern", "expected_tile"),
    [
        (CFAPattern.RGGB, (("R", "G"), ("G", "B"))),
        (CFAPattern.BGGR, (("B", "G"), ("G", "R"))),
        (CFAPattern.GRBG, (("G", "R"), ("B", "G"))),
        (CFAPattern.GBRG, (("G", "B"), ("R", "G"))),
        (CFAPattern.RCCC, (("R", "C"), ("C", "C"))),
        (CFAPattern.MONO, (("M", "M"), ("M", "M"))),
    ],
)
def test_all_pattern_tiles(
    pattern: CFAPattern,
    expected_tile: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    for y in range(2):
        for x in range(2):
            assert cfa_color_at(pattern, y, x) == expected_tile[y][x]


def test_cfa_tiling_repeats_every_two_pixels() -> None:
    assert cfa_color_at(CFAPattern.RGGB, 2, 2) == "R"
    assert cfa_color_at(CFAPattern.RGGB, 3, 5) == "B"
    assert cfa_color_at(CFAPattern.GBRG, 7, 4) == "R"


def test_pattern_strings_are_accepted() -> None:
    assert cfa_color_at("RGGB", 1, 1) == "B"
    assert cfa_masks("MONO", (2, 2))["M"].all()


def test_mono_has_one_channel_mask() -> None:
    masks = cfa_masks(CFAPattern.MONO, (4, 4))
    assert set(masks) == {"M"}
    assert masks["M"].all()


def test_rccc_masks() -> None:
    masks = cfa_masks(CFAPattern.RCCC, (4, 4))

    assert list(masks) == ["R", "C"]
    assert masks["R"].dtype == np.bool_
    assert masks["C"].dtype == np.bool_
    assert int(masks["R"].sum()) == 4
    assert int(masks["C"].sum()) == 12
    assert (masks["R"] | masks["C"]).all()
    assert not (masks["R"] & masks["C"]).any()
    expected_red = np.zeros((4, 4), dtype=bool)
    expected_red[0::2, 0::2] = True
    np.testing.assert_array_equal(masks["R"], expected_red)


@pytest.mark.parametrize(
    "pattern",
    [
        CFAPattern.RGGB,
        CFAPattern.BGGR,
        CFAPattern.GRBG,
        CFAPattern.GBRG,
        CFAPattern.RCCC,
        CFAPattern.MONO,
    ],
)
@pytest.mark.parametrize("shape", [(3, 5), (1, 1), (7, 4)])
def test_masks_partition_every_pixel(
    pattern: CFAPattern,
    shape: tuple[int, int],
) -> None:
    masks = cfa_masks(pattern, shape)

    coverage = np.zeros(shape, dtype=np.int16)
    for mask in masks.values():
        assert mask.shape == shape
        assert mask.dtype == np.bool_
        coverage += mask
    np.testing.assert_array_equal(coverage, np.ones(shape, dtype=np.int16))


def test_bayer_masks_use_one_green_mask() -> None:
    masks = cfa_masks(CFAPattern.RGGB, (4, 4))

    assert list(masks) == ["R", "G", "B"]
    assert int(masks["R"].sum()) == 4
    assert int(masks["G"].sum()) == 8
    assert int(masks["B"].sum()) == 4
    assert masks["G"][0, 1] and masks["G"][1, 0]


@pytest.mark.parametrize("pattern", list(CFAPattern))
@pytest.mark.parametrize("shape", [(2, 2), (3, 5), (4, 4)])
def test_masks_match_the_tiled_colour(pattern: CFAPattern, shape: tuple[int, int]) -> None:
    masks = cfa_masks(pattern, shape)

    for y in range(shape[0]):
        for x in range(shape[1]):
            colour = cfa_color_at(pattern, y, x)
            for name, mask in masks.items():
                assert bool(mask[y, x]) == (name == colour)


@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (-1, 4), (4,), (4, 4, 4), (4.0, 4)])
def test_cfa_masks_reject_invalid_shape(shape: object) -> None:
    with pytest.raises(ValueError, match="shape"):
        cfa_masks(CFAPattern.RGGB, shape)  # type: ignore[arg-type]


@pytest.mark.parametrize(("y", "x"), [(-1, 0), (0, -1), (1.0, 0), (0, 1.5), (True, 0)])
def test_cfa_color_rejects_invalid_coordinates(y: object, x: object) -> None:
    with pytest.raises(ValueError, match=r"[xy]"):
        cfa_color_at(CFAPattern.RGGB, y, x)  # type: ignore[arg-type]


@pytest.mark.parametrize("pattern", ["UNKNOWN", "rggb", 7])
def test_cfa_functions_reject_invalid_pattern(pattern: object) -> None:
    with pytest.raises(ValueError, match="pattern"):
        cfa_color_at(pattern, 0, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pattern"):
        cfa_masks(pattern, (2, 2))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Mosaicking channel planes
# --------------------------------------------------------------------------- #


def test_mosaic_places_channels() -> None:
    ch = {
        "R": np.full((4, 4), 1.0),
        "G": np.full((4, 4), 2.0),
        "B": np.full((4, 4), 3.0),
    }
    m = mosaic_from_channels(ch, CFAPattern.RGGB)
    assert m[0, 0] == 1.0 and m[0, 1] == 2.0 and m[1, 1] == 3.0
    assert m[1, 0] == 2.0 and m[2, 2] == 1.0


@pytest.mark.parametrize(
    ("pattern", "channels", "expected"),
    [
        (
            CFAPattern.BGGR,
            {"R": 1.0, "G": 2.0, "B": 3.0},
            [
                [3.0, 2.0, 3.0, 2.0],
                [2.0, 1.0, 2.0, 1.0],
                [3.0, 2.0, 3.0, 2.0],
                [2.0, 1.0, 2.0, 1.0],
            ],
        ),
        (
            CFAPattern.GRBG,
            {"R": 1.0, "G": 2.0, "B": 3.0},
            [
                [2.0, 1.0, 2.0, 1.0],
                [3.0, 2.0, 3.0, 2.0],
                [2.0, 1.0, 2.0, 1.0],
                [3.0, 2.0, 3.0, 2.0],
            ],
        ),
        (
            CFAPattern.GBRG,
            {"R": 1.0, "G": 2.0, "B": 3.0},
            [
                [2.0, 3.0, 2.0, 3.0],
                [1.0, 2.0, 1.0, 2.0],
                [2.0, 3.0, 2.0, 3.0],
                [1.0, 2.0, 1.0, 2.0],
            ],
        ),
    ],
)
def test_mosaic_bayer_variants(
    pattern: CFAPattern,
    channels: dict[str, float],
    expected: list[list[float]],
) -> None:
    planes = {name: np.full((4, 4), value) for name, value in channels.items()}

    mosaic = mosaic_from_channels(planes, pattern)

    np.testing.assert_array_equal(mosaic, np.asarray(expected))


def test_mosaic_rccc() -> None:
    red = np.full((4, 4), 5.0)
    clear = np.full((4, 4), 7.0)

    mosaic = mosaic_from_channels({"R": red, "C": clear}, CFAPattern.RCCC)

    assert mosaic[0, 0] == 5.0
    assert mosaic[0, 1] == 7.0
    assert mosaic[1, 0] == 7.0
    assert mosaic[1, 1] == 7.0
    assert int(np.count_nonzero(mosaic == 5.0)) == 4


def test_mosaic_mono_copies_the_plane() -> None:
    plane = np.arange(35.0).reshape(5, 7)

    mosaic = mosaic_from_channels({"M": plane}, CFAPattern.MONO)

    np.testing.assert_array_equal(mosaic, plane)
    assert mosaic is not plane
    plane[0, 0] = -1.0
    assert mosaic[0, 0] == 0.0


def test_mosaic_promotes_input_dtypes() -> None:
    planes = {
        "R": np.ones((2, 2), dtype=np.float32),
        "G": np.full((2, 2), 2, dtype=np.int32),
        "B": np.full((2, 2), 3.0, dtype=np.float64),
    }

    mosaic = mosaic_from_channels(planes, CFAPattern.RGGB)

    assert mosaic.dtype == np.result_type(planes["R"].dtype, planes["G"].dtype, planes["B"].dtype)


def test_mosaic_rejects_missing_and_extra_channels() -> None:
    shape = (4, 4)
    with pytest.raises(ValueError, match="missing"):
        mosaic_from_channels({"R": np.zeros(shape), "G": np.zeros(shape)}, CFAPattern.RGGB)
    with pytest.raises(ValueError, match="unexpected"):
        mosaic_from_channels(
            {
                "R": np.zeros(shape),
                "G": np.zeros(shape),
                "B": np.zeros(shape),
                "X": np.zeros(shape),
            },
            CFAPattern.RGGB,
        )


def test_mosaic_rejects_invalid_planes() -> None:
    with pytest.raises(ValueError, match="2-D"):
        mosaic_from_channels(
            {"R": np.zeros(4), "G": np.zeros((4, 4)), "B": np.zeros((4, 4))},
            CFAPattern.RGGB,
        )
    with pytest.raises(ValueError, match="same shape"):
        mosaic_from_channels(
            {"R": np.zeros((4, 4)), "G": np.zeros((3, 4)), "B": np.zeros((4, 4))},
            CFAPattern.RGGB,
        )
    with pytest.raises(ValueError, match="finite"):
        mosaic_from_channels(
            {
                "R": np.full((4, 4), np.nan),
                "G": np.zeros((4, 4)),
                "B": np.zeros((4, 4)),
            },
            CFAPattern.RGGB,
        )
    with pytest.raises(ValueError, match="numeric"):
        mosaic_from_channels(
            {
                "R": np.full((4, 4), "red"),
                "G": np.zeros((4, 4)),
                "B": np.zeros((4, 4)),
            },
            CFAPattern.RGGB,
        )


def test_mosaic_rejects_non_mapping_channels() -> None:
    with pytest.raises(ValueError, match="mapping"):
        mosaic_from_channels([np.zeros((4, 4))], CFAPattern.MONO)  # type: ignore[arg-type]


@pytest.mark.parametrize("pattern", list(CFAPattern))
def test_mosaic_matches_the_tiled_colour(pattern: CFAPattern) -> None:
    shape = (3, 5)
    colours = list(cfa_masks(pattern, shape))
    planes = {
        colour: np.arange(float(shape[0] * shape[1])).reshape(shape) + 100.0 * index
        for index, colour in enumerate(colours)
    }

    mosaic = mosaic_from_channels(planes, pattern)

    for y in range(shape[0]):
        for x in range(shape[1]):
            assert mosaic[y, x] == planes[cfa_color_at(pattern, y, x)][y, x]


# --------------------------------------------------------------------------- #
# Spectral projection
# --------------------------------------------------------------------------- #


def test_projection_shape_and_nonnegative(tmp_path: Path) -> None:
    qe_path, cfa_path, ir_path, qe, ir, cfa = _projection_files(tmp_path)

    projection = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    assert projection.shape == (3, GRID_NM.size)
    assert projection.dtype == np.float64
    assert np.all(np.isfinite(projection))
    assert np.all(projection >= 0.0)
    expected = np.vstack([qe * ir * cfa[name] for name in ("R", "G", "B")])
    np.testing.assert_allclose(projection, expected, rtol=1e-12, atol=1e-15)


def test_projection_channel_order_is_rgb(tmp_path: Path) -> None:
    qe_path, cfa_path, ir_path, qe, ir, cfa = _projection_files(tmp_path)

    projection = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    for index, name in enumerate(("R", "G", "B")):
        np.testing.assert_allclose(projection[index], qe * ir * cfa[name], rtol=1e-12, atol=1e-15)


def test_projection_honours_cfa_header_order(tmp_path: Path) -> None:
    qe = _qe_curve()
    ir = _ir_cut_curve()
    curves = _cfa_curves()
    qe_path = tmp_path / "qe.csv"
    ir_path = tmp_path / "ir_cut.csv"
    cfa_path = tmp_path / "cfa_reordered.csv"
    _write_curve(qe_path, qe)
    _write_curve(ir_path, ir)
    _write_cfa(cfa_path, {"B": curves["B"], "R": curves["R"], "G": curves["G"]})

    projection = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    expected = np.vstack([qe * ir * curves[name] for name in ("R", "G", "B")])
    np.testing.assert_allclose(projection, expected, rtol=1e-12, atol=1e-15)


def test_projection_accepts_headerless_cfa_in_canonical_order(tmp_path: Path) -> None:
    qe_path, cfa_path, ir_path, qe, ir, cfa = _projection_files(tmp_path, cfa_header=False)

    projection = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    expected = np.vstack([qe * ir * cfa[name] for name in ("R", "G", "B")])
    np.testing.assert_allclose(projection, expected, rtol=1e-12, atol=1e-15)


def test_projection_accepts_two_column_cfa_as_neutral_curve(tmp_path: Path) -> None:
    qe = _qe_curve()
    ir = _ir_cut_curve()
    neutral = np.full(GRID_NM.size, 0.5)
    qe_path = tmp_path / "qe.csv"
    cfa_path = tmp_path / "cfa_neutral.csv"
    ir_path = tmp_path / "ir_cut.csv"
    _write_curve(qe_path, qe)
    _write_curve(cfa_path, neutral)
    _write_curve(ir_path, ir)

    projection = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    expected = np.tile(qe * ir * neutral, (3, 1))
    np.testing.assert_allclose(projection, expected, rtol=1e-12, atol=1e-15)


def test_projection_rccc_channel_order(tmp_path: Path) -> None:
    qe = _qe_curve()
    ir = _ir_cut_curve()
    curves = {"R": _cfa_curves()["R"], "C": np.full(GRID_NM.size, 0.98)}
    qe_path = tmp_path / "qe.csv"
    cfa_path = tmp_path / "cfa_rccc.csv"
    ir_path = tmp_path / "ir_cut.csv"
    _write_curve(qe_path, qe)
    _write_cfa(cfa_path, curves)
    _write_curve(ir_path, ir)

    projection = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RCCC)

    assert projection.shape == (2, GRID_NM.size)
    np.testing.assert_allclose(projection[0], qe * ir * curves["R"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(projection[1], qe * ir * curves["C"], rtol=1e-12, atol=1e-15)


def test_mono_qe_exceeds_bayer_qe(tmp_path: Path) -> None:
    qe_path, cfa_path, ir_path, qe, ir, _cfa = _projection_files(tmp_path)

    bayer = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)
    mono = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.MONO)

    assert mono.shape == (1, GRID_NM.size)
    np.testing.assert_allclose(mono[0], qe * ir, rtol=1e-12, atol=1e-15)
    for channel in range(3):
        assert np.all(mono[0] > bayer[channel])


def test_mono_ignores_the_cfa_file(tmp_path: Path) -> None:
    qe_path = tmp_path / "qe.csv"
    ir_path = tmp_path / "ir_cut.csv"
    _write_curve(qe_path, _qe_curve())
    _write_curve(ir_path, _ir_cut_curve())

    projection = spectral_projection(
        qe_path,
        tmp_path / "does-not-exist.csv",
        ir_path,
        GRID,
        CFAPattern.MONO,
    )

    assert projection.shape == (1, GRID_NM.size)


def test_projection_accepts_an_array_like_grid(tmp_path: Path) -> None:
    qe_path, cfa_path, ir_path, *_ = _projection_files(tmp_path)

    from_grid = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)
    from_array = spectral_projection(qe_path, cfa_path, ir_path, GRID_NM, CFAPattern.RGGB)

    np.testing.assert_array_equal(from_grid, from_array)


def test_projection_accepts_headerless_qe_and_ir_curves(tmp_path: Path) -> None:
    qe = _qe_curve()
    ir = _ir_cut_curve()
    curves = _cfa_curves()
    qe_path = tmp_path / "qe.csv"
    cfa_path = tmp_path / "cfa.csv"
    ir_path = tmp_path / "ir_cut.csv"
    _write_curve(qe_path, qe, header=None)
    _write_cfa(cfa_path, curves)
    _write_curve(ir_path, ir, header=None)

    projection = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    expected = np.vstack([qe * ir * curves[name] for name in ("R", "G", "B")])
    np.testing.assert_allclose(projection, expected, rtol=1e-12, atol=1e-15)


def test_projection_interpolates_a_coarser_curve_onto_the_common_grid(tmp_path: Path) -> None:
    # A 10 nm source curve is linearly interpolated onto the 5 nm common grid.
    # Both the QE ramp and the constants are exactly representable, so the
    # expected values follow analytically.
    coarse_nm = np.arange(360.0, 831.0, 10.0)
    qe = 0.4 + (coarse_nm - 360.0) / 1000.0
    ir = np.full(coarse_nm.size, 0.9)
    cfa = {
        "R": np.full(coarse_nm.size, 0.50),
        "G": np.full(coarse_nm.size, 0.60),
        "B": np.full(coarse_nm.size, 0.70),
    }
    qe_path = tmp_path / "qe.csv"
    cfa_path = tmp_path / "cfa.csv"
    ir_path = tmp_path / "ir_cut.csv"
    _write_curve(qe_path, qe, wavelengths_nm=coarse_nm)
    _write_cfa(cfa_path, cfa, wavelengths_nm=coarse_nm)
    _write_curve(ir_path, ir, wavelengths_nm=coarse_nm)

    projection = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    expected_qe = 0.4 + (GRID_NM - 360.0) / 1000.0
    np.testing.assert_allclose(projection[0], expected_qe * 0.9 * 0.50, rtol=1e-12)
    np.testing.assert_allclose(projection[1], expected_qe * 0.9 * 0.60, rtol=1e-12)
    np.testing.assert_allclose(projection[2], expected_qe * 0.9 * 0.70, rtol=1e-12)


def test_projection_is_deterministic(tmp_path: Path) -> None:
    qe_path, cfa_path, ir_path, *_ = _projection_files(tmp_path)

    first = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)
    second = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    np.testing.assert_array_equal(first, second)


def test_projection_clamps_numerical_overhang_at_grid_edges(tmp_path: Path) -> None:
    # The source curves start and stop a few tenths of a nanometre inside the
    # grid. That is within the documented coverage tolerance, so the projection
    # must clamp to the endpoint values instead of extrapolating.
    source_nm = np.array([360.0000004, 500.0, 650.0, 829.9999996])
    qe = np.array([0.20, 0.60, 0.70, 0.40])
    ir = np.array([0.90, 0.95, 0.96, 0.03])
    cfa = {
        "R": np.array([0.10, 0.80, 0.90, 0.12]),
        "G": np.array([0.11, 0.82, 0.91, 0.13]),
        "B": np.array([0.12, 0.84, 0.92, 0.14]),
    }
    qe_path = tmp_path / "qe.csv"
    cfa_path = tmp_path / "cfa.csv"
    ir_path = tmp_path / "ir_cut.csv"
    _write_curve(qe_path, qe, wavelengths_nm=source_nm)
    _write_cfa(cfa_path, cfa, wavelengths_nm=source_nm)
    _write_curve(ir_path, ir, wavelengths_nm=source_nm)

    projection = spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    # Grid endpoints lie outside the measured range, so they take the first and
    # last measured values (no extrapolation).
    assert projection[0, 0] == pytest.approx(0.20 * 0.90 * 0.10)
    assert projection[0, -1] == pytest.approx(0.40 * 0.03 * 0.12)
    assert projection[2, 0] == pytest.approx(0.20 * 0.90 * 0.12)
    assert projection[2, -1] == pytest.approx(0.40 * 0.03 * 0.14)


def test_projection_rejects_uncovered_grid(tmp_path: Path) -> None:
    qe = _qe_curve()[8:-8]
    ir = _ir_cut_curve()[8:-8]
    cfa = {name: values[8:-8] for name, values in _cfa_curves().items()}
    qe_path = tmp_path / "qe.csv"
    cfa_path = tmp_path / "cfa.csv"
    ir_path = tmp_path / "ir_cut.csv"
    _write_curve(qe_path, qe, wavelengths_nm=GRID_NM[8:-8])
    _write_cfa(cfa_path, cfa, wavelengths_nm=GRID_NM[8:-8])
    _write_curve(ir_path, ir, wavelengths_nm=GRID_NM[8:-8])

    with pytest.raises(ValueError, match="cover"):
        spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)


def test_projection_rejects_cfa_curve_not_covering_grid(tmp_path: Path) -> None:
    # QE and IR-cut cover the grid, so the CFA coverage check is the one that
    # must fire; a short CFA curve would otherwise be silently extrapolated.
    qe_path = tmp_path / "qe.csv"
    cfa_path = tmp_path / "cfa.csv"
    ir_path = tmp_path / "ir_cut.csv"
    _write_curve(qe_path, _qe_curve())
    _write_curve(ir_path, _ir_cut_curve())
    cfa = {name: values[8:-8] for name, values in _cfa_curves().items()}
    _write_cfa(cfa_path, cfa, wavelengths_nm=GRID_NM[8:-8])

    with pytest.raises(ValueError, match="CFA curve"):
        spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)


def test_projection_rejects_dead_channel(tmp_path: Path) -> None:
    curves = _cfa_curves()
    curves["B"] = np.zeros(GRID_NM.size)
    qe_path, cfa_path, ir_path, *_ = _projection_files(tmp_path, cfa=curves)

    with pytest.raises(ValueError, match="zero integrated response"):
        spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)


@pytest.mark.parametrize(
    ("curve_name", "bad_curve"),
    [
        ("qe", np.full(GRID_NM.size, -0.1)),
        ("ir", np.full(GRID_NM.size, np.nan)),
        ("qe", np.full(GRID_NM.size, np.inf)),
    ],
)
def test_projection_rejects_invalid_curve_values(
    tmp_path: Path,
    curve_name: str,
    bad_curve: np.ndarray,
) -> None:
    kwargs = {curve_name: bad_curve}
    qe_path, cfa_path, ir_path, *_ = _projection_files(tmp_path, **kwargs)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"finite|non-negative|negative"):
        spectral_projection(qe_path, cfa_path, ir_path, GRID, CFAPattern.RGGB)


def test_projection_rejects_malformed_csvs(tmp_path: Path) -> None:
    _qe_path, cfa_path, ir_path, *_ = _projection_files(tmp_path)

    with pytest.raises(ValueError, match="cannot read"):
        spectral_projection(tmp_path / "missing.csv", cfa_path, ir_path, GRID, CFAPattern.RGGB)

    one_sample = tmp_path / "one_sample.csv"
    one_sample.write_text("wavelength_nm,value\n500,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least two"):
        spectral_projection(one_sample, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    decreasing = tmp_path / "decreasing.csv"
    decreasing.write_text("wavelength_nm,value\n600,0.5\n500,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        spectral_projection(decreasing, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    three_columns = tmp_path / "three_columns.csv"
    three_columns.write_text(
        "wavelength_nm,value,other\n500,0.5,0.1\n600,0.5,0.1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="two columns"):
        spectral_projection(three_columns, cfa_path, ir_path, GRID, CFAPattern.RGGB)

    non_numeric = tmp_path / "non_numeric.csv"
    non_numeric.write_text("wavelength_nm,value\n500,0.5\nnot-a-number,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not numeric"):
        spectral_projection(non_numeric, cfa_path, ir_path, GRID, CFAPattern.RGGB)


def test_projection_rejects_cfa_with_missing_or_extra_channels(tmp_path: Path) -> None:
    qe_path, _cfa_path, ir_path, *_ = _projection_files(tmp_path)
    curves = _cfa_curves()

    missing = tmp_path / "cfa_missing.csv"
    _write_cfa(missing, {"R": curves["R"], "G": curves["G"]})
    with pytest.raises(ValueError, match=r"CFA|channel"):
        spectral_projection(qe_path, missing, ir_path, GRID, CFAPattern.RGGB)

    extra = tmp_path / "cfa_extra.csv"
    _write_cfa(extra, {**curves, "X": curves["R"]})
    with pytest.raises(ValueError, match=r"CFA|channel"):
        spectral_projection(qe_path, extra, ir_path, GRID, CFAPattern.RGGB)


def test_projection_rejects_invalid_pattern(tmp_path: Path) -> None:
    qe_path, cfa_path, ir_path, *_ = _projection_files(tmp_path)

    with pytest.raises(ValueError, match="pattern"):
        spectral_projection(qe_path, cfa_path, ir_path, GRID, "YCMG")
