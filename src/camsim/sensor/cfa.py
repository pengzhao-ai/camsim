"""Colour-filter-array tiling and spectral projection matrices.

A colour sensor overlays a periodic filter mosaic on a monochrome photodiode
array. Each pixel integrates the scene spectrum through one filter, so a single
scalar RAW sample carries the response of one colour channel. The 2x2 Bayer
tile is the dominant consumer layout: one red, two green, and one blue site,
named row-major from the top-left corner (``RGGB``, ``BGGR``, ``GRBG``,
``GBRG``). ``RCCC`` is an automotive layout with one red and three clear sites
per tile, and ``MONO`` has no filters at all. Getting the tile origin or row
order wrong swaps red and blue or offsets the demosaic grid, producing colour
casts that no later ISP stage can recover because the spatial prior they rely
on is wrong.

For one CFA channel ``c`` the spectral projection matrix is

``P_c(lambda) = QE(lambda) * T_IR(lambda) * T_CFA,c(lambda)``

where every factor is dimensionless. ``QE`` is the silicon quantum efficiency
(electrons per incident photon), ``T_IR`` is the IR-cut filter transmittance,
and ``T_CFA,c`` is that colour filter's transmittance. The sensor later forms
the electron rate

``electron_rate_c = A * integral P_c(lambda) * E(lambda) * lambda / (h*c) dlambda``

with ``E`` in W/m2/nm, ``A`` the effective pixel area, and ``lambda/(h*c)`` the
inverse photon energy. Those factors (photon energy, area, exposure, lens
transmission, microlens gain) are deliberately *not* applied here: this module
returns the cacheable, dimensionless per-wavelength channel response, so a
sensor configuration and a scene spectrum can change independently and the
matrix can be audited against datasheets.

Real-world ranges: backside-illuminated CMOS QE peaks around 0.7-0.9 in the
green, with frontside parts losing blue response; colour-filter peak
transmission is roughly 0.6-0.95, and out-of-band absorption plus the mosaic's
spectral selection means a Bayer part collects less signal than a mono part of
the same silicon, typically by a factor of 1.5-2. IR-cut filters pass the
visible band and cut on near 650-700 nm with two to four decades of rejection
in the near infrared. QE, CFA, and IR-cut curves are normally tabulated from
350-1000 nm at 1-10 nm spacing; camsim interpolates them onto the shared
360-830 nm, 5 nm grid.

File contract. QE and IR-cut CSVs are two-column tables
``wavelength_nm,value``. A CFA CSV is either two-column (one neutral curve
applied to every colour of the pattern) or has one column per CFA colour, for
example ``wavelength_nm,R,G,B`` for Bayer or ``wavelength_nm,R,C`` for RCCC.
Headers are optional; ``#`` comment lines and blank lines are ignored. A
multi-column CFA file without a header is read in the canonical channel order
documented below, while a header naming the columns may appear in any order.
The source curves must cover the target wavelength grid; interpolation is
linear with endpoint clamping, and clamping only absorbs sub-nanometre
numerical round-off at the grid edges. Extrapolating a measured QE or filter
curve beyond its measured range would invent sensitivity the part does not
have, which is especially damaging in the near infrared where silicon and
IR-cut curves change by orders of magnitude.

Channel order is fixed and independent of the spatial tile: every Bayer pattern
returns rows ``R, G, B``, MONO returns ``M``, and RCCC returns ``R, C``. The
pattern therefore selects the filter set and the mosaic geometry, not a
different row ordering.

Failure artefacts guarded here: a transposed or one-pixel-offset Bayer tile
swaps colours; masks that overlap or leave gaps double-count or drop pixels;
mismatched mosaic plane shapes or missing colour keys silently lose a channel;
multiplying by QE, CFA, or IR-cut twice produces RAW values that are wrong by
orders of magnitude; a percent-valued QE curve (for example 85 instead of
0.85) produces implausibly large projections and is a units error the caller
must correct; a zero or dead channel, non-monotonic wavelengths, or a
malformed CSV must raise rather than be coerced. The loader validates
structure, finiteness, non-negativity, and grid coverage, and the projection
validates finite, non-negative rows with positive integrated response.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.core.frames import CFAPattern
from camsim.core.spectral import WavelengthGrid

__all__ = [
    "CFAColor",
    "cfa_color_at",
    "cfa_masks",
    "mosaic_from_channels",
    "spectral_projection",
]

CFAColor: TypeAlias = Literal["R", "G", "B", "C", "M"]
"""Colour labels used by camsim's CFA patterns."""

# Each 2x2 tile is written row-major from the top-left pixel, matching the
# convention CFAPattern documents. RCCC places its single red site at (0, 0);
# the remaining three sites are clear. MONO is tiled as a single colour so the
# mask code path has no special case.
_CFA_TILES: Final[dict[CFAPattern, tuple[tuple[CFAColor, CFAColor], tuple[CFAColor, CFAColor]]]] = {
    CFAPattern.RGGB: (("R", "G"), ("G", "B")),
    CFAPattern.BGGR: (("B", "G"), ("G", "R")),
    CFAPattern.GRBG: (("G", "R"), ("B", "G")),
    CFAPattern.GBRG: (("G", "B"), ("R", "G")),
    CFAPattern.RCCC: (("R", "C"), ("C", "C")),
    CFAPattern.MONO: (("M", "M"), ("M", "M")),
}

# The channel order returned by masks and projection matrices. Keeping this
# per-pattern mapping explicit prevents a spatial tile change (for example
# RGGB to BGGR) from silently reordering spectral rows.
_CHANNELS_BY_PATTERN: Final[dict[CFAPattern, tuple[CFAColor, ...]]] = {
    CFAPattern.RGGB: ("R", "G", "B"),
    CFAPattern.BGGR: ("R", "G", "B"),
    CFAPattern.GRBG: ("R", "G", "B"),
    CFAPattern.GBRG: ("R", "G", "B"),
    CFAPattern.MONO: ("M",),
    CFAPattern.RCCC: ("R", "C"),
}

# Coverage is a physical containment check. A 1e-6 nm allowance absorbs
# floating-point round-off in grids built with np.arange/np.linspace and lets
# np.interp clamp the last few tenths of a nanometre instead of extrapolating.
_COVERAGE_TOL_NM: Final[float] = 1e-6


def _coerce_pattern(pattern: CFAPattern | str) -> CFAPattern:
    """Return ``pattern`` as a :class:`CFAPattern`, accepting member strings."""
    if isinstance(pattern, CFAPattern):
        return pattern
    if isinstance(pattern, str):
        try:
            return CFAPattern(pattern)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in CFAPattern)
            raise ValueError(f"cfa pattern must be one of {allowed}; got {pattern!r}") from exc
    raise ValueError(f"cfa pattern must be a CFAPattern; got {type(pattern).__name__}")


def _coerce_dimension(value: object, field: str) -> int:
    """Return a positive integer dimension, rejecting bools and floats."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"shape {field} must be a positive integer; got {value!r}")
    dimension = int(value)
    if dimension <= 0:
        raise ValueError(f"shape {field} must be a positive integer; got {dimension}")
    return dimension


def _coerce_shape(shape: object) -> tuple[int, int]:
    """Return an ``(height, width)`` pair of positive Python integers."""
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise ValueError(f"shape must be a (height, width) sequence; got {shape!r}")
    dimensions = tuple(shape)
    if len(dimensions) != 2:
        raise ValueError(
            f"shape must contain exactly two dimensions (height, width); got {len(dimensions)}"
        )
    height = _coerce_dimension(dimensions[0], "height")
    width = _coerce_dimension(dimensions[1], "width")
    return height, width


def _coerce_index(value: object, field: str) -> int:
    """Return a non-negative integer pixel coordinate."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} coordinate must be a non-negative integer; got {value!r}")
    index = int(value)
    if index < 0:
        raise ValueError(f"{field} coordinate must be a non-negative integer; got {index}")
    return index


def cfa_color_at(pattern: CFAPattern | str, y: int, x: int) -> CFAColor:
    """Return the CFA colour at pixel ``(y, x)``.

    Parameters
    ----------
    pattern:
        A :class:`~camsim.core.frames.CFAPattern` or its string value. The 2x2
        tile is repeated in both directions, so any non-negative pixel
        coordinate is valid; odd image heights and widths simply crop the tile.
    y:
        Row index, zero based and non-negative.
    x:
        Column index, zero based and non-negative.

    Returns
    -------
    One of ``"R"``, ``"G"``, ``"B"``, ``"C"``, or ``"M"``. Bayer patterns
    always return the colour at the requested spatial site; MONO always
    returns ``"M"``.
    """
    tile = _CFA_TILES[_coerce_pattern(pattern)]
    row = _coerce_index(y, "y")
    column = _coerce_index(x, "x")
    return tile[row % 2][column % 2]


def cfa_masks(
    pattern: CFAPattern | str,
    shape: Sequence[int],
) -> dict[str, NDArray[np.bool_]]:
    """Return a boolean mask for every CFA colour present in the pattern.

    Parameters
    ----------
    pattern:
        A :class:`~camsim.core.frames.CFAPattern` or its string value.
    shape:
        ``(height, width)`` of the sensor in pixels. Both dimensions must be
        positive integers; odd dimensions crop the 2x2 tile.

    Returns
    -------
    A dict keyed by CFA colour in canonical order (``R, G, B`` for Bayer,
    ``M`` for MONO, ``R, C`` for RCCC). Every mask has ``shape`` and dtype
    ``bool``, and the masks partition the image: each pixel belongs to exactly
    one mask. Both green sites of a Bayer tile share the single ``G`` mask
    because the sensor and ISP treat them as one colour channel.

    A missing or double-counted site would bias white balance and demosaic
    interpolation, so the caller can rely on the masks being exhaustive and
    mutually exclusive.
    """
    resolved = _coerce_pattern(pattern)
    height, width = _coerce_shape(shape)
    masks: dict[str, NDArray[np.bool_]] = {
        colour: np.zeros((height, width), dtype=np.bool_)
        for colour in _CHANNELS_BY_PATTERN[resolved]
    }
    tile = _CFA_TILES[resolved]
    for row_offset in range(2):
        for column_offset in range(2):
            masks[tile[row_offset][column_offset]][row_offset::2, column_offset::2] = True
    return masks


def mosaic_from_channels(
    channels: Mapping[str, ArrayLike],
    cfa: CFAPattern | str,
) -> NDArray[Any]:
    """Interleave per-colour channel planes into a 2-D CFA mosaic.

    Parameters
    ----------
    channels:
        Mapping from CFA colour to a 2-D ``(H, W)`` plane in linear sensor
        units (electrons, volts, or digital numbers, depending on the caller).
        Bayer requires exactly ``R``, ``G``, and ``B``; MONO requires ``M``;
        RCCC requires ``R`` and ``C``. Planes must share one shape and contain
        finite real values. Missing and unexpected keys are rejected so a typo
        cannot silently drop a channel.
    cfa:
        A :class:`~camsim.core.frames.CFAPattern` or its string value.

    Returns
    -------
    A new 2-D array whose dtype is the NumPy promotion of the input planes.
    Each pixel takes the value of the plane selected by
    :func:`cfa_color_at`, which is the forward operation needed to simulate a
    sensor readout from separate colour images.

    Negative values are permitted because a black-level-corrected or noisy
    linear signal can be negative; non-finite values are not, because they
    would poison every downstream statistic.
    """
    resolved = _coerce_pattern(cfa)
    expected = _CHANNELS_BY_PATTERN[resolved]
    if not isinstance(channels, Mapping):
        raise ValueError(
            "channels must be a mapping of CFA colour to a 2-D plane; "
            f"got {type(channels).__name__}"
        )

    keys = set(channels)
    missing = [colour for colour in expected if colour not in keys]
    unexpected = [key for key in keys if key not in expected]
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(repr(key) for key in unexpected)}")
        raise ValueError(
            f"channels for {resolved.value} must be exactly {', '.join(expected)}; "
            f"got {'; '.join(details)}"
        )

    planes: dict[CFAColor, NDArray[Any]] = {}
    for colour in expected:
        plane = np.asarray(channels[colour])
        if plane.ndim != 2:
            raise ValueError(
                f"channel plane {colour!r} must be a 2-D (H, W) array; got shape {plane.shape}"
            )
        if plane.size == 0:
            raise ValueError(f"channel plane {colour!r} must contain at least one pixel")
        if not np.issubdtype(plane.dtype, np.number) or np.issubdtype(
            plane.dtype, np.complexfloating
        ):
            raise ValueError(
                f"channel plane {colour!r} must contain real numeric values; "
                f"got dtype {plane.dtype}"
            )
        if not np.all(np.isfinite(plane)):
            raise ValueError(f"channel plane {colour!r} must be finite (no NaN or infinity)")
        planes[colour] = plane

    shapes = {plane.shape for plane in planes.values()}
    if len(shapes) != 1:
        raise ValueError(
            "channel planes must have the same shape; got "
            + ", ".join(f"{colour!r}: {planes[colour].shape}" for colour in expected)
        )
    shape = planes[expected[0]].shape
    dtype = np.result_type(*(plane.dtype for plane in planes.values()))
    mosaic = np.empty(shape, dtype=dtype)
    # Slice directly from the 2x2 tile instead of materialising one boolean
    # mask per colour: a 50 MP frame would otherwise need hundreds of
    # megabytes of temporary mask storage for no additional correctness.
    tile = _CFA_TILES[resolved]
    for row_offset in range(2):
        for column_offset in range(2):
            colour = tile[row_offset][column_offset]
            mosaic[row_offset::2, column_offset::2] = planes[colour][
                row_offset::2, column_offset::2
            ]
    return mosaic


def _resolve_path(path: str | Path) -> Path:
    """Expand a user-supplied spectral CSV path."""
    return Path(path).expanduser()


def _is_numeric_row(columns: tuple[str, ...]) -> bool:
    """Return ``True`` when every column parses as a float."""
    try:
        for column in columns:
            float(column)
    except ValueError:
        return False
    return True


def _read_table(
    path: Path,
    description: str,
) -> tuple[tuple[str, ...] | None, NDArray[np.float64]]:
    """Read a comment-tolerant, optionally headed numeric CSV table.

    The first non-comment row is treated as a header when any of its columns
    is non-numeric; otherwise every row is data. Data rows must be rectangular,
    the first column must be a strictly increasing positive wavelength axis,
    and every value column must be finite and non-negative.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"cannot read {description} CSV {path}: {exc}") from exc

    rows: list[tuple[int, tuple[str, ...]]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append((line_number, tuple(column.strip() for column in line.split(","))))

    if not rows:
        raise ValueError(
            f"{description} CSV {path} must contain at least two wavelength samples; got 0"
        )

    header: tuple[str, ...] | None = None
    data_start = 0
    if not _is_numeric_row(rows[0][1]):
        header = rows[0][1]
        data_start = 1

    data_rows = rows[data_start:]
    if len(data_rows) < 2:
        raise ValueError(
            f"{description} CSV {path} must contain at least two wavelength samples; "
            f"got {len(data_rows)}"
        )

    width = len(data_rows[0][1])
    if width < 2:
        raise ValueError(
            f"{description} CSV must have at least two columns (wavelength_nm,value); "
            f"got {width} in {path}"
        )

    numeric_rows: list[list[float]] = []
    for line_number, columns in data_rows:
        if len(columns) != width:
            raise ValueError(
                f"{description} CSV line {line_number} has {len(columns)} columns; expected {width}"
            )
        try:
            numeric_rows.append([float(column) for column in columns])
        except ValueError:
            joined = ", ".join(columns)
            raise ValueError(
                f"{description} CSV line {line_number} is not numeric: {joined!r}"
            ) from None

    data = np.asarray(numeric_rows, dtype=np.float64)
    wavelengths = data[:, 0]
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError(f"{description} CSV wavelengths must be finite and positive")
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError(f"{description} CSV wavelengths must be strictly increasing")
    values = data[:, 1:]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{description} CSV values must be finite")
    if np.any(values < 0.0):
        raise ValueError(f"{description} CSV values must be non-negative")
    return header, data


def _read_single_curve(
    path: Path,
    description: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read a two-column ``(wavelength_nm, value)`` CSV curve."""
    _header, data = _read_table(path, description)
    if data.shape[1] != 2:
        raise ValueError(
            f"{description} CSV must have two columns (wavelength_nm,value); "
            f"got {data.shape[1]} in {path}"
        )
    return data[:, 0], data[:, 1]


def _read_cfa_curves(
    path: Path,
    pattern: CFAPattern,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read CFA transmission curves and order them by canonical channel.

    Two-column files carry a single neutral curve that is broadcast to every
    colour of the pattern. Wider files carry one curve per CFA colour; a header
    may name the columns in any order, while a headerless file is read in the
    canonical order for the pattern.
    """
    channels = _CHANNELS_BY_PATTERN[pattern]
    header, data = _read_table(path, "CFA")
    width = data.shape[1]
    value_columns = width - 1

    if value_columns == 1:
        wavelengths = np.asarray(data[:, 0], dtype=np.float64)
        curve = np.asarray(data[:, 1], dtype=np.float64)
        return wavelengths, np.asarray(np.tile(curve, (len(channels), 1)), dtype=np.float64)

    if value_columns != len(channels):
        expected = ", ".join(channels)
        raise ValueError(
            "CFA CSV must have either two columns (one neutral curve) or one column "
            f"per CFA channel ({expected}) for {pattern.value}; got {width} columns in {path}"
        )

    values = np.asarray(data[:, 1:], dtype=np.float64)
    if header is not None:
        if len(header) != width:
            raise ValueError(f"CFA CSV header has {len(header)} columns but data rows have {width}")
        names = [name.strip().upper() for name in header[1:]]
        expected_names = list(channels)
        if sorted(names) != sorted(expected_names):
            raise ValueError(
                f"CFA CSV columns {names} do not match the CFA channels "
                f"{expected_names} of {pattern.value}"
            )
        order = [names.index(name) for name in expected_names]
        values = values[:, order]

    return (
        np.asarray(data[:, 0], dtype=np.float64),
        np.ascontiguousarray(values.T, dtype=np.float64),
    )


def _require_coverage(
    source_nm: NDArray[np.float64],
    target_nm: NDArray[np.float64],
    description: str,
) -> None:
    """Raise when a measured curve does not span the target wavelength grid."""
    source_start = float(source_nm[0])
    source_stop = float(source_nm[-1])
    target_start = float(target_nm[0])
    target_stop = float(target_nm[-1])
    if (
        target_start < source_start - _COVERAGE_TOL_NM
        or target_stop > source_stop + _COVERAGE_TOL_NM
    ):
        raise ValueError(
            f"{description} curve covers {source_start:g}-{source_stop:g} nm and does not "
            f"cover the target grid {target_start:g}-{target_stop:g} nm"
        )


def spectral_projection(
    qe_csv: str | Path,
    cfa_csv: str | Path,
    ir_cut_csv: str | Path,
    grid: WavelengthGrid | ArrayLike,
    cfa: CFAPattern | str,
) -> NDArray[np.float64]:
    """Build per-channel spectral response matrices from measured curves.

    Parameters
    ----------
    qe_csv:
        Two-column ``wavelength_nm,qe`` CSV. QE is dimensionless, usually in
        ``[0, 1]``; a percent-valued file such as ``85`` instead of ``0.85`` is
        a units error the caller must correct.
    cfa_csv:
        CFA transmittance CSV. For MONO this path is ignored because there is
        no colour filter. Otherwise it is either two-column (one neutral curve
        broadcast to every colour) or has one column per colour of the pattern
        (``R,G,B`` for Bayer, ``R,C`` for RCCC). Columns may be header-named
        in any order; a headerless multi-column file uses canonical order.
    ir_cut_csv:
        Two-column ``wavelength_nm,transmittance`` CSV for the IR-cut filter.
    grid:
        The :class:`~camsim.core.spectral.WavelengthGrid` to sample, or an
        array-like of nanometres validated into one. All source curves must
        cover this grid's range.
    cfa:
        CFA pattern, as a :class:`~camsim.core.frames.CFAPattern` or string.

    Returns
    -------
    A ``(n_channels, n_wavelengths)`` float64 matrix in canonical channel
    order: ``R, G, B`` for every Bayer pattern, ``M`` for MONO, and ``R, C``
    for RCCC. Entries are the dimensionless product
    ``QE * T_IR * T_CFA`` sampled on ``grid``. Photon energy, pixel area,
    exposure, lens transmission, and microlens gain are intentionally absent;
    the sensor applies them when it integrates a scene spectrum.

    Raises
    ------
    ValueError
        If a file cannot be read, has a malformed table or header, contains
        non-finite, negative, or non-monotonic data, does not cover the target
        grid, or produces a channel with zero integrated response. These
        conditions otherwise yield plausible-looking but physically wrong
        electron counts.
    """
    target = (
        grid
        if isinstance(grid, WavelengthGrid)
        else WavelengthGrid(np.asarray(grid, dtype=np.float64))
    )
    pattern = _coerce_pattern(cfa)

    qe_nm, qe_values = _read_single_curve(_resolve_path(qe_csv), "QE")
    ir_nm, ir_values = _read_single_curve(_resolve_path(ir_cut_csv), "IR-cut")
    _require_coverage(qe_nm, target.nm, "QE")
    _require_coverage(ir_nm, target.nm, "IR-cut")

    # np.interp clamps to the endpoint values outside the source range. The
    # coverage check above keeps that fallback to sub-nanometre round-off, so
    # the matrix never encodes invented out-of-band sensitivity.
    qe_on_grid = np.asarray(np.interp(target.nm, qe_nm, qe_values), dtype=np.float64)
    ir_on_grid = np.asarray(np.interp(target.nm, ir_nm, ir_values), dtype=np.float64)
    common_response = qe_on_grid * ir_on_grid
    channels = _CHANNELS_BY_PATTERN[pattern]

    if pattern is CFAPattern.MONO:
        # Mono has no colour filters: it collects the bare silicon QE through
        # the IR-cut filter and therefore exceeds each Bayer channel.
        projection = common_response[None, :]
    else:
        cfa_nm, cfa_values = _read_cfa_curves(_resolve_path(cfa_csv), pattern)
        _require_coverage(cfa_nm, target.nm, "CFA")
        cfa_on_grid = np.asarray(
            [np.interp(target.nm, cfa_nm, row) for row in cfa_values],
            dtype=np.float64,
        )
        projection = cfa_on_grid * common_response[None, :]

    projection = np.ascontiguousarray(projection, dtype=np.float64)
    if not np.all(np.isfinite(projection)):
        raise ValueError("spectral projection contains non-finite values")
    if np.any(projection < 0.0):
        raise ValueError("spectral projection contains negative values")
    for channel, row_sum in zip(channels, projection.sum(axis=1), strict=True):
        if not row_sum > 0.0:
            raise ValueError(
                f"channel {channel} has zero integrated response; "
                "check the QE, CFA, and IR-cut curves"
            )
    return projection
