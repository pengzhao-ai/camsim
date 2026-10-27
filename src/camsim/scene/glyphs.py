"""Deterministic spectral glyph targets for machine-vision experiments.

A glyph target is an image-plane synthetic scene, not a three-dimensional
object: a character is painted as dark ink on light paper, and each material is
described by a spectral reflectance factor :math:`\\rho(\\lambda)`, a
dimensionless property in ``[0, 1]`` sampled on the common 360-830 nm, 5 nm
grid shared by the rest of the scene layer. A FreeType coverage mask
:math:`m(x, y) \\in [0, 1]` from an antialiased TrueType raster combines the two
materials by linear mixture::

    rho(x, y, lambda) = paper(lambda)
                        + (ink(lambda) - paper(lambda)) * m(x, y)

The mixture is justified because a sensor pixel integrates the reflected flux
over its finite area: a pixel half-covered by ink reflects half the paper flux
and half the ink flux. The mask is therefore a *coverage fraction*, not a
display-space intensity, and must never be gamma-encoded before it multiplies
reflectance. Reflectance is not colour either; it becomes sensor-plane
irradiance only after multiplication by the scene illuminant and the lens
formula in :mod:`camsim.scene.targets`.

Rendering and oversampling
--------------------------
Glyphs are rasterised on a grid ``oversampling`` times finer than the sensor in
each axis (four by default) so that thin strokes, diagonals, and the
antialiased edge itself carry sub-pixel geometry. The returned spectral cube is
on that oversampled grid; the downstream scene renderer box-averages it to the
sensor grid together with the PSF. Rendering at the sensor grid instead aliases
stroke edges into the passband, and nearest-neighbour rasterisation removes the
partial coverage that makes a glyph's edge look like a real lens image rather
than a binary stencil.

Randomization and determinism
-----------------------------
Each glyph gets a seeded random font size (50-70 % of the shorter frame side),
position (inside a margin that keeps the rotated glyph in frame), and rotation
in ``[-10, +10]`` degrees; the rotation bound models a printed sheet or a
handheld capture that is not perfectly aligned, while larger tilts are outside
the intended OCR scenario. Passing ``rotation_deg`` overrides the random draw
while keeping the size and position stream unchanged. All randomness comes from
``numpy.random.default_rng`` seeded per sample: there is no global RNG, so
repeated renders are bit-identical for a fixed Pillow/FreeType build. The
dataset additionally derives train and test samples from separate seed streams
and encodes the split in the sample-seed parity, which makes the two seed sets
disjoint by construction rather than merely unlikely to collide.

Default spectra and provenance
------------------------------
The default paper and ink are the measured ColorChecker Classic white
(``"white 9.5 (.05 D)"``) and black (``"black 2 (1.5 D)"``) patches loaded by
:func:`camsim.scene.spectral_data.load_colorchecker`. The white patch's
measured deep-violet roll-off (about 0.19 at 360 nm) is a real optical-
brightener property of the chart, not a rendering defect. Reflectances are
documented in :class:`GlyphDataset.provenance`, including the BabelColor
dataset and the rasteriser/font used. Custom spectra are accepted only when
they are one-dimensional, finite, in ``[0, 1]``, and sampled on the same
95-wavelength grid.

Real-world parameter ranges: glyph bodies in OCR datasets span roughly 20-60
sensor pixels, printed paper reflectance is 0.80-0.95, black ink reflectance is
0.02-0.05, and page skew of a few degrees is common. Fonts vary widely; this
layer fixes DejaVu Sans (the font bundled with Matplotlib) so class shapes are
reproducible across machines.

Failure artefacts guarded here: a font size that exceeds the frame clips the
glyph; a rotation beyond the documented bound can leave the frame; rendering
without antialiasing yields binary edges and inflates apparent task contrast;
applying the mask after gamma encoding produces the wrong ink density; spectra
of the wrong length silently shift the colour or fail far downstream; using a
global RNG makes two nominally identical datasets differ; and storing a full
spectral cube per sample costs tens of megabytes for a toy dataset, so the
dataset stores float16 masks and materialises cubes on demand.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

import matplotlib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageDraw, ImageFont

from camsim.scene.spectral_data import COMMON_WAVELENGTH_GRID, load_colorchecker

__all__ = [
    "DEFAULT_INK_PATCH",
    "DEFAULT_PAPER_PATCH",
    "GLYPH_CLASSES",
    "MAX_ROTATION_DEG",
    "GlyphDataset",
    "glyph_dataset",
    "render_glyph",
    "render_glyph_mask",
]

GlyphSplit: TypeAlias = Literal["train", "test"]

GLYPH_CLASSES: Final[tuple[str, ...]] = tuple("0123456789") + tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
"""The 36 OCR classes: digits 0-9 followed by uppercase letters A-Z."""

DEFAULT_PAPER_PATCH: Final[str] = "white 9.5 (.05 D)"
"""ColorChecker patch used as the default paper reflectance."""

DEFAULT_INK_PATCH: Final[str] = "black 2 (1.5 D)"
"""ColorChecker patch used as the default ink reflectance."""

MAX_ROTATION_DEG: Final[float] = 10.0
"""Largest absolute glyph rotation, in degrees, allowed or drawn at random."""

_DEFAULT_OVERSAMPLING: Final[int] = 4
_MIN_FONT_FRACTION: Final[float] = 0.50
_MAX_FONT_FRACTION: Final[float] = 0.70
_FRAME_MARGIN_FRACTION: Final[float] = 0.04
_FONT_FILENAME: Final[str] = "DejaVuSans.ttf"
_FONT_DESCRIPTION: Final[str] = "DejaVu Sans (Matplotlib-bundled)"
_SPLIT_CODES: Final[dict[str, int]] = {"train": 0, "test": 1}
_CLASS_STREAM: Final[int] = 100
_SEED_STREAM: Final[int] = 200
_ROTATION_STREAM: Final[int] = 0
_LAYOUT_STREAM: Final[int] = 1


def _require_integer(value: object, field_name: str, minimum: int) -> int:
    """Return an integer at least ``minimum``, rejecting booleans and floats."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be an integer; got {value!r}")
    integer = int(value)
    if integer < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}; got {integer}")
    return integer


def _validate_shape(shape: object) -> tuple[int, int]:
    """Return a positive ``(height, width)`` integer pair."""
    if isinstance(shape, (str, bytes)):
        raise ValueError(f"shape must be a (height, width) pair; got {shape!r}")
    try:
        dimensions: tuple[Any, ...] = tuple(shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"shape must be a (height, width) pair; {exc}") from exc
    if len(dimensions) != 2:
        raise ValueError(f"shape must contain exactly two dimensions; got {shape!r}")
    return (
        _require_integer(dimensions[0], "shape dimensions", 1),
        _require_integer(dimensions[1], "shape dimensions", 1),
    )


def _validate_char(char: object) -> str:
    """Return one of the 36 supported glyph characters."""
    if not isinstance(char, str) or char not in GLYPH_CLASSES:
        raise ValueError(
            f"char must be one of the {len(GLYPH_CLASSES)} glyph classes 0-9 or A-Z; got {char!r}"
        )
    return char


def _validate_rotation(rotation_deg: object) -> float:
    """Return a finite rotation within the documented degree bound."""
    try:
        rotation = float(rotation_deg)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rotation_deg must be a finite number; {exc}") from exc
    if not math.isfinite(rotation) or abs(rotation) > MAX_ROTATION_DEG:
        raise ValueError(
            f"rotation_deg must be finite and within +/-{MAX_ROTATION_DEG:g} degrees; "
            f"got {rotation_deg!r}"
        )
    return rotation


def _validate_split(split: object) -> GlyphSplit:
    """Return the dataset split name or raise ValueError."""
    if split == "train":
        return "train"
    if split == "test":
        return "test"
    raise ValueError(f"split must be 'train' or 'test'; got {split!r}")


def _validate_spectrum(values: ArrayLike, field_name: str) -> NDArray[np.float64]:
    """Return a read-only reflectance spectrum on the common wavelength grid."""
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a 1-D reflectance spectrum; {exc}") from exc
    if array.ndim != 1:
        raise ValueError(
            f"{field_name} must be a 1-D reflectance spectrum; got shape {array.shape}"
        )
    expected = len(COMMON_WAVELENGTH_GRID)
    if array.size != expected:
        first_nm = float(COMMON_WAVELENGTH_GRID.nm[0])
        last_nm = float(COMMON_WAVELENGTH_GRID.nm[-1])
        step_nm = float(COMMON_WAVELENGTH_GRID.nm[1] - COMMON_WAVELENGTH_GRID.nm[0])
        raise ValueError(
            f"{field_name} must be sampled on the common {first_nm:g}-{last_nm:g} nm, "
            f"{step_nm:g} nm grid ({expected} samples); got {array.size} samples"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} reflectance factors must be finite (no NaN or infinity)")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{field_name} reflectance factors must lie in [0, 1]")
    array.setflags(write=False)
    return array


@lru_cache(maxsize=1)
def _default_paper_ink() -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Return read-only default paper/ink spectra and their provenance.

    The ColorChecker white and black patches are measured material spectra, so
    they carry more physical meaning than arbitrary flat reflectance constants.
    The loader validates the upstream patch order before these names are
    indexed, and the returned arrays share the common wavelength grid.
    """
    chart = load_colorchecker()
    try:
        paper_index = chart.patches.index(DEFAULT_PAPER_PATCH)
        ink_index = chart.patches.index(DEFAULT_INK_PATCH)
    except ValueError as exc:  # pragma: no cover - loader pins the chart order
        raise RuntimeError(
            "the ColorChecker dataset no longer exposes the documented glyph paper/ink patches"
        ) from exc
    paper = np.array(chart.spectra[paper_index], dtype=np.float64, copy=True)
    ink = np.array(chart.spectra[ink_index], dtype=np.float64, copy=True)
    paper.setflags(write=False)
    ink.setflags(write=False)
    provenance: dict[str, Any] = {
        "paper_patch": DEFAULT_PAPER_PATCH,
        "ink_patch": DEFAULT_INK_PATCH,
        "paper_patch_index": paper_index,
        "ink_patch_index": ink_index,
        "spectra_source": dict(chart.provenance),
        "wavelength_range_nm": (
            float(COMMON_WAVELENGTH_GRID.nm[0]),
            float(COMMON_WAVELENGTH_GRID.nm[-1]),
        ),
        "wavelength_step_nm": float(COMMON_WAVELENGTH_GRID.nm[1] - COMMON_WAVELENGTH_GRID.nm[0]),
        "default_retrieval": "ColorChecker white and black patch reflectance factors",
    }
    return paper, ink, provenance


@lru_cache(maxsize=1)
def _font_path() -> Path:
    """Return the Matplotlib-bundled DejaVu Sans font file.

    A silent font-manager fallback would substitute a different typeface and
    change every glyph's class shape, so a missing bundled font is an error
    rather than a search that happens to succeed on one machine.
    """
    candidate = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / _FONT_FILENAME
    if not candidate.is_file():
        raise RuntimeError(
            "the Matplotlib-bundled DejaVu Sans font is unavailable; "
            "glyph rendering requires it for reproducible class shapes"
        )
    return candidate


@lru_cache(maxsize=64)
def _font(size: int) -> ImageFont.FreeTypeFont:
    """Return a cached FreeType font at ``size`` pixels."""
    return ImageFont.truetype(str(_font_path()), int(size))


def _rng(seed: int, stream: int) -> np.random.Generator:
    """Return a generator for one independent, named stream of a sample seed."""
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def _render_mask_and_rotation(
    char: str,
    shape: tuple[int, int],
    seed: int,
    rotation_deg: float | None,
    oversampling: int,
) -> tuple[NDArray[np.float64], float]:
    """Rasterise one glyph and return its coverage mask and applied rotation.

    Rotation and layout use independent seed streams so that forcing an
    explicit rotation does not perturb the seeded font size and position.
    """
    height, width = shape
    canvas_height = height * oversampling
    canvas_width = width * oversampling
    if rotation_deg is None:
        rotation = float(_rng(seed, _ROTATION_STREAM).uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG))
    else:
        rotation = rotation_deg

    layout_rng = _rng(seed, _LAYOUT_STREAM)
    min_side = min(canvas_height, canvas_width)
    font_size = max(
        1,
        round(min_side * float(layout_rng.uniform(_MIN_FONT_FRACTION, _MAX_FONT_FRACTION))),
    )
    font = _font(font_size)
    left, top, right, bottom = font.getbbox(char)
    glyph_width = right - left
    glyph_height = bottom - top
    if glyph_width <= 0 or glyph_height <= 0:  # pragma: no cover - tiny canvases
        raise RuntimeError(
            f"font rasterisation produced an empty mask for {char!r} at {font_size} px; "
            "use a larger shape"
        )

    # Rotate the glyph about its own centre. The rotated axis-aligned extent of
    # a w x h rectangle at angle theta is (w|cos| + h|sin|, w|sin| + h|cos|);
    # sampling the centre inside that envelope plus a margin guarantees the
    # glyph never leaves the frame, even at the maximum rotation.
    theta = math.radians(abs(rotation))
    cos_theta = abs(math.cos(theta))
    sin_theta = abs(math.sin(theta))
    rotated_width = glyph_width * cos_theta + glyph_height * sin_theta
    rotated_height = glyph_width * sin_theta + glyph_height * cos_theta
    margin = _FRAME_MARGIN_FRACTION * min_side
    half_width = rotated_width / 2.0 + margin
    half_height = rotated_height / 2.0 + margin
    if 2.0 * half_width > canvas_width or 2.0 * half_height > canvas_height:
        raise RuntimeError(
            f"glyph {char!r} at {font_size} px with {rotation:g} degrees does not fit "
            f"a {canvas_height} x {canvas_width} rendering canvas"
        )
    center_x = float(layout_rng.uniform(half_width, canvas_width - half_width))
    center_y = float(layout_rng.uniform(half_height, canvas_height - half_height))

    image = Image.new("L", (canvas_width, canvas_height), 0)
    draw = ImageDraw.Draw(image)
    # Place the text so the ink bounding box, not the font origin, is centred.
    draw.text(
        (
            center_x - (left + glyph_width / 2.0),
            center_y - (top + glyph_height / 2.0),
        ),
        char,
        font=font,
        fill=255,
    )
    if rotation != 0.0:
        image = image.rotate(
            rotation,
            resample=Image.Resampling.BICUBIC,
            center=(center_x, center_y),
            fillcolor=0,
        )
    mask = np.asarray(image, dtype=np.float64) / 255.0
    return mask, rotation


def render_glyph_mask(
    char: str,
    shape: tuple[int, int],
    seed: int,
    rotation_deg: float | None = None,
    oversampling: int = _DEFAULT_OVERSAMPLING,
) -> NDArray[np.float64]:
    """Return a glyph's antialiased coverage mask on the oversampled grid.

    Parameters
    ----------
    char:
        One of :data:`GLYPH_CLASSES` (digits ``0``-``9`` or uppercase
        ``A``-``Z``). Lowercase letters and punctuation are rejected because
        they are not classes of the downstream classifier.
    shape:
        Sensor-grid ``(height, width)``. The returned mask has shape
        ``(height * oversampling, width * oversampling)``.
    seed:
        Non-negative integer seed. It selects the font size, position, and,
        unless overridden, the rotation. No global random state is used.
    rotation_deg:
        ``None`` draws a rotation uniformly in
        ``[-MAX_ROTATION_DEG, +MAX_ROTATION_DEG]`` from the seed. A finite
        value overrides the draw and is itself bounded by the same limit.
        Positive angles rotate the glyph counter-clockwise about its centre.
    oversampling:
        Positive integer samples per sensor pixel per axis. Four is the
        conventional rendering factor; one disables oversampling.

    Returns
    -------
    A float64 ``(H * oversampling, W * oversampling)`` array in ``[0, 1]``.
    Values strictly between zero and one are antialiased edge coverage; the
    glyph interior is one and the paper background is zero.

    Raises
    ------
    ValueError
        If the character, shape, seed, rotation, or oversampling is invalid.
    RuntimeError
        If the font or a non-empty glyph bitmap is unavailable at the drawn
        size, which indicates a broken font installation rather than bad input.
    """
    glyph = _validate_char(char)
    sensor_shape = _validate_shape(shape)
    seed_value = _require_integer(seed, "seed", 0)
    rotation = None if rotation_deg is None else _validate_rotation(rotation_deg)
    factor = _require_integer(oversampling, "oversampling", 1)
    mask, _applied_rotation = _render_mask_and_rotation(
        glyph, sensor_shape, seed_value, rotation, factor
    )
    return mask


def render_glyph(
    char: str,
    shape: tuple[int, int],
    seed: int,
    paper_spectrum: ArrayLike | None = None,
    ink_spectrum: ArrayLike | None = None,
    rotation_deg: float | None = None,
    oversampling: int = _DEFAULT_OVERSAMPLING,
) -> NDArray[np.float64]:
    """Render one glyph as a spectral reflectance cube on the oversampled grid.

    The antialiased coverage mask ``m`` mixes the two material spectra:
    ``rho = paper + (ink - paper) * m``. The mapping is performed in linear
    reflectance space; no display encoding or tone curve belongs here.

    Parameters
    ----------
    char:
        One of :data:`GLYPH_CLASSES`.
    shape:
        Sensor-grid ``(height, width)``. The returned cube has shape
        ``(height * oversampling, width * oversampling, L)`` with ``L = 95``.
    seed:
        Non-negative rendering seed (font size, position, rotation).
    paper_spectrum:
        Optional reflectance on the common 360-830 nm, 5 nm grid. ``None``
        uses the ColorChecker white patch (see :data:`DEFAULT_PAPER_PATCH`).
    ink_spectrum:
        Optional reflectance on the same grid. ``None`` uses the ColorChecker
        black patch (see :data:`DEFAULT_INK_PATCH`).
    rotation_deg:
        ``None`` draws a rotation in ``[-10, +10]`` degrees from ``seed``;
        an explicit value overrides the draw. Positive angles are
        counter-clockwise.
    oversampling:
        Positive integer samples per sensor pixel per axis.

    Returns
    -------
    A float64 ``(H * oversampling, W * oversampling, L)`` cube of reflectance
    factors in ``[0, 1]`` on :data:`camsim.scene.spectral_data.COMMON_WAVELENGTH_GRID`.

    Raises
    ------
    ValueError
        If the character, shape, seed, spectra, rotation, or oversampling is
        invalid, including spectra that are not sampled on the common grid.
    """
    glyph = _validate_char(char)
    sensor_shape = _validate_shape(shape)
    seed_value = _require_integer(seed, "seed", 0)
    rotation = None if rotation_deg is None else _validate_rotation(rotation_deg)
    factor = _require_integer(oversampling, "oversampling", 1)
    default_paper, default_ink, _provenance = _default_paper_ink()
    paper = (
        default_paper
        if paper_spectrum is None
        else _validate_spectrum(paper_spectrum, "paper_spectrum")
    )
    ink = default_ink if ink_spectrum is None else _validate_spectrum(ink_spectrum, "ink_spectrum")

    mask, _applied_rotation = _render_mask_and_rotation(
        glyph, sensor_shape, seed_value, rotation, factor
    )
    cube = paper[None, None, :] + (ink - paper)[None, None, :] * mask[:, :, None]
    return np.asarray(cube, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class GlyphDataset:
    """Memory-efficient deterministic collection of rendered glyph samples.

    Parameters
    ----------
    masks:
        ``(n, H * oversampling, W * oversampling)`` float16 coverage masks in
        ``[0, 1]``. Float16 bounds the error to about 5e-4 reflectance, well
        below sensor and font-rendering variation, while halving the footprint
        of a float32 mask.
    classes:
        ``(n,)`` integer class indices into :data:`GLYPH_CLASSES`.
    chars:
        ``(n,)`` character labels aligned with ``classes``.
    seeds:
        ``(n,)`` rendering seeds. Datasets produced by :func:`glyph_dataset`
        carry even seeds for ``"train"`` and odd seeds for ``"test"``, which
        makes the two sets disjoint.
    rotations:
        ``(n,)`` applied rotation in degrees, recorded because it is part of
        the sample's ground truth.
    paper_spectra, ink_spectra:
        ``(n, L)`` reflectance spectra per sample. They share the common
        wavelength grid; storing them per sample keeps customised material
        variants representable without changing the mask representation.
    shape:
        Sensor-grid ``(height, width)``.
    split:
        ``"train"`` or ``"test"``.
    oversampling:
        Integer samples per sensor pixel per axis used for every mask.
    provenance:
        Mapping recording the font, rasteriser, default ColorChecker spectra,
        rotation bounds, and seed scheme.

    A full spectral cube for n=20 samples at 64 x 64 sensor pixels and four
    samples per pixel is about 1 GB in float64; the stored masks are a few
    megabytes. :meth:`materialize` builds a cube only when a consumer needs
    one.
    """

    masks: NDArray[np.float16]
    classes: NDArray[np.int64]
    chars: tuple[str, ...]
    seeds: NDArray[np.int64]
    rotations: NDArray[np.float64]
    paper_spectra: NDArray[np.float64]
    ink_spectra: NDArray[np.float64]
    shape: tuple[int, int]
    split: GlyphSplit
    oversampling: int
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sensor_shape = _validate_shape(self.shape)
        factor = _require_integer(self.oversampling, "oversampling", 1)
        split = _validate_split(self.split)
        masks = np.asarray(self.masks)
        if masks.ndim != 3:
            raise ValueError(
                f"masks must be a 3-D (n, height, width) array; got shape {masks.shape}"
            )
        count = int(masks.shape[0])
        if count < 1:
            raise ValueError("a glyph dataset must contain at least one sample")
        expected = (count, sensor_shape[0] * factor, sensor_shape[1] * factor)
        if masks.shape != expected:
            raise ValueError(
                f"masks must have shape {expected} for sensor shape {sensor_shape} at "
                f"oversampling {factor}; got {masks.shape}"
            )
        if masks.dtype != np.float16:
            raise ValueError(f"masks must be float16; got dtype {masks.dtype}")
        if not np.all(np.isfinite(masks)) or np.any(masks < 0.0) or np.any(masks > 1.0):
            raise ValueError("masks must contain finite coverage values in [0, 1]")

        classes = np.asarray(self.classes)
        seeds = np.asarray(self.seeds)
        rotations = np.asarray(self.rotations)
        paper = np.asarray(self.paper_spectra)
        ink = np.asarray(self.ink_spectra)
        if classes.shape != (count,) or not np.issubdtype(classes.dtype, np.integer):
            raise ValueError(f"classes must be an integer array of shape ({count},)")
        if len(self.chars) != count:
            raise ValueError(f"chars must contain {count} entries; got {len(self.chars)}")
        if not all(isinstance(char, str) and char in GLYPH_CLASSES for char in self.chars):
            raise ValueError("chars must all be members of GLYPH_CLASSES")
        if seeds.shape != (count,) or not np.issubdtype(seeds.dtype, np.integer):
            raise ValueError(f"seeds must be an integer array of shape ({count},)")
        if np.any(seeds < 0):
            raise ValueError("seeds must be non-negative")
        if rotations.shape != (count,):
            raise ValueError(f"rotations must have shape ({count},)")
        if not np.all(np.isfinite(rotations)) or np.any(np.abs(rotations) > MAX_ROTATION_DEG):
            raise ValueError(f"rotations must be finite and within +/-{MAX_ROTATION_DEG:g} degrees")
        wavelength_count = len(COMMON_WAVELENGTH_GRID)
        for name, spectra in (("paper_spectra", paper), ("ink_spectra", ink)):
            if spectra.shape != (count, wavelength_count):
                raise ValueError(
                    f"{name} must have shape ({count}, {wavelength_count}) on the common "
                    f"wavelength grid; got {spectra.shape}"
                )
            if not np.all(np.isfinite(spectra)) or np.any(spectra < 0.0) or np.any(spectra > 1.0):
                raise ValueError(f"{name} must contain finite reflectance factors in [0, 1]")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")

        for array in (masks, classes, seeds, rotations, paper, ink):
            array.setflags(write=False)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "chars", tuple(self.chars))
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "paper_spectra", paper)
        object.__setattr__(self, "ink_spectra", ink)
        object.__setattr__(self, "shape", sensor_shape)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "oversampling", factor)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def n(self) -> int:
        """Number of samples in the dataset."""
        return int(self.masks.shape[0])

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n

    def materialize(self, index: int) -> NDArray[np.float64]:
        """Build one sample's spectral reflectance cube on demand.

        Parameters
        ----------
        index:
            Integer sample index in ``[0, len(self))``.

        Returns
        -------
        A fresh float64 ``(H * oversampling, W * oversampling, L)`` cube. The
        returned array is not shared with the dataset and may be modified.
        """
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError(f"index must be an integer; got {index!r}")
        position = int(index)
        if position < 0 or position >= self.n:
            raise ValueError(f"index must be in [0, {self.n}); got {index!r}")
        mask = self.masks[position].astype(np.float64)
        paper = self.paper_spectra[position]
        delta = self.ink_spectra[position] - paper
        cube = paper[None, None, :] + delta[None, None, :] * mask[:, :, None]
        return np.asarray(cube, dtype=np.float64)


def glyph_dataset(
    n: int,
    seed: int,
    shape: tuple[int, int],
    split: GlyphSplit = "train",
) -> GlyphDataset:
    """Generate a deterministic, memory-efficient glyph classification dataset.

    Parameters
    ----------
    n:
        Number of samples; at least one.
    seed:
        Non-negative base seed for the dataset. Train and test splits derive
        independent streams from it, so calling both splits with the same base
        seed does not duplicate samples.
    shape:
        Sensor-grid ``(height, width)``. Masks are rendered at four samples per
        pixel, the conventional oversampling factor for this target.
    split:
        ``"train"`` or ``"test"``. The split is encoded in the per-sample seed
        parity: training seeds are even and test seeds are odd, which makes
        the two seed sets disjoint by construction.

    Returns
    -------
    GlyphDataset
        Float16 masks plus class labels, characters, seeds, rotations, and
        per-sample paper/ink spectra. No spectral cube is stored; call
        :meth:`GlyphDataset.materialize` for the sample a consumer needs.

    Notes
    -----
    Classes are drawn from a seeded permutation of all 36 classes and cycled
    when ``n`` exceeds 36, so small datasets are balanced and larger ones stay
    deterministic. A sample's mask, rotation, and seed are exactly the values
    :func:`render_glyph_mask` and :func:`render_glyph` produce for that seed
    and recorded rotation, up to float16 mask quantization.

    Raises
    ------
    ValueError
        If ``n``, ``seed``, ``shape``, or ``split`` is invalid.
    """
    count = _require_integer(n, "n", 1)
    seed_value = _require_integer(seed, "seed", 0)
    sensor_shape = _validate_shape(shape)
    split_name = _validate_split(split)
    split_code = _SPLIT_CODES[split_name]
    factor = _DEFAULT_OVERSAMPLING

    class_rng = _rng(seed_value, _CLASS_STREAM + split_code)
    order = class_rng.permutation(len(GLYPH_CLASSES))
    class_indices = np.asarray(np.resize(order, count), dtype=np.int64)
    chars = tuple(GLYPH_CLASSES[int(index)] for index in class_indices)

    seed_rng = _rng(seed_value, _SEED_STREAM + split_code)
    base_seeds = np.asarray(
        seed_rng.integers(0, 1 << 62, size=count, dtype=np.int64),
        dtype=np.int64,
    )
    # Doubling leaves the low bit free for the split; train is even and test
    # odd, so the sets are disjoint even if two streams happen to draw the same
    # base value.
    sample_seeds = base_seeds * 2 + split_code

    default_paper, default_ink, spectra_provenance = _default_paper_ink()
    masks = np.empty(
        (count, sensor_shape[0] * factor, sensor_shape[1] * factor),
        dtype=np.float16,
    )
    rotations = np.empty(count, dtype=np.float64)
    for position in range(count):
        mask, rotation = _render_mask_and_rotation(
            chars[position],
            sensor_shape,
            int(sample_seeds[position]),
            None,
            factor,
        )
        masks[position] = mask.astype(np.float16)
        rotations[position] = rotation

    provenance: dict[str, Any] = {
        "renderer": "Pillow FreeType text rasterisation with antialiasing",
        "font": _FONT_DESCRIPTION,
        "font_size_fraction_range": (_MIN_FONT_FRACTION, _MAX_FONT_FRACTION),
        "rotation_range_deg": (-MAX_ROTATION_DEG, MAX_ROTATION_DEG),
        "oversampling": factor,
        "mask_dtype": "float16",
        "shape": sensor_shape,
        "split": split_name,
        "base_seed": seed_value,
        "seed_scheme": (
            "independent SeedSequence streams per split; sample seeds are even for "
            "train and odd for test, so the sets are disjoint"
        ),
        "spectra": copy.deepcopy(spectra_provenance),
    }
    return GlyphDataset(
        masks=masks,
        classes=class_indices,
        chars=chars,
        seeds=sample_seeds,
        rotations=rotations,
        paper_spectra=np.tile(default_paper, (count, 1)),
        ink_spectra=np.tile(default_ink, (count, 1)),
        shape=sensor_shape,
        split=split_name,
        oversampling=factor,
        provenance=provenance,
    )
