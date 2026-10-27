"""Chart geometry and colour ground truth for the independent metrics layer.

A chart layout is the measurement plan for an image-quality test: it names the
rectangular regions to measure and, for colour charts, carries the reference
CIE Lab values those regions should produce. Two physical facts shape the
design.

* **Coordinates are sensor pixels, not scene millimetres.** The scene renderer
  rasterizes a target on a grid ``oversampling`` times finer than the sensor
  and box-averages it down. Only the sensor grid is ever measured, so a region
  must name whole sensor pixels. Measuring on the oversampled grid would
  overstate resolution by the oversampling factor; measuring on the target's
  physical grid would make results depend on chart size.
* **A ROI is half-open.** A region ``(x, y, width, height)`` covers columns
  ``x .. x + width - 1`` and rows ``y .. y + height - 1``, exactly NumPy's
  slice ``image[y : y + height, x : x + width]``. Half-open rectangles make
  tiling exact: two ROIs that merely touch share no pixel, which is why
  ``[0, 4)`` followed by ``[4, 8)`` covers eight columns with no double count.

Colour ground truth travels with the layout as a
:class:`~camsim.core.color.ColorReference`. A chart's reflectance spectra are
not its colour: the same 24 patches under D65 (about 6500 K) and under a 3000 K
incandescent lamp differ by tens of delta-E if the reference is not matched to
the illuminant. ``ColorReference`` is owned by ``camsim.core`` so the scene
layer can compute it spectrally and the metrics layer can consume it without
importing the scene layer. This module therefore imports only NumPy, typing,
and ``camsim.core.color``; the import direction is scene -> metrics, never the
reverse.

Real-world parameter ranges: ColorChecker patches are roughly 40 mm square on a
215 x 280 mm chart, covering tens of pixels at a 1-10 um pixel pitch;
slanted-edge analysis regions are typically 100-400 pixels tall and are inset
from the frame border so the edge fit is not biased by boundary pixels.
Camera metadata uses micrometres for pixel pitch (0.8-10 um for phone through
large-format sensors) and millimetres for focal length (2-50 mm); focal length
supplies the pixels-per-degree conversion for MTF expressed in cycles/degree.

Failure artefacts guarded here: transposing ``x`` and ``y`` silently measures
the wrong patch; a region that crosses a patch boundary averages two
reflectances; overlapping regions double-count pixels in a statistics sweep; a
layout with neither patches nor an edge gives ``measure_all`` nothing to do;
attaching a D65 reference to a warm-light capture injects a white-point error
that mimics a sensor colour error; accepting NaN or infinite samples poisons
every downstream mean and standard deviation; and returning a copy from
:func:`extract_patch` would silently break callers that expect to measure the
original buffer in place without extra allocation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camsim.core.color import ColorReference

__all__ = ["ChartLayout", "PatchROI", "extract_patch"]


def _require_non_empty_text(value: object, field_name: str) -> str:
    """Return a stripped, non-empty string or raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string; got {value!r}")
    return value.strip()


def _require_integer(value: object, field_name: str, minimum: int) -> int:
    """Return an integer at least ``minimum``, rejecting booleans and floats.

    Float coordinates are rejected rather than truncated: ``2.5`` almost always
    means the caller mixed pixel-centre and pixel-corner conventions, and
    silently rounding it moves the ROI by half a pixel.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be an integer; got {value!r}")
    integer = int(value)
    if integer < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}; got {integer}")
    return integer


def _require_positive_float(value: object, field_name: str) -> float:
    """Return a finite, strictly positive float or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field_name} must be a finite positive number; got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a finite positive number; got {value!r}")
    return number


def _require_mapping(payload: object, field_name: str) -> Mapping[str, Any]:
    """Return ``payload`` as a mapping or raise ``ValueError``."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping; got {type(payload).__name__}")
    return payload


@dataclass(frozen=True)
class PatchROI:
    """A rectangular region of a chart, in sensor-pixel coordinates.

    Parameters
    ----------
    name:
        Patch label such as ``"dark skin"`` or ``"neutral 8 (.23 D)"``. Unique
        within a :class:`ChartLayout`; stored stripped of surrounding
        whitespace.
    x, y:
        Zero-based column and row of the region's top-left pixel. Both are
        non-negative integers; floats and booleans are rejected.
    width, height:
        Region size in sensor pixels. Both are positive integers.

    Notes
    -----
    The region is half-open: it covers columns ``x .. x + width - 1`` and rows
    ``y .. y + height - 1``. It carries no image size, so it may be built
    before the frame is known; :func:`extract_patch` performs the containment
    check when the image is available.

    A real ColorChecker patch spans tens of pixels at typical sensor pitches.
    ROIs are usually inset 10-20 % from the painted patch boundary so that
    optical blur and demosaic artefacts from the surround never enter the mean.
    """

    name: str
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty_text(self.name, "patch name"))
        object.__setattr__(self, "x", _require_integer(self.x, "patch x", 0))
        object.__setattr__(self, "y", _require_integer(self.y, "patch y", 0))
        object.__setattr__(self, "width", _require_integer(self.width, "patch width", 1))
        object.__setattr__(self, "height", _require_integer(self.height, "patch height", 1))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the region's five fields."""
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a region from :meth:`to_dict` output or parsed JSON."""
        mapping = _require_mapping(payload, "PatchROI payload")
        try:
            name = mapping["name"]
            x = mapping["x"]
            y = mapping["y"]
            width = mapping["width"]
            height = mapping["height"]
        except KeyError as exc:
            raise ValueError(f"PatchROI payload is missing required key {exc.args[0]!r}") from exc
        return cls(name=name, x=x, y=y, width=width, height=height)


def _rectangles_overlap(first: PatchROI, second: PatchROI) -> bool:
    """Return ``True`` when two half-open rectangles share a positive-area pixel.

    Touching edges (``first.x + first.width == second.x``) do not overlap,
    because the region is half-open.
    """
    return (
        first.x < second.x + second.width
        and second.x < first.x + first.width
        and first.y < second.y + second.height
        and second.y < first.y + first.height
    )


@dataclass(frozen=True)
class ChartLayout:
    """Patch regions, optional edge region, colour truth, and camera metadata.

    Parameters
    ----------
    name:
        Chart name such as ``"colorchecker"``, ``"flat_patch"``, or
        ``"slanted_edge"``. Non-empty; the chart type is read from this field,
        not inferred from the patch count.
    patches:
        Patch regions in chart order. Names must be unique and no two regions
        may overlap with positive area. Stored as a tuple. Patches may be
        empty only when ``edge_roi`` is supplied, because a layout with
        nothing to measure is a caller error, not an empty result.
    edge_roi:
        Optional region containing the slanted edge, with the same non-negative,
        positive-size and non-overlap rules. It is the full analysis region for
        edge-based metrics; the metric may crop further inside it.
    reference:
        Optional per-patch CIE Lab ground truth for the illuminant that lit
        the chart. Required in practice for colour metrics under non-D65
        illumination: a D65 reference applied to a 3000 K capture produces a
        systematic delta-E offset. The layout does not infer a reference from
        its name; the scene layer computes it spectrally and binds it
        explicitly.
    pixel_pitch_um:
        Sensor pixel pitch in micrometres, finite and positive, when known.
        Needed to express MTF in cycles/degree and to convert spatial
        frequencies to physical units.
    focal_length_mm:
        Actual lens focal length in millimetres, finite and positive, when
        known. Combined with ``pixel_pitch_um`` it gives the angular pixel
        pitch that converts cycles/pixel to cycles/degree.

    Notes
    -----
    Coordinates refer to the sensor grid of the measured image, never to the
    oversampled rendering grid. The layout is a frozen value object: it can be
    shared between metric calls and serialized for a report's provenance
    without risk of a later stage editing the regions.
    """

    name: str
    patches: tuple[PatchROI, ...] = ()
    edge_roi: PatchROI | None = None
    reference: ColorReference | None = None
    pixel_pitch_um: float | None = None
    focal_length_mm: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty_text(self.name, "layout name"))
        try:
            patches = tuple(self.patches)
        except TypeError as exc:
            raise ValueError(f"patches must be an iterable of PatchROI instances; {exc}") from exc
        if not all(isinstance(patch, PatchROI) for patch in patches):
            raise ValueError("layout patches must all be PatchROI instances")
        names = [patch.name for patch in patches]
        if len(set(names)) != len(names):
            raise ValueError(f"layout patch names must be unique; got {names!r}")
        object.__setattr__(self, "patches", patches)

        edge_roi = self.edge_roi
        if edge_roi is not None and not isinstance(edge_roi, PatchROI):
            raise ValueError("edge_roi must be a PatchROI or None")
        if not patches and edge_roi is None:
            raise ValueError("a chart layout requires at least one patch or an edge_roi")

        for first, second in combinations(patches, 2):
            if _rectangles_overlap(first, second):
                raise ValueError(
                    f"patches {first.name!r} and {second.name!r} overlap; "
                    "patch regions must not share pixels"
                )
        if edge_roi is not None:
            for patch in patches:
                if _rectangles_overlap(patch, edge_roi):
                    raise ValueError(
                        f"patch {patch.name!r} overlaps edge ROI {edge_roi.name!r}; "
                        "patch and edge regions must not share pixels"
                    )

        if self.reference is not None and not isinstance(self.reference, ColorReference):
            raise ValueError("reference must be a ColorReference or None")
        object.__setattr__(
            self,
            "pixel_pitch_um",
            None
            if self.pixel_pitch_um is None
            else _require_positive_float(self.pixel_pitch_um, "pixel_pitch_um"),
        )
        object.__setattr__(
            self,
            "focal_length_mm",
            None
            if self.focal_length_mm is None
            else _require_positive_float(self.focal_length_mm, "focal_length_mm"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping, with the reference in its own schema."""
        return {
            "name": self.name,
            "patches": [patch.to_dict() for patch in self.patches],
            "edge_roi": None if self.edge_roi is None else self.edge_roi.to_dict(),
            "reference": None if self.reference is None else self.reference.to_dict(),
            "pixel_pitch_um": self.pixel_pitch_um,
            "focal_length_mm": self.focal_length_mm,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a layout from :meth:`to_dict` output or parsed JSON.

        The reference round-trips through
        :meth:`camsim.core.color.ColorReference.from_dict`, so its Lab values,
        white point, observer, and provenance survive serialization. Optional
        camera metadata absent from the payload stays ``None``.
        """
        mapping = _require_mapping(payload, "ChartLayout payload")
        try:
            name = mapping["name"]
        except KeyError as exc:
            raise ValueError("ChartLayout payload is missing required key 'name'") from exc

        raw_patches = mapping.get("patches", ())
        if isinstance(raw_patches, (str, bytes)) or not isinstance(raw_patches, Iterable):
            raise ValueError(
                "ChartLayout payload 'patches' must be a sequence of ROI mappings; "
                f"got {type(raw_patches).__name__}"
            )
        patches = tuple(PatchROI.from_dict(item) for item in raw_patches)

        raw_edge = mapping.get("edge_roi")
        edge_roi = None if raw_edge is None else PatchROI.from_dict(raw_edge)

        raw_reference = mapping.get("reference")
        if raw_reference is not None and not isinstance(raw_reference, Mapping):
            raise ValueError(
                "ChartLayout payload 'reference' must be a mapping; "
                f"got {type(raw_reference).__name__}"
            )
        reference = None if raw_reference is None else ColorReference.from_dict(raw_reference)

        return cls(
            name=name,
            patches=patches,
            edge_roi=edge_roi,
            reference=reference,
            pixel_pitch_um=mapping.get("pixel_pitch_um"),
            focal_length_mm=mapping.get("focal_length_mm"),
        )


def extract_patch(image: ArrayLike, roi: PatchROI) -> NDArray[Any]:
    """Return the pixels of ``roi`` from a 2-D or 3-D image as a view.

    Parameters
    ----------
    image:
        Real-valued ``(H, W)`` mono or ``(H, W, C)`` multi-channel array-like.
        Boolean, complex, non-numeric, and non-finite inputs are rejected:
        a NaN in a patch turns every mean, standard deviation, and delta-E
        computed from it into NaN.
    roi:
        A :class:`PatchROI`. It must lie fully inside the image:
        ``roi.x + roi.width <= W`` and ``roi.y + roi.height <= H``.

    Returns
    -------
    ndarray
        The half-open crop ``image[y : y + height, x : x + width]`` with its
        trailing axes and dtype preserved. For an ndarray input this is a
        **view**, not a copy: it shares memory with ``image``. Callers must not
        mutate the result, because the mutation would corrupt the source frame
        that other metrics may still measure.

    Raises
    ------
    TypeError
        If ``roi`` is not a :class:`PatchROI`.
    ValueError
        If the image is not 2-D or 3-D, is not real-valued and finite, or the
        ROI extends past the image boundary. A clear error is preferred to
        NumPy's silent truncation, which would measure a smaller region than
        requested and bias a patch mean towards the image border.
    """
    if not isinstance(roi, PatchROI):
        raise TypeError(f"roi must be a PatchROI; got {type(roi).__name__}")
    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"image must be a real 2-D or 3-D array-like; {exc}") from exc
    if array.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D (H, W) or 3-D (H, W, C); got shape {array.shape}")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError("image must contain real numbers, not boolean values")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("image must be real; complex values are not supported")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"image must contain real numbers; got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values (no NaN or infinity)")

    height, width = array.shape[:2]
    if roi.x + roi.width > width or roi.y + roi.height > height:
        raise ValueError(
            f"roi {roi.name!r} at x={roi.x}, y={roi.y}, "
            f"{roi.width}x{roi.height} lies outside image shape {(height, width)}"
        )
    return array[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
