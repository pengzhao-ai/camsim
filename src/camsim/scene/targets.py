"""Synthetic spectral scene targets and the scene-to-sensor renderer.

The scene layer turns a test chart into sensor-plane spectral irradiance. Each
target is a spatial map of spectral reflectance :math:`\\rho(\\lambda)`, a
dimensionless material property in ``[0, 1]``. A Lambertian surface under scene
illuminance :math:`E_{scene}(\\lambda)` has radiance
:math:`L = \\rho E_{scene} / \\pi`; a lens of transmittance
:math:`T(\\lambda)` and relative aperture :math:`N` images it with irradiance::

    E_img(lambda) = rho(lambda) * E_scene(lambda) * T(lambda)
                    / (4 N**2) * cos**4(theta)

where :math:`\\theta` is the field angle. The ``1 / (4 N**2)`` factor is what
makes absolute electron counts depend on aperture: opening from f/2.8 to f/1.4
quadruples the irradiance. The ``cos**4`` factor is supplied by
:func:`camsim.optics.vignetting.vignetting_map`, honouring the
``LensConfig.vignetting`` strength; with ``vignetting=None`` it is disabled.

Targets
-------
* **Slanted edge** - a binary reflectance step tilted a few degrees from
  vertical, the ISO 12233 structure used to estimate modulation transfer. The
  tilt spreads the edge across many pixel phases so the edge-spread function
  can be reconstructed at sub-pixel resolution.
* **ColorChecker** - the 24 BabelColor average reflectance spectra laid out as
  four rows of six square patches with a dark surround. Patch order is the
  physical chart order read left to right, top to bottom.
* **Flat patch** - a uniform reflectance field for SNR, shading, and
  flat-field tests.
* **Siemens star** - alternating radial spokes; a resolution and aliasing
  target whose spoke frequency rises towards the centre.

Sampling and rendering
----------------------
The target is rasterized on a grid ``oversampling`` times finer than the sensor
in each axis, then multiplied by the illuminant and lens terms, convolved with
the PSF, multiplied by vignetting, and box-averaged down to the sensor grid.
Four samples per pixel is the conventional choice: it bounds edge-position
quantization to a quarter pixel and pushes aliasing of the pixel aperture down
to a level that e-SFR can separate from optical blur. Two samples is adequate
for smooth targets; one sample is appropriate only when alias-free rendering is
not required.

The PSF argument is sampled on the **oversampled** grid, not the sensor grid.
A sensor-grid Gaussian of ``sigma`` pixels must be passed as
``gaussian_psf(sigma * oversampling)``; otherwise the blur is too narrow by the
oversampling factor. Layout coordinates, in contrast, always refer to the
sensor grid returned in ``SceneResult.irradiance``, because downstream metrics
extract patches and edges from that image.

Real-world parameter ranges: chart reflectance 0.02-0.95, slanted-edge tilts
2-7 degrees (ISO 12233 recommends about 5 degrees), illuminance 1-100,000 lux,
and lens f-numbers f/1.4-f/16 for these tests. A ColorChecker chart is
physically about 215 x 280 mm; its proportions, not its absolute size, are what
the simulation reproduces.

Failure artefacts guarded here: reflectance outside ``[0, 1]`` invents energy;
a chart whose patch ROIs overlap or leave the frame reports the wrong colour;
an edge angle near 45 degrees runs out of the frame and biases the estimate;
rendering at the sensor grid aliases the edge; applying a sensor-grid PSF to an
oversampled cube under-blurs; forgetting the ``1 / (4 N**2)`` or lens
transmittance term makes absolute exposure meaningless; and any global random
state would make two renders with the same seed differ.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import sobel  # type: ignore[import-untyped]

from camsim.config.schema import LensConfig
from camsim.core.color import ColorReference
from camsim.core.spectral import SpectralIrradiance
from camsim.optics.lens import transmittance_at
from camsim.optics.propagation import apply_psf, box_downsample
from camsim.optics.vignetting import vignetting_map
from camsim.scene.illuminants import illuminant as illuminant_spd
from camsim.scene.illuminants import lux_to_spectral_irradiance, reference_lab
from camsim.scene.spectral_data import load_colorchecker

__all__ = [
    "PatchROI",
    "SceneLayout",
    "SceneResult",
    "colorchecker_target",
    "estimate_edge_angle",
    "flat_patch_target",
    "render",
    "siemens_star_target",
    "slanted_edge_target",
]

TargetName: TypeAlias = Literal[
    "colorchecker",
    "slanted_edge",
    "flat_patch",
    "siemens_star",
]

_TARGET_NAMES: Final[tuple[TargetName, ...]] = (
    "colorchecker",
    "slanted_edge",
    "flat_patch",
    "siemens_star",
)

# ColorChecker Classic geometry: four rows of six square patches on a chart
# with a dark surround. The 6 % margin and 8 % inter-patch gap approximate the
# physical chart; ROIs are inset a further 15 % so transition pixels and
# demosaicing artefacts never enter a patch mean.
_CHART_COLUMNS: Final[int] = 6
_CHART_ROWS: Final[int] = 4
_CHART_MARGIN_FRACTION: Final[float] = 0.06
_CHART_GAP_FRACTION: Final[float] = 0.08
_CHART_ROI_INSET_FRACTION: Final[float] = 0.15
_CHART_BACKGROUND_REFLECTANCE: Final[float] = 0.05
_MIN_CHART_CELL_PX: Final[float] = 2.0

# A near-vertical slanted edge must stay inside the frame over the full image
# height for the row-centroid fit; beyond 45 degrees it exits through the top
# and bottom edges instead.
_MAX_EDGE_ANGLE_DEG: Final[float] = 45.0
_DEFAULT_EDGE_ANGLE_DEG: Final[float] = 5.0
_EDGE_BORDER_MARGIN_PX: Final[int] = 2
_EDGE_GRADIENT_THRESHOLD: Final[float] = 0.1
_MIN_EDGE_ROWS: Final[int] = 2

_DEFAULT_FLAT_REFLECTANCE: Final[float] = 0.5
_DEFAULT_STAR_SPOKES: Final[int] = 36

_REFERENCE_LENS_F_NUMBER: Final[float] = 1.8
_REFERENCE_LENS_FOCAL_LENGTH_MM: Final[float] = 6.4
_REFERENCE_LENS_DIAGONAL_MM: Final[float] = 11.46
_REFERENCE_LENS_TRANSMITTANCE: Final[float] = 0.9


def _require_non_empty_text(value: object, field_name: str) -> str:
    """Return a stripped, non-empty string or raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string; got {value!r}")
    return value.strip()


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


def _validate_reflectance(value: object, field_name: str) -> float:
    """Return a finite reflectance in ``[0, 1]``."""
    try:
        reflectance = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number in [0, 1]; {exc}") from exc
    if not math.isfinite(reflectance) or not 0.0 <= reflectance <= 1.0:
        raise ValueError(f"{field_name} must be a finite reflectance in [0, 1]; got {value!r}")
    return reflectance


@dataclass(frozen=True)
class PatchROI:
    """A rectangular region of a chart, in sensor-pixel coordinates.

    Parameters
    ----------
    name:
        Patch label such as ``"dark skin"`` or ``"neutral 8 (.23 D)"``. Unique
        within a :class:`SceneLayout`.
    x, y:
        Zero-based column and row of the region's top-left pixel.
    width, height:
        Region size in sensor pixels; both strictly positive.

    The region is half-open: it covers columns ``x .. x + width - 1`` and rows
    ``y .. y + height - 1``. This small layout type is intentionally local to
    the scene layer for the current phase; the metrics layer defines a richer
    ``ChartLayout`` with reference and camera metadata in a later phase.
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


@dataclass(frozen=True)
class SceneLayout:
    """Patch and edge locations for a rendered scene, in sensor pixels.

    Parameters
    ----------
    name:
        Target name, for example ``"colorchecker"`` or ``"slanted_edge"``.
    patches:
        Patch regions in chart order. Names must be unique. Empty for targets
        without patches.
    edge_roi:
        Optional region containing the slanted edge. It is the full analysis
        region for edge-based metrics; this phase supplies a conservative
        centred region, and the metrics layer may crop further.

    Coordinates always refer to the sensor grid of the returned
    :class:`SceneResult.irradiance`, not to the oversampled rendering grid.
    """

    name: str
    patches: tuple[PatchROI, ...] = ()
    edge_roi: PatchROI | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty_text(self.name, "layout name"))
        patches = tuple(self.patches)
        if not all(isinstance(patch, PatchROI) for patch in patches):
            raise ValueError("layout patches must all be PatchROI instances")
        names = [patch.name for patch in patches]
        if len(set(names)) != len(names):
            raise ValueError(f"layout patch names must be unique; got {names!r}")
        object.__setattr__(self, "patches", patches)
        if self.edge_roi is not None and not isinstance(self.edge_roi, PatchROI):
            raise ValueError("edge_roi must be a PatchROI or None")


@dataclass(frozen=True)
class SceneResult:
    """A rendered scene at the sensor plane plus its ground-truth metadata.

    Parameters
    ----------
    irradiance:
        Sensor-plane spectral irradiance in ``"W/m^2/nm"`` on the sensor grid,
        before the sensor's IR-cut filter and CFA. Its ``pixel_pitch_um``
        metadata names the sensor grid.
    layout:
        Patch and edge regions in sensor-pixel coordinates.
    reference:
        Per-patch CIE Lab reference for the render illuminant, computed
        spectrally from the ColorChecker reflectance and the CIE observer. It
        is the colour ground truth even for non-colour targets, where it
        provides the illuminant white point.
    ground_truth:
        Free-form scene metadata (target name, illuminant, lux, oversampling,
        grid shape, PSF and vignetting settings, and seed). Copied on
        construction.

    A scene result is a value object: mutate the arrays it exposes rather than
    the result fields, and pass it to the sensor layer exactly as returned.
    """

    irradiance: SpectralIrradiance
    layout: SceneLayout
    reference: ColorReference
    ground_truth: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.irradiance, SpectralIrradiance):
            raise ValueError("irradiance must be a SpectralIrradiance")
        if not isinstance(self.layout, SceneLayout):
            raise ValueError("layout must be a SceneLayout")
        if not isinstance(self.reference, ColorReference):
            raise ValueError("reference must be a ColorReference")
        if not isinstance(self.ground_truth, Mapping):
            raise ValueError("ground_truth must be a mapping")
        object.__setattr__(self, "ground_truth", dict(self.ground_truth))


def slanted_edge_target(
    shape: tuple[int, int],
    angle_deg: float = _DEFAULT_EDGE_ANGLE_DEG,
) -> NDArray[np.float64]:
    """Return a binary reflectance slanted edge.

    Parameters
    ----------
    shape:
        ``(height, width)`` of the rendered target grid.
    angle_deg:
        Signed tilt of the edge from vertical, in degrees. Positive values tilt
        the line rightwards as the row index (top to bottom) increases; zero is
        a perfectly vertical edge. The magnitude must be below 45 degrees so
        the line stays inside the frame across its full height.

    Returns
    -------
    A float64 ``(height, width)`` array of reflectance values exactly zero and
    one. A real edge target has finite contrast (for example 0.05 and 0.90);
    the ideal binary step maximizes edge energy for MTF estimation.

    The edge passes through the geometric image centre:
    ``(x - cx) cos(theta) - (y - cy) sin(theta) >= 0`` is the bright side.
    """
    height, width = _validate_shape(shape)
    try:
        angle = float(angle_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"angle_deg must be a finite number; {exc}") from exc
    if not math.isfinite(angle) or abs(angle) >= _MAX_EDGE_ANGLE_DEG:
        raise ValueError(
            f"angle_deg must be finite and strictly between -45 and 45 degrees; got {angle_deg!r}"
        )

    theta = math.radians(angle)
    rows, columns = np.mgrid[0:height, 0:width]
    signed_distance = (columns - (width - 1) / 2.0) * math.cos(theta) - (
        rows - (height - 1) / 2.0
    ) * math.sin(theta)
    edge: NDArray[np.float64] = np.asarray(signed_distance >= 0.0, dtype=np.float64)
    return edge


def estimate_edge_angle(image: ArrayLike) -> float:
    """Estimate a near-vertical slanted edge's tilt from image gradients.

    The horizontal Sobel gradient is computed for every row. Rows whose
    gradient peak is non-zero keep only the columns within 10 % of that peak,
    which rejects sensor noise and texture away from the edge. A gradient-
    energy-weighted centroid gives that row's edge column, and a weighted
    least-squares line fit to those centroids gives the edge slope
    ``dx / dy``. The tilt is ``arctan(slope)`` in degrees.

    Parameters
    ----------
    image:
        A finite 2-D array. It may be a reflectance target, a raw mono image,
        or a luma image; no normalization is required.

    Returns
    -------
    The signed edge angle in degrees, using the same convention as
    :func:`slanted_edge_target`: positive means the edge moves rightwards as
    the row index increases. Near-vertical edges only; the estimator does not
    wrap to a horizontal orientation.

    Raises
    ------
    ValueError
        If the image is not 2-D and finite, is too short for a line fit, or
        contains no edge gradient (for example a uniform field).

    Real lenses blur the edge and sensors add noise. The per-row threshold and
    energy weighting keep the estimate stable under both, as long as the edge
    spans enough rows; ISO 12233 analysis regions are typically hundreds of
    rows tall.
    """
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"image must be a 2-D array; got shape {array.shape}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("image must be non-empty and finite")
    height, width = array.shape
    if height <= 2 * _EDGE_BORDER_MARGIN_PX or width < 2:
        raise ValueError("image is too small for an edge line fit")

    gradient = np.asarray(sobel(array, axis=1), dtype=np.float64)
    weights = gradient * gradient
    columns = np.arange(width, dtype=np.float64)

    rows: list[float] = []
    centroids: list[float] = []
    energies: list[float] = []
    for row in range(_EDGE_BORDER_MARGIN_PX, height - _EDGE_BORDER_MARGIN_PX):
        row_weights = weights[row]
        peak = float(row_weights.max())
        if peak <= 0.0:
            continue
        row_weights = np.where(row_weights >= _EDGE_GRADIENT_THRESHOLD * peak, row_weights, 0.0)
        total = float(row_weights.sum())
        if total <= 0.0:
            continue
        rows.append(float(row))
        centroids.append(float(np.dot(row_weights, columns) / total))
        energies.append(total)

    if len(rows) < _MIN_EDGE_ROWS:
        raise ValueError("image has no usable horizontal gradient; cannot estimate an edge angle")

    coefficients = np.polyfit(
        np.asarray(rows, dtype=np.float64),
        np.asarray(centroids, dtype=np.float64),
        1,
        w=np.sqrt(np.asarray(energies, dtype=np.float64)),
    )
    slope = float(coefficients[0])
    angle_deg = math.degrees(math.atan(slope))
    if not math.isfinite(angle_deg):
        raise ValueError("edge fit did not produce a finite angle")
    return angle_deg


def flat_patch_target(
    shape: tuple[int, int],
    reflectance: float = _DEFAULT_FLAT_REFLECTANCE,
) -> NDArray[np.float64]:
    """Return a uniform reflectance field.

    Parameters
    ----------
    shape:
        ``(height, width)`` of the rendered target grid.
    reflectance:
        Constant reflectance in ``[0, 1]``. 0.5 is a neutral mid-grey that
        leaves highlight and shadow headroom in a 10-bit capture; 0.18 is the
        photographic 18 % grey card.

    Returns
    -------
    A float64 ``(height, width)`` constant array. A spectrally neutral patch
    models a grey card, whose reflectance is nearly constant across the visible
    band, rather than a coloured material.
    """
    height, width = _validate_shape(shape)
    value = _validate_reflectance(reflectance, "reflectance")
    return np.full((height, width), value, dtype=np.float64)


def siemens_star_target(
    shape: tuple[int, int],
    spokes: int = _DEFAULT_STAR_SPOKES,
) -> NDArray[np.float64]:
    """Return a binary Siemens-star reflectance target.

    The star is divided into ``spokes`` equal angular sectors of alternating
    zero and one reflectance. Its spatial frequency rises linearly towards the
    centre, so a single image reveals the spatial frequency at which contrast
    disappears (the resolution limit) and where aliasing or moire first
    appears. A 36-spoke star is the common printed chart; 4-16 spokes produce
    coarser wedges for low-resolution sensors.

    Parameters
    ----------
    shape:
        ``(height, width)`` of the rendered target grid.
    spokes:
        Number of angular sectors, an integer of at least two. Sectors
        alternate between dark and bright.

    Returns
    -------
    A float64 ``(height, width)`` array of zeros and ones.

    The pattern is centred on ``((height - 1) / 2, (width - 1) / 2)``. For even
    dimensions the exact centre lies between pixels, so no single pixel sits on
    the singularity; the sector through angle zero is bright.
    """
    height, width = _validate_shape(shape)
    if isinstance(spokes, bool) or not isinstance(spokes, (int, np.integer)):
        raise ValueError(f"spokes must be an integer of at least 2; got {spokes!r}")
    spoke_count = int(spokes)
    if spoke_count < 2:
        raise ValueError(f"spokes must be an integer of at least 2; got {spokes!r}")

    rows = np.arange(height, dtype=np.float64) - (height - 1) / 2.0
    columns = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    angle = np.arctan2(rows[:, None], columns[None, :])
    sector_width = 2.0 * math.pi / spoke_count
    sector = np.floor(np.mod(angle, 2.0 * math.pi) / sector_width).astype(np.int64)
    star: NDArray[np.float64] = np.asarray(1 - sector % 2, dtype=np.float64)
    return star


def _colorchecker_cells(shape: tuple[int, int]) -> tuple[tuple[str, int, int, int, int], ...]:
    """Return ``(name, x, y, width, height)`` paint rectangles for 24 patches.

    Patches are square and fitted inside a 6 % margin; the grid is centred in
    the frame, so a non-3:2 sensor shape gains extra margin along the long
    axis. The returned rectangles describe the full painted patch; ROIs are
    inset from them by :func:`_colorchecker_layout`.
    """
    height, width = _validate_shape(shape)
    usable_height = height * (1.0 - 2.0 * _CHART_MARGIN_FRACTION)
    usable_width = width * (1.0 - 2.0 * _CHART_MARGIN_FRACTION)
    cell = min(
        usable_width / _CHART_COLUMNS,
        usable_height / _CHART_ROWS,
    )
    if cell < _MIN_CHART_CELL_PX:
        raise ValueError(
            f"shape {(height, width)} is too small for a 6 x 4 ColorChecker; "
            f"each patch cell would be {cell:.2f} pixels wide"
        )

    grid_width = _CHART_COLUMNS * cell
    grid_height = _CHART_ROWS * cell
    x_origin = (width - grid_width) / 2.0
    y_origin = (height - grid_height) / 2.0
    gap = _CHART_GAP_FRACTION * cell
    patch_size = cell - gap
    names = load_colorchecker().patches

    cells: list[tuple[str, int, int, int, int]] = []
    for index, name in enumerate(names):
        row, column = divmod(index, _CHART_COLUMNS)
        left = x_origin + column * cell + gap / 2.0
        top = y_origin + row * cell + gap / 2.0
        x_start = max(0, min(round(left), width - 1))
        y_start = max(0, min(round(top), height - 1))
        x_stop = max(x_start + 1, min(round(left + patch_size), width))
        y_stop = max(y_start + 1, min(round(top + patch_size), height))
        cells.append((name, x_start, y_start, x_stop - x_start, y_stop - y_start))
    return tuple(cells)


def _colorchecker_layout(shape: tuple[int, int]) -> SceneLayout:
    """Return sensor-grid ROIs inset from each painted ColorChecker patch."""
    patches: list[PatchROI] = []
    for name, x, y, width, height in _colorchecker_cells(shape):
        roi_width = max(1, round(width * (1.0 - 2.0 * _CHART_ROI_INSET_FRACTION)))
        roi_height = max(1, round(height * (1.0 - 2.0 * _CHART_ROI_INSET_FRACTION)))
        roi_x = x + (width - roi_width) // 2
        roi_y = y + (height - roi_height) // 2
        patches.append(PatchROI(name, roi_x, roi_y, roi_width, roi_height))
    return SceneLayout("colorchecker", tuple(patches))


def colorchecker_target(
    shape: tuple[int, int],
) -> tuple[NDArray[np.float64], SceneLayout]:
    """Return the spectral ColorChecker reflectance cube and its layout.

    Parameters
    ----------
    shape:
        ``(height, width)`` of the rendered target grid. The 24 patches are
        fitted as square cells inside a 6 % margin; shapes smaller than about
        10 x 14 pixels cannot hold the grid and are rejected.

    Returns
    -------
    reflectance:
        A float64 ``(height, width, L)`` cube of reflectance factors in
        ``[0, 1]`` on the common 360-830 nm, 5 nm grid. Patch pixels carry the
        BabelColor average spectrum for that chart position; the surrounding
        margin is a neutral 0.05 reflectance.
    layout:
        A :class:`SceneLayout` whose 24 ROIs are inset 15 % from each patch
        edge and use the physical chart order and names.

    Reflectance is not colour: the same spectra look different under D65 and
    3000 K, and the colour ground truth for a particular illuminant comes from
    :class:`SceneResult.reference`, not from these raw spectra.
    """
    height, width = _validate_shape(shape)
    chart = load_colorchecker()
    cube = np.full(
        (height, width, len(chart.wavelengths_nm)),
        _CHART_BACKGROUND_REFLECTANCE,
        dtype=np.float64,
    )
    cells = _colorchecker_cells((height, width))
    for index, (_name, x, y, cell_width, cell_height) in enumerate(cells):
        cube[y : y + cell_height, x : x + cell_width, :] = chart.spectra[index]
    return cube, _colorchecker_layout((height, width))


def _centered_roi(name: str, shape: tuple[int, int]) -> PatchROI:
    """Return a centred half-size ROI, kept away from the frame border."""
    height, width = shape
    roi_width = max(1, width // 2)
    roi_height = max(1, height // 2)
    return PatchROI(
        name,
        (width - roi_width) // 2,
        (height - roi_height) // 2,
        roi_width,
        roi_height,
    )


def _reference_lens() -> LensConfig:
    """Return a fresh reference phone lens configuration."""
    return LensConfig(
        f_number=_REFERENCE_LENS_F_NUMBER,
        focal_length_mm=_REFERENCE_LENS_FOCAL_LENGTH_MM,
        sensor_diagonal_mm=_REFERENCE_LENS_DIAGONAL_MM,
        transmittance=_REFERENCE_LENS_TRANSMITTANCE,
    )


def _transmittance_curve(
    lens: LensConfig, wavelengths_nm: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the lens transmittance sampled at every scene wavelength."""
    if isinstance(lens.transmittance, str):
        return np.asarray(
            [transmittance_at(lens, float(wavelength)) for wavelength in wavelengths_nm],
            dtype=np.float64,
        )
    return np.full(wavelengths_nm.shape, float(lens.transmittance), dtype=np.float64)


def render(
    target: str,
    illuminant: str = "D65",
    lux: float = 100.0,
    lens: LensConfig | None = None,
    optics: ArrayLike | None = None,
    pixel_pitch_um: float = 1.12,
    shape: tuple[int, int] = (256, 256),
    oversampling: int = 4,
    seed: int | None = None,
) -> SceneResult:
    """Render a synthetic target to sensor-plane spectral irradiance.

    The target is rasterized at ``oversampling`` samples per sensor pixel in
    each axis and multiplied by the photometrically normalized illuminant, the
    lens transmittance, and ``1 / (4 N**2)``. A PSF is applied next, then the
    ``cos**4`` vignetting map, and finally a box average produces the sensor
    grid. Reflectance and all multiplication stay in linear radiometric units;
    no tone curve or display encoding belongs in the scene layer.

    Parameters
    ----------
    target:
        ``"colorchecker"``, ``"slanted_edge"``, ``"flat_patch"``, or
        ``"siemens_star"``. Target-specific options use their documented
        defaults (5 degree slanted edge, 36-spoke star, 0.5 flat reflectance).
    illuminant:
        ``"D65"``, ``"3000K"``, or ``"4000K"`` (see
        :func:`camsim.scene.illuminants.illuminant`).
    lux:
        Scene illuminance in lux. Zero is a valid dark scene; negative values
        are rejected.
    lens:
        A :class:`~camsim.config.schema.LensConfig`. ``None`` uses the
        reference phone lens: f/1.8, 6.4 mm focal length, 11.46 mm diagonal,
        transmittance 0.9, and no vignetting.
    optics:
        A 2-D PSF kernel sampled on the **oversampled** grid, or ``None`` for
        a diffraction- and aberration-free system. A sensor-grid Gaussian of
        ``sigma`` pixels must be passed as ``gaussian_psf(sigma * oversampling)``;
        passing the sensor-grid kernel directly under-blurs by the
        oversampling factor.
    pixel_pitch_um:
        Sensor pixel pitch in micrometres. The target grid is
        ``shape * oversampling``; its sample pitch is
        ``pixel_pitch_um / oversampling``.
    shape:
        Sensor-grid ``(height, width)``. The returned irradiance has exactly
        this spatial shape.
    oversampling:
        Positive integer samples per sensor pixel per axis. Four is the
        conventional rendering factor; one disables oversampling.
    seed:
        Non-negative seed recorded for stochastic target placement. The
        targets implemented in this phase are fully deterministic, so the
        value does not change the image; it is accepted and recorded now so
        later stochastic targets remain reproducible and no global RNG is
        needed.

    Returns
    -------
    SceneResult
        Sensor-plane :class:`~camsim.core.spectral.SpectralIrradiance` in
        ``"W/m^2/nm"`` with ``pixel_pitch_um`` metadata, a sensor-grid
        :class:`SceneLayout`, a spectrally computed :class:`ColorReference` for
        the illuminant, and a ground-truth metadata dict.

    Raises
    ------
    ValueError
        If the target, shape, oversampling, seed, lens, pitch, illuminant, or
        PSF is invalid, or if the ColorChecker grid cannot fit in ``shape``.

    Notes
    -----
    The ``cos**4`` factor is the natural-vignetting term of the lens formula.
    It is disabled when ``lens.vignetting`` is ``None`` or zero, which models a
    lens whose shading has been calibrated out; full strength reproduces the
    physical falloff. The PSF is wavelength-independent and applied to every
    spectral plane; a wavelength-dependent PSF would require one kernel per
    plane.
    """
    if target not in _TARGET_NAMES:
        raise ValueError(f"unsupported target {target!r}; expected one of {_TARGET_NAMES}")
    sensor_shape = _validate_shape(shape)
    sample_factor = _require_integer(oversampling, "oversampling", 1)
    seed_value: int | None = None if seed is None else _require_integer(seed, "seed", 0)
    if lens is None:
        resolved_lens = _reference_lens()
    elif isinstance(lens, LensConfig):
        resolved_lens = lens
    else:
        raise ValueError(f"lens must be a LensConfig; got {type(lens).__name__}")
    try:
        pitch_um = float(pixel_pitch_um)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pixel_pitch_um must be finite and positive; {exc}") from exc
    if not math.isfinite(pitch_um) or pitch_um <= 0.0:
        raise ValueError(f"pixel_pitch_um must be finite and positive; got {pixel_pitch_um!r}")

    scene_shape = (sensor_shape[0] * sample_factor, sensor_shape[1] * sample_factor)
    reflectance: NDArray[np.float64]
    layout: SceneLayout
    if target == "colorchecker":
        reflectance, _ = colorchecker_target(scene_shape)
        layout = _colorchecker_layout(sensor_shape)
    elif target == "slanted_edge":
        reflectance = slanted_edge_target(scene_shape)
        layout = SceneLayout(
            "slanted_edge",
            (),
            edge_roi=_centered_roi("edge", sensor_shape),
        )
    elif target == "flat_patch":
        reflectance = flat_patch_target(scene_shape)
        height, width = sensor_shape
        layout = SceneLayout("flat_patch", (PatchROI("flat", 0, 0, width, height),))
    else:
        reflectance = siemens_star_target(scene_shape)
        layout = SceneLayout("siemens_star")

    spd = illuminant_spd(illuminant)
    scene_irradiance = lux_to_spectral_irradiance(lux, spd)
    wavelengths_nm = spd.wavelengths_nm.nm
    if reflectance.ndim == 3:
        if reflectance.shape[2] != wavelengths_nm.size:
            raise RuntimeError(
                "target reflectance and illuminant do not share a wavelength grid; "
                f"got {reflectance.shape[2]} and {wavelengths_nm.size} samples"
            )
        spectral_reflectance = reflectance
    else:
        spectral_reflectance = reflectance[:, :, None]

    lens_factor = _transmittance_curve(resolved_lens, wavelengths_nm) / (
        4.0 * float(resolved_lens.f_number) ** 2
    )
    cube = spectral_reflectance * scene_irradiance.data * lens_factor
    cube = np.asarray(cube, dtype=np.float64)
    if optics is not None:
        cube = apply_psf(cube, optics)

    falloff = vignetting_map(
        scene_shape,
        resolved_lens,
        pitch_um / sample_factor,
    )
    cube *= falloff[:, :, None]
    cube = box_downsample(cube, sample_factor)

    ground_truth: dict[str, Any] = {
        "target": target,
        "illuminant": spd.name,
        "lux": float(lux),
        "shape": sensor_shape,
        "oversampling": sample_factor,
        "pixel_pitch_um": pitch_um,
        "psf_applied": optics is not None,
        "psf_grid": "oversampled" if optics is not None else None,
        "vignetting_strength": resolved_lens.vignetting,
        "seed": seed_value,
    }
    if target == "colorchecker":
        ground_truth["patch_names"] = tuple(patch.name for patch in layout.patches)
    return SceneResult(
        irradiance=SpectralIrradiance(
            cube,
            spd.wavelengths_nm,
            "W/m^2/nm",
            pixel_pitch_um=pitch_um,
        ),
        layout=layout,
        reference=reference_lab(spd.name),
        ground_truth=ground_truth,
    )
