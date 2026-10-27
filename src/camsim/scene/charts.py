"""Scene-owned chart layouts for the independent metrics harness.

This module is the one sanctioned bridge from the scene layer to the metrics
layer. Building a ColorChecker layout needs the chart's reflectance spectra, the
illuminant, and the CIE colour-matching functions, and all of those live in
``camsim.scene``. The returned type, however, is
:class:`camsim.metrics.chart.ChartLayout`, which knows nothing about spectra.
The dependency therefore runs scene -> metrics, never metrics -> scene, and
``import camsim.metrics`` stays free of the simulator packages.

The three layouts mirror what :func:`camsim.scene.targets.render` paints, in
sensor-pixel coordinates:

* ``colorchecker_layout`` reuses the renderer's own cell geometry
  (``_colorchecker_cells``) and ROI derivation (``_layout_from_painted_cells``)
  at the same ``shape`` and ``oversampling``, so every ROI pixel's full
  oversampling block lies inside its painted patch. A ROI that straddles a
  patch boundary averages two reflectances; re-deriving the rectangles here
  instead of sharing the renderer's code would let paint rounding and ROI
  rounding drift apart.
* ``flat_patch_layout`` names the full frame ``"flat"``, matching the uniform
  field used for SNR, shading, and fixed-pattern measurements.
* ``slanted_edge_layout`` supplies the centred half-size ``"edge"`` region used
  for ISO 12233-style e-SFR analysis. It is deliberately conservative: a
  region inset from the frame border avoids boundary pixels biasing the
  edge-angle fit, and the metric may crop further inside it.

``chart_layout_from_scene`` avoids the shape/oversampling mismatch entirely by
deriving the layout from a :class:`~camsim.scene.targets.SceneResult`; prefer
it when the render is already in hand.

Colour ground truth is bound, not assumed. ``colorchecker_layout`` attaches the
:class:`~camsim.core.color.ColorReference` computed spectrally by
:func:`camsim.scene.illuminants.reference_lab` for the requested illuminant, so
a 3000 K or 4000 K layout carries its own warm white point. An unknown
illuminant raises instead of silently falling back to D65. Flat-patch and
slanted-edge layouts carry no reference because they measure noise and
resolution, not colour.

Real-world parameter ranges: the default sensor shape ``(256, 256)`` and pixel
pitch ``1.12 um`` match :func:`camsim.scene.targets.render`'s defaults, so a
layout built with no arguments fits the default render. ``oversampling`` is
typically 2-4. Focal length is attached only when a
:class:`~camsim.config.schema.LensConfig` is supplied (2-50 mm across phone to
large-format lenses); without it a layout can still drive MTF in cycles/pixel
but not cycles/degree.

Failure artefacts guarded here: building a layout at a different shape or
oversampling from the render shifts every ROI; ignoring the oversampling
factor makes ROIs include surround pixels; omitting the illuminant reference
forces a D65 fallback that looks like a colour error; and passing a lens of the
wrong type would leave focal-length metadata either absent or wrong, biasing
cycles/degree values.
"""

from __future__ import annotations

from camsim.config.schema import LensConfig
from camsim.metrics.chart import ChartLayout
from camsim.metrics.chart import PatchROI as MetricsPatchROI
from camsim.scene.illuminants import reference_lab
from camsim.scene.spectral_data import load_colorchecker
from camsim.scene.targets import (
    PatchROI,
    SceneResult,
    _centered_roi,
    _colorchecker_cells,
    _layout_from_painted_cells,
    _require_integer,
    _validate_shape,
)

__all__ = [
    "chart_layout_from_scene",
    "colorchecker_layout",
    "flat_patch_layout",
    "slanted_edge_layout",
]

# Matches the renderer's default sensor pitch so a default layout fits a
# default render without a units mismatch.
_DEFAULT_PIXEL_PITCH_UM = 1.12


def _convert_roi(roi: PatchROI) -> MetricsPatchROI:
    """Convert a scene-layer ROI into the metrics-layer ROI value object.

    Both types carry the same five fields. The conversion is explicit rather
    than duck-typed so the metrics package never has to import ``camsim.scene``
    to accept a ROI.
    """
    return MetricsPatchROI(roi.name, roi.x, roi.y, roi.width, roi.height)


def _resolve_focal_length(lens: LensConfig | None) -> float | None:
    """Return the lens focal length in mm, or ``None`` when no lens is given."""
    if lens is None:
        return None
    if not isinstance(lens, LensConfig):
        raise ValueError(f"lens must be a LensConfig or None; got {type(lens).__name__}")
    return float(lens.focal_length_mm)


def chart_layout_from_scene(
    scene: SceneResult,
    lens: LensConfig | None = None,
) -> ChartLayout:
    """Return the metrics layout that exactly matches a rendered scene.

    Building a layout separately from the render is error-prone: the shape,
    oversampling factor, patch geometry, and illuminant reference must all
    match the scene, and a mismatch silently shifts every ROI. This helper
    derives the layout from the scene's own painted cells and reference, so a
    ColorChecker render and its measurement cannot disagree about
    ``oversampling`` or patch placement.

    Parameters
    ----------
    scene:
        The :class:`~camsim.scene.targets.SceneResult` returned by
        :func:`camsim.scene.targets.render`.
    lens:
        Optional lens used for the render. Supplying it records the focal
        length for cycles/degree conversion; ``None`` leaves the focal length
        unknown even though the renderer used a default lens, because the
        layout describes the camera actually used.

    Returns
    -------
    ChartLayout
        The scene's patch and edge regions in sensor-pixel coordinates, the
        scene's spectrally computed :class:`ColorReference`, and the sensor
        pixel pitch from the irradiance metadata.

    Raises
    ------
    ValueError
        If ``scene`` is not a :class:`~camsim.scene.targets.SceneResult`, or
        ``lens`` is neither ``None`` nor a
        :class:`~camsim.config.schema.LensConfig`.
    """
    if not isinstance(scene, SceneResult):
        raise ValueError(f"scene must be a SceneResult; got {type(scene).__name__}")
    patches = tuple(_convert_roi(patch) for patch in scene.layout.patches)
    edge_roi = None if scene.layout.edge_roi is None else _convert_roi(scene.layout.edge_roi)
    return ChartLayout(
        name=scene.layout.name,
        patches=patches,
        edge_roi=edge_roi,
        reference=scene.reference,
        pixel_pitch_um=scene.irradiance.pixel_pitch_um,
        focal_length_mm=_resolve_focal_length(lens),
    )


def colorchecker_layout(
    illuminant: str = "D65",
    lens: LensConfig | None = None,
    *,
    shape: tuple[int, int] = (256, 256),
    oversampling: int = 4,
    pixel_pitch_um: float = _DEFAULT_PIXEL_PITCH_UM,
) -> ChartLayout:
    """Return the 24-patch ColorChecker layout for a rendered scene.

    Parameters
    ----------
    illuminant:
        ``"D65"``, ``"3000K"``, or ``"4000K"``. The layout carries the
        spectrally computed Lab reference for exactly this illuminant. Any
        other name raises through
        :func:`camsim.scene.illuminants.reference_lab`; there is no silent
        D65 fallback, because a mismatched reference looks like a sensor
        colour error.
    lens:
        Optional :class:`~camsim.config.schema.LensConfig`. When supplied, its
        ``focal_length_mm`` is recorded for cycles/degree conversion. ``None``
        leaves the focal length unknown (``None``) even though the renderer
        has a reference lens default, because the layout describes the camera
        actually used, not a default.
    shape:
        Sensor-grid ``(height, width)`` of the render. Defaults to ``(256,
        256)``, matching :func:`camsim.scene.targets.render`.
    oversampling:
        Rendering samples per sensor pixel per axis. Must equal the value used
        for the render; the ROIs are derived from the painted cells on the
        ``shape * oversampling`` grid and would otherwise include surround
        pixels or miss the patch.
    pixel_pitch_um:
        Sensor pixel pitch in micrometres. Defaults to the renderer's
        ``1.12 um``.

    Returns
    -------
    ChartLayout
        Named ``"colorchecker"``, with 24 patches in physical chart order,
        each inset inside its painted cell and guaranteed to contain only
        patch colour. The reference is present for every supported
        illuminant, including D65.

    Raises
    ------
    ValueError
        If the illuminant is unsupported, the shape or oversampling is
        invalid, the shape is too small for clean patch ROIs, the lens is not
        a :class:`~camsim.config.schema.LensConfig`, or the pixel pitch is not
        finite and positive.
    """
    sensor_shape = _validate_shape(shape)
    sample_factor = _require_integer(oversampling, "oversampling", 1)
    focal_length_mm = _resolve_focal_length(lens)
    reference = reference_lab(illuminant)

    chart = load_colorchecker()
    scene_shape = (sensor_shape[0] * sample_factor, sensor_shape[1] * sample_factor)
    cells = _colorchecker_cells(scene_shape, chart.patches)
    scene_layout = _layout_from_painted_cells(cells, sample_factor, sensor_shape)
    patches = tuple(_convert_roi(patch) for patch in scene_layout.patches)
    return ChartLayout(
        name=scene_layout.name,
        patches=patches,
        reference=reference,
        pixel_pitch_um=pixel_pitch_um,
        focal_length_mm=focal_length_mm,
    )


def flat_patch_layout(
    shape: tuple[int, int] = (256, 256),
    *,
    pixel_pitch_um: float = _DEFAULT_PIXEL_PITCH_UM,
    focal_length_mm: float | None = None,
) -> ChartLayout:
    """Return the layout of the uniform flat patch used for SNR and shading.

    Parameters
    ----------
    shape:
        Sensor-grid ``(height, width)``. Defaults to ``(256, 256)``, matching
        :func:`camsim.scene.targets.render`.
    pixel_pitch_um:
        Sensor pixel pitch in micrometres; ``1.12 um`` matches the renderer.
    focal_length_mm:
        Optional actual focal length in millimetres for cycles/degree
        conversion.

    Returns
    -------
    ChartLayout
        Named ``"flat_patch"`` with a single full-frame ROI named ``"flat"``
        that covers every sensor pixel, exactly as the renderer's flat target
        does. ``reference`` is ``None``: a flat field measures noise, not
        colour.
    """
    height, width = _validate_shape(shape)
    return ChartLayout(
        name="flat_patch",
        patches=(MetricsPatchROI("flat", 0, 0, width, height),),
        pixel_pitch_um=pixel_pitch_um,
        focal_length_mm=focal_length_mm,
    )


def slanted_edge_layout(
    shape: tuple[int, int] = (256, 256),
    *,
    pixel_pitch_um: float = _DEFAULT_PIXEL_PITCH_UM,
    focal_length_mm: float | None = None,
) -> ChartLayout:
    """Return the layout of the centred slanted-edge analysis region.

    Parameters
    ----------
    shape:
        Sensor-grid ``(height, width)``. Defaults to ``(256, 256)``, matching
        :func:`camsim.scene.targets.render`.
    pixel_pitch_um:
        Sensor pixel pitch in micrometres; ``1.12 um`` matches the renderer.
    focal_length_mm:
        Optional actual focal length in millimetres for cycles/degree
        conversion.

    Returns
    -------
    ChartLayout
        Named ``"slanted_edge"`` with no patches and one centred, half-size
        ROI named ``"edge"``. Using the renderer's own centring helper keeps
        the region identical to ``render("slanted_edge", ...).layout``. A
        centred region avoids the frame border, where a slanted line exits the
        image and the edge-angle fit loses rows. ``reference`` is ``None``:
        e-SFR is luminance-based and illuminant-independent apart from scale.
    """
    sensor_shape = _validate_shape(shape)
    roi = _convert_roi(_centered_roi("edge", sensor_shape))
    return ChartLayout(
        name="slanted_edge",
        patches=(),
        edge_roi=roi,
        pixel_pitch_um=pixel_pitch_um,
        focal_length_mm=focal_length_mm,
    )
