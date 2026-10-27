"""Unified metric harness: run every group a chart layout and image support.

A camera measurement campaign produces one image and one measurement plan. The
plan - a :class:`~camsim.metrics.chart.ChartLayout` - names the regions to
measure: flat patches for noise and colour, an edge region for resolution and
sharpness. Not every image supports every group. A monochrome capture has no
colour; a flat-patch chart has no edge; a layout with no reference has no
colour ground truth. The harness's job is to run exactly the groups that are
well defined, skip the rest explicitly, and return one serializable
:class:`MetricReport` that records both the numbers and the decisions.

Applicability rules
-------------------
* **MTF and acutance** run only when ``layout.edge_roi`` exists. Both consume
  **linear light**, built once from ``image.linear_rgb`` when present and
  otherwise decoded from ``image.data`` with the exact sRGB/Display P3 EOTF
  (the two spaces share the transfer curve), then reduced to Rec. 709 luma for
  RGB or taken as the single plane for MONO. Gamma-encoded MTF input biases the
  curve, so the harness never measures the display buffer directly. The MTF
  additionally needs ``layout.pixel_pitch_um``; without it a cycles/pixel
  frequency cannot be expressed and the harness raises instead of inventing a
  scale. ``layout.focal_length_mm`` is optional: without it the cycles/pixel
  crossings are still reported and the cycles/degree fields are NaN.
* **SNR** runs for every patch when the layout has patches, always on the
  **linear** image. RGB patches report ``R``, ``G``, ``B``, and ``luma``; MONO
  patches report a single ``mono`` plane. Results are stored under the patch
  name in layout order. With no patches the group is ``None`` rather than an
  empty mapping, so a consumer can distinguish "not measured" from "measured
  nothing".
* **Colour** runs only for an RGB image, with at least one patch, against a
  resolvable reference (an explicit ``reference`` argument overrides
  ``layout.reference``). It consumes the **display-encoded** ``image.data`` and
  ``image.color_space``, exactly as :func:`camsim.metrics.color.delta_e_patches`
  requires. A layout whose name contains ``"colorchecker"`` (case-insensitive)
  and that has no reference raises: the metrics layer cannot infer an
  illuminant, and a silent D65 fallback would report a white-point error as a
  sensor colour error, so a non-D65 chart must carry its own reference. Any
  other layout without a reference simply skips colour.
* **Acutance** follows the edge ROI exactly like MTF and is measured on the same
  linear luma.

The report
----------
:class:`MetricReport` is a frozen dataclass of optional groups plus the layout
id and a metadata mapping. ``to_dict`` returns a JSON-friendly structure with
dataclasses expanded, arrays as lists, NumPy scalars as Python scalars, and
every non-finite float (NaN, +inf, -inf) as ``None``; the full MTF curve travels
in ``{"frequency_cp": [...], "mtf": [...]}``. ``to_json`` uses
``json.dumps(..., allow_nan=False)``, so the output is standards-compliant JSON
and a not-reached MTF crossing appears as ``null`` rather than the
non-standard ``NaN`` token. The metadata records the image's colour space and
channel layout, the layout's pixel pitch and focal length, the reference
illuminant, the fact that MTF was fed linear luma, and a ``skipped`` list
naming every group that did not run and why.

Real-world ranges: a rendered phone frame is 12-50 MP; a ColorChecker patch
spans tens of pixels; an ISO 12233-style edge region is 100-400 pixels tall;
pixel pitches run from about 0.8 um to 10 um, focal lengths from about 2 mm to
50 mm, and a good camera's chart-wide mean delta-E00 is about 1-5. MTF50 of
0.1-0.4 cycles/pixel and SNR of 10-45 dB on a mid-grey patch are typical.

Failure artefacts guarded here
------------------------------
* **Display-encoded MTF/SNR input.** The tone curve changes the edge profile and
  compresses high-light noise, so the harness linearizes once and feeds every
  linear metric the same buffer.
* **Missing references.** A colour chart with no reference raises instead of
  silently comparing against D65; non-chart layouts skip colour and say so.
* **Flat or misplaced edges.** An edge ROI that contains no usable gradient
  makes the MTF raise; acutance on a flat ROI is zero by definition.
* **Missing pixel pitch.** An edge layout without a pitch raises a clear
  ``ValueError`` rather than reporting cycles/pixel in a fictitious unit.
* **JSON NaN.** A sharp edge whose MTF never crosses 50 % inside the band has
  NaN crossings; serialization maps them to ``null`` so the report parses in
  every standards-compliant JSON reader.
* **Silent group loss.** Every skipped group is named in the metadata, so a
  report cannot look complete when a metric never ran.

The module imports NumPy, the standard library, and the other metrics modules
only - never the sensor, ISP, scene, or optics packages - so the harness cannot
depend on the model it is meant to judge. ``measure_all`` is deterministic and
never mutates the image or the layout.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from camsim.core.color import ColorReference
from camsim.core.frames import RenderedImage
from camsim.metrics.acutance import edge_acutance
from camsim.metrics.chart import ChartLayout, PatchROI, extract_patch
from camsim.metrics.color import ColorResult, _srgb_eotf, delta_e_patches
from camsim.metrics.mtf import MTFResult, slanted_edge_mtf
from camsim.metrics.snr import SNRResult, _luma_plane, snr_channels, snr_patch

__all__ = ["MetricReport", "measure_all"]

_RGB_LAYOUT: Final[str] = "RGB"
_MONO_CHANNEL: Final[str] = "mono"
_MTF_INPUT: Final[str] = "linear_luma"
_COLOUR_CHART_MARKER: Final[str] = "colorchecker"


def _require_non_empty_text(value: object, field: str) -> str:
    """Return a stripped, non-empty string or raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string; got {value!r}")
    return value.strip()


def _optional_layout_id(value: object) -> str | None:
    """Return ``None`` or a stripped, non-empty layout identifier."""
    if value is None:
        return None
    return _require_non_empty_text(value, "layout_id")


def _optional_acutance(value: object) -> float | None:
    """Return ``None`` or a finite, non-negative float, rejecting booleans."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"acutance must be a finite non-negative number; got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"acutance must be a finite non-negative number; got {value!r}")
    return number


def _snr_mapping(value: object) -> dict[str, dict[str, SNRResult]]:
    """Validate a patch → channel → result mapping into a plain ordered dict.

    Iterating the caller's mapping preserves its insertion order, which
    :func:`measure_all` fills in layout order. Patch and channel names must be
    non-empty strings; every leaf must be an :class:`SNRResult`.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"snr must be a mapping of patch name to channel results; got {value!r}")
    converted: dict[str, dict[str, SNRResult]] = {}
    for patch_key, channel_map in value.items():
        patch_name = _require_non_empty_text(patch_key, "SNR patch name")
        if not isinstance(channel_map, Mapping):
            raise ValueError(
                f"SNR patch {patch_name!r} must map channel names to SNRResult; "
                f"got {type(channel_map).__name__}"
            )
        channel_results: dict[str, SNRResult] = {}
        for channel_key, result in channel_map.items():
            channel_name = _require_non_empty_text(channel_key, "SNR channel name")
            if not isinstance(result, SNRResult):
                raise ValueError(
                    f"SNR channel {channel_name!r} must be an SNRResult; "
                    f"got {type(result).__name__}"
                )
            channel_results[channel_name] = result
        converted[patch_name] = channel_results
    return converted


def _metadata_mapping(value: object) -> dict[str, Any]:
    """Return a shallow copy of ``value`` as a string-keyed dict.

    ``None`` becomes the empty dict. The copy keeps a caller's dictionary from
    being mutated through the report; values are not deep-copied.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"metadata must be a mapping or None; got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _json_float(value: float) -> float | None:
    """Return a JSON-safe float: ``None`` for NaN and infinities."""
    number = float(value)
    return number if math.isfinite(number) else None


def _json_value(value: object) -> Any:
    """Convert a report value to a JSON-friendly Python object.

    Dataclasses expand field by field (in declaration order), arrays become
    nested lists, tuples become lists, NumPy scalars become Python scalars, and
    non-finite floats become ``None`` so that ``json.dumps(..., allow_nan=False)``
    stays standards-compliant. Unknown objects pass through unchanged; a caller
    who puts one in the report metadata makes ``to_json`` fail loudly rather
    than silently stringifying it.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _json_float(float(value))
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _json_float(value)
    if isinstance(value, np.ndarray):
        # tolist() handles 0-d arrays (it returns a scalar) and nested shapes;
        # recursing keeps NumPy scalars and non-finite samples on the same path.
        return _json_value(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        field_names = getattr(value, "__dataclass_fields__", {})
        return {name: _json_value(getattr(value, name)) for name in field_names}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class MetricReport:
    """One image's measured metric groups, layout identity, and provenance.

    Parameters
    ----------
    mtf:
        The slanted-edge :class:`~camsim.metrics.mtf.MTFResult`, or ``None``
        when the layout has no edge ROI. A sharp edge may legitimately carry
        NaN crossings ("not reached inside the measured band"), which serialize
        as ``null``.
    snr:
        Patch name → channel name → :class:`~camsim.metrics.snr.SNRResult`, in
        layout order, or ``None`` when the layout has no patches. RGB patches
        use ``"R"``, ``"G"``, ``"B"``, and ``"luma"``; MONO patches use
        ``"mono"``. A mapping is accepted and copied into plain dicts; names
        must be non-empty strings and leaves must be SNR results.
    color:
        The chart-wide :class:`~camsim.metrics.color.ColorResult`, or ``None``
        when colour was skipped (monochrome image, no patches, or no reference
        on a non-colour-chart layout).
    acutance:
        The normalized edge-gradient energy from
        :func:`camsim.metrics.acutance.edge_acutance`, or ``None`` when the
        layout has no edge ROI. Finite and non-negative; booleans are rejected.
    layout_id:
        The source layout's name, or ``None`` when built by hand. Non-empty
        when supplied; stored stripped.
    metadata:
        JSON-friendly provenance: colour space, channel layout, pixel pitch,
        focal length, reference illuminant, MTF input domain, and the
        ``skipped`` list. ``None`` becomes an empty dict; a mapping is copied.

    The dataclass is frozen, so a report can be shared between a sweep, a
    figure, and a serialized artifact without a later stage editing it. The
    nested SNR dicts are ordinary dicts for JSON compatibility; treat them as
    read-only.

    Notes
    -----
    :meth:`to_dict` expands dataclasses and converts arrays, tuples, NumPy
    scalars, and non-finite floats (to ``None``). :meth:`to_json` serializes
    that mapping with ``allow_nan=False``, so NaN crossings and undefined SNR
    ratios appear as ``null`` and the text parses with any standard JSON reader.
    """

    mtf: MTFResult | None = None
    snr: dict[str, dict[str, SNRResult]] | None = None
    color: ColorResult | None = None
    acutance: float | None = None
    layout_id: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mtf is not None and not isinstance(self.mtf, MTFResult):
            raise ValueError(f"mtf must be an MTFResult or None; got {type(self.mtf).__name__}")
        if self.color is not None and not isinstance(self.color, ColorResult):
            raise ValueError(
                f"color must be a ColorResult or None; got {type(self.color).__name__}"
            )
        if self.snr is not None:
            object.__setattr__(self, "snr", _snr_mapping(self.snr))
        object.__setattr__(self, "acutance", _optional_acutance(self.acutance))
        object.__setattr__(self, "layout_id", _optional_layout_id(self.layout_id))
        object.__setattr__(self, "metadata", _metadata_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-friendly mapping.

        The MTF group carries every scalar field plus the full sampled curve as
        ``{"frequency_cp": [...], "mtf": [...]}``. SNR is nested
        patch → channel → statistics, and colour carries its per-patch rows.
        Non-finite floats become ``None``; NumPy scalars become Python scalars.
        """
        payload: dict[str, Any] = {
            "layout_id": self.layout_id,
            "mtf": None
            if self.mtf is None
            else {
                "mtf50_cp": self.mtf.mtf50_cp,
                "mtf20_cp": self.mtf.mtf20_cp,
                "mtf50_cyc_per_deg": self.mtf.mtf50_cyc_per_deg,
                "mtf20_cyc_per_deg": self.mtf.mtf20_cyc_per_deg,
                "curve": {
                    "frequency_cp": self.mtf.curve.frequency_cp,
                    "mtf": self.mtf.curve.mtf,
                },
                "edge_angle_deg": self.mtf.edge_angle_deg,
                "nyquist_cp": self.mtf.nyquist_cp,
            },
            "snr": self.snr,
            "color": self.color,
            "acutance": self.acutance,
            "metadata": self.metadata,
        }
        return cast("dict[str, Any]", _json_value(payload))

    def to_json(self) -> str:
        """Return standards-compliant JSON; non-finite values serialize as null.

        ``allow_nan=False`` rejects the non-standard ``NaN`` and ``Infinity``
        tokens, so a not-reached crossing is emitted as ``null`` and the result
        parses in every strict JSON reader.
        """
        return json.dumps(self.to_dict(), allow_nan=False)


def _linear_image(image: RenderedImage) -> NDArray[np.float64]:
    """Return the image's linear-light buffer as float64.

    A camera-processed frame carries ``linear_rgb``; a hand-built or externally
    loaded frame may not, in which case the display data are decoded with the
    exact sRGB EOTF. Display P3 shares that transfer curve, so one decoder
    serves both output spaces. Decoding is never applied to an existing linear
    buffer - that would linearize twice.
    """
    if image.linear_rgb is not None:
        return np.asarray(image.linear_rgb, dtype=np.float64)
    return _srgb_eotf(image.data)


def _linear_luma(image: RenderedImage, linear: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a 2-D linear luma plane for an RGB frame, or the mono plane."""
    if image.channels == _RGB_LAYOUT:
        return _luma_plane(linear)
    return np.asarray(linear[..., 0], dtype=np.float64)


def measure_all(
    image: RenderedImage,
    layout: ChartLayout,
    reference: ColorReference | None = None,
) -> MetricReport:
    """Measure every metric group the image and layout support.

    Parameters
    ----------
    image:
        A :class:`~camsim.core.frames.RenderedImage`. Its ``data`` are
        display-referred and feed only the colour metric; every noise and
        resolution metric uses ``linear_rgb`` when present or the EOTF-decoded
        ``data`` otherwise. A mono frame has shape ``(H, W, 1)`` and is reduced
        to its single plane. Any other object is rejected: the harness needs
        the colour-space and channel metadata a bare array does not carry.
    layout:
        A :class:`~camsim.metrics.chart.ChartLayout`. Patch regions drive SNR
        and colour; ``edge_roi`` drives MTF and acutance. ``pixel_pitch_um`` is
        required when an edge ROI exists; ``focal_length_mm`` is optional.
        Regions are measured on the sensor grid of ``image``.
    reference:
        Optional per-patch CIE Lab ground truth. When ``None``,
        ``layout.reference`` is used. An explicit reference overrides the
        layout's, which supports comparing one capture against several
        candidate references. There is no implicit D65 fallback: a layout whose
        name contains ``"colorchecker"`` and that has no reference raises,
        because a reference that does not match the illuminant would turn a
        white-point error into an apparent camera colour error.

    Returns
    -------
    MetricReport
        The MTF result, per-patch SNR, chart-wide colour result, and acutance
        for the groups that apply; ``None`` for the rest; the layout name as
        ``layout_id``; and metadata recording the image and layout settings,
        the linear-luma input, the reference illuminant, and every skipped
        group with its reason.

    Raises
    ------
    TypeError
        If ``image`` is not a :class:`~camsim.core.frames.RenderedImage` or
        ``layout`` is not a :class:`~camsim.metrics.chart.ChartLayout`.
    ValueError
        If ``reference`` is neither a :class:`~camsim.core.color.ColorReference`
        nor ``None``; if the layout has an edge ROI but no
        ``pixel_pitch_um``; if a colour-checker layout has no resolvable
        reference; or if a metric rejects the supplied data (non-finite
        samples, an ROI outside the frame, a flat edge for MTF, a reference
        whose row count differs from the patch count, and so on).

    Notes
    -----
    The groups run in a fixed order and the metadata's ``skipped`` list uses a
    fixed order (MTF, acutance, SNR, colour), so repeated measurement is
    bit-identical. The image buffers and the layout are never modified:
    ``extract_patch`` returns views, and every metric copies what it needs.
    MTF is an ISO 12233-style estimate fed **linear luma**; hand it
    display-encoded data and the tone curve biases the curve low at high
    frequencies. Acutance is a correlation-study sharpness proxy, not a
    calibrated physical quantity.
    """
    if not isinstance(image, RenderedImage):
        raise TypeError(f"image must be a RenderedImage; got {type(image).__name__}")
    if not isinstance(layout, ChartLayout):
        raise TypeError(f"layout must be a ChartLayout; got {type(layout).__name__}")
    if reference is not None and not isinstance(reference, ColorReference):
        raise ValueError(
            f"reference must be a ColorReference or None; got {type(reference).__name__}"
        )
    resolved_reference = layout.reference if reference is None else reference

    linear = _linear_image(image)
    linear_luma = _linear_luma(image, linear)
    skipped: list[str] = []

    mtf_result: MTFResult | None = None
    acutance_value: float | None = None
    edge_roi: PatchROI | None = layout.edge_roi
    if edge_roi is None:
        skipped.append("mtf: layout has no edge_roi")
        skipped.append("acutance: layout has no edge_roi")
    else:
        pixel_pitch = layout.pixel_pitch_um
        if pixel_pitch is None:
            raise ValueError(
                "MTF requires layout.pixel_pitch_um (the sensor pixel pitch in "
                "micrometres); the layout supplies an edge_roi but no pitch, so "
                "cycles/pixel has no defined scale"
            )
        edge_plane = np.asarray(extract_patch(linear_luma, edge_roi), dtype=np.float64)
        mtf_result = slanted_edge_mtf(edge_plane, pixel_pitch, layout.focal_length_mm)
        acutance_value = edge_acutance(linear_luma, edge_roi)

    snr_results: dict[str, dict[str, SNRResult]] | None = None
    if layout.patches:
        snr_results = {}
        for patch in layout.patches:
            if image.channels == _RGB_LAYOUT:
                snr_results[patch.name] = snr_channels(linear, patch)
            else:
                snr_results[patch.name] = {_MONO_CHANNEL: snr_patch(linear_luma, patch)}
    else:
        skipped.append("snr: layout has no patches")

    color_result: ColorResult | None = None
    if image.channels != _RGB_LAYOUT:
        skipped.append("color: monochrome image")
    elif not layout.patches:
        skipped.append("color: layout has no patches")
    elif resolved_reference is None:
        if _COLOUR_CHART_MARKER in layout.name.lower():
            raise ValueError(
                "a colour chart requires a reference; attach a ColorReference to "
                "the layout or pass reference=. The metrics layer cannot infer an "
                "illuminant, and a D65 fallback would report a white-point error as "
                "a sensor colour error."
            )
        skipped.append("color: layout has no colour reference")
    else:
        # Colour is measured on the display-referred buffer, which is what the
        # delta-E pipeline decodes internally; SNR and resolution never see it.
        color_result = delta_e_patches(
            image.data,
            layout,
            reference=resolved_reference,
            color_space=image.color_space,
        )

    metadata: dict[str, Any] = {
        "color_space": image.color_space,
        "channels": image.channels,
        "pixel_pitch_um": layout.pixel_pitch_um,
        "focal_length_mm": layout.focal_length_mm,
        "reference_illuminant": (
            None if resolved_reference is None else resolved_reference.illuminant_name
        ),
        "mtf_input": _MTF_INPUT,
        "skipped": skipped,
    }
    return MetricReport(
        mtf=mtf_result,
        snr=snr_results,
        color=color_result,
        acutance=acutance_value,
        layout_id=layout.name,
        metadata=metadata,
    )
