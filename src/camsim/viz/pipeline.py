"""Stage-gallery plots for the ISP pipeline and demosaic comparisons.

An ISP trace is a sequence of buffers that live in different units, and a
contact sheet that ignores those units lies about what the pipeline did. The
panels therefore follow three explicit display rules:

* **Pre-BLC data are ADC codes.** The defective-pixel stage runs before the
  black pedestal is removed, so its buffer is in digital numbers (tens to
  thousands). The panel normalises that buffer by its own maximum only for
  visibility and labels it as code units.
* **Linear stages share one display transform.** Every stage from BLC through
  the last pre-tone stage is linear light, but its absolute level changes as
  the pipeline applies gains, white balance, and colour correction. All of
  those panels are divided by the *same* scale (the peak of the last pre-tone
  buffer) and then encoded with the sRGB OETF. A common scale is what makes
  brightness and colour changes between panels comparable; per-panel contrast
  stretching would make a dark stage look as bright as a bright one.
* **Tone and output panels are already display encoded.** They are shown as
  stored, clipped only to the display range. Applying the OETF a second time
  would double-encode them.

The sRGB opto-electronic transfer function is used as the display transform
because it is the pipeline's own standard output curve and because a linear
display would crush the shadows that the ISP's tone stage exists to lift. It is
a presentation choice: the numbers in the panels are not measurements, and a
figure never substitutes for the report's measured values. Nearest-neighbour
interpolation keeps individual pixels visible, which matters for demosaic
zippers, colour fringes, and sharpening halos; smooth interpolation would hide
the artefacts the gallery is meant to reveal.

The zoom view picks the highest-gradient location in the final display image
and crops the same region from every stage. A central crop would often land on
a flat patch and show nothing. The demosaic comparison places a reference RGB
image beside bilinear and Malvar-He-Cutler reconstructions and shows absolute
error maps on a shared scale, with max and mean errors printed in the summary
panel.

Real-world conventions: stage-gallery panels are 3-6 inches wide at 150 dpi,
nearest-neighbour is the honest choice for pixel-level defects, and a
difference map's colour scale must be shared across the methods being compared
or a small error in one method can look larger than a big error in another.

Failure artefacts guarded here: per-panel auto-normalisation that makes a
black-level or white-balance error invisible; applying the OETF twice to the
tone or output stage; mixing code units and linear light on one shared scale;
smooth interpolation that erases demosaic artefacts; difference maps with
independent scales; an unlabelled panel whose stage cannot be identified; and
figure paths whose parent directory does not exist.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from camsim.core.frames import StageOutput
from camsim.isp.tone import linear_to_srgb
from camsim.viz.spectra import save_figure

__all__ = [
    "plot_demosaic_comparison",
    "plot_pipeline_trace",
    "plot_zoom_crops",
]

_DISPLAY_STAGES: Final[frozenset[str]] = frozenset({"tone", "output"})
_DEFAULT_ZOOM_PX: Final[int] = 32
_MAX_PANEL_COLUMNS: Final[int] = 4
_GRID_COLOR: Final[str] = "#444444"
_LUMA_RED: Final[float] = 0.2126
_LUMA_GREEN: Final[float] = 0.7152
_LUMA_BLUE: Final[float] = 0.0722


def _image_array(data: Any, name: str) -> NDArray[np.float64]:
    """Return a stage buffer as a finite float64 2-D or RGB image."""
    try:
        array = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stage {name!r} does not hold a numeric image") from exc
    if array.ndim == 2:
        if array.size == 0:
            raise ValueError(f"stage {name!r} is empty")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"stage {name!r} contains non-finite values")
        return array
    if array.ndim == 3 and array.shape[-1] == 1:
        squeezed = np.asarray(array[..., 0], dtype=np.float64)
        if not np.all(np.isfinite(squeezed)):
            raise ValueError(f"stage {name!r} contains non-finite values")
        return squeezed
    if array.ndim == 3 and array.shape[-1] == 3:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"stage {name!r} contains non-finite values")
        return array
    raise ValueError(
        f"stage {name!r} must be a 2-D image or have one or three channels; got shape {array.shape}"
    )


def _stage_names(stages: Mapping[str, StageOutput]) -> list[str]:
    """Validate the trace and return its insertion order."""
    if not isinstance(stages, Mapping):
        raise ValueError(f"stages must be a mapping; got {type(stages).__name__}")
    names = list(stages)
    if not names:
        raise ValueError("stages must contain at least one stage")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"stage names must be non-empty strings; got {name!r}")
        if not isinstance(stages[name], StageOutput):
            raise ValueError(f"stage {name!r} must be a StageOutput")
    return names


def _is_pre_blc(name: str, names: list[str]) -> bool:
    """Return ``True`` when a stage runs before black-level correction."""
    return "blc" in names and names.index(name) < names.index("blc")


def _linear_scale(
    names: list[str],
    arrays: list[NDArray[np.float64]],
) -> float:
    """Return one shared peak for every non-display linear panel.

    The last stage before tone is the most processed linear buffer and carries
    the final signal scale; earlier stages are then divided by the same value.
    A zero or missing linear buffer falls back to one so the transform stays
    well defined.
    """
    pre_tone = names.index("tone") - 1 if "tone" in names else len(names) - 1
    for index in range(pre_tone, -1, -1):
        if not _is_pre_blc(names[index], names):
            peak = float(np.max(np.abs(arrays[index]))) if arrays[index].size else 0.0
            if math.isfinite(peak) and peak > 0.0:
                return peak
    return 1.0


def _display_panel(
    name: str,
    array: NDArray[np.float64],
    names: list[str],
    linear_scale: float,
) -> NDArray[np.float64]:
    """Map one stage buffer into the display range following the panel rules."""
    if name in _DISPLAY_STAGES:
        # Already encoded by the tone stage; never encode twice.
        return np.asarray(np.clip(array, 0.0, 1.0), dtype=np.float64)
    if _is_pre_blc(name, names):
        peak = float(np.max(np.abs(array))) if array.size else 0.0
        normalised = array / peak if peak > 0.0 else array
    else:
        normalised = array / linear_scale
    encoded = np.asarray(linear_to_srgb(np.clip(normalised, 0.0, 1.0)), dtype=np.float64)
    return encoded


def _unit_label(name: str, names: list[str]) -> str:
    """Return a short panel label naming the buffer's units."""
    if name in _DISPLAY_STAGES:
        return "encoded display"
    if _is_pre_blc(name, names):
        return "ADC code units (panel-normalised)"
    return "linear (shared display scale)"


def _style_image_axis(axis: Axes, title: str) -> None:
    """Remove ticks and frame a gallery panel consistently."""
    axis.set_title(title, fontsize=8.0, pad=3.0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(_GRID_COLOR)
        spine.set_linewidth(0.6)


def _show(
    axis: Axes,
    image: NDArray[np.float64],
    *,
    interpolation: str = "nearest",
) -> None:
    """Draw a 2-D or RGB panel with the display range fixed to ``[0, 1]``."""
    if image.ndim == 2:
        axis.imshow(
            image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation=interpolation,
            aspect="equal",
        )
    else:
        axis.imshow(
            np.clip(image, 0.0, 1.0),
            vmin=0.0,
            vmax=1.0,
            interpolation=interpolation,
            aspect="equal",
        )


def _save_if_requested(figure: Figure, save: str | Path | None) -> Path | None:
    """Save a figure with :func:`camsim.viz.save_figure` when a path is given."""
    if save is None:
        return None
    return save_figure(figure, save)


def plot_pipeline_trace(
    stages: Mapping[str, StageOutput],
    save: str | Path | None = None,
) -> tuple[Figure, list[Axes]]:
    """Draw a contact sheet of every ISP trace stage.

    Parameters
    ----------
    stages:
        Ordered ``StageOutput`` mapping from ``RenderedImage.stages`` (or an
        equivalent trace). Each ``data`` must be a 2-D mono image or a
        three-channel RGB image. The insertion order is the pipeline order.
    save:
        Optional PNG path. The parent directory is created by
        :func:`camsim.viz.save_figure`; ``None`` leaves the figure unsaved.

    Returns
    -------
    tuple[Figure, list[Axes]]
        The figure and its flat list of axes. Panels for a partially filled
        grid are hidden.

    Notes
    -----
    Pre-BLC panels are code-unit-normalised; the linear panels from BLC
    through the last pre-tone stage share one scale; tone and output are shown
    as encoded display. The suptitle states these rules so a reader cannot
    compare panels with the wrong unit assumption.
    """
    names = _stage_names(stages)
    arrays = [_image_array(stages[name].data, name) for name in names]
    linear_scale = _linear_scale(names, arrays)

    columns = min(_MAX_PANEL_COLUMNS, len(names))
    rows = math.ceil(len(names) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 2.9 * rows + 0.7),
        squeeze=False,
    )
    flat = [axis for row in axes for axis in row]
    for axis, name, array in zip(flat, names, arrays, strict=False):
        panel = _display_panel(name, array, names, linear_scale)
        _show(axis, panel)
        _style_image_axis(axis, f"{name}: {_unit_label(name, names)}")
    for axis in flat[len(names) :]:
        axis.set_visible(False)

    figure.suptitle(
        "ISP stage trace: DPC is in ADC code units; linear panels share one "
        "normalised display transform; tone/output are encoded display",
        fontsize=9.0,
    )
    figure.subplots_adjust(top=0.90, bottom=0.02, left=0.01, right=0.99, wspace=0.06, hspace=0.18)
    _save_if_requested(figure, save)
    return figure, flat[: len(names)]


def _choose_crop(image: NDArray[np.float64], size: int) -> tuple[int, int, int, int]:
    """Return a ``(row, column, height, width)`` crop around the strongest edge."""
    height, width = image.shape[:2]
    if image.ndim == 2:
        luma = image
    else:
        luma = _LUMA_RED * image[..., 0] + _LUMA_GREEN * image[..., 1] + _LUMA_BLUE * image[..., 2]
    gradient_rows, gradient_columns = np.gradient(luma)
    magnitude = np.hypot(gradient_rows, gradient_columns)
    if magnitude.size == 0 or not np.any(magnitude > 0.0):
        centre_row, centre_column = height // 2, width // 2
    else:
        centre_row, centre_column = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    crop_height = min(size, height)
    crop_width = min(size, width)
    top = min(max(int(centre_row) - crop_height // 2, 0), height - crop_height)
    left = min(max(int(centre_column) - crop_width // 2, 0), width - crop_width)
    return top, left, crop_height, crop_width


def plot_zoom_crops(
    stages: Mapping[str, StageOutput],
    save: str | Path | None = None,
) -> tuple[Figure, list[Axes]]:
    """Draw nearest-neighbour zoomed crops of every ISP trace stage.

    The crop is selected once, at the strongest gradient of the final display
    image, and the same sensor-pixel region is cut from every stage. This keeps
    the comparison spatially registered while showing the zipper, fringe, and
    halo artefacts that a full-frame contact sheet is too small to resolve.

    Parameters
    ----------
    stages:
        Ordered stage trace, as accepted by :func:`plot_pipeline_trace`.
    save:
        Optional PNG path.

    Returns
    -------
    tuple[Figure, list[Axes]]
        The figure and its flat list of axes.

    Notes
    -----
    The panel display rules are identical to :func:`plot_pipeline_trace`, so
    brightness, colour, and units remain comparable between the two figures.
    """
    names = _stage_names(stages)
    arrays = [_image_array(stages[name].data, name) for name in names]
    linear_scale = _linear_scale(names, arrays)
    final = _display_panel(names[-1], arrays[-1], names, linear_scale)
    top, left, crop_height, crop_width = _choose_crop(final, _DEFAULT_ZOOM_PX)

    columns = min(_MAX_PANEL_COLUMNS, len(names))
    rows = math.ceil(len(names) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 2.9 * rows + 0.7),
        squeeze=False,
    )
    flat = [axis for row in axes for axis in row]
    for axis, name, array in zip(flat, names, arrays, strict=False):
        panel = _display_panel(name, array, names, linear_scale)
        crop = panel[top : top + crop_height, left : left + crop_width]
        _show(axis, crop)
        _style_image_axis(axis, f"{name}: {_unit_label(name, names)}")
    for axis in flat[len(names) :]:
        axis.set_visible(False)

    figure.suptitle(
        "Stage-trace zoom at the strongest final-image edge "
        f"({crop_height}x{crop_width} px, nearest-neighbour; same display rules)",
        fontsize=9.0,
    )
    figure.subplots_adjust(top=0.90, bottom=0.02, left=0.01, right=0.99, wspace=0.06, hspace=0.18)
    _save_if_requested(figure, save)
    return figure, flat[: len(names)]


def _as_rgb(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a finite ``(H, W, 3)`` float64 image."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must be an (H, W, 3) RGB image; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def plot_demosaic_comparison(
    reference: ArrayLike,
    bilinear: ArrayLike,
    mhc: ArrayLike,
    save: str | Path | None = None,
) -> tuple[Figure, list[Axes]]:
    """Compare a reference RGB image with bilinear and MHC reconstructions.

    Parameters
    ----------
    reference:
        The full-resolution ``(H, W, 3)`` linear RGB image that was mosaiced.
    bilinear, mhc:
        Reconstructions of the same mosaic with the respective methods.
    save:
        Optional PNG path.

    Returns
    -------
    tuple[Figure, list[Axes]]
        A two-row figure: the reference and both reconstructions on the top
        row, and absolute-error maps plus a numeric summary on the bottom row.
        All three images use the same sRGB-encoded display transform; the two
        error maps share one linear-light colour scale.

    Raises
    ------
    ValueError
        If any input is not a non-empty finite ``(H, W, 3)`` array or the
        shapes differ.

    Notes
    -----
    The reference and reconstructions are linear light in ``[0, 1]``; the
    OETF is applied for display only. The error maps are computed in linear
    light and labelled as such. The summary reports max and mean absolute
    error over all samples, which is the same statistic the G3 tests gate.
    """
    reference_rgb = _as_rgb(reference, "reference")
    bilinear_rgb = _as_rgb(bilinear, "bilinear")
    mhc_rgb = _as_rgb(mhc, "mhc")
    if bilinear_rgb.shape != reference_rgb.shape or mhc_rgb.shape != reference_rgb.shape:
        raise ValueError(
            "reference, bilinear, and MHC images must share one shape; got "
            f"{reference_rgb.shape}, {bilinear_rgb.shape}, and {mhc_rgb.shape}"
        )

    bilinear_error = np.abs(bilinear_rgb - reference_rgb)
    mhc_error = np.abs(mhc_rgb - reference_rgb)
    error_scale = float(max(bilinear_error.max(), mhc_error.max()))
    if not math.isfinite(error_scale) or error_scale <= 0.0:
        error_scale = float(np.finfo(np.float64).eps)

    display_reference = np.asarray(
        linear_to_srgb(np.clip(reference_rgb, 0.0, 1.0)), dtype=np.float64
    )
    display_bilinear = np.asarray(linear_to_srgb(np.clip(bilinear_rgb, 0.0, 1.0)), dtype=np.float64)
    display_mhc = np.asarray(linear_to_srgb(np.clip(mhc_rgb, 0.0, 1.0)), dtype=np.float64)

    figure, axes = plt.subplots(2, 3, figsize=(10.0, 6.6))
    top = axes[0]
    top[0].imshow(display_reference, interpolation="nearest", vmin=0.0, vmax=1.0)
    top[0].set_title("Reference RGB", fontsize=9.0)
    top[1].imshow(display_bilinear, interpolation="nearest", vmin=0.0, vmax=1.0)
    top[1].set_title("Bilinear reconstruction", fontsize=9.0)
    top[2].imshow(display_mhc, interpolation="nearest", vmin=0.0, vmax=1.0)
    top[2].set_title("Malvar-He-Cutler reconstruction", fontsize=9.0)
    for axis in top:
        axis.set_xticks([])
        axis.set_yticks([])

    bottom = axes[1]
    error_image = bottom[0].imshow(
        bilinear_error.mean(axis=-1),
        cmap="viridis",
        vmin=0.0,
        vmax=error_scale,
        interpolation="nearest",
        aspect="equal",
    )
    bottom[0].set_title("Bilinear |error| (linear light)", fontsize=9.0)
    figure.colorbar(error_image, ax=bottom[0], fraction=0.046, pad=0.04)
    error_image_mhc = bottom[1].imshow(
        mhc_error.mean(axis=-1),
        cmap="viridis",
        vmin=0.0,
        vmax=error_scale,
        interpolation="nearest",
        aspect="equal",
    )
    bottom[1].set_title("MHC |error| (linear light)", fontsize=9.0)
    figure.colorbar(error_image_mhc, ax=bottom[1], fraction=0.046, pad=0.04)
    for axis in bottom[:2]:
        axis.set_xticks([])
        axis.set_yticks([])
    bottom[2].axis("off")
    bottom[2].text(
        0.02,
        0.95,
        "Maximum |error| (linear RGB)\n"
        f"bilinear: {bilinear_error.max():.3e}\n"
        f"MHC:      {mhc_error.max():.3e}\n\n"
        "Mean |error| (linear RGB)\n"
        f"bilinear: {bilinear_error.mean():.3e}\n"
        f"MHC:      {mhc_error.mean():.3e}\n\n"
        "Display panels are sRGB-encoded;\nthe error maps are linear light.",
        transform=bottom[2].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
    )

    figure.suptitle(
        "Siemens-star RGB mosaic: bilinear versus Malvar-He-Cutler",
        fontsize=10.5,
    )
    figure.subplots_adjust(top=0.90, bottom=0.03, left=0.03, right=0.97, wspace=0.18, hspace=0.18)
    _save_if_requested(figure, save)
    flat = [axis for row in axes for axis in row]
    return figure, flat
