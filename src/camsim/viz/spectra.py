"""Plotting and figure-saving helpers for spectra, PSFs, and MTF curves.

Spectral plots are line plots of a density per nanometre: an illuminant's
relative spectral power distribution :math:`S(\\lambda)`, a material's
reflectance factor :math:`\\rho(\\lambda)`, or a measured spectral response.
The x axis is wavelength in nanometres and is kept linear by default; a
logarithmic y axis is occasionally useful for showing faint Airy rings or a
small out-of-band response, but it hides the shape of the passband, so the
default is linear. Curves returned by the project's data types carry their own
wavelength grid, which makes a silent mismatch between data and axis
impossible; raw arrays must supply wavelengths explicitly.

A point-spread function is normalized to unit sum, so its absolute scale is
arbitrary. PSF profile plots therefore normalize each kernel to a peak of one
to compare shape, not radiometry. Two profile conventions are supported: a
central row cut, which is the profile whose discrete transform equals the
one-dimensional MTF at zero tangential frequency, and a radial average, which
shows the rings of an Airy pattern without one-pixel sampling noise. The radial
origin is the same sample (index ``size // 2``) that
:func:`camsim.optics.propagation.apply_psf` treats as the kernel centre.

The modulation transfer function is the magnitude of the normalized discrete
Fourier transform of the PSF,
:math:`\\mathrm{MTF}(f) = |\\sum_x h(x) e^{-i 2 \\pi f x}|`,
plotted against spatial frequency in cycles/pixel. The axis stops at the
Nyquist limit of 0.5 cycles/pixel: frequencies above it alias back into the
passband and can make a curve appear to rise again. MTF50 (the frequency where
contrast falls to one half) is the conventional single-number sharpness metric;
MTF20 is marked where a visible-contrast limit is more relevant. When the lens
geometry is known, an optional top axis converts cycles/pixel to
cycles/degree through the pixel angular pitch, the unit in which human acuity
and object detection are usually discussed.

Figure paths are explicit and deterministic. :func:`figure_dir` honours the
``CAMSIM_FIG_DIR`` environment variable and otherwise uses ``results/figures``;
:func:`save_figure` creates parent directories and writes a PNG with fixed
metadata, so repeated runs overwrite the same named file instead of
accumulating timestamped copies. Real-world figures use 100-300 dpi, 8-16
inches of width for a two-column layout, and six to eight categorical colours
at most.

Failure artefacts guarded here: a spectrum whose wavelengths are in metres
plots on a useless axis; a data/axis length mismatch silently truncates a curve
or raises deep inside Matplotlib; a PSF profile normalized by its sum looks
flat because the sum is one; plotting beyond Nyquist aliases high frequencies
into apparently real passband structure; a radial average over a non-square
kernel uses the wrong centre; a figure path whose parent does not exist loses
the figure with a bare ``FileNotFoundError``; and allowing Matplotlib's default
timestamped metadata makes otherwise identical figures differ byte for byte.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from camsim.optics.psf import psf_mtf, psf_mtf50
from camsim.scene.spectral_data import ColorCheckerData, SpectralPowerDistribution

__all__ = [
    "FIGURE_DIR_ENV",
    "figure_dir",
    "plot_mtf",
    "plot_psf",
    "plot_spectrum",
    "save_figure",
]

FIGURE_DIR_ENV: Final[str] = "CAMSIM_FIG_DIR"
"""Environment variable that overrides the default figure output directory."""

DEFAULT_FIGURE_DIR: Final[Path] = Path("results/figures")
"""Gallery directory used when ``CAMSIM_FIG_DIR`` is unset."""

SpectrumSource: TypeAlias = SpectralPowerDistribution | ColorCheckerData | ArrayLike
"""One project spectrum object, or an array of spectral samples."""

SpectrumInput: TypeAlias = SpectrumSource | Sequence[SpectrumSource]
"""A spectrum, or a sequence of project spectra such as several illuminants."""

KernelSource: TypeAlias = ArrayLike | Sequence[ArrayLike] | Mapping[str, ArrayLike]
"""A 2-D kernel, a sequence of kernels, or a label-to-kernel mapping."""

_MTF50_LEVEL: Final[float] = 0.5
_MTF20_LEVEL: Final[float] = 0.2
_DEFAULT_MTF_SAMPLES: Final[int] = 256
_DENSE_MTF_SAMPLES: Final[int] = 20001
_MAX_LEGEND_CURVES: Final[int] = 8
_LOG_FLOOR: Final[float] = 1e-6


def _validate_wavelengths(wavelengths_nm: ArrayLike, length: int) -> NDArray[np.float64]:
    """Return a finite, strictly increasing nanometre axis of the given length."""
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    if wavelengths.ndim != 1:
        raise ValueError(f"wavelengths_nm must be a 1-D array; got shape {wavelengths.shape}")
    if wavelengths.size != length:
        raise ValueError(
            f"wavelengths_nm has {wavelengths.size} samples but the spectra have {length}"
        )
    if wavelengths.size < 2:
        raise ValueError("wavelengths_nm must contain at least two samples")
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_nm must be finite and positive (nanometres, not metres)")
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("wavelengths_nm must be strictly increasing")
    return wavelengths


def _validate_curves(curves: Sequence[ArrayLike]) -> list[NDArray[np.float64]]:
    """Return validated finite one-dimensional float64 arrays."""
    validated: list[NDArray[np.float64]] = []
    for index, curve in enumerate(curves):
        values = np.asarray(curve, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"spectrum {index} must be one-dimensional; got shape {values.shape}")
        if values.size < 2:
            raise ValueError(f"spectrum {index} must contain at least two samples")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"spectrum {index} contains NaN or infinite values")
        validated.append(values)
    return validated


def _collect_spectra(
    spectra: SpectrumInput,
    wavelengths_nm: ArrayLike | None,
) -> tuple[list[NDArray[np.float64]], list[str], set[str], NDArray[np.float64]]:
    """Resolve any supported spectrum input into curves, labels, and a grid.

    The returned kind set is used only to choose a sensible default y-axis
    label: ``"spd"`` for illuminants, ``"reflectance"`` for chart spectra, and
    ``"array"`` for caller-supplied arrays.
    """
    if isinstance(spectra, SpectralPowerDistribution):
        curve = np.asarray(spectra.values, dtype=np.float64)
        return [curve], [spectra.name], {"spd"}, spectra.wavelengths_nm.nm

    if isinstance(spectra, ColorCheckerData):
        curves = [np.asarray(row, dtype=np.float64) for row in spectra.spectra]
        return curves, list(spectra.patches), {"reflectance"}, spectra.wavelengths_nm.nm

    if wavelengths_nm is not None:
        try:
            array = np.asarray(spectra, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "raw spectra must be a numeric 1-D or 2-D array when wavelengths_nm is given"
            ) from exc
        if array.ndim == 1:
            curves = [array]
        elif array.ndim == 2:
            curves = [array[index] for index in range(array.shape[0])]
        else:
            raise ValueError(f"raw spectra must be 1-D or 2-D; got shape {array.shape}")
        validated = _validate_curves(curves)
        raw_grid = _validate_wavelengths(wavelengths_nm, validated[0].size)
        return validated, [], {"array"}, raw_grid

    if isinstance(spectra, (str, bytes)):
        raise ValueError(
            "spectra must be a SpectralPowerDistribution, ColorCheckerData, an array with "
            "wavelengths_nm, or a sequence of those"
        )
    try:
        items = list(cast(Sequence[SpectrumSource], spectra))
    except TypeError as exc:
        raise ValueError(
            "spectra must be an SPD, ColorCheckerData, an array with wavelengths_nm, "
            "or a sequence of those"
        ) from exc
    if not items:
        raise ValueError("spectra must contain at least one curve")

    combined: list[NDArray[np.float64]] = []
    combined_labels: list[str] = []
    kinds: set[str] = set()
    shared_grid: NDArray[np.float64] | None = None
    for item in items:
        item_curves, item_labels, item_kinds, item_grid = _collect_spectra(item, None)
        if shared_grid is None:
            shared_grid = item_grid
        elif not np.array_equal(shared_grid, item_grid):
            raise ValueError("all spectra in a sequence must share the same wavelength grid")
        combined.extend(item_curves)
        combined_labels.extend(item_labels)
        kinds.update(item_kinds)
    if shared_grid is None:  # pragma: no cover - guarded by the non-empty check above
        raise ValueError("spectra must contain at least one curve")
    return _validate_curves(combined), combined_labels, kinds, shared_grid


def _resolve_labels(
    labels: str | Sequence[str] | None,
    default_labels: Sequence[str],
    count: int,
) -> list[str]:
    """Return one label per curve, overriding project-provided names if asked."""
    if labels is None:
        if len(default_labels) == count:
            return list(default_labels)
        return []
    resolved = [labels] if isinstance(labels, str) else [str(label) for label in labels]
    if len(resolved) != count:
        raise ValueError(f"expected {count} labels; got {len(resolved)}")
    return resolved


def _validate_kernel(kernel: ArrayLike, index: int) -> NDArray[np.float64]:
    """Return a finite, non-negative 2-D kernel with positive total energy."""
    try:
        array = np.asarray(kernel, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PSF {index} must be a numeric 2-D array") from exc
    if array.ndim != 2:
        raise ValueError(f"PSF {index} must be a 2-D kernel; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"PSF {index} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"PSF {index} contains NaN or infinite values")
    if np.any(array < 0.0):
        raise ValueError(f"PSF {index} must be non-negative")
    if float(array.sum()) <= 0.0:
        raise ValueError(f"PSF {index} must sum to a positive value")
    return array


def _collect_kernels(
    psfs: KernelSource,
    labels: str | Sequence[str] | None,
) -> tuple[list[NDArray[np.float64]], list[str]]:
    """Resolve a kernel, a sequence, or a mapping into validated kernels.

    Labels default to mapping keys when a mapping is supplied, to ``PSF n`` for
    a sequence, and to no label for a single kernel.
    """
    items: list[tuple[str | None, ArrayLike]]
    from_mapping = isinstance(psfs, Mapping)
    if from_mapping:
        items = [(str(key), value) for key, value in cast(Mapping[Any, ArrayLike], psfs).items()]
    else:
        array: NDArray[np.float64] | None
        try:
            array = np.asarray(psfs, dtype=np.float64)
        except (TypeError, ValueError):
            array = None
        if array is not None and array.ndim == 2:
            items = [(None, array)]
        elif array is not None and array.ndim == 3:
            items = [(None, array[index]) for index in range(array.shape[0])]
        else:
            try:
                sequence = list(cast(Sequence[ArrayLike], psfs))
            except TypeError as exc:
                raise ValueError("psfs must be a 2-D kernel, a sequence, or a mapping") from exc
            items = [(None, item) for item in sequence]
    if not items:
        raise ValueError("psfs must contain at least one kernel")

    kernels = [_validate_kernel(value, index) for index, (_, value) in enumerate(items)]
    default_labels = [key for key, _ in items if key is not None]
    if len(default_labels) != len(kernels):
        default_labels = (
            [f"PSF {index + 1}" for index in range(len(kernels))] if len(kernels) > 1 else []
        )
    resolved = _resolve_labels(labels, default_labels, len(kernels))
    return kernels, resolved


def figure_dir(default: str | Path = DEFAULT_FIGURE_DIR) -> Path:
    """Return the figure output directory, creating it when necessary.

    ``CAMSIM_FIG_DIR`` wins when it is set to a non-empty value; otherwise
    ``default`` is used. A relative path is resolved against the current
    working directory, which is why examples pass a repository-root-anchored
    default rather than relying on where they were launched. The directory is
    created eagerly so a later save cannot fail on a missing parent.
    """
    configured = os.environ.get(FIGURE_DIR_ENV)
    directory = Path(configured).expanduser() if configured else Path(default).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def save_figure(figure: Figure, destination: str | Path, *, dpi: int = 150) -> Path:
    """Write ``figure`` to ``destination`` and return the resolved path.

    Parent directories are created on demand. PNG output carries a fixed
    ``Software`` metadata field instead of Matplotlib's default version string,
    which keeps repeated runs reproducible across environments. The function
    does not close the figure; callers that generate a gallery in a loop should
    close each figure explicitly to release memory.

    Parameters
    ----------
    figure:
        A Matplotlib figure, typically returned by one of the ``plot_*``
        helpers in this module.
    destination:
        Output path. The suffix selects the format (``.png`` is the gallery
        convention); existing files are overwritten by design so figure names
        stay deterministic.
    dpi:
        Raster resolution in dots per inch. 100-300 is the useful range; 150 is
        the default compromise between screen review and print.

    Raises
    ------
    ValueError
        If ``dpi`` is not a positive integer.
    """
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError(f"dpi must be a positive integer; got {dpi!r}")
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, str] | None = (
        {"Software": "camsim"} if path.suffix.lower() == ".png" else None
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight", metadata=metadata)
    return path.resolve()


def plot_spectrum(
    spectra: SpectrumInput,
    wavelengths_nm: ArrayLike | None = None,
    *,
    labels: str | Sequence[str] | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
    show_legend: bool | None = None,
    linewidth: float = 1.5,
    alpha: float = 0.9,
) -> tuple[Figure, Axes]:
    """Plot illuminant SPDs or reflectance spectra against wavelength.

    Parameters
    ----------
    spectra:
        A :class:`~camsim.scene.spectral_data.SpectralPowerDistribution`, a
        :class:`~camsim.scene.spectral_data.ColorCheckerData`, a numeric array,
        or a sequence of the project types. Arrays require
        ``wavelengths_nm``; project objects carry their own grids.
    wavelengths_nm:
        Wavelength axis in nanometres for raw arrays. Must match the curve
        length and be strictly increasing. Ignored for project objects, whose
        validated grid is authoritative.
    labels:
        Legend labels overriding the object-provided names. A single string is
        used only when there is one curve; otherwise one label per curve is
        required.
    title, ylabel:
        Optional titles. When ``ylabel`` is omitted it is inferred as
        "Relative spectral power" for illuminants, "Reflectance factor" for
        chart spectra, and "Spectral value" for arrays.
    ax:
        Existing axes to draw on. ``None`` creates a new figure.
    show_legend:
        ``None`` shows a legend only when labels exist and at most eight
        curves are present, keeping a 24-patch chart readable.
    linewidth, alpha:
        Line styling in points and opacity.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes containing the curves.
    """
    curves, default_labels, kinds, wavelengths = _collect_spectra(spectra, wavelengths_nm)
    resolved_labels = _resolve_labels(labels, default_labels, len(curves))

    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = cast(Figure, axes.figure)

    for index, curve in enumerate(curves):
        label = resolved_labels[index] if index < len(resolved_labels) else None
        axes.plot(wavelengths, curve, linewidth=linewidth, alpha=alpha, label=label)

    if ylabel is None:
        if kinds == {"spd"}:
            ylabel = "Relative spectral power"
        elif kinds == {"reflectance"}:
            ylabel = "Reflectance factor"
        else:
            ylabel = "Spectral value"
    axes.set_xlabel("Wavelength (nm)")
    axes.set_ylabel(ylabel)
    axes.set_xlim(float(wavelengths[0]), float(wavelengths[-1]))
    if all(float(curve.min()) >= 0.0 for curve in curves):
        axes.set_ylim(bottom=0.0)
    if title is not None:
        axes.set_title(title)

    if show_legend is None:
        show_legend = bool(resolved_labels) and len(curves) <= _MAX_LEGEND_CURVES
    if show_legend and resolved_labels:
        axes.legend(loc="best")
    return figure, axes


def _psf_profile(
    kernel: NDArray[np.float64],
    mode: Literal["cut", "radial"],
    max_radius_px: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return coordinates and mean intensity for one PSF profile convention."""
    if mode == "cut":
        row = kernel[kernel.shape[0] // 2]
        coordinates = np.arange(kernel.shape[1], dtype=np.float64) - (kernel.shape[1] - 1) / 2.0
        values = np.asarray(row, dtype=np.float64)
        if max_radius_px is not None:
            keep = np.abs(coordinates) <= max_radius_px
            coordinates = coordinates[keep]
            values = values[keep]
        return coordinates, values

    height, width = kernel.shape
    rows, columns = np.indices(kernel.shape)
    radii = np.hypot(rows - height // 2, columns - width // 2)
    integer_radii = np.floor(radii + 0.5).astype(np.int64)
    counts = np.bincount(integer_radii.ravel())
    sums = np.bincount(integer_radii.ravel(), weights=kernel.ravel())
    populated = counts > 0
    coordinates = np.arange(counts.size, dtype=np.float64)[populated]
    values = np.asarray(sums[populated] / counts[populated], dtype=np.float64)
    if max_radius_px is not None:
        keep = coordinates <= max_radius_px
        coordinates = coordinates[keep]
        values = values[keep]
    return coordinates, values


def plot_psf(
    psfs: KernelSource,
    *,
    labels: str | Sequence[str] | None = None,
    mode: Literal["cut", "radial"] = "cut",
    max_radius_px: float | None = None,
    normalize: bool = True,
    log_scale: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
    show_legend: bool | None = None,
    linewidth: float = 1.5,
) -> tuple[Figure, Axes]:
    """Plot central-cut or radial profiles of one or more PSF kernels.

    Parameters
    ----------
    psfs:
        A 2-D kernel, a 3-D stack or sequence of kernels, or a mapping from
        label to kernel. A mapping's keys become the default legend labels.
    labels:
        Optional labels overriding mapping keys or generated ``PSF n`` names.
    mode:
        ``"cut"`` plots the central row through the kernel origin, the profile
        whose discrete transform is the conventional one-dimensional MTF.
        ``"radial"`` averages all samples at each rounded radius and is better
        for showing Airy rings.
    max_radius_px:
        Optional radius limit in pixels. For a cut the x axis is symmetric, so
        a value of 4 shows offsets from -4 to +4.
    normalize:
        Divide each profile by its peak. PSFs are already normalized to unit
        sum, which would flatten every profile to roughly ``1 / size**2`` and
        hide the shape being compared.
    log_scale:
        Use a logarithmic y axis, clipping at a small floor so exact zeros in
        an Airy ring do not produce missing-data warnings.
    title:
        Optional axes title.
    ax:
        Existing axes to draw on. ``None`` creates a new figure.
    show_legend:
        ``None`` shows a legend when labels exist and at most eight kernels are
        plotted.
    linewidth:
        Profile line width in points.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes containing the profiles.
    """
    if mode not in ("cut", "radial"):
        raise ValueError(f"mode must be 'cut' or 'radial'; got {mode!r}")
    if max_radius_px is not None and (not math.isfinite(max_radius_px) or max_radius_px <= 0.0):
        raise ValueError(f"max_radius_px must be finite and positive; got {max_radius_px!r}")
    kernels, resolved_labels = _collect_kernels(psfs, labels)

    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = cast(Figure, axes.figure)

    profile_extent = 0.0
    for index, kernel in enumerate(kernels):
        coordinates, values = _psf_profile(kernel, mode, max_radius_px)
        if normalize:
            peak = float(values.max()) if values.size else 0.0
            if peak <= 0.0:
                raise ValueError(f"PSF {index} has no positive values in the plotted profile")
            values = values / peak
        if log_scale:
            values = np.clip(values, _LOG_FLOOR, None)
        label = resolved_labels[index] if index < len(resolved_labels) else None
        axes.plot(coordinates, values, linewidth=linewidth, label=label)
        profile_extent = max(profile_extent, float(np.max(np.abs(coordinates))))

    axes.set_xlabel("Offset (pixels)" if mode == "cut" else "Radius (pixels)")
    axes.set_ylabel("Normalized intensity" if normalize else "Intensity")
    if log_scale:
        axes.set_yscale("log")
    else:
        axes.set_ylim(bottom=0.0)
    if mode == "cut":
        half_width = max_radius_px if max_radius_px is not None else profile_extent
        axes.set_xlim(-half_width, half_width)
    else:
        full_width = max_radius_px if max_radius_px is not None else profile_extent
        axes.set_xlim(0.0, full_width)
    if title is not None:
        axes.set_title(title)

    if show_legend is None:
        show_legend = bool(resolved_labels) and len(kernels) <= _MAX_LEGEND_CURVES
    if show_legend and resolved_labels:
        axes.legend(loc="best")
    return figure, axes


def _mtf_level_crossing(kernel: NDArray[np.float64], level: float) -> float | None:
    """Return the first frequency where the MTF falls through ``level``.

    The crossing is interpolated on a dense grid, matching the convention of
    :func:`camsim.optics.psf.psf_mtf50`. ``None`` means the MTF stays above the
    level through Nyquist, in which case there is no finite crossing to mark.
    """
    fractions = np.linspace(0.0, 0.5, _DENSE_MTF_SAMPLES)
    values = psf_mtf(kernel, fractions)
    below = np.flatnonzero(values < level)
    if below.size == 0:
        return None
    index = int(below[0])
    if index == 0:
        return 0.0
    low_frequency = float(fractions[index - 1])
    high_frequency = float(fractions[index])
    low_value = float(values[index - 1])
    high_value = float(values[index])
    fraction = (level - low_value) / (high_value - low_value)
    return low_frequency + (high_frequency - low_frequency) * float(fraction)


def plot_mtf(
    psfs: KernelSource,
    *,
    labels: str | Sequence[str] | None = None,
    frequencies_cp: ArrayLike | None = None,
    pixels_per_degree: float | None = None,
    mark_mtf50: bool = True,
    mark_mtf20: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
    show_legend: bool | None = None,
    linewidth: float = 1.7,
) -> tuple[Figure, Axes]:
    """Plot MTF curves for one or more PSF kernels.

    The curve is computed with :func:`camsim.optics.psf.psf_mtf` at
    frequencies from zero to Nyquist. MTF50 is located with
    :func:`camsim.optics.psf.psf_mtf50`, the same function the metrics layer
    uses, and MTF20 with an equivalent dense-grid interpolation. A crossing is
    marked only when the MTF actually reaches that level by 0.5 cycles/pixel.

    Parameters
    ----------
    psfs:
        A 2-D kernel, a sequence of kernels, or a mapping from label to kernel.
    labels:
        Optional labels overriding mapping keys or generated ``PSF n`` names.
    frequencies_cp:
        Optional 1-D frequency samples in cycles/pixel within ``[0, 0.5]``.
        The default is a uniform 256-sample grid.
    pixels_per_degree:
        Optional lens scale. When supplied, a secondary top axis converts
        cycles/pixel to cycles/degree through
        :meth:`camsim.config.schema.LensConfig.pixels_per_degree`.
    mark_mtf50, mark_mtf20:
        Draw horizontal reference lines and a marker at each curve's first
        crossing of 0.5 or 0.2.
    title:
        Optional axes title.
    ax:
        Existing axes to draw on. ``None`` creates a new figure.
    show_legend:
        ``None`` shows a legend when labels exist and at most eight kernels are
        plotted.
    linewidth:
        Curve line width in points.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes containing the MTF curves.

    Raises
    ------
    ValueError
        If frequencies leave ``[0, 0.5]``, if ``pixels_per_degree`` is not
        positive, or if a kernel is not a valid PSF.
    """
    kernels, resolved_labels = _collect_kernels(psfs, labels)
    if pixels_per_degree is not None:
        scale = float(pixels_per_degree)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"pixels_per_degree must be finite and positive; got {pixels_per_degree!r}"
            )

    if frequencies_cp is None:
        frequencies = np.linspace(0.0, 0.5, _DEFAULT_MTF_SAMPLES)
    else:
        frequencies = np.asarray(frequencies_cp, dtype=np.float64)
        if frequencies.ndim != 1 or frequencies.size < 2:
            raise ValueError("frequencies_cp must be a 1-D array with at least two samples")
        if not np.all(np.isfinite(frequencies)) or np.any(
            (frequencies < 0.0) | (frequencies > 0.5)
        ):
            raise ValueError("frequencies_cp must lie within [0, 0.5] cycles/pixel")

    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = cast(Figure, axes.figure)

    for index, kernel in enumerate(kernels):
        label = resolved_labels[index] if index < len(resolved_labels) else None
        mtf = psf_mtf(kernel, frequencies)
        (line,) = axes.plot(frequencies, mtf, linewidth=linewidth, label=label)
        color = line.get_color()

        if mark_mtf50:
            try:
                frequency_50 = psf_mtf50(kernel)
            except ValueError:
                frequency_50 = None
            if frequency_50 is not None:
                axes.plot(
                    [frequency_50],
                    [_MTF50_LEVEL],
                    marker="o",
                    markersize=5,
                    linestyle="none",
                    color=color,
                )

        if mark_mtf20:
            frequency_20 = _mtf_level_crossing(kernel, _MTF20_LEVEL)
            if frequency_20 is not None:
                axes.plot(
                    [frequency_20],
                    [_MTF20_LEVEL],
                    marker="s",
                    markersize=4,
                    linestyle="none",
                    color=color,
                )

    if mark_mtf50:
        axes.axhline(
            _MTF50_LEVEL,
            color="#666666",
            linewidth=0.8,
            linestyle="--",
            label="MTF50 (0.5)",
        )
    if mark_mtf20:
        axes.axhline(
            _MTF20_LEVEL,
            color="#999999",
            linewidth=0.8,
            linestyle=":",
            label="MTF20 (0.2)",
        )

    axes.set_xlabel("Spatial frequency (cycles/pixel)")
    axes.set_ylabel("Modulation transfer")
    axes.set_xlim(0.0, 0.5)
    axes.set_ylim(0.0, 1.05)
    if title is not None:
        axes.set_title(title)

    if pixels_per_degree is not None:
        scale = float(pixels_per_degree)

        def to_cycles_per_degree(frequency: ArrayLike) -> NDArray[np.float64]:
            """Convert a cycles/pixel frequency to cycles/degree."""
            return np.asarray(frequency, dtype=np.float64) * scale

        def to_cycles_per_pixel(frequency: ArrayLike) -> NDArray[np.float64]:
            """Convert a cycles/degree frequency back to cycles/pixel."""
            return np.asarray(frequency, dtype=np.float64) / scale

        secondary = axes.secondary_xaxis(
            "top",
            functions=(to_cycles_per_degree, to_cycles_per_pixel),
        )
        secondary.set_xlabel("Spatial frequency (cycles/degree)")

    if show_legend is None:
        show_legend = bool(resolved_labels) and len(kernels) <= _MAX_LEGEND_CURVES
    if show_legend and (resolved_labels or mark_mtf50 or mark_mtf20):
        axes.legend(loc="best")
    return figure, axes
