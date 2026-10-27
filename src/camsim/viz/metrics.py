"""Metric plots: MTF curves, SNR versus illuminance, delta-E, and acutance.

A metrics figure is a projection of measured numbers, not a second measurement.
The plots here therefore preserve the units and reference structure of the
quantities they display:

* **MTF** is the magnitude of the normalized transfer function, dimensionless
  and plotted from 0 to the 0.5 cycles/pixel Nyquist limit. The measured curve
  and its analytic reference share one axis, and MTF50/MTF20 are marked at the
  interpolated crossings so a reader can see where the single-number summary
  came from. Plotting past Nyquist would alias and can make MTF appear to rise.
* **SNR** is ``20 log10(mean / std)`` in decibels on a linear-light flat patch.
  Illuminance spans decades, so the lux axis is logarithmic; the SNR axis is
  linear because decibels are already logarithmic. A sensor sweep under constant
  exposure shows the transition from read-noise-limited to shot-noise-limited
  behaviour and eventually saturation.
* **Delta-E** is a CIEDE2000 colour difference per chart patch, dimensionless
  and non-negative. The heatmap keeps the physical chart arrangement (four rows
  of six ColorChecker patches) and prints the value in every cell, because a
  bare colour ramp invites reading hue as the data.
* **Acutance** is a contrast-normalized RMS gradient, dimensionless and only
  comparable between measurements with the same ROI geometry. It falls
  monotonically as a Gaussian blur widens on a fixed edge.

Real-world conventions: MTF50 of 0.1-0.4 cycles/pixel and mid-grey SNR of
10-45 dB are typical; a camera's mean ColorChecker CIEDE2000 is usually 1-5,
with individual patches several times that. Figure DPI is 150, and the
colorblind-safe Okabe-Ito palette keeps categorical SNR curves separable.

Failure artefacts guarded here: a measured curve drawn without its analytic
reference (hiding a calibration error), MTF plotted past Nyquist, an MTF marker
placed at an interpolated crossing the curve never reaches, a log axis applied
to already-logarithmic decibels, a delta-E heatmap without printed values, a
rainbow colour map that invents structure, and a figure whose output directory
does not exist. Every function validates its arrays and returns the Matplotlib
objects without closing them, so an interactive caller can keep editing; the
walkthroughs save through :func:`camsim.viz.save_figure` and close explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from camsim.viz.spectra import save_figure
from camsim.viz.style import PALETTE

__all__ = [
    "MTFPlotSeries",
    "plot_acutance",
    "plot_delta_e_heatmap",
    "plot_mtf_curve",
    "plot_snr_vs_lux",
]

_NYQUIST_CP = 0.5
_EPSILON = 1e-12
_MAX_MTF_PANELS = 4


def _finite_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a finite, non-empty 1-D float64 array or raise ``ValueError``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array; got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _save_if_requested(figure: Figure, save: str | Path | None) -> Path | None:
    """Save a figure with :func:`camsim.viz.save_figure` when a path is given."""
    if save is None:
        return None
    return save_figure(figure, save)


@dataclass(frozen=True)
class MTFPlotSeries:
    """One measured MTF curve plus its analytic reference and crossings.

    Parameters
    ----------
    label:
        Short case label such as ``"Gaussian sigma=1 px"``.
    frequency_cp:
        Strictly increasing frequencies in ``[0, 0.5]`` cycles/pixel starting at
        zero. Values past Nyquist alias and are rejected.
    measured_mtf:
        Measured modulation transfer factors at the same frequencies. Finite
        and non-negative; the metric clips its own estimate to ``[0, 1]``.
    analytic_mtf:
        Reference transfer function of the same PSF at the same frequencies.
    mtf50_cp, mtf20_cp:
        Interpolated crossings in cycles/pixel, or NaN when the curve never
        reaches that level inside Nyquist. A NaN crossing draws no marker.

    The arrays are converted to float64 and copied, so the series is a value
    object that a figure and a report can share without later mutation.
    """

    label: str
    frequency_cp: ArrayLike
    measured_mtf: ArrayLike
    analytic_mtf: ArrayLike
    mtf50_cp: float
    mtf20_cp: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(f"label must be a non-empty string; got {self.label!r}")
        frequency = _finite_vector(self.frequency_cp, "frequency_cp")
        measured = _finite_vector(self.measured_mtf, "measured_mtf")
        analytic = _finite_vector(self.analytic_mtf, "analytic_mtf")
        if not (frequency.size == measured.size == analytic.size):
            raise ValueError(
                "frequency_cp, measured_mtf, and analytic_mtf must share one length; "
                f"got {frequency.size}, {measured.size}, and {analytic.size}"
            )
        if frequency[0] != 0.0 or np.any(np.diff(frequency) <= 0.0):
            raise ValueError("frequency_cp must start at 0 and be strictly increasing")
        if frequency[-1] > _NYQUIST_CP + _EPSILON:
            raise ValueError(
                f"frequency_cp must not exceed Nyquist (0.5 cycles/pixel); got {frequency[-1]!r}"
            )
        if np.any(measured < 0.0) or np.any(analytic < 0.0):
            raise ValueError("MTF values must be non-negative")
        object.__setattr__(self, "frequency_cp", frequency)
        object.__setattr__(self, "measured_mtf", measured)
        object.__setattr__(self, "analytic_mtf", analytic)
        object.__setattr__(self, "mtf50_cp", float(self.mtf50_cp))
        object.__setattr__(self, "mtf20_cp", float(self.mtf20_cp))


def _marker(axis: Axes, frequency: float, level: float, label: str, colour: str) -> None:
    """Draw a crossing marker and its numeric label when the crossing exists."""
    if not math.isfinite(frequency):
        return
    axis.plot(
        [frequency],
        [level],
        marker="o",
        markersize=5.0,
        color=colour,
        linestyle="none",
        zorder=5,
    )
    axis.annotate(
        f"{label} {frequency:.4f}",
        xy=(frequency, level),
        xytext=(4.0, 8.0),
        textcoords="offset points",
        fontsize=7.5,
        color=colour,
    )


def plot_mtf_curve(
    series: Sequence[MTFPlotSeries],
    save: str | Path | None = None,
) -> tuple[Figure, list[Axes]]:
    """Draw measured MTF curves with their analytic references and crossings.

    Parameters
    ----------
    series:
        One to four :class:`MTFPlotSeries` values, each carrying the measured
        curve, the analytic reference, and the interpolated MTF50/MTF20
        crossings.
    save:
        Optional PNG path. The parent directory is created by
        :func:`camsim.viz.save_figure`.

    Returns
    -------
    tuple[Figure, list[Axes]]
        The figure and its flat list of axes, one per case.

    Raises
    ------
    ValueError
        If no series is supplied or a series is malformed.

    Notes
    -----
    The analytic reference is dashed and the measurement solid on the same
    axes, so a systematic bias is visible instead of hidden behind a single
    summary number. The horizontal axis stops at Nyquist. A NaN crossing draws
    no marker because the curve never reached that level inside the measured
    band.
    """
    cases = list(series)
    if not cases:
        raise ValueError("plot_mtf_curve requires at least one MTFPlotSeries")
    if len(cases) > _MAX_MTF_PANELS:
        raise ValueError(
            f"plot_mtf_curve supports at most {_MAX_MTF_PANELS} panels; got {len(cases)}"
        )
    columns = 2 if len(cases) > 1 else 1
    rows = math.ceil(len(cases) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 4.0 * rows), squeeze=False)
    flat = [axis for row in axes for axis in row]
    for axis, case in zip(flat, cases, strict=False):
        axis.plot(
            case.frequency_cp,
            case.measured_mtf,
            color=PALETTE[0],
            linewidth=1.6,
            label="measured e-SFR",
        )
        # Draw the reference after the measurement so its dashes remain
        # visible when the two curves agree to a few parts in a thousand; a
        # hidden reference would make the agreement impossible to judge.
        axis.plot(
            case.frequency_cp,
            case.analytic_mtf,
            color=PALETTE[1],
            linestyle="--",
            linewidth=2.0,
            label="analytic psf_mtf",
        )
        _marker(axis, case.mtf50_cp, 0.5, "MTF50", PALETTE[2])
        _marker(axis, case.mtf20_cp, 0.2, "MTF20", PALETTE[4])
        axis.set_xlim(0.0, _NYQUIST_CP)
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel("Spatial frequency (cycles/pixel)")
        axis.set_ylabel("MTF")
        axis.set_title(case.label, fontsize=10.0)
        axis.legend(loc="upper right", fontsize=8.0)
    for axis in flat[len(cases) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Slanted-edge e-SFR versus the analytic PSF transfer function",
        fontsize=10.5,
    )
    figure.tight_layout()
    _save_if_requested(figure, save)
    return figure, flat[: len(cases)]


def plot_snr_vs_lux(
    lux: ArrayLike,
    snr_db: Mapping[str, ArrayLike],
    save: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Draw flat-field SNR against scene illuminance for several channels.

    Parameters
    ----------
    lux:
        Scene illuminance in lux, one value per measurement. Finite, positive,
        and strictly increasing.
    snr_db:
        Channel label -> SNR in decibels at each lux level. Every series must
        have the same length as ``lux`` and contain only finite values. Typical
        keys are ``"luma"``, ``"R"``, ``"G"``, and ``"B"``.
    save:
        Optional PNG path.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and its only axis.

    Raises
    ------
    ValueError
        If the lux axis is not strictly increasing and positive, the mapping is
        empty, or a series has the wrong length or a non-finite sample.

    Notes
    -----
    The illuminance axis is logarithmic because a sweep spans decades; the SNR
    axis is linear because decibels are already a logarithmic quantity.
    """
    levels = _finite_vector(lux, "lux")
    if np.any(levels <= 0.0) or np.any(np.diff(levels) <= 0.0):
        raise ValueError("lux must be strictly increasing and positive")
    if not snr_db:
        raise ValueError("snr_db must contain at least one channel series")
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for index, (label, values) in enumerate(snr_db.items()):
        series = _finite_vector(values, f"snr_db[{label!r}]")
        if series.size != levels.size:
            raise ValueError(
                f"snr_db[{label!r}] has {series.size} samples but lux has {levels.size}"
            )
        colour = PALETTE[index % len(PALETTE)]
        axis.semilogx(levels, series, marker="o", markersize=4.0, color=colour, label=label)
    axis.set_xlabel("Scene illuminance (lux)")
    axis.set_ylabel("SNR (dB)")
    axis.set_title("Flat-field SNR versus illuminance (linear-light ROI)")
    axis.legend(loc="upper left", fontsize=8.5)
    figure.tight_layout()
    _save_if_requested(figure, save)
    return figure, axis


def plot_delta_e_heatmap(
    patch_names: Sequence[str],
    delta_e: ArrayLike,
    *,
    rows: int = 4,
    columns: int = 6,
    title: str = "Per-patch CIEDE2000",
    save: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Draw a chart-ordered delta-E heatmap with the value printed per cell.

    Parameters
    ----------
    patch_names:
        Patch labels in chart order, read left to right and top to bottom.
        Exactly ``rows * columns`` names are required.
    delta_e:
        Non-negative, finite colour differences in the same order.
    rows, columns:
        Chart geometry. The ColorChecker Classic is four rows of six patches.
    title:
        Figure title.
    save:
        Optional PNG path.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and its image axis.

    Raises
    ------
    ValueError
        If the geometry is invalid, the name or value count differs from it, or
        a value is negative or non-finite.

    Notes
    -----
    Values are printed because a bare colour ramp makes it easy to read hue as
    the measurement. The perceptually uniform ``viridis`` map avoids the false
    banding of a rainbow map, and the colour scale starts at zero because a
    colour difference cannot be negative.
    """
    names = [str(name) for name in patch_names]
    if rows < 1 or columns < 1:
        raise ValueError(f"rows and columns must be positive; got {rows} x {columns}")
    if len(names) != rows * columns:
        raise ValueError(
            f"patch_names must contain exactly rows * columns = {rows * columns} entries; "
            f"got {len(names)}"
        )
    values = _finite_vector(delta_e, "delta_e")
    if values.size != len(names):
        raise ValueError(f"delta_e has {values.size} values but patch_names has {len(names)}")
    if np.any(values < 0.0):
        raise ValueError("delta_e values must be non-negative")
    grid = values.reshape(rows, columns)

    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    # A grid over an image would draw lines that look like cell boundaries but
    # carry no data; the printed values and the colourbar carry the meaning.
    axis.grid(False)
    image = axis.imshow(grid, cmap="viridis", vmin=0.0, vmax=float(grid.max()), aspect="auto")
    for row in range(rows):
        for column in range(columns):
            value = float(grid[row, column])
            # The annotation colour flips at the midpoint so every value stays
            # legible against both ends of the colour ramp.
            threshold = 0.55 * float(grid.max()) if grid.max() > 0.0 else 0.0
            text_colour = "white" if value > threshold else "black"
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=text_colour,
            )
    axis.set_xticks(range(columns))
    axis.set_xticklabels([f"col {index + 1}" for index in range(columns)], fontsize=8.0)
    axis.set_yticks(range(rows))
    axis.set_yticklabels([f"row {index + 1}" for index in range(rows)], fontsize=8.0)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label="CIEDE2000")
    figure.tight_layout()
    _save_if_requested(figure, save)
    return figure, axis


def plot_acutance(
    sigma_px: ArrayLike,
    acutance: ArrayLike,
    save: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Draw edge acutance against Gaussian blur width.

    Parameters
    ----------
    sigma_px:
        Gaussian blur standard deviation in pixels, non-negative and strictly
        increasing. Zero denotes the unblurred edge.
    acutance:
        Contrast-normalized RMS gradient measured on the same edge ROI.
        Finite and non-negative.
    save:
        Optional PNG path.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and its only axis.

    Raises
    ------
    ValueError
        If the arrays differ in length, the blur axis is not increasing and
        non-negative, or a value is negative or non-finite.

    Notes
    -----
    Acutance is a correlation-study sharpness proxy, not a calibrated MTF. Its
    absolute value depends on the ROI geometry, so only measurements made with
    the same ROI belong on one curve.
    """
    sigma = _finite_vector(sigma_px, "sigma_px")
    values = _finite_vector(acutance, "acutance")
    if sigma.size != values.size:
        raise ValueError(
            f"sigma_px and acutance must share one length; got {sigma.size} and {values.size}"
        )
    if np.any(sigma < 0.0) or np.any(np.diff(sigma) <= 0.0):
        raise ValueError("sigma_px must be non-negative and strictly increasing")
    if np.any(values < 0.0):
        raise ValueError("acutance values must be non-negative")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(sigma, values, marker="o", markersize=4.5, color=PALETTE[0])
    axis.set_xlabel("Gaussian blur sigma (pixels)")
    axis.set_ylabel("Acutance (normalized gradient RMS)")
    axis.set_title("Edge acutance versus Gaussian blur")
    axis.set_ylim(bottom=0.0)
    figure.tight_layout()
    _save_if_requested(figure, save)
    return figure, axis
