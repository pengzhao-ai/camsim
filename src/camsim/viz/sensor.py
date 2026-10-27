"""Sensor and EMVA 1288-style characterization figures.

A sensor figure is a projection of a measurement, not a measurement in itself,
so every plot here states what is measured and in which units:

* **Spectral response.** Quantum efficiency is the dimensionless probability
  that an incident photon produces a collected electron; CFA transmittance and
  IR-cut transmittance are dimensionless fractions. All three are plotted
  against wavelength in nanometres and bounded by ``[0, 1]``. The curves are
  representative synthetic engineering estimates shipped with the preset, not
  vendor measurements of a commercial part.
* **Photon transfer curve.** For a pixel with mean signal ``mu_DN`` and temporal
  variance ``sigma_DN**2`` in black-level-corrected DN, Poisson shot noise plus
  input-referred read noise gives ``sigma_DN**2 = K mu_DN + (K sigma_read)**2``
  to first order, where ``K`` is the system gain in DN per electron. The plot
  is log-log because both axes span several decades; the reference line uses the
  measured ``K`` and read noise, while the points at the lowest signal also
  contain the phase-dependent ADC quantization variance that the linear model
  omits.
* **SNR versus irradiance.** ``SNR_dB = 20 log10(signal_e / noise_e)``. The
  solid curve uses row-demeaned pixel-temporal noise; the dashed curve uses the
  total frame noise including common-mode row noise. They separate when row
  noise is comparable to shot noise, which is exactly when a single image looks
  noisier than the per-pixel model predicts.
* **Dark current versus temperature.** Silicon dark current follows
  ``D(T) = D_ref * 2**((T - T_ref) / T_double)``; the model curve is plotted on
  a logarithmic y axis with optional measured differential rates. Near the
  reference temperature the dark signal can be smaller than one ADC code, so a
  measured point is meaningful only after averaging many pixels and frames.
* **LCG versus HCG.** High conversion gain trades full-well capacity for lower
  input-referred read noise. A grouped bar chart makes the trade explicit in
  electrons and decibels; a single normalised score would hide which side of
  the trade changed.
* **DSNU and PRNU histograms.** DSNU is the per-pixel dark offset in electrons;
  PRNU is the dimensionless per-pixel responsivity deviation, plotted in percent
  because datasheets quote it that way. The dark histogram is compressed by the
  sensor's zero-charge clamp and sub-LSB ADC at low signal, which is why the
  reported pre-clamp DSNU is larger than the raw spatial spread of the averaged
  dark frame. The annotation states both.

Real-world ranges: QE 0.1-0.9, system gain 0.01-10 DN/e-, read noise 0.3-10 e-
rms, saturation 2-50 ke-, DSNU 0.1-10 e-, PRNU 0.1-2 %, dark current
0.01-100 e-/s at 25 C, and dynamic range 40-90 dB.

Failure artefacts guarded here: plotting a percent-valued QE curve as if it were
a fraction; drawing a PTC on linear axes where the low-signal points collapse
onto the origin; showing SNR without saying whether row noise was removed;
plotting dark current on a linear axis where the doubling law looks flat;
comparing LCG and HCG in one mixed PTC (which has no single slope); and
labelling the raw post-clamp DSNU spread as the configured pre-clamp value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm  # type: ignore[import-untyped]

from camsim.characterization.report import EMVAReport
from camsim.config.schema import SensorConfig
from camsim.data import data_path
from camsim.sensor.noise import dark_current_e_per_s
from camsim.viz.style import PALETTE

__all__ = [
    "plot_dark_current_vs_temperature",
    "plot_fixed_pattern_histograms",
    "plot_mode_comparison",
    "plot_ptc",
    "plot_sensor_spectral_response",
    "plot_snr_vs_irradiance",
]

_MAX_LEGEND_CURVES: Final[int] = 8
_HISTOGRAM_BINS: Final[int] = 60


def _read_curve_table(
    path: str | Path,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Read a comment-tolerant CSV with a header row.

    The shipped spectral files have ``#`` comment headers followed by a
    ``wavelength_nm,<curve>...`` header and numeric rows. The first column is
    the wavelength axis; every remaining named column is returned as a curve.
    """
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"cannot read spectral curve CSV {path}: {exc}") from exc
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 2:
        raise ValueError(f"spectral curve CSV {path} must contain a header and at least one row")
    header = [item.strip() for item in lines[0].split(",")]
    if len(header) < 2:
        raise ValueError(f"spectral curve CSV {path} must contain a wavelength column and a curve")
    rows: list[list[float]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        values = line.split(",")
        if len(values) != len(header):
            raise ValueError(
                f"spectral curve CSV {path} row {line_number} has {len(values)} columns"
            )
        try:
            rows.append([float(value) for value in values])
        except ValueError as exc:
            raise ValueError(f"spectral curve CSV {path} row {line_number} is not numeric") from exc
    data = np.asarray(rows, dtype=np.float64)
    wavelengths = data[:, 0]
    if not np.all(np.isfinite(data)) or np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError(f"spectral curve CSV {path} must be finite and increasing in wavelength")
    curves = {
        name: np.asarray(data[:, index], dtype=np.float64)
        for index, name in enumerate(header[1:], start=1)
    }
    return wavelengths, curves


def _first_curve(curves: Mapping[str, NDArray[np.float64]], path: Path) -> NDArray[np.float64]:
    """Return the sole value column of a two-column spectral CSV."""
    if not curves:
        raise ValueError(f"spectral curve CSV {path} has no value column")
    name = next(iter(curves))
    return curves[name]


def plot_sensor_spectral_response(
    config: SensorConfig,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot QE, CFA, and IR-cut curves referenced by a sensor preset.

    Parameters
    ----------
    config:
        Sensor preset whose ``spectral`` paths name the bundled CSVs. Relative
        paths resolve through :func:`camsim.data.data_path`; an absolute path is
        read directly.
    ax:
        Optional axes to draw on.
    title:
        Optional axes title.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes containing one curve per filter/channel.
    """
    qe_path = data_path(config.spectral.qe_csv)
    cfa_path = data_path(config.spectral.cfa_csv)
    ir_path = data_path(config.spectral.ir_cut_csv)
    qe_nm, qe_curves = _read_curve_table(qe_path)
    cfa_nm, cfa_curves = _read_curve_table(cfa_path)
    ir_nm, ir_curves = _read_curve_table(ir_path)

    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = cast(Figure, axes.figure)

    axes.plot(qe_nm, _first_curve(qe_curves, qe_path), color=PALETTE[0], label="QE")
    for index, (name, values) in enumerate(cfa_curves.items()):
        axes.plot(cfa_nm, values, color=PALETTE[1 + index], label=f"CFA {name}")
    axes.plot(
        ir_nm,
        _first_curve(ir_curves, ir_path),
        color=PALETTE[5],
        linestyle="--",
        label="IR-cut",
    )
    axes.set_xlabel("Wavelength (nm)")
    axes.set_ylabel("Fraction (dimensionless)")
    axes.set_xlim(
        float(min(qe_nm[0], cfa_nm[0], ir_nm[0])), float(max(qe_nm[-1], cfa_nm[-1], ir_nm[-1]))
    )
    axes.set_ylim(0.0, 1.05)
    if title is not None:
        axes.set_title(title)
    if len(qe_curves) + len(cfa_curves) + len(ir_curves) <= _MAX_LEGEND_CURVES:
        axes.legend(loc="best")
    return figure, axes


def _ptc_arrays(report: EMVAReport) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return positive mean and variance points in DN from a report."""
    means = np.asarray([point.mean_signal_dn for point in report.snr_curve], dtype=np.float64)
    variances = np.asarray([point.noise_dn**2 for point in report.snr_curve], dtype=np.float64)
    keep = (means > 0.0) & (variances > 0.0)
    return means[keep], variances[keep]


def plot_ptc(
    reports: Mapping[str, EMVAReport],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot measured photon-transfer points and each report's linear PTC model.

    The reference line is ``variance = K * mean + (K * read_noise)**2`` using
    the report's measured system gain and read noise. At the lowest signals the
    measured variance also contains quantization, so the line is a model, not a
    fit through every point.
    """
    if not reports:
        raise ValueError("at least one EMVA report is required")
    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = cast(Figure, axes.figure)

    for index, (label, report) in enumerate(reports.items()):
        color = PALETTE[index % len(PALETTE)]
        means, variances = _ptc_arrays(report)
        if means.size == 0:
            continue
        axes.scatter(means, variances, s=18, color=color, label=f"{label} measured")
        gain = report.system_gain_dn_per_e
        read_dn = gain * report.read_noise_e
        line_x = np.geomspace(max(float(means.min()), 1e-3), float(means.max()), 160)
        line_y = gain * line_x + read_dn**2
        axes.plot(
            line_x,
            line_y,
            color=color,
            linestyle="--",
            linewidth=1.1,
            label=(f"{label} model: K={gain:.3f} DN/e-, RN={report.read_noise_e:.2f} e-"),
        )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("Mean signal (DN)")
    axes.set_ylabel("Temporal variance (DN$^2$)")
    if title is not None:
        axes.set_title(title)
    if len(reports) <= 2:
        axes.legend(loc="best")
    return figure, axes


def plot_snr_vs_irradiance(
    reports: Mapping[str, EMVAReport],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot SNR in dB against sensor-plane irradiance for one or more modes."""
    if not reports:
        raise ValueError("at least one EMVA report is required")
    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = cast(Figure, axes.figure)

    for index, (label, report) in enumerate(reports.items()):
        color = PALETTE[index % len(PALETTE)]
        points = sorted(report.snr_curve, key=lambda point: point.irradiance_w_per_m2_nm)
        x = np.asarray([point.irradiance_w_per_m2_nm for point in points], dtype=np.float64)
        snr = np.asarray([point.snr_db for point in points], dtype=np.float64)
        axes.plot(
            x, snr, marker="o", markersize=4, color=color, label=f"{label} (row noise removed)"
        )
        totals = [point.snr_db_total for point in points]
        if all(value is not None for value in totals):
            axes.plot(
                x,
                np.asarray(totals, dtype=np.float64),
                linestyle=":",
                color=color,
                label=f"{label} (total frame noise)",
            )
    axes.set_xscale("log")
    axes.set_xlabel("Sensor-plane irradiance (W/m$^2$/nm)")
    axes.set_ylabel("SNR (dB)")
    if title is not None:
        axes.set_title(title)
    axes.legend(loc="best")
    return figure, axes


def plot_dark_current_vs_temperature(
    config: SensorConfig,
    *,
    measured: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    differential_reference_c: float | None = None,
    note: str | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot the dark-current doubling model with optional measured points.

    Parameters
    ----------
    config:
        Sensor preset supplying each mode's reference dark current and the
        reference/doubling temperatures.
    measured:
        Optional mapping ``mode -> (temperatures_c, rates_e_per_s)``. Measured
        points are plotted on the same logarithmic axis; a rate near the
        reference temperature may be quantization-limited and should be
        described in the caption or the ``note``.
    differential_reference_c:
        Optional reference temperature. When supplied, each model curve is
        plotted as ``D(T) - D(T_ref)`` so it can be compared directly with
        differential measured points. Comparing a differential measurement
        with the absolute model biases every point by ``D(T_ref)``, which is
        about 2 % at 70 C but dominates the low-temperature points.
    note:
        Optional annotation placed in the lower-right corner, for example a
        reminder that low-temperature differential points are compressed by the
        zero-charge clamp and sub-LSB ADC.
    """
    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = cast(Figure, axes.figure)

    temperatures = np.linspace(
        min(0.0, config.temperature.reference_c),
        max(70.0, config.temperature.reference_c),
        160,
    )
    for index, mode_name in enumerate(("LCG", "HCG")):
        color = PALETTE[index]
        reference = float(config.modes[mode_name].dark_current_e_per_s)  # type: ignore[index]
        rates = [
            dark_current_e_per_s(reference, float(temperature), config.temperature)
            for temperature in temperatures
        ]
        if differential_reference_c is not None:
            reference_rate = dark_current_e_per_s(
                reference,
                float(differential_reference_c),
                config.temperature,
            )
            rates = [rate - reference_rate for rate in rates]
            label = f"{mode_name} model: differential from {differential_reference_c:g} C"
        else:
            label = f"{mode_name} model: {reference:g} e-/s at {config.temperature.reference_c:g} C"
        axes.plot(
            temperatures,
            np.asarray(rates, dtype=np.float64),
            color=color,
            linestyle="-" if mode_name == "LCG" else "--",
            label=label,
        )
        if measured is not None and mode_name in measured:
            measured_t, measured_rates = measured[mode_name]
            axes.plot(
                np.asarray(measured_t, dtype=np.float64),
                np.asarray(measured_rates, dtype=np.float64),
                marker="o",
                markersize=4,
                linestyle="none",
                color=color,
                label=f"{mode_name} measured",
            )
    axes.set_yscale("log")
    axes.set_xlabel("Sensor temperature (C)")
    axes.set_ylabel(
        "Differential dark current (e-/s)"
        if differential_reference_c is not None
        else "Dark current (e-/s)"
    )
    if title is not None:
        axes.set_title(title)
    if note is not None:
        axes.annotate(
            note,
            xy=(0.98, 0.04),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8.5,
        )
    axes.legend(loc="upper left")
    return figure, axes


def plot_mode_comparison(
    reports: Mapping[str, EMVAReport],
    *,
    title: str | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    """Compare read noise, saturation capacity, and dynamic range by mode.

    Three small panels keep the electron and decibel axes separate; a single
    combined score would hide which side of the dual-conversion-gain trade
    changed.
    """
    if not reports:
        raise ValueError("at least one EMVA report is required")
    figure, axes_array = plt.subplots(1, 3, figsize=(10.8, 3.6), layout="constrained")
    axes = cast(tuple[Axes, Axes, Axes], tuple(axes_array))
    labels = list(reports)
    positions = np.arange(len(labels), dtype=np.float64)
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(labels))]

    read_noise = [reports[label].read_noise_e for label in labels]
    saturation = [reports[label].saturation_capacity_e for label in labels]
    dynamic_range = [reports[label].dynamic_range_db for label in labels]

    panels = (
        (axes[0], read_noise, "Read noise (e- rms)", "{:.2f}"),
        (axes[1], saturation, "Saturation capacity (e-)", "{:.0f}"),
        (axes[2], dynamic_range, "Dynamic range (dB)", "{:.1f}"),
    )
    for axis, values, ylabel, format_string in panels:
        bars = axis.bar(positions, values, color=colors, width=0.55)
        axis.set_xticks(positions, labels)
        axis.set_ylabel(ylabel)
        for bar, value in zip(bars, values, strict=True):
            axis.annotate(
                format_string.format(value),
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    if title is not None:
        figure.suptitle(title)
    return figure, axes


def plot_fixed_pattern_histograms(
    dsnu_e: ArrayLike,
    prnu_fraction: ArrayLike,
    *,
    dsnu_configured_e: float | None = None,
    prnu_configured_percent: float | None = None,
    dsnu_reported_e: float | None = None,
    title: str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot per-pixel DSNU and PRNU distributions from measured frames.

    Parameters
    ----------
    dsnu_e:
        Per-pixel mean dark signal in electrons from an averaged dark frame.
        This is the raw, post-clamp, post-quantization distribution; at low
        signal it is narrower than the configured pre-clamp DSNU.
    prnu_fraction:
        Per-pixel relative responsivity deviation from an averaged flat field,
        as a fraction (0.005 is 0.5 %).
    dsnu_configured_e:
        Optional configured pre-clamp DSNU, overlaid as a Gaussian reference.
    prnu_configured_percent:
        Optional configured PRNU in percent, overlaid as a Gaussian reference.
    dsnu_reported_e:
        Optional input-referred DSNU from the characterization report, printed
        alongside the raw spread so the clamp/quantization compression is not
        mistaken for a failed measurement.
    """
    dsnu_values = np.asarray(dsnu_e, dtype=np.float64)
    prnu_values = np.asarray(prnu_fraction, dtype=np.float64) * 100.0
    if dsnu_values.size == 0 or prnu_values.size == 0:
        raise ValueError("fixed-pattern histograms need at least one pixel per map")
    if not np.all(np.isfinite(dsnu_values)) or not np.all(np.isfinite(prnu_values)):
        raise ValueError("fixed-pattern maps must be finite")

    figure, axes_array = plt.subplots(1, 2, figsize=(10.0, 3.7))
    axes = cast(tuple[Axes, Axes], tuple(axes_array))

    counts, edges = np.histogram(dsnu_values, bins=_HISTOGRAM_BINS)
    axes[0].stairs(counts, edges, color=PALETTE[0], fill=False, label="measured dark frame")
    if dsnu_configured_e is not None and dsnu_configured_e > 0.0:
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = float(edges[1] - edges[0])
        scale = dsnu_values.size * width
        axes[0].plot(
            centers,
            scale * norm.pdf(centers, 0.0, dsnu_configured_e),
            color=PALETTE[1],
            linestyle="--",
            label=f"configured N(0, {dsnu_configured_e:g} e-)",
        )
    raw_sigma = float(np.std(dsnu_values, ddof=1))
    annotation = f"avg-frame sigma: {raw_sigma:.3f} e-\n(+ residual temporal noise)"
    if dsnu_reported_e is not None:
        annotation += f"\ninput-referred: {dsnu_reported_e:.3f} e-"
    axes[0].annotate(
        annotation,
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        va="top",
        fontsize=8.0,
    )
    axes[0].set_xlabel("Per-pixel mean dark signal (e-)")
    axes[0].set_ylabel("Pixel count")
    axes[0].set_title("DSNU (averaged dark-frame distribution)")
    axes[0].legend(loc="upper right")

    counts, edges = np.histogram(prnu_values, bins=_HISTOGRAM_BINS)
    axes[1].stairs(counts, edges, color=PALETTE[0], fill=False, label="measured flat field")
    if prnu_configured_percent is not None and prnu_configured_percent > 0.0:
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = float(edges[1] - edges[0])
        scale = prnu_values.size * width
        axes[1].plot(
            centers,
            scale * norm.pdf(centers, 0.0, prnu_configured_percent),
            color=PALETTE[1],
            linestyle="--",
            label=f"configured N(0, {prnu_configured_percent:g} %)",
        )
    axes[1].annotate(
        f"spatial sigma: {float(np.std(prnu_values, ddof=1)):.3f} %",
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        va="top",
        fontsize=8.0,
    )
    axes[1].set_xlabel("Per-pixel responsivity deviation within CFA channel (%)")
    axes[1].set_ylabel("Pixel count")
    axes[1].set_title("PRNU (averaged flat field, per CFA channel)")
    axes[1].legend(loc="upper right")

    if title is not None:
        figure.suptitle(title)
    return figure, axes
