"""Colorblind-friendly plotting helpers for spectra, PSFs, MTF, sensors, figures.

A camsim figure is a projection of a physical quantity, not a measurement in
itself. A spectral plot carries wavelength in nanometres and either relative
spectral power, a dimensionless reflectance factor, or a measured response; a
PSF profile is a normalized irradiance distribution whose absolute scale is
arbitrary; an MTF curve is the magnitude of the normalized discrete Fourier
transform of that PSF, plotted only to the 0.5 cycles/pixel Nyquist limit. The
style layer fixes the Okabe-Ito colorblind-safe palette so categorical curves
stay separable under common dichromacies and in greyscale print.

Real-world conventions: visible-band spectral data run roughly 360-830 nm,
MTF axes stop at Nyquist, PSF kernels span a few to a few dozen pixels, raster
figures use 100-300 dpi, and a readable legend holds at most about eight
categorical curves. Figure output is deterministic: the walkthroughs write
fixed names under ``CAMSIM_FIG_DIR`` (default ``results/figures``) rather than
creating timestamped files.

Failure artefacts guarded here: an unlabelled wavelength axis or a log y axis
that hides the passband shape; a PSF profile normalized by its unit sum, which
makes every kernel look equally flat; an MTF curve drawn past Nyquist, where
aliasing can make it appear to rise; a rainbow colormap that invents structure;
a figure path whose parent does not exist; and Matplotlib's environment-
dependent metadata, which would make repeated runs differ byte for byte.

The public surface covers the Phase 1 optics figures (spectrum, PSF profile,
MTF), the Phase 2 sensor figures (spectral response, photon transfer curve,
SNR, dark current, dual-gain comparison, DSNU/PRNU histograms), and Phase 3's
ISP stage gallery (pipeline trace, zoomed stage crops, and the bilinear/MHC
demosaic comparison), plus a deterministic figure saver. Later phases extend
the same namespace with metric and experiment plots without changing these
entry points.
"""

from __future__ import annotations

from camsim.viz.pipeline import (
    plot_demosaic_comparison,
    plot_pipeline_trace,
    plot_zoom_crops,
)
from camsim.viz.sensor import (
    plot_dark_current_vs_temperature,
    plot_fixed_pattern_histograms,
    plot_mode_comparison,
    plot_ptc,
    plot_sensor_spectral_response,
    plot_snr_vs_irradiance,
)
from camsim.viz.spectra import figure_dir, plot_mtf, plot_psf, plot_spectrum, save_figure
from camsim.viz.style import PALETTE, set_style

__all__ = [
    "PALETTE",
    "figure_dir",
    "plot_dark_current_vs_temperature",
    "plot_demosaic_comparison",
    "plot_fixed_pattern_histograms",
    "plot_mode_comparison",
    "plot_mtf",
    "plot_pipeline_trace",
    "plot_psf",
    "plot_ptc",
    "plot_sensor_spectral_response",
    "plot_snr_vs_irradiance",
    "plot_spectrum",
    "plot_zoom_crops",
    "save_figure",
    "set_style",
]
