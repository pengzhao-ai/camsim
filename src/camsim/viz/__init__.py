"""Colorblind-friendly plotting helpers for spectra, PSFs, MTF, and figures.

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

The public surface is intentionally small for the current phase: a style
initializer, three physical plot types (spectrum, PSF profile, MTF), and a
deterministic figure saver. Later phases extend the same namespace with ISP,
colour, and metric plots without changing these entry points.
"""

from __future__ import annotations

from camsim.viz.spectra import figure_dir, plot_mtf, plot_psf, plot_spectrum, save_figure
from camsim.viz.style import PALETTE, set_style

__all__ = [
    "PALETTE",
    "figure_dir",
    "plot_mtf",
    "plot_psf",
    "plot_spectrum",
    "save_figure",
    "set_style",
]
