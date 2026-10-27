"""Colorblind-friendly plotting helpers for spectra, PSFs, MTF, and figures.

Figures are presentation artefacts, not measurements: their axes, units, and
normalizations are documented here so a curve cannot be mistaken for raw data.
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
