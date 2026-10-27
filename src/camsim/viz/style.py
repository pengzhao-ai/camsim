"""Colorblind-friendly Matplotlib defaults for every camsim figure.

Human colour vision compares the responses of three cone classes (S, M, L), and
about 8 % of men and 0.5 % of women have a colour-vision deficiency that removes
or distorts one of those comparisons. The common red-green deficiencies
(deuteranopia and protanopia) collapse distinctions that many default plotting
palettes rely on, so two curves that look clearly separate to a typical viewer
can become indistinguishable. camsim therefore uses the Okabe-Ito palette,
which varies lightness and blue-yellow opposition as well as red-green
opposition, and remains separable under all three common dichromacies and in
greyscale print.

The defaults also encode plotting conventions that matter for image-quality
work: a white figure background for print, a light grid behind the data
(``axes.axisbelow``), no top or right spines, and DejaVu Sans at 10 pt. These
are presentation choices, not physical claims: a spectral axis remains linear in
nanometres unless a caller explicitly changes it, and an image colormap is a
visual aid rather than calibrated colour. ``image.cmap`` is set to the
perceptually uniform ``viridis`` rather than a rainbow map, because rainbow
maps invent banding and non-monotonic lightness that readers mistake for real
structure in a PSF or irradiance image.

Real-world ranges: figure DPI is usually 100-300 (150 is a good compromise for
screen review and print), line widths 1-2 pt, font sizes 9-11 pt, and
colour-blind-safe palettes typically carry 6-10 distinguishable categorical
colours. Beyond about eight categories a legend stops being readable regardless
of palette, so later metrics modules should facet or label directly.

Failure artefacts guarded here: a red/green pair that collapses under
deuteranopia; a rainbow colormap that creates false edges in smooth data; a
transparent or dark figure background that becomes unreadable in a printed
report; grid lines drawn over the data and mistaken for measurements; and a
figure that silently depends on the caller's interactive backend, which breaks
headless CI. ``set_style`` changes only Matplotlib state and is idempotent, so
notebooks can call it at the top of every session without accumulating styles.
"""

from __future__ import annotations

from typing import Any, Final, cast

import matplotlib as mpl

__all__ = ["PALETTE", "set_style"]

PALETTE: Final[tuple[str, ...]] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
"""Okabe-Ito colorblind-safe categorical palette.

The order puts blue first because it survives every common dichromacy and
greyscale conversion; vermillion and bluish green stay distinguishable from
each other under deuteranopia, which the classic red/green pair does not.
"""

# rcParams accepts a Cycler object or its string representation. Building the
# string keeps the palette in one tuple and avoids importing cycler directly.
_PROP_CYCLE: Final[str] = "cycler('color', [" + ", ".join(repr(c) for c in PALETTE) + "])"

_STYLE: Final[dict[str, Any]] = {
    "figure.figsize": (8.0, 4.5),
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.titlesize": 11.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.0,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": _PROP_CYCLE,
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.7,
    "legend.frameon": False,
    "legend.fontsize": 9.0,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "image.cmap": "viridis",
}


def set_style() -> None:
    """Apply camsim's colorblind-friendly Matplotlib defaults.

    The function updates a fixed set of ``rcParams`` and returns ``None``. It
    sets the categorical colour cycle to :data:`PALETTE`, selects DejaVu Sans,
    turns on a light grid behind the data, removes the top and right spines,
    and chooses a perceptually uniform image colormap. Because only those keys
    are touched, an embedding application can layer its own house style before
    or after the call; because the values are fixed, repeated calls produce the
    same state and never depend on the previously active style.
    """
    # rcParams.update is typed against the full literal key union; the style
    # mapping is a plain dict, so the cast states that every key was checked by
    # Matplotlib's own rcParams validator at runtime.
    mpl.rcParams.update(cast(Any, _STYLE))
