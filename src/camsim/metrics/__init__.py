"""Image-quality metrics: chart layouts, MTF/e-SFR, SNR, colour, and acutance.

This package is import-independent by design: it operates on arrays plus
metadata and must never import ``camsim.sensor``, ``camsim.isp``,
``camsim.scene``, or ``camsim.optics``. The scene layer owns the convenience
constructors that bind a chart to its spectrally computed illuminant
reference; those constructors return the :class:`~camsim.metrics.chart.ChartLayout`
defined here, so the dependency runs scene -> metrics, never the reverse.

The package is the public metrics API:

* chart geometry - :class:`~camsim.metrics.chart.ChartLayout`,
  :class:`~camsim.metrics.chart.PatchROI`, and
  :func:`~camsim.metrics.chart.extract_patch`;
* resolution - :func:`~camsim.metrics.mtf.slanted_edge_mtf`,
  :func:`~camsim.metrics.mtf.extract_raw_plane`, and the
  :class:`~camsim.metrics.mtf.MTFCurve` / :class:`~camsim.metrics.mtf.MTFResult`
  value objects;
* noise - :func:`~camsim.metrics.snr.snr_patch`,
  :func:`~camsim.metrics.snr.snr_temporal`,
  :func:`~camsim.metrics.snr.snr_channels`, and
  :class:`~camsim.metrics.snr.SNRResult`;
* colour - :func:`~camsim.metrics.color.delta_e_patches`,
  :func:`~camsim.metrics.color.linear_rgb_to_lab`, and the
  :class:`~camsim.metrics.color.PatchColorResult` /
  :class:`~camsim.metrics.color.ColorResult` value objects;
* sharpness - :func:`~camsim.metrics.acutance.edge_acutance`;
* the unified harness - :func:`~camsim.metrics.summary.measure_all` and its
  :class:`~camsim.metrics.summary.MetricReport`.

The layout types are the shared vocabulary of every metric. A
:class:`~camsim.metrics.chart.PatchROI` names a half-open sensor-pixel region,
:class:`~camsim.metrics.chart.ChartLayout` groups those regions with an
optional edge region and colour/camera metadata, and
:func:`~camsim.metrics.chart.extract_patch` returns the pixels of one region
without copying. Metric modules (e-SFR, SNR, delta-E, acutance) consume these
types plus plain NumPy arrays.

Regions are measured in sensor pixels, never in millimetres or on the
oversampled rendering grid. Typical values: a ColorChecker patch covers tens of
pixels, an ISO 12233-style edge region is 100-400 pixels tall, pixel pitches run
from about 0.8 um (small phone sensors) to 10 um (large-format), and focal
lengths from about 2 mm to 50 mm. Colour references are bound to the
illuminant that produced the image; a D65 reference applied to a 3000 K capture
is a white-point error, not a sensor colour error.

Failure artefacts guarded here: transposed coordinates measure the wrong patch,
a region crossing a patch boundary averages two reflectances, overlapping
regions double-count pixels, a layout with no patches and no edge gives
``measure_all`` nothing to do, and importing a simulator package from this
layer would couple the measurement harness to the model it is meant to judge.
"""

from __future__ import annotations

from camsim.metrics.acutance import edge_acutance
from camsim.metrics.chart import ChartLayout, PatchROI, extract_patch
from camsim.metrics.color import ColorResult, PatchColorResult, delta_e_patches, linear_rgb_to_lab
from camsim.metrics.mtf import MTFCurve, MTFResult, extract_raw_plane, slanted_edge_mtf
from camsim.metrics.snr import SNRResult, snr_channels, snr_patch, snr_temporal
from camsim.metrics.summary import MetricReport, measure_all

__all__ = [
    "ChartLayout",
    "ColorResult",
    "MTFCurve",
    "MTFResult",
    "MetricReport",
    "PatchColorResult",
    "PatchROI",
    "SNRResult",
    "delta_e_patches",
    "edge_acutance",
    "extract_patch",
    "extract_raw_plane",
    "linear_rgb_to_lab",
    "measure_all",
    "slanted_edge_mtf",
    "snr_channels",
    "snr_patch",
    "snr_temporal",
]
