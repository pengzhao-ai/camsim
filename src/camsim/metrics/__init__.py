"""Image-quality metrics: chart layouts, MTF/e-SFR, SNR, colour, and acutance.

This package is import-independent by design: it operates on arrays plus
metadata and must never import ``camsim.sensor``, ``camsim.isp``,
``camsim.scene``, or ``camsim.optics``. The scene layer owns the convenience
constructors that bind a chart to its spectrally computed illuminant
reference; those constructors return the :class:`~camsim.metrics.chart.ChartLayout`
defined here, so the dependency runs scene -> metrics, never the reverse.

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

from camsim.metrics.chart import ChartLayout, PatchROI, extract_patch

__all__ = ["ChartLayout", "PatchROI", "extract_patch"]
