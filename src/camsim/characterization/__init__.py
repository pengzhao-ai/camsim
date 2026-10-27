"""EMVA 1288-style sensor characterization and validation reports.

The package exposes the sensor measurement entry point and the structured
report it produces, plus the shared G3 ISP validation and G4 metrics validation
used by the gallery examples and their tests. ``characterize_sensor`` drives a
simulated :class:`camsim.Sensor` through flat-field and dark-frame sweeps;
:class:`EMVAReport` carries the measured gain, noise, saturation, dynamic
range, fixed-pattern metrics, and the provenance needed to interpret them.
``measure_g3`` measures the ISP end to end and ``measure_g4`` measures the
independent metrics layer against analytic MTF, known-noise SNR, and
colour-science delta-E; both return JSON-serialisable reports.
``build_gallery`` produces the arrays used by the stage-gallery figures.
"""

from __future__ import annotations

from .emva import characterize_sensor
from .isp import G3Gallery, build_gallery, measure_g3
from .metrics import measure_g4
from .report import EMVAReport, SNRPoint, render_g3_report, render_g4_report

__all__ = [
    "EMVAReport",
    "G3Gallery",
    "SNRPoint",
    "build_gallery",
    "characterize_sensor",
    "measure_g3",
    "measure_g4",
    "render_g3_report",
    "render_g4_report",
]
