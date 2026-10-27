"""EMVA 1288-style sensor characterization and validation reports.

The package exposes one measurement entry point and the structured report it
produces. ``characterize_sensor`` drives a simulated :class:`camsim.Sensor`
through flat-field and dark-frame sweeps; :class:`EMVAReport` carries the
measured gain, noise, saturation, dynamic range, fixed-pattern metrics, and the
provenance needed to interpret them.
"""

from __future__ import annotations

from .emva import characterize_sensor
from .report import EMVAReport, SNRPoint

__all__ = ["EMVAReport", "SNRPoint", "characterize_sensor"]
