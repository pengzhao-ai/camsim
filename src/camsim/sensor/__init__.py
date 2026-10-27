"""Physically based sensor model from spectral irradiance to RAW frames.

The subpackage layers the sensor from the outside in:

* :mod:`camsim.sensor.cfa` — CFA tiling and dimensionless spectral projection
  matrices from QE, IR-cut, and colour-filter curves.
* :mod:`camsim.sensor.noise` — Poisson shot/dark sampling, DSNU/PRNU, and row
  noise.
* :mod:`camsim.sensor.defects` — dead, hot, and stuck pixel maps plus PDAF
  layouts.
* :mod:`camsim.sensor.gain` — full-well clipping, conversion gain, and analog
  gain.
* :mod:`camsim.sensor.adc` — quantization, black-level pedestal, and digital
  gain.
* :mod:`camsim.sensor.binning` — analog charge summation and digital code
  averaging.
* :mod:`camsim.sensor.sensor` — :class:`Sensor` and the non-ideality helpers
  (crosstalk and blooming) that tie the chain together.

Importing this package does not import the ISP, scene, optics, or metrics
layers; the sensor is usable with an externally supplied spectral cube.
"""

from __future__ import annotations

from camsim.sensor.binning import (
    BinningPlan,
    binning_plan,
    mean_blocks,
    reduce_mask,
    sum_blocks,
)
from camsim.sensor.sensor import (
    Sensor,
    apply_blooming,
    apply_spatial_crosstalk,
    apply_spectral_crosstalk,
)

__all__ = [
    "BinningPlan",
    "Sensor",
    "apply_blooming",
    "apply_spatial_crosstalk",
    "apply_spectral_crosstalk",
    "binning_plan",
    "mean_blocks",
    "reduce_mask",
    "sum_blocks",
]
