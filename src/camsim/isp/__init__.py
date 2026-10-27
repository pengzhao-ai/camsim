"""Linear-space ISP stages and the unified RAW-to-rendered pipeline.

Each module implements one operation of the fixed reconstruction chain
``DPC -> BLC -> LSC -> AWB -> demosaic -> CCM -> denoise -> sharpen -> tone``:
defective-pixel correction, black-level correction, lens-shading correction,
white balance, demosaicing, colour correction, denoise, sharpening, and the
final sRGB/Display P3 transfer. Every stage through sharpening accepts and
returns linear float32; only :mod:`camsim.isp.tone` is non-linear.

The integrated pipeline lives in :mod:`camsim.isp.pipeline` and is exported as
:class:`camsim.ISP`. This package is deliberately lazy: importing it executes
only this docstring and loads no stage modules, so ``import camsim.metrics``
stays independent of the sensor and ISP layers and ``import camsim`` stays
cheap. Individual stages remain importable on their own for analysis,
calibration, and the gallery notebooks.
"""
