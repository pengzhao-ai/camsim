"""Camera orchestration and the opt-in auto-exposure policy.

:class:`~camsim.camera.camera.Camera` binds a lens, a sensor, and an ISP into
one object that can turn a rendered scene into a display image. The default
capture is the fixed-exposure baseline from the sensor preset; auto-exposure is
an explicit request that probes the real sensor, meters the black-corrected
signal with a centre-weighted statistic, and reports every decision.

The package re-exports the three public value types:

* :class:`Camera` — capture and auto-exposure entry points.
* :class:`ExposureSettings` — a fixed exposure, analog gain, and readout mode.
* :class:`ExposureResult` — the measured settings and evidence from one
  auto-exposure run.

Importing the package loads the camera module, which in turn imports the
sensor, ISP, and scene layers; ``import camsim`` itself remains lazy.
"""

from __future__ import annotations

from camsim.camera.camera import Camera, ExposureResult, ExposureSettings

__all__ = ["Camera", "ExposureResult", "ExposureSettings"]
