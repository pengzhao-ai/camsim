"""RAW and rendered-image input/output helpers.

The two formats this package writes describe different physical quantities and
must never be interchanged:

* :mod:`camsim.io.raw` stores a :class:`~camsim.core.frames.RawFrame` as exact
  ``uint16`` ADC codes plus a mandatory JSON sidecar and an optional compressed
  mask archive. The codes still include the black-level pedestal and depend on
  the CFA pattern, bit depth, gains, exposure, mode, and sensor configuration.
* :mod:`camsim.io.images` stores a display-referred
  :class:`~camsim.core.frames.RenderedImage` as 8-bit PNG, 8-bit RGB TIFF, or
  16-bit grayscale TIFF, with a sidecar recording the colour space, channel
  layout, bit depth, and metadata. The values have already passed through the
  output transfer function and are not linear light.

Both loaders validate aggressively: a missing or mismatched sidecar, a dtype
or shape disagreement, an unknown schema version, and a lossy image format all
raise with the offending file named. Precision is part of the contract — 8-bit
display files quantize to 1/255 steps, and 16-bit mono TIFF to 1/65535 — so a
metric computed from a reloaded file must account for that container depth.

Typical RAW frames are 10-16 bits per pixel and rendered images are 8 bits per
channel (16 for scientific grayscale). Failure artefacts guarded here: reading
RAW codes as display values (gamma applied twice), assuming a missing image
sidecar means sRGB when it was Display P3, and letting a JPEG enter a
measurement chain where codec ringing masquerades as camera sharpness.
"""

from __future__ import annotations

from camsim.io.images import load_image, save_image
from camsim.io.raw import load_raw, save_raw

__all__ = ["load_image", "load_raw", "save_image", "save_raw"]
