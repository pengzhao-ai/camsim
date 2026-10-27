"""Radial cos-fourth natural vignetting for a thin lens.

Natural vignetting is geometric, not a defect of the glass. A pixel at image
height :math:`r` from the optical axis receives rays at field angle
:math:`\\theta = \\arctan(r / f)`, where :math:`f` is the focal length. The
oblique bundle is projected onto the image plane at an angle, its entrance
pupil appears foreshortened, and the distance from the exit pupil grows; the
product of those effects for a thin lens is the familiar ``cos**4(theta)``
illumination falloff::

    V(r) = cos**4(theta) = 1 / (1 + (r / f)**2)**2

normalized to one on axis. At the corner of a 1/1.3-inch sensor
(:math:`r \\approx 5.7` mm) behind a 6.4 mm phone lens, ``r / f`` is close to
0.9 and only about 31 % of the axial irradiance remains, roughly 1.7 stops.
Real lenses often fall off faster still because of pupil aberration and
microlens crosstalk, which is why lens-shading correction is calibrated from a
flat field rather than trusted from this analytic curve.

``LensConfig.vignetting`` is a strength in ``[0, 1]``. A value of one applies
the full physical curve, zero (or ``None``) disables the falloff, and
intermediate values blend linearly::

    map = 1 - strength * (1 - cos**4(theta))

The blend is useful for modelling a lens whose shading has already been
calibrated out by the ISP: disabling the scene-level falloff and leaving the
correction to the flat-field stage avoids applying it twice.

Real-world parameter ranges: focal lengths 1.5-50 mm for phone and machine
vision lenses, pixel pitches 0.6-10 um, and corner falloff from a few per cent
for a long telephoto to more than two stops for an ultra-wide phone lens.

Failure artefacts guarded here: using object-space field angle instead of
image-space angle; passing pixel indices where physical radius is required;
forgetting that an oversampled render has a proportionally smaller sample pitch
(the map then collapses towards one); applying ``cos**4`` once here and once
more in the lens formula; using the full sensor diagonal as the pixel radius;
and assuming an exact ``V = 1`` central sample when an even-sized grid has its
optical centre between pixels.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from camsim.config.schema import LensConfig

__all__ = ["vignetting_map"]

_UM_TO_MM: Final[float] = 1e-3


def _validate_shape(shape: object) -> tuple[int, int]:
    """Return a positive ``(height, width)`` integer pair."""
    if isinstance(shape, (str, bytes)):
        raise ValueError(f"shape must be a (height, width) pair; got {shape!r}")
    try:
        dimensions: tuple[Any, ...] = tuple(shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"shape must be a (height, width) pair; {exc}") from exc
    if len(dimensions) != 2:
        raise ValueError(f"shape must contain exactly two dimensions; got {shape!r}")
    checked: list[int] = []
    for dimension in dimensions:
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
            raise ValueError(f"shape dimensions must be integers; got {shape!r}")
        if int(dimension) < 1:
            raise ValueError(f"shape dimensions must be positive; got {shape!r}")
        checked.append(int(dimension))
    return (checked[0], checked[1])


def vignetting_map(
    shape: tuple[int, int],
    lens: LensConfig,
    pixel_pitch_um: float,
) -> NDArray[np.float64]:
    """Return the radial illumination falloff map in ``[0, 1]``.

    Parameters
    ----------
    shape:
        ``(height, width)`` of the sampled image. Pixel centres are assumed to
        sit at integer indices, so the optical centre is at
        ``((height - 1) / 2, (width - 1) / 2)`` in index coordinates. For an
        even dimension the axis falls between pixels and no sample reaches
        exactly one.
    lens:
        A validated :class:`~camsim.config.schema.LensConfig`. Its
        ``focal_length_mm`` sets the field-angle scale and its ``vignetting``
        strength blends between a flat map (``None`` or zero) and the full
        ``cos**4`` law (one).
    pixel_pitch_um:
        Sampling pitch of the grid in micrometres. For a scene rendered with
        oversampling ``N``, pass ``pixel_pitch_um / N`` because each rendered
        sample covers a proportionally smaller part of the sensor.

    Returns
    -------
    A float64 ``(height, width)`` map, one at the optical centre and falling
    radially outwards. The map multiplies scene irradiance; it is not a gain
    map for lens-shading correction.

    Raises
    ------
    ValueError
        If the shape is not two positive dimensions or the pitch is not finite
        and positive.
    """
    height, width = _validate_shape(shape)
    try:
        pitch_um = float(pixel_pitch_um)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pixel_pitch_um must be finite and positive; {exc}") from exc
    if not math.isfinite(pitch_um) or pitch_um <= 0.0:
        raise ValueError(f"pixel_pitch_um must be finite and positive; got {pixel_pitch_um!r}")

    # Coordinate 0 is the optical axis; the centre of a pixel grid falls on a
    # pixel for odd dimensions and between the two central pixels otherwise.
    y_mm = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) * pitch_um
    x_mm = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * pitch_um
    y_mm *= _UM_TO_MM
    x_mm *= _UM_TO_MM

    focal_length_mm = float(lens.focal_length_mm)
    radius_ratio_squared = (y_mm[:, None] ** 2 + x_mm[None, :] ** 2) / focal_length_mm**2
    # cos(theta) = 1 / sqrt(1 + (r/f)**2), so cos**4 = 1 / (1 + (r/f)**2)**2.
    cos_fourth: NDArray[np.float64] = 1.0 / (1.0 + radius_ratio_squared) ** 2

    strength = 0.0 if lens.vignetting is None else float(lens.vignetting)
    falloff: NDArray[np.float64] = 1.0 - strength * (1.0 - cos_fourth)
    return falloff
