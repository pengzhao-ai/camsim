"""Reference lens geometry and on-axis spectral transmission.

A lens performs two jobs that this module exposes separately. Geometrically it
maps field angle to sensor position: a focal length :math:`f` images a sensor
diagonal :math:`d` over the diagonal field of view :math:`2\\arctan(d / 2f)`,
and a pixel of pitch :math:`p` subtends :math:`\\arctan(p / f)` radians at the
rear principal plane. The reciprocal of that angle in degrees is the
pixels-per-degree scale used to express MTF in cycles/degree. For the reference
phone lens (6.4 mm focal length, 11.46 mm diagonal, 1.12 um pitch) the field is
about 84 degrees and the scale about 100 pixels/degree, which is why phone MTF
curves in cycles/degree look much better than the same curves in
cycles/pixel.

Radiometrically a lens transmits only part of the incident flux. Anti-reflection
coatings on modern phone optics transmit roughly 85-98 % on axis over the
visible band, falling towards the blue and near infrared; a bare glass surface
loses about 4 % per air-glass interface. The configuration stores either a
broadband scalar or a CSV path to a measured spectral curve. This module
resolves the path and interpolates the curve; it does not integrate the curve
against a scene spectrum, which belongs to the propagation stage.

Real-world parameter ranges: f-numbers from f/0.7 (specialist low-light optics)
to f/22 (diffraction-dominated), focal lengths from about 1.5 mm (ultra-wide
phone) to hundreds of millimetres (telephoto), and sensor diagonals from
1/6-inch (3 mm) to medium format (55 mm). Transmission curves are usually
tabulated from 350-1000 nm at 1-10 nm spacing.

Failure artefacts guarded here: a zero or negative f-number is unphysical and
the configuration schema rejects it; a transmittance above one would invent
energy; a curve with non-increasing wavelengths mislabels every sample; a
transmittance outside ``[0, 1]`` almost always means the file stores percent
rather than a fraction; and a missing or malformed CSV must raise instead of
silently falling back to full transmission, which would make every low-light
comparison optimistic.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from camsim.config.schema import LensConfig

__all__ = ["LensConfig", "transmittance_at"]

_CSV_COLUMNS: Final[int] = 2


def _read_transmittance_curve(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read a two-column ``wavelength_nm, transmittance`` CSV curve.

    A non-numeric header row and ``#`` comment lines are allowed. The returned
    wavelengths are finite, positive, and strictly increasing, and every
    transmittance lies in ``[0, 1]``: those are the preconditions for linear
    interpolation. Values of zero are permitted because a coating or IR-cut
    filter can genuinely block a band, even though the scalar configuration
    field requires a strictly positive broadband average.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read lens transmittance CSV {path}: {exc}") from exc

    samples: list[tuple[float, float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != _CSV_COLUMNS:
            raise ValueError(
                "lens transmittance CSV must have two columns "
                f"(wavelength_nm,transmittance); line {line_number} has {len(columns)}"
            )
        try:
            wavelength_nm = float(columns[0])
            transmittance = float(columns[1])
        except ValueError:
            if not samples:
                # A conventional header such as "wavelength_nm,transmittance"
                # carries no data; skip it once at the top of the file.
                continue
            raise ValueError(
                f"lens transmittance CSV line {line_number} is not numeric: {line!r}"
            ) from None
        samples.append((wavelength_nm, transmittance))

    if len(samples) < 2:
        raise ValueError(
            "lens transmittance CSV must contain at least two wavelength samples; "
            f"got {len(samples)}"
        )

    data = np.asarray(samples, dtype=np.float64)
    wavelengths = data[:, 0]
    values = data[:, 1]
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("lens transmittance wavelengths must be finite and positive")
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("lens transmittance wavelengths must be strictly increasing")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("lens transmittance values must be finite and within [0, 1]")
    return wavelengths, values


def transmittance_at(lens: LensConfig, wavelength_nm: float) -> float:
    """Return the lens's on-axis transmittance at ``wavelength_nm``.

    Parameters
    ----------
    lens:
        A validated :class:`~camsim.config.schema.LensConfig`. When
        ``lens.transmittance`` is a scalar it is returned unchanged for every
        wavelength; when it is a CSV path the two-column curve is interpolated
        linearly and held constant outside its measured range.
    wavelength_nm:
        Query wavelength in nanometres; must be finite and positive.

    Returns
    -------
    The dimensionless transmittance in ``[0, 1]``.

    A negative or non-finite wavelength has no physical meaning, and an
    out-of-band query without endpoint clamping would require extrapolating a
    coating curve beyond its measured range, inventing performance the
    datasheet does not support.
    """
    if not math.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError(f"wavelength_nm must be finite and positive; got {wavelength_nm!r}")

    specification = lens.transmittance
    if not isinstance(specification, str):
        return float(specification)

    wavelengths, values = _read_transmittance_curve(Path(specification).expanduser())
    return float(np.interp(wavelength_nm, wavelengths, values))
