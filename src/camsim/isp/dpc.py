"""Mask-guided defective-pixel correction in the RAW linear domain.

A production image sensor is not perfectly uniform. Dead pixels have no
photoresponse and read near the black level, hot pixels carry anomalously high
dark current that grows with exposure and temperature, and stuck pixels are
pinned to a fixed code by a shorted or open readout node. The sensor flags all
three classes in ``RawFrame.defect_mask``; their values are meaningless scene
samples, so the ISP replaces each one with an estimate from the surrounding
pixels that viewed the scene through the *same* colour filter.

Same-colour donor selection is the central physical constraint. A Bayer mosaic
samples one spectral band per site, so borrowing a green value for a red pixel
would mix spectral bands, create colour fringes around the defect, and bias any
white-balance or colour-matrix fit that follows. Dead and stuck pixels have no
useful signal, and hot pixels are dominated by dark current, so defective sites
must also be excluded from the donor set; otherwise the correction copies a bad
value into its neighbours. Phase-detection autofocus (PDAF) sites are not
defects: their photodiode is deliberately partially masked and records a
fraction of the light, so DPC leaves them untouched and does not use them as
donors.

For each defect the donor set is the same-CFA-colour samples within a
Chebyshev radius of two pixels (a 5x5 window). On a Bayer lattice that yields
up to eight donors for red and blue and up to twelve for green, whose two tile
phases share one colour channel. ``method="neighbor"`` replaces the site with
the mean of the valid donors; ``method="median"`` uses their median, which is
robust to a single bright neighbour such as an undetected hot pixel. If a
defect cluster or the image border leaves no valid donor, the original code is
left in place rather than inventing a value. Pixels that are not flagged are
copied bit-exactly, so DPC is a pass-through for the overwhelming majority of
the frame.

Typical defect rates are 1e-6 to 1e-4 for dead and stuck pixels and 1e-5 to
1e-3 for hot pixels; a mobile sensor commonly carries tens to a few hundred
defects. The stage works in RAW digital numbers and returns float32: it does not
subtract the black-level pedestal (BLC does that next) and applies no gain,
gamma, or other non-linear operation, preserving the linear-space invariant.

Failure artefacts guarded here: interpolating across CFA colours (colour
artefacts and white-balance bias), using a defective or PDAF site as a donor
(copying the fault), correcting a PDAF pixel (destroying the autofocus signal),
inventing values for an unsupported cluster (hallucinated detail), altering
unflagged pixels (breaking the deterministic pass-through contract), and
accepting a mask whose shape or dtype no longer matches the frame.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from camsim.config.schema import DPCConfig
from camsim.core.frames import RawFrame
from camsim.sensor.cfa import cfa_color_at

__all__ = ["DPCConfig", "correct_defects"]

# Two pixels is the smallest radius that reaches a same-colour neighbour on a
# Bayer lattice for red and blue (whose nearest same-colour sites are at a
# distance of two), while still collecting the diagonal green samples.
_NEIGHBOR_RADIUS = 2
_METHODS = ("neighbor", "median")


def correct_defects(raw: RawFrame, config: DPCConfig) -> NDArray[np.float32]:
    """Replace flagged defective pixels from same-CFA-colour neighbours.

    Parameters
    ----------
    raw:
        RAW frame whose ``data`` are ADC codes including the black-level
        pedestal. ``defect_mask`` marks dead, hot, and stuck sites; it may be
        ``None`` when the sensor has no flagged defects. ``pdaf_mask`` marks
        intentional phase-detection sites, which are neither corrected nor
        used as donors.
    config:
        Validated :class:`~camsim.config.schema.DPCConfig`. ``enabled=False``
        makes the stage a float32 pass-through. ``method="neighbor"`` uses the
        mean of the valid donors and ``"median"`` their median.

    Returns
    -------
    A new float32 array of shape ``(H, W)``. Flagged non-PDAF defects are
    replaced; every other sample equals ``raw.data`` cast exactly to float32.
    The pedestal is preserved, so black-level correction must still follow.

    Raises
    ------
    ValueError
        If ``raw`` or ``config`` has the wrong type, ``config.method`` is not a
        supported method, or the defect/PDAF masks no longer match the frame's
        shape or boolean dtype. Silently accepting a mismatched mask would
        correct the wrong pixels.

    Notes
    -----
    Donor search is local (5x5) and deterministic. Corrections use the codes of
    the donor sites without any colour or gain normalisation, because the CFA
    filter and analog gain are common to all donors of one colour. A defect
    surrounded entirely by defects or PDAF sites is left unchanged.
    """
    if not isinstance(raw, RawFrame):
        raise ValueError(f"raw must be a RawFrame; got {type(raw).__name__}")
    if not isinstance(config, DPCConfig):
        raise ValueError(f"config must be a DPCConfig; got {type(config).__name__}")
    data = raw.data
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.dtype != np.uint16:
        raise ValueError(
            "raw.data must be a 2-D uint16 array; "
            f"got {type(data).__name__} with dtype {getattr(data, 'dtype', None)}"
        )
    output = np.array(data, dtype=np.float32, copy=True)
    if not config.enabled:
        return output

    method = config.method
    if method not in _METHODS:
        raise ValueError(f"method must be one of {', '.join(_METHODS)}; got {method!r}")

    defect = _validate_mask(raw.defect_mask, "defect_mask", data.shape)
    pdaf = _validate_mask(raw.pdaf_mask, "pdaf_mask", data.shape)
    if defect is None or not bool(defect.any()):
        return output
    if pdaf is None:
        pdaf = np.zeros(data.shape, dtype=np.bool_)

    targets = defect & ~pdaf
    if not bool(targets.any()):
        return output

    height, width = data.shape
    row_indices, column_indices = np.nonzero(targets)
    for row, column in zip(row_indices.tolist(), column_indices.tolist(), strict=True):
        y = int(row)
        x = int(column)
        colour = cfa_color_at(raw.cfa, y, x)
        donors: list[float] = []
        for delta_y in range(-_NEIGHBOR_RADIUS, _NEIGHBOR_RADIUS + 1):
            neighbour_y = y + delta_y
            if neighbour_y < 0 or neighbour_y >= height:
                continue
            for delta_x in range(-_NEIGHBOR_RADIUS, _NEIGHBOR_RADIUS + 1):
                if delta_y == 0 and delta_x == 0:
                    continue
                neighbour_x = x + delta_x
                if neighbour_x < 0 or neighbour_x >= width:
                    continue
                if cfa_color_at(raw.cfa, neighbour_y, neighbour_x) != colour:
                    continue
                if defect[neighbour_y, neighbour_x] or pdaf[neighbour_y, neighbour_x]:
                    continue
                donors.append(float(data[neighbour_y, neighbour_x]))
        if not donors:
            continue
        if method == "neighbor":
            output[y, x] = np.float32(np.mean(donors))
        else:
            output[y, x] = np.float32(np.median(donors))
    return output


def _validate_mask(mask: Any, field: str, shape: tuple[int, int]) -> NDArray[np.bool_] | None:
    """Return a boolean mask matching ``shape``, or ``None``.

    ``RawFrame`` validates its masks at construction, but frames and their
    arrays are mutable; re-checking here turns a later mutation into a clear
    error instead of silently correcting the wrong pixels.
    """
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray):
        raise ValueError(
            f"{field} must be None or a boolean numpy array; got {type(mask).__name__}"
        )
    if mask.dtype != np.bool_:
        raise ValueError(f"{field} must have dtype bool; got dtype {mask.dtype}")
    if mask.shape != shape:
        raise ValueError(f"{field} shape {mask.shape} must match data shape {shape}")
    return cast(NDArray[np.bool_], mask)
