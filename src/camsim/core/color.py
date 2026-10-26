"""Per-patch CIE Lab colour references bound to an illuminant and observer.

A :class:`ColorReference` is the colour ground truth for a chart under one
specific illumination. It stores CIE 1976 L*a*b* values, one row per patch in
layout order, together with the white point that the Lab conversion used and
the observer and provenance that produced it. The scene layer computes these
values spectrally (patch reflectance x illuminant x colour-matching functions
-> XYZ -> Lab); metrics consume them without importing the scene layer.

CIE Lab is approximately perceptually uniform: L* runs from 0 (black) to 100
(diffuse white), while a* and b* encode green-red and blue-yellow opposition.
A 24-patch ColorChecker spans roughly L* 20-95 and a*/b* about -60 to 100.
White points are CIE 1931 chromaticity coordinates, for example D65 at
``(0.3127, 0.3290)`` or a 3000 K blackbody at ``(0.4369, 0.4041)``. Real
instruments and standards use either the CIE 1931 2-degree or the CIE 1964
10-degree observer; the two differ enough to shift computed Lab values, so the
observer is stored explicitly rather than implied.

Failure artefacts guarded here: applying a D65 reference to a 3000 K capture
produces a large systematic delta-E offset that looks like a sensor colour
error; mixing 2-degree and 10-degree observers shifts every patch slightly;
feeding display-encoded sRGB values into a Lab comparison double-applies the
transfer function; and reordering the patch rows silently mismatches the
layout, so the reference carries them in a validated ``(n, 3)`` array and
provenance can record how the order was derived. Non-finite values are
rejected because they poison delta-E statistics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray

_DEFAULT_OBSERVER: Final[str] = "CIE 1931 2 degree"


def _validate_lab(value: object) -> NDArray[np.float64]:
    """Return Lab patches as a copied, finite ``(n, 3)`` float64 array."""
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lab must be a numeric (n, 3) array; {exc}") from exc
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"lab must be an (n, 3) array of CIE Lab values; got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("lab must contain at least one colour patch")
    if not np.all(np.isfinite(array)):
        raise ValueError("lab values must be finite (no NaN or infinity)")
    return array


def _validate_white_point(value: object) -> tuple[float, float]:
    """Return a finite ``(x, y)`` chromaticity pair."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"white_point_xy must be a 2-tuple of finite (x, y); got {value!r}")
    coordinates = list(value)
    if len(coordinates) != 2:
        raise ValueError(
            "white_point_xy must contain exactly two chromaticity coordinates; "
            f"got {len(coordinates)}"
        )
    try:
        x = float(coordinates[0])
        y = float(coordinates[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"white_point_xy must contain real numbers; got {value!r}") from exc
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError(f"white_point_xy must be finite; got {value!r}")
    return (x, y)


def _validate_non_empty_text(value: object, field: str) -> str:
    """Return a non-empty, non-whitespace string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string; got {value!r}")
    return value


@dataclass(eq=False)
class ColorReference:
    """Reference CIE Lab patches for a colour chart under one illuminant.

    Parameters
    ----------
    lab:
        ``(n, 3)`` array of CIE 1976 L*a*b* values, one row per chart patch in
        layout order. L* spans 0-100; a* and b* are unbounded in principle but
        ColorChecker patches stay within about -60 to 100. Stored as a copied
        float64 array; NaN and infinity are rejected.
    white_point_xy:
        CIE 1931 chromaticity ``(x, y)`` of the illuminant white point, for
        example ``(0.3127, 0.3290)`` for D65 or ``(0.4369, 0.4041)`` for a
        3000 K blackbody.
    illuminant_name:
        Human-readable illuminant label such as ``"D65"``, ``"3000K"``, or
        ``"4000K"``. Non-empty; used to select matching references and to
        report provenance.
    observer:
        CIE standard observer used for the XYZ-to-Lab conversion. Defaults to
        ``"CIE 1931 2 degree"``; a metric that uses the 1964 10-degree
        observer must state it explicitly.
    provenance:
        Free-form metadata describing how the values were produced (dataset,
        source URL, retrieval date, assumptions). Defaults to an empty dict.

    Notes
    -----
    The reference is only comparable with measurements made under the same
    illuminant and observer, so those fields are part of the value rather than
    optional notes. Equality compares the Lab array and all metadata.
    """

    lab: NDArray[np.float64]
    white_point_xy: tuple[float, float]
    illuminant_name: str
    observer: str = _DEFAULT_OBSERVER
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lab = _validate_lab(self.lab)
        self.white_point_xy = _validate_white_point(self.white_point_xy)
        self.illuminant_name = _validate_non_empty_text(self.illuminant_name, "illuminant_name")
        self.observer = _validate_non_empty_text(self.observer, "observer")
        if not isinstance(self.provenance, dict):
            raise ValueError(
                f"provenance must be a dict of metadata; got {type(self.provenance).__name__}"
            )
        self.provenance = dict(self.provenance)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColorReference):
            return NotImplemented
        return (
            np.array_equal(self.lab, other.lab)
            and self.white_point_xy == other.white_point_xy
            and self.illuminant_name == other.illuminant_name
            and self.observer == other.observer
            and self.provenance == other.provenance
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping with Lab rows as nested lists."""
        return {
            "lab": self.lab.tolist(),
            "white_point_xy": [self.white_point_xy[0], self.white_point_xy[1]],
            "illuminant_name": self.illuminant_name,
            "observer": self.observer,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a colour reference from :meth:`to_dict` output or JSON."""
        return cls(
            lab=payload["lab"],
            white_point_xy=payload["white_point_xy"],
            illuminant_name=payload["illuminant_name"],
            observer=payload.get("observer", _DEFAULT_OBSERVER),
            provenance=payload.get("provenance", {}),
        )
