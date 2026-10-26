"""Tests for per-patch CIE Lab colour references.

A :class:`ColorReference` binds Lab ground truth to the illuminant, white
point, and observer that produced it. These tests pin down shape and dtype
validation, JSON round-tripping, and the default observer/provenance metadata
so non-D65 measurements cannot silently lose the context that makes them
comparable.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from camsim.core.color import ColorReference

LAB = np.array([[50.0, 10.0, -10.0], [90.0, -5.0, 15.0]])
D65_XY = (0.3127, 0.3290)


def make_reference(**overrides: Any) -> ColorReference:
    """Build a valid two-patch D65 reference, overriding individual fields."""
    values: dict[str, Any] = {
        "lab": LAB.copy(),
        "white_point_xy": D65_XY,
        "illuminant_name": "D65",
        "provenance": {"source": "CIE", "method": "spectral integration"},
    }
    values.update(overrides)
    return ColorReference(**values)


def test_color_reference_json_roundtrip() -> None:
    reference = make_reference()

    payload = json.loads(json.dumps(reference.to_dict()))

    assert isinstance(payload["lab"], list)
    assert payload["white_point_xy"] == [D65_XY[0], D65_XY[1]]
    restored = ColorReference.from_dict(payload)
    np.testing.assert_array_equal(restored.lab, reference.lab)
    assert restored.lab.dtype == np.float64
    assert restored.white_point_xy == D65_XY
    assert restored.illuminant_name == "D65"
    assert restored.observer == "CIE 1931 2 degree"
    assert restored.provenance == reference.provenance


def test_defaults_observer_and_provenance_are_independent() -> None:
    first = ColorReference(lab=LAB, white_point_xy=D65_XY, illuminant_name="D65")
    second = ColorReference(lab=LAB, white_point_xy=D65_XY, illuminant_name="4000K")

    assert first.observer == "CIE 1931 2 degree"
    assert first.provenance == {}
    assert first.provenance is not second.provenance


@pytest.mark.parametrize(
    "lab",
    [
        np.zeros(3),
        np.zeros((2, 2)),
        np.zeros((2, 4)),
        np.zeros((0, 3)),
    ],
)
def test_color_reference_rejects_bad_lab_shapes(lab: np.ndarray) -> None:
    with pytest.raises(ValueError, match="lab"):
        ColorReference(lab=lab, white_point_xy=D65_XY, illuminant_name="D65")


def test_color_reference_rejects_nonfinite_lab() -> None:
    with pytest.raises(ValueError, match="lab"):
        ColorReference(
            lab=np.array([[50.0, np.nan, -10.0]]),
            white_point_xy=D65_XY,
            illuminant_name="D65",
        )
    with pytest.raises(ValueError, match="lab"):
        ColorReference(
            lab=np.array([[50.0, np.inf, -10.0]]),
            white_point_xy=D65_XY,
            illuminant_name="D65",
        )


def test_color_reference_rejects_bad_white_point() -> None:
    with pytest.raises(ValueError, match="white_point_xy"):
        make_reference(white_point_xy=(0.3127,))
    with pytest.raises(ValueError, match="white_point_xy"):
        make_reference(white_point_xy=(np.nan, 0.3290))
    with pytest.raises(ValueError, match="white_point_xy"):
        make_reference(white_point_xy="D65")


def test_lab_is_stored_as_a_float64_copy() -> None:
    source = LAB.copy()
    reference = ColorReference(lab=source, white_point_xy=D65_XY, illuminant_name="D65")

    source[0, 0] = 0.0

    assert reference.lab.dtype == np.float64
    np.testing.assert_array_equal(reference.lab, LAB)

    float32_reference = ColorReference(
        lab=LAB.astype(np.float32), white_point_xy=D65_XY, illuminant_name="D65"
    )
    assert float32_reference.lab.dtype == np.float64
    np.testing.assert_array_equal(float32_reference.lab, LAB)


def test_color_reference_validates_text_and_provenance_fields() -> None:
    with pytest.raises(ValueError, match="illuminant_name"):
        make_reference(illuminant_name="")
    with pytest.raises(ValueError, match="observer"):
        make_reference(observer="")
    with pytest.raises(ValueError, match="provenance"):
        make_reference(provenance=["not", "a", "dict"])


def test_color_reference_to_dict_is_json_serializable() -> None:
    payload = make_reference().to_dict()

    assert isinstance(payload["lab"], list)
    assert isinstance(payload["white_point_xy"], list)
    assert isinstance(payload["provenance"], dict)
    json.dumps(payload)
