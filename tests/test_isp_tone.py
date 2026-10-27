"""Tests for the sRGB/Display P3 transfer functions and the tone stage.

The tone stage is the pipeline's final, deliberately non-linear step. Every
stage before it -- demosaic, colour correction, denoise, sharpen -- works on
linear radiance; tone mapping is what turns that radiance into display code
values. The standard curve is the sRGB opto-electronic transfer function
(OETF)::

    OETF(x) = 12.92 x                          for x <= 0.0031308
             1.055 x**(1/2.4) - 0.055          otherwise

with its exact inverse, the electro-optical transfer function (EOTF), using
the joint at 0.04045. The linear toe keeps the curve invertible and
numerically well behaved near black, where a pure power curve has an infinite
derivative; the power segment approximates the eye's response to luminance. A
plain ``x**(1/gamma)`` override is provided for comparisons and legacy
workflows (gamma 1.8-2.4), and produces visibly darker shadows than sRGB at
the same nominal exponent because it has no linear toe.

Exposure compensation is a gain of ``2**ev`` applied in linear light *before*
the curve, and the target-space matrix (sRGB is the identity, Display P3 uses
``SRGB_TO_P3``) is also applied before the curve. Encoding first and
matrixing second would mix three different non-linear transfer curves and
shift hues. After the matrix the stage clamps to ``[0, 1]``; that clamp is the
pipeline's final clip and is what turns highlight overshoot from earlier
stages into white and negative excursions into black. The transfer functions
themselves are pure pointwise maps: they are documented for the display
domain ``[0, 1]`` but accept any finite real input without clamping.

The tests pin the standard sRGB values and inverse, the joint between the two
branches, agreement with colour-science, the P3 conversion, exposure
compensation in linear light, the output range and clipping, gamma overrides,
the disabled identity, mono input, validation, dtype/contiguity, input
immutability, and determinism. Failure artefacts guarded here: encoding
before matrixing (hue shift), applying exposure to encoded data, a curve
without the linear toe, unclipped highlights or negative output, and silently
accepting malformed images or mutated configs.
"""

from __future__ import annotations

import colour
import numpy as np
import pytest
from colour.models import RGB_COLOURSPACES
from numpy.typing import NDArray

from camsim.config.schema import ToneConfig as SchemaToneConfig
from camsim.isp.colorspace import SRGB_TO_P3, linear_srgb_to_p3
from camsim.isp.tone import ToneConfig, apply_tone, linear_to_p3, linear_to_srgb, srgb_to_linear


def test_srgb_oetf_known_values():
    assert linear_to_srgb(np.array([0.0]))[0] == pytest.approx(0.0)
    assert linear_to_srgb(np.array([1.0]))[0] == pytest.approx(1.0)
    assert linear_to_srgb(np.array([0.0031308]))[0] == pytest.approx(0.04045, abs=1e-4)


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-7)


def test_tone_matches_colour_science():
    import colour

    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(
        linear_to_srgb(x), colour.cctf_encoding(x, function="sRGB"), atol=1e-7
    )


def _derived_space(name: str) -> colour.RGB_Colourspace:
    """Return a colour-science space whose matrices are derived from chromaticities."""
    space = RGB_COLOURSPACES[name].copy()
    space.use_derived_matrix_RGB_to_XYZ = True
    space.use_derived_matrix_XYZ_to_RGB = True
    return space


def _rgb_image(seed: int, shape: tuple[int, int] = (8, 8)) -> NDArray[np.float32]:
    """Return a reproducible linear RGB image spanning in- and out-of-gamut values."""
    return np.random.default_rng(seed).uniform(-0.25, 1.25, (*shape, 3)).astype(np.float32)


def test_tone_config_is_the_schema_model() -> None:
    """The re-exported config is the schema model, not a copy."""
    assert ToneConfig is SchemaToneConfig


def test_srgb_eotf_matches_colour_science() -> None:
    """The inverse curve matches colour's exact sRGB decoding."""
    x = np.linspace(-0.25, 1.5, 141)
    np.testing.assert_allclose(
        srgb_to_linear(x), colour.cctf_decoding(x, function="sRGB"), atol=1e-7
    )


def test_srgb_oetf_is_monotonic() -> None:
    """The curve is strictly increasing, including through the joint."""
    x = np.linspace(-0.25, 1.5, 1001)
    assert np.all(np.diff(linear_to_srgb(x)) > 0.0)


def test_srgb_transfer_functions_meet_at_the_joint() -> None:
    """The linear and power branches agree at the sRGB break points."""
    linear_joint = 0.0031308
    below = np.nextafter(linear_joint, 0.0)
    above = np.nextafter(linear_joint, 1.0)
    np.testing.assert_allclose(
        linear_to_srgb(np.array([below])),
        linear_to_srgb(np.array([above])),
        rtol=0.0,
        atol=1e-6,
    )
    values = np.array([0.0, below, linear_joint, above, 1.0])
    np.testing.assert_allclose(srgb_to_linear(linear_to_srgb(values)), values, atol=1e-7)
    np.testing.assert_allclose(
        srgb_to_linear(np.array([0.04045])), np.array([linear_joint]), rtol=0.0, atol=1e-7
    )
    encoded_below = np.nextafter(0.04045, 0.0)
    encoded_above = np.nextafter(0.04045, 1.0)
    np.testing.assert_allclose(
        srgb_to_linear(np.array([encoded_below])),
        srgb_to_linear(np.array([encoded_above])),
        rtol=0.0,
        atol=1e-7,
    )


def test_transfer_functions_are_pointwise_and_unclamped() -> None:
    """Any finite real value maps pointwise; negatives and >1 are not clipped."""
    values = np.random.default_rng(1).uniform(-0.5, 1.5, (3, 4, 3))
    pointwise = np.array(
        [linear_to_srgb(np.array([value]))[0] for value in values.ravel()]
    ).reshape(values.shape)
    np.testing.assert_allclose(linear_to_srgb(values), pointwise, rtol=0.0, atol=1e-7)

    np.testing.assert_allclose(linear_to_srgb(np.array([-0.5])), [-6.46], rtol=0.0, atol=1e-6)
    assert linear_to_srgb(np.array([2.0]))[0] > 1.0
    np.testing.assert_allclose(
        srgb_to_linear(np.array([-0.1])), [-0.1 / 12.92], rtol=0.0, atol=1e-7
    )
    assert srgb_to_linear(np.array([1.5]))[0] > 1.0


@pytest.mark.parametrize("function", [linear_to_srgb, srgb_to_linear])
@pytest.mark.parametrize(
    "bad", [np.array([np.nan]), np.array([np.inf]), np.array([True]), np.array([1 + 2j])]
)
def test_transfer_functions_reject_invalid_values(function, bad: NDArray) -> None:
    """A transfer function has no meaning for NaN, infinity, bool, or complex."""
    with pytest.raises(ValueError):
        function(bad)


@pytest.mark.parametrize("function", [linear_to_srgb, srgb_to_linear])
def test_transfer_functions_reject_float32_overflow(function) -> None:
    """A finite float64 value whose image cannot fit float32 raises."""
    with pytest.raises(ValueError, match="float32"):
        function(np.array([1e300]))


@pytest.mark.parametrize("function", [linear_to_srgb, srgb_to_linear])
def test_transfer_functions_return_new_float32_arrays(function) -> None:
    """Outputs are new C-contiguous float32 arrays of the input shape."""
    values = np.linspace(-0.25, 1.25, 32)
    before = values.copy()
    output = function(values)
    assert output.dtype == np.float32
    assert output.shape == values.shape
    assert output.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(output, values)
    np.testing.assert_array_equal(values, before)


def test_linear_to_p3_encodes_display_referred_p3() -> None:
    """P3 conversion happens in linear light; the OETF is applied last."""
    linear = np.random.default_rng(2).uniform(0.0, 1.0, (16, 3))
    expected = linear_to_srgb(linear @ SRGB_TO_P3.T)
    np.testing.assert_allclose(linear_to_p3(linear), expected, rtol=0.0, atol=1e-6)


def test_linear_to_p3_matches_colour_science() -> None:
    """Display-encoded P3 matches colour's linear conversion plus sRGB OETF."""
    srgb = _derived_space("sRGB")
    p3 = _derived_space("Display P3")
    linear = np.random.default_rng(3).uniform(0.0, 1.0, (256, 3))
    reference = colour.cctf_encoding(
        colour.RGB_to_RGB(linear, srgb, p3, chromatic_adaptation_transform=None),
        function="sRGB",
    )
    np.testing.assert_allclose(linear_to_p3(linear), reference, rtol=0.0, atol=1e-6)

    # colour's stored sRGB matrix is the rounded official table (~1e-4 away
    # from the chromaticity-derived matrix), which is far below one code value.
    official = colour.cctf_encoding(
        colour.RGB_to_RGB(
            linear,
            RGB_COLOURSPACES["sRGB"],
            RGB_COLOURSPACES["Display P3"],
            chromatic_adaptation_transform=None,
        ),
        function="sRGB",
    )
    np.testing.assert_allclose(linear_to_p3(linear), official, rtol=0.0, atol=5e-4)


def test_linear_to_p3_matches_the_linear_conversion_then_oetf() -> None:
    """``linear_to_p3`` is exactly the pure-linear map followed by the OETF."""
    linear = np.random.default_rng(4).uniform(-0.25, 1.25, (4, 4, 3))
    np.testing.assert_allclose(
        linear_to_p3(linear),
        linear_to_srgb(linear_srgb_to_p3(linear)),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.zeros((2, 2, 4)), np.array(0.5)])
def test_linear_to_p3_rejects_wrong_shapes(bad: NDArray[np.float64]) -> None:
    """P3 conversion needs a trailing axis of 3."""
    with pytest.raises(ValueError, match="trailing axis of 3"):
        linear_to_p3(bad)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_linear_to_p3_rejects_non_finite_values(value: float) -> None:
    """Non-finite input cannot be a linear intensity."""
    linear = np.zeros((1, 1, 3))
    linear[0, 0, 2] = value
    with pytest.raises(ValueError, match="finite"):
        linear_to_p3(linear)


def test_exposure_compensation() -> None:
    """Exposure is a 2**ev gain in linear light, applied before the curve."""
    image = np.array([[0.05, 0.1], [0.2, 0.35]])
    boosted = apply_tone(image, ToneConfig(exposure_compensation_ev=1.0))
    np.testing.assert_allclose(boosted, linear_to_srgb(np.clip(2.0 * image, 0.0, 1.0)), atol=1e-6)

    halved = apply_tone(image, ToneConfig(exposure_compensation_ev=-1.0))
    np.testing.assert_allclose(halved, linear_to_srgb(image / 2.0), atol=1e-6)

    neutral = apply_tone(image, ToneConfig())
    np.testing.assert_allclose(neutral, linear_to_srgb(image), atol=1e-6)


def test_output_range() -> None:
    """Every output is float32 in [0, 1], shape-preserving, with clipped tails."""
    image = np.random.default_rng(5).uniform(-0.5, 2.0, (16, 16, 3)).astype(np.float32)
    configs = [
        ToneConfig(),
        ToneConfig(color_space="DisplayP3"),
        ToneConfig(gamma=2.2),
        ToneConfig(exposure_compensation_ev=1.5),
        ToneConfig(exposure_compensation_ev=-2.0),
    ]
    for config in configs:
        output = apply_tone(image, config)
        assert output.dtype == np.float32
        assert output.shape == image.shape
        assert output.flags["C_CONTIGUOUS"]
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    clipped = apply_tone(np.array([[[2.0, -1.0, 0.5]]], np.float32), ToneConfig())
    assert clipped[0, 0, 0] == 1.0
    assert clipped[0, 0, 1] == 0.0
    assert clipped[0, 0, 2] == pytest.approx(float(linear_to_srgb(np.array([0.5]))[0]))


def test_tone_is_last_nonlinear_stage() -> None:
    """The curve is non-linear; exposure and colour matrixing are not."""
    mid_grey = 0.5
    encoded = float(linear_to_srgb(np.array([mid_grey]))[0])
    assert encoded == pytest.approx(0.7353569830524495, rel=1e-6)
    assert encoded > mid_grey

    # Exposure compensation multiplies in linear light before the curve, so a
    # +1 EV frame of 0.25 equals the curve at 0.5 -- not twice the curve at 0.25.
    boosted = apply_tone(np.array([[0.25]], np.float32), ToneConfig(exposure_compensation_ev=1.0))
    np.testing.assert_allclose(boosted, linear_to_srgb(np.array([[0.5]])), rtol=0.0, atol=1e-6)
    encoded_quarter = float(linear_to_srgb(np.array([0.25]))[0])
    assert float(boosted[0, 0]) != pytest.approx(2.0 * encoded_quarter, rel=1e-3)

    # The colour-space matrix is linear in linear light: gains commute with it.
    rgb = np.array([0.2, 0.4, 0.6])
    np.testing.assert_allclose(
        linear_srgb_to_p3(2.0 * rgb),
        2.0 * linear_srgb_to_p3(rgb).astype(np.float64),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        linear_srgb_to_p3(rgb + np.array([0.1, 0.1, 0.1])),
        linear_srgb_to_p3(rgb) + np.array([0.1, 0.1, 0.1]),
        rtol=0.0,
        atol=1e-6,
    )

    # Encoding before matrixing is a different, wrong operation (hue shift).
    encoded_before = linear_to_srgb(rgb).astype(np.float64) @ SRGB_TO_P3.T
    correct = linear_to_p3(rgb).astype(np.float64)
    assert np.max(np.abs(encoded_before - correct)) > 1e-3


def test_gamma_override() -> None:
    """A supplied gamma replaces the sRGB OETF with a plain power curve."""
    image = np.random.default_rng(6).uniform(0.0, 1.0, (6, 6, 3)).astype(np.float32)
    output = apply_tone(image, ToneConfig(gamma=2.2))
    np.testing.assert_allclose(output, image.astype(np.float64) ** (1.0 / 2.2), atol=1e-6)

    identity = apply_tone(image, ToneConfig(gamma=1.0))
    np.testing.assert_allclose(identity, image, rtol=0.0, atol=1e-7)

    p3 = apply_tone(image, ToneConfig(gamma=2.2, color_space="DisplayP3"))
    expected = np.clip(image.astype(np.float64) @ SRGB_TO_P3.T, 0.0, 1.0) ** (1.0 / 2.2)
    np.testing.assert_allclose(p3, expected, rtol=0.0, atol=1e-6)

    # The gamma curve is still monotonic and range-limited after clipping.
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_disabled_is_an_exact_identity() -> None:
    """``enabled=False`` returns an exact copy: no gain, matrix, clamp, or curve."""
    image = np.array([[[-0.5, 0.0, 1.5], [0.25, -2.0, 3.0]]], np.float32)
    config = ToneConfig(
        enabled=False,
        color_space="DisplayP3",
        exposure_compensation_ev=3.0,
        gamma=2.2,
    )
    output = apply_tone(image, config)
    np.testing.assert_array_equal(output, image)
    assert output.dtype == np.float32
    assert output.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(output, image)


def test_mono_output_stays_two_dimensional() -> None:
    """Mono input skips the colour matrix and keeps its single channel."""
    mono = np.random.default_rng(7).uniform(-0.25, 1.25, (8, 8)).astype(np.float32)
    expected = linear_to_srgb(np.clip(mono.astype(np.float64), 0.0, 1.0))
    for color_space in ("sRGB", "DisplayP3"):
        output = apply_tone(mono, ToneConfig(color_space=color_space))
        assert output.shape == (8, 8)
        assert output.dtype == np.float32
        assert output.flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(output, expected, rtol=0.0, atol=1e-6)

    boosted = apply_tone(mono, ToneConfig(exposure_compensation_ev=1.0))
    np.testing.assert_allclose(
        boosted, linear_to_srgb(np.clip(2.0 * mono.astype(np.float64), 0.0, 1.0)), atol=1e-6
    )


def test_display_p3_pipeline_matches_colour_science() -> None:
    """Exposure, P3 matrix, and OETF run in that order and match colour."""
    image = np.random.default_rng(8).uniform(0.0, 0.8, (8, 8, 3)).astype(np.float32)
    exposure_ev = 0.5
    output = apply_tone(
        image,
        ToneConfig(color_space="DisplayP3", exposure_compensation_ev=exposure_ev),
    )
    linear = image.astype(np.float64) * 2.0**exposure_ev
    reference = colour.cctf_encoding(
        np.clip(
            colour.RGB_to_RGB(
                linear.reshape(-1, 3),
                _derived_space("sRGB"),
                _derived_space("Display P3"),
                chromatic_adaptation_transform=None,
            ),
            0.0,
            1.0,
        ),
        function="sRGB",
    ).reshape(image.shape)
    np.testing.assert_allclose(output, reference, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(3),
        np.zeros((2, 2, 4)),
        np.zeros((2, 2, 1)),
        np.zeros((2, 2, 3, 1)),
        np.zeros((0, 0)),
        np.zeros((0, 0, 3)),
    ],
)
def test_apply_tone_rejects_wrong_shapes(bad: NDArray[np.float64]) -> None:
    """Only a non-empty (H, W) or (H, W, 3) image is an image."""
    with pytest.raises(ValueError):
        apply_tone(bad, ToneConfig())


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_apply_tone_rejects_non_finite_images(value: float) -> None:
    """A non-finite pixel cannot be tone mapped."""
    image = np.zeros((2, 2, 3), np.float32)
    image[1, 1, 0] = value
    with pytest.raises(ValueError, match="finite"):
        apply_tone(image, ToneConfig())


@pytest.mark.parametrize("dtype", [np.bool_, np.complex128])
def test_apply_tone_rejects_bool_and_complex_images(dtype: np.dtype) -> None:
    """Booleans and complex values are not linear intensities."""
    with pytest.raises(ValueError, match="real numeric"):
        apply_tone(np.zeros((2, 2, 3), dtype=dtype), ToneConfig())


@pytest.mark.parametrize("config", [None, {}, object(), SchemaToneConfig().model_dump()])
def test_apply_tone_rejects_non_config_objects(config: object) -> None:
    """The stage requires a validated ToneConfig instance, not a mapping."""
    with pytest.raises(ValueError, match="ToneConfig"):
        apply_tone(np.zeros((2, 2, 3), np.float32), config)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "update",
    [
        {"enabled": "yes"},
        {"enabled": 1},
        {"color_space": "AdobeRGB"},
        {"color_space": None},
        {"exposure_compensation_ev": float("nan")},
        {"exposure_compensation_ev": float("inf")},
        {"exposure_compensation_ev": "1.0"},
        {"exposure_compensation_ev": True},
        {"gamma": 0.0},
        {"gamma": -1.0},
        {"gamma": float("nan")},
        {"gamma": True},
    ],
)
def test_apply_tone_rejects_mutated_config_values(update: dict[str, object]) -> None:
    """A config mutated after construction bypasses pydantic and is re-checked."""
    config = ToneConfig().model_copy(update=update)
    with pytest.raises(ValueError):
        apply_tone(np.zeros((2, 2, 3), np.float32), config)


def test_apply_tone_rejects_exposure_overflow() -> None:
    """An exposure gain that overflows to infinity is a clear error, not NaN pixels."""
    with pytest.raises(ValueError, match="exposure"):
        apply_tone(
            np.zeros((2, 2, 3), np.float32),
            ToneConfig(exposure_compensation_ev=2000.0),
        )


def test_apply_tone_returns_new_float32_arrays() -> None:
    """Outputs are new C-contiguous float32 arrays and inputs are untouched."""
    image = np.random.default_rng(9).uniform(-0.5, 1.5, (5, 7, 3))
    before = image.copy()
    output = apply_tone(image, ToneConfig())
    assert output.dtype == np.float32
    assert output.shape == image.shape
    assert output.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(output, image)
    np.testing.assert_array_equal(image, before)


def test_apply_tone_is_deterministic() -> None:
    """Identical input and config give bit-identical output."""
    image = _rgb_image(10)
    config = ToneConfig(color_space="DisplayP3", exposure_compensation_ev=0.5)
    np.testing.assert_array_equal(apply_tone(image, config), apply_tone(image, config))
