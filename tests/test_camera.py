"""Camera orchestration, fixed-exposure capture, and opt-in auto-exposure.

The camera is the only object that owns all four layers at once: the lens
geometry travels into the rendered metadata, the sensor turns scene
irradiance into a RAW frame, and the ISP turns that frame into a display image.
These tests pin the two exposure policies the camera offers. The default is a
fixed exposure from the sensor preset, so every experiment that does not ask
for auto-exposure stays deterministic and comparable. Auto-exposure is opt-in:
it probes the real sensor (noise included), meters the black-corrected linear
signal with a centre-weighted statistic, and reports every decision in an
:class:`~camsim.camera.ExposureResult`.

The tests cover convergence across a bright, a normal, and a dark flat patch,
the documented physical limits (exposure window, analog-gain window, digital
gain), HCG selection in low light, the fixed-versus-auto metadata contract,
input validation, determinism, and the rule that capture never mutates the
scene or the ISP result it was derived from.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import camsim
from camsim.camera import Camera, ExposureResult, ExposureSettings
from camsim.camera.camera import _correction_factor, _is_clipped
from camsim.config.schema import (
    ADCConfig,
    AnalogGainConfig,
    ExposureConfig,
    LensConfig,
)
from camsim.core.frames import RenderedImage
from camsim.isp.pipeline import ISP
from camsim.scene.targets import render
from camsim.sensor import Sensor

# ---------------------------------------------------------------------------
# Brief: the three verbatim contracts
# ---------------------------------------------------------------------------


def test_auto_exposure_converges(camera, bright_scene, dark_scene):
    bright = camera.auto_expose(bright_scene)
    dark = camera.auto_expose(dark_scene)
    assert bright.converged and dark.converged
    assert dark.exposure_s > bright.exposure_s or dark.analog_gain_db > bright.analog_gain_db


def test_auto_exposure_respects_limits(camera, very_dark_scene):
    res = camera.auto_expose(very_dark_scene)
    cfg = camera.sensor.config
    assert res.exposure_s <= cfg.exposure.max_s
    assert res.analog_gain_db <= cfg.analog_gain.max_db
    assert res.mode in {"LCG", "HCG"}


def test_fixed_exposure_protocol_is_default(camera, scene):
    out = camera.capture(
        scene, exposure=ExposureSettings(exposure_s=0.01, analog_gain_db=0.0, mode="LCG")
    )
    assert out.metadata["exposure_policy"] == "fixed"


# ---------------------------------------------------------------------------
# Auto-exposure results
# ---------------------------------------------------------------------------


def test_auto_exposure_records_metered_value(camera, scene):
    """The reported metered value is the final measured fraction of full scale."""
    res = camera.auto_expose(scene)
    assert res.converged
    assert math.isfinite(res.metered_value)
    assert 0.0 <= res.metered_value <= 1.0
    assert abs(res.metered_value - 0.18) <= 0.02


def test_auto_exposure_iteration_bound(camera, scene):
    """Iterations count probes and never exceed the requested budget."""
    converged = camera.auto_expose(scene)
    assert 1 <= converged.iterations <= 12

    # A tolerance no real probe can meet forces the full budget.
    limited = camera.auto_expose(scene, max_iterations=1, tolerance=1e-6)
    assert limited.iterations == 1
    assert limited.converged is False
    # The returned settings are the ones that produced the reported probe.
    assert limited.exposure_s == camera.sensor.config.exposure.default_s
    assert limited.analog_gain_db == camera.sensor.config.analog_gain.default_db
    assert limited.digital_gain == 1.0


def test_mode_auto_selects_hcg_in_low_light(camera, very_dark_scene):
    res = camera.auto_expose(very_dark_scene)
    assert res.mode == "HCG"


def test_mode_lock_prevents_readout_mode_flapping(camera):
    """Only the first probe may request ``auto``; the rest reuse its mode.

    At 100 lux the first probe resolves HCG, while the converged settings have
    a mean charge above the 200 e- auto threshold. Re-resolving the mode at the
    target would therefore flip to LCG and change conversion gain mid-ramp.
    """
    flapping_scene = render(
        "flat_patch",
        illuminant="D65",
        lux=100.0,
        shape=(64, 64),
        oversampling=2,
    )
    with patch.object(camera.sensor, "expose", wraps=camera.sensor.expose) as spy:
        res = camera.auto_expose(flapping_scene)

    requested_modes = [recorded.kwargs["mode"] for recorded in spy.call_args_list]
    assert len(requested_modes) >= 2
    assert requested_modes[0] == "auto"
    assert all(mode == "HCG" for mode in requested_modes[1:])
    assert res.converged
    assert res.mode == "HCG"

    # Prove the rationale: an unlocked re-resolution at the final settings
    # would select LCG, because the converged mean charge exceeds the HCG
    # threshold.
    unlocked = camera.sensor.expose(
        flapping_scene.irradiance,
        res.exposure_s,
        mode="auto",
        analog_gain_db=res.analog_gain_db,
        digital_gain=res.digital_gain,
        seed=0,
    )
    assert unlocked.mode == "LCG"


def test_auto_exposure_all_scenes_reach_target(camera, scene, bright_scene, dark_scene):
    for target_scene in (scene, bright_scene, dark_scene):
        res = camera.auto_expose(target_scene)
        assert res.converged
        assert abs(res.metered_value - 0.18) <= 0.02


def test_auto_exposure_honours_a_different_target(camera, scene):
    res = camera.auto_expose(scene, target_fraction=0.35)
    assert res.converged
    assert abs(res.metered_value - 0.35) <= 0.02


def test_auto_exposure_drives_dark_scene_to_the_limits(camera, very_dark_scene):
    """A scene far below the fixed baseline needs every physical stop."""
    res = camera.auto_expose(very_dark_scene)
    cfg = camera.sensor.config
    assert res.exposure_s == pytest.approx(cfg.exposure.max_s)
    assert res.analog_gain_db == pytest.approx(cfg.analog_gain.max_db)
    # Digital gain is the only unbounded knob left, so it carries the residual.
    assert res.digital_gain > 1.0
    # The first probe resolves HCG and the policy locks it. At this light level
    # the auto threshold is never crossed, so the lock's actual effect is
    # covered by test_mode_lock_prevents_readout_mode_flapping.
    assert res.mode == "HCG"


def test_digital_gain_below_unity_is_last_resort(camera):
    """Digital gain only darkens when exposure and analog gain cannot."""
    narrow = camera.sensor.config.model_copy(
        update={
            "exposure": ExposureConfig(min_s=0.01, max_s=0.1, default_s=0.01),
            "analog_gain": AnalogGainConfig(min_db=0.0, max_db=24.0, step_db=1.0, default_db=0.0),
        }
    )
    limited_camera = Camera(camera.lens, Sensor(narrow, seed=1), camera.isp)
    # Bright enough to need an 0.5x correction at the minimum exposure, but
    # below the 6000 e- well so digital attenuation is a legitimate remedy:
    # a full-well-saturated scene can never be unclipped digitally.
    overexposed = render(
        "flat_patch",
        illuminant="D65",
        lux=4_000.0,
        shape=(64, 64),
        oversampling=2,
    )
    res = limited_camera.auto_expose(overexposed)
    assert res.converged
    assert res.exposure_s == pytest.approx(narrow.exposure.min_s)
    assert res.analog_gain_db == pytest.approx(narrow.analog_gain.min_db)
    assert 0.0 < res.digital_gain < 1.0


def test_clipped_probe_caps_the_correction_factor():
    """The 0.5 cap binds when the damped factor is still above it."""
    assert _correction_factor(0.3, 0.18, clipped=False) > 0.5
    assert _correction_factor(0.3, 0.18, clipped=True) == 0.5
    assert _correction_factor(0.03, 0.18, clipped=True) == 0.5


def test_clipping_detects_full_well_saturation(camera):
    """The well can saturate far below the ADC ceiling and still count."""
    hot_scene = render(
        "flat_patch",
        illuminant="D65",
        lux=500_000.0,
        shape=(64, 64),
        oversampling=2,
    )
    raw = camera.sensor.expose(
        hot_scene.irradiance,
        0.01,
        mode="LCG",
        analog_gain_db=0.0,
        digital_gain=1.0,
        seed=0,
    )
    largest_code = (1 << raw.bit_depth) - 1
    assert raw.data.max() < largest_code
    assert np.count_nonzero(raw.data >= largest_code) == 0
    assert _is_clipped(raw, camera.sensor.config) is True

    # The full-well-aware guard must not trap the search: AE still converges.
    res = camera.auto_expose(hot_scene)
    assert res.converged
    assert abs(res.metered_value - 0.18) <= 0.02


def test_auto_exposure_handles_a_clipped_probe(camera, bright_scene):
    """ADC saturation triggers the reduced-step guard and still converges."""
    clipped = camera.sensor.config.model_copy(
        update={
            "adc": ADCConfig(
                bits=10,
                full_scale_uv=200_000.0,
                black_level_dn=64,
                digital_gain=1.0,
            )
        }
    )
    clipped_camera = Camera(camera.lens, Sensor(clipped, seed=1), camera.isp)
    default_frame = clipped_camera.sensor.expose(
        bright_scene.irradiance,
        0.01,
        mode="LCG",
        analog_gain_db=0.0,
        digital_gain=1.0,
        seed=0,
    )
    assert default_frame.data.max() == (1 << default_frame.bit_depth) - 1
    assert _is_clipped(default_frame, clipped_camera.sensor.config) is True

    res = clipped_camera.auto_expose(bright_scene)
    assert res.converged
    assert abs(res.metered_value - 0.18) <= 0.02
    # Pulling a clipped frame down must leave the default exposure behind.
    assert res.exposure_s < clipped.exposure.default_s


def test_auto_exposure_rejects_a_frame_without_a_white_span(camera, scene):
    no_span = camera.sensor.config.model_copy(
        update={
            "adc": ADCConfig(
                bits=10,
                full_scale_uv=800_000.0,
                black_level_dn=1023,
                digital_gain=1.0,
            )
        }
    )
    broken = Camera(camera.lens, Sensor(no_span, seed=1), camera.isp)
    with pytest.raises(ValueError, match="white span"):
        broken.auto_expose(scene)


def test_auto_exposure_handles_an_unmeasurable_probe(camera, scene, monkeypatch):
    """An all-pedestal probe is treated as underexposed, not a divide by zero."""
    black = camera.sensor.expose(scene.irradiance, 0.01, mode="LCG", seed=0)
    black = black.with_data(np.full(black.data.shape, int(black.black_level_dn), dtype=np.uint16))
    monkeypatch.setattr(camera.sensor, "expose", lambda *args, **kwargs: black)

    res = camera.auto_expose(scene, max_iterations=2)
    assert res.converged is False
    assert res.iterations == 2
    assert res.metered_value == 0.0
    # The first probe jumps by the documented zero-meter factor of 8 from the
    # preset default (10 ms), and the second probe confirms it is still dark.
    assert res.exposure_s == pytest.approx(8.0 * camera.sensor.config.exposure.default_s)


def test_auto_exposure_rejects_an_unknown_readout_mode(camera, scene, monkeypatch):
    frame = camera.sensor.expose(scene.irradiance, 0.01, mode="LCG", seed=0)
    malformed = replace(frame, mode="XYZ")
    monkeypatch.setattr(camera.sensor, "expose", lambda *args, **kwargs: malformed)
    with pytest.raises(ValueError, match="readout mode"):
        camera.auto_expose(scene, max_iterations=1)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def test_capture_returns_rendered_image(camera, scene):
    out = camera.capture(scene)
    assert isinstance(out, RenderedImage)
    assert out.data.shape == (64, 64, 3)
    assert out.data.dtype == np.float32
    assert np.all(out.data >= 0.0)
    assert np.all(out.data <= 1.0)


def test_capture_default_policy_metadata(camera, scene):
    out = camera.capture(scene)
    cfg = camera.sensor.config
    metadata = out.metadata
    assert metadata["exposure_policy"] == "fixed"
    assert metadata["exposure_s"] == cfg.exposure.default_s
    assert metadata["analog_gain_db"] == cfg.analog_gain.default_db
    assert metadata["digital_gain"] == 1.0
    assert metadata["camera_mode"] == "LCG"
    assert metadata["focal_length_mm"] == camera.lens.focal_length_mm
    assert metadata["f_number"] == camera.lens.f_number
    assert "metered_value" not in metadata
    assert metadata["isp_config_hash"] == camera.isp.config_hash


def test_capture_uses_auto_exposure_result(camera, dark_scene):
    res = camera.auto_expose(dark_scene)
    out = camera.capture(dark_scene, exposure=res)
    metadata = out.metadata
    assert metadata["exposure_policy"] == "auto"
    assert metadata["exposure_s"] == res.exposure_s
    assert metadata["analog_gain_db"] == res.analog_gain_db
    assert metadata["digital_gain"] == res.digital_gain
    assert metadata["camera_mode"] == res.mode
    assert metadata["metered_value"] == res.metered_value
    assert metadata["iterations"] == res.iterations
    assert metadata["converged"] is res.converged


def test_capture_honours_fixed_settings(camera, scene):
    short = camera.capture(scene, ExposureSettings(0.001, 0.0, "LCG"))
    long = camera.capture(scene, ExposureSettings(0.05, 0.0, "LCG"))
    high_gain = camera.capture(scene, ExposureSettings(0.001, 12.0, "HCG"))

    assert short.metadata["exposure_s"] == 0.001
    assert long.metadata["exposure_s"] == 0.05
    assert high_gain.metadata["analog_gain_db"] == 12.0
    assert high_gain.metadata["camera_mode"] == "HCG"
    assert float(np.mean(long.linear_rgb)) > float(np.mean(short.linear_rgb))
    assert float(np.mean(high_gain.linear_rgb)) > float(np.mean(short.linear_rgb))


def test_capture_rejects_out_of_range_exposure(camera, scene):
    settings = ExposureSettings(exposure_s=1.0, analog_gain_db=0.0, mode="LCG")
    with pytest.raises(ValueError, match="exposure"):
        camera.capture(scene, exposure=settings)


def test_capture_rejects_out_of_range_gain(camera, scene):
    settings = ExposureSettings(exposure_s=0.01, analog_gain_db=30.0, mode="LCG")
    with pytest.raises(ValueError, match="analog_gain_db"):
        camera.capture(scene, exposure=settings)


def test_capture_rejects_out_of_range_auto_result(camera, scene):
    result = ExposureResult(
        exposure_s=camera.sensor.config.exposure.max_s * 10.0,
        analog_gain_db=0.0,
        digital_gain=1.0,
        mode="LCG",
        converged=True,
        iterations=1,
        metered_value=0.18,
    )
    with pytest.raises(ValueError, match="exposure"):
        camera.capture(scene, exposure=result)


def test_capture_rejects_unknown_exposure_type(camera, scene):
    with pytest.raises(ValueError, match="ExposureSettings or an ExposureResult"):
        camera.capture(scene, exposure="auto")


def test_capture_and_auto_expose_reject_a_wrong_scene(camera, scene):
    with pytest.raises(ValueError, match="SceneResult"):
        camera.capture(scene.irradiance)
    with pytest.raises(ValueError, match="SceneResult"):
        camera.auto_expose(scene.irradiance)


# ---------------------------------------------------------------------------
# Construction and value-type validation
# ---------------------------------------------------------------------------


def test_camera_validates_constructor_arguments(camera, sensor, isp_config):
    with pytest.raises(ValueError, match="LensConfig"):
        Camera("lens", sensor, ISP(isp_config))
    with pytest.raises(ValueError, match="Sensor"):
        Camera(camera.lens, "sensor", ISP(isp_config))
    with pytest.raises(ValueError, match="ISP"):
        Camera(camera.lens, sensor, "isp")


def test_camera_exposes_its_layers(camera, sensor):
    assert camera.sensor is sensor
    assert isinstance(camera.lens, LensConfig)
    assert isinstance(camera.isp, ISP)
    assert camera.lens.f_number == 1.8
    assert camera.lens.focal_length_mm == 6.4


def test_exposure_settings_validation():
    valid = ExposureSettings(0.01, 0.0, "HCG")
    assert valid.exposure_s == 0.01
    assert valid.analog_gain_db == 0.0
    assert valid.mode == "HCG"

    with pytest.raises(ValueError, match="exposure_s"):
        ExposureSettings(0.0, 0.0, "LCG")
    with pytest.raises(ValueError, match="exposure_s"):
        ExposureSettings(float("nan"), 0.0, "LCG")
    with pytest.raises(ValueError, match="exposure_s"):
        ExposureSettings(True, 0.0, "LCG")
    with pytest.raises(ValueError, match="analog_gain_db"):
        ExposureSettings(0.01, -1.0, "LCG")
    with pytest.raises(ValueError, match="analog_gain_db"):
        ExposureSettings(0.01, float("inf"), "LCG")
    with pytest.raises(ValueError, match="mode"):
        ExposureSettings(0.01, 0.0, "auto")


def _exposure_result(**overrides: object) -> ExposureResult:
    """Build a valid result with selected fields overridden."""
    values: dict[str, object] = {
        "exposure_s": 0.01,
        "analog_gain_db": 0.0,
        "digital_gain": 1.0,
        "mode": "LCG",
        "converged": True,
        "iterations": 2,
        "metered_value": 0.18,
    }
    values.update(overrides)
    return ExposureResult(**values)  # type: ignore[arg-type]


def test_exposure_result_validation():
    assert _exposure_result(iterations=0).iterations == 0

    with pytest.raises(ValueError, match="exposure_s"):
        _exposure_result(exposure_s=-0.01)
    with pytest.raises(ValueError, match="exposure_s"):
        _exposure_result(exposure_s=float("inf"))
    with pytest.raises(ValueError, match="analog_gain_db"):
        _exposure_result(analog_gain_db=-0.5)
    with pytest.raises(ValueError, match="digital_gain"):
        _exposure_result(digital_gain=0.0)
    with pytest.raises(ValueError, match="digital_gain"):
        _exposure_result(digital_gain=float("nan"))
    with pytest.raises(ValueError, match="mode"):
        _exposure_result(mode="auto")
    with pytest.raises(ValueError, match="converged"):
        _exposure_result(converged=1)
    with pytest.raises(ValueError, match="iterations"):
        _exposure_result(iterations=-1)
    with pytest.raises(ValueError, match="iterations"):
        _exposure_result(iterations=1.5)
    with pytest.raises(ValueError, match="iterations"):
        _exposure_result(iterations=True)
    with pytest.raises(ValueError, match="metered_value"):
        _exposure_result(metered_value=-0.1)
    with pytest.raises(ValueError, match="metered_value"):
        _exposure_result(metered_value=float("nan"))


def test_auto_exposure_validates_target(camera, scene):
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="target_fraction"):
            camera.auto_expose(scene, target_fraction=bad)


def test_auto_exposure_validates_tolerance(camera, scene):
    for bad in (0.0, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="tolerance"):
            camera.auto_expose(scene, tolerance=bad)


def test_auto_exposure_validates_iterations(camera, scene):
    for bad in (0, -1, 2.5, float("inf"), True):
        with pytest.raises(ValueError, match="max_iterations"):
            camera.auto_expose(scene, max_iterations=bad)


# ---------------------------------------------------------------------------
# Determinism and immutability
# ---------------------------------------------------------------------------


def test_auto_exposure_is_deterministic(camera, dark_scene):
    first = camera.auto_expose(dark_scene)
    second = camera.auto_expose(dark_scene)
    assert first == second
    camera.sensor.reseed(999)
    assert camera.auto_expose(dark_scene) == first


def test_capture_is_deterministic_with_a_reseeded_sensor(camera, scene):
    camera.sensor.reseed(5)
    first = camera.capture(scene)
    camera.sensor.reseed(5)
    second = camera.capture(scene)
    assert np.array_equal(first.data, second.data)
    assert np.array_equal(first.linear_rgb, second.linear_rgb)


def test_auto_exposure_does_not_advance_the_capture_stream(camera, scene):
    camera.sensor.reseed(5)
    first = camera.capture(scene)
    camera.sensor.reseed(5)
    camera.auto_expose(scene)
    second = camera.capture(scene)
    assert np.array_equal(first.data, second.data)


def test_capture_does_not_mutate_scene_or_isp_result(camera, scene, monkeypatch):
    cube_before = scene.irradiance.data.copy()
    truth_before = dict(scene.ground_truth)

    captured: dict[str, object] = {}
    original = camera.isp.process

    def spy(raw, trace=False):
        image = original(raw, trace=trace)
        captured["raw_codes"] = raw.data.copy()
        captured["raw"] = raw
        captured["image"] = image
        captured["metadata"] = dict(image.metadata)
        captured["data"] = image.data.copy()
        return image

    monkeypatch.setattr(camera.isp, "process", spy)
    out = camera.capture(scene)

    assert out is not captured["image"]
    assert captured["image"].metadata == captured["metadata"]
    assert np.array_equal(captured["image"].data, captured["data"])
    assert np.array_equal(captured["raw"].data, captured["raw_codes"])
    assert np.array_equal(scene.irradiance.data, cube_before)
    assert scene.ground_truth == truth_before


# ---------------------------------------------------------------------------
# Public export contract
# ---------------------------------------------------------------------------


def test_camera_is_a_lazy_export():
    assert camsim.Camera is Camera
    assert "Camera" in camsim.__all__
    assert camsim._LAZY_EXPORTS["Camera"] == ("camsim.camera.camera", "Camera")
    assert "Camera" in dir(camsim)
    # The documentation builder cannot run the lazy ``__getattr__``; the
    # type-checking alias block is what gives mkdocstrings a real symbol.
    source = Path(camsim.__file__).read_text(encoding="utf-8")
    assert "from camsim.camera.camera import Camera as Camera" in source


def test_camera_export_is_not_imported_eagerly():
    code = "\n".join(
        [
            "import sys",
            "import camsim",
            "assert 'camsim.camera' not in sys.modules",
            "camera = camsim.Camera",
            "assert camera.__name__ == 'Camera'",
            "assert 'camsim.camera' in sys.modules",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_clipping_uses_analog_gain_for_the_well_and_digital_gain_for_the_code(camera):
    """A sub-unity digital gain must not hide an ADC/full-well saturation.

    At 6 dB analog gain the HCG full-well voltage is above the ADC full scale,
    so the converter saturates first. A digital gain of 0.5 maps that saturated
    code to roughly half scale. Computing the full-well threshold with the
    post-ADC digital gain folded into the voltage would place the threshold at
    the top code and miss the saturation; the correct model clips the ADC ratio
    first and applies digital gain afterwards.
    """
    hot_scene = render(
        "flat_patch",
        illuminant="D65",
        lux=500_000.0,
        shape=(64, 64),
        oversampling=2,
    )
    raw = camera.sensor.expose(
        hot_scene.irradiance,
        0.01,
        mode="HCG",
        analog_gain_db=6.0,
        digital_gain=0.5,
        seed=0,
    )
    largest_code = (1 << raw.bit_depth) - 1
    assert raw.data.max() < largest_code
    assert _is_clipped(raw, camera.sensor.config) is True
