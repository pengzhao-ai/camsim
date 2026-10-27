"""Shared pytest fixtures for the camsim test suite.

Configuration fixtures load the shipped presets through the public loader
functions, so tests exercise the same validation path users hit. Fixtures that
depend on hardware models (``sensor``, ``raw_frame``, ``camera``, scenes) are
added here by later tasks; keeping them out of this file until they exist keeps
the suite runnable at every phase boundary.

Preset paths are derived from this file's location rather than the current
working directory, so the suite behaves identically when pytest is invoked from
the repository root or from a subdirectory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from camsim.camera import Camera
from camsim.config.io import load_isp, load_sensor
from camsim.config.schema import ISPConfig, LensConfig, SensorConfig
from camsim.core.frames import CFAPattern, RawFrame
from camsim.core.spectral import SpectralIrradiance
from camsim.isp.pipeline import ISP
from camsim.scene.targets import SceneResult, render
from camsim.sensor import Sensor

REPO_ROOT = Path(__file__).resolve().parents[1]
SENSOR_PRESET_PATH = (
    REPO_ROOT / "src" / "camsim" / "config" / "presets" / "sensors" / "flagship_phone_50mp.json"
)
ISP_PRESET_PATH = REPO_ROOT / "src" / "camsim" / "config" / "presets" / "isp" / "default.json"

# The preset's spectral grid: 360-830 nm in 5 nm steps.
_WAVELENGTHS_NM = np.arange(360.0, 831.0, 5.0)


def flat_irradiance(level: float = 2e-5, shape: tuple[int, int] = (32, 32)) -> SpectralIrradiance:
    """Return a low-light flat spectral cube on the preset grid.

    ``2e-5 W/m^2/nm`` is a dim laboratory scene that produces a few tens of
    electrons per pixel in the flagship part, so fixture frames exercise the
    pipeline without reaching full well. The level was raised from 1e-5 when
    the CFA was re-authored to conventional Bayer bands; the narrower passbands
    collect less light, and the old level left the darkest pixels at the
    zero-charge clamp.
    """
    data = np.full((*shape, _WAVELENGTHS_NM.size), level, dtype=np.float64)
    return SpectralIrradiance.from_array(
        data,
        _WAVELENGTHS_NM,
        "W/m^2/nm",
        pixel_pitch_um=1.12,
    )


@pytest.fixture
def sensor_config() -> SensorConfig:
    """The flagship sensor preset, validated by the production loader."""
    return load_sensor(SENSOR_PRESET_PATH)


@pytest.fixture
def isp_config() -> ISPConfig:
    """The default ISP preset, validated by the production loader."""
    return load_isp(ISP_PRESET_PATH)


@pytest.fixture
def mono_sensor_config(sensor_config: SensorConfig) -> SensorConfig:
    """The flagship sensor preset reconfigured as a CFA-free mono sensor.

    ``model_copy(update=...)`` is intentional: only the CFA changes, so the
    spectral projection, gains, and noise parameters stay identical to the
    colour preset and mono-versus-Bayer comparisons differ in one variable.
    """
    return sensor_config.model_copy(update={"cfa": CFAPattern.MONO})


@pytest.fixture
def sensor(sensor_config: SensorConfig) -> Sensor:
    """A sensor built from the flagship preset with a fixed frame-stream seed."""
    return Sensor(sensor_config, seed=1)


@pytest.fixture
def raw_frame(sensor: Sensor) -> RawFrame:
    """A low-signal 32x32 LCG RAW frame with full metadata and masks."""
    return sensor.expose(flat_irradiance(), 0.01, mode="LCG", seed=7)


@pytest.fixture
def mono_sensor(mono_sensor_config: SensorConfig) -> Sensor:
    """A CFA-free sensor sharing the flagship preset's non-CFA parameters."""
    return Sensor(mono_sensor_config, seed=1)


@pytest.fixture
def mono_raw_frame(mono_sensor: Sensor) -> RawFrame:
    """A low-signal 32x32 mono RAW frame (single-channel mosaic)."""
    return mono_sensor.expose(flat_irradiance(), 0.01, mode="LCG", seed=7)


# ---------------------------------------------------------------------------
# Camera and rendered scenes
# ---------------------------------------------------------------------------


def _reference_lens() -> LensConfig:
    """The design's reference phone lens: f/1.8, 6.4 mm, 11.46 mm, T = 0.9."""
    return LensConfig(
        f_number=1.8,
        focal_length_mm=6.4,
        sensor_diagonal_mm=11.46,
        transmittance=0.9,
    )


def _flat_scene(lux: float) -> SceneResult:
    """Render a D65 flat patch on the flagship sensor's 1.12 um pixel grid.

    ``oversampling=2`` keeps the render fast while resolving the patch edge
    cleanly; a flat target has no edge to resolve, but using the same render
    path as the chart scenes keeps the photometry representative.
    """
    return render(
        "flat_patch",
        illuminant="D65",
        lux=lux,
        shape=(64, 64),
        oversampling=2,
    )


@pytest.fixture
def camera(sensor: Sensor, isp_config: ISPConfig) -> Camera:
    """A camera with the flagship sensor, default ISP, and reference lens."""
    return Camera(_reference_lens(), sensor, ISP(isp_config))


@pytest.fixture
def scene() -> SceneResult:
    """A 2000 lux D65 flat patch that already sits near the 18 % target."""
    return _flat_scene(2000.0)


@pytest.fixture
def bright_scene() -> SceneResult:
    """A 50000 lux flat patch that saturates the default exposure."""
    return _flat_scene(50_000.0)


@pytest.fixture
def dark_scene() -> SceneResult:
    """A 20 lux flat patch that needs a longer exposure and some gain."""
    return _flat_scene(20.0)


@pytest.fixture
def very_dark_scene() -> SceneResult:
    """A 0.01 lux flat patch that needs HCG, every stop, and digital gain."""
    return _flat_scene(0.01)
