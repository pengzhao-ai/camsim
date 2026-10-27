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

from camsim.config.io import load_isp, load_sensor
from camsim.config.schema import ISPConfig, SensorConfig
from camsim.core.frames import CFAPattern, RawFrame
from camsim.core.spectral import SpectralIrradiance
from camsim.sensor import Sensor

REPO_ROOT = Path(__file__).resolve().parents[1]
SENSOR_PRESET_PATH = (
    REPO_ROOT / "src" / "camsim" / "config" / "presets" / "sensors" / "flagship_phone_50mp.json"
)
ISP_PRESET_PATH = REPO_ROOT / "src" / "camsim" / "config" / "presets" / "isp" / "default.json"

# The preset's spectral grid: 360-830 nm in 5 nm steps.
_WAVELENGTHS_NM = np.arange(360.0, 831.0, 5.0)


def flat_irradiance(level: float = 1e-5, shape: tuple[int, int] = (32, 32)) -> SpectralIrradiance:
    """Return a low-light flat spectral cube on the preset grid.

    ``1e-5 W/m^2/nm`` is a dim laboratory scene that produces a few tens of
    electrons per pixel in the flagship part, so fixture frames exercise the
    pipeline without reaching full well.
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
