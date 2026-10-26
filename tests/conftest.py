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

import pytest

from camsim.config.io import load_isp, load_sensor
from camsim.config.schema import ISPConfig, SensorConfig
from camsim.core.frames import CFAPattern

REPO_ROOT = Path(__file__).resolve().parents[1]
SENSOR_PRESET_PATH = (
    REPO_ROOT / "src" / "camsim" / "config" / "presets" / "sensors" / "flagship_phone_50mp.json"
)
ISP_PRESET_PATH = REPO_ROOT / "src" / "camsim" / "config" / "presets" / "isp" / "default.json"


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
