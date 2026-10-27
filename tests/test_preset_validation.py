"""G2 preset gate: the shipped flagship sensor recovers its datasheet-style values.

The G2 gate closes Phase 2 by proving that a sensor configured with known
parameters can be measured back through the EMVA 1288-style characterization.
This module is the executable form of that gate for the shipped preset: it
loads the preset through the production loader, characterizes the low
conversion-gain mode with 32 frames per level, and checks the measured read
noise, DSNU, PRNU, saturation capacity, linearity, and dynamic range against the
design tolerances.

Thirty-two frames per level is deliberate. The LCG dark offset is only about
0.06 DN, so an eight-frame dark average leaves a seed-dependent DSNU spread
that can exceed the 10 % gate; 32 frames average the residual row-noise and
quantization coupling enough for a stable measurement. The separate eight-frame
robustness sweep in ``tests/test_characterization.py`` carries a documented
20 % DSNU waiver and is not the G2 evidence.

The measured values belong to the preset in this checkout. A change to the ADC,
full well, conversion gain, DSNU, PRNU, or spectral curves invalidates them and
requires re-running the characterization and the G2 report.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import camsim
from camsim.characterization import characterize_sensor
from camsim.config.io import load_sensor

REPO_ROOT = Path(__file__).resolve().parents[1]
SENSOR_PRESET = (
    REPO_ROOT / "src" / "camsim" / "config" / "presets" / "sensors" / "flagship_phone_50mp.json"
)
G2_REPORT_PATH = REPO_ROOT / "results" / "reports" / "g2_sensor.md"
G2_FIGURE_NAMES = (
    "g2_qe_cfa_ir_cut.png",
    "g2_ptc.png",
    "g2_snr_vs_irradiance.png",
    "g2_dark_current_vs_temperature.png",
    "g2_lcg_hcg_comparison.png",
    "g2_dsnu_prnu_histograms.png",
)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_preset_g2_recovery() -> None:
    """The flagship LCG mode recovers every G2 quantity within tolerance."""
    cfg = load_sensor(SENSOR_PRESET)
    sensor = camsim.Sensor(cfg, seed=5)
    report = characterize_sensor(
        sensor,
        irradiance_levels=[2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 4e-2, 8e-2],
        frames_per_level=32,
        mode="LCG",
        exposure_s=0.01,
        base_seed=9,
    )
    assert report.linearity_residual < 0.05
    assert 30.0 < report.dynamic_range_db < 90.0
    assert report.read_noise_e == pytest.approx(2.0, rel=0.05)
    assert report.dsnu_e == pytest.approx(0.5, rel=0.10)
    assert report.prnu_percent == pytest.approx(0.5, rel=0.10)
    assert report.saturation_capacity_e == pytest.approx(6000.0, rel=0.10)


def test_g2_report_and_figures_are_present() -> None:
    """The committed G2 report and its six figures are gate evidence, not stubs."""
    text = G2_REPORT_PATH.read_text(encoding="utf-8")
    assert "**Status: PASS**" in text
    assert "Assumptions and limitations" in text
    for name in G2_FIGURE_NAMES:
        path = REPO_ROOT / "results" / "figures" / name
        assert path.stat().st_size > 0, f"missing G2 figure: {name}"
        assert path.read_bytes()[:8] == PNG_MAGIC
