"""Smoke test for the Phase 2 EMVA characterization walkthrough.

Example 04 is the G2 artifact generator: it characterizes the shipped flagship
preset, measures dark current versus temperature and fixed-pattern maps, writes
a machine-readable report JSON, and saves the ``g2_*.png`` figures. Running the
full 32-frame characterization inside the test suite would duplicate the G2
recovery test, so this smoke test uses the example's ``--quick`` mode, which
keeps the same code paths with fewer frames and a shorter temperature sweep.

The test runs the example in a subprocess with temporary figure and report
directories. A zero exit status alone would not catch a silently empty gallery,
so it also requires the six G2 figures and a non-empty report JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "04_emva_characterization.py"
G2_FIGURE_PREFIXES = (
    "g2_qe_cfa_ir_cut.png",
    "g2_ptc.png",
    "g2_snr_vs_irradiance.png",
    "g2_dark_current_vs_temperature.png",
    "g2_lcg_hcg_comparison.png",
    "g2_dsnu_prnu_histograms.png",
)


def test_example_04_quick_runs_and_writes_g2_artifacts(tmp_path: Path) -> None:
    """Run example 04 in quick mode and require its JSON and PNG deliverables."""
    figures = tmp_path / "figures"
    reports = tmp_path / "reports"
    env = {
        **os.environ,
        "CAMSIM_FIG_DIR": str(figures),
        "MPLBACKEND": "Agg",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE_PATH),
            "--quick",
            "--report-dir",
            str(reports),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for name in G2_FIGURE_PREFIXES:
        path = figures / name
        assert path.stat().st_size > 0, f"missing G2 figure: {name}"

    report_path = reports / "g2_sensor_results.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "camsim G2 sensor validation"
    for mode in ("LCG", "HCG"):
        assert payload[mode]["read_noise_e"] > 0.0
    assert payload["checks"]
    assert all("pass" in check for check in payload["checks"])
