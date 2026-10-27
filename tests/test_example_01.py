"""Smoke test for the Phase 1 optics walkthrough example.

The example is the executable contract behind notebook 01: a fresh checkout
must be able to run it headlessly and obtain deterministic ``01_*.png`` figures
without a display, a sensor model, or a network connection. Running it in a
subprocess with a temporary ``CAMSIM_FIG_DIR`` proves three things at once:
the script is importable as a ``__main__`` program, the environment variable
redirects output to a caller-chosen directory, and the gallery can be
regenerated from a clean temporary location.

A zero exit status alone would not catch a silently empty gallery, so the test
also requires at least three non-empty PNGs. It deliberately does not compare
image bytes: Matplotlib rendering can legitimately differ across font builds
and backends, while the requirement here is a reproducible pipeline run rather
than bit-identical antialiasing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "01_spectral_scene_to_raw.py"


def test_example_01_runs_and_writes_figures(tmp_path: Path) -> None:
    """Run example 01 headlessly and require at least three PNG figures."""
    env = {
        **os.environ,
        "CAMSIM_FIG_DIR": str(tmp_path),
        # The example sets Agg itself, but pinning it here keeps the test
        # honest if the script's backend selection regresses.
        "MPLBACKEND": "Agg",
    }
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    figures = sorted(tmp_path.glob("01_*.png"))
    assert len(figures) >= 3, f"expected at least three figures, got {figures}"
    assert all(path.stat().st_size > 0 for path in figures)
