"""Import-independence guard for the metrics layer.

The metrics harness must be able to judge a camera without depending on the
model of that camera. If ``import camsim.metrics`` loaded the sensor, ISP,
scene, or optics packages, a metrics bug could be compensated by a matching
change in the simulator, and the measurement could no longer serve as an
independent check. The test runs in a fresh subprocess so that modules imported
by the rest of the test session cannot mask a dependency introduced anywhere in
the metrics import graph.
"""

from __future__ import annotations


def test_metrics_does_not_import_simulator_modules():
    import subprocess, sys  # noqa: E401, I001

    code = "import camsim.metrics, sys; assert 'camsim.sensor' not in sys.modules; assert 'camsim.isp' not in sys.modules; assert 'camsim.scene' not in sys.modules; assert 'camsim.optics' not in sys.modules"  # noqa: E501
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
