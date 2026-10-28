"""Smoke tests for the Phase 5 documentation and release package.

A documentation site can rot without any simulation test noticing: a figure is
renamed, a nav entry points at a deleted page, the README quickstart drifts away
from the public API, or the coverage gate silently stops failing. These checks
make the editorial and release contract executable.

The tests run against repository-relative paths, matching the commands in the
README and the workflows CI executes from the repository root. The gallery
check parses every ``results/figures/*.png`` reference out of the README and
requires the file to exist, so a broken GitHub image cannot ship. The
documentation-site figures under ``docs/images`` are checked byte-for-byte
against their committed source, because the docs build must stay self-contained
without reaching outside ``docs/``.

The quickstart test rebuilds the documented sensor-to-metrics chain in process
at 64 x 64, and the benchmark test runs the public ``--quick`` mode in a
subprocess. Both use fixed seeds and assert structure, not machine-specific
timings.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
MKDOCS = REPO_ROOT / "mkdocs.yml"
DOCS = REPO_ROOT / "docs"
FIGURES = REPO_ROOT / "results" / "figures"


def test_readme_links_and_gallery_exist():
    readme = README.read_text(encoding="utf-8")
    assert "results/figures/" in readme
    assert (DOCS / "index.md").is_file()
    assert (DOCS / "validation.md").is_file()


def test_mkdocs_nav_covers_core_docs():
    text = MKDOCS.read_text(encoding="utf-8")
    for page in (
        "concepts.md",
        "architecture.md",
        "validation.md",
        "metrics.md",
        "datasheet_mapping.md",
    ):
        assert page in text


def test_readme_gallery_paths_exist():
    """Every figure the README displays must exist in the committed gallery."""
    readme = README.read_text(encoding="utf-8")
    referenced = set(re.findall(r"results/figures/[\w./-]+\.png", readme))
    assert referenced, "README shows no gallery figures"
    missing = sorted(path for path in referenced if not (REPO_ROOT / path).is_file())
    assert not missing, f"README references missing figures: {missing}"


def test_mkdocs_nav_covers_the_phase_5_pages():
    text = MKDOCS.read_text(encoding="utf-8")
    for page in ("isp_stages.md", "metrics.md", "api.md", "release.md"):
        assert page in text, f"nav missing {page}"
    assert "search" in text
    assert "mkdocstrings" in text


def test_docs_images_mirror_committed_figures():
    """The docs site is self-contained: docs/images mirrors results/figures."""
    images = DOCS / "images"
    assert images.is_dir(), "docs/images is missing"
    files = sorted(path.name for path in images.glob("*.png"))
    assert files, "docs/images contains no figures"
    for name in files:
        source = FIGURES / name
        assert source.is_file(), f"docs/images/{name} has no results/figures source"
        assert (images / name).read_bytes() == source.read_bytes(), (
            f"docs/images/{name} is stale; copy it from results/figures again"
        )


def test_docs_pages_and_release_workflow_exist():
    for page in ("index.md", "isp_stages.md", "metrics.md", "api.md", "release.md"):
        assert (DOCS / page).is_file(), f"missing docs page {page}"
    workflow = REPO_ROOT / ".github" / "workflows" / "release.yml"
    assert workflow.is_file(), "missing release workflow"
    assert (REPO_ROOT / ".github" / "workflows" / "docs.yml").is_file()


def test_api_page_renders_lazy_top_level_exports(tmp_path: Path) -> None:
    """The built API page must expose the five lazy public names as anchors."""
    pytest.importorskip("mkdocs")
    site_dir = tmp_path / "site"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str(site_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    page = (site_dir / "api" / "index.html").read_text(encoding="utf-8")
    for name in ("ISP", "Sensor", "SpectralIrradiance", "WavelengthGrid", "LensConfig"):
        assert f'id="camsim.{name}"' in page, f"API page has no anchor for camsim.{name}"


def test_release_workflow_tag_trigger_and_assets():
    text = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "tags:" in text and '"v*"' in text
    assert "contents: write" in text
    assert "uv build" in text
    assert "twine check" in text
    assert "gh release" in text or "softprops/action-gh-release" in text
    # Future tag releases publish through PyPI trusted publishing (OIDC).
    assert "pypi-publish" in text
    assert "environment: pypi" in text
    assert "id-token: write" in text
    assert "uv publish --trusted-publishing always" in text


def test_coverage_gate_is_enforced():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "*/camsim/viz/*" in pyproject
    assert "--cov-fail-under=90" in (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "--cov-fail-under=90" in (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )


def test_readme_quickstart_pipeline_runs() -> None:
    """Run the documented render -> expose -> ISP -> metrics chain at 64 x 64."""
    import camsim
    from camsim.config.io import PRESETS_DIR, load_isp, load_sensor
    from camsim.metrics import measure_all
    from camsim.scene.charts import slanted_edge_layout
    from camsim.scene.targets import render

    shape = (64, 64)
    scene = render("slanted_edge", lux=200.0, shape=shape, oversampling=2, seed=11)
    sensor = camsim.Sensor(
        load_sensor(PRESETS_DIR / "sensors" / "flagship_phone_50mp.json"), seed=1
    )
    raw = sensor.expose(scene.irradiance, 0.01, mode="LCG", seed=7)
    image = camsim.ISP(load_isp(PRESETS_DIR / "isp" / "default.json")).process(raw)
    report = measure_all(image, slanted_edge_layout(shape=shape))

    assert raw.data.dtype == np.uint16
    assert image.data.shape == (64, 64, 3)
    assert report.layout_id == "slanted_edge"
    assert report.mtf is not None
    assert report.acutance is not None and report.acutance > 0.0
    assert "mtf: layout has no edge_roi" not in report.metadata["skipped"]


def test_benchmark_quick_mode_runs() -> None:
    result = subprocess.run(
        [sys.executable, "benchmarks/benchmark_pipeline.py", "--quick"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "scene.render" in result.stdout
    assert "sensor.expose" in result.stdout
    assert "isp.process" in result.stdout
    assert "measure_all" in result.stdout
