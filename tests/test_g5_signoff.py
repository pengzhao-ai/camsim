"""G5 sign-off gate: the consolidated report, its links, and its claims.

G5 is the human decision that opens Phase 6. These tests make the candidate
artifact's contract executable without pretending to be the human decision:

* every phase gate G1-G4 is present in the consolidated report and its
  committed report and JSON still say PASS;
* the report contains the required assumptions, limitations, reproduction
  instructions, and the completed human sign-off checklist;
* every Markdown link into the repository resolves, the two G5 figures exist,
  and their ``docs/images`` mirrors are byte-identical;
* the quality-gate configuration still enforces 90 % coverage and documents
  the ``camsim.viz`` omission;
* the cross-cutting design criteria are measured directly with the public
  metrics and stages, not trusted from prose: denoise must raise flat-patch
  SNR with a two-sided MTF50 change bound and no edge ringing, sharpen must
  raise MTF50 and produce measurable overshoot, and camera auto-exposure must
  converge inside the preset's physical limits;
* ``camsim.tasks`` contains no implementation and imports no PyTorch;
* the report records the owner's sign-off and the `v0.1.0` release, while
  keeping machine-vision work deferred.

The tests do not create or inspect a git tag. The signed report is the
human-decision record; the git tag and package publication are release steps
performed after that record is committed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from camsim.camera import Camera
from camsim.config.io import PRESETS_DIR, load_isp, load_sensor
from camsim.config.schema import DenoiseConfig, LensConfig, SharpenConfig
from camsim.isp.denoise import denoise
from camsim.isp.pipeline import ISP
from camsim.isp.sharpen import unsharp_mask
from camsim.metrics import slanted_edge_mtf, snr_patch
from camsim.metrics.chart import PatchROI
from camsim.optics.propagation import apply_psf
from camsim.optics.psf import gaussian_psf
from camsim.scene.targets import render, slanted_edge_target
from camsim.sensor import Sensor

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "results" / "reports" / "validation_report.md"
FIGURE_DIR = REPO_ROOT / "results" / "figures"
DOCS_IMAGES = REPO_ROOT / "docs" / "images"
REPORT_DIR = REPO_ROOT / "results" / "reports"
SENSOR_PRESET = PRESETS_DIR / "sensors" / "flagship_phone_50mp.json"
ISP_PRESET = PRESETS_DIR / "isp" / "default.json"

PIXEL_PITCH_UM = 1.12
EDGE_SHAPE = (256, 256)
EDGE_ANGLE_DEG = 5.0
GAUSSIAN_SIGMA_PX = 1.0
GAUSSIAN_KERNEL_SIZE = 31
MTF_OVERSAMPLING = 8
EDGE_NOISE_SIGMA = 0.01
EDGE_NOISE_SEED = 71
FLAT_NOISE_SIGMA = 0.02
FLAT_NOISE_SEED = 72
FLAT_LEVEL = 0.4
FLAT_SHAPE = (64, 64)
DENOISE_MAX_MTF_CHANGE = 0.20
DENOISE_MAX_RINGING = 0.01
SHARPEN_OVERSHOOT_MARGIN = 1e-3


def _report() -> str:
    """Return the consolidated G5 report text."""
    assert REPORT_PATH.is_file(), f"missing G5 report: {REPORT_PATH}"
    return REPORT_PATH.read_text(encoding="utf-8")


def _blurred_edge() -> np.ndarray:
    """Return the same 5 degree, unit-sigma blurred edge as the generator."""
    edge = slanted_edge_target(EDGE_SHAPE, angle_deg=EDGE_ANGLE_DEG)
    blurred = apply_psf(
        edge[..., None],
        gaussian_psf(GAUSSIAN_SIGMA_PX, size=GAUSSIAN_KERNEL_SIZE),
    )[..., 0]
    return np.asarray(blurred, dtype=np.float64)


def _reported_number(label: str) -> float:
    """Read one numeric cell from the report's cross-cutting tables.

    The report is generated in one environment; CI runs the same tests on
    several. Float reductions can differ in the last digits across platforms,
    so the live measurement is compared with the reported value using a
    tolerance rather than a byte-for-byte string match.
    """
    match = re.search(
        rf"^\| {re.escape(label)} \| ([+-]?[0-9.]+(?:e[+-]?[0-9]+)?)(%?)",
        _report(),
        re.MULTILINE,
    )
    assert match, f"report has no numeric row {label!r}"
    value = float(match.group(1))
    return value / 100.0 if match.group(2) == "%" else value


def test_report_records_completed_signoff() -> None:
    """The report records the owner's explicit G5 sign-off."""
    text = _report()
    heading = "\n".join(text.splitlines()[:8]).lower()
    assert "signed off" in heading
    assert "awaiting human sign-off" not in heading
    assert not re.search(r"status:\s*pass", heading)

    # The gate table must carry the decision itself. G5 is a human gate, so it
    # is labelled signed off rather than a machine **PASS**.
    g5_rows = [line for line in text.splitlines() if line.startswith("| G5 |")]
    assert len(g5_rows) == 1, f"expected one G5 gate row, got {g5_rows}"
    assert "Signed off" in g5_rows[0]
    assert "v0.1.0" in g5_rows[0]
    assert "**PASS**" not in g5_rows[0]
    assert "awaiting human sign-off" not in g5_rows[0].lower()

    # The checklist and decision block are filled in.
    assert text.count("- [x]") >= 6
    assert "Owner: pengzhao-ai" in text
    assert "Decision:  [x] approve v0.1.0" in text


def test_all_phase_gates_are_present_and_pass() -> None:
    """G1-G4 appear as PASS rows and their artifacts still pass every check."""
    text = _report()
    for gate in ("G1", "G2", "G3", "G4"):
        pattern = rf"^\| {gate} \|.*\| \*\*PASS\*\* \|$"
        assert re.search(pattern, text, re.MULTILINE), f"{gate} is not a PASS row"
    g0_row = next(line for line in text.splitlines() if line.startswith("| G0 |"))
    assert "**PASS** (local)" in g0_row

    for name in ("g1_scene_optics.md", "g2_sensor.md", "g3_isp.md", "g4_metrics.md"):
        assert "**Status: PASS**" in (REPORT_DIR / name).read_text(encoding="utf-8")

    for name in ("g2_sensor_results.json", "g3_isp_results.json", "g4_metrics_results.json"):
        payload = json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert payload["gate_evidence"] is True
        assert payload.get("quick") is not True
        failed = [check["name"] for check in payload["checks"] if not check["pass"]]
        assert not failed, f"{name} has failing checks: {failed}"


def test_required_sections_and_signoff_checklist_exist() -> None:
    """Assumptions, limitations, reproduction, and the signed checklist exist."""
    text = _report()
    lower = text.lower()
    assert "## 6. assumptions" in lower
    assert "## 7. limitations" in lower
    assert "## 8. reproduction" in lower
    assert "## 10. human sign-off checklist" in lower
    assert "simulated sensor" in lower
    assert "provisional spectral data" in lower
    assert "in-sample colour calibration" in lower
    assert "iso 12233-style, not certified" in lower
    assert "no hdr, 3-d, motion blur, or autofocus" in lower
    assert "polychromatic psf" in lower
    assert "banding" in lower
    assert "owner:" in lower
    assert "decision:" in lower
    assert "approve v0.1.0" in lower


def test_linked_evidence_files_exist() -> None:
    """Every repository-relative Markdown link resolves."""
    text = _report()
    assert REPORT_PATH.parent.is_dir()
    missing: list[str] = []
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "#")):
            continue
        if not (REPORT_PATH.parent / target).resolve().exists():
            missing.append(target)
    assert not missing, f"report links to missing files: {missing}"
    for name in (
        "g1_scene_optics.md",
        "g2_sensor.md",
        "g2_sensor_results.json",
        "g3_isp.md",
        "g3_isp_results.json",
        "g4_metrics.md",
        "g4_metrics_results.json",
    ):
        assert (REPORT_DIR / name).is_file(), f"missing evidence {name}"


def test_phase_artifact_bindings_match_current_files() -> None:
    """The report's SHA-256 bindings must match the committed artifacts.

    This is the freshness guard the prose promises: changing a G1-G4 report or
    JSON without regenerating the consolidated report makes the binding stale
    and fails here. It proves consistency with the committed artifacts, not
    that the artifacts were regenerated from the current commit.
    """
    text = _report()
    bindings: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\| `([\w.-]+\.(?:md|json))` \| `([0-9a-f]{64})` \|", line)
        if match:
            bindings[match.group(1)] = match.group(2)
    expected = {
        "g1_scene_optics.md",
        "g2_sensor.md",
        "g2_sensor_results.json",
        "g3_isp.md",
        "g3_isp_results.json",
        "g4_metrics.md",
        "g4_metrics_results.json",
    }
    assert set(bindings) == expected, f"unexpected artifact bindings: {sorted(bindings)}"
    for name, digest in bindings.items():
        actual = hashlib.sha256((REPORT_DIR / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} changed without regenerating the G5 report"


def test_report_records_the_actual_regeneration() -> None:
    """The sign-off report, not only the task report, records the re-runs."""
    lower = _report().lower()
    assert "regeneration record" in lower
    assert "examples 01-04" in lower
    assert "notebooks 01-03" in lower
    assert "byte-identical" in lower
    assert "g1 remained pass" in lower


def test_g5_figures_exist_and_docs_mirror_is_byte_identical() -> None:
    """The two G5 figures exist and the documentation copies match exactly."""
    for name in ("g5_gate_summary.png", "g5_evidence.png"):
        source = FIGURE_DIR / name
        mirror = DOCS_IMAGES / name
        assert source.is_file(), f"missing figure {name}"
        assert mirror.is_file(), f"missing docs mirror {name}"
        assert mirror.read_bytes() == source.read_bytes(), f"stale docs mirror {name}"


def test_coverage_gate_and_viz_omission_are_documented() -> None:
    """The 90 % gate, its scope, and the whole-package exception are explicit."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "*/camsim/viz/*" in pyproject
    assert "branch = true" in pyproject
    assert "--cov-fail-under=90" in (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--cov-fail-under=90" in ci
    assert (REPO_ROOT / "coverage-whole.rc").is_file()
    validation = (REPO_ROOT / "docs" / "validation.md").read_text(encoding="utf-8").lower()
    assert "90" in validation
    assert "viz" in validation
    assert "whole-package" in validation

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "g5_validation_report",
        REPO_ROOT / "benchmarks" / "g5_validation_report.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    subject = float(module.COVERAGE_SUBJECT_PCT)
    whole = float(module.COVERAGE_WHOLE_PCT)
    assert subject >= 90.0
    assert whole < subject
    report = _report().lower()
    assert f"{subject:.2f}" in report
    assert f"{whole:.2f}" in report
    assert "subject-module" in report
    assert "whole-package" in report
    assert "reviewed exception" in report
    assert "*/camsim/viz/*" in report


def test_local_gallery_evidence_covers_examples_and_notebook_budgets() -> None:
    """Every example and notebook ran headlessly inside its documented budget."""
    examples = json.loads((REPORT_DIR / "examples_timings.json").read_text(encoding="utf-8"))
    notebooks = json.loads((REPORT_DIR / "notebooks_timings.json").read_text(encoding="utf-8"))
    assert examples["passed"] is True and notebooks["passed"] is True
    assert examples["timeout_s"] == 300.0
    assert notebooks["timeout_s"] == 120.0
    example_names = {path.name for path in (REPO_ROOT / "examples").glob("*.py")}
    notebook_names = {path.name for path in (REPO_ROOT / "notebooks").glob("*.py")}
    assert {item["name"] for item in examples["items"]} == example_names
    assert {item["name"] for item in notebooks["items"]} == notebook_names
    assert all(item["seconds"] <= examples["timeout_s"] for item in examples["items"])
    assert all(item["seconds"] <= notebooks["timeout_s"] for item in notebooks["items"])
    report = _report().lower()
    assert "two-minute" in report or "120 s" in report
    assert "ci execution is pending a push" in report
    assert "local equivalents passed" in report


def test_identity_ccm_and_large_frame_limits_are_disclosed() -> None:
    """The sign-off report names the identity-CCM baseline and the 50 MP limit."""
    lower = _report().lower()
    assert "identity ccm" in lower
    assert "fit_ccm" in lower
    assert "50 mp" in lower
    assert "memory" in lower
    assert "1024 x 1024" in lower


def test_no_machine_vision_implementation() -> None:
    """The gated namespace stays a docstring-only package before sign-off."""
    tasks_dir = REPO_ROOT / "src" / "camsim" / "tasks"
    assert sorted(path.name for path in tasks_dir.glob("*.py")) == ["__init__.py"]
    text = (tasks_dir / "__init__.py").read_text(encoding="utf-8")
    assert "torch" not in text
    module = ast.parse(text)
    executable = [node for node in module.body if not isinstance(node, ast.Expr | ast.ImportFrom)]
    assert not executable, "camsim.tasks contains executable statements"
    assert "no machine-vision" in _report().lower()


def test_denoise_raises_snr_with_bounded_mtf50_change_and_no_ringing() -> None:
    """Direct metrics calls reproduce the report's two-sided denoise evidence."""
    blurred = _blurred_edge()
    rng = np.random.default_rng(EDGE_NOISE_SEED)
    noisy = blurred + EDGE_NOISE_SIGMA * rng.standard_normal(blurred.shape)
    config = DenoiseConfig(
        method="bilateral",
        luma_strength=1.0,
        chroma_strength=0.0,
        radius=1.5,
    )
    denoised = np.asarray(denoise(noisy, config), dtype=np.float64)
    before = slanted_edge_mtf(noisy, PIXEL_PITCH_UM, oversampling=MTF_OVERSAMPLING)
    after = slanted_edge_mtf(denoised, PIXEL_PITCH_UM, oversampling=MTF_OVERSAMPLING)
    relative_change = after.mtf50_cp / before.mtf50_cp - 1.0
    assert abs(relative_change) <= DENOISE_MAX_MTF_CHANGE

    profile_before = blurred.mean(axis=0)
    profile_after = denoised.mean(axis=0)
    low = float(np.median(profile_before[:40]))
    high = float(np.median(profile_before[-40:]))
    contrast = high - low
    overshoot = (float(profile_after.max()) - high) / contrast
    undershoot = (low - float(profile_after.min())) / contrast
    assert abs(overshoot) <= DENOISE_MAX_RINGING
    assert abs(undershoot) <= DENOISE_MAX_RINGING

    flat_rng = np.random.default_rng(FLAT_NOISE_SEED)
    flat = FLAT_LEVEL + FLAT_NOISE_SIGMA * flat_rng.standard_normal(FLAT_SHAPE)
    denoised_flat = np.asarray(denoise(flat, config), dtype=np.float64)
    roi = PatchROI(name="flat", x=0, y=0, height=FLAT_SHAPE[0], width=FLAT_SHAPE[1])
    snr_before = snr_patch(flat, roi)
    snr_after = snr_patch(denoised_flat, roi)
    assert snr_after.snr_db > snr_before.snr_db

    assert math.isclose(_reported_number("MTF50 before denoise"), before.mtf50_cp, rel_tol=0.01)
    assert math.isclose(_reported_number("MTF50 after denoise"), after.mtf50_cp, rel_tol=0.01)
    assert math.isclose(
        _reported_number("Signed relative MTF50 change"),
        relative_change,
        rel_tol=0.02,
        abs_tol=1e-4,
    )
    assert math.isclose(_reported_number("Edge overshoot"), overshoot, rel_tol=0.05, abs_tol=1e-4)
    assert math.isclose(_reported_number("Edge undershoot"), undershoot, rel_tol=0.05, abs_tol=1e-4)
    assert math.isclose(_reported_number("SNR before denoise"), snr_before.snr_db, abs_tol=0.05)
    assert math.isclose(_reported_number("SNR after denoise"), snr_after.snr_db, abs_tol=0.05)
    assert math.isclose(
        _reported_number("SNR gain"),
        snr_after.snr_db - snr_before.snr_db,
        abs_tol=0.05,
    )


def test_sharpen_raises_mtf50_and_produces_overshoot() -> None:
    """Direct metrics calls reproduce the report's sharpen evidence."""
    blurred = _blurred_edge()
    config = SharpenConfig(amount=1.0, radius=1.0, threshold=0.0, luma_only=True)
    sharpened = np.asarray(unsharp_mask(blurred, config), dtype=np.float64)
    before = slanted_edge_mtf(blurred, PIXEL_PITCH_UM, oversampling=MTF_OVERSAMPLING)
    after = slanted_edge_mtf(sharpened, PIXEL_PITCH_UM, oversampling=MTF_OVERSAMPLING)
    assert after.mtf50_cp > before.mtf50_cp

    profile_before = blurred.mean(axis=0)
    profile_after = sharpened.mean(axis=0)
    low = float(np.median(profile_before[:40]))
    high = float(np.median(profile_before[-40:]))
    normalized_max = float((profile_after.max() - low) / (high - low))
    assert normalized_max > 1.0 + SHARPEN_OVERSHOOT_MARGIN

    assert math.isclose(_reported_number("MTF50 before sharpen"), before.mtf50_cp, rel_tol=0.01)
    assert math.isclose(_reported_number("MTF50 after sharpen"), after.mtf50_cp, rel_tol=0.01)
    assert math.isclose(
        _reported_number("Normalized profile maximum"), normalized_max, abs_tol=1e-3
    )


def test_camera_auto_exposure_converges_and_respects_limits() -> None:
    """The report cites the gate tests and a direct AE run still passes."""
    text = _report()
    assert "test_auto_exposure_converges" in text
    assert "test_auto_exposure_respects_limits" in text
    assert "0.01 lux" in text

    camera = Camera(
        LensConfig(
            f_number=1.8,
            focal_length_mm=6.4,
            sensor_diagonal_mm=11.46,
            transmittance=0.9,
        ),
        Sensor(load_sensor(SENSOR_PRESET), seed=1),
        ISP(load_isp(ISP_PRESET)),
    )
    config = camera.sensor.config
    for lux in (20.0, 0.01):
        scene = render("flat_patch", illuminant="D65", lux=lux, shape=(64, 64), oversampling=2)
        result = camera.auto_expose(scene)
        assert result.converged
        assert 1 <= result.iterations <= 12
        assert config.exposure.min_s <= result.exposure_s <= config.exposure.max_s
        assert config.analog_gain.min_db <= result.analog_gain_db <= config.analog_gain.max_db
        assert math.isfinite(result.digital_gain) and result.digital_gain >= 1.0
        assert result.mode in {"LCG", "HCG"}
        assert abs(result.metered_value - 0.18) <= 0.02


def test_rccc_baseline_is_cited_with_its_caveat() -> None:
    """The RCCC evidence names the existing tests and the Bayer caveat."""
    text = _report()
    assert "test_rccc_baseline_shape_and_measured_sites" in text
    assert "test_rccc_baseline_preserves_dc_everywhere" in text
    assert "test_g3_rccc_baseline_smoke" in text
    assert "RCCC/Bayer caveat" in text
    assert "advanced RCCC demosaicing is deferred" in text


def test_report_records_release_and_deferred_machine_vision() -> None:
    """The signed report records v0.1.0 and keeps Phase 6+ deferred."""
    lower = _report().lower()
    assert "awaiting human sign-off" not in lower
    assert "release record" in lower
    assert "v0.1.0" in lower
    assert "signed off" in lower
    assert "phase 6+ remains deferred" in lower
    assert "no machine-vision code is authorized" in lower
    # The report is the human-decision record; the tests never create the tag.
