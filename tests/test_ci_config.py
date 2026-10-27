"""G0/G5 CI contract: the operational artefacts exist and still mean what they say.

The Phase 0 gate is the exit criterion that a fresh clone installs, lints,
type-checks, tests, and builds its documentation. A missing workflow or
contribution document is not a runtime failure, so no simulation test would
catch it; this check makes the repository's operational contract executable.

The workflow is parsed as YAML rather than grepped as text. A string-presence
check passes when a tool name appears only in a comment, when a job is renamed
out of existence, or when a matrix loses a Python version; parsing the
structure asserts the jobs, steps, and matrix that actually run.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> dict[str, object]:
    payload = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / name).read_text())
    assert isinstance(payload, dict), f"{name} must parse as a mapping"
    return payload


def test_g0_docs_and_workflows_exist():
    assert (REPO_ROOT / ".github" / "workflows" / "ci.yml").is_file()
    assert (REPO_ROOT / ".github" / "workflows" / "docs.yml").is_file()
    assert (REPO_ROOT / "CONTRIBUTING.md").is_file()
    assert (REPO_ROOT / "docs" / "validation.md").is_file()
    assert (REPO_ROOT / "docs" / "datasheet_mapping.md").is_file()


def test_ci_quality_job_lints_and_typechecks():
    jobs = _workflow("ci.yml")["jobs"]
    assert isinstance(jobs, dict)
    quality = jobs["quality"]
    steps = quality["steps"]
    commands = [str(step.get("run", "")) for step in steps]
    assert any("ruff check" in command for command in commands)
    assert any("ruff format --check" in command for command in commands)
    assert any("mypy src" in command for command in commands)


def test_ci_test_matrix_covers_the_supported_python_versions():
    jobs = _workflow("ci.yml")["jobs"]
    matrix = jobs["test"]["strategy"]["matrix"]
    assert set(matrix["python-version"]) == {"3.11", "3.12", "3.13"}
    assert set(matrix["os"]) == {"ubuntu-latest", "macos-latest"}
    commands = [str(step.get("run", "")) for step in jobs["test"]["steps"]]
    assert any("pytest" in command and "--cov-fail-under=90" in command for command in commands)


def test_ci_has_examples_and_notebooks_jobs_with_timeouts():
    jobs = _workflow("ci.yml")["jobs"]
    for job_name, runner in (
        ("examples", "benchmarks/run_examples.py"),
        ("notebooks", "benchmarks/run_notebooks.py"),
    ):
        assert job_name in jobs, f"missing CI job {job_name}"
        job = jobs[job_name]
        assert job["timeout-minutes"] >= 10
        commands = [str(step.get("run", "")) for step in job["steps"]]
        assert any(runner in command for command in commands), f"{job_name} does not run {runner}"


def test_release_workflow_checks_the_tag_against_the_package_version():
    jobs = _workflow("release.yml")["jobs"]
    steps = jobs["release"]["steps"]
    commands = [str(step.get("run", "")) for step in steps]
    assert any(
        "GITHUB_REF_NAME" in command and "camsim.__version__" in command for command in commands
    ), "release must reject a tag that does not match camsim.__version__"
