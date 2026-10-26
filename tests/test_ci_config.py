"""G0 gate: the documentation and continuous-integration artefacts must exist.

The G0 gate is the Phase 0 exit criterion: a fresh clone installs, lints,
type-checks, tests, and builds its documentation. A missing workflow or
contribution document is not a runtime failure, so no simulation test would
catch it; this check makes the repository's operational contract executable.

It runs against repository-relative paths, matching the commands documented in
the README and executed by CI from the repository root. The test deliberately
reads the workflow files rather than only checking their existence, so an
accidental downgrade of the Python matrix or removal of a quality tool fails
the gate instead of silently narrowing coverage.
"""

from pathlib import Path


def test_g0_docs_and_workflows_exist():
    assert Path(".github/workflows/ci.yml").exists()
    assert Path(".github/workflows/docs.yml").exists()
    assert Path("CONTRIBUTING.md").exists()
    assert Path("docs/validation.md").exists()
    assert Path("docs/datasheet_mapping.md").exists()
    text = Path(".github/workflows/ci.yml").read_text()
    assert "3.11" in text and "3.12" in text and "3.13" in text
    assert "ruff" in text and "mypy" in text and "pytest" in text
