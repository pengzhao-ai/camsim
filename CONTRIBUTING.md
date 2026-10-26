# Contributing to camsim

Thank you for considering a contribution. `camsim` is a deterministic camera
sensor and ISP simulation library; correctness, traceability, and reproducible
numbers matter more than feature count. This guide describes the workflow and
the quality bar a change must meet.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Python 3.11, 3.12, or 3.13 is required. The recommended environment uses
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
uv run pre-commit install
```

pip/venv fallback:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all]"
pre-commit install
```

`uv sync --all-extras` installs the core, ML, tuning, docs, and development
extras, matching CI. The project-local `.venv/` is gitignored; never commit it,
generated notebooks, or build artefacts.

## Test-driven development

Write the failing test first, then the smallest implementation that passes,
then refactor. The repository convention is one test file per module, named
after the module it covers.

```bash
uv run pytest tests/test_units.py -v                    # one file
uv run pytest tests/test_units.py::test_name -v         # one test
uv run pytest -q                                        # the full suite
```

Tests must be deterministic and offline:

- Use explicit `numpy.random.Generator` objects; never the global NumPy RNG.
- Fix seeds for every stochastic routine and for PyTorch weights.
- Do not depend on current time, network access, locale, or test execution
  order.
- Cross-platform numeric equality is not promised for floating-point
  reductions. Assert with documented tolerances rather than bit equality when
  the result crosses platforms.
- Numerical code is implemented from first principles in NumPy and then
  cross-checked against reference libraries (`colour-science`,
  `colour-demosaicing`, `scikit-image`) in tests where a reference exists.

Changes to the sensor, ISP, or metrics layers must keep the phase validation
gates green (see below). A change that invalidates a published validation
report requires a re-run of that report and a new sign-off.

## Lint, type-check, and test

The Makefile targets and CI run the same commands:

```bash
make lint         # uv run ruff check . && uv run ruff format --check .
make format       # apply safe fixes and formatting
make typecheck    # uv run mypy src
make test         # uv run pytest
make coverage     # uv run pytest --cov=camsim --cov-report=term-missing
make docs         # uv run mkdocs build --strict
make gate         # sync, lint, typecheck, test, coverage, docs, build
```

`mypy` runs in strict mode on `src/`; new public functions need type
annotations. `ruff` enforces a 100-character line length and import order.
Production dependencies stay in `[project.dependencies]`; optional features go
in an extra, and `[all]` must compose every extra.

## Code and documentation style

- Public modules, classes, and functions carry NumPy-style docstrings.
- Every source file opens with a docstring covering the physical or
  mathematical foundation, realistic parameter ranges, and the failure
  artefacts the code guards against. Explain *why* a choice is made, not what
  a line of code mechanically does.
- Units belong in field and variable names (`pitch_um`, `full_well_e`,
  `exposure_s`). Stages before the tone curve operate in linear float32.
- Keep the dependency rules in `docs/architecture.md` intact: the sensor and
  ISP must not import scene, optics, or task code, and `camsim.metrics` must
  not import the simulation packages.
- Update the relevant docs page with behavioural changes. `mkdocs build
  --strict` treats broken navigation and links as errors.

## Notebooks and Jupytext

Notebooks are authored as Jupytext percent-format Python sources:

- Source of truth: `notebooks/*.py`, using `# %%` for code cells and
  `# %% [markdown]` for narrative.
- Never commit `.ipynb` files. Local pairing is configured in `jupytext.toml`
  and `.ipynb` is gitignored.
- Execute every notebook headlessly before opening a pull request:
  `make notebooks`. Outputs are written under `results/cache/` and are not
  committed.
- Each notebook must finish in under two minutes on Apple Silicon, use fixed
  seeds, and avoid network access.

Examples follow the same determinism and runtime rules: under five minutes
each on CPU in CI.

## Committing and pull requests

- Work on a feature branch and keep commits focused.
- Write an imperative subject line, for example `fix: reject negative
  exposure`, and explain the reasoning in the body.
- **Commit attribution:** every commit message ends with the trailer

  ```text
  Co-Authored-By: Claude Code <noreply@anthropic.com>
  ```

  Do not strip the trailer from existing commits, and do not rewrite published
  history.
- Open a pull request describing the change, the tests added, and any
  behaviour that changed. Link the phase gate or validation report affected.
- A pull request is mergeable when CI is green on Ubuntu and macOS for Python
  3.11, 3.12, and 3.13, the documentation builds strictly, and the change has
  been reviewed.

## Validation gates and the ML restriction

Phases end at formal gates; a gate is evidence, not ceremony:

| Gate | Scope | Evidence |
|---|---|---|
| G0 | install, lint, typecheck, tests, docs | `make gate` |
| G1 | scene and optics | `results/reports/g1_scene_optics.md` |
| G2 | sensor model | `results/reports/g2_sensor.md` |
| G3 | ISP | `results/reports/g3_isp.md` |
| G4 | metrics | `results/reports/g4_metrics.md` |
| G5 | human sign-off | `results/reports/validation_report.md` |

`docs/validation.md` is the public index of these reports, their tolerances,
and known limitations. Do not mark a gate complete without a reproducible
report.

**Machine-vision work does not begin before the G5 human sign-off.** No code
under `camsim.tasks`, no CNN training, and no task-accuracy experiments land
before G1–G4 are green, coverage is at least 90%, examples and notebooks run
within budget, and the project owner signs the validation report. Phase 6 is a
new, explicitly approved stage, not an automatic continuation.

## Third-party material and provenance

Every preset and dataset carries provenance: source, URL where public,
retrieval date, and an explicit list of engineering assumptions. Third-party
code, data, and fonts are recorded in [THIRD_PARTY.md](THIRD_PARTY.md) with
their licences. Do not vendor a dataset without adding its citation and
licence there.

## Versioning and deprecation

`camsim` follows Semantic Versioning. Before v1.0.0 the public API may change,
but every breaking change is recorded in [CHANGELOG.md](CHANGELOG.md).
After v1.0.0:

- deprecate a public name by documenting it in the changelog and emitting a
  warning for at least one minor release before removal;
- never remove a public name in a patch release;
- keep configuration schema compatibility explicit: bump `schema_version`
  when a preset can no longer be interpreted under the old semantics, and
  provide a migration note in the changelog.

## Release checklist

Releases are cut by the maintainer after a gate:

1. Version bumped in `src/camsim/_version.py` and `CITATION.cff`.
2. `CHANGELOG.md` has a dated release section.
3. `make gate` passes; CI is green across the matrix.
4. Coverage meets the active threshold (90% from G5 onward).
5. Docs build strictly and the validation report is signed off.
6. Wheel and sdist build with `uv build`; both are inspected.
7. Tag `vX.Y.Z` on GitHub. PyPI publication begins only at v1.0.0.
