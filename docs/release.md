# Release checklist

This page is the release contract for camsim. A tag is cut only after the G5
human sign-off in [Validation](validation.md); the repository version is
`0.1.0.dev0` and **no tag exists yet**. PyPI publication is deliberately
deferred to v1.0.0, after the Phase 6–9 machine-vision work is complete.

## Versioning policy

- `src/camsim/_version.py` is the single source of truth; `pyproject.toml`
  reads it through hatchling. Do not duplicate the version elsewhere.
- `0.1.0.dev0` is the development version for the pipeline/metrics release.
  Cutting `v0.1.0` requires the G5 sign-off; until then the version stays a
  `.dev0` pre-release.
- Future scope changes follow semantic versioning: patch releases fix defects,
  minor releases add backward-compatible capability, and v1.0.0 is the first
  PyPI release.
- A release tag is `vX.Y.Z` and matches the package version without the `v`.

## Checklist

Work through this list from top to bottom. Every item produces evidence in the
repository or the release page; a checked box without evidence is not a pass.

### 1. Scope and sign-off

- [ ] G1–G4 reports are present and green; see [Validation](validation.md) and
      `results/reports/g1_scene_optics.md`, `g2_sensor.md`, `g3_isp.md`, and
      `g4_metrics.md`.
- [ ] The consolidated `results/reports/validation_report.md` exists and the
      project owner has signed off G5.
- [ ] `CHANGELOG.md` has a section for this version, dated, with breaking
      changes called out. Unreleased entries are moved, not duplicated.
- [ ] `CITATION.cff` version and release date match the tag; author list is
      current.
- [ ] `THIRD_PARTY.md` and `NOTICE` cover any dependency added since the last
      release.
- [ ] No machine-vision code (`camsim.tasks`) ships before G5 sign-off.

### 2. Version and metadata

- [ ] Bump `0.1.0.dev0` to the release version in `src/camsim/_version.py`.
- [ ] Update the version-dependent badges and repository links: replace the
      `OWNER` placeholder in `README.md` with the GitHub account or
      organisation, and enable the CI, Docs, and Release badges.
- [ ] Confirm `requires-python = ">=3.11,<3.14"` and the classifier list still
      describe the supported interpreters.
- [ ] Confirm the package imports and reports the new version:
      `uv run python -c "import camsim; print(camsim.__version__)"`.

### 3. Quality gates

- [ ] `uv sync --all-extras` from a clean clone.
- [ ] `uv run pytest -q` passes.
- [ ] `uv run pytest --cov=camsim --cov-fail-under=90` passes. The gate covers
      statement and branch coverage of the simulation, metrics, configuration,
      and CLI modules; `camsim.viz` is excluded because the presentation layer
      is validated by the executed gallery examples and notebooks, not by unit
      tests. Do not lower the threshold to make a release; fix the gap or
      document a reviewed exception.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass.
- [ ] `uv run mypy src` passes under `strict`.
- [ ] `uv run mkdocs build --strict` passes with the docs extra installed.
- [ ] `make notebooks` executes every notebook headlessly, each under two
      minutes.
- [ ] `make figures` regenerates the committed gallery deterministically;
      copy any changed figures into `docs/images/` so the documentation site
      stays self-contained.
- [ ] `uv run python benchmarks/benchmark_pipeline.py --quick` passes; the
      full 512 × 512 benchmark runs and its timings are recorded in the release
      notes as measurements, not as guarantees.

### 4. Distribution

- [ ] `uv build` produces a wheel and an sdist in `dist/`.
- [ ] `uvx twine check dist/*` reports PASS for both artifacts.
- [ ] Install the wheel into a fresh environment and run a smoke import plus a
      small end-to-end example; confirm the wheel carries the bundled spectral
      CSVs under `camsim/data/spectra/`.
- [ ] Confirm the sdist includes `README.md`, `LICENSE`, `NOTICE`,
      `CHANGELOG.md`, `CITATION.cff`, `THIRD_PARTY.md`, `docs/`, `src/`, and
      `tests/` as configured in `pyproject.toml`.
- [ ] Do **not** upload to PyPI for v0.1.0; publication is deferred to
      v1.0.0.

### 5. Tag and publish

- [ ] Create an annotated tag `vX.Y.Z` on the signed-off commit and push it.
      The tag triggers `.github/workflows/release.yml`, which syncs all
      extras, runs the test suite, builds the distributions, runs
      `twine check`, and attaches the wheel, sdist, and `SHA256SUMS` to a
      GitHub release.
- [ ] Verify the GitHub release page shows the expected assets and release
      notes.
- [ ] Confirm GitHub Pages deployed from `main` with the Docs workflow and
      that the live site contains the current API, ISP-stages, metrics, and
      release pages.
- [ ] Announce the release with a link to the GitHub release and the
      documentation site.

## What the release workflow does

The workflow is triggered by a `v*` tag and has `contents: write` so it can
create or update the GitHub release. It runs:

```text
checkout → uv sync --all-extras → pytest → uv build → uvx twine check → gh release
```

It does **not** publish to PyPI. A failed test or `twine check` stops the job
before any release asset is uploaded, so a tag never quietly produces a broken
release without artifacts.

## Rollback

- If a release job fails before creating assets, delete the tag, fix the issue,
  and re-tag.
- If a release page was created with bad assets, delete the release and tag
  first, then re-run the workflow from a corrected commit; do not mutate the
  same tag in place.
- If a documentation deploy is wrong, revert the offending docs commit on
  `main` and let the Docs workflow redeploy; the release artifacts are
  unaffected.

## Deferred to v1.0.0

- PyPI publication and trusted publishing.
- Machine-vision experiments (`camsim.tasks`), the mono-versus-Bayer study,
  auto-tuning, and the metric-correlation whitepaper (Phases 6–9).
- `camsim.camera` convenience orchestration and auto-exposure, which are
  scaffold-only today.
- Any public compatibility promise for APIs outside the modules listed in the
  [API reference](api.md).
