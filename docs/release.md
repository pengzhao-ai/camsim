# Release checklist

This page is the release contract for camsim. v0.1.0 was approved at the G5
human sign-off on 2026-10-27 and is the first GitHub and PyPI release. The
checklist below remains the contract for future releases. PyPI publication of
later feature releases follows the same process; v1.0.0 is the feature-complete
release after the deferred Phase 6–9 machine-vision work.

## Versioning policy

- `src/camsim/_version.py` is the single source of truth; `pyproject.toml`
  reads it through hatchling. Do not duplicate the version elsewhere.
- `0.1.0` is the first pipeline/metrics release, approved at G5 on
  2026-10-27. Future work uses `X.Y.Z` from this baseline.
- Future scope changes follow semantic versioning: patch releases fix defects,
  minor releases add backward-compatible capability, and v1.0.0 is the
  feature-complete release after Phase 6–9.
- A release tag is `vX.Y.Z` and matches the package version without the `v`.
- PyPI releases use **Trusted Publishing** (OIDC) from GitHub Actions on a
  `v*` tag. The first 0.1.0 upload used a one-time account token; future
  releases require no stored PyPI token.

### PyPI trusted publisher setup

1. On PyPI, open the `camsim` project → **Settings → Publishing → Add a new
   publisher** and set owner `pengzhao-ai`, repository `camsim`, workflow
   `release.yml`, environment `pypi`.
2. In the GitHub repository, create an environment named `pypi`
   (**Settings → Environments**) and restrict it to the `main` branch/tags if
   desired.
3. Under **Settings → Actions → General → Workflow permissions**, allow
   read and write permissions so the release job can create GitHub release
   assets. The PyPI job requests `id-token: write` explicitly.
4. Push a `vX.Y.Z` tag whose version matches `src/camsim/_version.py`. The
   `Release` workflow runs tests, builds the wheel/sdist, creates the GitHub
   release assets, and publishes to PyPI with `uv publish
   --trusted-publishing always`.
5. Every publication needs a new version; never reuse a version number after
   PyPI accepts it.

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

- [x] Release version `0.1.0` recorded in `src/camsim/_version.py`.
- [ ] Update the version-dependent badges and repository links: replace the
      `pengzhao-ai` repository owner in `README.md` badges/links (already
      applied for v0.1.0) with the GitHub account or
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
