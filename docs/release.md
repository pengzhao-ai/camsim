# Release process

This page describes how camsim is versioned and published. The current release
is `0.1.2`; the package is published on PyPI and the documentation site is
deployed from `main`.

## Versioning policy

- `src/camsim/_version.py` is the single source of truth; `pyproject.toml`
  reads it through hatchling.
- Releases use semantic versioning. Patch releases fix defects or documentation;
  minor releases add backward-compatible capability; `1.0.0` is the first
  stable API release.
- A release tag is `vX.Y.Z` and must match the package version without the `v`.
- PyPI versions are permanent. Never reuse a version after PyPI accepts it.
- PyPI releases use Trusted Publishing (OIDC) from GitHub Actions on a `v*`
  tag. No long-lived PyPI token is stored in GitHub.

### PyPI trusted publisher setup

1. On PyPI, open the `camsim` project -> **Settings -> Publishing -> Add a new
   publisher** and set repository `pengzhao-ai/camsim`, workflow `release.yml`,
   environment `pypi`.
2. In the GitHub repository, create an environment named `pypi`
   (**Settings -> Environments**) and restrict it to `main` or tags if desired.
3. Under **Settings -> Actions -> General -> Workflow permissions**, allow read
   and write permissions so the release job can create GitHub release assets.
   The PyPI job requests `id-token: write` explicitly.
4. Push a `vX.Y.Z` tag whose version matches `src/camsim/_version.py`. The
   Release workflow runs tests, builds the wheel and sdist, creates the GitHub
   release assets, and publishes to PyPI with
   `uv publish --trusted-publishing always`.

## Release checklist

Work through this list from top to bottom. Every item produces evidence in the
repository or on the release page.

### 1. Validation and changelog

- [ ] Validation reports are present and green; see [Validation](validation.md)
      and `results/reports/`.
- [ ] `CHANGELOG.md` has a dated section for this version, with breaking
      changes called out. Unreleased entries are moved, not duplicated.
- [ ] `CITATION.cff` version and release date match the tag; author list is
      current.
- [ ] `THIRD_PARTY.md` and `NOTICE` cover any dependency added since the last
      release.

### 2. Version and metadata

- [ ] Release version recorded in `src/camsim/_version.py`.
- [ ] README badges and repository links point at the current release location.
- [ ] Confirm `requires-python` and the classifiers in `pyproject.toml`.
- [ ] Package metadata renders correctly; README images and links use absolute
      URLs so the PyPI project page renders.

### 3. Quality checks

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass.
- [ ] `uv run mypy src` passes under strict settings.
- [ ] `uv run pytest` passes.
- [ ] `uv run pytest --cov=camsim --cov-fail-under=90` passes.
- [ ] `uv run mkdocs build --strict` passes with the docs extra installed.
- [ ] `make notebooks` executes every notebook headlessly, each under two
      minutes.
- [ ] `make figures` regenerates the committed gallery deterministically; copy
      any changed figures into `docs/images/` so the documentation site stays
      self-contained.
- [ ] `uv run python benchmarks/benchmark_pipeline.py --quick` passes; full
      benchmark timings are recorded as measurements, not guarantees.

### 4. Distribution

- [ ] `uv build` produces a wheel and an sdist in `dist/`.
- [ ] `uvx twine check dist/*` reports PASS for both artifacts.
- [ ] Install the wheel into a fresh environment and run a smoke import plus a
      small end-to-end example; confirm the wheel carries the bundled spectral
      CSVs under `camsim/data/spectra/`.
- [ ] Confirm the sdist includes `README.md`, `LICENSE`, `NOTICE`,
      `CHANGELOG.md`, `CITATION.cff`, `THIRD_PARTY.md`, `docs/`, `src/`, and
      `tests/` as configured in `pyproject.toml`.
- [ ] Publish through the tag-triggered workflow with OIDC trusted publishing.
      Do not upload by hand; PyPI versions are permanent.

### 5. Tag and publish

- [ ] Create an annotated tag `vX.Y.Z` on the release commit and push it. The
      tag triggers `.github/workflows/release.yml`, which syncs all extras, runs
      the test suite, builds the distributions, runs `twine check`, attaches
      the wheel, sdist, and `SHA256SUMS` to a GitHub release, and publishes to
      PyPI with `uv publish --trusted-publishing always`.
- [ ] Verify the GitHub release page shows the expected assets and notes.
- [ ] Confirm GitHub Pages deployed from `main` with the Docs workflow and that
      the live site contains the current API, ISP-stages, metrics, and release
      pages.
- [ ] Announce the release with a link to the GitHub release and the docs site.

## What the release workflow does

The workflow is triggered by a `v*` tag and has `contents: write` so it can
create or update the GitHub release; the PyPI job has `id-token: write` for
OIDC. It runs:

```text
checkout -> uv sync --all-extras -> pytest -> uv build -> uvx twine check
         -> gh release -> uv publish --trusted-publishing always
```

A failed test or `twine check` stops the job before any asset is uploaded. PyPI
publication happens in a separate OIDC job after the GitHub release job.

## Rollback

- If a release job fails before creating assets, delete the tag, fix the issue,
  and re-tag.
- If a release page was created with bad assets, delete the release and tag
  first, then re-run the workflow from a corrected commit; do not mutate the
  same tag in place.
- If a documentation deploy is wrong, revert the offending docs commit on
  `main` and let the Docs workflow redeploy; the release artifacts are
  unaffected.
- If a version is already published to PyPI, do not reuse it. Yank it and
  publish a new patch version.
