# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo status

Phase 0 is in progress: the repository scaffold is in place — the `camsim` package installs, the CLI skeleton runs, and lint, typecheck, tests, and docs build. Simulation, ISP, metrics, and machine-vision modules are added phase by phase.

- **Authoritative spec:** `docs/superpowers/specs/2026-10-26-camsim-design.md` — read it before building anything.
- **Implementation plan:** `docs/superpowers/plans/2026-10-26-camsim-core-pipeline.md` — task-by-task requirements and gates.
- **Superseded plan:** `docs/reference/Only for reference - iq_for_machine_vision_project_plan_revised.md`. Kept for provenance only; do not implement from it.
- `archives/` is gitignored superseded material. Keep it on disk; do not delete or commit it.

Do not write machine-vision code before the G5 human sign-off.

## What this repo is

`camsim` — a professional, fully deterministic Python library that simulates a camera from spectral scene irradiance to a rendered image, measures image quality with standard metrics, and studies how optics, sensor noise, and ISP parameters affect machine-vision accuracy.

Three clean layers:

1. **Simulation pipeline** — `scene` and `optics` are supporting helpers; `sensor` (irradiance → RAW) and `isp` (RAW → rendered image) are the reusable core and never depend on them.
2. **Metrics** — `camsim.metrics` is import-independent: it operates on arrays plus metadata and must not import `camsim.sensor`, `camsim.isp`, `camsim.scene`, or `camsim.optics`.
3. **Machine vision** — `camsim.tasks` (PyTorch), gated behind G5.

## Commands

```bash
uv sync --all-extras          # install the package and every extra
make setup / make test / make lint / make typecheck / make docs
uv run pytest                 # full suite
uv run pytest tests/test_package.py::test_import_and_version -v   # single test
uv run ruff check . && uv run ruff format --check .
uv run mypy src               # strict, src/ only
uv run mkdocs build --strict
uv run jupytext --to ipynb --execute notebooks/NN_name.py --output /tmp/NN.ipynb
```

- Python 3.11–3.13, `uv` with a committed `uv.lock`; the pip/venv fallback is documented in the README.
- `make notebooks` executes every Jupytext notebook headlessly; notebooks must run in under 2 minutes each on Apple Silicon.
- The CLI entry point is `camsim = "camsim.cli:main"` with `validate`, `schema`, and `info` subcommands.
- `make figures` will regenerate the committed figure gallery once the plotting layer lands (Phase 4/5); it currently reports that the gallery is empty.

## Package architecture

```text
src/camsim/
├── __init__.py       # PEP 562 lazy export map (_LAZY_EXPORTS)
├── _version.py       # single source of the package version
├── cli.py            # argparse subcommands
├── core/             # units, spectral cubes, RAW/rendered frames, colour references
├── config/           # Pydantic schemas, JSON I/O, presets
├── scene/ optics/    # supporting simulation helpers
├── sensor/ isp/      # reusable core
├── characterization/ # EMVA 1288-style measurements
├── camera/           # convenience orchestration and auto-exposure
├── metrics/          # independent IQ harness
├── tasks/            # machine-vision experiments (gated)
├── viz/ io/          # presentation and file I/O
```

- `import camsim` must stay cheap. Public names are registered lazily in `_LAZY_EXPORTS` as `name -> (module path, attribute name)`; add new public types there as they land. Do not eagerly import simulation packages from `camsim/__init__.py`.
- Config uses Pydantic v2 with `extra="forbid"`, units in every field name, cross-field physical validators, and `provenance` in every preset.
- `RawFrame` carries every piece of metadata the ISP needs; no hidden globals.

## Non-negotiable conventions

- **Linear-space invariant:** DPC, BLC, LSC, AWB, demosaic, CCM, denoise, and sharpen operate in linear float32. The tone curve and output OETF are the only non-linear steps.
- **Spectral boundary:** the sensor consumes irradiance at the sensor plane before IR-cut/CFA/microlens. The sensor never resamples implicitly; spatial and wavelength resampling use explicit, energy-conserving helpers.
- **Determinism:** explicit `numpy.random.Generator` only, no global RNG; seeds and config hashes recorded in outputs. Bit-identical within one environment; tolerance-based across platforms.
- **From scratch first:** implement core numerics (MHC demosaic, e-SFR, ridge CCM, …) in NumPy, then validate against `colour-science`, `colour-demosaicing`, and `scikit-image` in tests.
- **Pedagogy:** every source file opens with a docstring covering the physical/mathematical foundation, real-world parameter ranges, and failure artefacts. Comments explain engineering intent, not code mechanics.
- **Tests:** one file per module in `tests/`; Hypothesis for invariants; deterministic seeds. Import-independence of `camsim.metrics` is enforced by test. Coverage target is ≥ 90% at G5.
- **Notebooks:** Jupytext percent format (`# %%` code, `# %% [markdown]` narrative) in `notebooks/*.py`; never commit `.ipynb` files.
- **Machine vision:** PyTorch on MPS when available with CPU fallback; manual seeding; no ML work before G5.
- `results/figures/` holds committed PNG figures; `results/cache/` is gitignored.
- **Commits:** end every commit message with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

## Reference hardware defaults

- Reference lens (design Section 7.1.1): f/1.8, 24 mm full-frame equivalent, ~84° diagonal FOV, T = 0.9, 1/1.3" sensor, ~6.4 mm actual focal length.
- Flagship preset (plan Global Constraints): LCG 100 µV/e⁻ / 6000 e⁻ full well / 2 e⁻ rms read noise; HCG 300 µV/e⁻ / 2000 e⁻ / 0.7 e⁻ rms; 10-bit ADC, 0.8 V full scale, black level 64 DN.
- These are overridable presets with provenance, not claims about a commercial part.

## Phases and gates

Phase 0 scaffold (G0: install, lint, typecheck, tests, docs) · Phase 1 scene/optics (G1) · Phase 2 sensor (G2) · Phase 3 ISP (G3) · Phase 4 metrics (G4) · Phase 5 consolidation and human sign-off (G5) · Phases 6–9 machine vision, mono-vs-Bayer, auto-tuning, metric correlation. Each gate's validation criteria are in the design spec, Section 10.
