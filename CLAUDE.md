# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo status

Phase 5 consolidation is complete and the repository is at the **G5 human
sign-off gate**: G1–G4 pass with committed evidence, the consolidated G5
candidate report is available, and the `v0.1.0` tag has not been created. CI
workflows (including examples and notebooks) exist, but CI execution is pending
a push; the local equivalents pass. Machine-vision code remains gated.

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
uv run pytest --cov=camsim --cov-fail-under=90          # subject-module gate
uv run pytest --cov=camsim --cov-config=coverage-whole.rc  # whole package incl. viz
uv run python benchmarks/run_examples.py   # 300 s per example
uv run python benchmarks/run_notebooks.py  # 120 s per notebook
uv run python benchmarks/benchmark_pipeline.py [--quick|--large]
uv run jupytext --to ipynb --execute notebooks/NN_name.py --output /tmp/NN.ipynb
```

- Python 3.11–3.13, `uv` with a committed `uv.lock`; the pip/venv fallback is documented in the README.
- `make notebooks` executes every Jupytext notebook headlessly through `benchmarks/run_notebooks.py` with a 120 s per-notebook timeout; `make figures` runs every example through `benchmarks/run_examples.py` with a 300 s per-example timeout. Both runners fail fast and write per-item timing evidence under `results/reports/`.
- The CLI entry point is `camsim = "camsim.cli:main"` with `validate`, `schema`, `info`, `render`, `characterize`, `measure`, and `figures` subcommands.
- `scene.render` and `Sensor.expose` enforce a configurable simulation-memory budget (8 GiB default; `CAMSIM_MAX_SIMULATION_BYTES` or `max_memory_bytes=`). Full 50 MP spectral simulation is not supported at v0.1.0.

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

- **Linear-space invariant:** DPC, BLC, LSC, AWB, demosaic, CCM, denoise, and sharpen operate in linear float32. The CCM non-negativity clamp is the one bounded exception and is inactive on in-gamut data; the G3 linearity check reports its pre-clamp margin. The tone curve and output OETF are the only intentionally non-linear steps.
- **Luma and primaries:** Rec. 709 luma weights assume linear sRGB. `measure_all` converts a linear Display P3 buffer to linear sRGB before SNR/MTF/acutance and records the conversion; direct 3-D metric calls assume sRGB input.
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

Phase 0 scaffold (G0: install, lint, typecheck, tests, docs) · Phase 1 scene/optics (G1) · Phase 2 sensor (G2) · Phase 3 ISP (G3) · Phase 4 metrics (G4) · **Phase 5 consolidation and human sign-off (G5, awaiting sign-off)** · Phases 6–9 machine vision, mono-vs-Bayer, auto-tuning, metric correlation. Each gate's validation criteria are in the design spec, Section 10.
