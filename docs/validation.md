# Validation

This page is the public index of camsim's phase-gate evidence. It is a stub in
Phase 0: the G1–G5 sections are filled as each phase completes, and a gate is
not marked green until its report is reproducible from a clean checkout.

## Lines of evidence

Every gate draws on four independent lines of evidence, so a bug must survive
more than one perspective before it can hide:

1. **Analytic invariants** — units and dimensional consistency, energy
   conservation, linear electron scaling with lux and exposure, Poisson
   variance in the shot-noise regime, clipping at full well and ADC saturation,
   known-PSF MTF, the sRGB OETF, and colour-difference identities.
2. **Round-trip characterization** — configure known sensor parameters, run
   EMVA 1288-style characterization, and recover the parameters within a stated
   tolerance.
3. **Reference-library cross-checks** — bilinear and Malvar-He-Cutler
   demosaicing against `colour-demosaicing`; colour transforms and ΔE against
   `colour-science`; filtering against `scipy` and `scikit-image`.
4. **Datasheet plausibility** — the flagship preset reproduces SNR, dynamic
   range, full well, and read noise within the ranges implied by public
   specifications, with assumptions listed explicitly.

Tolerances are stated per check in the phase report. Determinism is promised
within one environment (same OS, Python, NumPy, and SciPy); cross-platform
comparisons use tolerances because floating-point reduction order may differ.

## Gate status

| Gate | Phase | Scope | Report | Status |
|---|---|---|---|---|
| G0 | 0 | install, lint, typecheck, tests, docs build | this page | Green locally; CI runs the same commands |
| G1 | 1 | scene, illuminants, optics | `results/reports/g1_scene_optics.md` | Planned |
| G2 | 2 | sensor model and characterization | `results/reports/g2_sensor.md` | Planned |
| G3 | 3 | ISP stages and pipeline | `results/reports/g3_isp.md` | Planned |
| G4 | 4 | metrics harness | `results/reports/g4_metrics.md` | Planned |
| G5 | 5 | consolidated human sign-off | `results/reports/validation_report.md` | Planned |

## G0 — Foundation

The Phase 0 gate proves that a fresh environment can install, lint, type-check,
test, and document the package. Reproduce it from the repository root:

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run mkdocs build --strict
uv run python -c "import camsim; print(camsim.__version__)"
```

Expected: the suite passes, lint and formatting are clean, mypy reports no
issues under `strict`, the documentation builds with warnings treated as
errors, and the version prints `0.1.0.dev0`. The executable form of this gate
is `tests/test_ci_config.py` plus the CI workflows, and `make gate` runs the
same commands locally.

## G1–G4 — Phase validation reports

Each report will contain:

- the exact commands and environment used (OS, Python, dependency versions,
  configuration hash, and seeds);
- the metric, threshold, and measured value for every check;
- figures under `results/figures/` where a plot is clearer than a number;
- known limitations and assumptions, including any datasheet value that is an
  engineering estimate rather than a vendor measurement;
- the pass/fail verdict and the commit it applies to.

A report is appended to this page only when it has been regenerated against
the current commit. Changes that invalidate a report require a re-run.

## G5 — Human sign-off (blocks machine-vision work)

The G5 gate closes Phase 5 and opens the way to Phase 6. It requires:

- G1–G4 green;
- continuous integration green across Ubuntu and macOS on Python 3.11, 3.12,
  and 3.13;
- test coverage of at least 90%;
- a strict documentation build;
- examples and notebooks executing headless within their runtime budgets
  (notebooks under two minutes each, examples under five minutes each);
- a `results/reports/validation_report.md` with figures, tolerances,
  assumptions, and known limitations, reviewed by the project owner.

**No machine-vision code is written before this sign-off.** The CNN task
engine, training protocol, and task-accuracy sweeps in `camsim.tasks` do not
begin until the project owner explicitly approves the report. Subsequent
changes to sensor, ISP, or metrics code that invalidate the report require a
re-run and a new sign-off.

See [Datasheet mapping](datasheet_mapping.md) for how preset values are traced
to public specifications.
