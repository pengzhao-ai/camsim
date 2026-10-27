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
| G1 | 1 | scene, illuminants, optics | `results/reports/g1_scene_optics.md` | Green locally; report PASS and the G1 tests pass |
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

## G1 — Scene and optics

**Status: green locally.** The full report is
`results/reports/g1_scene_optics.md`; its four figures are
`results/figures/g1_lux_normalization.png`, `g1_psf_mtf.png`,
`g1_energy_conservation.png`, and `g1_edge_angle.png`. The report is generated
deterministically by `benchmarks/g1_scene_optics.py`, and
`tests/test_validation_g1.py` is the executable form of the gate.

Reproduce from the repository root:

```bash
uv run python benchmarks/g1_scene_optics.py
uv run pytest tests/test_validation_g1.py -v
uv run pytest tests/test_validation_g1.py tests/test_units.py \
  tests/test_spectral.py tests/test_lens.py tests/test_psf.py \
  tests/test_targets.py tests/test_propagation.py tests/test_glyphs.py -v
uv run pytest -q
uv run ruff check . && uv run mypy src
```

Measured results against the G1 tolerances:

| Check | Measured | Tolerance | Result |
|---|---|---|---|
| Photometric normalization (`683 * integral E(lambda) V(lambda) dlambda`, dlambda in nm) | max relative error 3.64e-16 over D65/3000 K/4000 K and 1-100,000 lux | < 1e-3 | PASS |
| Gaussian PSF MTF50, sigma = 1 px | 0.187391 cycles/pixel | within 0.01 of 0.1874 | PASS |
| Airy first zero, reference f/1.8 lens | 1.07810 px | within 1e-3 of 1.078 px; kernel matches `x = pi r / (lambda N / pitch)` within 1e-12 | PASS |
| Energy through PSF + 4x box mean | max relative error 8.88e-16 | < 1e-2 | PASS |
| Slanted-edge angle recovery | max absolute error 0.0426 deg | < 0.5 deg | PASS |
| Spectral cube/unit contract and resampling | 7/7 checks behaved as required | reject mismatches; conserve energy | PASS |
| Reference lens geometry | 83.677 deg diagonal FOV; 99.733 px/deg at 1.12 um | 84 +/- 1 deg; 99.73 +/- 0.1 % | PASS |

The report records these assumptions and known limitations:

- the optics layer is a **supporting model, not a calibrated lens**: ideal
  Gaussian, Airy, and box kernels with no aberrations, wavelength-dependent
  PSF, polychromatic diffraction, or field-dependent blur;
- **ISO 12233-style e-SFR is deferred to G4**; G1 validates only the
  target/estimator edge-angle convention;
- **D65 endpoint extrapolation**: `colour-science` tabulates D65 to 780 nm and
  camsim holds that value constant to 830 nm; V(lambda) is about 1.5e-5 at
  780 nm, so the photometric effect is far below the 1e-3 tolerance;
- the **photometric integral covers 360-830 nm**, with V(lambda) zero outside
  the CIE table by contract;
- the **reference lens** uses scalar transmittance 0.9, no vignetting, and a
  83.68 deg field that rounds to the documented approximately 84 deg;
- **energy conservation** is a round-off-level statement for symmetric kernels
  with reflected borders; asymmetric kernels carry the documented 1 % budget;
- **determinism** is promised within one environment; PNG bytes can differ
  across Matplotlib/font builds, so cross-platform checks compare the reported
  numbers with tolerances.

## G2–G4 — Phase validation reports

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
