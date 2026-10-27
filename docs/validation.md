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
| G2 | 2 | sensor model and characterization | `results/reports/g2_sensor.md` | Green locally; report PASS and the G2 tests pass |
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

## G2 — Sensor model and EMVA characterization

**Status: green locally.** The full report is `results/reports/g2_sensor.md`,
its machine-readable results are `results/reports/g2_sensor_results.json`, and
its six figures are `results/figures/g2_qe_cfa_ir_cut.png`, `g2_ptc.png`,
`g2_snr_vs_irradiance.png`, `g2_dark_current_vs_temperature.png`,
`g2_lcg_hcg_comparison.png`, and `g2_dsnu_prnu_histograms.png`. The report is
generated by `examples/04_emva_characterization.py`;
`tests/test_preset_validation.py` is the executable form of the flagship G2
recovery gate.

Reproduce from the repository root:

```bash
uv run pytest tests/test_sensor.py tests/test_noise.py tests/test_defects.py \
  tests/test_gain.py tests/test_adc.py tests/test_characterization.py \
  tests/test_preset_validation.py -v
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run python examples/04_emva_characterization.py
```

Measured results against the G2 tolerances (32 frames per level, 10 ms
exposure, G2 LCG sweep with base seed 9; HCG low-to-saturation sweep with base
seed 3):

| Check | Measured | Tolerance | Result |
|---|---|---|---|
| LCG read noise | 2.0449 e- rms | 2.0 +/- 5 % | PASS |
| LCG DSNU | 0.4995 e- | 0.5 +/- 10 % | PASS |
| LCG PRNU | 0.4959 % | 0.5 +/- 10 % | PASS |
| LCG saturation capacity | 6026.6 e- | 6000 +/- 10 % | PASS |
| LCG linearity residual | 1.79e-5 | < 0.05 | PASS |
| LCG dynamic range | 69.39 dB | 30-90 dB | PASS |
| LCG system gain relative error | 0.476 % | < 2 % | PASS |
| HCG read noise | 0.7153 e- rms | 0.7 +/- 5 % | PASS |
| HCG DSNU | 0.4989 e- | 0.5 +/- 10 % | PASS |
| HCG PRNU | 0.5014 % | 0.5 +/- 10 % | PASS |
| HCG saturation capacity | 2000.9 e- | 2000 +/- 10 % | PASS |
| HCG linearity residual | 6.81e-5 | < 0.05 | PASS |
| HCG dynamic range | 68.94 dB | 30-90 dB | PASS |
| HCG system gain relative error | 0.073 % | < 2 % | PASS |
| Dark current at 65 C | 60.98 e-/s | 63.00 +/- 25 % | PASS |
| Dark current at 70 C | 100.39 e-/s | 103.37 +/- 25 % | PASS |

The report records these assumptions and known limitations:

- the characterization is **simulated and EMVA 1288-style**, not a certified
  measurement of a physical sensor;
- the flagship preset is a **representative engineering design point**, not a
  measurement of a specific commercial part; the QE/CFA/IR-cut curves are
  synthetic estimates bundled in `camsim.data`;
- each PTC uses **one conversion-gain mode**; `auto` is resolved once because a
  mixed-mode PTC has no single slope;
- flat-field illumination is uniform, temperature is constant within a
  characterization, and the ADC is ideal and linear;
- LCG/DSNU and HCG read noise are **sub-LSB** (about 0.24-0.26 DN), so the fit
  uses the exact quantized forward model rather than a fixed 1/12 DN²
  correction;
- the **eight-frame LCG sweep** in `tests/test_characterization.py` is a
  robustness check with a documented 20 % DSNU waiver; the 32-frame result is
  the G2 evidence;
- **dark-current points below 60 C** are differential measurements compressed
  by the zero-charge clamp and sub-LSB ADC, so only 65 C and 70 C are
  gate criteria; those agree with the doubling model within 3.2 %;
- **determinism** is promised within one environment; cross-platform runs
  compare tolerances, not bit patterns.

## G3–G4 — Phase validation reports

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
