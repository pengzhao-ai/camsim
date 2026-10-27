# camsim G5 validation report

**Status: awaiting human sign-off.**

This is the consolidated G5 candidate report for the camsim core pipeline. G1-G4 are green, every cross-cutting check below passed on the validated source tree, and the engineering gates are green. The G5 gate itself is **not** closed: the project owner must review this report and the figure gallery, confirm the scope, assumptions, and limitations, and explicitly approve `v0.1.0`. Until that decision is recorded, the status stays **awaiting human sign-off**, no tag exists, and no machine-vision code may be written.

## 1. Scope and architecture

camsim is a deterministic, from-scratch camera simulator with three clean layers. The **simulation pipeline** renders spectral scene irradiance through a lens model, integrates it into electrons in the sensor, and processes the RAW frame through a modular ISP; `scene` and `optics` are supporting helpers, while `sensor` and `isp` are the reusable core and never depend on them. The **metrics** layer is an independent harness that operates on arrays plus metadata and must not import the simulator; `tests/test_metrics_independence.py` enforces that rule. The **machine-vision** layer (`camsim.tasks`) is gated behind G5 and is an empty namespace at this commit.

```text
scene reflectance x illuminant x lens
        -> spectral irradiance      camsim.scene / camsim.optics
        -> RAW frame                camsim.Sensor.expose
        -> rendered image           camsim.ISP.process
        -> image-quality report     camsim.metrics.measure_all
```

The enforced ISP order is `DPC -> BLC -> LSC -> AWB -> demosaic -> CCM -> denoise -> sharpen -> tone -> output`. Every stage before tone operates in linear float32 light; the tone curve and output OETF are the only non-linear steps.

## 2. Environment and validated source

| Item | Value |
|---|---|
| Python | `3.11.8` |
| camsim | `0.1.0.dev0` |
| NumPy | `2.4.6` |
| SciPy | `1.17.1` |
| colour-science | `0.4.7` |
| colour-demosaicing | `0.2.7` |
| Matplotlib | `3.11.2` |
| Platform | `Darwin (arm64)` |
| Validated source commit | `596c674f7284042970b0831d52346a9ea3cb5162` |
| Branch | `feat/camsim-core` |
| Report date | `2026-10-27` |
| Sensor preset | `src/camsim/config/presets/sensors/flagship_phone_50mp.json` |
| ISP preset | `src/camsim/config/presets/isp/default.json` |
| Sensor config hash | `dc383811bd9307f5d864d917da8fc7ac918e2983c802dc900ad56463d9b3b591` |
| ISP config hash | `790c97d8f00b352fe31f2ffd26fd46e89c47caf308622db51d397308ed49a627` |

The phase reports and this candidate were produced from the same source tree and software environment. The G5 candidate commit adds only this report, its tests, the two G5 figures, and documentation and metadata updates; it changes no simulation, sensor, ISP, metrics, camera, or CLI code. All numbers are simulated measurements, not measurements of physical hardware.

## 3. Gate status

| Gate | Phase | Scope | Evidence | Measured / passed | Status |
|---|---|---|---|---|---|
| G0 | 0 | install, lint, typecheck, tests, docs | [docs/validation.md](../../docs/validation.md) | 9/9 commands | **PASS** |
| G1 | 1 | scene, illuminants, optics | [g1_scene_optics.md](g1_scene_optics.md) | 7/7 checks | **PASS** |
| G2 | 2 | sensor model and EMVA characterization | [g2_sensor.md](g2_sensor.md) + [g2_sensor_results.json](g2_sensor_results.json) | 16/16 checks | **PASS** |
| G3 | 3 | ISP stages and pipeline | [g3_isp.md](g3_isp.md) + [g3_isp_results.json](g3_isp_results.json) | 19/19 checks | **PASS** |
| G4 | 4 | metrics harness | [g4_metrics.md](g4_metrics.md) + [g4_metrics_results.json](g4_metrics_results.json) | 31/31 checks | **PASS** |
| G5 | 5 | consolidated cross-cutting evidence | this report | 13/13 checks | **Awaiting human sign-off** |

A phase gate is evidence only when its report says PASS and every recorded check in its machine-readable results passed. The loader that builds this table refuses quick-mode artifacts and any JSON not marked `gate_evidence: true`.

## 4. G1-G4 measured evidence

The tables below are the verdict tables from the committed phase reports and their JSON. The values were not re-typed by hand: G2-G4 are read from the JSON, and G1 is parsed from its committed Markdown verdict table. A stale report therefore cannot silently satisfy G5.

### G1 - scene, illuminants, optics

Report: [`g1_scene_optics.md`](g1_scene_optics.md)

| Check | Measured | Tolerance | Result |
|---|---:|---|---|
| Photometric normalization | max relative error 3.64e-16 | < 0.001 | PASS |
| Gaussian PSF MTF50 | 0.187391 cycles/pixel | within 0.01 of 0.1874 | PASS |
| Airy first zero, reference lens | 1.07810 px | within 1e-3 of 1.078 px; closed form within 1e-12 | PASS |
| PSF + 4x box energy | max relative error 8.88e-16 | < 0.01 | PASS |
| Slanted-edge angle | max \|error\| 0.0426 deg | < 0.5 deg | PASS |
| Spectral cube contract | 7/7 checks behaved as required | all | PASS |
| Reference lens | 83.677 deg, 99.733 px/deg | 84 +/- 1 deg; 99.73 +/- 0.1 % | PASS |

7 of 7 checks pass; reported status **PASS**.

### G2 - sensor model and EMVA characterization

Report: [`g2_sensor.md`](g2_sensor.md)
Machine-readable results: [`g2_sensor_results.json`](g2_sensor_results.json)

| Check | Measured | Tolerance | Result |
|---|---:|---|---|
| LCG read noise (e- rms) | 2.02639 | 2 +/- 5 % | PASS |
| LCG DSNU (e-) | 0.477467 | 0.5 +/- 10 % | PASS |
| LCG PRNU (%) | 0.499267 | 0.5 +/- 10 % | PASS |
| LCG saturation capacity (e-) | 5996.84 | 6000 +/- 10 % | PASS |
| LCG linearity residual | 4.186e-05 | <= 0.05 | PASS |
| LCG dynamic range (dB) | 69.424 | 30 < x < 90 | PASS |
| LCG system gain relative error | 0.0001792 | <= 0.02 | PASS |
| HCG read noise (e- rms) | 0.721109 | 0.7 +/- 5 % | PASS |
| HCG DSNU (e-) | 0.5031 | 0.5 +/- 10 % | PASS |
| HCG PRNU (%) | 0.497099 | 0.5 +/- 10 % | PASS |
| HCG saturation capacity (e-) | 2015.27 | 2000 +/- 10 % | PASS |
| HCG linearity residual | 7.328e-05 | <= 0.05 | PASS |
| HCG dynamic range (dB) | 68.9266 | 30 < x < 90 | PASS |
| HCG system gain relative error | 0.00787696 | <= 0.02 | PASS |
| dark current at 65 C vs doubling model (e-/s) | 60.6753 | 61.8018 +/- 5 % | PASS |
| dark current at 70 C vs doubling model (e-/s) | 99.894 | 102.165 +/- 5 % | PASS |

16 of 16 checks pass; reported status **PASS**.

### G3 - ISP stages and pipeline

Report: [`g3_isp.md`](g3_isp.md)
Machine-readable results: [`g3_isp_results.json`](g3_isp_results.json)

| Check | Measured | Tolerance | Result |
|---|---:|---|---|
| ColorChecker mean deltaE2000 (D65) | 0.918931 | < 2 | PASS |
| ColorChecker mean deltaE2000 (3000 K) | 3.83422 | < 4 | PASS |
| AWB gain relative error, R (D65) | 0.00464816 | < 0.05 | PASS |
| AWB gain relative error, B (D65) | 0.00443304 | < 0.05 | PASS |
| AWB gain relative error, R (3000 K) | 0.00496836 | < 0.05 | PASS |
| AWB gain relative error, B (3000 K) | 0.00517455 | < 0.05 | PASS |
| AWB injected-cast recovery (max relative error) | 0.0066654 | < 0.05 | PASS |
| Bilinear demosaic vs colour-demosaicing (max abs) | 1.192e-07 | < 1e-05 | PASS |
| MHC demosaic vs colour-demosaicing (max abs) | 2.161e-07 | < 0.0001 | PASS |
| sRGB OETF vs colour.cctf_encoding (max abs) | 2.977e-08 | < 1e-07 | PASS |
| Stage trace tone reproduction (max abs) | 0 | < 1e-06 | PASS |
| Stage trace linear_rgb reproduction (Display P3, max abs) | 0 | < 1e-06 | PASS |
| Linear pre-tone scaling invariance (max relative error) | 0 | < 1e-05 | PASS |
| LSC flat-field residual after correction | 0.00618314 | < 0.02 | PASS |
| Denoise SNR gain (dB) | 13.0556 | > 3 | PASS |
| Denoise edge-contrast preservation | 4.333e-05 | < 0.1 | PASS |
| Sharpen edge-contrast gain | 0.587229 | > 0 | PASS |
| Sharpen overshoot | 0.117446 | > 0 | PASS |
| RCCC baseline output channels | 3 | = 3 +/- 0 | PASS |

19 of 19 checks pass; reported status **PASS**.

### G4 - metrics harness

Report: [`g4_metrics.md`](g4_metrics.md)
Machine-readable results: [`g4_metrics_results.json`](g4_metrics_results.json)

| Check | Measured | Tolerance | Result |
|---|---:|---|---|
| MTF50 vs analytic Gaussian sigma=1 px (absolute error, cp) | 6.338e-05 | <= 0.00474782 | PASS |
| MTF20 vs analytic Gaussian sigma=1 px (absolute error, cp) | 0.0001023 | <= 0.00671124 | PASS |
| MTF50 angle spread, Gaussian (relative) | 0.00150161 | < 0.05 | PASS |
| MTF50 vs analytic Box width=4 px (absolute error, cp) | 0.0001755 | <= 0.00387929 | PASS |
| MTF20 vs analytic Box width=4 px (absolute error, cp) | 0.0004548 | <= 0.00500934 | PASS |
| MTF50 angle spread, Box (relative) | 0.0104204 | < 0.05 | PASS |
| MTF50 vs analytic Airy f/4 (absolute error, cp) | 0.0001569 | <= 0.00521839 | PASS |
| MTF20 vs analytic Airy f/4 (absolute error, cp) | 0.0002694 | <= 0.00804524 | PASS |
| MTF50 angle spread, Airy f/4 (relative) | 0.00492637 | < 0.05 | PASS |
| MTF50 vs analytic Airy f/8 (absolute error, cp) | 0.0001242 | <= 0.00311163 | PASS |
| MTF20 vs analytic Airy f/8 (absolute error, cp) | 5.557e-05 | <= 0.00452605 | PASS |
| MTF50 angle spread, Airy f/8 (relative) | 0.00151456 | < 0.05 | PASS |
| MTF50 noise robustness, Gaussian (relative change) | 0.0118359 | < 0.05 | PASS |
| MTF curve DC normalization (max abs error) | 0 | < 1e-09 | PASS |
| MTF curve range [0, 1] (max violation) | 0 | < 1e-09 | PASS |
| MTF curve monotonicity, Gaussian full band (max rise) | -0.0001296 | < 1e-09 | PASS |
| MTF curve monotonicity, box main lobe (max rise) | -0.0004265 | < 1e-09 | PASS |
| SNR spatial per-channel agreement (max abs error, dB) | 0.0835339 | < 0.5 | PASS |
| SNR spatial luma agreement (abs error, dB) | 0.0156268 | < 0.5 | PASS |
| SNR temporal vs known noise (abs error, dB) | 0.0179613 | < 0.5 | PASS |
| SNR temporal fixed-pattern rejection (temporal minus spatial, dB) | 14.0558 | > 5 | PASS |
| SNR mono path agreement (abs error, dB) | 0.009943 | < 0.5 | PASS |
| DeltaE76 vs colour-science (max abs error) | 0 | < 1e-09 | PASS |
| DeltaE2000 vs colour-science (max abs error) | 0 | < 1e-09 | PASS |
| DeltaE identity (max abs error) | 0 | < 1e-12 | PASS |
| Sharma CIEDE2000 published pairs (max abs error) | 4.95e-05 | < 0.0001 | PASS |
| measure_all RGB chart completeness (colour + SNR) | 1 | = 1 +/- 0 | PASS |
| measure_all edge layout completeness (MTF + acutance) | 1 | = 1 +/- 0 | PASS |
| measure_all MONO report (SNR without colour) | 1 | = 1 +/- 0 | PASS |
| measure_all JSON serialization (NaN -> null) | 1 | = 1 +/- 0 | PASS |
| metrics import independence (forbidden modules absent) | 1 | = 1 +/- 0 | PASS |

31 of 31 checks pass; reported status **PASS**.

## 5. Cross-cutting checks

These criteria span phases or the engineering process, so no single phase report owns them. The measurements are direct calls to the public stages and metrics; the quality-gate outcomes are the recorded outputs of the documented commands.

### 5.1 Quality gates

| Command | Recorded result | Wall time | Verdict |
|---|---|---:|---|
| `uv run pytest -q` | 1749 passed | 125.62 s | **PASS** |
| `uv run pytest --cov=camsim --cov-fail-under=90` | 1749 passed; total coverage 91.82% (required 90%) | 151.85 s | **PASS** |
| `uv run ruff check .` | All checks passed! | 0.06 s | **PASS** |
| `uv run ruff format --check .` | 140 files already formatted | 0.03 s | **PASS** |
| `uv run mypy src` | Success: no issues found in 68 source files | 0.59 s | **PASS** |
| `uv run mkdocs build --strict` | Documentation built in 1.67 seconds | 1.86 s | **PASS** |
| `uv run python benchmarks/benchmark_pipeline.py --quick` | pipeline total 8.25 ms (128 x 128) | 1.30 s | **PASS** |
| `uv run python benchmarks/benchmark_pipeline.py` | pipeline total 114.71 ms (512 x 512) | 1.86 s | **PASS** |
| `uv build` | built camsim-0.1.0.dev0.tar.gz and camsim-0.1.0.dev0-py3-none-any.whl | 0.59 s | **PASS** |

Absolute timings are machine-specific and are evidence, not guarantees. The coverage gate enforces at least 90 % statement and branch coverage with `*/camsim/viz/*` omitted: the presentation layer is exercised by the executed examples and notebooks that generate the committed gallery, while the simulation, metrics, configuration, characterization, and CLI modules remain under the threshold. The threshold is not lowered for this candidate.

### 5.2 Cross-cutting measurement summary

| Check | Measured | Bound | Verdict |
|---|---|---|---|
| Denoise MTF50 preservation | 0.189840 -> 0.217722 cycles/pixel (+14.69%) | loss <= 10% | **PASS** |
| Denoise luma SNR gain | 25.946 -> 36.558 dB (+10.61 dB) | > 0 dB | **PASS** |
| Sharpen MTF50 gain | 0.187328 -> 0.249291 cycles/pixel (+33.08%) | after > before | **PASS** |
| Sharpen positive overshoot | max normalized profile 1.002107 (overshoot +0.21% of edge contrast) | > 1 + 0.001 | **PASS** |
| Camera AE convergence | 2000 lux: 1 probe(s), metered 0.1930; 20 lux: 4 probe(s), metered 0.1634; 0.01 lux: 5 probe(s), metered 0.1786 | converged and \|metered - 0.18\| <= 0.02 | **PASS** |
| Camera AE physical limits | 2000 lux: 10.00 ms, 0.0 dB analog, 1.00x digital, LCG; 20 lux: 100.00 ms, 7.0 dB analog, 1.26x digital, HCG; 0.01 lux: 100.00 ms, 24.0 dB analog, 44.34x digital, HCG | exposure <= preset max, analog <= preset max, iterations <= 12 | **PASS** |
| RCCC baseline shape and constants | (32, 48, 3), R=0.250, G=0.600, B=0.600 | finite (H, W, 3); R = 0.25; G = B = 0.60 | **PASS** |
| RCCC baseline linearity | max \|0.5 x output - output(0.5 x input)\| = 0.000e+00 | <= 1e-6 | **PASS** |
| Determinism (same seeds, same environment) | RAW equal=True, rendered equal=True (57344 bytes compared) | bit-identical arrays | **PASS** |
| Metrics import independence | forbidden modules loaded: none | sensor, ISP, scene, and optics absent from sys.modules | **PASS** |
| CLI render -> measure chain | render rc=0, measure rc=0, MTF50=0.433778 cycles/pixel, acutance=0.0768167 | both commands exit 0; finite MTF50 and acutance | **PASS** |
| Quick-mode isolation | quick artifacts in results/reports: none; full-run flags: {'g3_isp_results.json': True, 'g4_metrics_results.json': True, 'g2_sensor_results.json': True} | no quick artifacts; every committed JSON is full-run gate evidence | **PASS** |
| Machine-vision code remains gated | files=['__init__.py'], executable statements=0, torch import=False | tasks package contains only its docstring | **PASS** |

Every non-timing cross-cutting measurement is deterministic: fixed seeds, explicit `numpy.random.Generator` instances, and no global RNG. The derivation and raw values follow.

### 5.3 Denoise: SNR gain and MTF50 preservation

A 256 x 256 slanted edge at 5 degrees is blurred by the known unit-sigma Gaussian and corrupted with additive Gaussian noise (sigma = 0.01). MTF50 is measured on the noisy edge and again after the shipped bilateral stage (`method=bilateral`, luma strength 1.0, radius 1.5); the SNR measurement uses an independent 64 x 64 flat patch with its own generator. The resolution bound is a **loss** bound: a denoiser that suppresses noise by smearing the edge fails, while a slightly higher measured MTF50 passes because no resolution was lost.

| Quantity | Measured |
|---|---:|
| MTF50 before denoise | 0.189840 cycles/pixel |
| MTF50 after denoise | 0.217722 cycles/pixel |
| Relative MTF50 change | +14.69% |
| SNR before denoise | 25.946 dB |
| SNR after denoise | 36.558 dB |
| SNR gain | +10.61 dB |

Interpretation: the bilateral stage raises flat-patch SNR and does not lose MTF50 on this edge. The measured MTF50 rises because the full-strength bilateral filter sharpens the transition as it suppresses the noise floor; the check is deliberately a preservation bound, and the measured change is reported with its sign rather than described as a pure equality.

### 5.4 Sharpen: MTF50 gain and overshoot

The same blurred edge without noise is sharpened with the shipped unsharp mask at amount 1.0, radius 1.0, threshold 0, luma only. MTF50 must increase, and the bright-side profile maximum must exceed the high plateau by more than the measured margin; the overshoot is reported both as a fraction of edge contrast and as the normalized profile maximum.

| Quantity | Measured |
|---|---:|
| MTF50 before sharpen | 0.187328 cycles/pixel |
| MTF50 after sharpen | 0.249291 cycles/pixel |
| Relative MTF50 gain | +33.08% |
| High plateau | 1.000000 |
| Profile maximum after sharpen | 1.002107 |
| Normalized profile maximum | 1.002107 |
| Overshoot fraction of edge contrast | +0.211% |

### 5.5 Camera auto-exposure

`Camera.auto_expose` probes the real sensor, meters a centre-weighted black-corrected fraction, and ramps exposure, analog gain, readout mode, and digital gain. The existing gate tests are `tests/test_camera.py::test_auto_exposure_converges` and `::test_auto_exposure_respects_limits`; the measurement below re-executes the policy on three 64 x 64 flat patches.

| Scene | Converged | Probes | Mode | Exposure (ms) | Analog (dB) | Digital | Metered |
|---|---|---|---|---:|---:|---:|---:|
| 2000 lux | True | 1 | LCG | 10.00 | 0.0 | 1.00x | 0.1930 |
| 20 lux | True | 4 | HCG | 100.00 | 7.0 | 1.26x | 0.1634 |
| 0.01 lux | True | 5 | HCG | 100.00 | 24.0 | 44.34x | 0.1786 |

Target fraction 0.18 with tolerance 0.02; all results converge within 12 probes and stay inside the preset's exposure and analog-gain windows. At 0.01 lux the policy uses HCG, the exposure ceiling, maximum analog gain, and substantial digital gain; metering succeeds, but digital gain amplifies noise, so that extreme-low-light image quality is not claimed.

### 5.6 RCCC baseline

RCCC is a monochrome-plus-red automotive layout. The documented baseline uses the clear channel as the luma placeholder for G and B; it is a shape/linearity/smoke reconstruction, not an advanced demosaic. The existing tests are `tests/test_isp_demosaic.py::test_rccc_baseline_shape_and_measured_sites`, `::test_rccc_baseline_preserves_dc_everywhere`, and `tests/test_example_02.py::test_g3_rccc_baseline_smoke`; the G3 qualitative render is `../figures/02_isp_rccc_baseline.png`.

| Quantity | Measured |
|---|---:|
| Output shape | `(32, 48, 3)` |
| Finite pixels | True |
| Constant reconstruction | True |
| Linear-scaling max error | 0.000e+00 |

**RCCC/Bayer caveat.** The Bayer ColorChecker delta-E thresholds and MTF gates do not apply to RCCC: its colour reconstruction is a placeholder by design, and advanced RCCC demosaicing is deferred. This evidence shows that the baseline is finite, shape-correct, constant-preserving, and linear, no more.

### 5.7 Determinism and import independence

- Determinism: two runs of the same seeded render -> expose -> ISP chain produce bit-identical RAW (`True`) and rendered (`True`) arrays.
- Import independence: a fresh interpreter imports `camsim.metrics` and finds none of `camsim.sensor`, `camsim.isp`, `camsim.scene`, `camsim.optics` in `sys.modules` (pass=True).
- CLI chain: `camsim render` then `camsim measure` exit 0 and 0 and produce MTF50 0.433778 cycles/pixel with acutance 0.0768167.

Determinism is promised within one environment (same OS, Python, NumPy, and SciPy). Cross-platform comparisons use tolerances because floating-point reduction order and Matplotlib/font builds can differ; README and documentation figures should be compared numerically, not bit-for-bit, across platforms.

### 5.8 Quick-mode isolation and machine-vision scope

- Quick-mode isolation: quick artifacts in `results/reports` = `[]`; committed full-run flags = `{'g3_isp_results.json': True, 'g4_metrics_results.json': True, 'g2_sensor_results.json': True}`.
- Machine-vision scope: `camsim.tasks` modules = `['__init__.py']`, executable statements = 0, torch import = False.

Quick modes (`--quick`) use smaller grids and write suffixed `*_quick.json` artifacts; they are smoke tests and never gate evidence. No machine-vision code exists before the G5 sign-off, so this report makes no task-accuracy, CNN, or metric-correlation claim.

## 6. Assumptions

1. **Simulated sensor and preset.** The flagship preset is a representative engineering design point (LCG 100 uV/e-, 6000 e- full well, 2 e- rms read noise; HCG 300 uV/e-, 2000 e-, 0.7 e-; 10-bit ADC, 0.8 V full scale, 64 DN black level). It is not a measurement of, or a claim about, a specific commercial part.
2. **Provisional spectral data.** The QE, CFA, IR-cut, illuminant, and ColorChecker curves are bundled synthetic engineering estimates with recorded provenance, not digitized vendor or measured data. They are adequate to validate the physics and interfaces, not to predict a particular product's colour response.
3. **In-sample colour calibration.** The 3x3 CCM is fitted and evaluated on the same 24 ColorChecker patches. This measures whether the ISP can represent the calibration scene, not held-out generalisation, colour constancy, or performance under an unseen illuminant.
4. **ISO 12233-style, not certified.** The slanted-edge e-SFR follows the standard's principles (5 degree tilt, ESF derivative, Hamming window, native-Nyquist band) but makes no claim of conformance to its chart, windowing, or reporting details. The EMVA 1288-style sensor characterization is likewise simulated and uncertified.
5. **Determinism within one environment.** Every stochastic path takes an explicit `numpy.random.Generator` and a recorded seed. Bit-identical outputs are promised for one OS/Python/NumPy/SciPy environment; cross-platform comparisons use the stated tolerances.

## 7. Limitations

1. **Synthetic CFA re-authored during G3.** The original broad, nearly collinear CFA curves left a 45.6 % Luther-condition residual and no 3x3 matrix could meet the ColorChecker threshold. The shipped preset uses conventional Bayer bands (blue ~448 nm, green ~542 nm, red ~610 nm) with moderate out-of-band leakage, reducing the residual to 12.6 %. The delta-E thresholds (2.0 / 4.0) were not relaxed.
2. **In-sample calibration.** The measured mean delta-E values (0.919 under D65 and 3.834 under 3000 K) are in-sample; the 3000 K mean is 4.14 % below its limit and per-patch maxima reach 13.208. Held-out chart, illuminant-generalisation, and colour-constancy performance are not claimed.
3. **8x Airy MTF gate.** The G4 analytic MTF cases use eight analysis samples per pixel because four carries roughly a 2 % MTF50 bias near the diffraction limit. Airy f/2.8 is a documented diagnostic only: its MTF never reaches 0.20 inside Nyquist and its crossing sits at the band edge.
4. **Synthetic LSC, denoise, and sharpen measurements.** Lens shading is measured on a synthetic radial flat field, denoise on a synthetic noisy step edge and flat patch, and sharpen on a synthetic blurred step edge. These isolate the stages; they are not camera captures and do not include real lens shading, sensor texture, or scene content.
5. **Quick modes are not gate evidence.** `--quick` paths use smaller grids, write suffixed quick artifacts, and never overwrite the committed full-run JSON or Markdown. The committed G2-G4 JSON is full-run evidence (`gate_evidence: true`, `quick: false`).
6. **Auto-exposure digital-gain noise.** At 0.01 lux auto-exposure needs HCG, the exposure ceiling, maximum analog gain, and about 44x digital gain. Metering converges, but digital gain amplifies sensor noise and cannot create signal; extreme-low-light image quality is not claimed.
7. **No HDR, 3-D, motion blur, or autofocus.** The model has a single linear exposure, a planar scene, static optics, and no focus search, rolling shutter, or multi-frame HDR. These are Phase 6-9 or later scope, not G5 claims.
8. **RCCC is a baseline only.** The clear channel is a luma placeholder for G and B; Bayer colour and MTF thresholds do not apply, and advanced RCCC demosaicing is deferred.
9. **No machine-vision evidence.** `camsim.tasks` is an empty gated namespace. This report says nothing about CNN accuracy, task-relevant IQ, mono-versus-Bayer, auto-tuning, or metric correlation.

## 8. Reproduction

Run from the repository root with `uv` and all extras installed:

```bash
uv sync --all-extras

# Phase artifacts (regenerate only when the phase code changes)
uv run python benchmarks/g1_scene_optics.py
uv run python examples/04_emva_characterization.py
uv run python examples/02_isp_stage_gallery.py
uv run python examples/03_measure_iq.py

# Gallery: examples 01-04 full mode, then notebooks 01-03 headless
uv run python examples/01_spectral_scene_to_raw.py
uv run python examples/02_isp_stage_gallery.py
uv run python examples/03_measure_iq.py
uv run python examples/04_emva_characterization.py
make notebooks

# Consolidated G5 report and figures
uv run python benchmarks/g5_validation_report.py

# G5 quality gates
uv run pytest -q
uv run pytest --cov=camsim --cov-fail-under=90
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run mkdocs build --strict
uv run python benchmarks/benchmark_pipeline.py --quick
uv run python benchmarks/benchmark_pipeline.py
uv build
```

`make figures` runs examples 01-04 and is the gallery regeneration target; `make notebooks` executes every Jupytext notebook into `results/cache/notebooks` under a two-minute-per-notebook budget. Neither command tags, pushes, or publishes anything.

## 9. Figure gallery

| Figure | Shows |
|---|---|
| [g5_gate_summary.png](../figures/g5_gate_summary.png) | Gate/check status matrix; G1-G4 PASS, G5 recorded checks pass, human sign-off awaiting |
| [g5_evidence.png](../figures/g5_evidence.png) | Denoise SNR/MTF, sharpen profile/overshoot, auto-exposure convergence, and benchmark stage shares |
| [g1_lux_normalization.png](../figures/g1_lux_normalization.png) | Photometric normalization across illuminants and 1-100,000 lux |
| [g1_psf_mtf.png](../figures/g1_psf_mtf.png) | Gaussian, Airy, and box PSFs with their transfer functions |
| [g2_ptc.png](../figures/g2_ptc.png) | LCG/HCG photon-transfer curves and system gain |
| [g2_snr_vs_irradiance.png](../figures/g2_snr_vs_irradiance.png) | Sensor SNR across the shot- and read-noise regimes |
| [g3_check_summary.png](../figures/g3_check_summary.png) | G3 check summary with tolerances |
| [g3_delta_e.png](../figures/g3_delta_e.png) | Per-patch ColorChecker delta-E2000 under D65 and 3000 K |
| [02_isp_rccc_baseline.png](../figures/02_isp_rccc_baseline.png) | RCCC baseline qualitative render |
| [g4_check_summary.png](../figures/g4_check_summary.png) | G4 check summary against analytic references |
| [g4_mtf_curves.png](../figures/g4_mtf_curves.png) | Measured MTF against analytic PSF references |

The two G5 figures are also mirrored into `docs/images/` (`g5_gate_summary.png, g5_evidence.png`) and verified byte-identical so the documentation site stays self-contained.

## 10. Human sign-off checklist

G5 is a human decision, not an automated verdict. The owner must review the evidence below and record an explicit decision. Checking these boxes is the sign-off; until then the status line remains **awaiting human sign-off**.

- [ ] **Report reviewed.** Read this report end to end, including the G1-G4 verdict tables and the cross-cutting measurements.
- [ ] **Figures reviewed.** Inspect the figure gallery, especially `g5_gate_summary.png`, `g5_evidence.png`, the G3 ColorChecker delta-E figure, the RCCC baseline, and the G4 analytic MTF curves.
- [ ] **Scope confirmed.** Agree that the validated scope is the scene/optics helpers, sensor, ISP, metrics, camera orchestration, IO, and CLI, and that machine vision is out of scope for v0.1.0.
- [ ] **Assumptions confirmed.** Accept the simulated flagship preset, provisional synthetic spectral data, in-sample CCM, and the uncertified ISO 12233/EMVA-style measurement status.
- [ ] **Limitations confirmed.** Accept the synthetic CFA re-authoring, in-sample calibration, 8x Airy diagnostic, synthetic stage measurements, quick-mode exclusion, extreme-low-light digital-gain noise, and the absence of HDR/3-D/motion-blur/autofocus.
- [ ] **Release approved.** Approve tagging `v0.1.0` on the signed-off commit and starting Phase 6 machine-vision work afterwards.

```text
Owner: ______________________________
Date:  ______________________________
Decision:  [ ] approve v0.1.0     [ ] withhold (return to the failing gate)
Notes:   ______________________________
```

**This report does not create the `v0.1.0` tag, push a branch, or publish to PyPI.** Those actions require the recorded approval above and are performed in a separate release step. No machine-vision code may be written before that approval.

## Appendix A. Cross-cutting detail values

### denoise

| Field | Value |
|---|---|
| config | `method=bilateral, luma_strength=1, chroma_strength=0, radius=1.5, edge_noise_sigma=0.01, edge_seed=71, flat_noise_sigma=0.02, flat_seed=72, oversampling=8` |
| mtf50_before_cp | `0.18984` |
| mtf50_after_cp | `0.217722` |
| mtf50_relative_change | `0.146872` |
| snr_before_db | `25.9456` |
| snr_after_db | `36.5583` |
| snr_gain_db | `10.6127` |

### sharpen

| Field | Value |
|---|---|
| config | `amount=1, radius=1, threshold=0, luma_only=true` |
| mtf50_before_cp | `0.187328` |
| mtf50_after_cp | `0.249291` |
| mtf50_relative_gain | `0.330776` |
| profile_low_plateau | `4.788e-17` |
| profile_high_plateau | `1` |
| profile_max_after | `1.00211` |
| normalized_max | `1.00211` |
| overshoot_fraction | `0.00210719` |

### rccc

| Field | Value |
|---|---|
| shape | `[32, 48, 3]` |
| finite | `true` |
| constant_reconstruction | `true` |
| linearity_max_error | `0` |

### determinism

| Field | Value |
|---|---|
| raw_equal | `true` |
| image_equal | `true` |
| raw_shape | `[64, 64]` |
| image_shape | `[64, 64, 3]` |

### import_independence

| Field | Value |
|---|---|
| passed | `true` |
| forbidden | `[camsim.sensor, camsim.isp, camsim.scene, camsim.optics]` |

### cli_chain

| Field | Value |
|---|---|
| render_returncode | `0` |
| measure_returncode | `0` |
| mtf50_cp | `0.433778` |
| acutance | `0.0768167` |

### quick_isolation

| Field | Value |
|---|---|
| quick_files | `[]` |
| full_flags | `g3_isp_results.json=true, g4_metrics_results.json=true, g2_sensor_results.json=true` |

### machine_vision_scope

| Field | Value |
|---|---|
| files | `[__init__.py]` |
| executable_statements | `0` |
| imports_torch | `false` |

