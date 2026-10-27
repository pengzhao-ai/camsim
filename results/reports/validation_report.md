# camsim G5 validation report

**Status: awaiting human sign-off.**

This is the consolidated G5 candidate report for the camsim core pipeline. G1-G4 are green, every cross-cutting check below passed on the validated source tree, and the local engineering gates are green. CI workflows for lint, types, tests, coverage, docs, build, examples, and notebooks were added in this candidate, but their execution is pending a push; the owner must accept that residual at sign-off. The G5 gate itself is **not** closed: the project owner must review this report and the figure gallery, confirm the scope, assumptions, and limitations, and explicitly approve `v0.1.0`. Until that decision is recorded, the status stays **awaiting human sign-off**, no tag exists, and no machine-vision code may be written.

## 1. Scope and architecture

camsim is a deterministic, from-scratch camera simulator with three clean layers. The **simulation pipeline** renders spectral scene irradiance through a lens model, integrates it into electrons in the sensor, and processes the RAW frame through a modular ISP; `scene` and `optics` are supporting helpers, while `sensor` and `isp` are the reusable core and never depend on them. The **metrics** layer is an independent harness that operates on arrays plus metadata and must not import the simulator; `tests/test_metrics_independence.py` enforces that rule. The **machine-vision** layer (`camsim.tasks`) is gated behind G5 and is an empty namespace at this commit.

```text
scene reflectance x illuminant x lens
        -> spectral irradiance      camsim.scene / camsim.optics
        -> RAW frame                camsim.Sensor.expose
        -> rendered image           camsim.ISP.process
        -> image-quality report     camsim.metrics.measure_all
```

The enforced ISP order is `DPC -> BLC -> LSC -> AWB -> demosaic -> CCM -> denoise -> sharpen -> tone -> output`. Every stage before tone operates in linear float32 light. The CCM's non-negativity clamp is the one bounded exception: it is inactive on in-gamut data and is measured explicitly in G3. The tone curve and output OETF are the only intentionally non-linear steps.

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
| Validated code commit | `d5e6abb7a88b9a86423e79849ff45c69d96053dc` |
| Tag target | the commit containing this report (`HEAD` at sign-off) |
| Branch | `feat/camsim-core` |
| Report date | `2026-10-27` |
| Sensor preset | `src/camsim/config/presets/sensors/flagship_phone_50mp.json` |
| ISP preset | `src/camsim/config/presets/isp/default.json` |
| Sensor config hash | `dc383811bd9307f5d864d917da8fc7ac918e2983c802dc900ad56463d9b3b591` |
| ISP config hash | `790c97d8f00b352fe31f2ffd26fd46e89c47caf308622db51d397308ed49a627` |

The code commit validated by this report is `d5e6abb`; the report itself, its tests, and the documentation corrections are committed alongside it, and the commit containing this report is the intended `v0.1.0` tag target. The report was generated from the working tree that became `d5e6abb`; later changes to this file are report, test, and documentation corrections only. The phase reports and this candidate were produced from the same source tree and software environment. This candidate includes the release-blocking fixes from the whole-branch review: physical dead/stuck/hot defect behaviour and a separate hot mask, Display P3 luma conversion, an AWB+CCM linearity check, a two-sided denoise MTF bound with an edge-ringing check, a simulation-memory guard, and the CI/notebook runners. G3 evidence was regenerated; G1, G2, and G4 remained byte-identical. All numbers are simulated measurements, not measurements of physical hardware.

## 3. Gate status

| Gate | Phase | Scope | Evidence | Measured / passed | Status |
|---|---|---|---|---|---|
| G0 | 0 | install, lint, typecheck, tests, docs | [docs/validation.md](../../docs/validation.md) | 11/11 commands | **PASS** (local) |
| G1 | 1 | scene, illuminants, optics | [g1_scene_optics.md](g1_scene_optics.md) | 7/7 checks | **PASS** |
| G2 | 2 | sensor model and EMVA characterization | [g2_sensor.md](g2_sensor.md) + [g2_sensor_results.json](g2_sensor_results.json) | 16/16 checks | **PASS** |
| G3 | 3 | ISP stages and pipeline | [g3_isp.md](g3_isp.md) + [g3_isp_results.json](g3_isp_results.json) | 21/21 checks | **PASS** |
| G4 | 4 | metrics harness | [g4_metrics.md](g4_metrics.md) + [g4_metrics_results.json](g4_metrics_results.json) | 31/31 checks | **PASS** |
| G5 | 5 | consolidated cross-cutting evidence | this report | 16/16 checks | **Awaiting human sign-off** |

A phase gate is evidence only when its report says PASS and every recorded check in its machine-readable results passed. The loader that builds this table refuses quick-mode artifacts and any JSON not marked `gate_evidence: true`. G0 is a local gate run. The CI workflow now defines quality, test, docs, build, examples, and notebooks jobs, but GitHub Actions cannot be executed from this environment: the workflows are added, the local equivalents passed, and CI execution is pending a push. The owner must accept that residual at sign-off. No claim that CI is green is made here.

### 3.1 Phase artifact bindings

Every phase artifact this report reads is bound by SHA-256. The G5 test suite recomputes these hashes and fails if an artifact changes without this report being regenerated, so the consolidated tables and the committed evidence cannot silently diverge. The binding proves that the report describes the committed artifacts; it does **not** prove that those artifacts were regenerated from this exact commit, and no such proof is claimed.

| Artifact | SHA-256 |
|---|---|
| `g1_scene_optics.md` | `9d57875d370aef49702234c33ba7d7872b0332fa7e69fc8b21a4b9065fda8140` |
| `g2_sensor.md` | `324009aa10394db7a8842ceff0b3d92d17443b5c062d264585d2fdc788419692` |
| `g2_sensor_results.json` | `aacbf03da1659b238b32cdbd4930563c88412d03fde0dd09d0acd8872feae796` |
| `g3_isp.md` | `c8e33bf06550f350298c124f3512e8521eb5b79d5a39460c2790eab9f44cbdef` |
| `g3_isp_results.json` | `497703fce429fc1cb841a383a31ef64fb521bde512535f413704b267830be03f` |
| `g4_metrics.md` | `2a7b33233f34f3ebf9365259c532957520dbf07b5f30b7b58bdb354915a40127` |
| `g4_metrics_results.json` | `24de5b0f85a20790ead3d25a0e01417355f7f95e426910371cbf5305b68601bd` |

**Regeneration record for this candidate.** `benchmarks/g1_scene_optics.py`, examples 01-04 in full mode, and notebooks 01-03 headless were re-run on the validated source tree. G1 remained PASS and its report and four figures were byte-identical; G2 and G4 reports and JSON were byte-identical; the example and notebook gallery figures were byte-identical except the G3 check-summary figure, which reflects the added AWB/CCM linearity checks. G3 was regenerated because its linearity check now runs with AWB enabled and a fitted CCM, and its report wording was corrected. G1-G4 each reported PASS after regeneration. The preset's defect rates (1e-5 dead, 1e-5 hot, 1e-6 stuck) put fewer than one expected defect in the G2 characterization fields, so G2 was byte-identical after the defect behaviour change; the end-to-end defect tests use raised custom rates. Quick modes were not run against the gate directories. The per-item example and notebook runtimes are recorded in section 5.9.

## 4. G1-G4 measured evidence

The tables below are the verdict tables from the committed phase reports and their JSON. The values were not re-typed by hand: G2-G4 are read from the JSON, and G1 is parsed from its committed Markdown verdict table. The hash bindings in section 3.1 tie this report to those exact artifacts. They verify full-run PASS evidence and artifact/report consistency; they do not by themselves prove that the artifacts were regenerated from the current commit. The regeneration record in section 3.1 states what was actually re-run for this candidate.

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
| AWB gain scale invariance (max relative error) | 0 | < 0.001 | PASS |
| Linear pre-tone scaling invariance with AWB and CCM (max relative error) | 0 | < 1e-05 | PASS |
| CCM pre-clamp minimum on linearity data | 0.0248324 | > 0 | PASS |
| LSC flat-field residual after correction | 0.00618314 | < 0.02 | PASS |
| Denoise SNR gain (dB) | 13.0556 | > 3 | PASS |
| Denoise edge-contrast preservation | 4.333e-05 | < 0.1 | PASS |
| Sharpen edge-contrast gain | 0.587229 | > 0 | PASS |
| Sharpen overshoot | 0.117446 | > 0 | PASS |
| RCCC baseline output channels | 3 | = 3 +/- 0 | PASS |

21 of 21 checks pass; reported status **PASS**.

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
| `uv run pytest -q` | 1775 passed | 126.16 s | **PASS** |
| `uv run pytest --cov=camsim --cov-fail-under=90` | 1775 passed; subject-module coverage 91.73% (required 90%) | 150.48 s | **PASS** |
| `uv run pytest --cov=camsim --cov-config=coverage-whole.rc` | 1775 passed; whole-package coverage 83.55% including viz | 150.17 s | **PASS** |
| `uv run ruff check .` | All checks passed! | 0.06 s | **PASS** |
| `uv run ruff format --check .` | 143 files already formatted | 0.03 s | **PASS** |
| `uv run mypy src` | Success: no issues found in 69 source files | 0.59 s | **PASS** |
| `uv run mkdocs build --strict` | Documentation built in 1.78 seconds | 1.97 s | **PASS** |
| `uv run python benchmarks/benchmark_pipeline.py --quick` | pipeline total 8.34 ms (128 x 128) | 1.31 s | **PASS** |
| `uv run python benchmarks/benchmark_pipeline.py` | pipeline total 114.58 ms (512 x 512) | 1.87 s | **PASS** |
| `uv run python benchmarks/benchmark_pipeline.py --large` | pipeline total 453.90 ms (1024 x 1024, 1x oversampling) | 3.12 s | **PASS** |
| `uv build` | built camsim-0.1.0.dev0.tar.gz and camsim-0.1.0.dev0-py3-none-any.whl | 0.77 s | **PASS** |

Absolute timings are machine-specific and are evidence, not guarantees. Coverage is reported with two scopes, not one headline number: **subject-module coverage is 91.73 %** with `*/camsim/viz/*` omitted, and **whole-package coverage including viz is 83.55 %**. The 90 % statement-and-branch gate applies to the subject modules (simulation, metrics, configuration, characterization, IO, camera, and CLI), which contain the numerical contracts the gates depend on. The presentation layer is exercised by the executed examples and notebooks that generate the committed gallery; its inclusion lowers the whole-package number. That gap is an explicit **reviewed exception** recorded in the sign-off checklist, not a silent claim that 91.73 % is total coverage.

### 5.2 Cross-cutting measurement summary

| Check | Measured | Bound | Verdict |
|---|---|---|---|
| Examples headless (local) | 01_spectral_scene_to_raw.py: 2.07 s; 02_isp_stage_gallery.py: 2.44 s; 03_measure_iq.py: 2.57 s; 04_emva_characterization.py: 14.62 s | all exit 0; each item <= 300 s | **PASS** |
| Notebooks headless (local) | 01_optics_and_sensor.py: 14.82 s; 02_isp_color_fundamentals.py: 2.85 s; 03_measuring_iq.py: 2.57 s | all exit 0; each item <= 120 s | **PASS** |
| Denoise MTF50 change | 0.189840 -> 0.217722 cycles/pixel (+14.69%) | \|change\| <= 20% | **PASS** |
| Denoise edge ringing (overshoot/undershoot) | overshoot +0.07%, undershoot +0.07% of edge contrast | \|overshoot\| and \|undershoot\| <= 1% | **PASS** |
| Denoise luma SNR gain | 25.946 -> 36.558 dB (+10.61 dB) | > 0 dB | **PASS** |
| Sharpen MTF50 gain | 0.187328 -> 0.249291 cycles/pixel (+33.08%) | after > before | **PASS** |
| Sharpen positive overshoot | max normalized profile 1.002107 (overshoot +0.21% of edge contrast) | > 1 + 0.001 | **PASS** |
| Camera AE convergence | 2000 lux: 1 probe(s), metered 0.1930; 20 lux: 4 probe(s), metered 0.1634; 0.01 lux: 5 probe(s), metered 0.1786 | converged and \|metered - 0.18\| <= 0.02 | **PASS** |
| Camera AE physical limits | 2000 lux: 10.00 ms, 0.0 dB analog, 1.00x digital, LCG; 20 lux: 100.00 ms, 7.0 dB analog, 1.26x digital, HCG; 0.01 lux: 100.00 ms, 24.0 dB analog, 44.34x digital, HCG | exposure within preset min/max, analog within preset min/max, digital >= 1x, iterations <= 12, mode LCG/HCG | **PASS** |
| RCCC baseline shape and constants | (32, 48, 3), R=0.250, G=0.600, B=0.600 | finite (H, W, 3); R = 0.25; G = B = 0.60 | **PASS** |
| RCCC baseline linearity | max \|0.5 x output - output(0.5 x input)\| = 0.000e+00 | <= 1e-6 | **PASS** |
| Determinism (same seeds, same environment) | RAW equal=True, rendered equal=True (57344 bytes compared) | bit-identical arrays | **PASS** |
| Metrics import independence | forbidden modules loaded: none | sensor, ISP, scene, and optics absent from sys.modules | **PASS** |
| CLI render -> measure chain | render rc=0, measure rc=0, MTF50=0.433778 cycles/pixel, acutance=0.0768167 | both commands exit 0; finite MTF50 and acutance | **PASS** |
| Quick-mode isolation | quick artifacts in results/reports: none; full-run flags: {'g3_isp_results.json': True, 'g4_metrics_results.json': True, 'g2_sensor_results.json': True} | no quick artifacts; every committed JSON is full-run gate evidence | **PASS** |
| Machine-vision code remains gated | files=['__init__.py'], executable statements=0, torch import=False | tasks package contains only its docstring | **PASS** |

Every non-timing cross-cutting measurement is deterministic: fixed seeds, explicit `numpy.random.Generator` instances, and no global RNG. The derivation and raw values follow.

### 5.3 Denoise: SNR gain, MTF50 change, and edge ringing

A 256 x 256 slanted edge at 5 degrees is blurred by the known unit-sigma Gaussian and corrupted with additive Gaussian noise (sigma = 0.01). MTF50 is measured on the noisy edge and again after the shipped bilateral stage (`method=bilateral`, luma strength 1.0, radius 1.5); the SNR measurement uses an independent 64 x 64 flat patch with its own generator. The MTF50 criterion is **two-sided**: |relative change| <= 20%. A denoiser that smears the edge or invents contrast both fail. The same denoised profile must also stay within 1% of the edge contrast on both the bright and dark sides, so a ringing halo cannot pass as sharpening.

| Quantity | Measured |
|---|---:|
| MTF50 before denoise | 0.189840 cycles/pixel |
| MTF50 after denoise | 0.217722 cycles/pixel |
| Signed relative MTF50 change | +14.69% |
| Edge overshoot | +0.072% of contrast |
| Edge undershoot | +0.065% of contrast |
| SNR before denoise | 25.946 dB |
| SNR after denoise | 36.558 dB |
| SNR gain | +10.61 dB |

Interpretation: the bilateral stage raises flat-patch SNR. The measured MTF50 rises because the full-strength bilateral filter steepens the transition as it suppresses the noise floor; that rise is inside the stated two-sided bound and is reported with its sign. The edge profile shows no meaningful ringing: overshoot and undershoot are well below the 1 % bound.

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

### 5.9 Examples and notebooks (local gallery evidence)

CI workflows now include `examples` and `notebooks` jobs. This environment cannot execute GitHub Actions, so the equivalent local sweeps were run through `benchmarks/run_examples.py` and `benchmarks/run_notebooks.py`, which enforce per-item subprocess timeouts and fail fast. **CI execution is pending a push; local equivalents passed.** The owner must accept that residual at sign-off. The two-minute-per-notebook condition is enforced by the 120 s runner timeout and verified by the times below.

| Item | Sweep | Wall time | Budget | Verdict |
|---|---|---:|---:|---|
| `01_spectral_scene_to_raw.py` | example | 2.07 s | 300 s | **PASS** |
| `02_isp_stage_gallery.py` | example | 2.44 s | 300 s | **PASS** |
| `03_measure_iq.py` | example | 2.57 s | 300 s | **PASS** |
| `04_emva_characterization.py` | example | 14.62 s | 300 s | **PASS** |
| `01_optics_and_sensor.py` | notebook | 14.82 s | 120 s | **PASS** |
| `02_isp_color_fundamentals.py` | notebook | 2.85 s | 120 s | **PASS** |
| `03_measuring_iq.py` | notebook | 2.57 s | 120 s | **PASS** |

The timing JSON files are committed as local evidence: `results/reports/examples_timings.json` and `results/reports/notebooks_timings.json`. They are machine-specific wall times, not cross-platform promises.

## 6. Assumptions

1. **Simulated sensor and preset.** The flagship preset is a representative engineering design point (LCG 100 uV/e-, 6000 e- full well, 2 e- rms read noise; HCG 300 uV/e-, 2000 e-, 0.7 e-; 10-bit ADC, 0.8 V full scale, 64 DN black level). It is not a measurement of, or a claim about, a specific commercial part.
2. **Provisional spectral data.** The QE, CFA, IR-cut, illuminant, and ColorChecker curves are bundled synthetic engineering estimates with recorded provenance, not digitized vendor or measured data. They are adequate to validate the physics and interfaces, not to predict a particular product's colour response.
3. **In-sample colour calibration.** The 3x3 CCM is fitted and evaluated on the same 24 ColorChecker patches. This measures whether the ISP can represent the calibration scene, not held-out generalisation, colour constancy, or performance under an unseen illuminant.
4. **ISO 12233-style, not certified.** The slanted-edge e-SFR follows the standard's principles (5 degree tilt, ESF derivative, Hamming window, native-Nyquist band) but makes no claim of conformance to its chart, windowing, or reporting details. The EMVA 1288-style sensor characterization is likewise simulated and uncertified.
5. **Determinism within one environment.** Every stochastic path takes an explicit `numpy.random.Generator` and a recorded seed. Bit-identical outputs are promised for one OS/Python/NumPy/SciPy environment; cross-platform comparisons use the stated tolerances.
6. **Default ISP is an identity CCM baseline.** The shipped preset's `ccm.matrix` is the identity. It preserves a neutral baseline and lets the stage measurements isolate optics and sensor effects, but it is not a colour-accurate profile. A colour-accurate render must fit a matrix with `camsim.isp.ccm.fit_ccm` and supply it in `ccm.matrix`.
7. **Defect conventions are explicit.** Dead pixels have zero mean photoresponse, stuck pixels are pinned high to the ADC white code, and hot pixels keep a dark-current multiplier with a separate hot mask. Stuck-black is equally physical but not the chosen deterministic convention.

## 7. Limitations

1. **Synthetic CFA re-authored during G3.** The original broad, nearly collinear CFA curves left a 45.6 % Luther-condition residual and no 3x3 matrix could meet the ColorChecker threshold. The shipped preset uses conventional Bayer bands (blue ~448 nm, green ~542 nm, red ~610 nm) with moderate out-of-band leakage, reducing the residual to 12.6 %. The delta-E thresholds (2.0 / 4.0) were not relaxed.
2. **In-sample calibration.** The measured mean delta-E values (0.919 under D65 and 3.834 under 3000 K) are in-sample; the 3000 K mean is 4.14 % below its limit and per-patch maxima reach 13.208. Held-out chart, illuminant-generalisation, and colour-constancy performance are not claimed.
3. **8x Airy MTF gate.** The G4 analytic MTF cases use eight analysis samples per pixel because four carries roughly a 2 % MTF50 bias near the diffraction limit. Airy f/2.8 is a documented diagnostic only: its MTF never reaches 0.20 inside Nyquist and its crossing sits at the band edge.
4. **Synthetic LSC, denoise, and sharpen measurements.** Lens shading is measured on a synthetic radial flat field, denoise on a synthetic noisy step edge and flat patch, and sharpen on a synthetic blurred step edge. These isolate the stages; they are not camera captures and do not include real lens shading, sensor texture, or scene content.
5. **Quick modes are not gate evidence.** `--quick` paths use smaller grids, write suffixed quick artifacts, and never overwrite the committed full-run JSON or Markdown. The committed G2 JSON records `gate_evidence: true` (G2 has no quick-mode payload); G3 and G4 additionally record `quick: false`.
6. **Auto-exposure digital-gain noise.** At 0.01 lux auto-exposure needs HCG, the exposure ceiling, maximum analog gain, and about 44x digital gain. Metering converges, but digital gain amplifies sensor noise and cannot create signal; extreme-low-light image quality is not claimed.
7. **No HDR, 3-D, motion blur, or autofocus.** The model has a single linear exposure, a planar scene, static optics, and no focus search, rolling shutter, or multi-frame HDR. These are Phase 6-9 or later scope, not G5 claims.
8. **RCCC is a baseline only.** The clear channel is a luma placeholder for G and B; Bayer colour and MTF thresholds do not apply, and advanced RCCC demosaicing is deferred.
9. **Identity default CCM is not colour accurate.** The default preset renders a ColorChecker at roughly 40 mean delta-E2000 at its default 10 ms exposure (measured 37.6 at 300 lux), about 11 when well exposed, and about 16 under 3000 K. The G3 values near 1-4 come from a per-illuminant fitted matrix, not from the shipped preset. `measure_all` records an identity-CCM warning in its metadata when colour is measured through the default.
10. **Full 50 MP spectral simulation is not supported at v0.1.0.** A 50 MP frame on the 95-wavelength grid would need about 38 GB for one float64 cube before working copies. `scene.render` and `Sensor.expose` estimate peak memory before allocating and reject a request above the default 8 GiB budget with guidance to reduce shape/oversampling or process in chunks. The validated practical large-frame case is 1024 x 1024 at 1x oversampling (about 2.2 GiB estimated peak, measured in section 5.1); 512 x 512 at 4x oversampling has a 3.2 GB cube and fits only with sufficient RAM. The guard's peak estimate for a 50 MP render is about 106 GiB, of which one float64 cube alone is about 38 GB. Chunked spectral simulation and a full 50 MP path are future work.
11. **CI is added but not executed here.** The workflow includes quality, test-matrix, docs, build, examples, and notebooks jobs. GitHub Actions cannot run in this environment; the local equivalents passed and CI execution is pending a push. The owner must accept that residual, and no claim that CI is green is made.
12. **Polychromatic PSF and banding are not v0.1.0.** The optics model uses a wavelength-independent PSF, and row noise is modelled without periodic banding. Both are documented future capabilities, not silent omissions.
13. **Display P3 files carry sidecar metadata only.** A P3 PNG is tagged in its sidecar; the PNG itself has no reliably interoperable P3 colour profile in this release. Consumers must read the sidecar.
14. **Reduced frame contract.** `RawFrame` omits wall-clock timestamp and package version, and `RenderedImage` omits `linear_xyz`; acquisition provenance lives in the surrounding experiment record. The reduction is documented, not hidden.
15. **No machine-vision evidence.** `camsim.tasks` is an empty gated namespace. This report says nothing about CNN accuracy, task-relevant IQ, mono-versus-Bayer, auto-tuning, or metric correlation.

## 8. Reproduction

Run from the repository root with `uv` and all extras installed:

```bash
uv sync --all-extras

# Phase artifacts (regenerate only when the phase code changes)
uv run python benchmarks/g1_scene_optics.py
uv run python examples/04_emva_characterization.py
uv run python examples/02_isp_stage_gallery.py
uv run python examples/03_measure_iq.py

# Gallery: examples 01-04 and notebooks 01-03, with per-item timeouts
uv run python benchmarks/run_examples.py
uv run python benchmarks/run_notebooks.py

# Consolidated G5 report and figures
uv run python benchmarks/g5_validation_report.py

# G5 quality gates
uv run pytest -q
uv run pytest --cov=camsim --cov-fail-under=90
# Whole-package coverage including viz, for the two-scope report:
uv run pytest --cov=camsim --cov-config=coverage-whole.rc --cov-report=term
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run mkdocs build --strict
uv run python benchmarks/benchmark_pipeline.py --quick
uv run python benchmarks/benchmark_pipeline.py
uv run python benchmarks/benchmark_pipeline.py --large
uv build
```

`make figures` runs examples 01-04 and is the gallery regeneration target; `make notebooks` executes every Jupytext notebook into `results/cache/notebooks` under a two-minute-per-notebook budget. Neither command tags, pushes, or publishes anything.

**Pre-commit snapshot caveat.** The generator records `git rev-parse HEAD` and `date.today()` at generation time. Re-running it on the candidate commit therefore rewrites the validated-commit and report-date fields and dirties the working tree. The committed report is a pre-commit snapshot of that candidate; regenerate it only as part of a new candidate, and commit the regenerated report with the change that invalidated it.

## 9. Figure gallery

| Figure | Shows |
|---|---|
| [g5_gate_summary.png](../figures/g5_gate_summary.png) | Gate/check status matrix; G1-G4 PASS, G5 recorded checks pass, human sign-off awaiting |
| [g5_evidence.png](../figures/g5_evidence.png) | Denoise SNR, signed MTF50 change and ringing, sharpen profile/overshoot, auto-exposure convergence, and benchmark stage shares |
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
- [ ] **Limitations confirmed.** Accept the synthetic CFA re-authoring, in-sample calibration, 8x Airy diagnostic, synthetic stage measurements, quick-mode exclusion, extreme-low-light digital-gain noise, the identity default CCM, the absence of HDR/3-D/motion-blur/autofocus, no polychromatic PSF or banding, P3 sidecar-only tagging, and the reduced RawFrame/RenderedImage contract.
- [ ] **Full-50-MP limitation accepted.** Confirm that v0.1.0 supports single-process spectral simulation up to the configured memory budget (validated at 1024 x 1024, 1x oversampling) and rejects larger grids with a clear error rather than attempting a 50 MP spectral cube.
- [ ] **Coverage exception accepted.** Accept that subject-module coverage is 91.73 % and whole-package coverage including `camsim.viz` is lower, with viz exercised by the executed examples and notebooks. The 90 % gate applies to the numerical subject modules, not to the plotting layer.
- [ ] **CI and notebook evidence accepted.** Confirm that the CI workflows and runners were added, every example and notebook passed locally under the 300 s / 120 s per-item timeouts, and CI execution is pending a push. Accept that residual.
- [ ] **Two-minute-per-notebook condition confirmed.** Accept the per-notebook local times in section 5.9, all below 120 s, and the runner/CI timeout that enforces the budget.
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
| overshoot_fraction | `0.0007194` |
| undershoot_fraction | `0.0006522` |
| ringing_fraction | `0.0007194` |
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

