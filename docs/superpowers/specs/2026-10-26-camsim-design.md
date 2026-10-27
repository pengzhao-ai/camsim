# `camsim` — Design Document

**Status:** Draft for review
**Date:** 2026-10-26
**Supersedes:** `Only for reference - iq_for_machine_vision_project_plan_revised.md`
**License:** Apache-2.0

---

## 1. Intent

`camsim` is a professional, reusable, fully deterministic Python library for simulating a
camera's sensor and image signal processor (ISP) from a spectral irradiance map to a
rendered image, measuring image quality with standard metrics, and studying how the
imaging chain affects machine-vision task accuracy.

The project has three clean layers:

1. **Simulation pipeline** — scene/optics helpers (not the reusable core), a rigorous
   sensor model (irradiance → RAW), and a modular ISP (RAW → rendered image).
2. **Image-quality metrics** — an independent harness for e-SFR/MTF, SNR, colour error,
   and acutance, usable on simulated or real images.
3. **Machine-vision experiments** — a small PyTorch task engine and experiment suite
   showing how optics, sensor, and ISP parameters affect model accuracy.

The reusable core is the **sensor + ISP** pair. It is designed so that a user with real
optical simulation output can bypass `camsim.scene` and `camsim.optics` and feed a
`SpectralIrradiance` directly into the sensor. A convenience `Camera` orchestration layer
exists, but neither the sensor nor the ISP depends on it.

The project is intended to be published on GitHub, documented, tested, and cited. It is
not a notebook-only teaching demo. Pedagogy remains a requirement, but it is expressed
through rigorous docstrings, validation reports, and worked examples rather than through
simplified code.

---

## 2. Goals

### G1 — Reusable sensor/ISP toolbox
- `Sensor.expose(irradiance) -> RawFrame` and `ISP.process(raw) -> RenderedImage` are
  independent, typed, documented entry points.
- The sensor accepts spectral irradiance with explicit units and wavelength sampling.
- The ISP accepts a `RawFrame` that is self-describing (CFA, bit depth, black level,
  gains, mode, geometry, colour metadata).
- Both are configurable from versioned JSON with schema validation and provenance.
- Mono and RGGB are first-class end-to-end; RCCC is sensor-level with a documented
  baseline reconstruction. Mono is not a post-hoc desaturation of an RGB pipeline.
- The toolbox can be used without `camsim.scene`, `camsim.optics`, or `camsim.tasks`.

### G2 — Physically rigorous, published-quality sensor model
- Full spectral chain: illuminant × reflectance × lens/IR-cut × CFA × QE → electrons.
- Per-pixel noise: shot, dark (temperature-dependent), read, row/banding, DSNU, PRNU.
- Gain chain: LCG/HCG conversion gain, analog PGA, digital gain, black level,
  full-well and ADC saturation, quantization.
- Non-idealities: spectral and spatial crosstalk, defective pixels, PDAF occlusion,
  binning, optional blooming.
- A datasheet-grounded flagship-phone preset, plus a documented mapping from datasheet
  fields to JSON.
- Validation by EMVA 1288-style self-characterization with round-trip parameter recovery,
  analytic invariants, and reference-library cross-checks. No external RAW data required.

### G3 — Professional ISP
- DPC → BLC → LSC → AWB → demosaic → CCM → denoise → sharpen → tone → output.
- Linear-space invariance before the tone stage.
- Bilinear and Malvar-He-Cutler demosaicing; ridge-regularized CCM with row-sum
  constraint; luma/chroma denoise; thresholded unsharp mask; sRGB and Display P3 output.
- Per-stage buffers and metadata for visualization and debugging.
- Auto-estimation for AWB and optional CCM fitting; fixed or auto exposure in `Camera`.

### G4 — Independent, validated metrics harness
- ISO 12233-style slanted-edge e-SFR → MTF50/MTF20 in cycles/pixel and cycles/degree.
- SNR per channel and luma on flat patches; temporal and spatial variants.
- ΔE76 and ΔE2000 per patch against spectrally computed ground truth.
- Acutance for metric-correlation studies.
- Unified `measure_all(image, layout, reference=None) -> MetricReport`.
- `camsim.metrics` must not import `camsim.sensor`, `camsim.isp`, `camsim.scene`, or
  `camsim.optics`; it operates on arrays plus metadata.

### G5 — Machine-vision experiments
- Synthetic glyph classification with a small PyTorch CNN.
- Clean-training / degraded-evaluation as the primary protocol, plus a retraining
  control to separate information loss from domain shift.
- Sweeps over lux, blur, exposure, gain mode, denoise/sharpen, mono vs Bayer, and
  illuminant/white balance.
- Mono-vs-Bayer matched-accuracy study.
- NSGA-II auto-tuning against task loss and photographic loss.
- Metric-to-task correlation analysis.
- Machine-vision work does not begin until the pipeline and metrics pass a formal
  sign-off gate (Section 10).

### G6 — Publishable engineering
- Python 3.11–3.13, `src/` layout, `uv`, `hatchling`, `ruff`, `mypy`, `pytest`,
  Hypothesis, GitHub Actions, pre-commit, MkDocs Material, coverage, changelog,
  citation metadata, Apache-2.0.
- Deterministic stochastic routines with explicit RNG objects and seeds.
- Tested examples and Jupytext notebooks; no committed `.ipynb` files.
- A validation report and figure gallery that a skeptical engineer can inspect.

---

## 3. Non-goals (v1)

- Full 3D scene rendering, path tracing, or physically based materials.
- Lens design, ray tracing, aberrations beyond Gaussian/Airy/defocus approximations.
- Autofocus, optical image stabilization, motion blur, rolling-shutter motion artefacts.
- Dual-gain HDR merge or multi-frame HDR.
- Consumer camera features such as face detection, 3D LUTs, local tone mapping, or
  advanced computational photography.
- Deployment/runtime optimization for mobile hardware.
- Certification against ISO 12233 or EMVA 1288; the implementation follows their
  methods and reports assumptions, but is not a certified test lab.
- Claiming that a generic preset represents a specific commercial part's measured
  performance. Presets are representative and provenance-labelled.

## 4. Anti-goals (failure even if the code runs)

- A visually plausible pipeline that does not conserve energy or units.
- A "validated" claim backed only by smoke tests.
- Hidden unit ambiguity (lux vs W/m², electrons vs DN, per-nm vs integrated).
- A sensor that cannot accept real irradiance without editing internals.
- Metrics coupled to simulator internals.
- A mono mode that is an RGB afterthought.
- Non-deterministic noise or training that cannot be reproduced.
- Notebooks or sweeps that take so long they are not actually run.
- ML results built on an unvalidated pipeline.
- A config schema that cannot represent a real datasheet without code changes.

---

## 5. Users and use cases

1. **Camera/ISP engineer** — compare sensor modes, pixel sizes, noise floors, and ISP
   parameter sets before hardware exists; produce design-review figures.
2. **Machine-vision engineer** — ask which image-quality metrics predict task accuracy
   and how much SNR/MTF/colour accuracy a model actually needs.
3. **Researcher/educator** — use the sensor and ISP as a reproducible testbed, cite the
   validation methodology, and extend the stage library.
4. **Optical simulation user** — feed an externally computed spectral irradiance cube
   into the sensor and use the ISP/metrics without adopting `camsim.optics`.

Primary application domains: consumer phones, AR/VR passthrough and tracking cameras.
The model remains domain-agnostic.

---

## 6. Architecture

### 6.1 Package layout

```
camsim/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── NOTICE / THIRD_PARTY.md
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── Makefile
├── jupytext.toml
├── .pre-commit-config.yaml
├── .github/workflows/{ci,docs,release}.yml
├── archives/              # superseded material, not part of the package
├── docs/
│   ├── index.md
│   ├── concepts.md
│   ├── architecture.md
│   ├── datasheet_mapping.md
│   ├── isp_stages.md
│   ├── metrics.md
│   ├── validation.md
│   ├── whitepaper.md
│   ├── reference/          # superseded plan, kept for provenance
│   ├── api/
│   └── superpowers/specs/2026-10-26-camsim-design.md
├── src/camsim/
│   ├── __init__.py
│   ├── _version.py
│   ├── core/
│   │   ├── units.py
│   │   ├── spectral.py
│   │   ├── frames.py
│   │   ├── color.py
│   │   └── constants.py
│   ├── config/
│   │   ├── schema.py
│   │   ├── io.py
│   │   └── presets/
│   │       ├── sensors/flagship_phone_50mp.json
│   │       └── isp/default.json
│   ├── scene/
│   │   ├── targets.py
│   │   ├── glyphs.py
│   │   ├── spectral_data.py
│   │   └── illuminants.py
│   ├── optics/
│   │   ├── lens.py
│   │   ├── psf.py
│   │   ├── propagation.py
│   │   └── vignetting.py
│   ├── sensor/
│   │   ├── sensor.py
│   │   ├── cfa.py
│   │   ├── noise.py
│   │   ├── gain.py
│   │   ├── adc.py
│   │   ├── defects.py
│   │   └── binning.py
│   ├── isp/
│   │   ├── pipeline.py
│   │   ├── dpc.py
│   │   ├── blc.py
│   │   ├── lsc.py
│   │   ├── awb.py
│   │   ├── demosaic.py
│   │   ├── ccm.py
│   │   ├── denoise.py
│   │   ├── sharpen.py
│   │   ├── tone.py
│   │   └── colorspace.py
│   ├── metrics/
│   │   ├── mtf.py
│   │   ├── snr.py
│   │   ├── color.py
│   │   ├── acutance.py
│   │   ├── chart.py
│   │   └── summary.py
│   ├── characterization/
│   │   ├── emva.py
│   │   └── report.py
│   ├── camera/
│   │   └── camera.py
│   ├── tasks/
│   │   ├── glyphs.py
│   │   ├── cnn.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── sweeps.py
│   │   ├── tune.py
│   │   └── correlation.py
│   ├── viz/
│   │   ├── style.py
│   │   ├── spectra.py
│   │   ├── pipeline.py
│   │   ├── metrics.py
│   │   └── experiments.py
│   ├── io/
│   │   ├── raw.py
│   │   └── images.py
│   └── cli.py
├── tests/
│   ├── test_units.py
│   ├── test_spectral.py
│   ├── test_scene.py
│   ├── test_optics.py
│   ├── test_sensor.py
│   ├── test_cfa.py
│   ├── test_noise.py
│   ├── test_adc.py
│   ├── test_isp_*.py
│   ├── test_metrics_*.py
│   ├── test_characterization.py
│   ├── test_config.py
│   ├── test_camera.py
│   ├── test_determinism.py
│   ├── test_validation_*.py
│   └── test_metrics_independence.py
├── examples/
│   ├── 01_spectral_scene_to_raw.py
│   ├── 02_isp_stage_gallery.py
│   ├── 03_measure_iq.py
│   ├── 04_emva_characterization.py
│   ├── 05_glyph_task.py
│   ├── 06_mono_vs_bayer.py
│   ├── 07_autotune.py
│   └── 08_metric_correlation.py
├── notebooks/
│   ├── 01_optics_and_sensor.py
│   ├── 02_isp_color_fundamentals.py
│   ├── 03_measuring_iq.py
│   ├── 04_iq_vs_task_performance.py
│   ├── 05_mono_vs_bayer.py
│   ├── 06_toy_autotune.py
│   └── 07_which_metric_predicts_task.py
├── results/
│   ├── figures/
│   ├── data/
│   ├── reports/
│   └── cache/          # ignored by git
└── benchmarks/
```

`camsim.metrics` is a sibling of the simulation packages, not a submodule of the sensor or
ISP. The import-independence test enforces this.

### 6.2 Core data contracts

**`WavelengthGrid`** — 1-D `float64` nm array; strictly increasing, uniform; validated
range (default 360–830 nm, 5 nm step).

**`SpectralIrradiance`** — 2-D spatial × spectral `float32`/`float64` array plus:
- `wavelengths_nm: WavelengthGrid`
- `unit`: `"W/m^2/nm"` or `"photons/s/m^2/nm"`
- optional `pixel_pitch_um` and geometry metadata
- validation of shape, non-negativity, and unit conversion helpers

The boundary is explicit: irradiance at the **sensor plane, before the sensor's own
spectral filters** (IR-cut, CFA, microlens). A user supplying data that already includes
the IR-cut disables it in the sensor config. This prevents double counting.

**External irradiance contract.** The core `Sensor.expose` API requires the cube to be
on the sensor pixel grid: shape `(H, W, λ)` with `H` and `W` equal to the active pixel
count, and a wavelength grid matching the sensor's spectral responses by default. If
`pixel_pitch_um` is supplied in the metadata and differs from the sensor pitch, the
sensor raises a clear error. Two explicit helpers make other inputs usable without
silent resampling:

- `SpectralIrradiance.resample_spatial(target_shape, method="box")` — area-conserving
  box averaging to the sensor grid.
- `SpectralIrradiance.resample_wavelengths(target_grid, method="integrate")` —
  overlap-weighted integration that conserves energy; it raises if the input range
  does not cover the sensor range, or zero-fills with a warning when explicitly allowed.

The sensor never resamples implicitly. This is the reuse contract for real optical
simulation output: explicit, validated, and unit-safe.

**`RawFrame`** — `uint16` data plus complete metadata:
- CFA pattern, bit depth, black level DN, white/analog/digital gains, exposure, mode
  (LCG/HCG), temperature, binning, sensor config hash
- defective-pixel, hot-pixel, PDAF, and binning masks where applicable
- geometry: pixel pitch, focal length (optional), pixel count
- simulation provenance: package version, frame seed, frame index, timestamp

Every ISP decision that depends on the sensor must be derivable from `RawFrame`; no
hidden globals.

**`RenderedImage`** — display-referred `float32` image in `[0, 1]`, plus:
- `color_space`: `"sRGB"` or `"DisplayP3"`
- `channels`: `"RGB"` or `"MONO"`
- optional `linear_rgb` / `linear_xyz` for metrics
- optional `stages: dict[str, StageOutput]` when tracing
- metadata (white point, exposure, ISP config hash)

**`MetricReport`** — Pydantic model with optional metric groups (MTF, SNR, colour,
acutance), layout id, units, and provenance. Serializes to dict/JSON.

**`ColorReference`** — owned by `camsim.core.color` so both the scene and metrics layers
can use it without a circular dependency. Contains per-patch reference Lab values, the
white point, illuminant name/SPD (optional), observer, and provenance. The scene layer
computes it spectrally; `measure_all` consumes it. This is how non-D65 ground truth
reaches the metrics layer without `camsim.metrics` importing `camsim.scene`.

### 6.3 Sensor/ISP separation

```python
sensor = camsim.Sensor.from_json("presets/sensors/flagship_phone_50mp.json")
isp = camsim.ISP.from_json("presets/isp/default.json")
lens = camsim.optics.LensConfig(f_number=1.8, focal_length_mm=6.4, transmittance=0.9)

irradiance = camsim.scene.render(..., lens=lens)  # or an external SpectralIrradiance
raw = sensor.expose(
    irradiance, exposure_s=10e-3, mode="HCG", analog_gain_db=12,
    temperature_c=25, seed=42,
)
rendered = isp.process(raw, trace=True)

layout = camsim.scene.colorchecker_layout(illuminant="D65", lens=lens)
report = camsim.metrics.measure_all(rendered, layout=layout)
```

A `Camera` convenience wrapper composes scene → optics → sensor → ISP and can run an
auto-exposure loop, but the sensor and ISP never require it.

### 6.4 Configuration and provenance

- Pydantic v2 models, `extra="forbid"`, units in every field name.
- Cross-field validators (e.g., full-well × conversion gain vs ADC full scale; HCG full
  well ≤ LCG full well; QE ∈ [0,1]; crosstalk and CCM row sums; non-negative noise).
- Every preset carries `provenance: {source, url, retrieved, notes, assumptions}`.
- JSON Schema exported by `camsim schema sensor|isp`; validated in CI.
- A datasheet-mapping guide maps datasheet terms to JSON fields and documents how to
  derive missing values. A filled example is included.
- `camsim validate CONFIG` returns errors/warnings, including physically suspicious
  combinations.

---

## 7. Physics specification

### 7.1 Scene and optics (supporting modules, not the reusable core)

- 2-D spectral reflectance maps rendered on a 4× oversampled grid.
- Targets: slanted edge, 24-patch ColorChecker, flat patches, Siemens star, text glyphs.
- Illuminants: D65, 3000 K, 4000 K, custom SPD.
- Lambertian scene radiance and image-plane irradiance:
  `E_img(λ) = ρ(λ) · E_scene(λ) · T_lens(λ) / (4 N²) · cos⁴θ`.
- PSFs: Gaussian (σ in pixels), diffraction-limited Airy (first zero `1.22 λ N`), and
  optional defocus. A `PolychromaticPSF` supports per-wavelength convolution (accurate)
  or a weighted-average kernel (fast), with the choice recorded in metadata.
- Pixel-aperture integration by box filtering before downsampling to the sensor grid.
- Optional `cos⁴` vignetting.
- Output: `SpectralIrradiance` at the sensor plane, before IR-cut/CFA.
- **Real optical data:** the user can construct `SpectralIrradiance` directly from a
  spectral cube; the optics module is never imported by the sensor.

#### 7.1.1 Reference camera and lux normalization

Absolute electron scale needs a lens reference. `camsim.optics.LensConfig` carries
`f_number`, `focal_length_mm`, `field_of_view_deg` (derived or supplied),
`transmittance` (scalar or spectral CSV), and optional vignetting coefficients. It is
independent of the sensor preset; `CameraConfig` composes lens + sensor + ISP.

The reference lens for v1 is a flagship-phone main camera:

- f/1.8, 24 mm full-frame equivalent, ~84° diagonal FOV, T = 0.9.
- With a 1/1.3" sensor (~9.2 × 6.9 mm active area, 50 MP at 1.12 µm), the actual focal
  length is ~6.4 mm and the crop factor ~3.8.
- The lens configuration is a preset and is fully overridable.

Photometric normalization is explicit. For an illuminant SPD in relative units and a
target illuminance in lux:

```
E_scene(λ) = lux · SPD(λ) / (K_m · ∫ SPD(λ) · V(λ) dλ),   K_m = 683 lm/W
```

with `V(λ)` the CIE photopic luminous efficiency. The image-plane spectral irradiance
for a Lambertian scene is:

```
E_img(λ) = ρ(λ) · E_scene(λ) · T_lens(λ) / (4 N²) · cos⁴θ
```

`scene.render(lux=..., lens=...)` applies this chain and returns a `SpectralIrradiance`
at the sensor plane. The sensor itself never needs lux or lens parameters; it consumes
irradiance. `characterize_sensor` prefers `irradiance_levels`; the `lux_levels`
convenience requires an explicit illuminant and lens (defaulting to D65 and the reference
lens) and records them in the report. This removes the ambiguity between photometric lux
and radiometric irradiance, and makes the absolute electron scale auditable.

### 7.2 Sensor signal chain

For a pixel with effective aperture fraction `η_eff = min(1, fill_factor · microlens_gain)`
and area `A = pitch² · η_eff`:

```
electron_rate = ∫ QE(λ) · T_IR(λ) · T_CFA(λ) · E(λ) · A / (h·c/λ) dλ
```

with `E` in W/m²/nm, `λ` in metres, and `h·c/λ` in joules per photon. All spectral
integrals are precomputed as projection matrices per CFA channel and cached.

Signal chain order (documented and tested):

1. Mean signal electrons `S` from the spectral integral.
2. PRNU: `S_pix = S · (1 + prnu_pix)`.
3. Dark current: `D(T) = I_dark(T_ref) · 2^((T - T_ref)/T_double) · t_exp`; add DSNU
   offset `dsnu_pix`.
4. Sample `N ~ Poisson(S_pix + D_pix)`; clip to full well (charge domain).
5. Optional blooming: redistribute overflow to neighbours.
6. Optional spectral crosstalk on channel means, and optional spatial crosstalk kernel.
7. Add input-referred read noise `N(0, σ_read)`; clip to full well.
8. Row noise (electrons, common per row) and optional banding in the electron
   domain, before conversion.
9. Conversion gain: `V = N · CG_mode` (µV/e⁻).
10. Analog gain: `V ·= 10^(G_db/20)`.
11. ADC: `DN = round(V / V_ref · (2^bits - 1 - black_level)) + black_level`;
    clip to `[0, 2^bits - 1]`.
12. Digital gain applied to signal DN before re-adding black level; clip.

`RawFrame` stores the mode, gains, black level, bit depth, and masks. Saturation can be
full-well-limited or ADC-limited; the validator warns when the configured gain makes the
ADC the limiting factor.

**LCG/HCG:** each mode defines conversion gain, full well, read noise, and dark current.
`mode="auto"` selects HCG below a configured signal threshold and LCG otherwise.
Dual-gain HDR merge is a non-goal for v1.

**Binning:** analog (charge-domain sum before full well, full well scales by factor²) or
digital (post-ADC average). Read-noise treatment differs and is documented/tested.

**Mono / RGGB / RCCC:** mono and RGGB are first-class end-to-end. Mono uses bare silicon
QE with no CFA absorption, single-channel RAW, and an ISP bypass for demosaic/AWB/CCM.
RCCC is supported by the sensor and by a documented baseline ISP reconstruction; it is
covered by shape, linearity, and smoke tests plus a qualitative image, not by the Bayer
ΔE/MTF gates. Advanced RCCC demosaic and RCCC-specific AWB/CCM tuning are deferred to
v1.1.

**Defects/PDAF:** dead, hot, and stuck pixels generated from rates and a seed; masks in
`RawFrame`. PDAF pixels have a reduced effective aperture and are flagged, not treated as
defects. DPC ignores PDAF pixels.

#### 7.2.1 RNG, temperature, and determinism contract

- `Sensor(config, seed=...)` creates a `Generator` for fixed-pattern maps (defects, DSNU,
  PRNU, PDAF masks) from `pattern_seed`. Those maps are generated once and reused across
  frames.
- `expose(..., seed=None, temperature_c=None)`. If `seed` is supplied, a fresh
  `Generator(seed)` is used for that frame; the frame seed and frame index are recorded
  in `RawFrame`. If omitted, the sensor's frame generator advances from its base seed.
  `sensor.reseed(seed)` resets the frame stream.
- Characterization uses `base_seed + frame_index` so frames are independent and
  reproducible. The report records the base seed.
- Temperature is a per-call override; it defaults to the preset value and is recorded.
  Dark current uses the Arrhenius doubling model in Section 7.2.
- **Determinism scope:** within one environment (same OS, Python, NumPy, SciPy versions)
  and the same seeds, RAW output is bit-identical. Across platforms, floating-point
  reductions may differ; CI uses tolerance-based golden tests, and integer RAW hashes are
  only asserted within a platform. This is the single determinism contract; there is no
  conflicting cross-platform bit-identity promise.

#### 7.2.2 Auto-exposure policy

`Camera.auto_expose()` meters the linear signal after black-level subtraction with a
configurable center-weighted statistic, targeting 18% of usable full scale by default.
It adjusts in this order:

1. exposure time within `[min_s, max_s]`;
2. analog gain within `[min_db, max_db]`;
3. digital gain;
4. HCG/LCG `mode="auto"` selects HCG below a signal threshold, then re-meters.

The loop is damped, bounded by a maximum iteration count, and records convergence,
exposure, gains, mode, and metered value in an `ExposureResult`.

Primary physics and ML sweeps use **fixed exposure, gain, and mode** except when one of
them is the independent variable. This prevents AE from equalizing brightness across a
lux sweep and hiding the sensor differences the experiment is meant to measure. AE is a
separate experiment and an explicit, recorded policy for selected ML conditions; the lux
sweep reports the fixed-exposure result first and the AE variant separately.

### 7.3 ISP stages

Default order (enforced; stages individually toggleable):

```
RAW → DPC → BLC → LSC → AWB → Demosaic → CCM → Denoise → Sharpen → Tone → Output
```

- **DPC:** mask-guided neighbour interpolation (same CFA channel); median fallback.
- **BLC:** subtract scalar or 2×2-pattern black level; clamp; convert to float32.
- **LSC:** analytic radial gain or supplied map; optional flat-field calibration.
- **AWB:** gray-world, white-patch, and shades-of-gray on RAW; per-CFA-channel gains;
  returns gains, illuminant estimate, method, confidence. Mono is a no-op.
- **Demosaic:** bilinear and Malvar-He-Cutler (2004). Mono bypass. RCCC baseline.
- **CCM:** 3×3 least-squares fit on ColorChecker patches per illuminant with ridge (L2)
  regularization and row-sum constraint (Σrow = 1); precomputed matrix or runtime fit;
  applied in linear RGB.
- **Denoise:** linear opponent luma/chroma split; Gaussian and bilateral methods;
  independent luma/chroma strengths.
- **Sharpen:** unsharp mask with amount, radius, threshold; luma-only by default.
- **Tone:** sRGB OETF or configurable gamma; optional exposure compensation; output
  sRGB or Display P3.
- **Trace:** every stage can record its output and metadata. `trace=True` returns a
  contact-sheet-ready dictionary; tracing is off by default for large sweeps.

**Linear-space invariant:** DPC, BLC, LSC, AWB, demosaic, CCM, denoise, and sharpen all
operate in linear float32. Tone and the output transfer function are the only non-linear
steps. Tests assert linearity (e.g., scaling the input scales the pre-tone output) and
that no stage silently gamma-encodes.

### 7.4 Metrics

**MTF / e-SFR (ISO 12233-style):**
- ROI extraction, edge-angle regression, 4× oversampled ESF, LSF derivative with Hamming
  window, derivative-response correction, FFT, DC normalization.
- Report MTF50/MTF20 in cycles/pixel and cycles/degree; curve, edge angle, Nyquist.
- `pixel_pitch_um` and `focal_length_mm` (or pixels-per-degree) from metadata.
- Raw mono, raw green-plane, and post-demosaic channels. The green-plane quincunx
  sampling caveat is documented; results are reported with their sampling lattice.
- Validated against analytic Gaussian, Airy, and box PSFs. Described as ISO 12233-style,
  not certified.

**SNR:**
- Flat-patch mean/std per channel and linear luma; `SNR_dB = 20 log10(mean/std)`.
- Spatial (single frame) and temporal (multi-frame) variants.
- Clipping/black-level handling; report patch mean, std, and validity flags.

**Colour:**
- ΔE76 and ΔE2000 per ColorChecker patch.
- Ground truth computed spectrally: patch reflectance × illuminant × CIE 1931 2°
  observer → XYZ → Lab under the same white point. This makes 3000 K/4000 K ground
  truth correct rather than assuming D65.
- Rendered RGB linearized, converted via the output colour space matrix to XYZ/Lab.
- Report mean, max, per-patch values, and patch validity.

**Acutance:** documented edge-acutance metric for correlation studies.

**`measure_all(image, layout, reference=None)`:** runs only the metric groups supported
by the supplied layout and image; returns a `MetricReport`. Works for mono and RGB. No
simulator imports.

Colour ground truth travels as a `ColorReference` (Section 6.2). `ChartLayout` may carry
one; `measure_all` uses `layout.reference` when `reference` is omitted. For sRGB images
with no reference, it falls back to standard D65 ColorChecker Lab with a warning. For
non-D65 illuminants a reference is required, so 3000 K/4000 K results cannot silently
fall back to D65. The scene layer computes the reference spectrally; the metrics layer
only consumes it.

### 7.5 EMVA 1288-style characterization

`camsim.characterization` measures the simulated sensor:
- photon transfer curve (mean vs variance), system gain K [DN/e⁻], read noise,
  saturation capacity, absolute sensitivity threshold, dynamic range, SNR curve;
- DSNU from dark frames; PRNU from flat-field frames;
- linearity residuals; optional dark-current vs temperature.

The module cites EMVA 1288 definitions and states assumptions. It is a validation tool,
not a certified EMVA report. Round-trip tests configure a known sensor, characterize it,
and recover the inputs within tolerance.

---

## 8. API examples

```python
import camsim

sensor = camsim.Sensor.from_json("flagship_phone_50mp.json")
isp = camsim.ISP.from_json("default.json")
lens = camsim.optics.LensConfig(f_number=1.8, focal_length_mm=6.4, transmittance=0.9)

irr = camsim.scene.render(
    target="colorchecker",
    illuminant="D65",
    lux=200,
    lens=lens,
    optics=camsim.optics.GaussianPSF(sigma_px=1.2),
    pixel_pitch_um=1.12,
    shape=(512, 512),
)

raw = sensor.expose(
    irr, exposure_s=10e-3, mode="HCG", analog_gain_db=6,
    temperature_c=25, seed=42,
)
rendered = isp.process(raw, trace=True)

layout = camsim.scene.colorchecker_layout(illuminant="D65", lens=lens)
report = camsim.metrics.measure_all(rendered, layout=layout)

camsim.viz.plot_pipeline_trace(rendered.stages, save="results/figures/isp_trace.png")
camsim.viz.plot_mtf(report.mtf, save="results/figures/mtf.png")
camsim.viz.plot_colorchecker(rendered, report.color, save="results/figures/color.png")
```

```python
# Real optical simulation input: bypass scene/optics entirely.
irr = camsim.SpectralIrradiance.from_array(
    data=my_spectral_cube,               # (H, W, λ) on the sensor pixel grid
    wavelengths_nm=my_wavelengths,       # must match the sensor spectral grid
    unit="W/m^2/nm",
    pixel_pitch_um=1.12,
)
raw = sensor.expose(irr, exposure_s=5e-3, mode="LCG", seed=43)
```

```python
# Characterization. irradiance_levels is the preferred radiometric form.
report = camsim.characterization.characterize_sensor(
    sensor, lux_levels=[1, 2, 5, 10, 20, 50, 100, 200, 500],
    frames_per_level=32, mode="HCG", temperature_c=25,
    illuminant="D65", lens=lens, base_seed=1234,
)
report.to_json("results/reports/emva_flagship.json")
camsim.viz.plot_emva(report, save="results/figures/emva.png")
```

---

## 9. Configuration example

```json
{
  "schema_version": "1.0",
  "name": "flagship_phone_50mp",
  "provenance": {
    "source": "public datasheet + representative assumptions",
    "url": "https://example.com/datasheet",
    "retrieved": "2026-10-26",
    "notes": "Values marked assumed are engineering estimates, not vendor measurements.",
    "assumptions": ["QE curve representative of backside-illuminated silicon",
                    "IR-cut cut-on 650 nm", "microlens gain 1.15"]
  },
  "pixel": {"pitch_um": 1.12, "fill_factor": 0.85, "microlens_gain": 1.15},
  "cfa": "RGGB",
  "spectral": {
    "wavelength_nm": {"start": 360, "stop": 830, "step": 5},
    "qe_csv": "data/qe_flagship.csv",
    "cfa_csv": "data/cfa_rggb.csv",
    "ir_cut_csv": "data/ir_cut.csv",
    "spectral_crosstalk": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  },
  "modes": {
    "LCG": {"conversion_gain_uv_per_e": 100.0, "full_well_e": 6000,
            "read_noise_e_rms": 2.0, "dark_current_e_per_s": 1.2},
    "HCG": {"conversion_gain_uv_per_e": 300.0, "full_well_e": 2000,
            "read_noise_e_rms": 0.7, "dark_current_e_per_s": 1.2}
  },
  "analog_gain": {"min_db": 0.0, "max_db": 24.0, "step_db": 1.0, "default_db": 0.0},
  "adc": {"bits": 10, "full_scale_uv": 800000.0, "black_level_dn": 64,
          "digital_gain": 1.0},
  "exposure": {"min_s": 0.0001, "max_s": 0.1, "default_s": 0.01},
  "temperature": {"reference_c": 25.0, "default_c": 25.0, "dark_current_doubling_k": 7.0},
  "defects": {"dead_rate": 1e-5, "hot_rate": 1e-5, "stuck_rate": 1e-6, "seed": 1234},
  "pdaf": {"fraction": 0.01, "qe_factor": 0.5, "pattern": "sparse_green"},
  "binning": {"factor": 1, "mode": "analog"},
  "blooming": {"enabled": false, "fraction": 0.02, "iterations": 2}
}
```

Exact field names will be finalized in Phase 0; units are part of every name. The
preset is explicitly representative, not a claim about a specific commercial part.

---

## 10. Validation and sign-off

Validation uses four independent lines of evidence:

1. **Analytic invariants** — units, energy conservation, linear lux/exposure scaling,
   Poisson variance, clipping, known-PSF MTF, sRGB OETF, ΔE identities.
2. **Round-trip characterization** — configure known sensor parameters, run EMVA-style
   characterization, recover them within tolerance.
3. **Reference-library cross-checks** — bilinear/MHC demosaic vs `colour-demosaicing`,
   colour transforms and ΔE vs `colour-science`, filtering vs `scipy`/`scikit-image`.
4. **Datasheet plausibility** — preset parameters yield SNR, dynamic range, full well,
   and read noise in the ranges implied by public specifications/provenance.

### Gate G1 — Scene/optics
- Spectral cube shape/unit validation; energy conservation through optics; PSF
  normalization; analytic PSF MTF matches the implemented blur; 4× downsampling correct.
- Lux → spectral irradiance normalization validated against `V(λ)` and `K_m = 683 lm/W`.
- External `SpectralIrradiance` contract: matching-grid input accepted; mismatched
  spatial pitch/shape and uncovered wavelength ranges raise; explicit resampling helpers
  conserve energy.
- Reference lens parameters and cycles/degree metadata propagate correctly.

### Gate G2 — Sensor
- Linear electron scaling with lux and exposure (R² > 0.999 below saturation).
- Shot-noise variance slope = mean within 5% in the shot-noise regime.
- Read noise, dark current, DSNU, PRNU recovered within 5% (FPN within 10%).
- Full-well and ADC clipping at expected DN; LCG/HCG recovery within 5%.
- Mono/RGGB/RCCC mosaics correct; binning and crosstalk tests pass.
- Same seed → bit-identical RAW within the test environment (same OS/Python/NumPy/SciPy);
  different seed → different noise. Cross-platform comparisons use tolerances.
- Per-call seed and temperature overrides and fixed-pattern-map reuse follow Section
  7.2.1.
- Datasheet preset passes validation and yields plausible EMVA metrics.

### Gate G3 — ISP
- DPC/BLC/LSC unit tests; flat-field correction reduces vignetting to < 2%.
- AWB recovers known gains; gray-world/white-patch behave as expected.
- Bilinear and MHC match reference implementations within tight tolerance.
- CCM recovers a known matrix; row sums = 1; ridge reduces noise; ΔE improves.
- Denoise raises SNR and preserves MTF within stated bounds; sharpen raises MTF50 and
  produces measurable overshoot; both are identities when disabled.
- sRGB OETF matches the standard exactly; linearity invariant holds; stage trace
  reproduces the final output.
- End-to-end ColorChecker ΔE2000 mean < 2.0 under D65 and < 4.0 under 3000 K after CCM
  on the validation scene (thresholds revisited with the final sensor/ISP).
- RCCC baseline: shape, linearity, and smoke tests plus a qualitative reconstruction; the
  Bayer ΔE/MTF thresholds do not apply.
- `Camera.auto_expose()` converges on synthetic scenes, respects exposure/gain/mode
  limits, and records the `ExposureResult`.

### Gate G4 — Metrics
- MTF50/20 of analytic Gaussian/Airy/box PSFs within 2%; robust across edge angles and
  moderate noise; MTF monotonic and normalized.
- SNR within 0.5 dB of known synthetic noise; channel/luma variants correct.
- ΔE76/ΔE2000 match `colour-science` exactly for known pairs.
- `measure_all` complete, serializable, mono/RGB safe; import-independence test passes.
- Chart layouts locate patches/edges correctly on rendered charts.

### Gate G5 — Human sign-off (blocks ML)
- G1–G4 green; CI green; coverage ≥ 90%; docs build; examples and notebooks run
  headless and meet the runtime budget.
- A `results/reports/validation_report.md` with figures, tolerances, assumptions, and
  known limitations is reviewed by the project owner.
- Only after explicit sign-off: tag `v0.1.0` and begin Phase 6.
- Any later change to sensor/ISP/metrics that invalidates the report requires a re-run
  and re-sign-off.

### Runtime budget
- Notebooks: < 2 minutes each on Apple Silicon.
- Examples: < 5 minutes each in CI (CPU).
- Full sweeps and Optuna: CLI, cached, no notebook runtime limit.

---

## 11. Machine-vision experiments

### 11.1 Task
- 36-class glyph classification (0–9, A–Z), synthetic renderings, a small 3-layer CNN.
- Primary variant: grayscale glyphs (paper/ink spectra) for the mono-vs-Bayer study.
- Colour variant: random ink/paper spectra and illuminants, so AWB/CCM/ΔE affect the
  task and the colour pipeline is exercised. Planned as a Phase 6 extension.
- Scene is image-plane synthetic (glyph patch → optics → sensor → ISP); it is not a 3-D
  scene. This is sufficient to study the imaging chain and is documented as such.

### 11.2 Protocol
- Each sensor modality is trained on its own clean, high-lux, low-blur images with the
  default ISP and a fixed seed. The mono CNN sees a 1-channel rendered image; the Bayer
  CNN sees 3-channel RGB. Same scenes, seeds, architecture family, and training budget;
  report parameter counts.
- Evaluate each frozen model on degraded conditions. This measures real deployment-like
  hardware effects.
- Control experiment: retrain or fine-tune on each degraded condition. The gap between
  frozen and adapted accuracy separates information loss from domain shift.
- Cross-evaluation (mono model on Bayer images and vice versa) is a diagnostic for domain
  shift, not the headline comparison.
- RAW-to-CNN diagnostic: for selected conditions, compare the ISP-rendered input with
  normalized mono RAW and with Bayer RAW reduced to a green/luma representation, to
  quantify what the ISP adds or removes for machine vision.
- Report top-1 accuracy, cross-entropy, per-class accuracy, confusion matrix, and
  optionally expected calibration error.
- Fixed test set per condition; deterministic seeds; no train/test leakage.

### 11.3 Sweeps
- Lux (5–1000), fixed and auto exposure.
- Optical blur (σ 0.5–3 px).
- LCG vs HCG and analog gain.
- Denoise (luma/chroma) and sharpen (amount/radius) at low lux.
- Mono vs Bayer at matched optics/pixel pitch.
- Illuminant/white balance (D65, 3000 K, 4000 K), AWB on/off, CCM on/off.
- Caching: scene spectral cubes, clean training set, RAW per condition, and early ISP
  stages; downstream sweeps only recompute fast operators.

### 11.4 Mono vs Bayer
Same pixel pitch, optics, exposure, scenes, architecture family, and training budget.
Each modality gets its own clean-trained model: mono input is 1-channel, Bayer input is
3-channel RGB after its ISP. Report accuracy vs lux, MTF/SNR, the lux at which Bayer
matches mono, and the required MTF20/PPD increase for Bayer to match mono. A luma-matched
ablation (both modalities at 1-channel luma) is an optional diagnostic, not the headline.
Cross-evaluation between modalities is reported separately as a domain-shift check.

### 11.5 Auto-tuning
Optuna NSGA-II over 4–6 ISP parameters. Objective 1: task loss. Objective 2:
photographic loss = mean ΔE2000 + noise penalty + overshoot/halo penalty. Cached RAW;
notebook uses a small trial budget, CLI a larger one. Report the Pareto front and knee.

### 11.6 Metric correlation
Aggregate sweeps; Spearman rank correlation, standardized linear regression with
cross-validation, and decision-tree/random-forest feature importance. Report
multicollinearity diagnostics. Deliver guidance for machine-centric camera specs.

---

## 12. Engineering standards

- **Environment:** Python 3.11–3.13; `uv` with committed `uv.lock`; a project-local
  `.venv/` created by `uv sync` (pip fallback: `python -m venv .venv`), gitignored and
  not part of the source tree; `hatchling`; both workflows documented in the README.
- **Core dependencies:** `numpy`, `scipy`, `pydantic>=2`, `matplotlib`, `colour-science`,
  `pillow`.
- **Extras:** `[ml]` (torch, scikit-learn), `[tune]` (optuna), `[docs]`
  (mkdocs-material, mkdocstrings), `[dev]` (pytest, pytest-cov, hypothesis, ruff, mypy,
  pre-commit, jupytext, nbconvert, pandas, pyarrow). `[all]` composes them.
- **Quality:** ruff (lint + format), mypy strict on `src/`, pytest with coverage gate,
  Hypothesis for invariants, one test file per module, deterministic seeds.
- **Docs:** MkDocs Material, mkdocstrings API pages, concepts/architecture, datasheet
  mapping, ISP stages, metrics, validation report, tutorials, changelog, contributing,
  code of conduct, third-party notices. The whitepaper is a full v1.0 deliverable, not a
  stub.
- **CI:** GitHub Actions on Ubuntu + macOS, Python 3.11/3.12/3.13: lint, typecheck,
  tests, coverage, build, docs, example smoke tests, notebook execution (CPU).
- **Packaging:** wheel + sdist; entry point `camsim = "camsim.cli:main"`. GitHub tag at
  v0.1.0; PyPI trusted publishing and `pip install camsim` at v1.0.0 (name verified free;
  re-check at release).
- **Determinism:** explicit `numpy.random.Generator`; no global RNG; PyTorch manual
  seeding; config hash + seed in every report. Bit-identical across platforms is not
  promised for float reductions; tests use tolerances and document this.
- **Performance:** vectorized spectral projection matrices, FFT convolution for large
  PSFs, float32 defaults, caching, optional tracing. Profile before optimizing.
- **Style:** NumPy docstrings covering physical/mathematical foundation, real-world
  parameter ranges, and failure artefacts (per the repo's pedagogical convention).
  Comments explain intent, not mechanics.
- **Versioning:** SemVer; `v0.1.0` is a GitHub tag after G5 (no PyPI); `v1.0.0` is the
  PyPI release with the full whitepaper after G9; deprecation policy in `CONTRIBUTING.md`.
- **Licensing:** Apache-2.0; third-party data/font/code licences documented; provenance
  for every preset and dataset.

---

## 13. Phases and gates

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | Scaffold: `git init` here; move the superseded plan to `docs/reference/`; keep `archives/`; rewrite `CLAUDE.md`; pyproject, package skeleton, config schema, CLI, CI, pre-commit, docs skeleton, units/spectral types | G0: install, lint, typecheck, tests, docs build |
| 1 | Scene, illuminants, optics, spectral data; examples/notebook 01 | G1: scene/optics validation |
| 2 | Sensor model, flagship preset, datasheet guide, EMVA characterization; example 04 | G2: sensor validation report |
| 3 | ISP stages + pipeline + trace + default ISP; example 02 | G3: ISP validation report |
| 4 | Metrics harness + chart layouts + viz; example 03 | G4: metrics validation report |
| 5 | Consolidation: README, docs site, gallery, benchmarks, validation report, release | **G5: human sign-off; GitHub tag v0.1.0 (no PyPI)** |
| 6 | Glyph dataset, CNN, protocol, sweeps, caching; notebook 04 | G6: ML validation report |
| 7 | Mono vs Bayer study; notebook 05 | G7: study report |
| 8 | NSGA-II auto-tuning; notebook 06 | G8: Pareto report |
| 9 | Metric correlation, guidance, full whitepaper, PyPI release v1.0.0 | **G9: human sign-off; tag v1.0.0** |

No machine-vision code is written before G5. Phase 6 is a new, explicitly approved
stage, not an automatic continuation.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scope too large | Hard gates; v1 non-goals; reuse core types; defer HDR/3D/advanced RCCC |
| No real RAW data | EMVA-style round-trip + analytic invariants + reference libs; honest limitations |
| Spectral data licensing | Use `colour-science` datasets or public-domain sources; cite; allow user CSV |
| Datasheet gaps | Representative values with provenance and assumption flags; sensitivity analysis |
| MHC/reference mismatch | Implement from the paper; cross-check against `colour-demosaicing`; document boundary handling |
| MTF complexity | Validate against analytic PSFs; call it ISO 12233-style, not certified |
| Performance | Cache spectral projections, FFT, float32, precompute, profile |
| ML runtime | Small CNN, small images, MPS, cached data, CLI for full runs |
| Determinism across platforms | Explicit RNG; tolerance-based tests; document float caveats |
| Scope creep | Freeze feature list per phase; roadmap file for v1.1+ |
| Name collision | `camsim` verified free on PyPI; re-check at release |

---

## 15. Decisions made by the builder (check these)

1. **Pydantic v2 + JSON Schema** for configs, with `extra="forbid"` and provenance in
   every preset.
2. **Pure numerical functions + thin stateful wrappers**: `Sensor`, `ISP`, and `Camera`
   hold configuration and precomputed state (FPN maps, projection matrices); algorithms
   stay functions.
3. **`RawFrame` carries all metadata the ISP needs**; no hidden globals.
4. **Stage tracing is opt-in** and returns a dict of arrays/metadata.
5. **A separate `characterization` package** for EMVA-style measurement.
6. **`camsim.metrics` import-independence** is enforced by test.
7. **Python 3.11–3.13, uv, hatchling, ruff, mypy, pytest + Hypothesis.**
8. **MkDocs Material + GitHub Actions + pre-commit.**
9. **sRGB default, Display P3 option; mono output is single-channel, replicated only for
   visualization.**
10. **Default experiment sizes:** 512×512 sensor images; 4× scene oversampling; 64×64
    glyph patches (rendered at 4×); notebooks < 2 min; full sweeps via CLI.
11. **Representative flagship-phone preset** rather than naming a commercial part;
    provenance and assumptions explicit.
12. **No HDR merge, 3-D scenes, lens design, motion blur, or autofocus in v1.**
13. **Auto-exposure policy** as in Section 7.2.2: primary sweeps fixed; AE a separate,
    recorded experiment.
14. **Colour ground truth** via `ColorReference` owned by `camsim.core.color`; the scene
    computes it, metrics consumes it, and non-D65 results cannot silently use D65.
15. **RNG/temperature/determinism contract** as in Section 7.2.1.
16. **External irradiance contract** as in Section 6.2: sensor-grid input, matching
    wavelength grid, explicit resampling helpers, no silent resampling.
17. **RCCC scope:** sensor-level support plus a baseline ISP reconstruction with
    shape/linearity/smoke tests; not held to Bayer ΔE/MTF thresholds; advanced RCCC
    demosaic in v1.1.
18. **RAW-to-CNN diagnostic** in Phase 6, comparing ISP-rendered input with mono RAW and
    Bayer green/luma RAW.
19. **Colour glyph variant** is a Phase 6 extension; grayscale is the primary task.

## 16. Decisions resolved with the project owner

1. **Repository:** initialize this folder as the `camsim` repository; move the superseded
   plan to `docs/reference/`; keep `archives/`; rewrite `CLAUDE.md` for `camsim`. The
   local folder name may remain `IQ_machine_vision`; the package and GitHub repo are
   `camsim`.
2. **Releases:** GitHub tag `v0.1.0` after G5 (no PyPI); PyPI release and full whitepaper
   at `v1.0.0` after G9.
3. **Mono vs Bayer:** per-modality clean-trained models with the same scenes, seeds,
   architecture family, and training budget; mono input 1-channel, Bayer input 3-channel
   RGB; cross-evaluation is a diagnostic, not the headline; luma-matched ablation is
   optional.
4. **Reference lens:** flagship-phone main camera, f/1.8, 24 mm equivalent, ~84° diagonal
   FOV, T = 0.9, 1/1.3" sensor, ~6.4 mm actual focal length; fully configurable.
5. **Primary task:** grayscale glyph classification; coloured-glyph variant as a Phase 6
   extension.
6. **RCCC:** sensor-level support and baseline reconstruction, as in Section 15.17.
7. **Determinism:** bit-identical within one environment; tolerance-based across the CI
   matrix.

### Operational details left for Phase 0

- Exact GitHub owner/repo URL (`camsim` under the owner's account; confirm at release).
- Exact public datasheet/source URLs and citation wording for the preset provenance.
- Final JSON field names and preset values, validated against the schema and Phase 2
  characterization.

These do not block planning; they are settled during Phase 0/2 and recorded in the
preset provenance.

---

## 17. References

- ISO 12233:2017 — Photography — Electronic still picture imaging — Resolution and
  spatial frequency responses.
- EMVA Standard 1288 — Standard for Characterization of Image Sensors and Cameras.
- Malvar, He, Cutler (2004), "High-quality linear interpolation for demosaicing of
  Bayer-patterned color images," ICASSP.
- CIE 15:2004 — Colorimetry; CIE 1931 2° standard observer.
- `colour-science`, `colour-demosaicing`, `scikit-image` documentation and source.
- Public flagship-phone sensor datasheets and teardown reporting (used for
  representative parameter ranges, with provenance).
- The reference plan: `Only for reference - iq_for_machine_vision_project_plan_revised.md`.
