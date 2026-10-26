# Master Specification: `iq-for-machine-vision`

## 1. Project Overview & Architectural Blueprint

`iq-for-machine-vision` is an educational, fully deterministic Python repository designed to simulate a camera pipeline from physical scene radiometry to machine vision inference. Its primary objective is to demonstrate how physical optics, sensor noise, and ISP parameter choices impact downstream machine learning model accuracy compared to traditional human-perceptual image quality (IQ) metrics.

### Core Architectural Split
* **Core Simulation Pipeline (Phases 0–3):** A standalone, fully modular Python package (`src/iqlab/`) that implements scene physics, radiometric conversions, optical blur, sensor noise injection, ISP reconstruction, and standard image quality measurements ($e\text{-SFR}/MTF$, $SNR$, $\Delta E_{2000}$).
* **Applied Engineering Experiments (Phases 4–7):** Downstream analysis tools and Jupytext percent-format notebooks that feed simulated images into a PyTorch neural network to evaluate hardware design trade-offs, machine-centric ISP auto-tuning, and metric-to-task performance correlations.

---

## 2. Target Repository Directory Structure

```
iq-for-machine-vision/
├── .gitignore
├── jupytext.toml
├── pyproject.toml              # Managed via uv (with pip/venv fallback)
├── Makefile                    # setup, test, lint, notebooks, figures
├── README.md                   # Setup, pipeline diagram, gallery, roadmap
├── docs/
│   └── concepts.md             # Glossary + Mermaid pipeline diagram
├── src/
│   └── iqlab/
│       ├── __init__.py
│       ├── units.py            # Radiometry, PPD, unit conversions
│       ├── scene.py            # Target generation (slanted edge, ColorChecker, text glyphs)
│       ├── illuminant.py       # Standard illuminants (D65, 3000K, 4000K)
│       ├── optics.py           # Gaussian/Airy PSF, box downsampling, system MTF
│       ├── sensor.py           # Bayer/Mono CFA, QE, shot/read/dark noise, ADC, gain
│       ├── isp/
│       │   ├── __init__.py
│       │   ├── blc.py          # Black level subtraction
│       │   ├── lsc.py          # Lens shading correction (optional)
│       │   ├── awb.py          # Gray-World & White-Patch white balance
│       │   ├── demosaic.py     # Bilinear & Malvar-He-Cutler (MHC) demosaicing
│       │   ├── color.py        # Ridge-regularized CCM with row-sum constraints
│       │   ├── denoise.py      # Gaussian & Bilateral filtering (Luma/Chroma split)
│       │   ├── sharpen.py      # Unsharp masking (amount, radius, threshold)
│       │   ├── tone.py         # sRGB OETF & global tone mapping curves
│       │   └── pipeline.py     # ISPParams dataclass + unified run() entry point
│       └── metrics/
│           ├── __init__.py
│           ├── mtf.py          # ISO 12233 slanted-edge e-SFR (MTF50, MTF20)
│           ├── color.py        # ΔE76 & ΔE2000 calculations via colour-science
│           ├── snr.py          # Flat patch SNR per channel and luma
│           └── summary.py      # measure_all() unified metric harness
├── tasks/
│   ├── __init__.py
│   ├── text_recog.py           # PyTorch CNN character classifier & evaluator
│   ├── tune.py                 # Optuna wrappers (Task Loss vs Photographic Loss)
│   └── viz.py                  # Standardized figure rendering helpers
├── notebooks/
│   ├── 01_optics_and_sensor.py
│   ├── 02_isp_color_fundamentals.py
│   ├── 03_measuring_iq.py
│   ├── 04_iq_vs_task_performance.py
│   ├── 05_mono_vs_bayer.py
│   ├── 06_toy_autotune.py
│   └── 07_which_metric_predicts_task.py
├── tests/                      # Pytest suite (one test file per module)
└── results/
    └── figures/                # Committed PNG figures generated for README.md
```

---

## 3. Engineering Hygiene & Technical Principles

1. **Package & Environment Management:**
   * Python 3.11 managed via `uv` (`uv sync --all-extras`), with explicit `pip` fallback instructions in `README.md`.
   * Dependencies defined in `pyproject.toml` with optional extras: `[ocr]`, `[tune]`, `[dev]`.
   * Code formatting and linting enforced via `ruff`; unit tests executed via `pytest`.
2. **Notebook Format:**
   * All walkthroughs live in `notebooks/*.py` formatted using Jupytext "percent" notation (`# %%` for code, `# %% [markdown]` for narrative).
   * Do **NOT** commit `.ipynb` binaries to git. A `jupytext.toml` configuration enables local pairing to `.ipynb` if desired.
3. **Linear Space Pipeline Invariance:**
   * Black level subtraction, white balance, demosaicing, denoise, CCM, and sharpening **MUST** operate strictly in linear `float32` space.
   * Non-linear sRGB OETF and global tone mapping curves are applied as the final reconstruction step.
4. **Pedagogical Standards:**
   * Every source file must begin with a docstring detailing: physical/mathematical foundation, real-world parameter ranges, and potential failure artifacts.
   * Core numerical routines (e.g., MHC demosaic, e-SFR, Ridge CCM) are implemented from scratch in NumPy first, then validated against reference packages (`colour-science`, `colour-demosaicing`, `scikit-image`).
   * Comments explain engineering intent ("why") rather than code mechanics ("what").
5. **Execution Speed & Reproducibility:**
   * All stochastic routines (noise generation, PyTorch weights) use fixed random seeds.
   * Every notebook must execute in under 2 minutes on Apple Silicon using PyTorch MPS (with CPU fallback).

---

## 4. Execution Phases

### Phase 0 — Repository Scaffold
* **Deliverables:**
  * Skeleton layout, `pyproject.toml`, `Makefile`, `jupytext.toml`, `.gitignore`, and `docs/concepts.md` (containing a Mermaid pipeline diagram).
  * `README.md` containing macOS setup instructions (`uv sync --all-extras`), VS Code interactive execution guidelines, and a 5-line explanation of why Jupytext percent files outperform `.ipynb` in git workflows.
  * Empty package that imports cleanly, single passing unit test (`pytest`), and initial git commit.

### Phase 1 — Photons to RAW
* **Modules:** `units.py`, `scene.py`, `illuminant.py`, `optics.py`, `sensor.py`, `notebooks/01_optics_and_sensor.py`.
* **Physics & Mathematics:**
  * Render synthetic scenes (slanted edge, 24-patch ColorChecker, text glyphs) as linear reflectance arrays on a $4\times$ oversampled spatial grid.
  * Image-plane illuminance: $E_{\text{img}} \approx E_{\text{scene}} \cdot \frac{R \cdot T}{4 N^2}$, where $N$ is f-number, $T$ is lens transmittance, and $R$ is reflectance.
  * Photon flux conversion: $\approx 4.09 \times 10^3 \text{ photons/s/}\mu\text{m}^2/\text{lux}$ at $555\text{ nm}$.
  * Electron accumulation: $e^- = \text{QE} \cdot \text{Flux} \cdot \text{Pixel Area} \cdot t_{\text{exp}}$.
  * Default Sensor Parameters: $1.12\,\mu\text{m}$ pixel pitch, $f/2.2$, $T=0.9$, $10\text{ ms}$ exposure, Full Well $6000\,e^-$, Read Noise $2\,e^-\text{ rms}$, 10-bit ADC, Black Level 64 DN.
  * Color vs. Mono Sensor: RGGB CFA mosaic with camera-native spectral response ($RGB_{\text{cam}} \neq sRGB$). Monochrome variant omits filter absorption and demosaicing requirements.
  * Optics Model: Gaussian PSF ($\sigma$) combined with diffraction-limited Airy PSF options, box-downsampled for pixel aperture integration.
* **Notebook 01 Outputs:** Display RAW mosaic, channel histograms, noise regime transitions (shot vs. read noise), and Photon Transfer Curve (PTC).
* **Unit Tests:** Verify linear electron scaling with lux, Poisson variance behavior in shot-noise regimes, and clipping bounds at full well / ADC max.

### Phase 2 — ISP Reconstruction Engine
* **Modules:** `isp/blc.py`, `isp/lsc.py`, `isp/awb.py`, `isp/demosaic.py`, `isp/color.py`, `isp/denoise.py`, `isp/sharpen.py`, `isp/tone.py`, `isp/pipeline.py`, `notebooks/02_isp_color_fundamentals.py`.
* **Pipeline Sequencing:**
  $$\text{RAW} \rightarrow \text{BLC} \rightarrow (\text{LSC}) \rightarrow \text{AWB} \rightarrow \text{Demosaic} \rightarrow \text{CCM} \rightarrow \text{Denoise} \rightarrow \text{Sharpen} \rightarrow \text{Tone/Gamma} \rightarrow \text{sRGB}$$
* **Core Algorithms:**
  * **AWB:** Gray-World and White-Patch estimation on ColorChecker gray patches.
  * **Demosaic:** Bilinear interpolation and Malvar-He-Cutler (MHC 2004) gradient-corrected interpolation.
  * **CCM:** $3 \times 3$ matrix fit via least-squares on ColorChecker patches under D65/3000K/4000K illuminants, incorporating **Ridge ($L_2$) regularization** and a **row-sum constraint ($\sum \text{row} = 1$)** to prevent noise explosion under low lux.
  * **Denoise & Sharpen:** Gaussian / Bilateral filtering split across Luma ($Y$) and Chroma ($CbCr$); Unsharp masking with strength, radius, and threshold controls.
  * **Tone Mapping:** sRGB OETF conversion and global tone curve application.
* **Notebook 02 Outputs:** Side-by-side intermediate images after each block, zoomed crops on edges, and demosaic artifact comparisons on a Siemens star target.

### Phase 3 — Image Quality Metrics Harness
* **Modules:** `metrics/mtf.py`, `metrics/color.py`, `metrics/snr.py`, `metrics/summary.py`, `notebooks/03_measuring_iq.py`.
* **Metrics Implementation:**
  * **ISO 12233 Slanted-Edge e-SFR:** Edge detection $\rightarrow$ linear regression angle fit $\rightarrow 4\times$ oversampled Edge Spread Function (ESF) $\rightarrow$ Line Spread Function (LSF via derivative + Hamming window) $\rightarrow \text{FFT} \rightarrow \text{MTF}$. Report $MTF_{50}$ and $MTF_{20}$ in cycles/pixel and cycles/degree.
  * **Demosaic Isolation:** Calculate e-SFR on both raw sensor channels (monochrome / green) and post-demosaic RGB to isolate optical blur from demosaicing artifacts.
  * **Color Accuracy:** Calculate $\Delta E_{76}$ and $\Delta E_{2000}$ per patch against ground-truth ColorChecker values using `colour-science`.
  * **Noise Evaluation:** Calculate SNR (in dB) on flat patches per channel and on luma ($Y$).
* **Unified Interface:** `metrics/summary.py::measure_all(img, chart_layout) -> dict` returning a standardized dictionary of scalar metrics.

---

### Phase 4 — Core Machine Vision Task Engine
* **Modules:** `tasks/text_recog.py`, `notebooks/04_iq_vs_task_performance.py`.
* **Implementation:**
  * Lightweight 3-layer PyTorch CNN character recognition network trained on clean synthetic glyphs (digits/letters).
  * Evaluator accepts ISP output images and computes continuous **Cross-Entropy Task Loss** and **Top-1 Classification Accuracy**.
* **Experimental Sweeps:**
  1. Lux sweep ($5 \rightarrow 1000\text{ lux}$) under fixed optical blur.
  2. Blur sweep (PSF $\sigma$) under high lux.
  3. Denoise and sharpening parameter sweep under low lux ($10\text{ lux}$).
* **Notebook 04 Outputs:** Accuracy vs. Lux plot with $MTF_{50}$ and $SNR$ on twin axes; 2D heatmap of Accuracy vs. $\text{PPD} \times MTF_{20}(\text{cycles/deg})$; comparison crops of human-preferred vs. machine-optimal images.

### Phase 5 — Hardware Comparison: Mono vs. Bayer
* **Modules:** `notebooks/05_mono_vs_bayer.py`.
* **Experimental Setup:** Pass identical synthetic scenes through two sensor architectures under identical optics, pixel size, and illuminance:
  1. Monochrome sensor (no CFA, higher effective QE, no demosaicing required).
  2. RGGB Bayer sensor + ISP reconstruction pipeline.
* **Notebook 05 Outputs:** Quantitative evaluation determining the required $MTF_{20}$ or PPD increase for a Bayer sensor to match a Monochrome sensor's classification accuracy under low-light conditions.

### Phase 6 — Machine-Centric Auto-Tuning
* **Modules:** `tasks/tune.py`, `notebooks/06_toy_autotune.py`.
* **Implementation:**
  * Multi-objective Optuna optimization (NSGA-II) tuning 4–6 ISP parameters (denoise luma/chroma strength, sharpen amount/radius, tone gamma) across multiple lux levels.
  * **Objective 1 (Task Loss):** PyTorch CNN Cross-Entropy Loss.
  * **Objective 2 (Photographic Loss):** Mean $\Delta E_{2000}$ + noise penalty + edge overshoot/halo penalty.
  * **Speed Optimization:** Pre-compute and cache RAW mosaic states and early ISP stages so trial loops evaluate only fast downstream ISP operators (sub-2-minute total execution time).
* **Notebook 06 Outputs:** Pareto front plot illustrating the trade-off between photographic aesthetics and machine accuracy.

### Phase 7 — Metric Correlation & Specification Analysis
* **Modules:** `notebooks/07_which_metric_predicts_task.py`.
* **Implementation:**
  * Aggregate all sweep datasets generated across Phases 4, 5, and 6.
  * Regress CNN Task Loss and Accuracy on traditional metrics ($MTF_{50}$, $MTF_{20}$, $SNR$, $\Delta E_{2000}$, and Acutance).
  * Compute Spearman rank correlations, standardized linear regression coefficients, and decision tree feature importances.
* **Notebook 07 Outputs:** Summary tables and guidance for writing camera hardware specifications tailored to automated machine vision systems.