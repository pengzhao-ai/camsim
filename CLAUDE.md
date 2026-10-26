# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo status: pre-scaffold (Phase 0 not started)

No code exists yet. The authoritative spec is `iq_for_machine_vision_project_plan_revised.md` at the repo root — read it in full before building anything. The `archives/` folder is superseded material; ignore it.

The user will clarify phased-implementation details before the build starts — do not begin scaffolding until they confirm. Everything below describes the planned target state; the commands do not exist until Phase 0 creates them.

## What this repo is

`iq-for-machine-vision` — an educational, fully deterministic Python repo that simulates a camera pipeline from physical scene radiometry to machine-vision inference. It demonstrates how optics, sensor noise, and ISP parameter choices affect downstream ML model accuracy, contrasted against traditional human-perceptual IQ metrics.

Architectural split:

- **Core simulation pipeline (Phases 0–3):** standalone, modular package `src/iqlab/` — scene physics, radiometry, optical blur, sensor noise, ISP reconstruction, IQ measurement (e-SFR/MTF, SNR, ΔE2000).
- **Applied experiments (Phases 4–7):** `tasks/` plus Jupytext notebooks that feed simulated images into a PyTorch CNN to evaluate hardware design trade-offs, machine-centric ISP auto-tuning, and metric-to-task correlations.

## Commands (planned, per the plan)

- Setup (macOS): `uv sync --all-extras`; the README must also give the pip/venv fallback
- `make setup` / `make test` / `make lint` / `make notebooks` / `make figures`
  - `make notebooks` runs every notebook headless (jupytext + nbconvert)
- Lint/format: `ruff`; tests: `pytest`; single test: `uv run pytest tests/test_units.py::test_name`
- Python 3.11, managed with `uv`; `pyproject.toml` extras: `[ocr]`, `[tune]`, `[dev]`

## Pipeline architecture (target)

Physics chain in `src/iqlab/`:

- `units.py` — radiometry, PPD, unit conversions
- `scene.py` — synthetic targets (slanted edge, 24-patch ColorChecker, text glyphs) as linear reflectance on a 4× oversampled grid
- `illuminant.py` — D65 / 3000K / 4000K
- `optics.py` — Gaussian PSF plus optional diffraction-limited Airy PSF, box downsampling for pixel-aperture integration, system MTF
- `sensor.py` — RGGB Bayer or mono (no CFA, higher effective QE), QE, shot/read/dark noise, ADC, gain
  - default sensor: 1.12 µm pitch, f/2.2, T = 0.9, 10 ms exposure, full well 6000 e⁻, read noise 2 e⁻ rms, 10-bit ADC, black level 64 DN

ISP in `src/iqlab/isp/`; `pipeline.py` holds the `ISPParams` dataclass and the unified `run()` entry point:

```
RAW → BLC → (LSC) → AWB → Demosaic → CCM → Denoise → Sharpen → Tone/Gamma → sRGB
```

- AWB: Gray-World and White-Patch estimation
- Demosaic: bilinear and Malvar-He-Cutler (2004)
- CCM: 3×3 least-squares fit on ColorChecker patches per illuminant, with ridge (L2) regularization and a row-sum constraint (Σrow = 1) to prevent noise explosion at low lux
- Denoise/sharpen: Gaussian/bilateral split across luma (Y) and chroma (CbCr); unsharp mask (amount, radius, threshold)

Metrics in `src/iqlab/metrics/`:

- `mtf.py` — ISO 12233 slanted-edge e-SFR → MTF50/MTF20 in cycles/pixel and cycles/degree; computed on raw mono/green channels as well as post-demosaic RGB to isolate demosaicing artifacts from optical blur
- `color.py` — ΔE76 and ΔE2000 per ColorChecker patch via colour-science; `snr.py` — SNR (dB) on flat patches per channel and on luma
- `summary.py::measure_all(img, chart_layout) -> dict` — the unified harness every downstream notebook uses

Tasks live at the repo root in `tasks/` (not under `src/iqlab/`): `text_recog.py` (lightweight 3-layer PyTorch CNN character classifier plus evaluator reporting cross-entropy task loss and top-1 accuracy), `tune.py` (Optuna), `viz.py` (figure helpers).

Phases: 0 scaffold · 1 photons→RAW · 2 ISP engine · 3 metrics harness · 4 CNN task engine + sweeps · 5 mono vs Bayer · 6 auto-tune (NSGA-II Pareto) · 7 metric correlation.

## Non-negotiable conventions

- **Linear-space invariance:** BLC, WB, demosaic, denoise, CCM, and sharpening operate strictly in linear float32 space; the non-linear sRGB OETF and tone curve are applied as the final reconstruction step.
- **Notebooks:** Jupytext percent format (`# %%` code, `# %% [markdown]` narrative) in `notebooks/*.py`; never commit `.ipynb` files; `jupytext.toml` enables local pairing. Every notebook must execute in under 2 minutes on Apple Silicon.
- **Determinism:** fixed random seeds for all stochastic routines (noise generation, PyTorch weights); the repo must be fully deterministic.
- **From scratch first:** core numerical routines (MHC demosaic, e-SFR, ridge CCM, …) are implemented in NumPy first, then validated against reference packages (`colour-science`, `colour-demosaicing`, `scikit-image`) in tests or notebook cells.
- **Pedagogy:** every source file opens with a docstring covering the physical/mathematical foundation, real-world parameter ranges, and potential failure artifacts. Comments explain engineering intent ("why"), not code mechanics ("what").
- PyTorch runs on MPS when available, with CPU fallback.
- **Tests:** one file per module in `tests/`; Phase 1 physics checks include linear electron scaling with lux, Poisson variance in the shot-noise regime, and clipping at full well / ADC max.
- `results/figures/` holds committed PNG figures used by the README gallery.
