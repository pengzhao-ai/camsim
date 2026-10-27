# camsim Core Pipeline and Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the reusable `camsim` sensor + ISP toolbox and its independent image-quality metrics harness (Phases 0–5), ending in a human sign-off gate before any machine-vision work.

**Architecture:** A `src/`-layout Python package with three clean layers: simulation (`scene`, `optics`, `sensor`, `isp`), metrics (`metrics`, import-independent), and later machine vision (`tasks`, gated). The sensor consumes a validated `SpectralIrradiance` and returns a self-describing `RawFrame`; the ISP consumes `RawFrame` and returns a `RenderedImage` with optional per-stage trace. Physics is validated by analytic invariants, EMVA 1288-style round-trip characterization, and reference-library cross-checks.

**Tech Stack:** Python 3.11–3.13, `uv`, `hatchling`, NumPy, SciPy, Pydantic v2, `colour-science`, Matplotlib, Pillow, pytest, Hypothesis, ruff, mypy, MkDocs Material, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-26-camsim-design.md`

---

## Global Constraints

- Python 3.11–3.13; `uv sync --all-extras`; committed `uv.lock`; project-local `.venv/` is gitignored.
- `src/camsim/` layout; Apache-2.0; package, import, and PyPI name `camsim`.
- Pydantic v2 configs with `extra="forbid"`, units in every field name, and provenance in every preset.
- Determinism: explicit `np.random.Generator`; per-call `seed=` overrides; temperature overrides; bit-identical within one environment, tolerance-based across CI.
- ISP linear-space invariant: DPC, BLC, LSC, AWB, demosaic, CCM, denoise, and sharpen operate in linear float32; tone/output transfer is the only non-linear step.
- `camsim.metrics` must not import `camsim.sensor`, `camsim.isp`, `camsim.scene`, or `camsim.optics`.
- No `.ipynb` committed; notebooks are Jupytext percent-format `.py`; `make notebooks` executes headless; each notebook < 2 minutes on Apple Silicon.
- No machine-vision code before G5 sign-off.
- Reference lens: f/1.8, 24 mm equivalent, ~84° diagonal FOV, T = 0.9, 1/1.3" sensor, ~6.4 mm actual focal length.
- Reference preset: LCG 100 µV/e⁻ / 6000 e⁻ / 2 e⁻ rms; HCG 300 µV/e⁻ / 2000 e⁻ / 0.7 e⁻ rms; 10-bit ADC, 0.8 V full scale, black level 64 DN.
- Coverage ≥ 90% at G5; ruff and mypy strict on `src/` must pass.
- Test snippets show the assertions; every test file imports `numpy as np` and `pytest` at the top, plus the module under test.
- Every commit message ends with: `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

## Review Focus

Five input classes the spec implies but no single task naturally owns. Each gets a test in the listed task.

1. **Mismatched external irradiance** (wrong spatial shape/pitch, wavelength coverage, or unit): must raise a precise error or require an explicit resampling helper, never silently produce a wrong RAW. Tests in Tasks 2 and 15.
2. **Non-D65 colour ground truth**: 3000 K/4000 K ΔE must not silently fall back to D65; a missing non-D65 `ColorReference` raises. Tests in Tasks 28, 31, and 32.
3. **Extreme exposure/gain/empty input**: saturation, black frames, negative irradiance, and NaN must clip or raise deterministically, with no NaN in RAW. Tests in Tasks 2, 14, and 15.
4. **Mode confusion mono vs RGB**: mono through colour metrics, or RGB through the mono path, must skip or raise clearly rather than crash or return nonsense. Tests in Tasks 25, 31, and 32.
5. **Seed/call-order determinism**: the same per-call seed must reproduce a frame after other frames, and fixed-pattern maps must stay stable. Tests in Tasks 13, 15, and 18.

## File Structure

| Area | Files | Responsibility |
|---|---|---|
| Core types | `src/camsim/core/{units,spectral,frames,color,constants}.py` | Units, wavelength grids, spectral cubes, RAW/rendered frames, colour references |
| Config | `src/camsim/config/{schema,io}.py`, `presets/` | Pydantic schemas, JSON load/validate, datasheet-grounded presets |
| Scene/optics | `src/camsim/scene/*`, `src/camsim/optics/*` | Illuminants, spectral datasets, targets, glyphs, lens, PSFs, propagation |
| Sensor | `src/camsim/sensor/*` | CFA/projection, noise/FPN, gain/ADC, binning, `Sensor.expose` |
| Characterization | `src/camsim/characterization/*` | EMVA 1288-style measurement and reports |
| ISP | `src/camsim/isp/*` | DPC, BLC, LSC, AWB, demosaic, CCM, denoise, sharpen, tone, pipeline trace |
| Metrics | `src/camsim/metrics/*` | Chart layouts, e-SFR/MTF, SNR, ΔE, acutance, `measure_all` |
| Camera | `src/camsim/camera/camera.py` | Convenience orchestration and auto-exposure |
| Visualization | `src/camsim/viz/*` | Consistent plots for spectra, sensor chain, ISP trace, metrics, experiments |
| CLI | `src/camsim/cli.py` | `validate`, `schema`, `info`, later `render`, `characterize`, `measure` |
| Tests | `tests/test_*.py` | One file per module plus validation, determinism, import-independence |
| Examples/notebooks | `examples/*.py`, `notebooks/*.py` | Tested, reproducible walkthroughs |
| Results | `results/{figures,data,reports}/` | Committed gallery figures, sweep data, validation reports |

---

## Phase 0 — Foundation

### Task 1: Repository, packaging, tooling, CI, and repo hygiene

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `Makefile`, `jupytext.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `LICENSE`, `README.md`, `mkdocs.yml`, `docs/index.md`, `docs/concepts.md`, `docs/architecture.md`, `src/camsim/__init__.py`, `src/camsim/_version.py`, `tests/test_package.py`
- Modify: `CLAUDE.md`
- Move: `Only for reference - iq_for_machine_vision_project_plan_revised.md` → `docs/reference/`
- Keep: `archives/`

**Interfaces:**
- Produces: installable `camsim` package, `camsim.__version__`, `make` targets, CI and docs build, `camsim` console entry point stub.

- [ ] **Step 1: Write the failing package test**

```python
# tests/test_package.py
import camsim

def test_import_and_version():
    assert camsim.__version__ == "0.1.0.dev0"
    assert hasattr(camsim, "__all__")
```

- [ ] **Step 2: Run it to confirm failure**

Run: `uv run pytest tests/test_package.py -v`
Expected: FAIL because the package does not exist.

- [ ] **Step 3: Install uv if needed, initialize the repo, and create the package skeleton**

Run: `uv --version` (if missing: `curl -LsSf https://astral.sh/uv/install.sh | sh` and restart the shell).
Then:
```bash
git init
mkdir -p src/camsim/{core,config,scene,optics,sensor,isp,metrics,characterization,camera,tasks,viz,io}
mkdir -p tests docs/reference docs/superpowers/{specs,plans} examples notebooks results/{figures,data,reports,cache}
mv "Only for reference - iq_for_machine_vision_project_plan_revised.md" docs/reference/
```
Create `pyproject.toml` with hatchling, `requires-python = ">=3.11,<3.14"`, core deps (`numpy`, `scipy`, `pydantic>=2`, `matplotlib`, `colour-science`, `pillow`), extras `[ml]`, `[tune]`, `[docs]`, `[dev]`, `[all]`; `[dev]` includes `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`, `pre-commit`, `jupytext`, `nbconvert`, `colour-demosaicing`, `scikit-image`, `pandas`, `pyarrow`; add ruff/mypy/pytest/coverage config and entry point `camsim = "camsim.cli:main"`. Create `src/camsim/__init__.py` exporting `__version__ = "0.1.0.dev0"`, `__all__`, and a PEP 562 lazy `__getattr__` export map so `import camsim.metrics` does not eagerly import `sensor`/`isp`/`scene`/`optics`; later tasks add their public names to the map. Add `.venv/`, `.ipynb`, `results/cache/`, caches, and build artifacts to `.gitignore`. Rewrite `CLAUDE.md` for the camsim architecture, commands, and conventions.

- [ ] **Step 4: Sync, run the test, lint, typecheck, and docs build**

Run:
```bash
uv sync --all-extras
uv run pytest tests/test_package.py -v
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run mkdocs build --strict
```
Expected: test passes; lint/typecheck/docs succeed.

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "chore: scaffold camsim package, tooling, and CI

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: Core units and spectral types

**Files:**
- Create: `src/camsim/core/units.py`, `src/camsim/core/spectral.py`, `src/camsim/core/constants.py`, `tests/test_units.py`, `tests/test_spectral.py`

**Interfaces:**
- Produces:
  - `photon_energy_j(wavelength_nm: ArrayLike) -> NDArray[np.float64]`
  - `irradiance_w_to_photons(irradiance: ArrayLike, wavelength_nm: ArrayLike) -> NDArray[np.float64]`
  - `photons_to_irradiance_w(photons: ArrayLike, wavelength_nm: ArrayLike) -> NDArray[np.float64]`
  - `WavelengthGrid(nm: NDArray[np.float64])` — validated 1-D, strictly increasing, uniform, positive
  - `SpectralIrradiance(data, wavelengths_nm, unit, pixel_pitch_um=None)` with `from_array`, `to_unit`, `resample_spatial`, `resample_wavelengths`
  - Modify `src/camsim/__init__.py` to add `SpectralIrradiance` and `WavelengthGrid` to the lazy export map

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_units.py
import numpy as np
import pytest
from camsim.core import units

def test_photon_energy_555nm():
    assert units.photon_energy_j(555.0) == pytest.approx(3.578e-19, rel=1e-3)

def test_irradiance_photon_roundtrip():
    w = np.array([450.0, 550.0, 650.0])
    e = np.array([1.0, 2.0, 0.5])
    np.testing.assert_allclose(
        units.photons_to_irradiance_w(units.irradiance_w_to_photons(e, w), w),
        e, rtol=1e-12,
    )

# tests/test_spectral.py
from camsim.core.spectral import SpectralIrradiance, WavelengthGrid

def test_wavelength_grid_rejects_nonuniform():
    with pytest.raises(ValueError, match="uniform"):
        WavelengthGrid(np.array([400.0, 410.0, 430.0]))

def test_spectral_irradiance_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        SpectralIrradiance.from_array(
            -np.ones((2, 2, 3)), wavelengths_nm=np.array([450., 550., 650.]),
            unit="W/m^2/nm",
        )

def test_resample_spatial_conserves_energy():
    grid = np.array([450.0, 550.0, 650.0])
    cube = np.ones((8, 8, 3))
    out = SpectralIrradiance.from_array(cube, grid, "W/m^2/nm").resample_spatial((4, 4))
    assert out.data.shape == (4, 4, 3)
    np.testing.assert_allclose(out.data.mean(), 1.0)
```

Additional tests: `test_unit_conversion_roundtrip`, `test_resample_wavelengths_conserves_energy`, `test_resample_wavelengths_raises_on_uncovered_range`, `test_nan_irradiance_raises`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_units.py tests/test_spectral.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the signatures**

`units.py`: constants `H_PLANCK`, `C_LIGHT`; photon energy `h*c/λ` with λ in metres; conversions per nm bandwidth. `spectral.py`: frozen `WavelengthGrid` with `__post_init__` validation; `SpectralIrradiance` dataclass with shape `(H, W, L)`, unit literal, non-negative/NaN checks, area-conserving spatial box resampling, and overlap-weighted wavelength integration that raises when the target range is not covered unless `allow_zero_fill=True`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_units.py tests/test_spectral.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/core/units.py src/camsim/core/spectral.py src/camsim/core/constants.py tests/test_units.py tests/test_spectral.py
git commit -m "feat: add radiometric units and spectral irradiance types

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: Frames, stage outputs, and colour references

**Files:**
- Create: `src/camsim/core/frames.py`, `src/camsim/core/color.py`, `tests/test_frames.py`, `tests/test_color_reference.py`

**Interfaces:**
- Produces:
  - `CFAPattern` enum: `RGGB`, `BGGR`, `GRBG`, `GBRG`, `MONO`, `RCCC`
  - `RawFrame` dataclass with `data`, `cfa`, `bit_depth`, `black_level_dn`, `exposure_s`, `analog_gain_db`, `digital_gain`, `mode`, `temperature_c`, `seed`, `frame_index`, `pixel_pitch_um`, `defect_mask`, `pdaf_mask`, `sensor_config_hash`, and `with_data(data) -> RawFrame` for tests that need a modified copy
  - `StageOutput`, `RenderedImage` (display data, `linear_rgb`, `color_space`, `channels`, `stages`, metadata)
  - `ColorReference` (per-patch Lab, white point, illuminant, observer, provenance)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_frames.py
import numpy as np
import pytest
from camsim.core.frames import CFAPattern, RawFrame, RenderedImage

def test_raw_frame_validates_bit_depth_and_black_level():
    raw = RawFrame(
        data=np.zeros((4, 4), dtype=np.uint16), cfa=CFAPattern.RGGB,
        bit_depth=10, black_level_dn=64, exposure_s=1e-3, analog_gain_db=0.0,
        digital_gain=1.0, mode="LCG", temperature_c=25.0, seed=1, frame_index=0,
        pixel_pitch_um=1.12, sensor_config_hash="test",
    )
    assert raw.black_level_dn == 64
    with pytest.raises(ValueError, match="black level"):
        RawFrame(**{**raw.__dict__, "black_level_dn": 2000})

def test_rendered_image_mono_and_rgb_shapes():
    rgb = RenderedImage(data=np.zeros((4, 4, 3), np.float32), color_space="sRGB", channels="RGB")
    mono = RenderedImage(data=np.zeros((4, 4, 1), np.float32), color_space="sRGB", channels="MONO")
    assert rgb.channels == "RGB"
    assert mono.channels == "MONO"
    with pytest.raises(ValueError, match="channels"):
        RenderedImage(data=np.zeros((4, 4, 3), np.float32), color_space="sRGB", channels="MONO")
```

Additional tests: `test_color_reference_json_roundtrip`, `test_raw_frame_metadata_json_roundtrip`, `test_rendered_image_display_range`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_frames.py tests/test_color_reference.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the dataclasses**

Validate bit depth (8–16), black level `< 2**bit_depth`, CFA/mode consistency, display range `[0, 1]` (allow small tolerance), and channel/shape consistency. `ColorReference` stores `lab: (n, 3)`, `white_point_xy`, `illuminant_name`, `observer`, `provenance`; provide `to_dict`/`from_dict`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_frames.py tests/test_color_reference.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/core/frames.py src/camsim/core/color.py tests/test_frames.py tests/test_color_reference.py
git commit -m "feat: add RAW, rendered image, and color reference contracts

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: Configuration schema, presets, and CLI skeleton

**Files:**
- Create: `src/camsim/config/schema.py`, `src/camsim/config/io.py`, `src/camsim/config/presets/sensors/flagship_phone_50mp.json`, `src/camsim/config/presets/isp/default.json`, `src/camsim/cli.py`, `tests/test_config.py`, `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `SensorConfig`, `ISPConfig`, `LensConfig` Pydantic models
  - `load_sensor(path) -> SensorConfig`, `load_isp(path) -> ISPConfig`, `validate_config(path) -> ValidationReport`
  - `sensor_schema() -> dict`, `isp_schema() -> dict`
  - CLI: `camsim validate PATH`, `camsim schema {sensor,isp}`, `camsim info`

**Shared test fixtures (add to `tests/conftest.py` in this task):**
- `sensor_config` — `load_sensor` of the flagship preset.
- `isp_config` — `load_isp` of the default preset.
- `sensor` — `camsim.Sensor(sensor_config, seed=1)` (added in Task 15).
- `raw_frame` — a 32×32 `Sensor.expose` result at low lux (added in Task 15).
- `mono_sensor_config` — `sensor_config.model_copy(update={"cfa": CFAPattern.MONO})`.
- `mono_raw_frame` — the mono sensor output (added in Task 15).
- `camera`, `scene`, `bright_scene`, `dark_scene`, `very_dark_scene` — added in Task 33.

Fixtures that depend on later tasks are added to `tests/conftest.py` in those tasks so the
plan stays executable in order.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
import json
import pytest
from pydantic import ValidationError
from camsim.config.io import load_sensor, validate_config

def test_valid_sensor_preset_loads():
    cfg = load_sensor("src/camsim/config/presets/sensors/flagship_phone_50mp.json")
    assert cfg.pixel.pitch_um == pytest.approx(1.12)
    assert cfg.adc.bits == 10
    assert cfg.modes["HCG"].full_well_e == 2000

def test_unknown_field_rejected(tmp_path):
    data = json.loads(open("src/camsim/config/presets/sensors/flagship_phone_50mp.json").read())
    data["unknown_field"] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        load_sensor(path)

def test_full_well_vs_adc_warning(tmp_path):
    data = json.loads(open("src/camsim/config/presets/sensors/flagship_phone_50mp.json").read())
    data["adc"]["full_scale_uv"] = 100.0
    path = tmp_path / "adc_limited.json"
    path.write_text(json.dumps(data))
    report = validate_config(path)
    assert any("ADC" in w and "full well" in w for w in report.warnings)
```

Additional tests: `test_ccm_row_sum_rejected`, `test_schema_export_contains_units`, `test_cli_validate_exit_codes`, `test_cli_schema_prints_json`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_config.py tests/test_cli.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the schema and CLI**

Models with `model_config = ConfigDict(extra="forbid")`, units in field names, range validators, and cross-field validators for full well × conversion gain vs ADC full scale, HCG ≤ LCG full well, QE ∈ [0,1], crosstalk/CCM row sums. `ValidationReport` carries errors and warnings. `cli.py` uses `argparse` subcommands and prints JSON for schema. Create the flagship preset with the values in Global Constraints and provenance; create the default ISP config with stages enabled and sRGB output.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/config src/camsim/cli.py tests/test_config.py tests/test_cli.py
git commit -m "feat: add validated sensor/ISP config schema and CLI

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: Phase 0 docs, CI workflows, and G0 gate

**Files:**
- Create: `.github/workflows/{ci,docs}.yml`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `CITATION.cff`, `NOTICE`, `THIRD_PARTY.md`, `docs/datasheet_mapping.md` (stub), `docs/validation.md` (stub), `Makefile` targets
- Modify: `README.md`, `docs/concepts.md`, `docs/architecture.md`, `mkdocs.yml`

**Interfaces:**
- Produces: `make setup|test|lint|typecheck|docs|notebooks|figures|clean`; CI runs lint, typecheck, tests, docs, build.

- [ ] **Step 1: Write a failing CI-config test**

```python
# tests/test_ci_config.py
from pathlib import Path

def test_g0_docs_and_workflows_exist():
    assert Path(".github/workflows/ci.yml").exists()
    assert Path(".github/workflows/docs.yml").exists()
    assert Path("CONTRIBUTING.md").exists()
    assert Path("docs/validation.md").exists()
    assert Path("docs/datasheet_mapping.md").exists()
    text = Path(".github/workflows/ci.yml").read_text()
    assert "3.11" in text and "3.12" in text and "3.13" in text
    assert "ruff" in text and "mypy" in text and "pytest" in text
```

- [ ] **Step 2: Run it to confirm failure**

Run: `uv run pytest tests/test_ci_config.py -v`
Expected: FAIL because the workflow is missing.

- [ ] **Step 3: Add docs, CI, and Makefile targets**

CI: Ubuntu + macOS, Python 3.11/3.12/3.13, `uv sync --all-extras`, ruff, mypy, pytest with coverage, `uv build`, `mkdocs build --strict`. Docs workflow deploys to GitHub Pages on `main`. Makefile targets call the same commands. README includes badges, install (uv + pip/venv fallback), a 10-line quickstart placeholder, pipeline diagram, and roadmap. `docs/concepts.md` has a Mermaid pipeline diagram. Add Apache-2.0 `LICENSE`, third-party notices for `colour-science`, DejaVu Sans, and the ColorChecker data source, plus `CITATION.cff`.

- [ ] **Step 4: Run the G0 gate**

Run:
```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv run mkdocs build --strict
uv run python -c "import camsim; print(camsim.__version__)"
```
Expected: all pass; version prints `0.1.0.dev0`.

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "docs: add Phase 0 documentation, CI, and G0 gate

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Phase 1 — Scene and Optics

### Task 6: Spectral datasets, illuminants, and colour ground truth

**Files:**
- Create: `src/camsim/scene/spectral_data.py`, `src/camsim/scene/illuminants.py`, `data/spectra/` CSVs, `tests/test_spectral_data.py`
- Data: ColorChecker 2005 spectra, D65/3000 K/4000 K, CIE 1931 2° observer, photopic V(λ); all sourced and cited in `THIRD_PARTY.md`

**Interfaces:**
- Produces:
  - `load_colorchecker() -> ColorCheckerData` (24 patches, spectra on a grid, names)
  - `illuminant(name: str) -> SpectralPowerDistribution` (`"D65"`, `"3000K"`, `"4000K"`)
  - `lux_to_spectral_irradiance(lux: float, spd) -> SpectralIrradiance`
  - `reference_lab(illuminant: str) -> ColorReference`
  - `photopic_luminous_efficiency(wavelength_nm) -> NDArray`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_spectral_data.py
import numpy as np
import pytest
from camsim.scene import spectral_data, illuminants

def test_colorchecker_has_24_patches():
    cc = spectral_data.load_colorchecker()
    assert len(cc.patches) == 24
    assert cc.spectra.shape[-1] == len(cc.wavelengths_nm)

def test_lux_normalization_d65():
    e = illuminants.lux_to_spectral_irradiance(1.0, illuminants.illuminant("D65"))
    v = illuminants.photopic_luminous_efficiency(e.wavelengths_nm.nm)
    luminous = 683.0 * np.trapezoid(e.data[0, 0] * v, e.wavelengths_nm.nm * 1e-9)
    assert luminous == pytest.approx(1.0, rel=1e-3)

def test_reference_lab_changes_with_illuminant():
    d65 = illuminants.reference_lab("D65")
    warm = illuminants.reference_lab("3000K")
    assert np.max(np.abs(d65.lab - warm.lab)) > 1.0
```

Additional tests: `test_d65_reference_matches_colour_science`, `test_photopic_peak_at_555nm`, `test_illuminant_provenance_present`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_spectral_data.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the data loaders and photometric conversion**

Use `colour-science` datasets where licensed; vendor CSVs where needed, with source URLs in `THIRD_PARTY.md`. Interpolate all spectra onto a common 360–830 nm, 5 nm grid. Implement `lux_to_spectral_irradiance` per Section 7.1.1 of the spec. Compute `ColorReference` by integrating patch reflectance × illuminant × CMFs → XYZ → Lab under the illuminant white point.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_spectral_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/scene/spectral_data.py src/camsim/scene/illuminants.py data/spectra tests/test_spectral_data.py THIRD_PARTY.md
git commit -m "feat: add spectral datasets, illuminants, and color ground truth

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: Lens configuration and PSFs

**Files:**
- Create: `src/camsim/optics/lens.py`, `src/camsim/optics/psf.py`, `tests/test_lens.py`, `tests/test_psf.py`

**Interfaces:**
- Produces:
  - `LensConfig(f_number, focal_length_mm, sensor_diagonal_mm, transmittance, vignetting=None)` with derived `field_of_view_deg` and `pixels_per_degree(pixel_pitch_um)`
  - `gaussian_psf(sigma_px, size=None) -> NDArray`
  - `airy_psf(f_number, wavelength_nm, pixel_pitch_um, size=None) -> NDArray`
  - `psf_mtf(psf, frequencies_cp) -> NDArray`
  - `psf_mtf50(psf) -> float`
  - Modify `src/camsim/__init__.py` to add `LensConfig` to the lazy export map

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_psf.py
import numpy as np
import pytest
from camsim.optics.psf import gaussian_psf, psf_mtf, psf_mtf50, airy_psf

def test_gaussian_psf_normalized():
    psf = gaussian_psf(1.0, size=21)
    assert psf.sum() == pytest.approx(1.0)

def test_gaussian_mtf50_analytic():
    assert psf_mtf50(gaussian_psf(1.0, size=31)) == pytest.approx(0.1874, abs=0.01)

def test_airy_first_zero_radius():
    psf = airy_psf(1.8, 550.0, 1.12, size=41)
    assert psf.shape == (41, 41)
    assert psf.sum() == pytest.approx(1.0)

# tests/test_lens.py
def test_lens_fov_for_reference_phone():
    lens = LensConfig(f_number=1.8, focal_length_mm=6.4, sensor_diagonal_mm=11.46, transmittance=0.9)
    assert lens.field_of_view_deg == pytest.approx(84.0, abs=1.0)
```

Additional tests: `test_psf_mtf_dc_is_one`, `test_box_psf_sinc`, `test_lens_spectral_transmittance`, `test_lens_rejects_invalid_f_number`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_lens.py tests/test_psf.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the lens and PSF functions**

Gaussian PSF from a normalized sampled Gaussian; Airy PSF from `(2 J1(x)/x)²` with `x = π r / (1.22 λ N / pitch)`; normalize to sum 1. MTF via FFT magnitude with DC normalization. Lens FOV `2 atan(d/(2f))`; PPD from pixel pitch and focal length.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_lens.py tests/test_psf.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/optics/lens.py src/camsim/optics/psf.py tests/test_lens.py tests/test_psf.py
git commit -m "feat: add reference lens model and Gaussian/Airy PSFs

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 8: Scene targets, propagation, oversampling, and vignetting

**Files:**
- Create: `src/camsim/scene/targets.py`, `src/camsim/optics/propagation.py`, `src/camsim/optics/vignetting.py`, `tests/test_targets.py`, `tests/test_propagation.py`

**Interfaces:**
- Produces:
  - `render(target, illuminant, lux, lens, optics, pixel_pitch_um, shape, oversampling=4, seed=None) -> SceneResult`
  - `SceneResult(irradiance, layout, reference, ground_truth)`
  - `slanted_edge_target`, `colorchecker_target`, `flat_patch_target`, `siemens_star_target`
  - `apply_psf(cube, psf)`, `box_downsample(cube, factor)`, `vignetting_map(shape, lens, ...)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_propagation.py
import numpy as np
from camsim.optics.propagation import box_downsample, apply_psf
from camsim.optics.psf import gaussian_psf

def test_box_downsample_conserves_mean():
    x = np.ones((8, 8, 3))
    assert box_downsample(x, 2).mean() == pytest.approx(1.0)

def test_energy_conservation_through_psf_and_downsample():
    x = np.zeros((64, 64, 3)); x[30:34, 30:34, :] = 1.0
    y = box_downsample(apply_psf(x, gaussian_psf(1.0, size=11)), 4)
    assert y.sum() * 16 == pytest.approx(x.sum(), rel=1e-2)

def test_slanted_edge_angle_is_estimated():
    from camsim.scene.targets import slanted_edge_target
    from camsim.scene.targets import estimate_edge_angle
    img = slanted_edge_target((256, 256), angle_deg=5.0)
    assert estimate_edge_angle(img) == pytest.approx(5.0, abs=0.5)
```

Additional tests: `test_colorchecker_layout_has_24_patches`, `test_vignetting_cos4_falloff`, `test_render_returns_spectral_irradiance`, `test_render_is_seed_deterministic`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_targets.py tests/test_propagation.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement targets, propagation, and rendering**

Render targets on a 4× grid in reflectance, multiply by the illuminant and lens, convolve with the PSF per wavelength (or the documented fast weighted-kernel path), apply `cos⁴` vignetting, and box-downsample to the sensor grid. Return a `SceneResult` with the spectral cube, chart layout, and spectrally computed `ColorReference`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_targets.py tests/test_propagation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/scene/targets.py src/camsim/optics/propagation.py src/camsim/optics/vignetting.py tests/test_targets.py tests/test_propagation.py
git commit -m "feat: add scene targets, optics propagation, and rendering

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 9: Glyph scene generator

**Files:**
- Create: `src/camsim/scene/glyphs.py`, `tests/test_glyphs.py`

**Interfaces:**
- Produces:
  - `GLYPH_CLASSES: tuple[str, ...]` (36 entries)
  - `render_glyph(char, shape, seed, paper_spectrum=None, ink_spectrum=None, rotation_deg=0.0) -> NDArray`
  - `glyph_dataset(n, seed, shape, split="train") -> GlyphDataset` with deterministic samples

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_glyphs.py
import numpy as np
import pytest
from camsim.scene.glyphs import GLYPH_CLASSES, render_glyph, glyph_dataset

def test_glyph_classes_are_36():
    assert len(GLYPH_CLASSES) == 36
    assert "0" in GLYPH_CLASSES and "A" in GLYPH_CLASSES

def test_glyph_render_is_deterministic():
    a = render_glyph("A", (64, 64), seed=7)
    b = render_glyph("A", (64, 64), seed=7)
    c = render_glyph("A", (64, 64), seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)

def test_glyph_dataset_split_has_no_overlap():
    train = glyph_dataset(20, seed=1, shape=(64, 64), split="train")
    test = glyph_dataset(20, seed=2, shape=(64, 64), split="test")
    assert train.classes.shape == (20,) and test.classes.shape == (20,)
```

Additional tests: `test_glyph_ink_and_paper_spectra`, `test_glyph_rendered_at_4x`, `test_glyph_rotation_bounds`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_glyphs.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement glyph rendering**

Use Pillow with the Matplotlib-bundled DejaVu Sans font; render at 4× resolution with random size/position/rotation; map the antialiased mask to reflectance between paper and ink spectra; return a spectral reflectance cube. Generate deterministic datasets from explicit seeds, with disjoint train/test seeds.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_glyphs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/scene/glyphs.py tests/test_glyphs.py
git commit -m "feat: add deterministic spectral glyph scene generator

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 10: Visualisation foundation and notebook 01

**Files:**
- Create: `src/camsim/viz/style.py`, `src/camsim/viz/spectra.py`, `examples/01_spectral_scene_to_raw.py`, `notebooks/01_optics_and_sensor.py`, `tests/test_example_01.py`
- Modify: `Makefile` (`figures`, `notebooks`)

**Interfaces:**
- Produces: `set_style()`, `plot_spectrum`, `plot_psf`, `plot_mtf`, `save_figure`; example/notebook generate `results/figures/01_*.png`.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_example_01.py
import os, subprocess, sys
from pathlib import Path

def test_example_01_runs_and_writes_figures(tmp_path):
    env = {**os.environ, "CAMSIM_FIG_DIR": str(tmp_path)}
    result = subprocess.run([sys.executable, "examples/01_spectral_scene_to_raw.py"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert any(tmp_path.glob("*.png"))
```

- [ ] **Step 2: Run it to confirm failure**

Run: `uv run pytest tests/test_example_01.py -v`
Expected: FAIL because the example is missing.

- [ ] **Step 3: Implement the plotting helpers and example/notebook**

Set a colorblind-friendly Matplotlib style. Plot illuminant SPDs, QE/CFA curves, PSF profiles, and MTF curves. The example renders a scene through optics and saves a contact sheet; the Jupytext notebook narrates the same steps with markdown cells. All examples and notebooks honor `CAMSIM_FIG_DIR` (default `results/figures`). Keep it under 2 minutes.

- [ ] **Step 4: Run the test and notebook headless**

Run:
```bash
uv run pytest tests/test_example_01.py -v
uv run jupytext --to ipynb --execute notebooks/01_optics_and_sensor.py --output /tmp/01.ipynb
```
Expected: PASS and exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/viz examples/01_spectral_scene_to_raw.py notebooks/01_optics_and_sensor.py tests/test_example_01.py Makefile
git commit -m "feat: add optics/sensor visualization foundation and notebook 01

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 11: G1 validation report

**Files:**
- Create: `results/reports/g1_scene_optics.md`, `results/figures/g1_*.png`, `tests/test_validation_g1.py`
- Modify: `docs/validation.md`

**Interfaces:**
- Produces: a G1 report with the checks in Section 10 of the spec.

- [ ] **Step 1: Write the failing G1 validation tests**

```python
# tests/test_validation_g1.py
import numpy as np, pytest
from camsim.scene.illuminants import lux_to_spectral_irradiance, illuminant, photopic_luminous_efficiency
from camsim.optics.psf import gaussian_psf, psf_mtf50
from camsim.optics.propagation import box_downsample

def test_g1_lux_normalization():
    e = lux_to_spectral_irradiance(100.0, illuminant("D65"))
    v = photopic_luminous_efficiency(e.wavelengths_nm.nm)
    assert 683.0 * np.trapezoid(e.data * v, e.wavelengths_nm.nm * 1e-9) == pytest.approx(100.0, rel=1e-3)

def test_g1_analytic_mtf():
    assert psf_mtf50(gaussian_psf(1.0, size=31)) == pytest.approx(0.1874, abs=0.01)

def test_g1_downsample_mean():
    assert box_downsample(np.ones((8, 8, 3)), 2).mean() == pytest.approx(1.0)
```

- [ ] **Step 2: Run them to confirm failure**

Run: `uv run pytest tests/test_validation_g1.py -v`
Expected: FAIL until the report/checks are complete.

- [ ] **Step 3: Generate the G1 report and figures**

Add a script or notebook cell that writes the G1 report with measured values, tolerances, and figures: lux normalization, PSF MTF vs analytic, energy conservation, and edge-angle accuracy. Run the full test suite.

- [ ] **Step 4: Run the G1 gate**

Run:
```bash
uv run pytest tests/test_validation_g1.py tests/test_units.py tests/test_spectral.py tests/test_lens.py tests/test_psf.py tests/test_targets.py tests/test_propagation.py tests/test_glyphs.py -v
uv run ruff check . && uv run mypy src
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add results/reports/g1_scene_optics.md results/figures tests/test_validation_g1.py docs/validation.md
git commit -m "test: add G1 scene/optics validation report

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Phase 2 — Sensor

### Task 12: CFA patterns and spectral projection

**Files:**
- Create: `src/camsim/sensor/cfa.py`, `tests/test_cfa.py`

**Interfaces:**
- Produces:
  - `cfa_color_at(pattern, y, x) -> Literal["R","G","B","C","M"]`
  - `cfa_masks(pattern, shape) -> dict[str, NDArray[bool]]`
  - `spectral_projection(qe_csv, cfa_csv, ir_cut_csv, grid, cfa) -> NDArray` shape `(n_channels, n_wavelengths)`
  - `mosaic_from_channels(channels, cfa) -> NDArray`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cfa.py
import numpy as np
from camsim.sensor.cfa import cfa_color_at, cfa_masks, mosaic_from_channels
from camsim.core.frames import CFAPattern

def test_rggb_layout():
    assert cfa_color_at(CFAPattern.RGGB, 0, 0) == "R"
    assert cfa_color_at(CFAPattern.RGGB, 0, 1) == "G"
    assert cfa_color_at(CFAPattern.RGGB, 1, 0) == "G"
    assert cfa_color_at(CFAPattern.RGGB, 1, 1) == "B"

def test_mono_has_one_channel_mask():
    masks = cfa_masks(CFAPattern.MONO, (4, 4))
    assert set(masks) == {"M"}
    assert masks["M"].all()

def test_mosaic_places_channels():
    ch = {"R": np.full((4, 4), 1.0), "G": np.full((4, 4), 2.0), "B": np.full((4, 4), 3.0)}
    m = mosaic_from_channels(ch, CFAPattern.RGGB)
    assert m[0, 0] == 1.0 and m[0, 1] == 2.0 and m[1, 1] == 3.0
```

Additional tests: `test_rccc_masks`, `test_projection_shape_and_nonnegative`, `test_mono_qe_exceeds_bayer_qe`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_cfa.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement CFA masks and projection**

Define the four Bayer patterns, mono, and RCCC. Build channel masks by tiling. Compute projection matrices by interpolating QE/CFA/IR-cut CSVs onto the wavelength grid and multiplying; validate non-negativity and row sums.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cfa.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/sensor/cfa.py tests/test_cfa.py
git commit -m "feat: add CFA patterns and spectral projection matrices

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 13: Noise, dark current, FPN, defects, and PDAF

**Files:**
- Create: `src/camsim/sensor/noise.py`, `src/camsim/sensor/defects.py`, `tests/test_noise.py`, `tests/test_defects.py`

**Interfaces:**
- Produces:
  - `poisson_electrons(mean, rng) -> NDArray`
  - `dark_current_e_per_s(dark_current_ref_e_per_s: float, temperature_c: float, config: TemperatureConfig) -> float`
  - `fixed_pattern_maps(config, shape, seed) -> FPPMaps` (`dsnu_e`, `prnu_fraction`, masks)
  - `sample_dark_current(mean_e, rng) -> NDArray`
  - `defect_masks(config, shape, seed) -> DefectMasks`
  - `pdaf_mask(config, shape, seed) -> NDArray[bool]`
  - `row_noise(row_sigma_e, height, rng) -> NDArray`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_noise.py
import numpy as np
from camsim.sensor.noise import poisson_electrons, dark_current_e_per_s

def test_poisson_variance_matches_mean():
    rng = np.random.default_rng(0)
    x = poisson_electrons(np.full(200_000, 100.0), rng)
    assert x.mean() == pytest.approx(100.0, abs=0.5)
    assert x.var() == pytest.approx(100.0, abs=2.0)

def test_dark_current_doubling():
    from camsim.config.schema import TemperatureConfig
    cfg = TemperatureConfig(reference_c=25.0, default_c=25.0, dark_current_doubling_k=7.0)
    assert dark_current_e_per_s(1.0, 32.0, cfg) == pytest.approx(2.0, rel=1e-3)

def test_fixed_pattern_reproducible():
    a = fixed_pattern_maps(cfg, (32, 32), seed=11)
    b = fixed_pattern_maps(cfg, (32, 32), seed=11)
    np.testing.assert_array_equal(a.dsnu_e, b.dsnu_e)
    np.testing.assert_array_equal(a.prnu_fraction, b.prnu_fraction)
```

Additional tests: `test_defect_rates_plausible`, `test_pdaf_mask_fraction`, `test_row_noise_common_per_row`, `test_prnu_multiplicative`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_noise.py tests/test_defects.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the noise and defect models**

Use `rng.poisson` and `rng.normal`; dark current follows the doubling model; DSNU/PRNU maps are generated once from `pattern_seed`; defect masks use the configured rates; PDAF masks select the configured pattern; row noise is a per-row Gaussian added to all columns.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_noise.py tests/test_defects.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/sensor/noise.py src/camsim/sensor/defects.py tests/test_noise.py tests/test_defects.py
git commit -m "feat: add noise, dark current, FPN, defects, and PDAF models

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 14: Gain, full well, ADC, and quantization

**Files:**
- Create: `src/camsim/sensor/gain.py`, `src/camsim/sensor/adc.py`, `tests/test_gain.py`, `tests/test_adc.py`

**Interfaces:**
- Produces:
  - `clamp_full_well(electrons, full_well_e) -> NDArray`
  - `electrons_to_voltage(electrons, conversion_gain_uv_per_e) -> NDArray`
  - `apply_analog_gain(voltage_uv, gain_db) -> NDArray`
  - `adc_quantize(voltage_uv, bits, full_scale_uv, black_level_dn, digital_gain=1.0) -> NDArray[uint16]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_adc.py
import numpy as np
from camsim.sensor.adc import adc_quantize
from camsim.sensor.gain import clamp_full_well

def test_full_well_clips():
    x = clamp_full_well(np.array([0.0, 5999.0, 6000.0, 7000.0]), 6000.0)
    np.testing.assert_array_equal(x, [0.0, 5999.0, 6000.0, 6000.0])

def test_black_level_and_saturation():
    bits, black, fs = 10, 64, 800_000.0
    dn = adc_quantize(np.array([0.0, fs]), bits, fs, black)
    assert dn[0] == black
    assert dn[1] == 2**bits - 1
    assert dn.dtype == np.uint16

def test_analog_gain_doubles_signal():
    from camsim.sensor.gain import apply_analog_gain
    v = np.array([100.0])
    np.testing.assert_allclose(apply_analog_gain(v, 6.0206), [200.0], rtol=1e-3)
```

Additional tests: `test_lcg_hcg_full_well_differs`, `test_quantization_step`, `test_digital_gain_preserves_black_level`, `test_adc_clips_at_max`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_gain.py tests/test_adc.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the gain/ADC chain**

Implement `V = N · CG`, analog gain in dB, ADC mapping with usable range `2**bits - 1 - black_level`, clipping, and digital gain applied to signal DN before re-adding black level. Return `uint16` with values within `[0, 2**bits - 1]`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gain.py tests/test_adc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/sensor/gain.py src/camsim/sensor/adc.py tests/test_gain.py tests/test_adc.py
git commit -m "feat: add conversion gain, full-well, and ADC quantization

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 15: Sensor pipeline, binning, crosstalk, blooming, and external contract

**Files:**
- Create: `src/camsim/sensor/sensor.py`, `src/camsim/sensor/binning.py`, `src/camsim/sensor/__init__.py`, `tests/test_sensor.py`, `tests/test_binning.py`
- Modify: `tests/conftest.py` (add `sensor`, `raw_frame`, `mono_sensor_config`, `mono_raw_frame`)

**Interfaces:**
- Produces:
  - `Sensor(config, seed=0)` with `.config`, `expose(irradiance, exposure_s, mode="auto", analog_gain_db=None, digital_gain=1.0, temperature_c=None, seed=None) -> RawFrame`, and `reseed(seed)`
  - `BinningConfig` handling analog (charge-domain sum, full well × factor²) and digital (post-ADC average)
  - `apply_spectral_crosstalk(channels, matrix)`, `apply_spatial_crosstalk(mosaic, kernel)`
  - `apply_blooming(electrons, full_well_e, fraction, iterations)`
  - Modify `src/camsim/__init__.py` to add `Sensor` to the lazy export map

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sensor.py
import numpy as np
import pytest
import camsim
from camsim.core.spectral import SpectralIrradiance
from camsim.core.frames import CFAPattern

def make_irr(level=100.0, shape=(32, 32)):
    w = np.arange(360.0, 831.0, 5.0)  # matches the flagship preset grid
    data = np.full((*shape, len(w)), level, dtype=np.float32)
    return SpectralIrradiance.from_array(data, w, "W/m^2/nm", pixel_pitch_um=1.12)

def test_expose_linear_with_lux(sensor_config):
    sensor = camsim.Sensor(sensor_config, seed=1)
    means = [sensor.expose(make_irr(l), 0.01, mode="LCG", seed=1).data.mean() for l in (1, 2, 4, 8)]
    assert np.corrcoef(means, [1, 2, 4, 8])[0, 1] > 0.999

def test_same_seed_bit_identical(sensor_config):
    sensor = camsim.Sensor(sensor_config, seed=1)
    a = sensor.expose(make_irr(), 0.01, mode="HCG", seed=7)
    b = sensor.expose(make_irr(), 0.01, mode="HCG", seed=7)
    np.testing.assert_array_equal(a.data, b.data)

def test_per_call_seed_independent_of_call_order(sensor_config):
    sensor = camsim.Sensor(sensor_config, seed=1)
    a = sensor.expose(make_irr(), 0.01, mode="HCG", seed=7).data
    sensor.expose(make_irr(), 0.01, mode="HCG", seed=8)
    c = sensor.expose(make_irr(), 0.01, mode="HCG", seed=7).data
    np.testing.assert_array_equal(a, c)

def test_external_irradiance_wrong_shape_raises(sensor_config):
    sensor = camsim.Sensor(sensor_config, seed=1)
    grid = np.arange(360.0, 831.0, 5.0)  # correct wavelength grid, wrong spatial shape
    irr = SpectralIrradiance.from_array(
        np.ones((8, 8, len(grid))), grid, "W/m^2/nm", pixel_pitch_um=1.12,
    )
    with pytest.raises(ValueError, match="sensor grid"):
        sensor.expose(irr, 0.01)
```

Additional tests: `test_exposure_linearity`, `test_shot_noise_variance_slope`, `test_temperature_override_increases_dark`, `test_wrong_pixel_pitch_raises`, `test_wavelength_mismatch_raises`, `test_mono_output_single_channel`, `test_rggb_mosaic_pattern`, `test_binning_analog_vs_digital`, `test_crosstalk_row_sums`, `test_blooming_spreads_saturated_charge`, `test_no_nan_in_raw`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_sensor.py tests/test_binning.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement `Sensor.expose` and the supporting models**

Follow the signal-chain order in Section 7.2 of the spec exactly: spectral integration → PRNU → dark + DSNU → Poisson + full-well clip → crosstalk → read noise + clip → conversion gain → analog gain → ADC → digital gain → row noise. Validate irradiance shape/pitch/wavelength grid before integration and raise precise errors. Generate fixed-pattern maps once from `pattern_seed`. Implement analog/digital binning and optional blooming. Add the `sensor`, `raw_frame`, `mono_sensor_config`, and `mono_raw_frame` fixtures to `tests/conftest.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sensor.py tests/test_binning.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/sensor/sensor.py src/camsim/sensor/binning.py src/camsim/sensor/__init__.py tests/test_sensor.py tests/test_binning.py
git commit -m "feat: add full sensor pipeline with binning, crosstalk, and blooming

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 16: EMVA 1288-style characterization

**Files:**
- Create: `src/camsim/characterization/emva.py`, `src/camsim/characterization/report.py`, `src/camsim/characterization/__init__.py`, `tests/test_characterization.py`

**Interfaces:**
- Produces:
  - `characterize_sensor(sensor, irradiance_levels=None, lux_levels=None, frames_per_level=32, exposure_s=1e-3, mode="auto", temperature_c=25.0, illuminant="D65", lens=None, base_seed=0) -> EMVAReport`
  - `EMVAReport` fields: `system_gain_dn_per_e`, `read_noise_e`, `saturation_capacity_e`, `absolute_sensitivity_threshold_e`, `dynamic_range_db`, `dsnu_e`, `prnu_percent`, `snr_curve`, `linearity_residual`, `provenance`

- [ ] **Step 1: Write the failing round-trip tests**

```python
# tests/test_characterization.py
import pytest
import camsim
from camsim.characterization import characterize_sensor

def test_characterization_recovers_read_noise(sensor_config):
    sensor = camsim.Sensor(sensor_config, seed=1)
    report = characterize_sensor(sensor, irradiance_levels=[1, 2, 4, 8, 16], frames_per_level=16, mode="HCG", exposure_s=0.01, base_seed=3)
    assert report.read_noise_e == pytest.approx(0.7, rel=0.10)

def test_characterization_recovers_saturation(sensor_config):
    sensor = camsim.Sensor(sensor_config, seed=1)
    report = characterize_sensor(sensor, irradiance_levels=[1, 2, 4, 8, 16, 32], frames_per_level=8, mode="HCG", exposure_s=0.01, base_seed=3)
    assert report.saturation_capacity_e == pytest.approx(2000.0, rel=0.10)

def test_characterization_reports_dynamic_range(sensor_config):
    sensor = camsim.Sensor(sensor_config, seed=1)
    report = characterize_sensor(sensor, irradiance_levels=[1, 2, 4, 8], frames_per_level=8, mode="LCG", exposure_s=0.01, base_seed=3)
    assert report.dynamic_range_db > 30.0
```

Additional tests: `test_ptc_slope_matches_system_gain`, `test_dsnu_prnu_recovered`, `test_linearity_residual_small`, `test_report_json_roundtrip`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_characterization.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the EMVA-style measurement**

Generate flat-field frames per level with independent seeds; subtract black level; compute mean/variance per level; fit the photon transfer curve; derive system gain, read noise, saturation, absolute sensitivity threshold, dynamic range, DSNU, PRNU, and SNR curve. Document assumptions and cite EMVA 1288 in the module docstring.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_characterization.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/characterization tests/test_characterization.py
git commit -m "feat: add EMVA 1288-style sensor characterization

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 17: Flagship preset, datasheet guide, sensor example, and G2 report

**Files:**
- Create: `examples/04_emva_characterization.py`, `results/reports/g2_sensor.md`, `results/figures/g2_*.png`, `tests/test_preset_validation.py`
- Modify: `src/camsim/config/presets/sensors/flagship_phone_50mp.json`, `docs/datasheet_mapping.md`, `notebooks/01_optics_and_sensor.py`

**Interfaces:**
- Consumes: `Sensor`, `characterize_sensor`, `viz` helpers.
- Produces: a datasheet mapping table, a validated flagship preset, G2 figures/report.

- [ ] **Step 1: Write the failing preset/G2 tests**

```python
# tests/test_preset_validation.py
import pytest
import camsim
from camsim.characterization import characterize_sensor
from camsim.config.io import load_sensor

def test_preset_linearity_and_dynamic_range():
    cfg = load_sensor("src/camsim/config/presets/sensors/flagship_phone_50mp.json")
    sensor = camsim.Sensor(cfg, seed=5)
    report = characterize_sensor(sensor, irradiance_levels=[1, 2, 4, 8, 16], frames_per_level=8, mode="LCG", exposure_s=0.01, base_seed=9)
    assert report.linearity_residual < 0.05
    assert 30.0 < report.dynamic_range_db < 90.0
```

- [ ] **Step 2: Run it to confirm failure**

Run: `uv run pytest tests/test_preset_validation.py -v`
Expected: FAIL until the preset/report are complete.

- [ ] **Step 3: Finalize the preset, datasheet guide, and examples**

Fill the flagship preset with value-justified QE/CFA/IR-cut curves and provenance; document how each datasheet field maps to JSON and which values are assumptions. Add the EMVA example and sensor notebook sections. Generate G2 plots: PTC, SNR vs irradiance, dark current vs temperature, LCG/HCG comparison, DSNU/PRNU histograms.

- [ ] **Step 4: Run the G2 gate**

Run:
```bash
uv run pytest tests/test_sensor.py tests/test_noise.py tests/test_defects.py tests/test_gain.py tests/test_adc.py tests/test_characterization.py tests/test_preset_validation.py -v
uv run ruff check . && uv run mypy src
uv run python examples/04_emva_characterization.py
```
Expected: PASS and figures written.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/config/presets/sensors/flagship_phone_50mp.json docs/datasheet_mapping.md examples/04_emva_characterization.py results tests/test_preset_validation.py notebooks/01_optics_and_sensor.py
git commit -m "feat: add flagship sensor preset and G2 validation report

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Phase 3 — ISP

### Task 18: DPC, BLC, and LSC

**Files:**
- Create: `src/camsim/isp/dpc.py`, `src/camsim/isp/blc.py`, `src/camsim/isp/lsc.py`, `tests/test_isp_dpc.py`, `tests/test_isp_blc.py`, `tests/test_isp_lsc.py`

**Interfaces:**
- Produces:
  - `correct_defects(raw: RawFrame, config: DPCConfig) -> NDArray[np.float32]`
  - `subtract_black_level(image, black_level_dn, pattern=None) -> NDArray[np.float32]`
  - `apply_lsc(image, config: LSCConfig, cfa) -> NDArray[np.float32]`
  - `calibrate_lsc(flat_field_raw) -> NDArray[np.float32]`
  - `DPCConfig(enabled=True, method="neighbor"|"median")`
  - `LSCConfig(enabled=True, mode="none"|"radial"|"flat_field", map=None, coefficients=None, center=None)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_isp_blc.py
import numpy as np
from camsim.isp.blc import subtract_black_level

def test_blc_subtracts_and_clamps():
    x = np.array([[64, 100, 1000]], dtype=np.uint16)
    y = subtract_black_level(x, 64)
    np.testing.assert_array_equal(y, [[0, 36, 936]])

# tests/test_isp_lsc.py
def test_lsc_flat_field_corrects_vignetting():
    from camsim.isp.lsc import apply_lsc, LSCConfig
    yy, xx = np.mgrid[0:64, 0:64]
    r2 = ((xx - 31.5) ** 2 + (yy - 31.5) ** 2) / (31.5 ** 2 + 31.5 ** 2)
    img = (1.0 - 0.3 * r2).astype(np.float32)
    out = apply_lsc(img, LSCConfig(mode="flat_field", map=1.0 / img), cfa=None)
    assert out.max() / out.min() < 1.02
```

Additional tests: `test_dpc_interpolates_defects`, `test_blc_pattern`, `test_lsc_disabled_identity`, `test_lsc_preserves_linearity`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_isp_dpc.py tests/test_isp_blc.py tests/test_isp_lsc.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the stages**

DPC uses the defect mask and same-CFA-channel neighbours; BLC subtracts scalar or 2×2 pattern and clamps; LSC applies a supplied map or analytic radial gain and can be calibrated from a flat field. All operate in float32 linear space.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_isp_dpc.py tests/test_isp_blc.py tests/test_isp_lsc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/isp/dpc.py src/camsim/isp/blc.py src/camsim/isp/lsc.py tests/test_isp_dpc.py tests/test_isp_blc.py tests/test_isp_lsc.py
git commit -m "feat: add ISP defect correction, black level, and lens shading

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 19: Auto white balance

**Files:**
- Create: `src/camsim/isp/awb.py`, `tests/test_isp_awb.py`

**Interfaces:**
- Produces:
  - `AWBResult(gains, illuminant_estimate, method, confidence)`
  - `AWBConfig(enabled=True, method="gray_world", p=6.0)`
  - `estimate_awb(raw_or_rgb, cfa, config) -> AWBResult`
  - `apply_awb(image, gains, cfa) -> NDArray`
  - Methods: `"gray_world"`, `"white_patch"`, `"shades_of_gray"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_isp_awb.py
import numpy as np
from camsim.isp.awb import estimate_awb, apply_awb, AWBConfig

def test_gray_world_equalizes_channel_means():
    img = np.zeros((16, 16, 3), np.float32)
    img[..., 0] = 2.0; img[..., 1] = 1.0; img[..., 2] = 0.5
    res = estimate_awb(img, cfa=None, config=AWBConfig(method="gray_world"))
    out = apply_awb(img, res.gains, cfa=None)
    means = out.reshape(-1, 3).mean(axis=0)
    np.testing.assert_allclose(means, means.mean(), rtol=1e-2)

def test_white_patch_uses_brightest_pixel():
    img = np.ones((8, 8, 3), np.float32)
    img[0, 0] = [4.0, 2.0, 1.0]
    res = estimate_awb(img, cfa=None, config=AWBConfig(method="white_patch"))
    assert res.gains[0] < res.gains[1] < res.gains[2]
```

Additional tests: `test_shades_of_gray_p6`, `test_mono_awb_is_noop`, `test_awb_confidence_range`, `test_awb_preserves_linearity`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_isp_awb.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement AWB**

Compute channel statistics over valid pixels; gray-world uses means, white-patch uses maxima, shades-of-gray uses the p-norm; normalize gains to green; confidence from channel spread. Apply per-CFA-channel gains in linear RAW or per-RGB-channel gains after demosaic.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_isp_awb.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/isp/awb.py tests/test_isp_awb.py
git commit -m "feat: add gray-world, white-patch, and shades-of-gray AWB

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 20: Demosaicing

**Files:**
- Create: `src/camsim/isp/demosaic.py`, `tests/test_isp_demosaic.py`

**Interfaces:**
- Produces:
  - `demosaic_bilinear(raw, cfa) -> NDArray[np.float32]`
  - `demosaic_mhc(raw, cfa) -> NDArray[np.float32]`
  - `demosaic_rccc_baseline(raw) -> NDArray[np.float32]`
  - `demosaic(raw, cfa, method) -> NDArray`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_isp_demosaic.py
import numpy as np
import pytest
from camsim.isp.demosaic import demosaic_bilinear, demosaic_mhc
from camsim.core.frames import CFAPattern

def _mosaic_from_rgb(rgb, cfa=CFAPattern.RGGB):
    from camsim.sensor.cfa import mosaic_from_channels
    return mosaic_from_channels({"R": rgb[...,0], "G": rgb[...,1], "B": rgb[...,2]}, cfa)

def test_bilinear_matches_reference_on_smooth_image():
    from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
    rgb = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    raw = _mosaic_from_rgb(rgb)
    ours = demosaic_bilinear(raw, CFAPattern.RGGB)
    ref = demosaicing_CFA_Bayer_bilinear(raw, "RGGB")
    np.testing.assert_allclose(ours, ref, atol=1e-5)

def test_mhc_matches_reference_on_smooth_image():
    from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
    rgb = np.random.default_rng(1).random((32, 32, 3)).astype(np.float32)
    raw = _mosaic_from_rgb(rgb)
    ours = demosaic_mhc(raw, CFAPattern.RGGB)
    ref = demosaicing_CFA_Bayer_Malvar2004(raw, "RGGB")
    np.testing.assert_allclose(ours, ref, atol=1e-4)
```

Additional tests: `test_mono_bypass`, `test_rccc_baseline_shape`, `test_demosaic_preserves_dc`, `test_invalid_cfa_raises`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_isp_demosaic.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement bilinear and MHC**

Implement both from scratch in NumPy. Bilinear interpolates missing channels from same-colour neighbours; MHC uses the 5×5 gradient-corrected kernels from the 2004 paper. Mono returns the single channel (replicated only for RGB visualization at the output boundary). RCCC uses a documented baseline: reconstruct C and R planes, derive a luminance/chroma image, and return RGB with a caveat.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_isp_demosaic.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/isp/demosaic.py tests/test_isp_demosaic.py
git commit -m "feat: add bilinear, MHC, and baseline RCCC demosaicing

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 21: Colour correction matrix

**Files:**
- Create: `src/camsim/isp/ccm.py`, `tests/test_isp_ccm.py`

**Interfaces:**
- Produces:
  - `fit_ccm(camera_rgb, target_rgb, ridge=1e-4, row_sum=True) -> NDArray` (3×3)
  - `apply_ccm(rgb, matrix) -> NDArray`
  - `CCMConfig(matrix=None, ridge=1e-4, row_sum=True)`
  - `CCMResult(matrix, residual, delta_e_before, delta_e_after)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_isp_ccm.py
import numpy as np
import pytest
from camsim.isp.ccm import fit_ccm, apply_ccm

def test_ccm_recovers_known_matrix():
    rng = np.random.default_rng(0)
    target = rng.random((24, 3))
    true = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    camera = target @ np.linalg.inv(true).T
    fitted = fit_ccm(camera, target, ridge=0.0)
    np.testing.assert_allclose(fitted, true, atol=1e-6)

def test_ccm_rows_sum_to_one():
    rng = np.random.default_rng(1)
    target = rng.random((24, 3))
    camera = target @ np.array([[1.1, -0.2, 0.1], [-0.1, 1.2, -0.1], [0.0, -0.1, 1.1]]).T
    fitted = fit_ccm(camera, target, ridge=1e-4, row_sum=True)
    np.testing.assert_allclose(fitted.sum(axis=1), 1.0, atol=1e-6)
```

Additional tests: `test_apply_ccm_identity`, `test_ridge_reduces_noise_amplification`, `test_ccm_improves_delta_e`, `test_negative_values_clamped`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_isp_ccm.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement ridge CCM fitting**

Solve the regularized least-squares problem with an optional row-sum constraint (augment the system or project rows). Validate that the matrix is finite and rows sum to one. Apply in linear RGB.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_isp_ccm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/isp/ccm.py tests/test_isp_ccm.py
git commit -m "feat: add ridge-regularized row-sum-constrained CCM

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 22: Denoise and sharpen

**Files:**
- Create: `src/camsim/isp/denoise.py`, `src/camsim/isp/sharpen.py`, `tests/test_isp_denoise.py`, `tests/test_isp_sharpen.py`

**Interfaces:**
- Produces:
  - `DenoiseConfig(method="gaussian"|"bilateral", luma_strength, chroma_strength, radius)`
  - `denoise(image, config) -> NDArray`
  - `SharpenConfig(amount, radius, threshold, luma_only=True)`
  - `unsharp_mask(image, config) -> NDArray`
  - Linear opponent-space helpers `rgb_to_opponent`, `opponent_to_rgb`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_isp_denoise.py
import numpy as np
from camsim.isp.denoise import denoise, DenoiseConfig

def test_denoise_identity_when_disabled():
    x = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    np.testing.assert_allclose(denoise(x, DenoiseConfig(luma_strength=0, chroma_strength=0)), x)

def test_denoise_reduces_flat_patch_noise():
    rng = np.random.default_rng(0)
    x = np.full((64, 64, 3), 0.5, np.float32) + rng.normal(0, 0.02, (64, 64, 3)).astype(np.float32)
    y = denoise(x, DenoiseConfig(method="gaussian", luma_strength=1.0, chroma_strength=1.0, radius=1.0))
    assert y.std() < x.std()

# tests/test_isp_sharpen.py
def test_sharpen_identity_when_amount_zero():
    x = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    np.testing.assert_allclose(unsharp_mask(x, SharpenConfig(amount=0.0)), x)

def test_sharpen_increases_edge_contrast():
    x = np.zeros((64, 64, 3), np.float32); x[:, 32:] = 1.0
    y = unsharp_mask(x, SharpenConfig(amount=1.0, radius=1.0, threshold=0.0))
    assert np.abs(np.diff(y[32, :, 0])).max() > np.abs(np.diff(x[32, :, 0])).max()
```

Additional tests: `test_bilateral_edge_preservation`, `test_sharpen_threshold_suppresses_noise`, `test_luma_only_preserves_chroma`, `test_denoise_linearity`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_isp_denoise.py tests/test_isp_sharpen.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement denoise and sharpen**

Denoise in a linear opponent space (documented matrix), Gaussian or bilateral; apply independent luma/chroma strengths. Unsharp mask with amount, radius, threshold; luma-only by default. Both are linear-space operations and must be identities when disabled.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_isp_denoise.py tests/test_isp_sharpen.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/isp/denoise.py src/camsim/isp/sharpen.py tests/test_isp_denoise.py tests/test_isp_sharpen.py
git commit -m "feat: add luma/chroma denoise and thresholded unsharp mask

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 23: Tone curves and output colour spaces

**Files:**
- Create: `src/camsim/isp/tone.py`, `src/camsim/isp/colorspace.py`, `tests/test_isp_tone.py`, `tests/test_isp_colorspace.py`

**Interfaces:**
- Produces:
  - `linear_to_srgb(linear_rgb) -> NDArray`
  - `srgb_to_linear(display_rgb) -> NDArray`
  - `linear_to_p3(linear_rgb) -> NDArray`
  - `apply_tone(linear_rgb, config: ToneConfig) -> NDArray`
  - `ToneConfig(color_space="sRGB"|"DisplayP3", exposure_compensation_ev=0.0, gamma=None)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_isp_tone.py
import numpy as np
import pytest
from camsim.isp.tone import linear_to_srgb, srgb_to_linear

def test_srgb_oetf_known_values():
    assert linear_to_srgb(np.array([0.0]))[0] == pytest.approx(0.0)
    assert linear_to_srgb(np.array([1.0]))[0] == pytest.approx(1.0)
    assert linear_to_srgb(np.array([0.0031308]))[0] == pytest.approx(0.04045, abs=1e-4)

def test_srgb_roundtrip():
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-7)

def test_tone_matches_colour_science():
    import colour
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(linear_to_srgb(x), colour.cctf_encoding(x, function="sRGB"), atol=1e-7)
```

Additional tests: `test_p3_matrices`, `test_exposure_compensation`, `test_output_range`, `test_tone_is_last_nonlinear_stage`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_isp_tone.py tests/test_isp_colorspace.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the transfer functions and colour spaces**

Implement the sRGB OETF/EOTF exactly (linear segment and power segment), Display P3 primaries/white point matrices with references, and exposure compensation in linear space before tone. Validate against `colour-science`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_isp_tone.py tests/test_isp_colorspace.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/isp/tone.py src/camsim/isp/colorspace.py tests/test_isp_tone.py tests/test_isp_colorspace.py
git commit -m "feat: add sRGB/P3 transfer functions and tone stage

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 24: ISP pipeline, trace, and default configuration

**Files:**
- Create: `src/camsim/isp/pipeline.py`, `src/camsim/isp/__init__.py`, `tests/test_isp_pipeline.py`
- Modify: `src/camsim/config/presets/isp/default.json`

**Interfaces:**
- Produces:
  - `ISP(config: ISPConfig)` with `from_json(path)` and `process(raw: RawFrame, trace=False) -> RenderedImage`
  - `ISPConfig` stage toggles and parameters
  - `StageOutput` trace keyed by stage name
  - Pipeline order enforced: `dpc, blc, lsc, awb, demosaic, ccm, denoise, sharpen, tone, output`
  - Modify `src/camsim/__init__.py` to add `ISP` to the lazy export map

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_isp_pipeline.py
import numpy as np
import pytest
import camsim
from camsim.core.frames import CFAPattern

def test_pipeline_order_and_trace(sensor_config, isp_config, sensor, raw_frame):
    isp = camsim.ISP(isp_config)
    out = isp.process(raw_frame, trace=True)
    assert out.stages is not None
    assert list(out.stages)[:2] == ["dpc", "blc"]
    assert out.data.shape == (raw_frame.data.shape[0], raw_frame.data.shape[1], 3)
    assert 0.0 <= out.data.min() and out.data.max() <= 1.0

def test_linearity_before_tone(sensor_config, isp_config, sensor, raw_frame):
    # Scaling the raw signal scales the linear output before tone.
    isp = camsim.ISP(isp_config)
    a = isp.process(raw_frame, trace=True)
    scaled_data = np.clip(
        (raw_frame.data.astype(np.float32) - raw_frame.black_level_dn) * 2
        + raw_frame.black_level_dn,
        0,
        2 ** raw_frame.bit_depth - 1,
    ).astype(np.uint16)
    scaled = raw_frame.with_data(scaled_data)
    b = isp.process(scaled, trace=True)
    np.testing.assert_allclose(b.stages["ccm"].data, 2 * a.stages["ccm"].data, rtol=1e-3, atol=1e-4)

def test_mono_pipeline_bypasses_demosaic_and_awb(mono_sensor_config, isp_config, mono_raw_frame):
    isp = camsim.ISP(isp_config)
    out = isp.process(mono_raw_frame, trace=True)
    assert out.channels == "MONO"
    assert "demosaic" not in out.stages
```

Additional tests: `test_stage_disable`, `test_final_output_matches_manual_chain`, `test_rggb_pipeline`, `test_rccc_baseline_runs`, `test_trace_is_opt_in`, `test_no_nan_output`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_isp_pipeline.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the pipeline**

Apply stages in the enforced order, skip stages disabled by config or inapplicable to mono/RCCC, record `StageOutput` only when `trace=True`, and return a `RenderedImage` with display `data`, `linear_rgb`, color space, channels, and metadata. Default ISP config enables DPC/BLC/AWB/demosaic (MHC)/CCM (precomputed or fit)/denoise (off by default)/sharpen (off by default)/tone sRGB.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_isp_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/isp/pipeline.py src/camsim/isp/__init__.py src/camsim/config/presets/isp/default.json tests/test_isp_pipeline.py
git commit -m "feat: add unified ISP pipeline with stage tracing

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 25: ISP stage gallery, notebook 02, and G3 report

**Files:**
- Create: `src/camsim/viz/pipeline.py`, `examples/02_isp_stage_gallery.py`, `notebooks/02_isp_color_fundamentals.py`, `results/reports/g3_isp.md`, `results/figures/g3_*.png`, `tests/test_example_02.py`

**Interfaces:**
- Produces: `plot_pipeline_trace(stages, save=None)`, `plot_zoom_crops(stages, save=None)`, `plot_demosaic_comparison(...)`; G3 report with the spec's thresholds.

- [ ] **Step 1: Write the failing smoke/report tests**

```python
# tests/test_example_02.py
import subprocess, sys
from pathlib import Path

def test_example_02_runs(tmp_path):
    r = subprocess.run([sys.executable, "examples/02_isp_stage_gallery.py"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(list(Path("results/figures").glob("02_*.png"))) >= 3
```

Additional tests: `test_g3_colorchecker_delta_e_threshold`, `test_g3_awb_recovers_gains`, `test_g3_demosaic_matches_reference`, `test_g3_tone_matches_standard`, `test_g3_stage_trace_reproduces_final`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_example_02.py -v`
Expected: FAIL because the example is missing.

- [ ] **Step 3: Implement the gallery and report**

Generate a contact sheet after each stage, zoomed edge crops, and a bilinear/MHC comparison on a Siemens star. Measure ColorChecker ΔE2000 under D65 and 3000 K, AWB recovery, demosaic reference agreement, and stage-trace consistency. Write the G3 report with measured values and tolerances.

- [ ] **Step 4: Run the G3 gate**

Run:
```bash
uv run pytest tests/test_isp_*.py tests/test_example_02.py -v
uv run ruff check . && uv run mypy src
uv run jupytext --to ipynb --execute notebooks/02_isp_color_fundamentals.py --output /tmp/02.ipynb
```
Expected: PASS; notebook executes.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/viz/pipeline.py examples/02_isp_stage_gallery.py notebooks/02_isp_color_fundamentals.py results tests/test_example_02.py
git commit -m "feat: add ISP stage gallery and G3 validation report

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Phase 4 — Metrics

### Task 26: Chart layouts and colour reference integration

**Files:**
- Create: `src/camsim/metrics/chart.py`, `src/camsim/scene/charts.py`, `tests/test_chart.py`

**Interfaces:**
- Produces:
  - `PatchROI(name, x, y, width, height)`, `ChartLayout(name, patches, edge_roi=None, reference=None, pixel_pitch_um=None, focal_length_mm=None)`
  - `colorchecker_layout(illuminant="D65", lens=None) -> ChartLayout` (scene-owned convenience)
  - `flat_patch_layout(...)`, `slanted_edge_layout(...)`
  - `extract_patch(image, roi) -> NDArray`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chart.py
import numpy as np
import pytest
from camsim.metrics.chart import ChartLayout, PatchROI, extract_patch
from camsim.scene.charts import colorchecker_layout

def test_colorchecker_layout_has_24_patches_and_reference():
    layout = colorchecker_layout(illuminant="D65")
    assert len(layout.patches) == 24
    assert layout.reference is not None
    assert layout.reference.illuminant_name == "D65"

def test_extract_patch_returns_pixels():
    img = np.arange(100, dtype=np.float32).reshape(10, 10, 1)
    roi = PatchROI("p", 2, 2, 3, 3)
    assert extract_patch(img, roi).shape == (3, 3, 1)

def test_layout_json_roundtrip():
    layout = colorchecker_layout(illuminant="D65")
    restored = ChartLayout.from_dict(layout.to_dict())
    assert restored.name == layout.name
    assert len(restored.patches) == 24
```

Additional tests: `test_flat_patch_layout`, `test_slanted_edge_roi`, `test_reference_required_for_non_d65`, `test_layout_rejects_overlapping_patches`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_chart.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement layouts**

Define ROIs and serialization. `colorchecker_layout` lives in the scene layer (it depends on spectral data) but returns `camsim.metrics.chart.ChartLayout`, preserving the import direction: scene → metrics types, never metrics → scene. Non-D65 layouts must carry a `ColorReference`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_chart.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/metrics/chart.py src/camsim/scene/charts.py tests/test_chart.py
git commit -m "feat: add chart layouts and spectral color references

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 27: Slanted-edge e-SFR / MTF

**Files:**
- Create: `src/camsim/metrics/mtf.py`, `tests/test_metrics_mtf.py`

**Interfaces:**
- Produces:
  - `MTFCurve(frequency_cp, mtf)`
  - `MTFResult(mtf50_cp, mtf20_cp, mtf50_cyc_per_deg, mtf20_cyc_per_deg, curve: MTFCurve, edge_angle_deg, nyquist_cp)`
  - `slanted_edge_mtf(image_2d, pixel_pitch_um, focal_length_mm=None, oversampling=4, window="hamming") -> MTFResult`
  - `extract_raw_plane(raw, channel="G") -> (NDArray, float)` with the documented sampling pitch

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics_mtf.py
import numpy as np
import pytest
from camsim.metrics.mtf import slanted_edge_mtf

def test_gaussian_mtf50_analytic():
    from camsim.optics.psf import gaussian_psf
    from camsim.optics.propagation import apply_psf
    from camsim.scene.targets import slanted_edge_target
    edge = slanted_edge_target((256, 256), angle_deg=5.0)
    blurred = apply_psf(edge[..., None], gaussian_psf(1.0, size=31))[..., 0]
    res = slanted_edge_mtf(blurred, pixel_pitch_um=1.12, focal_length_mm=6.4)
    assert res.mtf50_cp == pytest.approx(0.1874, abs=0.01)

def test_mtf_curve_is_normalized_and_monotonic():
    from camsim.scene.targets import slanted_edge_target
    edge = slanted_edge_target((256, 256), angle_deg=5.0)
    res = slanted_edge_mtf(edge, pixel_pitch_um=1.12)
    assert res.curve.mtf[0] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(res.curve.mtf) <= 1e-6)

def test_edge_angle_robustness():
    from camsim.scene.targets import slanted_edge_target
    for angle in (5, 10, 15, 20):
        img = slanted_edge_target((256, 256), angle_deg=angle)
        res = slanted_edge_mtf(img, pixel_pitch_um=1.12)
        assert res.edge_angle_deg == pytest.approx(angle, abs=1.0)
```

Additional tests: `test_mtf_cycles_per_degree`, `test_mtf_noise_robustness`, `test_raw_green_plane_sampling_note`, `test_box_psf_first_zero`, `test_airy_mtf`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_metrics_mtf.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the ISO 12233-style e-SFR**

Implement ROI extraction, edge-angle regression, oversampled ESF, LSF derivative with Hamming window, derivative-response correction, FFT, DC normalization, and MTF50/20 interpolation. Convert to cycles/degree with `focal_length_mm` and `pixel_pitch_um`. Document that this follows ISO 12233 principles but is not certified.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics_mtf.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/metrics/mtf.py tests/test_metrics_mtf.py
git commit -m "feat: add ISO 12233-style slanted-edge e-SFR

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 28: SNR metrics

**Files:**
- Create: `src/camsim/metrics/snr.py`, `tests/test_metrics_snr.py`

**Interfaces:**
- Produces:
  - `SNRResult(mean, std, snr_db, channel, valid)`
  - `snr_patch(image, roi, channel=None) -> SNRResult`
  - `snr_temporal(frames, roi, channel=None) -> SNRResult`
  - Per-channel and luma helpers

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics_snr.py
import numpy as np
import pytest
from camsim.metrics.snr import snr_patch
from camsim.metrics.chart import PatchROI

def test_snr_known_gaussian_noise():
    rng = np.random.default_rng(0)
    img = 10.0 + rng.normal(0, 1.0, (64, 64, 1)).astype(np.float32)
    res = snr_patch(img, PatchROI("p", 0, 0, 64, 64), channel=0)
    assert res.snr_db == pytest.approx(20.0, abs=0.5)

def test_snr_clipped_patch_invalid():
    img = np.ones((8, 8, 1), np.float32)
    res = snr_patch(img, PatchROI("p", 0, 0, 8, 8), channel=0)
    assert not res.valid

def test_snr_luma():
    img = np.full((16, 16, 3), 0.5, np.float32)
    res = snr_patch(img, PatchROI("p", 0, 0, 16, 16), channel="luma")
    assert res.channel == "luma"
```

Additional tests: `test_snr_per_channel`, `test_snr_temporal`, `test_snr_mono`, `test_snr_black_level_handling`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_metrics_snr.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement SNR**

Compute mean and standard deviation on the ROI in linear units; SNR_dB = 20 log10(mean/std). Provide spatial and temporal variants; luma uses Rec. 709 weights in linear space. Flag clipped or black patches as invalid.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics_snr.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/metrics/snr.py tests/test_metrics_snr.py
git commit -m "feat: add per-channel, luma, spatial, and temporal SNR

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 29: Colour metrics

**Files:**
- Create: `src/camsim/metrics/color.py`, `tests/test_metrics_color.py`

**Interfaces:**
- Produces:
  - `ColorResult(delta_e76, delta_e2000, per_patch, valid)`
  - `delta_e_patches(image, layout, reference=None, color_space="sRGB") -> ColorResult`
  - `linear_rgb_to_lab(image, color_space, white_point)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics_color.py
import numpy as np
import pytest
import colour
from camsim.metrics.color import delta_e_patches
from camsim.metrics.chart import ChartLayout, PatchROI
from camsim.core.color import ColorReference

def test_delta_e_identity():
    from camsim.metrics.color import _delta_e2000
    lab = np.array([50.0, 10.0, -10.0])
    assert _delta_e2000(lab, lab) == pytest.approx(0.0, abs=1e-9)

def test_delta_e_patches_uses_layout_reference():
    ref = ColorReference(lab=np.array([[50.0, 10.0, -10.0]]), white_point_xy=(0.3127, 0.3290), illuminant_name="D65")
    layout = ChartLayout(name="t", patches=(PatchROI("p", 0, 0, 8, 8),), reference=ref)
    img = np.full((8, 8, 3), 0.5, np.float32)
    res = delta_e_patches(img, layout)
    assert res.per_patch[0].delta_e2000 > 0.0

def test_delta_e_matches_colour_science():
    lab1 = np.array([50.0, 10.0, -10.0]); lab2 = np.array([45.0, 12.0, -8.0])
    expected = colour.delta_E(lab1, lab2, method="CIE 2000")
    # Use a synthetic ColorReference and a precomputed Lab image to exercise the formula.
    # (The implementation exposes `_delta_e2000` for this exact comparison.)
    from camsim.metrics.color import _delta_e2000
    assert _delta_e2000(lab1, lab2) == pytest.approx(expected, rel=1e-9)

def test_non_d65_reference_is_used():
    ref_d65 = ColorReference(lab=np.array([[50.0, 0.0, 0.0]]), white_point_xy=(0.3127, 0.3290), illuminant_name="D65")
    ref_warm = ColorReference(lab=np.array([[50.0, 5.0, 10.0]]), white_point_xy=(0.4369, 0.4041), illuminant_name="3000K")
    layout = ChartLayout(name="t", patches=(PatchROI("p", 0, 0, 8, 8),), reference=ref_warm)
    img = np.full((8, 8, 3), 0.5, np.float32)
    res = delta_e_patches(img, layout, reference=ref_warm)
    assert res.per_patch[0].delta_e2000 != pytest.approx(0.0, abs=1e-3)
```

Additional tests: `test_delta_e76_formula`, `test_mono_raises_clear_error`, `test_missing_reference_non_d65_raises`, `test_white_patch_reference`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_metrics_color.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement colour metrics**

Linearize display RGB, convert to XYZ using the output colour-space matrix and white point, convert to Lab, and compare with the reference Lab. Implement ΔE76 and ΔE2000; cross-check ΔE2000 against `colour-science`. Mono images raise a clear error for colour metrics.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics_color.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/metrics/color.py tests/test_metrics_color.py
git commit -m "feat: add delta-E76 and delta-E2000 colour metrics

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 30: Acutance and unified `measure_all`

**Files:**
- Create: `src/camsim/metrics/acutance.py`, `src/camsim/metrics/summary.py`, `src/camsim/metrics/__init__.py`, `tests/test_metrics_summary.py`, `tests/test_metrics_independence.py`

**Interfaces:**
- Produces:
  - `edge_acutance(image, roi) -> float`
  - `MetricReport(mtf=None, snr=None, color=None, acutance=None, layout_id=None, metadata=None)` with `to_dict`/`to_json`
  - `measure_all(image, layout, reference=None) -> MetricReport`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics_summary.py
import numpy as np
import pytest
from camsim.metrics.summary import measure_all
from camsim.metrics.chart import ChartLayout, PatchROI
from camsim.core.color import ColorReference
from camsim.core.frames import RenderedImage

def test_measure_all_complete_report():
    ref = ColorReference(lab=np.zeros((1, 3)), white_point_xy=(0.3127, 0.3290), illuminant_name="D65")
    layout = ChartLayout(name="t", patches=(PatchROI("p", 0, 0, 16, 16),), reference=ref)
    img = RenderedImage(data=np.full((32, 32, 3), 0.5, np.float32), color_space="sRGB", channels="RGB")
    report = measure_all(img, layout)
    assert report.color is not None
    assert report.to_dict()["layout_id"] == "t"

def test_measure_all_skips_missing_groups():
    layout = ChartLayout(name="flat", patches=(PatchROI("p", 0, 0, 8, 8),))
    img = RenderedImage(data=np.full((16, 16, 3), 0.5, np.float32), color_space="sRGB", channels="RGB")
    report = measure_all(img, layout)
    assert report.mtf is None and report.color is None

def test_measure_all_mono_skips_color():
    layout = ChartLayout(name="flat", patches=(PatchROI("p", 0, 0, 8, 8),))
    img = RenderedImage(data=np.full((16, 16, 1), 0.5, np.float32), color_space="sRGB", channels="MONO")
    report = measure_all(img, layout)
    assert report.color is None

# tests/test_metrics_independence.py
def test_metrics_does_not_import_simulator_modules():
    import subprocess, sys
    code = "import camsim.metrics, sys; assert 'camsim.sensor' not in sys.modules; assert 'camsim.isp' not in sys.modules; assert 'camsim.scene' not in sys.modules; assert 'camsim.optics' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
```

Additional tests: `test_measure_all_json_roundtrip`, `test_acutance_known_edge`, `test_missing_non_d65_reference_raises`, `test_measure_all_rgb_and_mono`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_metrics_summary.py tests/test_metrics_independence.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement acutance and `measure_all`**

Edge acutance from the normalized edge profile; `measure_all` runs MTF only when an edge ROI exists, SNR and colour only when patches/reference exist, and acutance only when an edge ROI exists. It must not import simulator modules. Non-D65 layouts without a reference raise.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics_summary.py tests/test_metrics_independence.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/metrics/acutance.py src/camsim/metrics/summary.py src/camsim/metrics/__init__.py tests/test_metrics_summary.py tests/test_metrics_independence.py
git commit -m "feat: add acutance and unified measure_all harness

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 31: Metrics gallery, notebook 03, and G4 report

**Files:**
- Create: `src/camsim/viz/metrics.py`, `examples/03_measure_iq.py`, `notebooks/03_measuring_iq.py`, `results/reports/g4_metrics.md`, `results/figures/g4_*.png`, `tests/test_example_03.py`

**Interfaces:**
- Produces: MTF/SNR/colour/acutance plots; G4 report with analytic tolerances.

- [ ] **Step 1: Write the failing smoke/report tests**

```python
# tests/test_example_03.py
import subprocess, sys
from pathlib import Path

def test_example_03_runs():
    r = subprocess.run([sys.executable, "examples/03_measure_iq.py"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(list(Path("results/figures").glob("03_*.png"))) >= 3
```

Additional tests: `test_g4_gaussian_mtf50`, `test_g4_airy_mtf`, `test_g4_snr_known_noise`, `test_g4_delta_e_matches_reference`, `test_g4_measure_all_serialization`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_example_03.py -v`
Expected: FAIL because the example is missing.

- [ ] **Step 3: Implement the gallery and report**

Plot MTF curves with MTF50/20 markers, SNR vs lux, ColorChecker ΔE heatmap, and acutance. Write the G4 report with analytic PSF comparisons, SNR recovery, ΔE cross-checks, and import-independence evidence.

- [ ] **Step 4: Run the G4 gate**

Run:
```bash
uv run pytest tests/test_metrics_*.py tests/test_example_03.py -v
uv run ruff check . && uv run mypy src
uv run jupytext --to ipynb --execute notebooks/03_measuring_iq.py --output /tmp/03.ipynb
```
Expected: PASS; notebook executes.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/viz/metrics.py examples/03_measure_iq.py notebooks/03_measuring_iq.py results tests/test_example_03.py
git commit -m "feat: add metrics gallery and G4 validation report

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Phase 5 — Consolidation and Sign-Off

### Task 32: README, docs site, API docs, benchmarks, and release package

**Files:**
- Modify: `README.md`, `mkdocs.yml`, `docs/index.md`, `docs/architecture.md`, `docs/validation.md`, `docs/datasheet_mapping.md`
- Create: `docs/isp_stages.md`, `docs/metrics.md`, `benchmarks/benchmark_pipeline.py`, `.github/workflows/docs.yml`, `.github/workflows/release.yml`, `results/figures/` gallery figures, `tests/test_docs_smoke.py`

**Interfaces:**
- Produces: a buildable docs site, a README gallery, benchmark timings, and a release checklist.

- [ ] **Step 1: Write the failing docs smoke test**

```python
# tests/test_docs_smoke.py
from pathlib import Path

def test_readme_links_and_gallery_exist():
    readme = Path("README.md").read_text()
    assert "results/figures/" in readme
    assert Path("docs/index.md").exists()
    assert Path("docs/validation.md").exists()

def test_mkdocs_nav_covers_core_docs():
    text = Path("mkdocs.yml").read_text()
    for page in ("concepts.md", "architecture.md", "validation.md", "metrics.md", "datasheet_mapping.md"):
        assert page in text
```

- [ ] **Step 2: Run it to confirm failure**

Run: `uv run pytest tests/test_docs_smoke.py -v`
Expected: FAIL until the docs/gallery exist.

- [ ] **Step 3: Build the docs, gallery, and benchmarks**

Add README hero figure, install instructions (uv + pip/venv), quickstart, pipeline diagram, feature list, validation summary, and gallery. Add MkDocs nav, mkdocstrings API pages, validation report, datasheet guide, ISP stages, metrics. Add a benchmark script that times scene→RAW, RAW→ISP, and metrics on 512×512. Add a release checklist (version, changelog, CITATION, coverage, docs, wheel/sdist).

- [ ] **Step 4: Run the Phase 5 checks**

Run:
```bash
uv run pytest -q
uv run pytest --cov=camsim --cov-fail-under=90
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run mkdocs build --strict
uv run python benchmarks/benchmark_pipeline.py --quick
uv build
```
Expected: all pass; coverage ≥90%; docs build; build artifacts created.

- [ ] **Step 5: Commit**

```bash
git add README.md mkdocs.yml docs benchmarks .github/workflows results/figures tests/test_docs_smoke.py
git commit -m "docs: add README gallery, docs site, and release package

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 33: Camera orchestration and auto-exposure

**Files:**
- Create: `src/camsim/camera/camera.py`, `src/camsim/camera/__init__.py`, `tests/test_camera.py`
- Modify: `tests/conftest.py` (add `camera`, `scene`, `bright_scene`, `dark_scene`, `very_dark_scene`)

**Interfaces:**
- Produces:
  - `Camera(lens: LensConfig, sensor: Sensor, isp: ISP)`
  - `ExposureSettings(exposure_s, analog_gain_db, mode)`
  - `ExposureResult(exposure_s, analog_gain_db, digital_gain, mode, converged, iterations, metered_value)`
  - `auto_expose(scene, target_fraction=0.18, max_iterations=12, tolerance=0.02) -> ExposureResult`
  - `capture(scene, exposure=None) -> RenderedImage`
  - Modify `src/camsim/__init__.py` to add `Camera` to the lazy export map

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_camera.py
import pytest
import camsim

def test_auto_exposure_converges(camera, bright_scene, dark_scene):
    bright = camera.auto_expose(bright_scene)
    dark = camera.auto_expose(dark_scene)
    assert bright.converged and dark.converged
    assert dark.exposure_s > bright.exposure_s or dark.analog_gain_db > bright.analog_gain_db

def test_auto_exposure_respects_limits(camera, very_dark_scene):
    res = camera.auto_expose(very_dark_scene)
    cfg = camera.sensor.config
    assert res.exposure_s <= cfg.exposure.max_s
    assert res.analog_gain_db <= cfg.analog_gain.max_db
    assert res.mode in {"LCG", "HCG"}

def test_fixed_exposure_protocol_is_default(camera, scene):
    out = camera.capture(scene, exposure=ExposureSettings(exposure_s=0.01, analog_gain_db=0.0, mode="LCG"))
    assert out.metadata["exposure_policy"] == "fixed"
```

Additional tests: `test_auto_exposure_records_metered_value`, `test_auto_exposure_iteration_bound`, `test_mode_auto_selects_hcg_in_low_light`, `test_capture_returns_rendered_image`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_camera.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the Camera and AE policy**

Meter the linear signal after black level with a center-weighted statistic; target 18% of usable full scale; adjust exposure, then analog gain, then digital gain, with damped iterations and mode auto-selection. Record every decision in `ExposureResult`. Default experiment capture is fixed-exposure; AE is opt-in. Add the `camera` and scene fixtures to `tests/conftest.py`.

- [ ] **Step 4: Run the tests and the full G5 suite**

Run:
```bash
uv run pytest tests/test_camera.py -v
uv run pytest -q
uv run pytest --cov=camsim --cov-fail-under=90
```
Expected: PASS; coverage ≥90%.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/camera tests/test_camera.py
git commit -m "feat: add camera orchestration and auto-exposure policy

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 34: RAW/image IO and CLI commands

**Files:**
- Create: `src/camsim/io/raw.py`, `src/camsim/io/images.py`, `src/camsim/io/__init__.py`, `tests/test_io.py`, `tests/test_cli_commands.py`
- Modify: `src/camsim/cli.py`

**Interfaces:**
- Produces:
  - `save_raw(raw: RawFrame, directory: Path) -> Path` (`.npy` + `.json` sidecar)
  - `load_raw(path: Path) -> RawFrame`
  - `save_image(image: RenderedImage, path: Path) -> Path`
  - `load_image(path: Path) -> RenderedImage`
  - CLI: `camsim render`, `camsim characterize`, `camsim measure`, `camsim figures`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_io.py
import numpy as np
from camsim.io.raw import save_raw, load_raw
from camsim.io.images import save_image, load_image

def test_raw_roundtrip(tmp_path, raw_frame):
    path = save_raw(raw_frame, tmp_path)
    loaded = load_raw(path)
    np.testing.assert_array_equal(loaded.data, raw_frame.data)
    assert loaded.cfa == raw_frame.cfa
    assert loaded.black_level_dn == raw_frame.black_level_dn

def test_image_roundtrip(tmp_path):
    from camsim.core.frames import RenderedImage
    img = RenderedImage(data=np.full((16, 16, 3), 0.5, np.float32), color_space="sRGB", channels="RGB")
    path = save_image(img, tmp_path / "out.png")
    loaded = load_image(path)
    assert loaded.channels == "RGB"
    np.testing.assert_allclose(loaded.data, img.data, atol=1.0 / 255.0)
```

```python
# tests/test_cli_commands.py
import subprocess, sys
from pathlib import Path

def test_cli_render_writes_output(tmp_path):
    r = subprocess.run([sys.executable, "-m", "camsim.cli", "render",
                        "--scene", "colorchecker", "--lux", "200",
                        "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.glob("*.png"))
    assert list(tmp_path.glob("*.npy"))
```

Additional tests: `test_cli_measure_writes_json`, `test_cli_characterize_writes_report`, `test_cli_figures_smoke`, `test_load_raw_preserves_masks`.

- [ ] **Step 2: Run the tests to confirm failure**

Run: `uv run pytest tests/test_io.py tests/test_cli_commands.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement IO and CLI commands**

Use `.npy` for arrays and a JSON sidecar for metadata and masks; PNG/TIFF via Pillow. The CLI `render` composes `scene.render` → `Sensor.expose` → `ISP.process`, writes raw, rendered image, and layout sidecar. `characterize` writes an EMVA JSON report. `measure` loads an image + layout and writes `MetricReport` JSON. `figures` regenerates the gallery.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_io.py tests/test_cli_commands.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camsim/io src/camsim/cli.py tests/test_io.py tests/test_cli_commands.py
git commit -m "feat: add RAW/image IO and render/characterize/measure CLI

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 35: G5 validation report and human sign-off (blocks ML)

**Files:**
- Create: `results/reports/validation_report.md`, `docs/validation.md` (final), `results/figures/g5_*.png`
- Modify: `CHANGELOG.md`, `CITATION.cff`, `docs/index.md`

**Interfaces:**
- Produces: a consolidated G1–G4 validation report with assumptions, limitations, and figures; a GitHub tag `v0.1.0` only after explicit human approval.

- [ ] **Step 1: Write the failing sign-off checklist test**

```python
# tests/test_g5_signoff.py
from pathlib import Path

def test_validation_report_has_all_gates():
    text = Path("results/reports/validation_report.md").read_text()
    for gate in ("G1", "G2", "G3", "G4"):
        assert gate in text
    assert "limitations" in text.lower()
    assert "assumptions" in text.lower()
```

- [ ] **Step 2: Run it to confirm failure**

Run: `uv run pytest tests/test_g5_signoff.py -v`
Expected: FAIL until the report exists.

- [ ] **Step 3: Generate the consolidated validation report**

Aggregate G1–G4 results with measured values, tolerances, figures, assumptions, known limitations, and the exact commands used. Regenerate all gallery figures. Confirm the preset provenance and datasheet mapping are complete.

- [ ] **Step 4: Run the full G5 gate and stop for human review**

Run:
```bash
uv run pytest -q
uv run pytest --cov=camsim --cov-fail-under=90
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run mkdocs build --strict
uv run python benchmarks/benchmark_pipeline.py --quick
```
Expected: all pass. Then **stop**. Present `results/reports/validation_report.md` and the figure gallery to the project owner. Do not write machine-vision code until the owner explicitly signs off.

- [ ] **Step 5: After explicit sign-off only — tag v0.1.0**

```bash
git add results/reports/validation_report.md docs/validation.md results/figures CHANGELOG.md CITATION.cff docs/index.md tests/test_g5_signoff.py
git commit -m "docs: add G5 validation report and release v0.1.0 candidate

Co-Authored-By: Claude Code <noreply@anthropic.com>"
git tag -a v0.1.0 -m "camsim v0.1.0: validated sensor, ISP, and metrics core"
```
No PyPI release at this stage. If sign-off is withheld, return to the failing gate and do not tag.

---

## Phase 6–9 — Machine-Vision Roadmap (gated after G5)

The design requires G5 sign-off before any machine-vision code. The following is the reviewable scope; detailed TDD tasks will be written as a separate plan after G5, using the validated sensor/ISP/metrics APIs.

### Phase 6 — Glyph task engine and IQ sweeps
- `tasks/glyphs.py` dataset wrapper, `tasks/cnn.py` (3-layer CNN), `tasks/train.py`, `tasks/evaluate.py`, `tasks/sweeps.py`.
- Protocol: per-modality clean training; frozen evaluation; retraining control; cross-evaluation diagnostic; RAW-to-CNN diagnostic.
- Sweeps: lux, blur, LCG/HCG, analog gain, denoise/sharpen, illuminant/AWB/CCM.
- Caching: spectral cubes, clean training set, RAW and early ISP stages.
- Deliverables: `notebooks/04_iq_vs_task_performance.py`, sweep CSVs, accuracy/lux and heatmap figures, G6 report.
- Tests: clean accuracy > 99%, deterministic training, monotonic degradation, no leakage, cached-vs-uncached equality.

### Phase 7 — Mono vs Bayer
- Per-modality clean-trained models, same scenes/seeds/architecture/budget; mono 1-channel, Bayer 3-channel RGB; optional luma ablation and cross-evaluation.
- Deliverables: `notebooks/05_mono_vs_bayer.py`, matched-accuracy analysis, MTF/SNR comparison, G7 report.
- Tests: identical optics/pixel pitch/exposure, parameter-count parity reporting, reproducible matched-lux estimate.

### Phase 8 — NSGA-II auto-tuning
- `tasks/tune.py` with Optuna NSGA-II over 4–6 ISP parameters; objectives task loss and photographic loss (ΔE2000 + noise + overshoot); cached RAW/early stages.
- Deliverables: `notebooks/06_toy_autotune.py`, Pareto front, knee point, G8 report.
- Tests: cached-vs-uncached equality, objective reproducibility, Pareto dominance filtering.

### Phase 9 — Metric correlation and v1.0
- `tasks/correlation.py`: aggregate sweeps, Spearman, standardized regression with cross-validation, tree feature importance, multicollinearity diagnostics.
- Deliverables: `notebooks/07_which_metric_predicts_task.py`, full `docs/whitepaper.md`, guidance for machine-centric specs, PyPI release v1.0.0, G9 report.
- Tests: correlation reproducibility, no target leakage, report schema, whitepaper figure regeneration.

---

## Spec Coverage Map

| Spec section | Tasks |
|---|---|
| 2 Goals | 1–35 |
| 3 Non-goals / 4 Anti-goals | 1, 15, 24, 30 (enforced by scope and tests) |
| 6.1 Package layout | 1–4, 10, 17, 24, 26, 34 |
| 6.2 Data contracts | 2, 3, 15, 26, 30, 34 |
| 6.3 Sensor/ISP separation | 15, 24 |
| 6.4 Config/provenance | 4, 17 |
| 7.1 Scene/optics/lens/lux | 6–11 |
| 7.2 Sensor chain | 12–17 |
| 7.2.1 RNG/temperature/determinism | 13, 15, 18 |
| 7.2.2 Auto-exposure | 33 |
| 7.3 ISP stages | 18–24 |
| 7.4 Metrics | 26–30 |
| 7.5 EMVA characterization | 16, 17 |
| 8 API examples | 10, 17, 24, 31, 34 |
| 9 Configuration example | 4, 17 |
| 10 Validation/sign-off | 11, 17, 25, 31, 35 |
| 11 Machine-vision experiments | Phase 6–9 roadmap |
| 12 Engineering standards | 1, 5, 32, 35 |
| 13 Phases/gates | 1–35 + roadmap |
| 14 Risks | 1, 5, 17, 35 |
| 15/16 Decisions | 1, 4, 15, 17, 30, 35 |

## Self-Review Notes

- **Spec coverage:** all design sections map to tasks; the gaps found during self-review (Camera/AE and IO/CLI) are now Tasks 33 and 34.
- **Step scan:** every step names a test, a signature, a command, or a commit; no "handle edge cases" placeholders.
- **Type consistency:** `SpectralIrradiance`, `RawFrame`, `RenderedImage`, `ColorReference`, `ChartLayout`, `MTFResult`, `SNRResult`, `ColorResult`, `MetricReport`, `EMVAReport`, and `ExposureResult` are defined once in their owning tasks and referenced consistently.
- **Review Focus:** five items each have named tests in their owning tasks.
- **Proportion:** the plan is detailed at the interface and test level; implementation bodies are left to the builder as intended.

## Execution Handoff

The plan is split into a detailed core (Phases 0–5, Tasks 1–35) and a gated ML roadmap (Phases 6–9). Do not start Phase 6 until the G5 human sign-off in Task 35.
