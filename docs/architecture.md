# Architecture

`camsim` is organized into independent layers with explicit data contracts and
a small dependency surface. This page describes the package layout, the
contracts exchanged between layers, and the rules the code enforces.

## Layers and dependency rules

| Layer | Packages | Depends on |
|---|---|---|
| Core contracts | `camsim.core` | NumPy, SciPy, Pydantic, colour-science |
| Simulation | `camsim.scene`, `camsim.optics`, `camsim.sensor`, `camsim.isp` | Core contracts |
| Characterization | `camsim.characterization` | Sensor, ISP, core contracts |
| Metrics | `camsim.metrics` | Core contracts only |
| Orchestration | `camsim.camera` | Scene, optics, sensor, ISP |
| Presentation | `camsim.viz`, `camsim.config` | Core types |
| File I/O | `camsim.io` | Core types |

Two rules are non-negotiable:

1. The sensor and ISP are reusable without the scene, optics, or camera
   packages. Only `camsim.core` is shared.
2. `camsim.metrics` never imports `camsim.sensor`, `camsim.isp`,
   `camsim.scene`, or `camsim.optics`. An import-independence test enforces
   this in a fresh subprocess.

`camsim` exposes public names lazily (PEP 562), so importing the package or a
sibling subpackage does not eagerly load the heavy simulation modules. The
scene-owned chart layout constructors return the metrics-layer type, so the
one sanctioned bridge runs scene → metrics, never the reverse.

## Package layout

```text
src/camsim/
├── __init__.py          # lazy public export map
├── _version.py          # single version source
├── cli.py               # validate / schema / info / render / characterize / measure / figures
├── core/                # units, spectral cubes, frames, colour references
├── config/              # Pydantic schemas, JSON I/O, presets
├── data/                # bundled QE/CFA/IR-cut CSVs (package resources)
├── scene/               # targets, glyphs, illuminants, chart layouts
├── optics/              # lens, PSF, propagation, vignetting
├── sensor/              # CFA, noise, gain, ADC, defects, binning, expose()
├── isp/                 # DPC, BLC, LSC, AWB, demosaic, CCM, denoise, sharpen, tone
├── characterization/    # EMVA 1288-style measurements and reports
├── metrics/             # e-SFR/MTF, SNR, colour, acutance, measure_all()
├── viz/                 # plotting helpers
├── camera/              # capture orchestration and auto-exposure
└── io/                  # RAW and rendered-image IO with sidecars
```

The public API is indexed in the [API reference](api.md). `camsim.camera` and
`camsim.io` are part of the public pipeline and covered by the validation
suite.

## Data contracts

- **`WavelengthGrid`** — validated 1-D nanometre grid, strictly increasing and
  uniform (default 360–830 nm, 5 nm step).
- **`SpectralIrradiance`** — `(H, W, λ)` cube with explicit unit, geometry, and
  optional pixel pitch. The sensor-plane boundary is before the sensor's own
  spectral filters.
- **`RawFrame`** — `uint16` data plus every piece of metadata the ISP needs
  (CFA, bit depth, black level, gains, mode, temperature, geometry, the defect
  union, a separate hot-pixel mask, PDAF mask, and configuration hash). No
  hidden globals. The frame contract omits wall-clock timestamp and package
  version; those belong to the surrounding experiment record.
- **`RenderedImage`** — display-referred `float32` image in `[0, 1]` with colour
  space, channel layout, optional linear RGB, and stage traces. The
  `linear_rgb` buffer is the unclipped pre-tone data that resolution and noise
  metrics consume; its primaries follow `color_space` (linear Display P3 for a
  P3 image), and `measure_all` converts it to linear sRGB before applying
  Rec. 709 luma. The image contract has no `linear_xyz`, timestamp, or
  package-version field.
- **`MetricReport`** — frozen dataclass with optional metric groups, layout id,
  units, and provenance; serializes to dict/JSON with non-finite values as
  `null`.
- **`ColorReference`** — per-patch reference Lab values, white point,
  illuminant, observer, and provenance, owned by `camsim.core.color` so the
  scene and metrics layers share it without a circular dependency.

## Configuration and provenance

Configuration uses Pydantic v2 models with `extra="forbid"` and units in every
field name. Cross-field validators reject physically inconsistent combinations.
Every preset carries `provenance` metadata distinguishing datasheet values from
engineering assumptions. JSON Schema export and config validation are exposed
through the `camsim schema` and `camsim validate` subcommands. Spectral data
referenced by a preset live under `camsim.data` and resolve through
`importlib.resources`, so an installed wheel constructs the same sensor without
a source checkout.

The ISP runs a fixed stage order and records its configuration hash on every
rendered image. See [ISP stages](isp_stages.md) for the stage contract, the
linear-space invariant, trace semantics, and MONO/RCCC applicability rules.

## Measurement layer

`camsim.metrics` consumes arrays plus `ChartLayout`/`ColorReference` metadata.
Chart geometry is expressed in sensor pixels, never the oversampled render
grid, and every patch ROI is half-open. The unified `measure_all` harness
selects the groups a layout supports, records skipped groups with reasons, and
emits JSON-able reports. See [Metrics](metrics.md) for definitions and
limitations.

## Presentation and coverage policy

`camsim.viz` is the presentation layer: colourblind-friendly plotting helpers,
file naming, and figure styling. It is exercised by the example scripts and
notebooks that generate the committed gallery, not by the unit suite. The
statement-and-branch coverage gate in `pyproject.toml` therefore omits
`camsim.viz`; the numerical modules must meet the 90 % threshold. This is a
deliberate split between correctness logic, which unit tests must cover, and
plotting code, which the executed gallery demonstrates. See
[Validation](validation.md) for the policy and evidence.

## Repository artifacts

- `results/reports/` holds the committed validation reports and, where
  applicable, machine-readable JSON. They are repository artifacts, not pages
  on the documentation site.
- `results/figures/` holds the committed PNG gallery, regenerated by
  `make figures`.
- `docs/images/` mirrors selected committed figures so the MkDocs build is
  self-contained and strict-buildable. Copies must stay byte-identical to
  their source.
- `benchmarks/` holds the validation report generators and the pipeline
  benchmark; the benchmark prints environment and timing tables and does not
  commit machine-specific numbers.
- `.github/workflows/` holds CI (including headless examples and notebooks
  jobs), the Pages deployment, and the tag-triggered release workflow that
  attaches wheel and sdist assets to a GitHub release.
- `scene.render` and `Sensor.expose` enforce a configurable simulation-memory
  budget (8 GiB default via `CAMSIM_MAX_SIMULATION_BYTES` or
  `max_memory_bytes=`) because the current implementation does not chunk
  spectral cubes. Full 50 MP spectral simulation is not supported.

The release process is described in [Release process](release.md).

## Validation

The validation suite covers:

- analytic invariants (units, energy conservation, linear scaling, Poisson
  variance, clipping, known-PSF MTF);
- EMVA 1288-style round-trip characterization;
- reference-library cross-checks (`colour-science`, `colour-demosaicing`,
  `scikit-image`);
- import-independence and determinism tests.

The measured results, tolerances, coverage policy, and reproduction commands
live in [Validation](validation.md). How public datasheet parameters become
validated preset fields is documented in
[Datasheet mapping](datasheet_mapping.md).
