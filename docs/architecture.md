# Architecture

camsim has completed Phase 5 consolidation: the simulation pipeline (G1–G4),
camera orchestration, IO, and CLI are implemented, and the G5 human sign-off
is pending. Machine-vision code is gated behind G5. This page describes the
package layout, data contracts, and dependency rules that the code enforces.

## Layers and dependency rules

| Layer | Packages | Depends on | Status |
|---|---|---|---|
| Core contracts | `camsim.core` | NumPy, SciPy, Pydantic, colour-science | implemented |
| Simulation | `camsim.scene`, `camsim.optics`, `camsim.sensor`, `camsim.isp` | Core contracts | implemented |
| Characterization | `camsim.characterization` | Sensor, ISP, core contracts | implemented |
| Metrics | `camsim.metrics` | Core contracts only | implemented |
| Orchestration | `camsim.camera` | Scene, optics, sensor, ISP | implemented |
| Machine vision | `camsim.tasks` | PyTorch | gated behind G5 |
| Presentation | `camsim.viz`, `camsim.config` | Core types | implemented |
| File I/O | `camsim.io` | Core types | implemented |

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
├── cli.py               # validate / schema / info
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
├── io/                  # RAW and rendered-image IO with sidecars
└── tasks/               # machine-vision experiments (gated)
```

The implemented public API is indexed in the [API reference](api.md).
`camsim.camera` and `camsim.io` are implemented and covered by the validation
gates; `camsim.tasks` does not exist as working code until G5 is signed off.

## Data contracts

- **`WavelengthGrid`** — validated 1-D nanometre grid, strictly increasing and
  uniform (default 360–830 nm, 5 nm step).
- **`SpectralIrradiance`** — `(H, W, λ)` cube with explicit unit, geometry, and
  optional pixel pitch. The sensor-plane boundary is before the sensor's own
  spectral filters.
- **`RawFrame`** — `uint16` data plus every piece of metadata the ISP needs
  (CFA, bit depth, black level, gains, mode, temperature, geometry, masks,
  simulation provenance). No hidden globals.
- **`RenderedImage`** — display-referred `float32` image in `[0, 1]` with colour
  space, channel layout, optional linear RGB, and stage traces. The
  `linear_rgb` buffer is the unclipped pre-tone data that resolution and noise
  metrics consume.
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
`camsim.viz`; the simulation, metrics, configuration, CLI, and characterization
modules remain under the 90 % threshold. This is a deliberate split between
correctness logic, which unit tests must cover, and plotting code, which the
executed gallery demonstrates. See [Validation](validation.md) for the policy
and evidence.

## Repository artifacts

- `results/reports/` holds the G1–G4 Markdown reports and, for G2–G4,
  machine-readable JSON. They are repository artifacts, not pages on the
  documentation site.
- `results/figures/` holds the committed PNG gallery, regenerated by
  `make figures`.
- `docs/images/` mirrors selected committed figures so the MkDocs build is
  self-contained and strict-buildable. Copies must stay byte-identical to
  their source.
- `benchmarks/` holds the G1 report generator and the pipeline benchmark; the
  benchmark prints environment and timing tables and does not commit
  machine-specific numbers.
- `.github/workflows/` holds CI, the Pages deployment, and the tag-triggered
  release workflow that attaches wheel and sdist assets to a GitHub release.

The release contract is in [Release](release.md).

## Quality gates

Phase gates validate each layer before the next begins:

- analytic invariants (units, energy conservation, linear scaling, Poisson
  variance, clipping, known-PSF MTF);
- EMVA 1288-style round-trip characterization;
- reference-library cross-checks (`colour-science`, `colour-demosaicing`,
  `scikit-image`);
- import-independence and determinism tests.

The gate definitions, current status, coverage policy, and reproduction
commands live in [Validation](validation.md). How public datasheet parameters
become validated preset fields is documented in
[Datasheet mapping](datasheet_mapping.md).

Machine-vision work does not begin until the pipeline and metrics pass the G5
human sign-off.

## Process material

The design spec and implementation plans live under `docs/superpowers/`. The
superseded reference plan is kept in `docs/reference/` for provenance. Both are
excluded from the published documentation site.
