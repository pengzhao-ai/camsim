# Architecture

## Layers and dependency rules

| Layer | Packages | Depends on |
|---|---|---|
| Core contracts | `camsim.core` | NumPy, SciPy, Pydantic, colour-science |
| Simulation | `camsim.scene`, `camsim.optics`, `camsim.sensor`, `camsim.isp` | Core contracts |
| Characterization | `camsim.characterization` | Sensor, core contracts |
| Metrics | `camsim.metrics` | Core contracts only |
| Orchestration | `camsim.camera` | Scene, optics, sensor, ISP |
| Machine vision | `camsim.tasks` | Gated behind G5 sign-off |
| Presentation | `camsim.viz`, `camsim.io`, `camsim.config` | Core types |

Two rules are non-negotiable:

1. The sensor and ISP are reusable without the scene, optics, or camera
   packages. Only `camsim.core` is shared.
2. `camsim.metrics` never imports `camsim.sensor`, `camsim.isp`,
   `camsim.scene`, or `camsim.optics`.

`camsim` exposes public names lazily (PEP 562), so importing the package or a
sibling subpackage does not eagerly load the heavy simulation modules.

## Package layout

```text
src/camsim/
├── __init__.py          # lazy public export map
├── _version.py          # single version source
├── cli.py               # validate / schema / info
├── core/                # units, spectral cubes, frames, colour references
├── config/              # Pydantic schemas, JSON I/O, presets
├── scene/               # targets, glyphs, illuminants, spectral data
├── optics/              # lens, PSF, propagation, vignetting
├── sensor/              # CFA, noise, gain, ADC, defects, binning, expose()
├── isp/                 # DPC, BLC, LSC, AWB, demosaic, CCM, denoise, sharpen, tone
├── characterization/    # EMVA 1288-style measurements and reports
├── camera/              # convenience orchestration and auto-exposure
├── metrics/             # e-SFR/MTF, SNR, colour, acutance, measure_all()
├── tasks/               # PyTorch task engine and experiments (gated)
├── viz/                 # plotting helpers
└── io/                  # RAW and image I/O
```

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
  space, channel layout, optional linear RGB/XYZ, and stage traces.
- **`MetricReport`** — Pydantic model with optional metric groups, layout id,
  units, and provenance; serializes to dict/JSON.
- **`ColorReference`** — per-patch reference Lab values, white point,
  illuminant, observer, and provenance, owned by `camsim.core.color` so the
  scene and metrics layers share it without a circular dependency.

## Configuration and provenance

Configuration uses Pydantic v2 models with `extra="forbid"` and units in every
field name. Cross-field validators reject physically inconsistent combinations.
Every preset carries `provenance` metadata distinguishing datasheet values from
engineering assumptions. JSON Schema export and config validation are exposed
through the `camsim schema` and `camsim validate` subcommands.

## Quality gates

Phase gates validate each layer before the next begins:

- analytic invariants (units, energy conservation, linear scaling, Poisson
  variance, clipping, known-PSF MTF);
- EMVA 1288-style round-trip characterization;
- reference-library cross-checks (`colour-science`, `colour-demosaicing`,
  `scikit-image`);
- import-independence and determinism tests.

Machine-vision work does not begin until the pipeline and metrics pass the G5
human sign-off.

## Process material

The design spec and implementation plans live under `docs/superpowers/`. The
superseded reference plan is kept in `docs/reference/` for provenance. Both are
excluded from the published documentation site.
