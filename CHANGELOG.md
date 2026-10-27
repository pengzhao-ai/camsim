# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The `[0.1.0]` heading names the **release candidate**, not the distribution
version: `src/camsim/_version.py` remains `0.1.0.dev0` until the tag is cut
after the G5 human sign-off. The candidate is not tagged or published. After
tagging, the heading is dated and a fresh `Unreleased` section opens.

## [0.1.0] - unreleased candidate

**Status: awaiting G5 human sign-off.** The core pipeline is validated end to
end; the tag, push, and publication are separate steps performed only after
the project owner approves `results/reports/validation_report.md`.

### Added

#### Phase 0 — Foundation

- Installable `camsim` package with a `src/` layout, a single version source
  (`0.1.0.dev0`), lazy (PEP 562) public exports, and a `camsim` console entry
  point with `info`, `schema`, `validate`, `render`, `characterize`, `measure`,
  and `figures` subcommands.
- Radiometric unit conversions and validated spectral types: `WavelengthGrid`,
  `SpectralIrradiance`, photon-energy helpers, area-conserving spatial
  resampling, and overlap-weighted wavelength integration.
- Frame and colour contracts: `RawFrame` with self-describing metadata,
  `RenderedImage`, `CFAPattern`, `ColorSpace`, and `ColorReference`.
- Versioned Pydantic v2 configuration schema for sensors and ISPs with
  unit-bearing field names, `extra="forbid"`, cross-field physical validation,
  JSON Schema export, and provenance metadata.
- Representative sensor and ISP presets (`flagship_phone_50mp`, default ISP).
- MkDocs Material documentation site with concepts, architecture, validation,
  and datasheet-mapping pages, including a Mermaid imaging-chain diagram.
- Continuous integration on Ubuntu and macOS across Python 3.11, 3.12, and
  3.13: ruff lint and format check, strict mypy, pytest with coverage, strict
  documentation build, and wheel/sdist build.
- Documentation workflow that builds with warnings as errors on pull requests
  and deploys the site to GitHub Pages from `main`.
- Contribution guide, Contributor Covenant code of conduct, changelog,
  citation metadata, third-party notices, and Apache-2.0 licence.
- Pre-commit hooks, Jupytext notebook pairing, Makefile targets (`setup`,
  `test`, `lint`, `format`, `typecheck`, `coverage`, `docs`, `build`,
  `notebooks`, `figures`, `gate`, `clean`), and the executable G0 gate test.

#### Phase 1 — Scene and optics

- Spectral scene rendering with photometric lux normalization, D65/3000 K/4000 K
  illuminants, ColorChecker reflectance, slanted-edge, flat-patch, Siemens-star,
  and glyph targets, and explicit energy-conserving spatial resampling.
- Gaussian, Airy, and box point-spread functions with analytic MTF helpers,
  lens geometry, and vignetting (uniform for the reference lens).
- G1 validation report and figures covering photometry, PSF MTF, energy
  conservation, edge-angle recovery, the spectral-cube contract, and the
  reference lens.

#### Phase 2 — Sensor

- Spectral projection through QE, CFA, and IR-cut curves; shot, read, dark, and
  fixed-pattern noise; full-well and ADC saturation with clipping masks; LCG/HCG
  dual conversion gain; defects; binning; and the self-describing `RawFrame`.
- EMVA 1288-style characterization recovering read noise, DSNU, PRNU, full well,
  linearity, dynamic range, system gain, and dark current within stated
  tolerances.
- G2 validation report, machine-readable results, and six figures.

#### Phase 3 — ISP

- Fixed-order ISP: DPC, BLC, LSC, AWB, bilinear/MHC demosaicing, ridge CCM,
  linear-light denoise, thresholded unsharp mask, tone/OETF, and output
  primaries. Every stage before tone is linear float32.
- RCCC baseline reconstruction (shape/linearity/smoke only) and a documented
  Bayer-only colour and MTF caveat.
- G3 validation report, machine-readable results, and the ISP stage gallery,
  including ColorChecker ΔE2000, AWB, demosaic reference cross-checks, LSC,
  denoise, sharpen, and RCCC evidence.

#### Phase 4 — Metrics

- Import-independent metrics harness: ISO 12233-style slanted-edge e-SFR/MTF,
  spatial and temporal SNR, CIEDE2000/CIEDE76, acutance, chart layouts, and the
  unified `measure_all` report with strict JSON serialization.
- G4 validation report, machine-readable results, and analytic-reference
  figures for MTF, SNR, colour, and acutance.

#### Phase 5 — Consolidation

- `Camera` orchestration with fixed exposure and opt-in auto-exposure
  (`Camera.auto_expose`) that probes the real sensor, respects the preset's
  physical windows, and reports every decision.
- RAW and image IO, layout/report serialization, and the `render`,
  `characterize`, `measure`, and `figures` CLI commands.
- Deterministic pipeline benchmark with quick and full modes.
- Consolidated G5 validation report (`results/reports/validation_report.md`),
  cross-cutting denoise/sharpen/auto-exposure/RCCC checks, determinism and
  import-independence evidence, and the `g5_*` summary figures.
- G5 sign-off tests, documentation updates, citation metadata, and the release
  candidate metadata.

### Validation

- G0: quality-gate commands pass locally; CI workflows (including examples
  and notebooks) were added but execution is pending a push.
- G1: 7/7 checks; G2: 16/16; G3: 21/21; G4: 31/31; G5 cross-cutting: 16/16.
- Test suite: all tests pass; subject-module coverage is 91.73% against the
  90% gate with the documented `*/camsim/viz/*` omission, and whole-package
  coverage including viz is 83.55% (reviewed exception).
- The validation is simulated: it is not a certified ISO 12233 or EMVA 1288
  measurement, and the flagship preset is a representative engineering design
  point, not a measurement of a specific commercial part.
- **No machine-vision code or claim is included.** `camsim.tasks` remains a
  gated docstring-only namespace.

[0.1.0]: https://github.com/OWNER/camsim/commits/main
