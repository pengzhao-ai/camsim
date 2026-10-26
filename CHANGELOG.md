# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until the first tagged release (`v0.1.0`, after the G5 human sign-off), changes
accumulate under `Unreleased`; version numbers shown here follow
`src/camsim/_version.py`.

## [Unreleased]

### Added

#### Phase 0 — Foundation

- Installable `camsim` package with a `src/` layout, a single version source
  (`0.1.0.dev0`), lazy (PEP 562) public exports, and a `camsim` console entry
  point with `info`, `schema`, and `validate` subcommands.
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

[Unreleased]: https://github.com/OWNER/camsim/commits/main
