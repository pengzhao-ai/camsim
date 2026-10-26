# camsim

<!-- Replace OWNER with the GitHub account before the public release. -->
[![CI](https://github.com/OWNER/camsim/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/camsim/actions/workflows/ci.yml)
[![Docs](https://github.com/OWNER/camsim/actions/workflows/docs.yml/badge.svg)](https://github.com/OWNER/camsim/actions/workflows/docs.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Deterministic camera sensor and ISP simulation with image-quality metrics.

`camsim` models the imaging chain from spectral scene irradiance to a rendered
image, measures image quality with standard metrics, and studies how optics,
sensor noise, and ISP parameter choices affect machine-vision accuracy. It is
built for camera and ISP engineers, machine-vision engineers, and researchers
who need a reproducible, physically grounded testbed rather than a
notebook-only demo.

> **Status:** early Phase 0 (repository scaffold). The package installs, the CLI
> skeleton runs, and CI/lint/typecheck/docs are wired up. The simulation, ISP,
> metrics, and machine-vision modules arrive in later phases.

## Layers

1. **Simulation pipeline** — spectral scenes and optics (supporting modules), a
   rigorous sensor model (irradiance → RAW), and a modular ISP (RAW → rendered
   image). The sensor and ISP are the reusable core and do not depend on the
   scene or optics helpers.
2. **Image-quality metrics** — an independent harness for e-SFR/MTF, SNR,
   colour error (ΔE76/ΔE2000), and acutance, operating on arrays plus metadata.
3. **Machine-vision experiments** — a small PyTorch task engine and experiment
   suite, gated behind a formal pipeline/metrics sign-off (G5).

```
spectral scene → optics → sensor irradiance → RAW → ISP → rendered image → metrics
```

The documentation site covers the [concepts](docs/concepts.md),
[architecture](docs/architecture.md), [validation gates](docs/validation.md),
and the [datasheet-to-JSON mapping](docs/datasheet_mapping.md).

## Roadmap

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | Scaffold, core types, config schema, CLI, CI, docs | **G0: install, lint, typecheck, tests, docs** |
| 1 | Scene, illuminants, optics, spectral data | G1: scene/optics report |
| 2 | Sensor model, flagship preset, EMVA characterization | G2: sensor report |
| 3 | ISP stages and pipeline | G3: ISP report |
| 4 | Metrics harness | G4: metrics report |
| 5 | Consolidation, gallery, benchmarks | **G5: human sign-off; tag v0.1.0** |
| 6–9 | Machine-vision experiments, studies, auto-tuning, whitepaper | G6–G9; v1.0.0 |

Machine-vision code is deliberately gated behind G5; see
[docs/validation.md](docs/validation.md) for the sign-off conditions.

## Installation

Python 3.11–3.13 is required.

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync --all-extras
```

pip/venv fallback:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all]"
```

Core dependencies are NumPy, SciPy, Pydantic v2, Matplotlib, Pillow, and
colour-science. Optional extras: `[ml]`, `[tune]`, `[docs]`, `[dev]`, `[all]`.

## Quickstart

```python
import camsim

print(camsim.__version__)

from camsim.config.io import load_sensor

sensor = load_sensor("src/camsim/config/presets/sensors/flagship_phone_50mp.json")
print(sensor.name, sensor.pixel.pitch_um)
```

```bash
camsim info
camsim validate src/camsim/config/presets/sensors/flagship_phone_50mp.json
camsim schema sensor
```

The end-to-end sensor → ISP → metrics quickstart is added with the simulation
modules.

## Development

The same commands back the Makefile targets and CI:

```bash
make setup        # uv sync --all-extras
make test         # uv run pytest
make lint         # ruff check + ruff format --check
make format       # ruff check --fix + ruff format
make typecheck    # uv run mypy src
make coverage     # pytest with coverage, as in CI
make docs         # uv run mkdocs build --strict
make build        # uv build (wheel + sdist)
make notebooks    # execute Jupytext notebooks headlessly
make gate         # the full G0 gate: sync, lint, typecheck, test, coverage, docs, build
```

A single test can be run with
`uv run pytest tests/test_package.py::test_import_and_version -v`.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. The
project follows the [Contributor Covenant](CODE_OF_CONDUCT.md), records changes
in [CHANGELOG.md](CHANGELOG.md), and documents third-party material in
[THIRD_PARTY.md](THIRD_PARTY.md).

## Repository conventions

- `src/` layout; Apache-2.0; package and import name `camsim`.
- `uv` with a committed `uv.lock`; the project-local `.venv/` is gitignored.
- Notebooks are Jupytext percent-format `.py` files; `.ipynb` files are never
  committed.
- Stochastic routines use explicit NumPy generators and seeds; the pipeline is
  deterministic within one environment.
- Phase gates G0–G5 are documented in [docs/validation.md](docs/validation.md);
  no machine-vision code lands before the G5 human sign-off.
- Citation metadata lives in [CITATION.cff](CITATION.cff). The design spec
  lives at `docs/superpowers/specs/`. The superseded reference plan is kept in
  `docs/reference/` for provenance and is not part of the documentation site.

## License

Apache-2.0. See [LICENSE](LICENSE).
