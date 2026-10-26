# camsim

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
```

```bash
camsim info
```

The end-to-end sensor → ISP → metrics quickstart is added with the simulation
modules.

## Development

The same commands back the Makefile targets and CI:

```bash
make setup        # uv sync --all-extras
make test         # uv run pytest
make lint         # ruff check + ruff format --check
make typecheck    # uv run mypy src
make docs         # uv run mkdocs build --strict
make notebooks    # execute Jupytext notebooks headlessly
```

A single test can be run with
`uv run pytest tests/test_package.py::test_import_and_version -v`.

## Repository conventions

- `src/` layout; Apache-2.0; package and import name `camsim`.
- `uv` with a committed `uv.lock`; the project-local `.venv/` is gitignored.
- Notebooks are Jupytext percent-format `.py` files; `.ipynb` files are never
  committed.
- Stochastic routines use explicit NumPy generators and seeds; the pipeline is
  deterministic within one environment.
- The design spec lives at `docs/superpowers/specs/`. The superseded reference
  plan is kept in `docs/reference/` for provenance and is not part of the
  documentation site.

## License

Apache-2.0. See [LICENSE](LICENSE).
