# camsim

Deterministic camera sensor and ISP simulation with image-quality metrics.

`camsim` is a professional Python library that simulates a camera from spectral
scene irradiance to a rendered image, measures image quality with standard
metrics, and studies how optics, sensor noise, and ISP parameter choices affect
machine-vision task accuracy.

!!! note "Project status"
    Early Phase 0 (repository scaffold). The package installs, the CLI skeleton
    runs, and lint, typecheck, tests, and documentation build in CI. The
    simulation, ISP, metrics, and machine-vision modules arrive in later phases.

## Three clean layers

1. **Simulation pipeline** — spectral scenes and optics helpers, a rigorous
   sensor model (irradiance → RAW), and a modular ISP (RAW → rendered image).
   The reusable core is the sensor + ISP pair; neither depends on the scene or
   optics packages.
2. **Image-quality metrics** — an independent harness for e-SFR/MTF, SNR,
   colour error, and acutance. It operates on arrays plus metadata and never
   imports the simulator.
3. **Machine-vision experiments** — a small PyTorch task engine and experiment
   suite, gated behind the G5 human sign-off.

## Install

```bash
uv sync --all-extras
```

or, with pip and a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Quick check

```python
import camsim

print(camsim.__version__)
```

```bash
camsim info
```

## Where to go next

- [Concepts](concepts.md) — the imaging chain, units, linear-space invariant,
  and determinism contract.
- [Architecture](architecture.md) — package layout, data contracts, and
  dependency rules.
