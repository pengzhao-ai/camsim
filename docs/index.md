# camsim

Deterministic camera sensor and ISP simulation with image-quality metrics.

`camsim` is a professional Python library that simulates a camera from spectral
scene irradiance to a rendered image, measures image quality with standard
metrics, and studies how optics, sensor noise, and ISP parameter choices affect
machine-vision task accuracy.

![ISP stage contact sheet](images/02_isp_stage_contact_sheet.png)

*The default ISP applied to a spectral ColorChecker scene, shown stage by
stage: RAW, defect correction, black-level correction, lens shading, white
balance, demosaicing, colour correction, denoise, sharpen, and tone. The figure
mirrors `results/figures/02_isp_stage_contact_sheet.png` in the repository.*

!!! note "Project status"
    **v0.1.0 released; G5 signed off.** G1–G4 are PASS with committed
    validation reports, all G5 cross-cutting checks pass, and the signed
    report is `results/reports/validation_report.md` in the repository. The
    version is `0.1.0` and the tag is `v0.1.0`. Machine-vision code is gated
    behind G5 and is not implemented; Phase 6+ is deferred.

## Three clean layers

1. **Simulation pipeline** — spectral scene and optics helpers, a rigorous
   sensor model (irradiance → RAW), and a modular ISP (RAW → rendered image).
   The reusable core is the sensor + ISP pair; neither depends on the scene or
   optics packages.
2. **Image-quality metrics** — an independent harness for e-SFR/MTF, SNR,
   colour error, and acutance. It operates on arrays plus metadata and never
   imports the simulator.
3. **Machine-vision experiments** — a small PyTorch task engine and experiment
   suite, gated behind the G5 human sign-off.

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
```

Or with pip and a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
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

## End-to-end example

```python
import camsim
from camsim.config.io import PRESETS_DIR, load_isp, load_sensor
from camsim.metrics import measure_all
from camsim.scene.charts import slanted_edge_layout
from camsim.scene.targets import render

scene = render("slanted_edge", lux=200.0, shape=(256, 256), oversampling=2, seed=11)
sensor = camsim.Sensor(
    load_sensor(PRESETS_DIR / "sensors" / "flagship_phone_50mp.json"), seed=1
)
raw = sensor.expose(scene.irradiance, exposure_s=0.01, mode="LCG", seed=7)
image = camsim.ISP(load_isp(PRESETS_DIR / "isp" / "default.json")).process(raw)
report = measure_all(image, slanted_edge_layout(shape=(256, 256)))

print(image.data.shape, image.color_space)
print(report.mtf.mtf50_cp, report.acutance)
```

## Where to go next

- [Concepts](concepts.md) — the imaging chain, units, linear-space invariant,
  and determinism contract.
- [Architecture](architecture.md) — package layout, data contracts, and
  dependency rules.
- [ISP stages](isp_stages.md) — every stage's role, parameters, trace, and
  linear-space invariant.
- [Metrics](metrics.md) — chart layouts, e-SFR, SNR, colour, acutance, and the
  unified report.
- [Validation](validation.md) — gate evidence and the coverage policy.
- [Datasheet mapping](datasheet_mapping.md) — how preset fields trace to
  public specifications.
- [API reference](api.md) — the public modules and types.
- [Release](release.md) — the release checklist and versioning policy.

!!! note "Images on this site"
    The figures under `docs/images/` are copies of selected files in the
    repository's committed gallery, `results/figures/`. The copies keep the
    documentation build self-contained, so MkDocs strict mode never depends on
    paths outside `docs/`. Regenerate the source figures with `make figures`,
    then copy the changed files into `docs/images/`.
