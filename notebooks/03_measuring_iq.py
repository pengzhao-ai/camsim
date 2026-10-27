"""Notebook 03: measuring image quality with MTF, SNR, colour, and acutance.

This notebook narrates the Phase 4 metrics layer with the same shared
measurement that produces the G4 gate report. It is a teaching companion, not
a gate artifact: it uses `measure_g4(quick=True)`, displays figures inline,
writes no report, and never overwrites the committed
`results/reports/g4_metrics.md`.

The physics in the figures:

* **MTF and e-SFR.** A slanted edge crosses every sub-pixel phase as it runs
  down the frame. Projecting the pixels onto the edge normal reconstructs the
  edge-spread function (ESF); differentiating it gives the line-spread function
  (LSF); the magnitude of the normalized Fourier transform of the LSF is the
  modulation transfer function (MTF). MTF50 and MTF20 are the frequencies where
  the curve falls to one half and one fifth. The measurement is displayed
  against the exact discrete transfer function of the known PSF, so a
  systematic bias is visible.
* **SNR, spatial versus temporal.** Spatial SNR is `20 log10(mean / std)` over
  one frame and therefore includes fixed-pattern noise. Temporal SNR pools the
  per-pixel temporal variances over a stack, so a static per-pixel pattern
  cancels; the difference between the two is the fixed-pattern rejection.
* **CIEDE2000.** Lab is a nonlinear remapping of XYZ designed so that equal
  numeric distances approximate equal perceived differences. CIEDE2000
  rescales the a* axis and adds a blue-region rotation term, which is why it
  disagrees with the plain Euclidean delta-E76 exactly where the eye does.
* **Acutance.** The contrast-normalized RMS gradient of one edge ROI is a
  cheap sharpness proxy for correlation studies. It is not a calibrated MTF and
  is only comparable between measurements made with the same ROI geometry.
* **The unified harness.** `measure_all` runs the groups a chart layout and
  image support, skips the rest explicitly, and returns a strict-JSON report
  with not-reached MTF crossings as `null`.

The notebook executes in well under two minutes on a laptop. The committed
`.py` source is the notebook; no `.ipynb` is committed.
"""

# %% [markdown]
# # 03 - Measuring image quality
#
# This notebook uses the shared quick measurement from
# `camsim.characterization.measure_g4`. Quick mode exercises every code path on
# smaller grids and is **not** G4 gate evidence; the committed report comes
# from the full `examples/03_measure_iq.py` run.

# %%
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

from camsim.characterization import measure_g4
from camsim.core.frames import RenderedImage
from camsim.metrics import ChartLayout, PatchROI, measure_all, snr_patch, snr_temporal
from camsim.viz import (
    MTFPlotSeries,
    plot_acutance,
    plot_delta_e_heatmap,
    plot_mtf_curve,
    set_style,
)

# When executed by Jupyter, switch to the inline backend so every figure is
# embedded in the notebook. Outside Jupyter the Agg backend stays active.
_ipython = get_ipython()
if _ipython is not None:
    _ipython.run_line_magic("matplotlib", "inline")

set_style()

payload = measure_g4(quick=True)
print(f"G4 quick status: {payload['status']} (quick mode is not gate evidence)")
print(f"checks: {len(payload['checks'])}, all pass: {all(c['pass'] for c in payload['checks'])}")

# %% [markdown]
# ## 1. MTF from a slanted edge
#
# The measurement chain has five steps:
#
# 1. project every pixel in the analysis region onto the edge normal and bin
#    the samples at eight sub-pixel intervals per pixel;
# 2. differentiate the edge-spread function to obtain the line-spread function;
# 3. taper the LSF with a Hamming window to suppress the rectangular window's
#    sidelobes;
# 4. take the Fourier magnitude, normalize by the DC term, and divide out the
#    central-difference estimator's own `sinc` roll-off;
# 5. truncate to the native 0.5 cycles/pixel Nyquist limit and interpolate the
#    MTF50 and MTF20 crossings.
#
# Eight samples per pixel are used for the analytic gate cases because the
# conventional four carry roughly a 2 % MTF50 bias for near-diffraction-limited
# systems. The dashed reference is the exact discrete transfer function of the
# same PSF, not a fit to the measured curve.

# %%
series = [
    MTFPlotSeries(
        label=case["name"],
        frequency_cp=case["frequency_cp"],
        measured_mtf=case["measured_mtf"],
        analytic_mtf=case["analytic_mtf"],
        mtf50_cp=case["measured_mtf50_cp"],
        mtf20_cp=case["measured_mtf20_cp"],
    )
    for case in payload["mtf"]["cases"]
]
figure, _axes = plot_mtf_curve(series)
plt.show()

for case in payload["mtf"]["cases"]:
    print(
        f"{case['name']:22s} MTF50 {case['measured_mtf50_cp']:.5f} cp "
        f"({100.0 * case['relative_error_mtf50']:+.3f} % vs analytic)"
    )

# %% [markdown]
# The Airy f/2.8 case appears only as a diagnostic. Its transfer function never
# reaches 0.20 inside Nyquist, so it has no finite MTF20, and its MTF50 lies
# closest to the band edge where the estimator carries more bias.

# %%
for diagnostic in payload["mtf"]["diagnostics"]:
    print(
        f"{diagnostic['name']}: MTF50 {diagnostic['mtf50_cp']:.5f} cp, "
        f"MTF20 {diagnostic['mtf20_cp']} (not reached)"
    )

# %% [markdown]
# ## 2. SNR: spatial versus temporal
#
# A single-frame spatial SNR is the sample standard deviation of every pixel in
# a flat patch. It counts random noise **and** fixed-pattern noise, so a sensor
# with strong pixel-to-pixel gain or offset variation reports a worse SNR than
# its temporal noise alone would suggest.
#
# The temporal estimator takes each pixel's variance over a stack and pools
# those variances. A static pattern shifts every frame equally and cancels; the
# reported mean still carries it, but the standard deviation does not. The
# difference between the two estimates is the fixed-pattern rejection.
#
# The synthetic demonstration below adds a static per-pixel pattern of sigma
# 0.02 and temporal noise of sigma 0.01 to a 0.5-mean patch. The temporal SNR
# should recover about `20 log10(0.5 / 0.01) = 34 dB`, while the single-frame
# spatial SNR is much lower.

# %%
rng = np.random.default_rng(2026)
frames, size = 24, 32
fixed_pattern = 0.02 * rng.standard_normal((size, size))
stack = 0.5 + fixed_pattern[None, :, :] + 0.01 * rng.standard_normal((frames, size, size))
roi = PatchROI(name="flat", x=0, y=0, width=size, height=size)
temporal = snr_temporal(stack, roi)
spatial = snr_patch(stack[0], roi, channel="mono")
print(f"spatial single-frame SNR: {spatial.snr_db:6.2f} dB (includes the fixed pattern)")
print(f"temporal SNR:             {temporal.snr_db:6.2f} dB (fixed pattern rejected)")
print(f"fixed-pattern rejection:  {temporal.snr_db - spatial.snr_db:6.2f} dB")

# %% [markdown]
# The measured quick payload repeats the same logic at the gate sizes and adds
# known-noise references. The spatial per-channel values are exact against
# their own sample statistics by construction; the temporal check is the
# meaningful one because its reference is the known temporal sigma, not the
# estimator under test.

# %%
for row in payload["snr"]["spatial"]["measurements"]:
    print(
        f"spatial {row['channel']:4s} mean {row['mean']:.4f} "
        f"std {row['std']:.4f} SNR {row['snr_db']:.2f} dB"
    )
temporal_payload = payload["snr"]["temporal"]
print(
    f"temporal: SNR {temporal_payload['snr_db']:.2f} dB vs known-noise "
    f"{temporal_payload['expected_snr_db']:.2f} dB "
    f"(error {temporal_payload['error_db']:+.3f} dB)"
)

# %% [markdown]
# ## 3. Colour difference: delta-E76 and CIEDE2000
#
# `delta-E76` is the Euclidean distance in CIE Lab. Lab is not perfectly
# perceptually uniform, and CIEDE2000 corrects the worst of it: it rescales the
# a* axis by a chroma-dependent factor, weights lightness, chroma, and hue
# separately, and adds a rotation term that compresses differences in the blue
# region around hue 275 degrees. The hue bookkeeping is the fragile part: hue
# differences wrap modulo 360, a pair exactly 180 degrees apart must take the
# shorter sign-preserving branch, and a neutral colour has no defined hue.
#
# The measurement compares both implementations with `colour-science` on the 34
# published Sharma et al. (2005) pairs plus constructed neutral, lightness,
# hue-rotation, and identity pairs. The maximum absolute disagreement is at
# floating-point round-off, not at a tolerance-setting level.

# %%
colour_section = payload["color"]
print(f"pairs compared: {colour_section['pair_count']}")
print(f"max |deltaE76  - colour-science|: {colour_section['max_abs_error_delta_e76']:.3e}")
print(f"max |deltaE2000 - colour-science|: {colour_section['max_abs_error_delta_e2000']:.3e}")
print(
    "max |deltaE2000 - published Sharma|: "
    f"{colour_section['sharma_published_max_abs_error_delta_e2000']:.3e}"
)
for pair in colour_section["pairs"][:4]:
    print(
        f"{pair['name']:12s} our {pair['delta_e2000']:8.4f} "
        f"reference {pair['reference_delta_e2000']:8.4f}"
    )

# %% [markdown]
# The heatmap below is a synthetic ColorChecker-like chart built from the
# scene's spectrally computed reference Lab, with a deterministic per-patch
# perturbation so the chart carries real structure. It is a measurement
# vehicle, not a camera render, and its mean delta-E is not a camera
# colour-accuracy claim.

# %%
chart = payload["measure_all"]["chart"]
figure, _axes = plot_delta_e_heatmap(
    chart["patch_names"],
    chart["per_patch_delta_e2000"],
    title=f"Synthetic ColorChecker CIEDE2000 (mean {chart['mean_delta_e2000']:.2f})",
)
plt.show()

# %% [markdown]
# ## 4. Acutance versus blur
#
# Acutance is the contrast-normalized RMS gradient of an edge ROI. Dividing out
# the edge's own `max - min` makes it insensitive to exposure and chart
# contrast; a Gaussian blur spreads the transition and lowers the number. It is
# a correlation-study proxy, not a calibrated resolution measurement: noise and
# sharpening halos also raise it, and its absolute value depends on ROI size.

# %%
figure, _axes = plot_acutance(
    payload["acutance_sweep"]["sigma_px"],
    payload["acutance_sweep"]["acutance"],
)
plt.show()

for sigma, value in zip(
    payload["acutance_sweep"]["sigma_px"],
    payload["acutance_sweep"]["acutance"],
    strict=True,
):
    print(f"sigma {sigma:.2f} px: acutance {value:.5f}")

# %% [markdown]
# ## 5. The unified harness and strict JSON
#
# `measure_all` executes exactly the groups a layout and image support. A chart
# layout with patch ROIs enables SNR and colour; an edge ROI enables MTF and
# acutance; a MONO image skips colour and records why. Non-finite crossings are
# written as JSON `null` so the report parses with any strict reader.
#
# The quick payload below shows the synthetic chart's colour/SNR summary, the
# separate edge layout's MTF and acutance, the mono path, and the
# import-independence evidence.

# %%
print(
    f"chart: {chart['patch_count']} patches, mean deltaE2000 "
    f"{chart['mean_delta_e2000']:.3f}, SNR patches x channels "
    f"{chart['snr_patch_count']} x {len(chart['snr_channels'])}"
)
edge = payload["measure_all"]["edge"]
print(f"edge: MTF50 {edge['mtf50_cp']} (not reached), acutance {edge['acutance']:.5f}")
mono = payload["measure_all"]["mono"]
print(f"mono: SNR channels {mono['snr_channels']}, colour present {mono['color_present']}")
serialization = payload["measure_all"]["serialization"]
print(
    "serialization: round trip "
    f"{serialization['roundtrip_ok']}, NaN tokens absent "
    f"{serialization['nan_token_absent']}, null crossing {serialization['null_crossing']}"
)
independence = payload["measure_all"]["import_independence"]
print(f"import independence: {'PASS' if independence['passed'] else 'FAIL'}")
print(f"forbidden modules: {', '.join(independence['forbidden_modules'])}")

# %% [markdown]
# A small hand-built example makes the layout contract concrete: one flat patch
# and no colour reference produce an SNR group and skip colour explicitly. A
# chart whose name contains "colorchecker" and that has no reference raises,
# because a silent D65 fallback would report a white-point error as a sensor
# colour error.

# %%
demo_layout = ChartLayout(
    name="flat_patch",
    patches=(PatchROI(name="grey", x=4, y=4, width=24, height=24),),
)
demo_image = np.full((32, 32, 3), 0.5, dtype=np.float32)
demo_noise = 0.01 * np.random.default_rng(7).standard_normal(demo_image.shape)
demo_image += demo_noise.astype(np.float32)
demo_report = measure_all(
    RenderedImage(data=demo_image, color_space="sRGB", channels="RGB"),
    demo_layout,
)
print(f"demo colour group: {demo_report.color}")
print(f"demo SNR channels: {list(demo_report.snr['grey'])}")
print(f"demo skipped: {demo_report.metadata['skipped']}")

# %% [markdown]
# ## 6. Summary
#
# The quick payload is green, but it is **not** gate evidence: it uses smaller
# MTF, SNR, chart, and acutance grids. The committed
# `results/reports/g4_metrics_results.json` and `results/reports/g4_metrics.md`
# come from the full `examples/03_measure_iq.py` run, and
# `tests/test_example_03.py` compares every report table cell with that JSON.

# %%
print(f"quick status: {payload['status']}")
print(f"quick gate evidence: {payload['gate_evidence']}")
passing = sum(1 for check in payload["checks"] if check["pass"])
print(f"checks passing: {passing} / {len(payload['checks'])}")
