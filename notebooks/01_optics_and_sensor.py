"""Notebook 01: illuminants, lens geometry, PSFs, MTF, and the sensor plane.

This notebook mirrors ``examples/01_spectral_scene_to_raw.py`` with narrative.
It follows the implemented Phase 1 chain from a spectral scene to sensor-plane
spectral irradiance:

* An illuminant is a relative spectral power distribution :math:`S(\\lambda)`
  on the common 360-830 nm, 5 nm grid. It is not lux; a target illuminance
  scales it photometrically through the luminous efficiency function.
* ColorChecker reflectance :math:`\\rho(\\lambda)` is a material property in
  ``[0, 1]``, not a colour. Reflectance times illuminant, integrated against the
  CIE observer, is what produces tristimulus values.
* A lens maps field angle to sensor position. The diagonal field of view is
  :math:`2\\arctan(d / 2f)` radians, converted to degrees by :math:`180/\\pi`.
  A pixel of pitch :math:`p` subtends :math:`\\arctan(p / f)` radians, so
  angular sampling is :math:`\\pi / (180 \\arctan(p / f))` pixels per degree
  (99.7 px/degree for the reference lens at 1.12 um pitch).
* A PSF is the normalized irradiance from a point source. Gaussian,
  diffraction-limited Airy, and square box kernels cover the useful range;
  profiles are normalized to peak one because unit-sum kernels would all look
  equally flat.
* The MTF is the magnitude of the normalized discrete Fourier transform of the
  PSF. It is plotted only to Nyquist (0.5 cycles/pixel), where higher
  frequencies would alias, and MTF50/MTF20 crossings are marked only when they
  exist by that limit.
* Scene rendering multiplies reflectance by illuminant and the lens formula
  :math:`T / (4N^2)`, convolves the PSF on an oversampled grid, applies the
  lens vignetting map (uniform for the reference lens, whose ``vignetting`` is
  ``None``), and box-averages to sensor pixels.

Real-world parameter ranges used here: phone lenses f/1.4-f/8, focal lengths
1.5-50 mm, pixel pitch 0.6-2 um, and scene illuminance 1-100,000 lux. The
reference lens is f/1.8, 6.4 mm, 11.46 mm diagonal, T = 0.9, and 1.12 um pitch.

The sensor and RAW stages (QE, CFA, shot/read noise, ADC) are not implemented
yet. The final section is an explicit placeholder; no RAW frame or tone-mapped
image is fabricated. Figures honour ``CAMSIM_FIG_DIR`` (default
``results/figures``) and use deterministic ``01_*.png`` names.

Failure artefacts guarded here: a PSF sampled on the sensor grid but applied to
the oversampled scene under-blurs by the oversampling factor; plotting MTF
beyond Nyquist aliases high frequencies into the passband; per-panel display
normalization makes brightness comparisons between panels invalid; and a stale
figure directory silently sends outputs somewhere other than the gallery.
"""

# %% [markdown]
# # 01 - Optics and the sensor-plane scene
#
# This notebook walks from an illuminant spectrum to the irradiance a sensor
# would receive. It does not model the sensor itself: everything below stays in
# linear radiometric units, and the last section marks where the sensor/RAW
# stage will attach.
#
# Run it from the repository root (the `make notebooks` target does this), or
# set `CAMSIM_FIG_DIR` to redirect the PNG output.

# %%
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from matplotlib.ticker import NullFormatter
from numpy.typing import NDArray

from camsim.config.schema import LensConfig
from camsim.optics.psf import airy_psf, box_psf, gaussian_psf, psf_mtf50
from camsim.scene.illuminants import illuminant, photopic_luminous_efficiency
from camsim.scene.spectral_data import load_colorchecker
from camsim.scene.targets import SceneResult, render
from camsim.viz import (
    PALETTE,
    figure_dir,
    plot_mtf,
    plot_psf,
    plot_spectrum,
    save_figure,
    set_style,
)

# When executed by Jupyter, switch to the inline backend so each figure is
# embedded in the notebook. Outside Jupyter the Agg backend above stays active.
_ipython = get_ipython()
if _ipython is not None:
    _ipython.run_line_magic("matplotlib", "inline")


def find_repo_root() -> Path:
    """Return the nearest ancestor containing pyproject.toml."""
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


REPO_ROOT = find_repo_root()
set_style()
FIGURES = figure_dir(REPO_ROOT / "results" / "figures")
print(f"figure directory: {FIGURES}")

PIXEL_PITCH_UM = 1.12
REFERENCE_F_NUMBER = 1.8
REFERENCE_FOCAL_LENGTH_MM = 6.4
REFERENCE_DIAGONAL_MM = 11.46
REFERENCE_TRANSMITTANCE = 0.9
SCENE_LUX = 200.0
OVERSAMPLING = 2
COLORCHECKER_SHAPE = (96, 128)
EDGE_SHAPE = (96, 96)
AIRY_F_NUMBER = 4.0
AIRY_WAVELENGTH_NM = 550.0


def reference_lens(f_number: float = REFERENCE_F_NUMBER) -> LensConfig:
    """Return the reference phone lens, optionally stopped down."""
    return LensConfig(
        f_number=f_number,
        focal_length_mm=REFERENCE_FOCAL_LENGTH_MM,
        sensor_diagonal_mm=REFERENCE_DIAGONAL_MM,
        transmittance=REFERENCE_TRANSMITTANCE,
    )


def luminance(scene: SceneResult) -> NDArray[np.float64]:
    """Integrate sensor-plane irradiance against the photopic curve."""
    wavelengths = scene.irradiance.wavelengths_nm.nm
    weights = photopic_luminous_efficiency(wavelengths)
    return np.asarray(
        np.trapezoid(
            np.asarray(scene.irradiance.data, dtype=np.float64) * weights,
            wavelengths,
            axis=-1,
        ),
        dtype=np.float64,
    )


def display_normalize(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply a display-only linear stretch to [0, 1]; no ISP or tone curve."""
    peak = float(np.max(image)) if image.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(image)
    return np.clip(image / peak, 0.0, 1.0)


# %% [markdown]
# ## 1. Illuminant spectra and scene reflectance
#
# D65 is the CIE daylight illuminant (about 6504 K). The 3000 K and 4000 K
# curves are Planckian radiators. All three are relative SPDs normalized to 100
# at 560 nm, so their shapes can be compared directly; the absolute radiometric
# scale is applied later by `lux_to_spectral_irradiance`. Tungsten's rising red
# tail and daylight's blue-rich peak are the physical reason white balance
# exists - not a sensor defect.
#
# Reflectance is the second half of the spectral scene. A patch's spectrum is a
# material property in [0, 1]; it is not a colour until it multiplies an
# illuminant and is integrated against the CIE observer. The neutral patches
# are nearly flat, while the red, green, and blue patches rise and fall in
# different wavelength bands.

# %%
illuminants = [illuminant(name) for name in ("D65", "3000K", "4000K")]
figure, _axes = plot_spectrum(
    illuminants,
    title="Illuminant spectral power distributions (relative, normalized at 560 nm)",
    ylabel="Relative spectral power",
)
save_figure(figure, FIGURES / "01_illuminant_spds.png")
plt.show()

selected_patches = (
    "white 9.5 (.05 D)",
    "neutral 5 (.70 D)",
    "black 2 (1.5 D)",
    "red",
    "green",
    "blue",
)
chart = load_colorchecker()
indices = [chart.patches.index(name) for name in selected_patches]
figure, _axes = plot_spectrum(
    chart.spectra[indices],
    chart.wavelengths_nm.nm,
    labels=selected_patches,
    title="ColorChecker reflectance factors (selected patches)",
    ylabel="Reflectance factor",
)
save_figure(figure, FIGURES / "01_colorchecker_reflectance.png")
plt.show()

# %% [markdown]
# ## 2. Lens geometry
#
# A 6.4 mm lens on an 11.46 mm diagonal sees about 84 degrees diagonally. That
# wide field comes at a cost: the same pixel pitch subtends a smaller angle, so
# angular sampling is only about 100 pixels/degree. A telephoto lens raises
# pixels/degree but narrows the field. MTF in cycles/degree is the
# cycles/pixel curve multiplied by this scale.

# %%
focal_lengths = np.geomspace(1.5, 50.0, 120)
fields_of_view: list[float] = []
scales: list[float] = []
for focal_length in focal_lengths:
    lens = LensConfig(
        f_number=REFERENCE_F_NUMBER,
        focal_length_mm=float(focal_length),
        sensor_diagonal_mm=REFERENCE_DIAGONAL_MM,
        transmittance=REFERENCE_TRANSMITTANCE,
    )
    fields_of_view.append(lens.field_of_view_deg)
    scales.append(lens.pixels_per_degree(PIXEL_PITCH_UM))

reference = reference_lens()
reference_scale = reference.pixels_per_degree(PIXEL_PITCH_UM)

figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
axes[0].plot(focal_lengths, fields_of_view, color=PALETTE[0])
axes[0].plot(
    [REFERENCE_FOCAL_LENGTH_MM],
    [reference.field_of_view_deg],
    marker="o",
    linestyle="none",
    color=PALETTE[1],
    label=f"reference {REFERENCE_FOCAL_LENGTH_MM:g} mm",
)
axes[0].set_xscale("log")
axes[0].set_xlabel("Focal length (mm)")
axes[0].set_ylabel("Diagonal field of view (degrees)")
axes[0].set_title("Field of view")
axes[0].legend(loc="best")

axes[1].plot(focal_lengths, scales, color=PALETTE[0])
axes[1].plot(
    [REFERENCE_FOCAL_LENGTH_MM],
    [reference_scale],
    marker="o",
    linestyle="none",
    color=PALETTE[1],
    label=f"reference {reference_scale:.0f} px/deg",
)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Focal length (mm)")
axes[1].set_ylabel("Angular sampling (pixels/degree)")
axes[1].set_title("Angular sampling at 1.12 um pitch")
axes[1].legend(loc="best")

focal_ticks = (2.0, 3.0, 5.0, 10.0, 20.0, 50.0)
for axis in axes:
    axis.set_xticks(focal_ticks)
    axis.set_xticklabels([f"{value:g}" for value in focal_ticks])
    axis.xaxis.set_minor_formatter(NullFormatter())

figure.suptitle("Reference lens geometry: field of view versus angular sampling")
save_figure(figure, FIGURES / "01_lens_geometry.png")
plt.show()

# %% [markdown]
# ## 3. Point-spread functions
#
# A Gaussian with sigma = 1 pixel models aberration and defocus. The Airy
# pattern is diffraction from a circular pupil; at f/4 and 550 nm its first dark
# ring lies about 2.4 pixels from the centre at this pitch. The box kernel is a
# deliberate 1.5-pixel box blur, not the physical 1-pixel aperture of this
# sensor: `render` already integrates that footprint. At 1.5 pixels the box's
# sinc-shaped transfer crosses half contrast inside the Nyquist band, which
# makes it a useful deliberately-soft comparison.
#
# The radial profile averages every sample at each rounded radius. Profiles are
# normalized to peak one: the kernels themselves sum to one, so plotting raw
# values would make every curve look equally flat.

# %%
kernels = {
    "Gaussian (sigma = 1.0 px)": gaussian_psf(1.0),
    "Airy (f/4, 550 nm)": airy_psf(AIRY_F_NUMBER, AIRY_WAVELENGTH_NM, PIXEL_PITCH_UM),
    "Box blur (1.5 px)": box_psf(1.5),
}

figure, _axes = plot_psf(
    kernels,
    mode="radial",
    max_radius_px=5.0,
    title="Point-spread functions at the sensor grid (each normalized to peak one)",
)
save_figure(figure, FIGURES / "01_psf_profiles.png")
plt.show()

# %% [markdown]
# ## 4. Modulation transfer
#
# The MTF is the magnitude of the normalized Fourier transform of the PSF. A
# Gaussian with sigma = 1 pixel has the analytic MTF50
# `sqrt(ln 2 / (2 pi^2)) = 0.187` cycles/pixel, a useful check on the curve.
# Stop bands are not shown: frequencies above 0.5 cycles/pixel alias into the
# passband and would make the plot lie about resolution.
#
# The top axis converts cycles/pixel to cycles/degree using the reference
# lens's 99.7 pixels/degree scale.

# %%
figure, _axes = plot_mtf(
    kernels,
    pixels_per_degree=reference_scale,
    title="Modulation transfer of the sensor-grid PSFs",
)
save_figure(figure, FIGURES / "01_mtf_curves.png")
plt.show()

print("MTF50 in cycles/pixel and cycles/degree:")
for name, kernel in kernels.items():
    try:
        frequency = psf_mtf50(kernel)
    except ValueError:
        print(f"  {name}: no finite MTF50 (MTF stays above 0.5 through Nyquist)")
        continue
    print(
        f"  {name}: {frequency:.3f} cycles/pixel ({frequency * reference_scale:.1f} cycles/degree)"
    )

# %% [markdown]
# ## 5. A spectral scene at the sensor plane
#
# `render` rasterizes the target on a 2x oversampled grid, multiplies by the
# D65 illuminant (photometrically normalized to 200 lux) and by
# `T / (4 N^2)`, applies the PSF on that same oversampled grid, applies the
# lens vignetting map (uniform here because the reference lens sets
# `vignetting=None`), and box-averages down to sensor pixels.
#
# The panels show spectrally weighted luminance with a per-panel display
# stretch. This is visualization only: no RAW data, white balance, demosaicing,
# colour correction, or tone curve has been applied. Per-panel normalization
# also means relative brightness between panels is not meaningful.


# %%
def render_luminance(
    target: str,
    *,
    lens: LensConfig,
    optics: NDArray[np.float64] | None,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Render one target and return its photopically weighted luminance."""
    result = render(
        target,
        illuminant="D65",
        lux=SCENE_LUX,
        lens=lens,
        optics=optics,
        pixel_pitch_um=PIXEL_PITCH_UM,
        shape=shape,
        oversampling=OVERSAMPLING,
        seed=0,
    )
    return luminance(result)


colorchecker = render_luminance(
    "colorchecker",
    lens=reference,
    optics=gaussian_psf(1.0 * OVERSAMPLING),
    shape=COLORCHECKER_SHAPE,
)
edge_sharp = render_luminance(
    "slanted_edge",
    lens=reference,
    optics=None,
    shape=EDGE_SHAPE,
)
edge_gaussian = render_luminance(
    "slanted_edge",
    lens=reference,
    optics=gaussian_psf(1.5 * OVERSAMPLING),
    shape=EDGE_SHAPE,
)
edge_airy = render_luminance(
    "slanted_edge",
    lens=reference_lens(AIRY_F_NUMBER),
    optics=airy_psf(
        AIRY_F_NUMBER,
        AIRY_WAVELENGTH_NM,
        PIXEL_PITCH_UM / OVERSAMPLING,
    ),
    shape=EDGE_SHAPE,
)

panels = (
    ("ColorChecker\nD65, Gaussian 1.0 px", colorchecker),
    ("Slanted edge\npixel aperture only", edge_sharp),
    ("Slanted edge\nGaussian 1.5 px", edge_gaussian),
    ("Slanted edge\nAiry f/4, 550 nm", edge_airy),
)
figure, axes = plt.subplots(1, len(panels), figsize=(12.0, 4.0))
figure.subplots_adjust(top=0.76, bottom=0.02, left=0.01, right=0.99, wspace=0.08)
for axis, (title, image) in zip(axes, panels, strict=True):
    axis.imshow(
        display_normalize(image),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.set_title(title, fontsize=9, pad=6)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#444444")
        spine.set_linewidth(0.6)
figure.suptitle("Sensor-plane luminance, display-normalized per panel (no RAW, ISP, or tone curve)")
save_figure(figure, FIGURES / "01_scene_contact_sheet.png")
plt.show()

# %% [markdown]
# ## 6. Where the sensor and RAW stages attach
#
# The cube returned by `render` is a `SpectralIrradiance` in `W/m^2/nm` on the
# sensor grid, before the sensor's IR-cut filter and CFA. The missing stages are
# physical, not cosmetic:
#
# - integrate each pixel's spectral irradiance against QE and CFA transmission
#   to obtain electrons per second; aperture and lens transmittance are already
#   included in the scene irradiance;
# - multiply by exposure time to count electrons, then add shot noise
#   (Poisson), read noise, and dark current;
# - clip at full well, apply conversion gain, digitize with the ADC, and add the
#   black-level pedestal.
#
# Those stages are deliberately absent from this phase. No RAW frame is
# produced or simulated here; the placeholder below is the exact boundary a
# later phase will fill.

# %%
print("Sensor/RAW stage: not implemented in this phase.")
print("This notebook ends at sensor-plane spectral irradiance; no RAW data are faked.")
print(
    f"Reference lens: {reference.field_of_view_deg:.1f} deg diagonal, "
    f"{reference_scale:.1f} pixels/degree at {PIXEL_PITCH_UM:g} um pitch."
)
print(f"Rendered ColorChecker luminance shape: {colorchecker.shape}")
print(f"Rendered slanted-edge luminance shape: {edge_gaussian.shape}")
