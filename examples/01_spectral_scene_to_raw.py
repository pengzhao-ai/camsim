"""Example 01: from a spectral scene through a lens to the sensor plane.

This walkthrough follows light through the implemented Phase 1 chain:

1. **Illuminant spectra** - D65, 3000 K, and 4000 K on the common 360-830 nm,
   5 nm grid. An SPD is relative power per nanometre, not lux; the absolute
   scale is set later by a target illuminance.
2. **Scene reflectance** - selected ColorChecker patches. Reflectance is a
   material property, not a colour; only reflectance multiplied by an
   illuminant and integrated against the CIE observer becomes tristimulus.
3. **Lens geometry** - focal length trades diagonal field of view against
   angular sampling in pixels/degree. The reference phone lens is 6.4 mm at
   f/1.8 with an 11.46 mm sensor diagonal and a 1.12 um pixel pitch.
4. **Point-spread functions** - a Gaussian aberration/defocus model, a
   diffraction-limited Airy pattern, and a deliberately coarse 1.5 px square
   box blur (not the physical 1 px pixel aperture, which the renderer already
   integrates). Each kernel is normalized to unit energy; the figure normalizes
   profiles to peak one to compare shape.
5. **Modulation transfer** - the MTF of those kernels in cycles/pixel, with
   MTF50 and MTF20 crossings marked and a cycles/degree axis derived from the
   lens geometry.
6. **Scene render** - a ColorChecker and a slanted edge are rendered to
   sensor-plane spectral irradiance. The scene renderer multiplies reflectance
   by photometrically normalized illuminance and the lens formula
   ``T / (4 N**2)``, convolves the PSF on the oversampled grid, applies the
   lens vignetting map (uniform for the reference lens, whose ``vignetting`` is
   ``None``), and box-averages to sensor pixels.

The filename is aspirational for the full project. The sensor and RAW stages
(QE, CFA, shot/read noise, ADC) do not exist yet, so this example stops at
spectral irradiance and prints a clearly marked placeholder for the section a
later phase will add. It never fabricates a RAW frame or a tone-mapped image.

The contact sheet uses a display-only linear stretch of spectrally weighted
luminance; that normalization is a plotting convenience, not an ISP stage or a
calibrated photometric quantity. The spectrum, PSF, MTF, and lens figures plot
their physical quantities directly (relative SPD, reflectance factor,
peak-normalized PSF profile, and modulation transfer). Output paths honour
``CAMSIM_FIG_DIR`` (default ``results/figures``) and use deterministic
``01_*.png`` names.

Failure artefacts guarded here: a sensor-grid PSF applied directly to the
oversampled scene under-blurs by the oversampling factor; the Airy kernel is
sampled at the oversampled pitch for the same reason. Per-panel display
normalization hides relative exposure differences by design, so the contact
sheet must not be read as a radiometric comparison. Plotting MTF beyond Nyquist
would alias high frequencies into the passband, and a missing or relative
figure directory would scatter outputs instead of regenerating one gallery.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

PIXEL_PITCH_UM: Final[float] = 1.12
REFERENCE_F_NUMBER: Final[float] = 1.8
REFERENCE_FOCAL_LENGTH_MM: Final[float] = 6.4
REFERENCE_DIAGONAL_MM: Final[float] = 11.46
REFERENCE_TRANSMITTANCE: Final[float] = 0.9

SCENE_LUX: Final[float] = 200.0
OVERSAMPLING: Final[int] = 2
COLORCHECKER_SHAPE: Final[tuple[int, int]] = (96, 128)
EDGE_SHAPE: Final[tuple[int, int]] = (96, 96)

GAUSSIAN_SIGMA_PX: Final[float] = 1.0
EDGE_GAUSSIAN_SIGMA_PX: Final[float] = 1.5
AIRY_F_NUMBER: Final[float] = 4.0
AIRY_WAVELENGTH_NM: Final[float] = 550.0
BOX_BLUR_PX: Final[float] = 1.5

ILLUMINANT_NAMES: Final[tuple[str, ...]] = ("D65", "3000K", "4000K")
REFLECTANCE_PATCHES: Final[tuple[str, ...]] = (
    "white 9.5 (.05 D)",
    "neutral 5 (.70 D)",
    "black 2 (1.5 D)",
    "red",
    "green",
    "blue",
)


def _reference_lens(f_number: float = REFERENCE_F_NUMBER) -> LensConfig:
    """Return the reference phone lens, optionally stopped down."""
    return LensConfig(
        f_number=f_number,
        focal_length_mm=REFERENCE_FOCAL_LENGTH_MM,
        sensor_diagonal_mm=REFERENCE_DIAGONAL_MM,
        transmittance=REFERENCE_TRANSMITTANCE,
    )


def _luminance(scene: SceneResult) -> NDArray[np.float64]:
    """Integrate sensor-plane irradiance against the photopic curve.

    This is a visualization proxy for what a human observer would weigh most
    strongly, not a calibrated luminance: the cube is a per-nanometre
    irradiance density and the integration is a plain trapezoidal sum. The
    absolute scale is discarded by the display normalization below.
    """
    wavelengths = scene.irradiance.wavelengths_nm.nm
    weights = photopic_luminous_efficiency(wavelengths)
    luminance = np.trapezoid(
        np.asarray(scene.irradiance.data, dtype=np.float64) * weights,
        wavelengths,
        axis=-1,
    )
    return np.asarray(luminance, dtype=np.float64)


def _display_normalize(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stretch a luminance image to ``[0, 1]`` for grayscale display.

    This linear stretch only makes the scene structure visible; it is not
    exposure correction, gamma encoding, or an ISP tone curve. Each panel is
    normalized independently, so relative brightness between panels is not
    meaningful.
    """
    peak = float(np.max(image)) if image.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(image)
    return np.clip(image / peak, 0.0, 1.0)


def _save_illuminant_figure(directory: Path) -> Path:
    """Plot the three supported illuminant SPDs and save the figure."""
    spectra = [illuminant(name) for name in ILLUMINANT_NAMES]
    figure, _axes = plot_spectrum(
        spectra,
        title="Illuminant spectral power distributions (relative, normalized at 560 nm)",
        ylabel="Relative spectral power",
    )
    destination = save_figure(figure, directory / "01_illuminant_spds.png")
    plt.close(figure)
    return destination


def _save_reflectance_figure(directory: Path) -> Path:
    """Plot selected ColorChecker reflectance spectra.

    Reflectance is a material property in ``[0, 1]``, not a colour: the curve
    shapes only become the familiar patch colours after multiplication by an
    illuminant and integration against the CIE observer. Neutral patches are
    nearly flat; the red, green, and blue patches rise and fall in different
    bands, which is why a single measured reflectance selects different
    tristimulus values under D65 and tungsten.
    """
    chart = load_colorchecker()
    indices = [chart.patches.index(name) for name in REFLECTANCE_PATCHES]
    figure, _axes = plot_spectrum(
        chart.spectra[indices],
        chart.wavelengths_nm.nm,
        labels=REFLECTANCE_PATCHES,
        title="ColorChecker reflectance factors (selected patches)",
        ylabel="Reflectance factor",
    )
    destination = save_figure(figure, directory / "01_colorchecker_reflectance.png")
    plt.close(figure)
    return destination


def _save_lens_figure(directory: Path) -> tuple[Path, LensConfig, float]:
    """Plot field of view and angular sampling versus focal length."""
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

    reference = _reference_lens()
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

    # A logarithmic axis otherwise labels several minor ticks per decade and
    # the focal-length numbers collide; fixed major ticks keep the axis legible.
    focal_ticks = (2.0, 3.0, 5.0, 10.0, 20.0, 50.0)
    for axis in axes:
        axis.set_xticks(focal_ticks)
        axis.set_xticklabels([f"{value:g}" for value in focal_ticks])
        axis.xaxis.set_minor_formatter(NullFormatter())

    figure.suptitle("Reference lens geometry: field of view versus angular sampling")
    destination = save_figure(figure, directory / "01_lens_geometry.png")
    plt.close(figure)
    return destination, reference, reference_scale


def _sensor_psfs() -> dict[str, NDArray[np.float64]]:
    """Return the Gaussian, Airy, and box kernels used throughout the figures.

    The Airy kernel is sampled at f/4 rather than the reference lens's f/1.8.
    At f/1.8 the first dark ring lies at about 1.08 px, so the Airy core is
    undersampled on the sensor grid and the discrete MTF stays above 0.5
    through Nyquist; MTF50 is undefined for that sampled kernel. Stopping down
    to f/4 moves the first zero to about 2.4 px, resolves the rings, and
    produces a measurable MTF50 while the continuous diffraction cutoff remains
    just above Nyquist. The box is a deliberate 1.5 px blur, not the physical
    1 px pixel aperture; the wider box makes its sinc-shaped MTF cross 0.5
    inside the Nyquist band.
    """
    return {
        "Gaussian (sigma = 1.0 px)": gaussian_psf(GAUSSIAN_SIGMA_PX),
        "Airy (f/4, 550 nm)": airy_psf(
            AIRY_F_NUMBER,
            AIRY_WAVELENGTH_NM,
            PIXEL_PITCH_UM,
        ),
        "Box blur (1.5 px)": box_psf(BOX_BLUR_PX),
    }


def _save_psf_figure(directory: Path, kernels: dict[str, NDArray[np.float64]]) -> Path:
    """Plot normalized radial profiles through each PSF kernel."""
    figure, _axes = plot_psf(
        kernels,
        mode="radial",
        max_radius_px=5.0,
        title="Point-spread functions at the sensor grid (each normalized to peak one)",
    )
    destination = save_figure(figure, directory / "01_psf_profiles.png")
    plt.close(figure)
    return destination


def _save_mtf_figure(
    directory: Path,
    kernels: dict[str, NDArray[np.float64]],
    pixels_per_degree: float,
) -> Path:
    """Plot MTF curves with MTF50/MTF20 markers and a cycles/degree axis."""
    figure, _axes = plot_mtf(
        kernels,
        pixels_per_degree=pixels_per_degree,
        title="Modulation transfer of the sensor-grid PSFs",
    )
    destination = save_figure(figure, directory / "01_mtf_curves.png")
    plt.close(figure)
    return destination


def _render_scene(
    target: str,
    *,
    lens: LensConfig,
    optics: NDArray[np.float64] | None,
    shape: tuple[int, int],
    illuminant_name: str = "D65",
) -> NDArray[np.float64]:
    """Render one target and return its display luminance image."""
    result = render(
        target,
        illuminant=illuminant_name,
        lux=SCENE_LUX,
        lens=lens,
        optics=optics,
        pixel_pitch_um=PIXEL_PITCH_UM,
        shape=shape,
        oversampling=OVERSAMPLING,
        seed=0,
    )
    return _luminance(result)


def _save_scene_figure(directory: Path) -> Path:
    """Render four scene/optics combinations and save a contact sheet."""
    reference = _reference_lens()
    stopped_down = _reference_lens(AIRY_F_NUMBER)
    airy_kernel = airy_psf(
        AIRY_F_NUMBER,
        AIRY_WAVELENGTH_NM,
        PIXEL_PITCH_UM / OVERSAMPLING,
    )

    colorchecker = _render_scene(
        "colorchecker",
        lens=reference,
        optics=gaussian_psf(GAUSSIAN_SIGMA_PX * OVERSAMPLING),
        shape=COLORCHECKER_SHAPE,
    )
    edge_sharp = _render_scene(
        "slanted_edge",
        lens=reference,
        optics=None,
        shape=EDGE_SHAPE,
    )
    edge_gaussian = _render_scene(
        "slanted_edge",
        lens=reference,
        optics=gaussian_psf(EDGE_GAUSSIAN_SIGMA_PX * OVERSAMPLING),
        shape=EDGE_SHAPE,
    )
    edge_airy = _render_scene(
        "slanted_edge",
        lens=stopped_down,
        optics=airy_kernel,
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
            _display_normalize(image),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=9, pad=6)
        axis.set_xticks([])
        axis.set_yticks([])
        # A white half-image is otherwise indistinguishable from the page
        # background; the frame makes each panel's extent explicit.
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#444444")
            spine.set_linewidth(0.6)
    figure.suptitle(
        "Sensor-plane luminance, display-normalized per panel (no RAW, ISP, or tone curve)"
    )
    destination = save_figure(figure, directory / "01_scene_contact_sheet.png")
    plt.close(figure)
    return destination


def main() -> int:
    """Run the example, write the Phase 1 figure gallery, and return exit code 0."""
    set_style()
    figures = figure_dir(REPO_ROOT / "results" / "figures")
    print("camsim example 01: spectral scene -> optics -> sensor plane")
    print(f"figure directory: {figures}")

    written: list[Path] = []
    written.append(_save_illuminant_figure(figures))
    written.append(_save_reflectance_figure(figures))
    lens_path, reference, pixels_per_degree = _save_lens_figure(figures)
    written.append(lens_path)

    kernels = _sensor_psfs()
    written.append(_save_psf_figure(figures, kernels))
    written.append(_save_mtf_figure(figures, kernels, pixels_per_degree))
    written.append(_save_scene_figure(figures))

    print("\nWritten figures:")
    for path in written:
        print(f"  {path.name} ({path.stat().st_size / 1024:.0f} KiB)")

    print("\nReference lens:")
    print(
        f"  f/{REFERENCE_F_NUMBER:g}, {REFERENCE_FOCAL_LENGTH_MM:g} mm, "
        f"{REFERENCE_DIAGONAL_MM:g} mm diagonal, T = {REFERENCE_TRANSMITTANCE:g}"
    )
    print(f"  diagonal field of view: {reference.field_of_view_deg:.1f} deg")
    print(f"  angular sampling at {PIXEL_PITCH_UM:g} um pitch: {pixels_per_degree:.1f} px/deg")

    print("\nPSF MTF50:")
    for name, kernel in kernels.items():
        try:
            frequency = psf_mtf50(kernel)
        except ValueError:
            print(f"  {name}: no finite MTF50 (MTF stays above 0.5 through Nyquist)")
        else:
            print(
                f"  {name}: {frequency:.3f} cycles/pixel "
                f"({frequency * pixels_per_degree:.1f} cycles/degree)"
            )

    print(
        "\nRendered scenes: ColorChecker D65 and slanted edge D65, "
        f"{SCENE_LUX:g} lux, {OVERSAMPLING}x oversampling, seed 0"
    )

    # ------------------------------------------------------------------ #
    # Sensor and RAW stage: intentionally not implemented in this phase.
    # A later task will convert the SceneResult above into electrons and RAW
    # codes (QE, CFA, shot/read/dark noise, full well, ADC, black level).
    # No sensor is constructed here and no RAW array is produced.
    # ------------------------------------------------------------------ #
    print(
        "\n[sensor/RAW stage - placeholder]\n"
        "  The sensor model is not implemented yet, so this example stops at\n"
        "  sensor-plane spectral irradiance. A later phase will add the\n"
        "  irradiance -> electrons -> RAW conversion; no RAW data are faked here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
