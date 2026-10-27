"""Tests for scene targets, chart layouts, vignetting, and scene rendering.

A scene target is a spatial map of spectral reflectance on a sampling grid. The
slanted edge, flat patch, Siemens star, and 24-patch ColorChecker are the
standard image-quality charts: the edge measures modulation transfer, the flat
patch measures noise and shading, the star exposes resolution and aliasing, and
the ColorChecker anchors colour. Reflectance is a dimensionless material
property in ``[0, 1]``; it becomes sensor-plane irradiance only after
multiplication by the scene illuminance and the lens formula

``E_img(lambda) = rho(lambda) * E_scene(lambda) * T(lambda) / (4 N**2) * cos**4(theta)``.

The renderer samples the target on an ``oversampling x oversampling`` grid,
applies the PSF and vignetting there, and box-averages down to the sensor grid.
That ordering matters: rendering the sharp target directly at sensor resolution
aliases high frequencies into the passband, and applying a sensor-grid PSF to an
oversampled cube would use the wrong blur width. The PSF argument is therefore
documented as sampled on the oversampled grid: a Gaussian of ``s`` sensor pixels
must be passed as ``gaussian_psf(s * oversampling)``.

Vignetting follows the physical ``cos**4`` natural-illumination falloff, where
``theta`` is the field angle and ``tan(theta) = r / f`` for a sensor radius
``r``. A reference phone lens at f/1.8 and a 5.7 mm half-diagonal loses roughly
70 % of its corner illumination; wide apertures and short focal lengths lose
more. ``LensConfig.vignetting`` blends the physical curve towards no falloff,
which models lenses whose shading has been calibrated out.

Failure artefacts guarded here: reflectance outside ``[0, 1]`` is not a passive
material; chart ROIs that overlap or fall outside the sensor grid silently
report the wrong patch; rendering at the sensor grid instead of oversampling
aliases the edge; using a PSF sampled on the sensor grid blurs by the
oversampling factor too little; a missing ``1 / (4 N**2)`` factor makes absolute
electron counts meaningless; and using a global RNG would make repeated renders
non-reproducible.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

from camsim.config.schema import LensConfig
from camsim.core.color import ColorReference
from camsim.core.spectral import SpectralIrradiance
from camsim.optics.psf import gaussian_psf
from camsim.optics.vignetting import vignetting_map
from camsim.scene.illuminants import (
    illuminant,
    lux_to_spectral_irradiance,
    photopic_luminous_efficiency,
)
from camsim.scene.spectral_data import COMMON_WAVELENGTH_GRID, load_colorchecker
from camsim.scene.targets import (
    PatchROI,
    SceneLayout,
    SceneResult,
    colorchecker_target,
    estimate_edge_angle,
    flat_patch_target,
    render,
    siemens_star_target,
    slanted_edge_target,
)


def _lens(**updates: object) -> LensConfig:
    """Return the reference phone lens with optional field overrides."""
    lens = LensConfig(
        f_number=1.8,
        focal_length_mm=6.4,
        sensor_diagonal_mm=11.46,
        transmittance=0.9,
    )
    return lens.model_copy(update=updates) if updates else lens


def _render_flat(**overrides: object) -> SceneResult:
    """Render the flat patch with test-sized defaults."""
    options: dict[str, object] = {
        "illuminant": "D65",
        "lux": 100.0,
        "lens": _lens(),
        "optics": None,
        "pixel_pitch_um": 1.12,
        "shape": (24, 32),
        "oversampling": 2,
        "seed": None,
    }
    options.update(overrides)
    return render("flat_patch", **options)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Targets and chart layouts
# --------------------------------------------------------------------------- #


def test_colorchecker_layout_has_24_patches() -> None:
    cube, layout = colorchecker_target((160, 240))

    assert cube.shape == (160, 240, len(COMMON_WAVELENGTH_GRID.nm))
    assert layout.name == "colorchecker"
    assert len(layout.patches) == 24
    assert tuple(patch.name for patch in layout.patches) == load_colorchecker().patches

    for patch in layout.patches:
        assert patch.width > 0 and patch.height > 0
        assert patch.x >= 0 and patch.x + patch.width <= 240
        assert patch.y >= 0 and patch.y + patch.height <= 160

    for first, second in combinations(layout.patches, 2):
        assert not (
            first.x < second.x + second.width
            and second.x < first.x + first.width
            and first.y < second.y + second.height
            and second.y < first.y + first.height
        )


def test_colorchecker_patches_use_their_spectra() -> None:
    cube, layout = colorchecker_target((160, 240))
    chart = load_colorchecker()
    first = layout.patches[0]
    center = cube[first.y + first.height // 2, first.x + first.width // 2]

    np.testing.assert_allclose(center, chart.spectra[0], atol=1e-12)
    # The margin between patches is a dark neutral surround, not a patch colour.
    assert cube[0, 0, 0] == pytest.approx(0.05)


def test_colorchecker_target_rejects_a_shape_too_small_for_24_patches() -> None:
    with pytest.raises(ValueError, match="too small"):
        colorchecker_target((2, 2))


@pytest.mark.parametrize("shape", [(10, 14), (11, 17), (12, 19), (16, 24)])
def test_colorchecker_layout_stays_valid_on_compact_shapes(
    shape: tuple[int, int],
) -> None:
    _cube, layout = colorchecker_target(shape)

    assert len(layout.patches) == 24
    for patch in layout.patches:
        assert patch.x >= 0 and patch.x + patch.width <= shape[1]
        assert patch.y >= 0 and patch.y + patch.height <= shape[0]
    for first, second in combinations(layout.patches, 2):
        assert not (
            first.x < second.x + second.width
            and second.x < first.x + first.width
            and first.y < second.y + second.height
            and second.y < first.y + first.height
        )


def test_flat_patch_target_is_uniform() -> None:
    target = flat_patch_target((8, 12))

    assert target.shape == (8, 12)
    np.testing.assert_allclose(target, 0.5)
    np.testing.assert_allclose(flat_patch_target((8, 12), reflectance=0.2), 0.2)


@pytest.mark.parametrize("reflectance", [-0.1, 1.1, math.nan])
def test_flat_patch_target_rejects_invalid_reflectance(reflectance: float) -> None:
    with pytest.raises(ValueError, match="reflectance"):
        flat_patch_target((8, 8), reflectance=reflectance)


def test_slanted_edge_target_is_binary_and_tilted() -> None:
    target = slanted_edge_target((256, 256), angle_deg=5.0)

    assert target.shape == (256, 256)
    assert set(np.unique(target)) <= {0.0, 1.0}
    assert target.mean() == pytest.approx(0.5, abs=0.01)
    # A positive angle moves the edge rightwards as rows increase, so the upper
    # rows contain more bright pixels than the lower rows.
    assert target[10].sum() > target[-11].sum()


@pytest.mark.parametrize("angle_deg", [-10.0, -5.0, 5.0, 10.0])
def test_estimate_edge_angle_recovers_the_tilt(angle_deg: float) -> None:
    target = slanted_edge_target((256, 256), angle_deg=angle_deg)

    assert estimate_edge_angle(target) == pytest.approx(angle_deg, abs=0.5)


def test_estimate_edge_angle_rejects_an_image_without_an_edge() -> None:
    with pytest.raises(ValueError, match="gradient"):
        estimate_edge_angle(np.zeros((64, 64)))


def test_siemens_star_target_has_alternating_spokes() -> None:
    target = siemens_star_target((129, 129), spokes=4)

    assert target.shape == (129, 129)
    assert set(np.unique(target)) <= {0.0, 1.0}
    assert target.mean() == pytest.approx(0.5, abs=0.01)
    # With four spokes the horizontal pair is bright and the vertical pair dark.
    assert target[64, 84] == 1.0
    assert target[84, 64] == 0.0


@pytest.mark.parametrize("spokes", [0, -4, 3.5])
def test_siemens_star_rejects_invalid_spokes(spokes: object) -> None:
    with pytest.raises(ValueError, match="spokes"):
        siemens_star_target((64, 64), spokes=spokes)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Vignetting
# --------------------------------------------------------------------------- #


def test_vignetting_cos4_falloff() -> None:
    lens = _lens(vignetting=1.0)
    shape = (101, 101)
    pitch_um = 40.0

    falloff = vignetting_map(shape, lens, pitch_um)

    assert falloff.shape == shape
    assert falloff[50, 50] == pytest.approx(1.0)
    corner_radius_mm = math.hypot(50.0, 50.0) * pitch_um * 1e-3
    expected_corner = 1.0 / (1.0 + (corner_radius_mm / lens.focal_length_mm) ** 2) ** 2
    assert falloff[0, 0] == pytest.approx(expected_corner, rel=1e-12)
    assert falloff[0, 0] == pytest.approx(0.6999, abs=1e-3)
    assert falloff[50, 50] > falloff[50, 90] > falloff[0, 90]


def test_vignetting_strength_blends_between_none_and_full_cos4() -> None:
    shape = (65, 65)
    pitch_um = 40.0

    np.testing.assert_allclose(vignetting_map(shape, _lens(), pitch_um), 1.0)
    full = vignetting_map(shape, _lens(vignetting=1.0), pitch_um)
    half = vignetting_map(shape, _lens(vignetting=0.5), pitch_um)
    np.testing.assert_allclose(half, 1.0 - 0.5 * (1.0 - full))


@pytest.mark.parametrize(
    ("shape", "pitch_um", "message"),
    [
        ((0, 10), 1.12, "shape"),
        ((10, 10, 10), 1.12, "shape"),
        ((10, 10), 0.0, "pixel_pitch_um"),
        ((10, 10), math.inf, "pixel_pitch_um"),
    ],
)
def test_vignetting_map_rejects_invalid_inputs(
    shape: tuple[int, ...], pitch_um: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        vignetting_map(shape, _lens(vignetting=1.0), pitch_um)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_render_returns_spectral_irradiance() -> None:
    result = _render_flat(shape=(24, 32))

    assert isinstance(result.irradiance, SpectralIrradiance)
    assert isinstance(result.layout, SceneLayout)
    assert isinstance(result.reference, ColorReference)
    assert result.irradiance.data.shape == (24, 32, len(COMMON_WAVELENGTH_GRID.nm))
    np.testing.assert_array_equal(result.irradiance.wavelengths_nm.nm, COMMON_WAVELENGTH_GRID.nm)
    assert result.irradiance.unit == "W/m^2/nm"
    assert result.irradiance.pixel_pitch_um == pytest.approx(1.12)
    assert np.all(np.isfinite(result.irradiance.data))
    assert np.all(result.irradiance.data >= 0.0)
    assert result.reference.illuminant_name == "D65"
    assert result.layout.name == "flat_patch"
    assert len(result.layout.patches) == 1
    assert result.layout.patches[0].name == "flat"


def test_render_is_seed_deterministic() -> None:
    first = _render_flat(seed=7)
    second = _render_flat(seed=7)

    np.testing.assert_array_equal(first.irradiance.data, second.irradiance.data)
    assert first.layout == second.layout
    assert first.reference == second.reference
    assert first.ground_truth == second.ground_truth


@pytest.mark.parametrize(
    ("target", "patch_count", "has_edge_roi"),
    [
        ("colorchecker", 24, False),
        ("flat_patch", 1, False),
        ("slanted_edge", 0, True),
        ("siemens_star", 0, False),
    ],
)
def test_render_supports_every_target(target: str, patch_count: int, has_edge_roi: bool) -> None:
    result = render(
        target,
        lens=_lens(),
        shape=(40, 48),
        oversampling=1,
        optics=None,
    )

    assert result.irradiance.data.shape == (40, 48, len(COMMON_WAVELENGTH_GRID.nm))
    assert result.layout.name == target
    assert len(result.layout.patches) == patch_count
    assert (result.layout.edge_roi is not None) is has_edge_roi


def test_render_colorchecker_layout_stays_on_the_sensor_grid() -> None:
    result = render(
        "colorchecker",
        lens=_lens(),
        shape=(80, 120),
        oversampling=2,
        optics=None,
    )

    assert len(result.layout.patches) == 24
    for patch in result.layout.patches:
        assert patch.x >= 0 and patch.x + patch.width <= 120
        assert patch.y >= 0 and patch.y + patch.height <= 80


def test_render_colorchecker_patch_radiance_follows_patch_spectrum() -> None:
    # Oversampling is two so the test also proves that sensor-grid ROIs line up
    # with patches painted on the finer rendering grid.
    result = render(
        "colorchecker",
        lens=_lens(),
        lux=100.0,
        shape=(80, 120),
        oversampling=2,
        optics=None,
    )
    chart = load_colorchecker()
    scene = lux_to_spectral_irradiance(100.0, illuminant("D65")).data[0, 0]
    expected = chart.spectra[0] * scene * 0.9 / (4.0 * 1.8**2)
    first = result.layout.patches[0]
    measured = result.irradiance.data[first.y + first.height // 2, first.x + first.width // 2]

    np.testing.assert_allclose(measured, expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("shape", "oversampling"),
    [
        ((10, 14), 2),
        ((10, 14), 3),
        ((10, 14), 4),
        ((11, 17), 2),
        ((11, 17), 3),
        ((11, 17), 4),
        ((12, 19), 2),
        ((12, 19), 3),
        ((12, 19), 4),
    ],
)
def test_render_colorchecker_compact_shapes_have_clean_rois(
    shape: tuple[int, int], oversampling: int
) -> None:
    # Regression: the layout used to be rounded independently on the sensor
    # grid while patches were painted on the oversampled grid. At compact
    # shapes the two roundings drifted, so ROI-centre pixels landed on the dark
    # surround. Every one of the 24 ROI centres must carry its own spectrum.
    result = render(
        "colorchecker",
        lens=_lens(),
        lux=100.0,
        shape=shape,
        oversampling=oversampling,
        optics=None,
    )
    chart = load_colorchecker()
    scene = lux_to_spectral_irradiance(100.0, illuminant("D65")).data[0, 0]
    scale = scene * 0.9 / (4.0 * 1.8**2)

    assert len(result.layout.patches) == 24
    for index, patch in enumerate(result.layout.patches):
        centre = result.irradiance.data[patch.y + patch.height // 2, patch.x + patch.width // 2]
        np.testing.assert_allclose(centre, chart.spectra[index] * scale, rtol=1e-12)


def test_render_rejects_a_colorchecker_shape_without_a_clean_sensor_pixel() -> None:
    # At four samples per pixel this shape leaves no sensor pixel whose whole
    # 4x4 rendering block sits inside a patch, so no uncontaminated ROI exists.
    with pytest.raises(ValueError, match="fully covered sensor pixel"):
        render("colorchecker", lens=_lens(), shape=(4, 6), oversampling=4, optics=None)


def test_render_scales_with_inverse_f_number_squared() -> None:
    wide = _render_flat(lens=_lens(f_number=1.8))
    narrow = _render_flat(lens=_lens(f_number=3.6))

    ratio = wide.irradiance.data.mean() / narrow.irradiance.data.mean()
    assert ratio == pytest.approx(4.0, rel=1e-12)


def test_render_scales_linearly_with_lens_transmittance() -> None:
    clear = _render_flat(lens=_lens(transmittance=0.9))
    coated = _render_flat(lens=_lens(transmittance=0.45))

    ratio = clear.irradiance.data.mean() / coated.irradiance.data.mean()
    assert ratio == pytest.approx(2.0, rel=1e-12)


def test_render_scales_linearly_with_lux() -> None:
    bright = _render_flat(lux=100.0)
    dim = _render_flat(lux=50.0)

    ratio = bright.irradiance.data.mean() / dim.irradiance.data.mean()
    assert ratio == pytest.approx(2.0, rel=1e-12)


def test_render_absolute_photometric_scale() -> None:
    # At f/0.5 the lens formula's 1 / (4 N**2) factor is exactly one; with unit
    # transmittance and a 0.5-reflectance target, 100 lux in must produce the
    # photometric equivalent of 50 lux at the sensor plane.
    result = render(
        "flat_patch",
        lens=_lens(f_number=0.5, transmittance=1.0),
        lux=100.0,
        shape=(16, 16),
        oversampling=1,
        optics=None,
    )
    wavelengths_nm = result.irradiance.wavelengths_nm.nm
    spectrum = result.irradiance.data[8, 8]
    luminous = 683.0 * float(
        np.trapezoid(spectrum * photopic_luminous_efficiency(wavelengths_nm), wavelengths_nm)
    )

    assert luminous == pytest.approx(50.0, rel=1e-9)


def test_render_psf_conserves_energy_and_softens_an_edge() -> None:
    plain = _render_flat(shape=(32, 32), oversampling=2)
    blurred = _render_flat(shape=(32, 32), oversampling=2, optics=gaussian_psf(2.0, size=13))
    assert blurred.irradiance.data.sum() == pytest.approx(plain.irradiance.data.sum(), rel=1e-2)

    sharp_edge = render("slanted_edge", lens=_lens(), shape=(64, 64), oversampling=2, optics=None)
    soft_edge = render(
        "slanted_edge",
        lens=_lens(),
        shape=(64, 64),
        oversampling=2,
        optics=gaussian_psf(2.0, size=13),
    )
    sharp_gradient = np.abs(np.diff(sharp_edge.irradiance.data.mean(axis=2), axis=1)).max()
    soft_gradient = np.abs(np.diff(soft_edge.irradiance.data.mean(axis=2), axis=1)).max()
    assert soft_gradient < 0.75 * sharp_gradient


def test_render_applies_cos4_vignetting() -> None:
    lens = _lens(vignetting=1.0)
    result = render(
        "flat_patch",
        lens=lens,
        shape=(65, 65),
        pixel_pitch_um=80.0,
        oversampling=1,
        optics=None,
    )
    image = result.irradiance.data.mean(axis=2)
    falloff = vignetting_map((65, 65), lens, 80.0)

    np.testing.assert_allclose(image / image[32, 32], falloff, rtol=1e-12, atol=1e-15)
    assert image[0, 0] < 0.7 * image[32, 32]


@pytest.mark.parametrize("oversampling", [2, 4])
def test_render_vignetting_uses_the_oversampled_pixel_pitch(oversampling: int) -> None:
    # The vignetting map is sampled on the rendering grid, so render must pass
    # pixel_pitch_um / oversampling. Passing the sensor pitch would make the
    # falloff nearly flat at N = 2-4 and fail this comparison by many per cent.
    lens = _lens(vignetting=1.0)
    shape = (33, 33)
    pitch_um = 80.0
    result = render(
        "flat_patch",
        lens=lens,
        shape=shape,
        pixel_pitch_um=pitch_um,
        oversampling=oversampling,
        optics=None,
    )
    image = result.irradiance.data.mean(axis=2)
    falloff = vignetting_map(shape, lens, pitch_um)

    # The oversampled map is box-averaged into each sensor pixel, which differs
    # from the point-sampled map by O((pitch / focal_length)**2) ~ 1e-5.
    np.testing.assert_allclose(image / image[16, 16], falloff, rtol=1e-4, atol=1e-7)
    assert image[0, 0] < 0.9 * image[16, 16]


def _edge_width_10_90(profile: np.ndarray) -> float:
    """Return the 10-90 % ascending-edge width of a monotonic 1-D profile."""
    low = float(profile.min())
    high = float(profile.max())
    normalized = (profile - low) / (high - low)
    columns = np.arange(profile.size, dtype=np.float64)
    return float(np.interp(0.9, normalized, columns) - np.interp(0.1, normalized, columns))


def test_render_psf_width_matches_the_oversampled_sampling_convention() -> None:
    # gaussian_psf(2.0) passed to a 2x render is a one-sensor-pixel Gaussian.
    # A Gaussian edge has a 10-90 % width of 2 * 1.2816 * sigma = 2.563 sensor
    # pixels; pixel-aperture averaging and the 5 degree tilt add a few per cent.
    # Passing a sensor-grid gaussian_psf(1.0) instead would measure ~1.9 px.
    shape = (128, 128)
    sigma_sensor_px = 1.0
    oversampling = 2
    result = render(
        "slanted_edge",
        lens=_lens(),
        shape=shape,
        oversampling=oversampling,
        optics=gaussian_psf(sigma_sensor_px * oversampling, size=41),
    )
    profile = result.irradiance.data[64].mean(axis=1)
    width = _edge_width_10_90(profile)

    assert width == pytest.approx(2.563, rel=0.15)

    wider = render(
        "slanted_edge",
        lens=_lens(),
        shape=shape,
        oversampling=oversampling,
        optics=gaussian_psf(2.0 * sigma_sensor_px * oversampling, size=41),
    )
    wider_profile = wider.irradiance.data[64].mean(axis=1)
    assert _edge_width_10_90(wider_profile) == pytest.approx(2.0 * width, rel=0.08)


def test_render_records_scene_parameters_as_ground_truth() -> None:
    result = _render_flat(seed=3, lux=250.0)

    ground_truth = result.ground_truth
    assert ground_truth["target"] == "flat_patch"
    assert ground_truth["illuminant"] == "D65"
    assert ground_truth["lux"] == 250.0
    assert ground_truth["shape"] == (24, 32)
    assert ground_truth["oversampling"] == 2
    assert ground_truth["seed"] == 3
    assert ground_truth["psf_applied"] is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target": "unknown"}, "target"),
        ({"oversampling": 0}, "oversampling"),
        ({"shape": (0, 10)}, "shape"),
        ({"lux": -1.0}, "lux"),
        ({"illuminant": "D50"}, "illuminant"),
        ({"optics": np.ones(3)}, "PSF"),
    ],
)
def test_render_rejects_invalid_arguments(overrides: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "illuminant": "D65",
        "lux": 100.0,
        "lens": _lens(),
        "optics": None,
        "pixel_pitch_um": 1.12,
        "shape": (16, 16),
        "oversampling": 1,
    }
    arguments.update(overrides)
    target = str(arguments.pop("target", "flat_patch"))
    with pytest.raises(ValueError, match=message):
        render(target, **arguments)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Layout value validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "arguments",
    [
        ("", 0, 0, 2, 2),
        ("patch", -1, 0, 2, 2),
        ("patch", 0, 0, 0, 2),
        ("patch", 0, 0, 2, -2),
    ],
)
def test_patch_roi_rejects_invalid_geometry(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        PatchROI(*arguments)  # type: ignore[arg-type]


def test_scene_layout_rejects_duplicate_patch_names() -> None:
    patches = (PatchROI("p", 0, 0, 2, 2), PatchROI("p", 4, 4, 2, 2))

    with pytest.raises(ValueError, match="unique"):
        SceneLayout("chart", patches)


# --------------------------------------------------------------------------- #
# Simulation memory budget and scene-derived metrics layout
# --------------------------------------------------------------------------- #


def test_render_memory_guard_rejects_before_allocating() -> None:
    """A huge spectral request fails before any cube is allocated."""
    with pytest.raises(ValueError, match="peak memory"):
        render(
            "flat_patch",
            illuminant="D65",
            lux=100.0,
            shape=(100_000, 100_000),
            oversampling=1,
            max_memory_bytes=8 * 1024**3,
        )


def test_chart_layout_from_scene_matches_the_render() -> None:
    """The layout helper derives geometry from the scene, not caller guesses."""
    from camsim.scene.charts import chart_layout_from_scene

    scene = render(
        "colorchecker",
        illuminant="3000K",
        lux=100.0,
        shape=(64, 64),
        oversampling=2,
    )
    layout = chart_layout_from_scene(scene, lens=_lens())
    assert layout.name == "colorchecker"
    assert len(layout.patches) == 24
    assert layout.patches[0].name == scene.layout.patches[0].name
    assert layout.pixel_pitch_um == scene.irradiance.pixel_pitch_um
    assert layout.reference is not None
    assert layout.reference.illuminant_name == "3000K"
    assert layout.focal_length_mm == pytest.approx(6.4)
