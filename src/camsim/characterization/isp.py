"""G3 ISP validation: spectral scene, sensor, and ISP measured end to end.

The G3 gate asks a different question from G1 and G2. G1 proved that a scene
carries the right photometry and optics; G2 proved that a sensor converts
irradiance into electrons and RAW codes with the configured physics. G3 asks
whether the reconstruction chain

    RAW -> DPC -> BLC -> LSC -> AWB -> demosaic -> CCM -> denoise -> sharpen
        -> tone -> output

turns that RAW data into a display-referred image that preserves colour,
linearity, and spatial detail. This module is the shared measurement used by
``examples/02_isp_stage_gallery.py`` and by ``tests/test_example_02.py``; the
report and the tests therefore cannot silently measure two different things.

**Colour path.** A ColorChecker patch has a spectral reflectance
``rho(lambda)``. Under an illuminant ``S(lambda)`` its camera channels are the
sensor projection integrals

    c_k = integral rho(lambda) S(lambda) P_k(lambda) / E_photon(lambda) dlambda

with ``P_k = QE * T_IR * T_CFA,k``, while the colorimetric reference is

    XYZ = integral rho(lambda) S(lambda) [xbar, ybar, zbar] dlambda
          / integral S(lambda) ybar(lambda) dlambda.

The reference Lab in :class:`~camsim.scene.targets.SceneResult` is constructed
against the scene illuminant's own white point. To fit a camera-to-sRGB matrix
we convert it back to XYZ with that white point, apply a Bradford chromatic
adaptation to D65, and convert XYZ to *linear* sRGB (``colour.XYZ_to_RGB``
without a transfer function). The camera channels are white-balanced with the
analytic spectral response to a perfect diffuser, not with the chart's white
patch: a row-sum matrix preserves the neutral axis, so the white reference must
be the perfect diffuser that defines the Lab scale. The fitted matrix has
``target ~= camera @ M.T`` and rows summing to one.

The matrix is fitted and evaluated on the same 24 patches (in-sample
calibration-scene validation). That is deliberate and is stated as a limitation:
a held-out chart or a second illuminant would measure generalisation, while G3
measures whether the ISP can represent the calibration scene at all. The
residual is still limited by metamerism: a 3x3 matrix can only be exact when
the sensor's spectral sensitivities are a linear combination of the CIE colour
matching functions (the Luther condition).

**White balance.** The gray-world estimator on a uniform, spectrally neutral
flat field estimates the per-channel response ratios. The expected gains are
computed independently from the analytic spectral response to a perfect
diffuser, so the test does not define its expectation with the estimator it is
testing. A second check injects a known per-CFA channel scaling into a neutral
frame and requires the estimator to invert it.

**Demosaic and tone.** Bilinear and Malvar-He-Cutler reconstruction are
compared against ``colour-demosaicing`` on a synthetic Siemens-star RGB mosaic.
The sRGB OETF is compared against ``colour.cctf_encoding``. The stage trace is
checked by applying the tone configuration to the last linear trace stage and
comparing both the display array and the pre-clamp linear buffer (including
exposure compensation and the sRGB-to-Display-P3 matrix).

**Lens shading, denoise, and sharpen.** LSC is measured by calibrating a gain
map from one noisy flat field and applying it to an independent flat field.
Denoise is measured as the SNR gain on a noisy flat region plus the relative
change of a step-edge contrast with the bilateral filter. Sharpening is
measured as the contrast gain and the positive overshoot of a thresholded
unsharp mask on a step edge. Formal ISO 12233-style MTF50/MTF20 measurement
lives in the independent G4 metrics harness and the G5 cross-cutting evidence;
the G3 report points to those checks rather than duplicating them.

**RCCC.** An RCCC mosaic is reconstructed by the documented luma-placeholder
baseline. It is a shape/linearity/smoke test only; the Bayer ΔE and MTF
thresholds do not apply.

**Real-world ranges.** The validation scene uses 200-500 lux, a 10-bit sensor
with a 64 DN pedestal, exposures inside the preset's 0.1 ms-100 ms range,
illuminants D65 and 3000 K, and the shipped reference lens. The ColorChecker
reflectance is a material property in ``[0, 1]``; a percent-valued patch or a
display-encoded target is a units error. White-balance gains of 0.5-4,
exposures of 60-80 % of the ADC white span, and 24 in-sample patches are the
representative operating point.

**Determinism.** Every sensor exposure uses an explicit per-call seed and every
synthetic noise field uses an explicit :class:`numpy.random.Generator`; no
global random state is read or written. Repeated calls in one software
environment produce identical dictionaries within float round-off.

**Failure artefacts guarded here.** A chart rendered without an exposure
calibration (clipped highlights); a camera RGB matrix paired with an absolute
target at the wrong scale (a row-sum CCM that cannot reach the target white); a
target computed against the wrong illuminant white point (a systematic colour
cast); an AWB expectation copied from the estimator under test (a tautological
check); a demosaic comparison on a grey image (a trivially exact result); a
tone check that compares the pipeline with itself instead of the published
standard; a stale report that no longer matches the JSON; and a quick-mode run
mistaken for gate evidence.
"""

from __future__ import annotations

import math
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import colour
import numpy as np
from colour.difference import delta_E_CIE2000
from numpy.typing import NDArray

from camsim._version import __version__ as camsim_version
from camsim.config.io import PRESETS_DIR, load_isp, load_sensor
from camsim.config.schema import (
    DenoiseConfig,
    ISPConfig,
    LSCConfig,
    SensorConfig,
    SharpenConfig,
)
from camsim.core.color import ColorReference
from camsim.core.frames import CFAPattern, RawFrame, RenderedImage, StageOutput
from camsim.core.units import photon_energy_j
from camsim.data import data_path
from camsim.isp.ccm import fit_ccm
from camsim.isp.colorspace import linear_srgb_to_p3
from camsim.isp.demosaic import demosaic_bilinear, demosaic_mhc
from camsim.isp.denoise import denoise
from camsim.isp.lsc import apply_lsc, calibrate_lsc
from camsim.isp.pipeline import ISP
from camsim.isp.sharpen import unsharp_mask
from camsim.isp.tone import apply_tone, linear_to_srgb, srgb_to_linear
from camsim.scene.illuminants import illuminant, lux_to_spectral_irradiance
from camsim.scene.targets import PatchROI, SceneResult, render, siemens_star_target
from camsim.sensor import Sensor
from camsim.sensor.cfa import cfa_color_at, mosaic_from_channels, spectral_projection

__all__ = ["G3Gallery", "build_gallery", "measure_g3"]

G3_KIND: Final[str] = "camsim G3 ISP validation"
SCHEMA_VERSION: Final[str] = "1.0"

# D65 chromaticity used for the colorimetric reference and for interpreting the
# rendered sRGB image. The value is the CIE 1931 2-degree D65 white point.
D65_XY: Final[NDArray[np.float64]] = np.array([0.3127, 0.3290], dtype=np.float64)

# Scene and acquisition constants. 300 lux is a bright indoor scene: it keeps
# the calibrated exposure inside the preset's 0.1 ms-100 ms range and leaves
# enough electrons that read noise does not dominate the patch means.
SCENE_LUX: Final[float] = 300.0
OVERSAMPLING: Final[int] = 2
PROBE_EXPOSURE_S: Final[float] = 1e-3
TARGET_WHITE_FRACTION: Final[float] = 0.70
COLORCHECKER_WHITE_PATCH: Final[str] = "white 9.5 (.05 D)"

FULL_COLORCHECKER_SHAPE: Final[tuple[int, int]] = (96, 128)
QUICK_COLORCHECKER_SHAPE: Final[tuple[int, int]] = (64, 96)
FULL_STAR_SHAPE: Final[tuple[int, int]] = (96, 96)
QUICK_STAR_SHAPE: Final[tuple[int, int]] = (64, 64)
FULL_NOISE_SHAPE: Final[int] = 96
QUICK_NOISE_SHAPE: Final[int] = 64
RCCC_SHAPE: Final[tuple[int, int]] = (64, 64)

CCM_RIDGE: Final[float] = 1e-4
AWB_TOLERANCE: Final[float] = 0.05
DELTA_E_D65_MAX: Final[float] = 2.0
DELTA_E_WARM_MAX: Final[float] = 4.0
DEMOSAIC_BILINEAR_MAX: Final[float] = 1e-5
DEMOSAIC_MHC_MAX: Final[float] = 1e-4
TONE_MAX_ABS: Final[float] = 1e-7
TRACE_MAX_ABS: Final[float] = 1e-6
LINEARITY_MAX_REL: Final[float] = 1e-5
LINEARITY_AWB_MAX_REL: Final[float] = 1e-3
LSC_RESIDUAL_MAX: Final[float] = 0.02
DENOISE_SNR_GAIN_MIN: Final[float] = 3.0
DENOISE_EDGE_MAX: Final[float] = 0.10

# Fixed seeds. Each exposure gets its own seed so the frame data are
# independent of call order; the synthetic noise fields use separate generators.
_PROBE_SEED: Final[int] = 100
_SCENE_SEED: Final[int] = 101
_FLAT_SEED: Final[int] = 111
_LSC_SEED: Final[int] = 61
_DENOISE_SEED: Final[int] = 62

_SENSOR_PRESET: Final[Path] = PRESETS_DIR / "sensors" / "flagship_phone_50mp.json"
_ISP_PRESET: Final[Path] = PRESETS_DIR / "isp" / "default.json"


@dataclass(frozen=True)
class G3Gallery:
    """Arrays used by the G3 stage-gallery figures.

    Parameters
    ----------
    stages:
        Ordered ISP trace from a ColorChecker capture with denoise and
        sharpening enabled, one :class:`~camsim.core.frames.StageOutput` per
        executed stage plus the final ``"output"`` entry.
    star_reference:
        Synthetic Siemens-star RGB reference image in linear ``[0, 1]``.
    star_bilinear:
        Bilinear reconstruction of the star mosaic.
    star_mhc:
        Malvar-He-Cutler reconstruction of the star mosaic.
    rccc_image:
        Rendered RCCC baseline image, shape ``(H, W, 3)``.

    Notes
    -----
    The dataclass is frozen, but the arrays and the stage dictionary are not
    deep-frozen; treat them as read-only snapshots.
    """

    stages: dict[str, StageOutput]
    star_reference: NDArray[np.float32]
    star_bilinear: NDArray[np.float32]
    star_mhc: NDArray[np.float32]
    rccc_image: NDArray[np.float32]


def measure_g3(*, quick: bool = False) -> dict[str, Any]:
    """Run the G3 ISP measurements and return a JSON-serialisable report.

    Parameters
    ----------
    quick:
        Use smaller ColorChecker, star, and noise fields. Quick mode exercises
        every code path but is **not** gate evidence; the returned dictionary
        records ``gate_evidence=False``. The full run uses the committed gate
        scene sizes and is the source of the committed report.

    Returns
    -------
    A dictionary with ``kind``, ``gate_evidence``, ``checks`` (each carrying
    ``name``, ``measured``, ``tolerance``, ``pass``, ``notes``, and the exact
    display strings used by the Markdown report), and the raw measured values
    under ``colorchecker``, ``awb``, ``demosaic``, ``tone``, ``trace``,
    ``linearity``, ``lsc``, ``denoise``, ``sharpen``, and ``rccc``. Every value
    is a Python float, int, bool, string, list, or nested list, so
    ``json.dumps(..., allow_nan=False)`` succeeds.

    The ColorChecker criterion is reported honestly. The shipped flagship
    preset uses synthetic **conventional Bayer bands** (blue ~448 nm, green
    ~542 nm, red ~610 nm) with moderate out-of-band leakage. The in-sample 3x3
    CCM meets the design thresholds under both illuminants; the
    Luther-condition residual is reported as a diagnostic of how well the CFA
    spans the CIE observer. The original broad, nearly collinear CFA curves were
    replaced after the G3 gate showed that no 3x3 matrix could colour-correct
    them. The CCM is still fitted and evaluated in-sample, which is a stated
    limitation.
    """
    sensor_config, isp_config = _load_configs()
    colorchecker_shape = QUICK_COLORCHECKER_SHAPE if quick else FULL_COLORCHECKER_SHAPE
    star_shape = QUICK_STAR_SHAPE if quick else FULL_STAR_SHAPE
    noise_shape = QUICK_NOISE_SHAPE if quick else FULL_NOISE_SHAPE

    projection = _spectral_projection(sensor_config)
    colorchecker, colorchecker_checks, runs = _measure_colorchecker(
        sensor_config,
        isp_config,
        projection,
        shape=colorchecker_shape,
    )
    awb, awb_checks = _measure_awb(sensor_config, isp_config, projection, runs)
    demosaic, demosaic_checks = _measure_demosaic(star_shape)
    tone, tone_checks = _measure_tone()
    trace, trace_checks = _measure_trace(isp_config, runs["D65"])
    linearity, linearity_checks = _measure_linearity(noise_shape)
    lsc, lsc_checks = _measure_lsc(noise_shape)
    denoise, denoise_checks = _measure_denoise(noise_shape)
    sharpen, sharpen_checks = _measure_sharpen(noise_shape)
    rccc, rccc_checks = _measure_rccc(isp_config)

    checks = [
        *colorchecker_checks,
        *awb_checks,
        *demosaic_checks,
        *tone_checks,
        *trace_checks,
        *linearity_checks,
        *lsc_checks,
        *denoise_checks,
        *sharpen_checks,
        *rccc_checks,
    ]
    status = "PASS" if all(bool(check["pass"]) for check in checks) else "FAIL"

    d65_run = runs["D65"]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": G3_KIND,
        "gate_evidence": not quick,
        "quick": quick,
        "status": status,
        "environment": _environment(),
        "seeds": {
            "probe": _PROBE_SEED,
            "scene": _SCENE_SEED,
            "flat": _FLAT_SEED,
            "lsc": _LSC_SEED,
            "denoise": _DENOISE_SEED,
            "note": (
                "Each illuminant reuses the scene and flat seeds with its own "
                "spectrum; LSC and denoise use explicit numpy Generator seeds; "
                "sharpen and the star/RCCC targets are deterministic without RNG."
            ),
        },
        "presets": {
            "sensor": _relative_preset(_SENSOR_PRESET),
            "isp": _relative_preset(_ISP_PRESET),
            "sensor_name": sensor_config.name,
            "isp_name": isp_config.name,
            "sensor_config_hash": d65_run["sensor_config_hash"],
            "isp_config_hash": ISP(isp_config).config_hash,
        },
        "scene": {
            "lux": SCENE_LUX,
            "oversampling": OVERSAMPLING,
            "colorchecker_shape": list(colorchecker_shape),
            "white_patch": COLORCHECKER_WHITE_PATCH,
            "target_white_fraction": TARGET_WHITE_FRACTION,
            "exposure_ceiling_s": float(sensor_config.exposure.max_s),
            "exposures_s": {name: run["exposure_s"] for name, run in runs.items()},
            "white_fractions": {name: run["white_fraction"] for name, run in runs.items()},
        },
        "colorchecker": {"illuminants": colorchecker},
        "sensor_colorimetry": _luther_diagnostic(projection, _sensor_wavelengths(sensor_config)),
        "awb": awb,
        "demosaic": demosaic,
        "tone": tone,
        "trace": trace,
        "linearity": linearity,
        "lsc": lsc,
        "denoise": denoise,
        "sharpen": sharpen,
        "rccc": rccc,
        "checks": checks,
    }
    return payload


def build_gallery(*, quick: bool = False) -> G3Gallery:
    """Render the stage-gallery and demosaic-comparison arrays for figures.

    The gallery is built from a separate ColorChecker capture with the fitted
    per-illuminant CCM and with denoise and sharpening enabled, so the contact
    sheet shows every mandatory stage. The star arrays are the same synthetic
    mosaic used by the demosaic check. The function is deterministic and does
    not write files; the example owns the figure names and formats.
    """
    sensor_config, isp_config = _load_configs()
    shape = QUICK_COLORCHECKER_SHAPE if quick else FULL_COLORCHECKER_SHAPE
    star_shape = QUICK_STAR_SHAPE if quick else FULL_STAR_SHAPE
    projection = _spectral_projection(sensor_config)
    run = _colorchecker_run(
        sensor_config,
        isp_config,
        projection,
        illuminant_name="D65",
        shape=shape,
        trace=True,
        denoise=DenoiseConfig(
            method="gaussian",
            luma_strength=0.35,
            chroma_strength=0.6,
            radius=1.0,
        ),
        sharpen=SharpenConfig(amount=0.6, radius=1.0, threshold=0.0, luma_only=True),
    )
    stages = run["rendered"].stages
    if stages is None:  # pragma: no cover - trace=True always records stages
        raise RuntimeError("gallery capture did not produce an ISP trace")
    star = _star_case(star_shape)
    rccc = _rccc_case(isp_config)
    return G3Gallery(
        stages=stages,
        star_reference=star["reference"],
        star_bilinear=star["bilinear"],
        star_mhc=star["mhc"],
        rccc_image=rccc["image"],
    )


# ---------------------------------------------------------------------------
# Configuration, spectral response, and colour conversion
# ---------------------------------------------------------------------------


def _load_configs() -> tuple[SensorConfig, ISPConfig]:
    """Load the shipped flagship sensor and default ISP presets."""
    return load_sensor(_SENSOR_PRESET), load_isp(_ISP_PRESET)


def _relative_preset(path: Path) -> str:
    """Return a repository-relative preset path when possible."""
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_csv(path: str) -> Path:
    """Resolve a preset spectral CSV against the working tree or package data.

    The order matches the sensor's own resolution: an absolute or
    working-directory-relative file wins, then the repository root (for source
    checkouts), then the installed ``camsim.data`` package.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    fallback = Path(__file__).resolve().parents[3] / candidate
    if fallback.exists():
        return fallback
    try:
        return data_path(candidate)
    except FileNotFoundError:
        return candidate


def _sensor_wavelengths(config: SensorConfig) -> NDArray[np.float64]:
    """Return the sensor preset's uniform wavelength grid in nanometres."""
    spectral = config.spectral.wavelength_nm
    start = float(spectral.start)
    stop = float(spectral.stop)
    step = float(spectral.step)
    count = round((stop - start) / step)
    return np.asarray(start + step * np.arange(count + 1), dtype=np.float64)


def _spectral_projection(config: SensorConfig) -> NDArray[np.float64]:
    """Return the ``(3, L)`` QE x IR-cut x CFA projection for the sensor."""
    return spectral_projection(
        _resolve_csv(config.spectral.qe_csv),
        _resolve_csv(config.spectral.cfa_csv),
        _resolve_csv(config.spectral.ir_cut_csv),
        _sensor_wavelengths(config),
        config.cfa,
    )


def _analytic_white_response(
    config: SensorConfig,
    projection: NDArray[np.float64],
    illuminant_name: str,
    lux: float = SCENE_LUX,
) -> NDArray[np.float64]:
    """Return the per-channel electron-rate response to a perfect diffuser.

    A perfect diffuser has unit spectral reflectance, so its sensor-plane
    irradiance is the illuminant's spectral irradiance scaled by the lens
    terms. The lens terms, pixel area, and exposure are common to all channels
    and cancel in the gain ratios; photon energy does not cancel and is
    therefore included, exactly as :class:`camsim.Sensor` integrates it.
    """
    wavelengths = _sensor_wavelengths(config)
    irradiance = lux_to_spectral_irradiance(lux, illuminant(illuminant_name)).data[0, 0]
    photons = irradiance / photon_energy_j(wavelengths)
    response = np.asarray(photons @ projection.T, dtype=np.float64)
    crosstalk = np.asarray(config.spectral.spectral_crosstalk, dtype=np.float64)
    if crosstalk.shape == (3, 3) and not np.allclose(crosstalk, np.eye(3)):
        response = np.asarray(crosstalk @ response, dtype=np.float64)
    if response.size != 3 or not np.all(np.isfinite(response)) or np.any(response <= 0.0):
        raise ValueError("the analytic white response must be three positive finite channels")
    return response


def _analytic_gains(
    config: SensorConfig,
    projection: NDArray[np.float64],
    illuminant_name: str,
) -> tuple[float, float, float]:
    """Return green-normalised white-balance gains from the spectral response."""
    response = _analytic_white_response(config, projection, illuminant_name)
    return (
        float(response[1] / response[0]),
        1.0,
        float(response[1] / response[2]),
    )


def _target_rgb_from_reference(reference: ColorReference) -> NDArray[np.float64]:
    """Convert reference Lab to linear sRGB under D65.

    The reference Lab was computed against the scene illuminant's white point.
    ``Lab_to_XYZ`` reverses that with the same white, a Bradford chromatic
    adaptation maps the result to the CIE D65 white, and ``XYZ_to_RGB`` returns
    **linear** sRGB (``apply_cctf_encoding=False``), which is the domain in
    which the CCM is fitted and applied.
    """
    white_xy = np.asarray(reference.white_point_xy, dtype=np.float64)
    xyz = np.asarray(colour.Lab_to_XYZ(reference.lab, illuminant=white_xy), dtype=np.float64)
    adapted = np.asarray(
        colour.adaptation.chromatic_adaptation_VonKries(
            xyz,
            colour.xy_to_XYZ(white_xy),
            colour.xy_to_XYZ(D65_XY),
            transform="Bradford",
        ),
        dtype=np.float64,
    )
    rgb = np.asarray(
        colour.XYZ_to_RGB(
            adapted,
            colourspace="sRGB",
            illuminant=D65_XY,
            apply_cctf_encoding=False,
        ),
        dtype=np.float64,
    )
    if rgb.shape != (24, 3) or not np.all(np.isfinite(rgb)):
        raise ValueError("the ColorChecker target must be a finite (24, 3) linear sRGB array")
    return rgb


def _luther_diagnostic(
    projection: NDArray[np.float64],
    wavelengths: NDArray[np.float64],
) -> dict[str, Any]:
    """Summarise how well the sensor's spectral sensitivities span colourimetry.

    The Luther condition asks whether the camera sensitivities are a linear
    combination of the CIE colour matching functions. The best 3x3 fit and its
    relative spectral residual are a diagnostic for the ColorChecker ΔE: a
    large residual means no 3x3 CCM can reproduce the observer exactly.
    """
    from colour.colorimetry import MSDS_CMFS

    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    cmf = np.column_stack(
        [
            np.interp(wavelengths, np.asarray(cmfs.wavelengths), np.asarray(cmfs.values)[:, i])
            for i in range(3)
        ]
    )
    radiometric = projection / photon_energy_j(wavelengths)
    transform = np.linalg.lstsq(radiometric.T, cmf, rcond=None)[0].T
    residual = float(np.linalg.norm(radiometric.T @ transform.T - cmf) / np.linalg.norm(cmf))
    correlation = np.atleast_2d(np.corrcoef(projection))
    singular_values = np.asarray(np.linalg.svd(projection, compute_uv=False), dtype=np.float64)
    return {
        "luther_relative_residual": residual,
        "channel_correlation": [[float(value) for value in row] for row in correlation],
        "projection_singular_values": [float(value) for value in singular_values],
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "interpretation": (
            "A large Luther residual means the CFA sensitivities are not a linear "
            "combination of the CIE observer; a 3x3 CCM then cannot reproduce "
            "saturated patches exactly, no matter how it is fitted."
        ),
    }


# ---------------------------------------------------------------------------
# Scene rendering, exposure calibration, and patch extraction
# ---------------------------------------------------------------------------


def _find_patch(scene: SceneResult, name: str) -> PatchROI:
    """Return the named patch ROI from a rendered scene."""
    for patch in scene.layout.patches:
        if patch.name == name:
            return patch
    raise ValueError(f"scene {scene.layout.name!r} has no patch named {name!r}")


def _roi_array(frame: RawFrame, roi: PatchROI) -> tuple[NDArray[np.uint16], NDArray[np.bool_]]:
    """Return the ROI data and a mask excluding defects and PDAF sites."""
    data = frame.data[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    mask = np.ones(data.shape, dtype=np.bool_)
    if frame.defect_mask is not None:
        mask &= ~frame.defect_mask[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    if frame.pdaf_mask is not None:
        mask &= ~frame.pdaf_mask[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    return data, mask


def _channel_means(frame: RawFrame, roi: PatchROI) -> dict[str, float]:
    """Return black-level-corrected, white-span-normalised per-CFA-channel means.

    The ROI's top-left pixel may fall on any phase of the 2x2 tile; the phase
    is taken from the absolute sensor coordinate, not from the crop origin, so
    a red site is never averaged into a blue mean.
    """
    data, mask = _roi_array(frame, roi)
    black = float(frame.black_level_dn)
    span = float(2**frame.bit_depth - 1) - black
    if span <= 0.0:
        raise ValueError("the frame has no usable white span")
    values: dict[str, list[NDArray[np.float64]]] = {}
    for row_phase in (0, 1):
        for column_phase in (0, 1):
            colour = cfa_color_at(
                frame.cfa,
                (roi.y + row_phase) % 2,
                (roi.x + column_phase) % 2,
            )
            samples = data[row_phase::2, column_phase::2][mask[row_phase::2, column_phase::2]]
            if samples.size:
                values.setdefault(colour, []).append(samples.astype(np.float64) - black)
    means: dict[str, float] = {}
    for channel, parts in values.items():
        samples = np.concatenate(parts)
        mean = float(samples.mean()) / span
        if not math.isfinite(mean) or mean <= 0.0:
            raise ValueError(f"channel {channel} has a non-positive mean in patch {roi.name!r}")
        means[channel] = mean
    if not means:
        raise ValueError(f"patch {roi.name!r} contains no usable samples")
    return means


def _calibrate_exposure(
    sensor: Sensor,
    scene: SceneResult,
    *,
    patch_name: str,
    target_fraction: float,
    probe_s: float,
    seed: int,
) -> tuple[float, bool]:
    """Scale a probe exposure so the brightest white-patch channel hits a target.

    The sensor is linear below saturation, so one proportional step from a
    deliberately dim probe is enough. The result is clamped to the preset's
    exposure limits; the returned flag records whether the requested exposure
    was outside those limits, and the caller records the actually achieved
    white fraction so a clipped calibration cannot masquerade as the requested
    one.
    """
    patch = _find_patch(scene, patch_name)
    probe = sensor.expose(scene.irradiance, probe_s, mode="LCG", seed=seed)
    means = _channel_means(probe, patch)
    brightest = max(means.values())
    requested = probe_s * target_fraction / brightest
    minimum = float(sensor.config.exposure.min_s)
    maximum = float(sensor.config.exposure.max_s)
    clamped = requested > maximum + 1e-12 or requested < minimum - 1e-12
    return float(min(max(requested, minimum), maximum)), clamped


def _render_scene(
    target: str,
    illuminant_name: str,
    shape: tuple[int, int],
    *,
    seed: int,
) -> SceneResult:
    """Render one target at the G3 scene illuminance and sensor grid."""
    return render(
        target,
        illuminant=illuminant_name,
        lux=SCENE_LUX,
        shape=shape,
        oversampling=OVERSAMPLING,
        seed=seed,
    )


def _patch_means(image: NDArray[Any], patches: tuple[PatchROI, ...]) -> NDArray[np.float64]:
    """Return the mean colour of every patch ROI in a 2-D or 3-channel image."""
    values = np.asarray(image, dtype=np.float64)
    if values.ndim == 2:
        values = values[..., np.newaxis]
    if values.ndim != 3 or values.shape[-1] not in (1, 3):
        raise ValueError(f"image must have one or three channels; got shape {values.shape}")
    means = [
        values[patch.y : patch.y + patch.height, patch.x : patch.x + patch.width]
        .reshape(-1, values.shape[-1])
        .mean(axis=0)
        for patch in patches
    ]
    return np.asarray(means, dtype=np.float64)


def _display_lab(rendered: RenderedImage) -> NDArray[np.float64]:
    """Decode a rendered sRGB image to CIE Lab under D65.

    The display buffer carries the sRGB OETF; ``srgb_to_linear`` reverses it,
    the sRGB matrix gives XYZ D65, and ``XYZ_to_Lab`` uses the D65 white. The
    linear buffer is deliberately not used here: the design gate is stated on
    the rendered display, and clipping is part of what the display shows.
    """
    if rendered.color_space != "sRGB":
        raise ValueError(
            f"the ColorChecker ΔE gate requires an sRGB render; got {rendered.color_space!r}"
        )
    linear = np.asarray(srgb_to_linear(rendered.data), dtype=np.float64)
    xyz = np.asarray(
        colour.RGB_to_XYZ(
            linear,
            colourspace="sRGB",
            illuminant=D65_XY,
            apply_cctf_decoding=False,
        ),
        dtype=np.float64,
    )
    return np.asarray(colour.XYZ_to_Lab(xyz, illuminant=D65_XY), dtype=np.float64)


def _delta_e(
    rendered: RenderedImage,
    reference_lab: NDArray[np.float64],
    patches: tuple[PatchROI, ...],
) -> NDArray[np.float64]:
    """Return per-patch CIEDE2000 between the rendered display and the reference."""
    lab = _patch_means(_display_lab(rendered), patches)
    differences = np.asarray(delta_E_CIE2000(lab, reference_lab), dtype=np.float64)
    if differences.shape != (24,) or not np.all(np.isfinite(differences)):
        raise ValueError("the ColorChecker ΔE calculation did not produce 24 finite values")
    return differences


# ---------------------------------------------------------------------------
# ISP configuration helpers
# ---------------------------------------------------------------------------


def _matrix_as_list(matrix: NDArray[np.float64]) -> list[list[float]]:
    """Return a 3x3 matrix as nested Python floats for a pydantic config."""
    return [[float(value) for value in row] for row in matrix]


def _with_stages(
    base: ISPConfig,
    *,
    awb_gains: tuple[float, float, float] | None = None,
    ccm_matrix: NDArray[np.float64] | None = None,
    tone_exposure_ev: float = 0.0,
    color_space: str = "sRGB",
    denoise: DenoiseConfig | None = None,
    sharpen: SharpenConfig | None = None,
) -> ISPConfig:
    """Return an ISP configuration with the requested calibration applied.

    ``ccm_matrix=None`` disables the CCM stage instead of applying the preset's
    identity matrix, so the camera RGB used for fitting is genuinely
    pre-matrix. A supplied matrix enables the stage and is converted to nested
    Python floats, which keeps the pipeline's configuration hash serialisable.
    """
    awb = base.awb.model_copy(update={"gains": awb_gains})
    ccm = base.ccm.model_copy(
        update={
            "enabled": ccm_matrix is not None,
            "matrix": None if ccm_matrix is None else _matrix_as_list(ccm_matrix),
        }
    )
    tone = base.tone.model_copy(
        update={"exposure_compensation_ev": float(tone_exposure_ev), "color_space": color_space}
    )
    output = base.output.model_copy(update={"color_space": color_space})
    update: dict[str, Any] = {"awb": awb, "ccm": ccm, "tone": tone, "output": output}
    if denoise is not None:
        update["denoise"] = denoise
    if sharpen is not None:
        update["sharpen"] = sharpen
    return base.model_copy(update=update)


def _pre_tone_stage(stages: Mapping[str, StageOutput]) -> StageOutput:
    """Return the last linear stage immediately before the tone stage."""
    names = list(stages)
    if "tone" not in names:
        raise ValueError("the ISP trace has no tone stage")
    index = names.index("tone")
    if index == 0:
        raise ValueError("the tone stage is the first trace entry; no linear input exists")
    return stages[names[index - 1]]


# ---------------------------------------------------------------------------
# ColorChecker measurement
# ---------------------------------------------------------------------------


def _colorchecker_run(
    sensor_config: SensorConfig,
    isp_config: ISPConfig,
    projection: NDArray[np.float64],
    *,
    illuminant_name: str,
    shape: tuple[int, int],
    trace: bool = False,
    denoise: DenoiseConfig | None = None,
    sharpen: SharpenConfig | None = None,
) -> dict[str, Any]:
    """Capture, white-balance, fit, and evaluate one ColorChecker scene."""
    scene = _render_scene("colorchecker", illuminant_name, shape, seed=0)
    if len(scene.layout.patches) != 24:
        raise RuntimeError("the ColorChecker scene must expose 24 patch ROIs")
    sensor = Sensor(sensor_config, seed=_SCENE_SEED)
    exposure, exposure_clamped = _calibrate_exposure(
        sensor,
        scene,
        patch_name=COLORCHECKER_WHITE_PATCH,
        target_fraction=TARGET_WHITE_FRACTION,
        probe_s=PROBE_EXPOSURE_S,
        seed=_PROBE_SEED,
    )
    frame = sensor.expose(scene.irradiance, exposure, mode="LCG", seed=_SCENE_SEED)
    white = _find_patch(scene, COLORCHECKER_WHITE_PATCH)
    white_fraction = max(_channel_means(frame, white).values())

    # The perfect-diffuser camera level comes from a 50 % flat field at the
    # same exposure. Green has unit AWB gain, so twice its normalised signal is
    # the level a unit-reflectance diffuser would produce.
    flat_scene = _render_scene("flat_patch", illuminant_name, shape, seed=1)
    flat_frame = sensor.expose(flat_scene.irradiance, exposure, mode="LCG", seed=_FLAT_SEED)
    flat_means = _channel_means(flat_frame, flat_scene.layout.patches[0])
    if "G" not in flat_means:
        raise RuntimeError("the flat field did not expose a green CFA channel")
    white_level = 2.0 * flat_means["G"]
    if not math.isfinite(white_level) or not 0.0 < white_level < 1.0:
        raise ValueError(
            f"the perfect-diffuser camera level must lie in (0, 1); got {white_level!r}"
        )
    saturated_fraction = float(np.mean(frame.data >= 2**frame.bit_depth - 1))

    gains = _analytic_gains(sensor_config, projection, illuminant_name)
    camera_isp = ISP(
        _with_stages(
            isp_config,
            awb_gains=gains,
            ccm_matrix=None,
            denoise=DenoiseConfig(enabled=False),
            sharpen=SharpenConfig(enabled=False),
        )
    )
    camera_out = camera_isp.process(frame, trace=True)
    if camera_out.stages is None or "demosaic" not in camera_out.stages:
        raise RuntimeError("the pre-CCM camera run did not record a demosaic stage")
    camera_patch = _patch_means(camera_out.stages["demosaic"].data, scene.layout.patches)
    target_patch = _target_rgb_from_reference(scene.reference)
    camera_normalised = camera_patch / white_level
    matrix = fit_ccm(camera_normalised, target_patch, ridge=CCM_RIDGE, row_sum=True)
    residual = float(np.sqrt(np.mean((camera_normalised @ matrix.T - target_patch) ** 2)))

    # The CCM maps camera/white_level to the absolute target. The pipeline sees
    # camera (white level w), so the tone exposure gain 1/w restores the
    # absolute scale before the OETF. This is a linear-light exposure trim, not
    # a display-domain brightness change.
    tone_ev = float(np.log2(1.0 / white_level))
    render_isp = ISP(
        _with_stages(
            isp_config,
            awb_gains=gains,
            ccm_matrix=matrix,
            tone_exposure_ev=tone_ev,
            denoise=denoise,
            sharpen=sharpen,
        )
    )
    rendered = render_isp.process(frame, trace=trace)
    delta_e = _delta_e(rendered, scene.reference.lab, scene.layout.patches)

    linear_preclip = rendered.linear_rgb
    if linear_preclip is None:  # pragma: no cover - the pipeline always sets it
        raise RuntimeError("the rendered image has no linear buffer")
    preclip_xyz = np.asarray(
        colour.RGB_to_XYZ(
            np.asarray(linear_preclip, dtype=np.float64),
            colourspace="sRGB",
            illuminant=D65_XY,
        ),
        dtype=np.float64,
    )
    preclip_lab = np.asarray(colour.XYZ_to_Lab(preclip_xyz, illuminant=D65_XY), dtype=np.float64)
    delta_e_preclip = np.asarray(
        delta_E_CIE2000(
            _patch_means(preclip_lab, scene.layout.patches),
            scene.reference.lab,
        ),
        dtype=np.float64,
    )

    awb_stage = camera_out.stages.get("awb")
    awb_method = None if awb_stage is None else str(awb_stage.metadata.get("method"))
    awb_measured = None if awb_stage is None else awb_stage.metadata.get("gains")
    return {
        "scene": scene,
        "frame": frame,
        "flat_frame": flat_frame,
        "exposure_s": float(exposure),
        "exposure_clamped": bool(exposure_clamped),
        "white_fraction": float(white_fraction),
        "white_level": float(white_level),
        "saturated_fraction": saturated_fraction,
        "gains": gains,
        "awb_method": awb_method,
        "awb_measured_gains": None if awb_measured is None else list(awb_measured),
        "camera_patch": camera_patch,
        "target_patch": target_patch,
        "matrix": matrix,
        "ccm_residual": residual,
        "tone_ev": tone_ev,
        "rendered": rendered,
        "delta_e": delta_e,
        "delta_e_preclip": delta_e_preclip,
        "clip_fraction": float(np.mean(rendered.data >= 1.0 - 1e-6)),
        "sensor_config_hash": rendered.metadata["sensor_config_hash"],
        "isp_config_hash": render_isp.config_hash,
    }


def _measure_colorchecker(
    sensor_config: SensorConfig,
    isp_config: ISPConfig,
    projection: NDArray[np.float64],
    *,
    shape: tuple[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Measure the end-to-end ColorChecker colour accuracy for both illuminants."""
    sections: dict[str, Any] = {}
    runs: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for name in ("D65", "3000K"):
        run = _colorchecker_run(
            sensor_config,
            isp_config,
            projection,
            illuminant_name=name,
            shape=shape,
        )
        runs[name] = run
        delta_e = run["delta_e"]
        sections[name] = {
            "exposure_s": run["exposure_s"],
            "exposure_clamped": run["exposure_clamped"],
            "white_fraction": run["white_fraction"],
            "white_level": run["white_level"],
            "tone_exposure_ev": run["tone_ev"],
            "saturated_fraction": run["saturated_fraction"],
            "clip_fraction": run["clip_fraction"],
            "awb_gains": list(run["gains"]),
            "awb_method": run["awb_method"],
            "awb_measured_gains": run["awb_measured_gains"],
            "ccm": _matrix_as_list(run["matrix"]),
            "ccm_row_sums": [float(value) for value in run["matrix"].sum(axis=1)],
            "ccm_residual": run["ccm_residual"],
            "calibration_isp_config_hash": run["isp_config_hash"],
            "patch_names": [patch.name for patch in run["scene"].layout.patches],
            "camera_rgb": run["camera_patch"].tolist(),
            "target_rgb": run["target_patch"].tolist(),
            "delta_e2000": [float(value) for value in delta_e],
            "delta_e2000_mean": float(delta_e.mean()),
            "delta_e2000_max": float(delta_e.max()),
            "delta_e2000_preclip_mean": float(run["delta_e_preclip"].mean()),
            "delta_e2000_preclip_max": float(run["delta_e_preclip"].max()),
        }
        maximum = DELTA_E_D65_MAX if name == "D65" else DELTA_E_WARM_MAX
        label = "D65" if name == "D65" else "3000 K"
        checks.append(
            _check(
                f"ColorChecker mean deltaE2000 ({label})",
                float(delta_e.mean()),
                maximum=maximum,
                notes=(
                    "In-sample 3x3 ridge CCM on the same 24 patches; rendered sRGB "
                    "display decoded with the sRGB EOTF to XYZ D65 and CIE Lab; "
                    "mean CIEDE2000 against the spectrally computed reference."
                ),
            )
        )
    return sections, checks, runs


# ---------------------------------------------------------------------------
# White balance measurement
# ---------------------------------------------------------------------------


def _injected_scaling_frame(
    frame: RawFrame,
    injection: tuple[float, float, float],
) -> RawFrame:
    """Return a frame with a known per-CFA channel cast injected in the signal.

    The pedestal is preserved: the scaling multiplies ``DN - black_level`` and
    the black level is added back, because a physical cast multiplies light,
    not the ADC offset. The result is rounded and clipped to the code range.
    """
    scale = np.empty(frame.data.shape, dtype=np.float64)
    for row_phase in (0, 1):
        for column_phase in (0, 1):
            colour = cfa_color_at(frame.cfa, row_phase, column_phase)
            index = {"R": 0, "G": 1, "B": 2}[colour]
            scale[row_phase::2, column_phase::2] = injection[index]
    signal = (frame.data.astype(np.float64) - frame.black_level_dn) * scale
    maximum = float(2**frame.bit_depth - 1)
    data = np.clip(np.round(signal) + frame.black_level_dn, 0.0, maximum).astype(np.uint16)
    return frame.with_data(data)


def _awb_gains_from(
    isp_config: ISPConfig,
    frame: RawFrame,
    *,
    awb_gains: tuple[float, float, float] | None = None,
    ccm_matrix: NDArray[np.float64] | None = None,
) -> tuple[list[float], str, float]:
    """Run the pipeline and return the AWB gains, method, and confidence."""
    config = _with_stages(
        isp_config,
        awb_gains=awb_gains,
        ccm_matrix=ccm_matrix,
        denoise=DenoiseConfig(enabled=False),
        sharpen=SharpenConfig(enabled=False),
    )
    rendered = ISP(config).process(frame, trace=True)
    stages = rendered.stages
    if stages is None or "awb" not in stages:
        raise RuntimeError("the AWB measurement did not record an awb stage")
    metadata = stages["awb"].metadata
    return (
        [float(value) for value in metadata["gains"]],
        str(metadata["method"]),
        float(metadata["confidence"]),
    )


def _measure_awb(
    sensor_config: SensorConfig,
    isp_config: ISPConfig,
    projection: NDArray[np.float64],
    runs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure gray-world recovery on flat fields and under an injected cast."""
    sections: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    for name in ("D65", "3000K"):
        run = runs[name]
        frame = run["flat_frame"]
        expected = np.asarray(_analytic_gains(sensor_config, projection, name), dtype=np.float64)
        measured, method, confidence = _awb_gains_from(isp_config, frame)
        measured_array = np.asarray(measured, dtype=np.float64)
        errors = np.abs(measured_array / expected - 1.0)
        sections[name] = {
            "exposure_s": run["exposure_s"],
            "expected_gains": [float(value) for value in expected],
            "measured_gains": [float(value) for value in measured_array],
            "relative_errors": [float(value) for value in errors],
            "method": method,
            "confidence": confidence,
        }
        for index, channel in ((0, "R"), (2, "B")):
            checks.append(
                _check(
                    f"AWB gain relative error, {channel} ({'D65' if name == 'D65' else '3000 K'})",
                    float(errors[index]),
                    maximum=AWB_TOLERANCE,
                    notes=(
                        "Expected gain from the analytic spectral response to a "
                        "perfect diffuser; gray-world estimate on a 50 % neutral flat "
                        "field; green is normalised to 1."
                    ),
                )
            )

    injection = (1.6, 1.0, 0.7)
    injected_sections: dict[str, Any] = {}
    max_error = 0.0
    for name in ("D65", "3000K"):
        frame = runs[name]["flat_frame"]
        injected = _injected_scaling_frame(frame, injection)
        expected = np.asarray(_analytic_gains(sensor_config, projection, name), dtype=np.float64)
        expected = expected / np.asarray(injection, dtype=np.float64)
        expected = expected / expected[1]
        measured, method, confidence = _awb_gains_from(isp_config, injected)
        measured_array = np.asarray(measured, dtype=np.float64)
        errors = np.abs(measured_array / expected - 1.0)
        max_error = max(max_error, float(max(errors[0], errors[2])))
        injected_sections[name] = {
            "expected_gains": [float(value) for value in expected],
            "measured_gains": [float(value) for value in measured_array],
            "relative_errors": [float(value) for value in errors],
            "method": method,
            "confidence": confidence,
        }
    checks.append(
        _check(
            "AWB injected-cast recovery (max relative error)",
            float(max_error),
            maximum=AWB_TOLERANCE,
            notes=(
                "A known per-CFA scaling (R x1.6, G x1.0, B x0.7) is injected into "
                "a neutral flat frame; the gray-world estimate must invert it on "
                "top of the illuminant cast."
            ),
        )
    )
    return (
        {
            "flat_reflectance": 0.5,
            "illuminants": sections,
            "injected_scaling": {
                "injection": list(injection),
                "illuminants": injected_sections,
                "max_relative_error": float(max_error),
            },
        },
        checks,
    )


# ---------------------------------------------------------------------------
# Demosaic, tone, trace, and linearity
# ---------------------------------------------------------------------------


def _star_reference(shape: tuple[int, int]) -> NDArray[np.float32]:
    """Return a colour Siemens-star RGB image with independent channel structure.

    A grey star would make bilinear and MHC agree trivially. The red channel
    uses 36 spokes, the green channel 18, and the blue channel concentric rings,
    so every interpolation kernel encounters independent per-channel edges.
    """
    height, width = shape
    red = siemens_star_target((height, width), spokes=36)
    green = siemens_star_target((height, width), spokes=18)
    rows, columns = np.mgrid[0:height, 0:width]
    radius = np.hypot(rows - (height - 1) / 2.0, columns - (width - 1) / 2.0)
    blue = 0.5 + 0.5 * np.cos(radius / 2.0)
    return np.ascontiguousarray(np.stack((red, green, blue), axis=-1), dtype=np.float32)


def _star_case(
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Demosaic the synthetic star with both methods and both references."""
    try:
        from colour_demosaicing import (
            demosaicing_CFA_Bayer_bilinear,
            demosaicing_CFA_Bayer_Malvar2004,
        )
    except ImportError as exc:  # pragma: no cover - dev extra is installed in CI
        raise RuntimeError(
            "the G3 demosaic cross-check needs colour-demosaicing; install the "
            "dev extra with 'uv sync --all-extras'"
        ) from exc

    reference = _star_reference(shape)
    mosaic = mosaic_from_channels(
        {
            "R": reference[..., 0],
            "G": reference[..., 1],
            "B": reference[..., 2],
        },
        CFAPattern.RGGB,
    )
    bilinear = np.asarray(demosaic_bilinear(mosaic, CFAPattern.RGGB), dtype=np.float32)
    mhc = np.asarray(demosaic_mhc(mosaic, CFAPattern.RGGB), dtype=np.float32)
    reference_bilinear = np.asarray(
        demosaicing_CFA_Bayer_bilinear(mosaic, "RGGB"), dtype=np.float64
    )
    reference_mhc = np.asarray(demosaicing_CFA_Bayer_Malvar2004(mosaic, "RGGB"), dtype=np.float64)
    bilinear_error = np.abs(bilinear.astype(np.float64) - reference_bilinear)
    mhc_error = np.abs(mhc.astype(np.float64) - reference_mhc)
    return {
        "reference": reference,
        "bilinear": bilinear,
        "mhc": mhc,
        "bilinear_max_abs": float(bilinear_error.max()),
        "mhc_max_abs": float(mhc_error.max()),
        "bilinear_mean_abs": float(bilinear_error.mean()),
        "mhc_mean_abs": float(mhc_error.mean()),
    }


def _measure_demosaic(
    shape: tuple[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare bilinear and MHC demosaicing with ``colour-demosaicing``."""
    case = _star_case(shape)
    height, width = shape
    section: dict[str, Any] = {
        "shape": [int(height), int(width)],
        "height": int(height),
        "width": int(width),
        "source": "synthetic Siemens-star RGB mosaic (RGGB), independent R/G/B structure",
        "bilinear_max_abs": case["bilinear_max_abs"],
        "mhc_max_abs": case["mhc_max_abs"],
        "bilinear_mean_abs": case["bilinear_mean_abs"],
        "mhc_mean_abs": case["mhc_mean_abs"],
    }
    checks = [
        _check(
            "Bilinear demosaic vs colour-demosaicing (max abs)",
            case["bilinear_max_abs"],
            maximum=DEMOSAIC_BILINEAR_MAX,
            notes=(
                "Full-frame max absolute difference on a Siemens-star RGB mosaic; "
                "the reference uses symmetric-reflection borders, the same rule "
                "the implementation follows."
            ),
        ),
        _check(
            "MHC demosaic vs colour-demosaicing (max abs)",
            case["mhc_max_abs"],
            maximum=DEMOSAIC_MHC_MAX,
            notes=(
                "Malvar-He-Cutler full-frame max absolute difference against the "
                "reference implementation on the same star mosaic."
            ),
        ),
    ]
    return section, checks


def _measure_tone() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare the sRGB OETF with the colour-science standard."""
    samples = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    ours = np.asarray(linear_to_srgb(samples), dtype=np.float64)
    reference = np.asarray(colour.cctf_encoding(samples, function="sRGB"), dtype=np.float64)
    max_abs = float(np.max(np.abs(ours - reference)))
    section: dict[str, Any] = {
        "sample_count": int(samples.size),
        "max_abs": max_abs,
        "domain": [0.0, 1.0],
    }
    checks = [
        _check(
            "sRGB OETF vs colour.cctf_encoding (max abs)",
            max_abs,
            maximum=TONE_MAX_ABS,
            notes=(
                "1001 linear samples in [0, 1] encoded by camsim.isp.tone.linear_to_srgb "
                "and compared with colour.cctf_encoding(function='sRGB')."
            ),
        )
    ]
    return section, checks


def _measure_trace(
    isp_config: ISPConfig,
    run: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify that the pre-tone trace stage reproduces the rendered buffers."""
    exposure_ev = 0.5
    p3_config = _with_stages(
        isp_config,
        awb_gains=run["gains"],
        ccm_matrix=run["matrix"],
        tone_exposure_ev=exposure_ev,
        color_space="DisplayP3",
        denoise=DenoiseConfig(enabled=False),
        sharpen=SharpenConfig(enabled=False),
    )
    rendered = ISP(p3_config).process(run["frame"], trace=True)
    stages = rendered.stages
    if stages is None:
        raise RuntimeError("the trace validation did not record stages")
    pre = np.asarray(_pre_tone_stage(stages).data, dtype=np.float32)
    reproduced_display = np.asarray(apply_tone(pre, p3_config.tone), dtype=np.float64)
    display_error = float(np.max(np.abs(reproduced_display - rendered.data.astype(np.float64))))

    expected_linear = np.asarray(pre, dtype=np.float64) * 2.0**exposure_ev
    expected_p3 = np.asarray(linear_srgb_to_p3(expected_linear), dtype=np.float64)
    linear_buffer = rendered.linear_rgb
    if linear_buffer is None:  # pragma: no cover - the pipeline always sets it
        raise RuntimeError("the trace validation produced no linear buffer")
    linear_error = float(np.max(np.abs(expected_p3 - np.asarray(linear_buffer, dtype=np.float64))))
    names = list(stages)
    section: dict[str, Any] = {
        "stage_names": names,
        "color_space": "DisplayP3",
        "exposure_ev": exposure_ev,
        "tone_max_abs": display_error,
        "linear_rgb_max_abs": linear_error,
    }
    checks = [
        _check(
            "Stage trace tone reproduction (max abs)",
            display_error,
            maximum=TRACE_MAX_ABS,
            notes=(
                "apply_tone on the last linear trace stage (pre-tone) against "
                "RenderedImage.data with +0.5 EV and Display P3 output."
            ),
        ),
        _check(
            "Stage trace linear_rgb reproduction (Display P3, max abs)",
            linear_error,
            maximum=TRACE_MAX_ABS,
            notes=(
                "The pre-tone linear stage scaled by 2**0.5 and converted with "
                "linear_srgb_to_p3 against RenderedImage.linear_rgb, before clipping."
            ),
        ),
    ]
    return section, checks


def _linearity_frame(code_signal: int, shape: tuple[int, int]) -> RawFrame:
    """Return a synthetic RGGB frame with a spatial gradient at a known level."""
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width]
    base = 0.12 + 0.22 * columns / max(width - 1, 1) + 0.08 * rows / max(height - 1, 1)
    planes = {
        "R": (base * 0.98).astype(np.float32),
        "G": base.astype(np.float32),
        "B": (base * 0.94).astype(np.float32),
    }
    mosaic = mosaic_from_channels(planes, CFAPattern.RGGB)
    bit_depth = 10
    black = 64
    data = np.clip(np.round(black + mosaic * code_signal), 0, 2**bit_depth - 1).astype(np.uint16)
    return RawFrame(
        data=data,
        cfa=CFAPattern.RGGB,
        bit_depth=bit_depth,
        black_level_dn=black,
        exposure_s=0.01,
        analog_gain_db=0.0,
        digital_gain=1.0,
        mode="LCG",
        temperature_c=25.0,
        seed=0,
        frame_index=0,
        pixel_pitch_um=1.12,
        sensor_config_hash="g3-linearity",
    )


def _linearity_ccm() -> NDArray[np.float64]:
    """Return a positively fitted 3x3 CCM for the linearity frame.

    The matrix is fitted with :func:`camsim.isp.ccm.fit_ccm` from a
    deterministic, well-conditioned set of in-gamut camera/target samples,
    using a tiny ridge and the row-sum constraint. It is not the ColorChecker
    matrix; its purpose is to exercise the CCM stage in the linearity test with
    a real fit rather than a hand-typed identity. The matrix is chosen so that
    every output on the synthetic linearity frame is positive, which keeps the
    stage's documented non-negativity clamp inactive and the linearity check
    meaningful.
    """
    target = np.array(
        [
            [0.08, 0.10, 0.12],
            [0.18, 0.16, 0.14],
            [0.28, 0.24, 0.20],
            [0.38, 0.34, 0.30],
            [0.48, 0.46, 0.42],
            [0.58, 0.54, 0.50],
            [0.68, 0.64, 0.60],
            [0.78, 0.74, 0.70],
            [0.12, 0.42, 0.22],
            [0.22, 0.62, 0.32],
            [0.32, 0.52, 0.72],
            [0.72, 0.52, 0.32],
            [0.52, 0.72, 0.42],
            [0.42, 0.22, 0.62],
            [0.62, 0.32, 0.12],
            [0.82, 0.42, 0.22],
            [0.72, 0.82, 0.52],
            [0.52, 0.62, 0.82],
            [0.20, 0.30, 0.80],
            [0.80, 0.30, 0.20],
            [0.30, 0.80, 0.30],
            [0.40, 0.40, 0.40],
            [0.60, 0.60, 0.60],
            [0.90, 0.85, 0.80],
        ],
        dtype=np.float64,
    )
    reference = np.array(
        [
            [0.98, 0.03, -0.01],
            [-0.02, 1.04, -0.02],
            [0.01, -0.03, 1.02],
        ],
        dtype=np.float64,
    )
    camera = target @ np.linalg.inv(reference).T
    fitted = fit_ccm(camera, target, ridge=1e-6, row_sum=True)
    return np.asarray(fitted, dtype=np.float64)


def _measure_linearity(
    shape: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify that doubling the RAW signal doubles the pre-tone linear output.

    Three properties are measured. First, gray-world AWB gain *estimation* is
    scale invariant: doubling the signal must not change the estimated gains.
    Second, with AWB enabled (pinned to those estimated gains) and a positively
    fitted CCM, doubling the black-level-corrected RAW signal must double the
    last linear trace stage exactly; the fitted matrix is checked to produce
    strictly positive pre-clamp values, so its documented non-negativity clamp
    is inactive. Third, the result is reported with its pre-clamp minimum so
    the exception is explicit rather than asserted in prose.

    Denoise and sharpen are disabled because both are optional enhancement
    stages and the shipped baseline leaves them off. Tone is excluded because
    it is the one intentionally non-linear stage.
    """
    isp_config = load_isp(_ISP_PRESET)
    frame = _linearity_frame(code_signal=200, shape=(shape, shape))
    black = frame.black_level_dn
    doubled_data = np.clip(
        np.round((frame.data.astype(np.float64) - black) * 2.0 + black),
        0,
        2**frame.bit_depth - 1,
    ).astype(np.uint16)
    doubled = frame.with_data(doubled_data)

    awb_config = _with_stages(
        isp_config,
        awb_gains=None,
        ccm_matrix=None,
        denoise=DenoiseConfig(enabled=False),
        sharpen=SharpenConfig(enabled=False),
    )
    awb_first = ISP(awb_config).process(frame, trace=True)
    awb_second = ISP(awb_config).process(doubled, trace=True)
    if awb_first.stages is None or awb_second.stages is None:
        raise RuntimeError("the AWB linearity check did not record stages")
    gains_first = np.asarray(awb_first.stages["awb"].metadata["gains"], dtype=np.float64)
    gains_second = np.asarray(awb_second.stages["awb"].metadata["gains"], dtype=np.float64)
    gain_denominator = float(np.max(np.abs(gains_first)))
    if gain_denominator <= 0.0:
        raise ValueError("the AWB linearity check produced an all-zero gain vector")
    awb_scale_error = float(np.max(np.abs(gains_second - gains_first)) / gain_denominator)

    matrix = _linearity_ccm()
    config = _with_stages(
        isp_config,
        awb_gains=(float(gains_first[0]), float(gains_first[1]), float(gains_first[2])),
        ccm_matrix=matrix,
        denoise=DenoiseConfig(enabled=False),
        sharpen=SharpenConfig(enabled=False),
    )
    first = ISP(config).process(frame, trace=True)
    second = ISP(config).process(doubled, trace=True)
    if first.stages is None or second.stages is None:
        raise RuntimeError("the linearity check did not record stages")
    before = np.asarray(_pre_tone_stage(first.stages).data, dtype=np.float64)
    after = np.asarray(_pre_tone_stage(second.stages).data, dtype=np.float64)
    doubled_before = 2.0 * before
    denominator = float(np.max(np.abs(doubled_before)))
    if denominator <= 0.0:
        raise ValueError("the linearity check produced an all-zero pre-tone image")
    max_relative = float(np.max(np.abs(after - doubled_before)) / denominator)
    demosaiced = np.asarray(first.stages["demosaic"].data, dtype=np.float64)
    pre_clamp = demosaiced @ matrix.T
    pre_clamp_min = float(pre_clamp.min())

    section: dict[str, Any] = {
        "scale": 2.0,
        "max_relative_error": max_relative,
        "pixels_compared": int(before.size),
        "awb_estimated_gains": [float(value) for value in gains_first],
        "awb_scale_relative_error": awb_scale_error,
        "ccm": _matrix_as_list(matrix),
        "ccm_row_sums": [float(value) for value in matrix.sum(axis=1)],
        "ccm_pre_clamp_min": pre_clamp_min,
        "ccm_clamp_active": bool(pre_clamp_min < 0.0),
        "source": (
            "synthetic RGGB gradient frame; signal doubled after BLC; AWB enabled "
            "(estimated gains) and a fitted, in-gamut positive CCM"
        ),
    }
    checks = [
        _check(
            "AWB gain scale invariance (max relative error)",
            awb_scale_error,
            maximum=LINEARITY_AWB_MAX_REL,
            notes=(
                "Gray-world gain estimation on the frame and on its doubled, "
                "black-level-corrected signal; normalised gain ratios are scale-free."
            ),
        ),
        _check(
            "Linear pre-tone scaling invariance with AWB and CCM (max relative error)",
            max_relative,
            maximum=LINEARITY_MAX_REL,
            notes=(
                "Doubling the black-level-corrected RAW signal must double the last "
                "linear trace stage exactly with AWB enabled and a fitted CCM; the "
                "CCM clamp is inactive because every pre-clamp output is positive."
            ),
        ),
        _check(
            "CCM pre-clamp minimum on linearity data",
            pre_clamp_min,
            minimum=0.0,
            notes=(
                "The CCM non-negativity clamp is the one bounded exception to the "
                "linear-space invariant; it is inactive on this in-gamut frame."
            ),
        ),
    ]
    return section, checks


# ---------------------------------------------------------------------------
# Lens shading, denoise, sharpen, and RCCC
# ---------------------------------------------------------------------------


def _vignette_field(size: int) -> NDArray[np.float64]:
    """Return a radial ``1 - 0.3 r**2`` flat-field response."""
    rows, columns = np.mgrid[0:size, 0:size]
    radius_squared = ((columns - (size - 1) / 2.0) ** 2 + (rows - (size - 1) / 2.0) ** 2) / (
        2.0 * ((size - 1) / 2.0) ** 2
    )
    return np.asarray(1.0 - 0.3 * radius_squared, dtype=np.float64)


def _measure_lsc(
    size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calibrate LSC on one noisy flat and correct an independent flat."""
    vignette = _vignette_field(size)
    rng = np.random.default_rng(_LSC_SEED)
    noise_sigma = 0.0005
    flat_one = vignette * (1.0 + noise_sigma * rng.standard_normal(vignette.shape))
    flat_two = vignette * (1.0 + noise_sigma * rng.standard_normal(vignette.shape))
    gain_map = np.asarray(calibrate_lsc(flat_one), dtype=np.float64)
    corrected = apply_lsc(flat_two, LSCConfig(mode="flat_field", map=gain_map))
    corrected_same = apply_lsc(flat_one, LSCConfig(mode="flat_field", map=gain_map))
    residual_before = float((vignette.max() - vignette.min()) / vignette.mean())
    residual_after = float((corrected.max() - corrected.min()) / corrected.mean())
    same_frame = float((corrected_same.max() - corrected_same.min()) / corrected_same.mean())
    section: dict[str, Any] = {
        "noise_sigma": noise_sigma,
        "residual_before": residual_before,
        "residual_after": residual_after,
        "same_frame_residual": same_frame,
        "map_min": float(gain_map.min()),
        "map_max": float(gain_map.max()),
        "size": int(size),
    }
    checks = [
        _check(
            "LSC flat-field residual after correction",
            residual_after,
            maximum=LSC_RESIDUAL_MAX,
            notes=(
                "A gain map calibrated from one noisy flat field is applied to an "
                "independent flat field; residual is (max - min) / mean. "
                "tests/test_isp_lsc.py::test_lsc_flat_field_corrects_vignetting is "
                "the exact-reciprocal unit check."
            ),
        )
    ]
    return section, checks


def _luma(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return BT.709 luma for an RGB image."""
    return np.asarray(
        0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2],
        dtype=np.float64,
    )


def _step_image(size: int, left: float = 0.2, right: float = 0.6) -> NDArray[np.float32]:
    """Return an RGB step edge in linear light."""
    image = np.empty((size, size, 3), dtype=np.float32)
    image[:, : size // 2] = left
    image[:, size // 2 :] = right
    return image


def _measure_denoise(
    size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure denoise SNR gain and step-edge contrast preservation."""
    rng = np.random.default_rng(_DENOISE_SEED)
    noise_sigma = 0.02
    clean = _step_image(size)
    noisy = clean + noise_sigma * rng.standard_normal(clean.shape).astype(np.float32)
    config = DenoiseConfig(
        method="bilateral",
        luma_strength=1.0,
        chroma_strength=1.0,
        radius=1.5,
    )
    denoised = np.asarray(denoise(noisy, config), dtype=np.float64)
    noisy64 = noisy.astype(np.float64)
    third = size // 8
    left_slice = (slice(third, size - third), slice(third, size // 2 - third))
    right_slice = (slice(third, size - third), slice(size // 2 + third, size - third))
    before_flat = _luma(noisy64)[left_slice]
    after_flat = _luma(denoised)[left_slice]
    snr_before = float(20.0 * math.log10(abs(before_flat.mean()) / before_flat.std(ddof=1)))
    snr_after = float(20.0 * math.log10(abs(after_flat.mean()) / after_flat.std(ddof=1)))
    contrast_before = float(_luma(noisy64)[right_slice].mean() - _luma(noisy64)[left_slice].mean())
    contrast_after = float(_luma(denoised)[right_slice].mean() - _luma(denoised)[left_slice].mean())
    edge_error = float(abs(contrast_after / contrast_before - 1.0))
    section: dict[str, Any] = {
        "method": config.method,
        "radius": config.radius,
        "luma_strength": config.luma_strength,
        "chroma_strength": config.chroma_strength,
        "noise_sigma": noise_sigma,
        "snr_before_db": snr_before,
        "snr_after_db": snr_after,
        "snr_gain_db": snr_after - snr_before,
        "edge_contrast_before": contrast_before,
        "edge_contrast_after": contrast_after,
        "edge_preservation_error": edge_error,
    }
    checks = [
        _check(
            "Denoise SNR gain (dB)",
            section["snr_gain_db"],
            minimum=DENOISE_SNR_GAIN_MIN,
            notes=(
                "Bilateral luma/chroma denoise on a noisy flat region; SNR is "
                "20 log10(|mean| / std) of BT.709 luma. Formal slanted-edge MTF "
                "is measured independently by the G4 metrics harness."
            ),
        ),
        _check(
            "Denoise edge-contrast preservation",
            edge_error,
            maximum=DENOISE_EDGE_MAX,
            notes=(
                "Relative change of the step-edge contrast between flat regions "
                "away from the edge; the bilateral range gate must preserve it "
                "within 10 percent."
            ),
        ),
    ]
    return section, checks


def _measure_sharpen(
    size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure unsharp-mask edge-contrast gain and positive overshoot."""
    clean = _step_image(size)
    config = SharpenConfig(amount=0.8, radius=1.5, threshold=0.0, luma_only=True)
    sharpened = np.asarray(unsharp_mask(clean, config), dtype=np.float64)
    profile_before = _luma(clean.astype(np.float64)).mean(axis=0)
    profile_after = _luma(sharpened).mean(axis=0)
    contrast_before = float(profile_before.max() - profile_before.min())
    contrast_after = float(profile_after.max() - profile_after.min())
    contrast_gain = float(contrast_after / contrast_before - 1.0)
    overshoot = float(profile_after.max() - profile_before.max())
    section: dict[str, Any] = {
        "amount": config.amount,
        "radius": config.radius,
        "threshold": config.threshold,
        "luma_only": config.luma_only,
        "edge_contrast_before": contrast_before,
        "edge_contrast_after": contrast_after,
        "contrast_gain": contrast_gain,
        "overshoot": overshoot,
    }
    checks = [
        _check(
            "Sharpen edge-contrast gain",
            contrast_gain,
            minimum=0.0,
            notes=(
                "Thresholded unsharp mask on a clean step edge; the full-profile "
                "contrast must increase. Formal ISO 12233 MTF50 validation is "
                "measured independently by the G4 metrics harness; the harness "
                "exists in this release."
            ),
        ),
        _check(
            "Sharpen overshoot",
            overshoot,
            minimum=0.0,
            notes=(
                "The bright-side profile maximum must exceed the unsharpened "
                "plateau, i.e. the stage produces measurable positive overshoot."
            ),
        ),
    ]
    return section, checks


def _rccc_case(
    isp_config: ISPConfig,
) -> dict[str, Any]:
    """Render the RCCC baseline smoke image and its serialisable summary."""
    height, width = RCCC_SHAPE
    columns = np.linspace(0.0, 6.0 * math.pi, width)
    rows = np.linspace(0.0, 4.0 * math.pi, height)
    red = 0.25 + 0.2 * np.cos(columns)[None, :] * np.ones((height, 1))
    clear = 0.45 + 0.25 * np.sin(rows)[:, None] * np.ones((1, width))
    mosaic = mosaic_from_channels(
        {"R": red.astype(np.float32), "C": clear.astype(np.float32)},
        CFAPattern.RCCC,
    )
    bit_depth = 10
    black = 64
    span = 2**bit_depth - 1 - black
    data = np.clip(np.round(black + mosaic * span), 0, 2**bit_depth - 1).astype(np.uint16)
    frame = RawFrame(
        data=data,
        cfa=CFAPattern.RCCC,
        bit_depth=bit_depth,
        black_level_dn=black,
        exposure_s=0.01,
        analog_gain_db=0.0,
        digital_gain=1.0,
        mode="LCG",
        temperature_c=25.0,
        seed=0,
        frame_index=0,
        pixel_pitch_um=1.12,
        sensor_config_hash="g3-rccc",
    )
    rendered = ISP(isp_config).process(frame, trace=True)
    image = np.asarray(rendered.data, dtype=np.float32)
    stages = rendered.stages
    summary = {
        "shape": list(image.shape),
        "channels": int(image.shape[-1]),
        "finite": bool(np.all(np.isfinite(image))),
        "min": float(image.min()),
        "max": float(image.max()),
        "stages": list(stages) if stages is not None else [],
    }
    return {"image": image, "summary": summary}


def _measure_rccc(
    isp_config: ISPConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the RCCC baseline smoke test; no Bayer colour or MTF gate applies."""
    case = _rccc_case(isp_config)
    summary = case["summary"]
    valid = (
        summary["channels"] == 3
        and summary["finite"]
        and 0.0 <= summary["min"] <= summary["max"] <= 1.0
    )
    checks = [
        _check(
            "RCCC baseline output channels",
            float(summary["channels"]),
            target=3.0,
            absolute=0.0,
            notes=(
                "RCCC is reconstructed with the documented baseline demosaic; AWB "
                "and CCM are skipped by the pipeline. Output must be a finite "
                "(H, W, 3) image. The Bayer ΔE and MTF thresholds do not apply; "
                "advanced RCCC demosaicing is deferred."
            ),
        )
    ]
    if not valid:  # pragma: no cover - the baseline is expected to be valid
        checks[0]["pass"] = False
    return summary, checks


# ---------------------------------------------------------------------------
# Checks, formatting, and environment
# ---------------------------------------------------------------------------


def _format_number(value: float) -> str:
    """Format a measured value consistently for JSON, Markdown, and tests."""
    if not math.isfinite(value):
        raise ValueError(f"cannot format a non-finite measured value: {value!r}")
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e5:
        return f"{value:.4g}"
    return f"{value:.6g}"


def _format_tolerance(tolerance: Mapping[str, float]) -> str:
    """Format a tolerance mapping as a compact human-readable condition."""
    if "minimum" in tolerance and "maximum" in tolerance:
        return (
            f"{_format_number(tolerance['minimum'])} < x < {_format_number(tolerance['maximum'])}"
        )
    if "maximum" in tolerance:
        return f"< {_format_number(tolerance['maximum'])}"
    if "minimum" in tolerance:
        return f"> {_format_number(tolerance['minimum'])}"
    if "target" in tolerance:
        target = _format_number(tolerance["target"])
        if "relative" in tolerance:
            return f"{target} +/- {100.0 * tolerance['relative']:.4g} %"
        if "absolute" in tolerance:
            return f"= {target} +/- {_format_number(tolerance['absolute'])}"
        return f"= {target}"
    raise ValueError(f"unsupported tolerance mapping: {dict(tolerance)!r}")


def _check(
    name: str,
    measured: float,
    *,
    maximum: float | None = None,
    minimum: float | None = None,
    target: float | None = None,
    relative: float | None = None,
    absolute: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Return one serialisable gate check with an honest verdict.

    A ``maximum`` is a strict upper bound, a ``minimum`` is a strict lower
    bound, and a ``target`` uses ``math.isclose`` with the supplied relative or
    absolute tolerance. The display strings are stored in the JSON so the
    Markdown report and the report-consistency test compare exactly.
    """
    value = float(measured)
    if not math.isfinite(value):
        raise ValueError(f"check {name!r} measured a non-finite value: {value!r}")
    tolerance: dict[str, float] = {}
    passed = True
    if maximum is not None:
        tolerance["maximum"] = float(maximum)
        passed = passed and value < float(maximum)
    if minimum is not None:
        tolerance["minimum"] = float(minimum)
        passed = passed and value > float(minimum)
    if target is not None:
        tolerance["target"] = float(target)
        rel = 0.0 if relative is None else float(relative)
        absolute_tolerance = 0.0 if absolute is None else float(absolute)
        if relative is not None:
            tolerance["relative"] = rel
        if absolute is not None:
            tolerance["absolute"] = absolute_tolerance
        passed = passed and math.isclose(
            value, float(target), rel_tol=rel, abs_tol=absolute_tolerance
        )
    return {
        "name": name,
        "measured": value,
        "tolerance": tolerance,
        "pass": bool(passed),
        "notes": notes,
        "measured_text": _format_number(value),
        "tolerance_text": _format_tolerance(tolerance),
    }


def _environment() -> dict[str, str]:
    """Return the software environment used by the report."""
    import scipy  # type: ignore[import-untyped]

    try:
        import colour_demosaicing

        demosaicing_version = str(colour_demosaicing.__version__)
    except ImportError:  # pragma: no cover - measure_g3 needs it for the star case
        demosaicing_version = "not installed"
    return {
        "python": platform.python_version(),
        "camsim": camsim_version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "colour_science": colour.__version__,
        "colour_demosaicing": demosaicing_version,
        "platform": platform.system(),
    }
