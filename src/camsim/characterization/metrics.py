"""G4 metrics validation: analytic MTF, known-noise SNR, colour, and the harness.

The G4 gate asks whether the independent metrics layer measures what it claims
to measure. Its evidence is deliberately built from references that share no
code with the metric under test, so a bug cannot cancel between the simulator
and its judge.

**Analytic MTF.** A slanted edge is blurred by a known PSF and measured with
:func:`camsim.metrics.slanted_edge_mtf`. The reference is the exact
discrete-time transfer function :func:`camsim.optics.psf.psf_mtf` of the same
kernel, evaluated at arbitrary frequencies rather than interpolated from an FFT
grid. Three physical models cover the useful range: a Gaussian aberration core
(``sigma = 1 px``), the square pixel aperture (``width = 4 px``, whose transfer
function is ``|sinc(4 f)|``), and diffraction (Airy at f/4 and f/8, 550 nm,
1.12 um pitch). The metric is run at eight analysis samples per pixel: four is
the ISO 12233 convention, but near diffraction-limited systems carry about a
2 % MTF50 bias at four, so the gate cases use eight and the gallery may use
four. Airy f/2.8 is reported only as a diagnostic because its first zero sits
so close to Nyquist that the measured crossings are noisier and its transfer
function never reaches 0.20 inside the band.

**Known-noise SNR.** A synthetic flat patch is generated with known per-channel
means and Gaussian noise. The spatial estimator is compared with
``20 log10(mu / sigma)`` using the **generating** means and noise sigma, not the
estimator's own sample statistics; an estimator checked against its own
arithmetic would always read zero error and prove nothing. The Rec. 709 luma
reference uses the weighted generating means and the quadrature-weighted sigma,
because the channel noise draws are independent. A second stack adds a static
per-pixel fixed pattern on top of independent temporal noise; the temporal
estimator is compared with ``20 log10(TEMPORAL_MEAN / TEMPORAL_SIGMA)`` from the
generating constants, while the single-frame spatial estimate is materially
worse because it counts the fixed pattern as noise. This is the fixed-pattern
rejection that separates the two estimators. A mono path is measured against
the generating R-channel mean and sigma with the same rules. A separate
sample-statistics consistency diagnostic records the (identically zero) error
against each patch's own sample mean and std; it is reported for transparency
and is not gate evidence.

**Colour difference.** :func:`camsim.metrics.color._delta_e76` and
:func:`camsim.metrics.color._delta_e2000` are compared with
``colour.delta_E(..., method="CIE 1976"/"CIE 2000")`` on neutral, lightness,
hue-rotation, and the published Sharma et al. (2005) CIEDE2000 test pairs. The
tolerance is ``1e-9``: both implementations evaluate the same closed-form
equation, so anything larger is a real disagreement, not accumulated
statistics. The published Sharma values are printed to four decimals and are
checked at ``1e-4`` as an independent published-source confirmation.

**Unified harness.** A synthetic ColorChecker-like display image is built from
the spectrally computed reference Lab (reference white -> Bradford D65 ->
linear sRGB -> sRGB OETF), with a deterministic per-patch Lab perturbation and
a small amount of display noise so every SNR patch is valid. ``measure_all``
must return complete colour and SNR groups and, for a separate edge layout,
MTF and acutance. A perfectly sharp synthetic edge legitimately reports no
finite MTF50/MTF20 (the curve never falls to one half inside Nyquist); the
report must serialize that as ``null`` rather than the non-standard ``NaN``
token. A MONO image must return SNR with no colour group. Finally, a fresh
subprocess checks that importing :mod:`camsim.metrics` does not load the
sensor, ISP, scene, or optics packages.

Real-world context: phone lenses measure MTF50 of 0.1-0.4 cycles/pixel, f/8
diffraction-limited systems near 0.1, and mid-grey SNR of roughly 10-45 dB;
camera colour accuracy is usually reported as a mean CIEDE2000 of 1-5 on a
ColorChecker. The synthetic chart here is a measurement vehicle, not a claim
about a physical camera.

Failure artefacts guarded here: a quick smoke run mistaken for gate evidence;
gamma-encoded data fed to MTF or SNR; a reference PSF sampled on a different
grid than the blurred edge; a metric compared with a reference that shares its
own estimator; a fixed pattern counted as temporal noise; a colour pair with
wrapped hue treated linearly; a missing colour reference silently replaced by
D65; a NaN crossing written as invalid JSON; and a metrics package that imports
the simulator it is supposed to judge. Every seed is explicit and every output
is deterministic within one software environment.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, TypeAlias

import colour
import numpy as np
from numpy.typing import NDArray

from camsim._version import __version__ as camsim_version
from camsim.core.color import ColorReference
from camsim.core.frames import RenderedImage
from camsim.metrics import (
    ChartLayout,
    PatchROI,
    edge_acutance,
    measure_all,
    slanted_edge_mtf,
    snr_channels,
    snr_patch,
    snr_temporal,
)
from camsim.metrics.color import _delta_e76, _delta_e2000
from camsim.optics.propagation import apply_psf
from camsim.optics.psf import airy_psf, box_psf, gaussian_psf, psf_mtf
from camsim.scene.charts import colorchecker_layout
from camsim.scene.targets import slanted_edge_target

__all__ = ["measure_g4"]

G4_KIND: Final[str] = "camsim G4 metrics validation"
SCHEMA_VERSION: Final[str] = "1.0"

# Reference sensor geometry for the analytic PSF cases: the shipped flagship
# preset's 1.12 um pitch at the photopic 550 nm reference wavelength.
PIXEL_PITCH_UM: Final[float] = 1.12
WAVELENGTH_NM: Final[float] = 550.0

# The primary edge tilt is ISO 12233's conventional 5 degrees; the robustness
# sweep covers the 5-20 degree range common on printed and synthetic charts.
PRIMARY_ANGLE_DEG: Final[float] = 5.0
EDGE_ANGLES_DEG: Final[tuple[float, ...]] = (5.0, 10.0, 15.0, 20.0)

# Eight samples per pixel removes the ~2 % near-diffraction-limited MTF50 bias
# that four samples per pixel carries; the gate cases use eight.
MTF_OVERSAMPLING: Final[int] = 8
FULL_MTF_SHAPE: Final[tuple[int, int]] = (256, 256)
QUICK_MTF_SHAPE: Final[tuple[int, int]] = (128, 128)

# Design tolerances from the G4 task brief.
MTF_RELATIVE_TOLERANCE: Final[float] = 0.02
# A reference crossing below _MTF_ABSOLUTE_FLOOR_CP would make a pure relative
# tolerance meaningless, so the gate budget is the 2 % relative allowance plus a
# 1e-3 cycles/pixel absolute floor: |measured - reference| <= 0.02 * |reference|
# + 1e-3. The relative error is still reported for readability.
MTF_ABSOLUTE_FLOOR_CP: Final[float] = 1e-3
ANGLE_SPREAD_TOLERANCE: Final[float] = 0.05
NOISE_ROBUSTNESS_TOLERANCE: Final[float] = 0.05
NOISE_SIGMA: Final[float] = 0.01
CURVE_TOLERANCE: Final[float] = 1e-9
SNR_TOLERANCE_DB: Final[float] = 0.5
FPN_REJECTION_MIN_DB: Final[float] = 5.0
DELTA_E_TOLERANCE: Final[float] = 1e-9
IDENTITY_TOLERANCE: Final[float] = 1e-12
SHARMA_PUBLISHED_TOLERANCE: Final[float] = 1e-4

# A box transfer function is |sinc(width * f)|: it reaches its first zero at
# 1 / width cycles/pixel and then carries a sidelobe, so monotonicity is
# enforced only over the main lobe, exactly as the Airy oscillation is
# documented rather than hidden.
BOX_MAIN_LOBE_LEVEL: Final[float] = 0.02

# Synthetic flat-patch SNR cases. The three means span a typical colour chart;
# the noise sigma is a moderate, easily measured level in linear light.
SPATIAL_MEANS: Final[tuple[float, float, float]] = (0.30, 0.52, 0.18)
SPATIAL_SIGMA: Final[float] = 0.02
FULL_SNR_PATCH: Final[int] = 64
QUICK_SNR_PATCH: Final[int] = 48

TEMPORAL_PATCH: Final[int] = 32
QUICK_TEMPORAL_PATCH: Final[int] = 24
FULL_TEMPORAL_FRAMES: Final[int] = 32
QUICK_TEMPORAL_FRAMES: Final[int] = 16
FIXED_PATTERN_SIGMA: Final[float] = 0.05
TEMPORAL_SIGMA: Final[float] = 0.01
TEMPORAL_MEAN: Final[float] = 0.5

# Synthetic ColorChecker-like chart. The perturbation makes the delta-E heatmap
# show real per-patch structure instead of a uniform numerical floor.
FULL_CHART_SHAPE: Final[tuple[int, int]] = (96, 128)
QUICK_CHART_SHAPE: Final[tuple[int, int]] = (64, 96)
CHART_OVERSAMPLING: Final[int] = 2
CHART_NOISE_SIGMA: Final[float] = 0.003
# Some reference colours (notably the sRGB cyan patch) land at an encoded
# channel value of exactly zero. A channel sitting on the display floor has a
# mean comparable to the added noise and a meaningless SNR, so every encoded
# channel is lifted to this small floor; the chart is a measurement vehicle,
# not a colour-accuracy target.
CHART_DISPLAY_FLOOR: Final[float] = 0.02
CHART_PERTURBATION: Final[tuple[float, float, float]] = (2.0, 1.6, 1.4)
EDGE_LAYOUT_SHAPE: Final[int] = 128
EDGE_LAYOUT_MARGIN: Final[int] = 16

# Acutance-versus-blur gallery sweep. Zero is the unblurred binary edge.
ACUTANCE_SIGMAS_PX: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
FULL_ACUTANCE_SHAPE: Final[int] = 192
QUICK_ACUTANCE_SHAPE: Final[int] = 128

REC709_WEIGHTS: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)
D65_XY: Final[NDArray[np.float64]] = np.array([0.3127, 0.3290], dtype=np.float64)

# Fixed seeds. Each measurement owns its generator, so changing one does not
# perturb another and repeated calls are bit-identical.
_MTF_NOISE_SEED: Final[int] = 41
_SNR_SPATIAL_SEED: Final[int] = 42
_SNR_TEMPORAL_SEED: Final[int] = 43
_CHART_NOISE_SEED: Final[int] = 44

_FORBIDDEN_IMPORTS: Final[tuple[str, ...]] = (
    "camsim.sensor",
    "camsim.isp",
    "camsim.scene",
    "camsim.optics",
)

# The published CIEDE2000 test data of Sharma, Wu, and Dalal (2005), printed to
# four decimals in the paper. Each row is (Lab1, Lab2, published delta E00).
_LabTriple: TypeAlias = tuple[float, float, float]
_SharmaPair: TypeAlias = tuple[_LabTriple, _LabTriple, float]
_ConstructedPair: TypeAlias = tuple[str, str, _LabTriple, _LabTriple]

_SHARMA_PAIRS: Final[tuple[_SharmaPair, ...]] = (
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -1.1848, -84.8006), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -0.9009, -85.5211), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, -1.0, 2.0), (50.0, 0.0, 0.0), 2.3669),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0009), 7.1792),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.001), 7.1792),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0011), 7.2195),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0012), 7.2195),
    ((50.0, -0.001, 2.49), (50.0, 0.0009, -2.49), 4.8045),
    ((50.0, -0.001, 2.49), (50.0, 0.001, -2.49), 4.8045),
    ((50.0, -0.001, 2.49), (50.0, 0.0011, -2.49), 4.7461),
    ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
    ((50.0, 2.5, 0.0), (56.0, -27.0, -3.0), 31.9030),
    ((50.0, 2.5, 0.0), (58.0, 24.0, 15.0), 19.4535),
    ((50.0, 2.5, 0.0), (50.0, 3.1736, 0.5854), 1.0000),
    ((50.0, 2.5, 0.0), (50.0, 3.2972, 0.0), 1.0000),
    ((50.0, 2.5, 0.0), (50.0, 1.8634, 0.5757), 1.0000),
    ((50.0, 2.5, 0.0), (50.0, 3.2592, 0.335), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
)

# Constructed pairs that exercise the branches a published table alone does not
# isolate: neutral colors (undefined hue), lightness-only differences, and hue
# rotations around the blue-region rotation term.
_CONSTRUCTED_PAIRS: Final[tuple[_ConstructedPair, ...]] = (
    ("Neutral lightness", "neutral", (20.0, 0.0, 0.0), (80.0, 0.0, 0.0)),
    ("Neutral step", "neutral", (50.0, 0.0, 0.0), (55.0, 0.0, 0.0)),
    ("Lightness inversion", "lightness", (95.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
    ("Lightness offset", "lightness", (20.0, 10.0, 10.0), (80.0, 10.0, 10.0)),
    ("Hue rotation 1", "hue rotation", (50.0, 30.0, 0.0), (50.0, 0.0, 30.0)),
    ("Hue rotation 2", "hue rotation", (50.0, 20.0, -20.0), (50.0, -20.0, 20.0)),
    ("Hue rotation 3", "hue rotation", (60.0, 40.0, 10.0), (60.0, -30.0, 25.0)),
    ("Identity neutral", "identity", (50.0, 0.0, 0.0), (50.0, 0.0, 0.0)),
    ("Identity chromatic", "identity", (50.0, 20.0, -30.0), (50.0, 20.0, -30.0)),
)


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
    inclusive_maximum: bool = False,
    minimum: float | None = None,
    target: float | None = None,
    relative: float | None = None,
    absolute: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Return one serializable gate check with an honest verdict.

    A ``maximum`` is an upper bound (strict unless ``inclusive_maximum`` is
    set), a ``minimum`` a strict lower bound, and a ``target`` uses
    ``math.isclose`` with the supplied relative or absolute tolerance. The
    display strings are stored in the JSON so the Markdown report and the
    report-consistency test compare exactly.
    """
    value = float(measured)
    if not math.isfinite(value):
        raise ValueError(f"check {name!r} measured a non-finite value: {value!r}")
    tolerance: dict[str, float] = {}
    passed = True
    if maximum is not None:
        tolerance["maximum"] = float(maximum)
        if inclusive_maximum:
            passed = passed and value <= float(maximum)
        else:
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
    tolerance_text = _format_tolerance(tolerance)
    if inclusive_maximum and "maximum" in tolerance:
        tolerance_text = f"<= {_format_number(tolerance['maximum'])}"
    return {
        "name": name,
        "measured": value,
        "tolerance": tolerance,
        "pass": bool(passed),
        "notes": notes,
        "measured_text": _format_number(value),
        "tolerance_text": tolerance_text,
    }


def _optional_float(value: float | None) -> float | None:
    """Return a Python float or ``None`` when the value is NaN or absent."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _crossing(frequencies: NDArray[np.float64], mtf: NDArray[np.float64], level: float) -> float:
    """Return the first frequency where ``mtf`` falls below ``level``.

    The crossing is linearly interpolated between the bracketing samples. NaN
    means "not reached inside the measured band", which is the honest answer for
    a system sharper than Nyquist rather than an invented crossing.
    """
    below = np.flatnonzero(mtf < level)
    if below.size == 0:
        return math.nan
    index = int(below[0])
    if index == 0:
        return math.nan
    lower_mtf = float(mtf[index - 1])
    upper_mtf = float(mtf[index])
    lower_frequency = float(frequencies[index - 1])
    upper_frequency = float(frequencies[index])
    if upper_mtf == lower_mtf:
        return upper_frequency
    fraction = (level - lower_mtf) / (upper_mtf - lower_mtf)
    return lower_frequency + (upper_frequency - lower_frequency) * fraction


def _reference_crossings(psf: NDArray[np.float64]) -> tuple[float, float]:
    """Return MTF50 and MTF20 of a discrete PSF on a dense frequency grid.

    ``psf_mtf`` evaluates the exact discrete-time transform at each frequency;
    the 20,001-sample grid puts the linear-interpolation error far below the
    0.1 % level that a 2 % gate would tolerate.
    """
    frequencies = np.linspace(0.0, 0.5, 20001, dtype=np.float64)
    mtf = psf_mtf(psf, frequencies)
    return _crossing(frequencies, mtf, 0.5), _crossing(frequencies, mtf, 0.2)


def _relative_error(reference: float, measured: float) -> float:
    """Return ``(measured - reference) / |reference|`` for reporting.

    The gate itself uses the absolute budget
    ``MTF_RELATIVE_TOLERANCE * |reference| + MTF_ABSOLUTE_FLOOR_CP``; this
    relative value is kept for the report table and is only meaningful when the
    reference is comfortably above zero.
    """
    return (measured - reference) / abs(reference)


def _mtf_tolerance(reference: float) -> float:
    """Return the absolute crossing budget: 2 % of the reference plus 1e-3 cp."""
    return MTF_RELATIVE_TOLERANCE * abs(reference) + MTF_ABSOLUTE_FLOOR_CP


def _odd_at_least(minimum: int) -> int:
    """Return the smallest odd integer greater than or equal to ``minimum``."""
    return minimum if minimum % 2 else minimum + 1


def _luma(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return Rec. 709 luma for a trailing-axis RGB array."""
    return np.asarray(
        REC709_WEIGHTS[0] * image[..., 0]
        + REC709_WEIGHTS[1] * image[..., 1]
        + REC709_WEIGHTS[2] * image[..., 2],
        dtype=np.float64,
    )


def _snr_db(mean: float, std: float) -> float:
    """Return ``20 log10(mean / std)``; caller guarantees positive values."""
    return float(20.0 * math.log10(mean / std))


def _measure_mtf(shape: tuple[int, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure the analytic PSF cases and their robustness checks."""
    specs: list[dict[str, Any]] = [
        {
            "key": "gaussian_sigma_1",
            "name": "Gaussian sigma=1 px",
            "short": "Gaussian",
            "psf": gaussian_psf(1.0, size=31),
            "monotonic_band": "full band",
        },
        {
            "key": "box_width_4",
            "name": "Box width=4 px",
            "short": "Box",
            "psf": box_psf(4.0, size=9),
            "monotonic_band": "main lobe",
        },
        {
            "key": "airy_f4",
            "name": "Airy f/4",
            "short": "Airy f/4",
            "psf": airy_psf(4.0, WAVELENGTH_NM, PIXEL_PITCH_UM),
            "monotonic_band": "excluded (oscillatory)",
        },
        {
            "key": "airy_f8",
            "name": "Airy f/8",
            "short": "Airy f/8",
            "psf": airy_psf(8.0, WAVELENGTH_NM, PIXEL_PITCH_UM),
            "monotonic_band": "excluded (oscillatory)",
        },
    ]

    cases: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []
    blurred_edges: dict[str, NDArray[np.float64]] = {}
    checks: list[dict[str, Any]] = []

    for spec in specs:
        psf = spec["psf"]
        reference_50, reference_20 = _reference_crossings(psf)
        results: dict[float, Any] = {}
        for angle in EDGE_ANGLES_DEG:
            edge = slanted_edge_target(shape, angle_deg=angle)
            blurred = apply_psf(edge[..., None], psf)[..., 0]
            if angle == PRIMARY_ANGLE_DEG:
                # The noise check reuses the primary-angle edge, so keep only
                # that one instead of the last loop iteration's 20 degree edge.
                blurred_edges[spec["key"]] = blurred
            results[angle] = slanted_edge_mtf(
                blurred,
                PIXEL_PITCH_UM,
                oversampling=MTF_OVERSAMPLING,
            )
        primary = results[PRIMARY_ANGLE_DEG]
        frequencies = np.asarray(primary.curve.frequency_cp, dtype=np.float64)
        measured_mtf = np.asarray(primary.curve.mtf, dtype=np.float64)
        analytic_mtf = np.asarray(psf_mtf(psf, frequencies), dtype=np.float64)
        error_50 = _relative_error(reference_50, float(primary.mtf50_cp))
        error_20 = _relative_error(reference_20, float(primary.mtf20_cp))
        absolute_error_50 = abs(float(primary.mtf50_cp) - reference_50)
        absolute_error_20 = abs(float(primary.mtf20_cp) - reference_20)
        tolerance_50 = _mtf_tolerance(reference_50)
        tolerance_20 = _mtf_tolerance(reference_20)

        cases.append(
            {
                "key": spec["key"],
                "name": spec["name"],
                "short": spec["short"],
                "psf_shape": [int(psf.shape[0]), int(psf.shape[1])],
                "reference_mtf50_cp": float(reference_50),
                "reference_mtf20_cp": float(reference_20),
                "measured_mtf50_cp": float(primary.mtf50_cp),
                "measured_mtf20_cp": float(primary.mtf20_cp),
                "relative_error_mtf50": float(error_50),
                "relative_error_mtf20": float(error_20),
                "absolute_error_mtf50_cp": float(absolute_error_50),
                "absolute_error_mtf20_cp": float(absolute_error_20),
                "tolerance_mtf50_cp": float(tolerance_50),
                "tolerance_mtf20_cp": float(tolerance_20),
                "within_tolerance_mtf50": bool(absolute_error_50 <= tolerance_50),
                "within_tolerance_mtf20": bool(absolute_error_20 <= tolerance_20),
                "measured_edge_angle_deg": float(primary.edge_angle_deg),
                "frequency_cp": [float(value) for value in frequencies],
                "measured_mtf": [float(value) for value in measured_mtf],
                "analytic_mtf": [float(value) for value in analytic_mtf],
            }
        )

        values = [float(results[angle].mtf50_cp) for angle in EDGE_ANGLES_DEG]
        spread = (max(values) - min(values)) / float(np.mean(values))
        angle_rows.append(
            {
                "case": spec["short"],
                "key": spec["key"],
                "angles_deg": [float(angle) for angle in EDGE_ANGLES_DEG],
                "mtf50_cp": values,
                "measured_angles_deg": [
                    float(results[angle].edge_angle_deg) for angle in EDGE_ANGLES_DEG
                ],
                "spread_fraction": float(spread),
            }
        )

        # Normalization and monotonicity. Gaussian is passive and monotone over
        # the whole band; the width-4 box reaches a zero at 0.25 cycles/pixel
        # and then carries a sidelobe, so only its main lobe is monotone.
        dc = float(measured_mtf[0])
        if spec["monotonic_band"] == "full band":
            band = measured_mtf
        elif spec["monotonic_band"] == "main lobe":
            below = np.flatnonzero(measured_mtf < BOX_MAIN_LOBE_LEVEL)
            end = int(below[0]) + 1 if below.size else int(measured_mtf.size)
            band = measured_mtf[:end]
        else:
            band = None
        max_rise = float(np.max(np.diff(band))) if band is not None and band.size > 1 else 0.0
        normalized.append(
            {
                "case": spec["short"],
                "key": spec["key"],
                "dc": dc,
                "min": float(measured_mtf.min()),
                "max": float(measured_mtf.max()),
                "monotonic_band": spec["monotonic_band"],
                "max_rise": max_rise,
            }
        )

        for label, absolute_error, tolerance in (
            ("MTF50", absolute_error_50, tolerance_50),
            ("MTF20", absolute_error_20, tolerance_20),
        ):
            checks.append(
                _check(
                    f"{label} vs analytic {spec['name']} (absolute error, cp)",
                    float(absolute_error),
                    maximum=float(tolerance),
                    inclusive_maximum=True,
                    notes=(
                        "Slanted-edge e-SFR at 5 degrees and 8x analysis oversampling "
                        "against psf_mtf evaluated exactly on the discrete PSF "
                        "kernel; budget = 2 % of the reference plus a 1e-3 "
                        "cycles/pixel absolute floor."
                    ),
                )
            )
        checks.append(
            _check(
                f"MTF50 angle spread, {spec['short']} (relative)",
                float(spread),
                maximum=ANGLE_SPREAD_TOLERANCE,
                notes=(
                    "Measured MTF50 at 5, 10, 15, and 20 degrees; spread is "
                    "(max - min) / mean over the four estimates."
                ),
            )
        )

    # Noise robustness: additive Gaussian noise on the clean Gaussian edge.
    gaussian_blurred = blurred_edges["gaussian_sigma_1"]
    clean = slanted_edge_mtf(
        gaussian_blurred,
        PIXEL_PITCH_UM,
        oversampling=MTF_OVERSAMPLING,
    )
    rng = np.random.default_rng(_MTF_NOISE_SEED)
    noisy = gaussian_blurred + NOISE_SIGMA * rng.standard_normal(gaussian_blurred.shape)
    noisy_result = slanted_edge_mtf(
        noisy,
        PIXEL_PITCH_UM,
        oversampling=MTF_OVERSAMPLING,
    )
    noise_change = (float(noisy_result.mtf50_cp) - float(clean.mtf50_cp)) / float(clean.mtf50_cp)
    checks.append(
        _check(
            "MTF50 noise robustness, Gaussian (relative change)",
            abs(float(noise_change)),
            maximum=NOISE_ROBUSTNESS_TOLERANCE,
            notes=(
                f"Gaussian noise sigma={NOISE_SIGMA:g} added to the blurred edge; "
                "the MTF50 must stay within 5 % of the clean estimate."
            ),
        )
    )

    # Airy f/2.8 is a diagnostic: its MTF does not reach 0.20 inside Nyquist,
    # and its MTF50 sits close enough to the band edge that the estimate is
    # noisier than the gate cases.
    diagnostic_psf = airy_psf(2.8, WAVELENGTH_NM, PIXEL_PITCH_UM)
    diagnostic_reference_50, diagnostic_reference_20 = _reference_crossings(diagnostic_psf)
    diagnostic_edge = slanted_edge_target(shape, angle_deg=PRIMARY_ANGLE_DEG)
    diagnostic_blurred = apply_psf(diagnostic_edge[..., None], diagnostic_psf)[..., 0]
    diagnostic_result = slanted_edge_mtf(
        diagnostic_blurred,
        PIXEL_PITCH_UM,
        oversampling=MTF_OVERSAMPLING,
    )
    diagnostics = [
        {
            "key": "airy_f2.8",
            "name": "Airy f/2.8 (diagnostic)",
            "mtf50_cp": float(diagnostic_result.mtf50_cp),
            "mtf20_cp": _optional_float(diagnostic_result.mtf20_cp),
            "reference_mtf50_cp": float(diagnostic_reference_50),
            "reference_mtf20_cp": _optional_float(diagnostic_reference_20),
            "note": (
                "Diagnostic only, not a gate case: Airy f/2.8 never reaches "
                "MTF20 inside Nyquist and its MTF50 is closest to the band "
                "edge, where the ESF estimator and Hamming window carry more bias."
            ),
        }
    ]

    max_dc_error = max(abs(row["dc"] - 1.0) for row in normalized)
    max_out_of_range = max(max(row["max"] - 1.0, -row["min"], 0.0) for row in normalized)
    checks.append(
        _check(
            "MTF curve DC normalization (max abs error)",
            float(max_dc_error),
            maximum=CURVE_TOLERANCE,
            notes="Every measured curve has mtf[0] equal to one by construction.",
        )
    )
    checks.append(
        _check(
            "MTF curve range [0, 1] (max violation)",
            float(max_out_of_range),
            maximum=CURVE_TOLERANCE,
            notes=(
                "A passive system cannot amplify modulation; the estimator clips "
                "noise/window excursions to the physical range."
            ),
        )
    )
    gaussian_row = next(row for row in normalized if row["key"] == "gaussian_sigma_1")
    box_row = next(row for row in normalized if row["key"] == "box_width_4")
    checks.append(
        _check(
            "MTF curve monotonicity, Gaussian full band (max rise)",
            float(gaussian_row["max_rise"]),
            maximum=CURVE_TOLERANCE,
            notes="A Gaussian transfer function is monotone non-increasing over [0, 0.5].",
        )
    )
    checks.append(
        _check(
            "MTF curve monotonicity, box main lobe (max rise)",
            float(box_row["max_rise"]),
            maximum=CURVE_TOLERANCE,
            notes=(
                "A width-4 box transfer function is |sinc(4 f)|, which has a zero "
                "at 0.25 cycles/pixel and a sidelobe; monotonicity is enforced on "
                "the main lobe, and the sidelobe is documented rather than hidden."
            ),
        )
    )

    section = {
        "cases": cases,
        "angle_robustness": angle_rows,
        "noise": {
            "sigma": NOISE_SIGMA,
            "clean_mtf50_cp": float(clean.mtf50_cp),
            "noisy_mtf50_cp": float(noisy_result.mtf50_cp),
            "relative_change": float(noise_change),
        },
        "normalization": {
            "cases": normalized,
            "max_dc_error": float(max_dc_error),
            "max_out_of_range": float(max_out_of_range),
        },
        "diagnostics": diagnostics,
    }
    return section, checks


def _measure_snr(quick: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure spatial, temporal, and mono SNR against known synthetic noise."""
    checks: list[dict[str, Any]] = []

    patch_size = QUICK_SNR_PATCH if quick else FULL_SNR_PATCH
    rng = np.random.default_rng(_SNR_SPATIAL_SEED)
    image = np.empty((patch_size, patch_size, 3), dtype=np.float64)
    for index, mean in enumerate(SPATIAL_MEANS):
        image[..., index] = mean + SPATIAL_SIGMA * rng.standard_normal((patch_size, patch_size))
    roi = PatchROI(name="flat", x=0, y=0, width=patch_size, height=patch_size)
    results = snr_channels(image, roi)

    measurements: list[dict[str, Any]] = []
    per_channel_errors: list[float] = []
    consistency_errors: list[float] = []
    for index, channel in enumerate(("R", "G", "B")):
        patch = image[..., index]
        # The reference is the generating mean and sigma, not the estimator's
        # own patch statistics. Using the sample statistics would make the
        # error identically zero and prove nothing about noise recovery.
        expected = _snr_db(SPATIAL_MEANS[index], SPATIAL_SIGMA)
        sample_expected = _snr_db(float(patch.mean()), float(patch.std(ddof=1)))
        result = results[channel]
        error = float(result.snr_db) - expected
        consistency_error = float(result.snr_db) - sample_expected
        per_channel_errors.append(abs(error))
        consistency_errors.append(abs(consistency_error))
        measurements.append(
            {
                "channel": channel,
                "mean": float(result.mean),
                "std": float(result.std),
                "snr_db": float(result.snr_db),
                "expected_snr_db": float(expected),
                "error_db": error,
                "valid": bool(result.valid),
                "reference_mean": float(SPATIAL_MEANS[index]),
                "reference_sigma": float(SPATIAL_SIGMA),
                "sample_expected_snr_db": float(sample_expected),
                "sample_consistency_error_db": float(consistency_error),
            }
        )
    # Each channel's noise is drawn independently, so the Rec. 709 luma noise
    # variance is the weighted sum of the per-channel variances. The reference
    # mean and sigma therefore come from the generating constants, not from the
    # realized luma plane.
    luma = _luma(image)
    luma_reference_mean = sum(
        weight * mean for weight, mean in zip(REC709_WEIGHTS, SPATIAL_MEANS, strict=True)
    )
    luma_reference_sigma = math.sqrt(
        sum((weight * SPATIAL_SIGMA) ** 2 for weight in REC709_WEIGHTS)
    )
    luma_expected = _snr_db(luma_reference_mean, luma_reference_sigma)
    luma_sample_expected = _snr_db(float(luma.mean()), float(luma.std(ddof=1)))
    luma_result = results["luma"]
    luma_error = float(luma_result.snr_db) - luma_expected
    luma_consistency_error = float(luma_result.snr_db) - luma_sample_expected
    consistency_errors.append(abs(luma_consistency_error))
    measurements.append(
        {
            "channel": "luma",
            "mean": float(luma_result.mean),
            "std": float(luma_result.std),
            "snr_db": float(luma_result.snr_db),
            "expected_snr_db": float(luma_expected),
            "error_db": float(luma_error),
            "valid": bool(luma_result.valid),
            "reference_mean": float(luma_reference_mean),
            "reference_sigma": float(luma_reference_sigma),
            "sample_expected_snr_db": float(luma_sample_expected),
            "sample_consistency_error_db": float(luma_consistency_error),
        }
    )

    max_channel_error = max(per_channel_errors)
    checks.append(
        _check(
            "SNR spatial per-channel agreement (max abs error, dB)",
            float(max_channel_error),
            maximum=SNR_TOLERANCE_DB,
            notes=(
                "snr_patch per R/G/B channel against 20 log10(mu / sigma) using the "
                "generating means and noise sigma; channel noise draws are "
                "independent."
            ),
        )
    )
    checks.append(
        _check(
            "SNR spatial luma agreement (abs error, dB)",
            abs(float(luma_error)),
            maximum=SNR_TOLERANCE_DB,
            notes=(
                "Rec. 709 luma in linear light against the generating luma mean "
                "0.2126 mu_R + 0.7152 mu_G + 0.0722 mu_B and the quadrature "
                "weighted sigma, since the channel noise draws are independent."
            ),
        )
    )

    # Temporal stack: a static per-pixel fixed pattern plus independent noise.
    temporal_size = QUICK_TEMPORAL_PATCH if quick else TEMPORAL_PATCH
    frames = QUICK_TEMPORAL_FRAMES if quick else FULL_TEMPORAL_FRAMES
    temporal_rng = np.random.default_rng(_SNR_TEMPORAL_SEED)
    fixed_pattern = FIXED_PATTERN_SIGMA * temporal_rng.standard_normal(
        (temporal_size, temporal_size)
    )
    stack = (
        TEMPORAL_MEAN
        + fixed_pattern[None, :, :]
        + TEMPORAL_SIGMA * temporal_rng.standard_normal((frames, temporal_size, temporal_size))
    )
    temporal_roi = PatchROI(
        name="temporal",
        x=0,
        y=0,
        width=temporal_size,
        height=temporal_size,
    )
    temporal = snr_temporal(stack, temporal_roi)
    # Reference the generating mean and temporal sigma, never the realized
    # stack mean, so the check measures noise recovery rather than restating the
    # estimator's own arithmetic.
    temporal_expected = _snr_db(TEMPORAL_MEAN, TEMPORAL_SIGMA)
    temporal_error = float(temporal.snr_db) - temporal_expected
    temporal_sample_expected = _snr_db(float(temporal.mean), float(temporal.std))
    temporal_consistency_error = float(temporal.snr_db) - temporal_sample_expected
    consistency_errors.append(abs(temporal_consistency_error))
    spatial_single = snr_patch(stack[0], temporal_roi, channel="mono")
    fpn_rejection = float(temporal.snr_db) - float(spatial_single.snr_db)
    checks.append(
        _check(
            "SNR temporal vs known noise (abs error, dB)",
            abs(float(temporal_error)),
            maximum=SNR_TOLERANCE_DB,
            notes=(
                f"Pooled per-pixel temporal variance over {frames} frames with a "
                f"static fixed pattern (sigma={FIXED_PATTERN_SIGMA:g}) and temporal "
                f"noise (sigma={TEMPORAL_SIGMA:g}); the reference is "
                f"20 log10({TEMPORAL_MEAN:g} / {TEMPORAL_SIGMA:g}) from the "
                "generating constants, not the measured stack mean."
            ),
        )
    )
    checks.append(
        _check(
            "SNR temporal fixed-pattern rejection (temporal minus spatial, dB)",
            float(fpn_rejection),
            minimum=FPN_REJECTION_MIN_DB,
            notes=(
                "The temporal estimator must exclude the static fixed pattern, so "
                "it must exceed the single-frame spatial SNR by a material margin."
            ),
        )
    )

    # Mono path: the same flat field without a CFA, measured as one plane. The
    # plane is the synthetic R channel, so its generating constants are the
    # reference; the sample-statistics version is kept only as a diagnostic.
    mono_result = snr_patch(image[..., 0], roi, channel="mono")
    mono_expected = _snr_db(SPATIAL_MEANS[0], SPATIAL_SIGMA)
    mono_sample_expected = _snr_db(
        float(image[..., 0].mean()),
        float(image[..., 0].std(ddof=1)),
    )
    mono_error = float(mono_result.snr_db) - mono_expected
    mono_consistency_error = float(mono_result.snr_db) - mono_sample_expected
    consistency_errors.append(abs(mono_consistency_error))
    mono_channels = list(snr_channels(image[..., 0], roi))
    checks.append(
        _check(
            "SNR mono path agreement (abs error, dB)",
            abs(float(mono_error)),
            maximum=SNR_TOLERANCE_DB,
            notes=(
                "A 2-D mono patch reports one 'mono' channel with the same "
                "spatial statistics; its reference is the generating R-channel "
                "mean and sigma, and no colour channel is invented."
            ),
        )
    )

    section = {
        "spatial": {
            "shape": [patch_size, patch_size],
            "means": [float(value) for value in SPATIAL_MEANS],
            "noise_sigma": SPATIAL_SIGMA,
            "luma_reference_mean": float(luma_reference_mean),
            "luma_reference_sigma": float(luma_reference_sigma),
            "measurements": measurements,
        },
        "temporal": {
            "frames": int(frames),
            "patch_shape": [temporal_size, temporal_size],
            "fixed_pattern_sigma": FIXED_PATTERN_SIGMA,
            "temporal_sigma": TEMPORAL_SIGMA,
            "reference_mean": TEMPORAL_MEAN,
            "reference_sigma": TEMPORAL_SIGMA,
            "mean": float(temporal.mean),
            "std": float(temporal.std),
            "snr_db": float(temporal.snr_db),
            "expected_snr_db": float(temporal_expected),
            "error_db": float(temporal_error),
            "sample_expected_snr_db": float(temporal_sample_expected),
            "sample_consistency_error_db": float(temporal_consistency_error),
            "valid": bool(temporal.valid),
            "spatial_single_frame_snr_db": float(spatial_single.snr_db),
            "fpn_rejection_db": float(fpn_rejection),
        },
        "mono": {
            "channel": mono_result.channel,
            "mean": float(mono_result.mean),
            "std": float(mono_result.std),
            "snr_db": float(mono_result.snr_db),
            "expected_snr_db": float(mono_expected),
            "error_db": float(mono_error),
            "reference_mean": float(SPATIAL_MEANS[0]),
            "reference_sigma": float(SPATIAL_SIGMA),
            "sample_expected_snr_db": float(mono_sample_expected),
            "sample_consistency_error_db": float(mono_consistency_error),
            "valid": bool(mono_result.valid),
            "channels": mono_channels,
        },
        "consistency": {
            "max_abs_error_db": float(max(consistency_errors)),
            "note": (
                "Estimator against the same patch's sample mean and standard "
                "deviation. This is zero by construction; it is reported as a "
                "diagnostic and is not the gate reference. The gate compares "
                "against the generating means and sigmas."
            ),
        },
    }
    return section, checks


def _delta_e_pair_rows() -> list[dict[str, Any]]:
    """Return every colour pair with its published source and expected value."""
    rows: list[dict[str, Any]] = []
    for index, (first, second, published) in enumerate(_SHARMA_PAIRS, start=1):
        rows.append(
            {
                "name": f"Sharma {index:02d}",
                "source": "Sharma",
                "lab1": [float(value) for value in first],
                "lab2": [float(value) for value in second],
                "published_delta_e2000": float(published),
            }
        )
    for name, source, first, second in _CONSTRUCTED_PAIRS:
        rows.append(
            {
                "name": name,
                "source": source,
                "lab1": [float(value) for value in first],
                "lab2": [float(value) for value in second],
                "published_delta_e2000": None,
            }
        )
    return rows


def _measure_color() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare the in-house delta-E implementations with colour-science."""
    pairs = _delta_e_pair_rows()
    max_76 = 0.0
    max_2000 = 0.0
    max_published = 0.0
    identity_max = 0.0
    for pair in pairs:
        first = np.asarray(pair["lab1"], dtype=np.float64)
        second = np.asarray(pair["lab2"], dtype=np.float64)
        ours_76 = float(_delta_e76(first, second))
        ours_2000 = float(_delta_e2000(first, second))
        reference_76 = float(colour.delta_E(first, second, method="CIE 1976"))
        reference_2000 = float(colour.delta_E(first, second, method="CIE 2000"))
        pair["delta_e76"] = ours_76
        pair["reference_delta_e76"] = reference_76
        pair["delta_e2000"] = ours_2000
        pair["reference_delta_e2000"] = reference_2000
        max_76 = max(max_76, abs(ours_76 - reference_76))
        max_2000 = max(max_2000, abs(ours_2000 - reference_2000))
        if pair["published_delta_e2000"] is not None:
            max_published = max(
                max_published,
                abs(ours_2000 - float(pair["published_delta_e2000"])),
            )
        identity_max = max(
            identity_max,
            abs(float(_delta_e76(first, first))),
            abs(float(_delta_e2000(first, first))),
            abs(float(_delta_e76(second, second))),
            abs(float(_delta_e2000(second, second))),
        )

    checks = [
        _check(
            "DeltaE76 vs colour-science (max abs error)",
            float(max_76),
            maximum=DELTA_E_TOLERANCE,
            notes=(
                "Euclidean Lab distance on the Sharma, neutral, lightness, and "
                "hue-rotation pairs against colour.delta_E(method='CIE 1976')."
            ),
        ),
        _check(
            "DeltaE2000 vs colour-science (max abs error)",
            float(max_2000),
            maximum=DELTA_E_TOLERANCE,
            notes=(
                "CIEDE2000 with kL=kC=kH=1 against colour.delta_E(method='CIE 2000') "
                "on the same pair set."
            ),
        ),
        _check(
            "DeltaE identity (max abs error)",
            float(identity_max),
            maximum=IDENTITY_TOLERANCE,
            notes="A colour compared with itself must have exactly zero difference.",
        ),
        _check(
            "Sharma CIEDE2000 published pairs (max abs error)",
            float(max_published),
            maximum=SHARMA_PUBLISHED_TOLERANCE,
            notes=(
                "The 34 published Sharma et al. (2005) pairs printed to four "
                "decimals; this is an independent published-source check on both "
                "implementations."
            ),
        ),
    ]

    section = {
        "pair_count": len(pairs),
        "pairs": pairs,
        "max_abs_error_delta_e76": float(max_76),
        "max_abs_error_delta_e2000": float(max_2000),
        "sharma_published_max_abs_error_delta_e2000": float(max_published),
        "identity_max_abs": float(identity_max),
    }
    return section, checks


def _target_rgb_from_reference(reference: ColorReference) -> NDArray[np.float64]:
    """Convert reference Lab to linear sRGB under D65.

    The reference Lab is relative to the scene illuminant's white point. It is
    converted back to XYZ with that white, adapted to D65 with Bradford, and
    converted to linear sRGB. This mirrors the G3 colour path but is duplicated
    here so the synthetic chart generator is self-contained.
    """
    white_xy = np.asarray(reference.white_point_xy, dtype=np.float64)
    xyz = np.asarray(
        colour.Lab_to_XYZ(np.asarray(reference.lab, dtype=np.float64), illuminant=white_xy),
        dtype=np.float64,
    )
    adapted = np.asarray(
        colour.adaptation.chromatic_adaptation_VonKries(
            xyz,
            colour.xy_to_XYZ(white_xy),
            colour.xy_to_XYZ(D65_XY),
            transform="Bradford",
        ),
        dtype=np.float64,
    )
    return np.asarray(
        colour.XYZ_to_RGB(
            adapted,
            colourspace="sRGB",
            illuminant=D65_XY,
            apply_cctf_encoding=False,
        ),
        dtype=np.float64,
    )


def _synthetic_chart(layout: ChartLayout) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Build a synthetic display chart and its linear-light counterpart.

    Each patch is filled with the reference colour plus a deterministic Lab
    perturbation, encoded with the sRGB OETF, clipped to the display range, and
    lifted to a small display floor so no channel sits at zero with a
    meaningless noise-only SNR. A small display-domain noise field then gives
    every patch a nonzero standard deviation. The linear buffer is the EOTF
    decoding of the clipped display, so SNR measures the image that is actually
    displayed.
    """
    reference = layout.reference
    if reference is None:  # pragma: no cover - colorchecker_layout always binds one
        raise ValueError("the synthetic chart layout has no colour reference")
    lab = np.asarray(reference.lab, dtype=np.float64)
    index = np.arange(lab.shape[0], dtype=np.float64)
    perturbation = np.stack(
        [
            CHART_PERTURBATION[0] * np.sin(0.9 * index),
            CHART_PERTURBATION[1] * np.cos(1.3 * index),
            CHART_PERTURBATION[2] * np.sin(0.7 * index + 0.3),
        ],
        axis=1,
    )
    perturbed = ColorReference(
        lab=lab + perturbation,
        white_point_xy=reference.white_point_xy,
        illuminant_name=reference.illuminant_name,
    )
    linear_target = _target_rgb_from_reference(perturbed)
    display_target = np.asarray(
        colour.cctf_encoding(np.clip(linear_target, 0.0, 1.0), function="sRGB"),
        dtype=np.float64,
    )
    display_target = np.maximum(display_target, CHART_DISPLAY_FLOOR)

    height = max(patch.y + patch.height for patch in layout.patches)
    width = max(patch.x + patch.width for patch in layout.patches)
    image = np.zeros((height, width, 3), dtype=np.float64)
    for patch, patch_colour in zip(layout.patches, display_target, strict=True):
        image[patch.y : patch.y + patch.height, patch.x : patch.x + patch.width, :] = patch_colour

    rng = np.random.default_rng(_CHART_NOISE_SEED)
    noisy = np.clip(
        image + CHART_NOISE_SIGMA * rng.standard_normal(image.shape),
        0.0,
        1.0,
    )
    linear = np.asarray(colour.cctf_decoding(noisy, function="sRGB"), dtype=np.float64)
    return noisy.astype(np.float32), linear.astype(np.float32)


def _measure_all_sections(
    quick: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run ``measure_all`` on a synthetic chart, an edge layout, and a mono image."""
    chart_shape = QUICK_CHART_SHAPE if quick else FULL_CHART_SHAPE
    layout = colorchecker_layout(
        "D65",
        shape=chart_shape,
        oversampling=CHART_OVERSAMPLING,
        pixel_pitch_um=PIXEL_PITCH_UM,
    )
    display, linear = _synthetic_chart(layout)
    rendered = RenderedImage(
        data=display,
        color_space="sRGB",
        channels="RGB",
        linear_rgb=linear,
    )
    chart_report = measure_all(rendered, layout)
    if chart_report.color is None or chart_report.snr is None:  # pragma: no cover
        raise RuntimeError("the synthetic RGB chart did not produce colour and SNR groups")
    channel_names = list(next(iter(chart_report.snr.values())))
    snr_results = [
        result for channel_map in chart_report.snr.values() for result in channel_map.values()
    ]
    all_snr_valid = all(result.valid for result in snr_results)
    per_patch_snr: list[dict[str, Any]] = []
    for patch in layout.patches:
        channel_map = chart_report.snr[patch.name]
        row: dict[str, Any] = {"patch": patch.name}
        for channel in channel_names:
            row[channel] = _optional_float(float(channel_map[channel].snr_db))
        per_patch_snr.append(row)
    chart_section = {
        "shape": [int(chart_shape[0]), int(chart_shape[1])],
        "patch_count": len(layout.patches),
        "valid": bool(chart_report.color.valid),
        "mean_delta_e76": float(chart_report.color.delta_e76),
        "mean_delta_e2000": float(chart_report.color.delta_e2000),
        "patch_names": [patch.name for patch in layout.patches],
        "per_patch_delta_e76": [float(patch.delta_e76) for patch in chart_report.color.per_patch],
        "per_patch_delta_e2000": [
            float(patch.delta_e2000) for patch in chart_report.color.per_patch
        ],
        "snr_patch_count": len(chart_report.snr),
        "snr_channels": channel_names,
        "all_snr_valid": bool(all_snr_valid),
        "per_patch_snr_db": per_patch_snr,
        "mean_luma_snr_db": float(
            np.mean([channel_map["luma"].snr_db for channel_map in chart_report.snr.values()])
        ),
        "skipped": list((chart_report.metadata or {}).get("skipped", [])),
    }

    # A separate edge layout contains only an edge ROI; a perfectly sharp
    # synthetic step never crosses 50 % inside Nyquist, so the crossings are
    # legitimately null after serialization.
    edge_image = np.empty((EDGE_LAYOUT_SHAPE, EDGE_LAYOUT_SHAPE, 3), dtype=np.float32)
    edge_image[:, : EDGE_LAYOUT_SHAPE // 2, :] = 0.2
    edge_image[:, EDGE_LAYOUT_SHAPE // 2 :, :] = 0.8
    edge_layout = ChartLayout(
        name="slanted_edge",
        edge_roi=PatchROI(
            name="edge",
            x=EDGE_LAYOUT_MARGIN,
            y=EDGE_LAYOUT_MARGIN,
            width=EDGE_LAYOUT_SHAPE - 2 * EDGE_LAYOUT_MARGIN,
            height=EDGE_LAYOUT_SHAPE - 2 * EDGE_LAYOUT_MARGIN,
        ),
        pixel_pitch_um=PIXEL_PITCH_UM,
    )
    edge_rendered = RenderedImage(
        data=edge_image,
        color_space="sRGB",
        channels="RGB",
        linear_rgb=edge_image.copy(),
    )
    edge_report = measure_all(edge_rendered, edge_layout)
    if edge_report.mtf is None or edge_report.acutance is None:  # pragma: no cover
        raise RuntimeError("the edge layout did not produce MTF and acutance")
    edge_section = {
        "shape": [EDGE_LAYOUT_SHAPE, EDGE_LAYOUT_SHAPE],
        "mtf50_cp": _optional_float(edge_report.mtf.mtf50_cp),
        "mtf20_cp": _optional_float(edge_report.mtf.mtf20_cp),
        "acutance": float(edge_report.acutance),
        "skipped": list((edge_report.metadata or {}).get("skipped", [])),
    }

    # A MONO image has patches and a reference but no colour group.
    luma = _luma(display.astype(np.float64))
    mono_data = luma[..., None].astype(np.float32)
    mono = RenderedImage(
        data=mono_data,
        color_space="sRGB",
        channels="MONO",
        linear_rgb=mono_data.copy(),
    )
    mono_report = measure_all(mono, layout)
    if mono_report.snr is None:  # pragma: no cover
        raise RuntimeError("the MONO chart did not produce SNR")
    mono_channels = list(next(iter(mono_report.snr.values())))
    mono_section = {
        "color_present": mono_report.color is not None,
        "patch_count": len(mono_report.snr),
        "snr_channels": mono_channels,
        "valid": all(
            channel.valid
            for channel_map in mono_report.snr.values()
            for channel in channel_map.values()
        ),
        "skipped": list((mono_report.metadata or {}).get("skipped", [])),
    }

    # Strict serialization evidence. NaN crossings must become JSON null, and
    # the parsed mapping must equal to_dict() exactly.
    chart_json = chart_report.to_json()
    edge_json = edge_report.to_json()
    mono_json = mono_report.to_json()
    parsed_edge = json.loads(edge_json)
    serialization = {
        "roundtrip_ok": bool(json.loads(chart_json) == chart_report.to_dict()),
        "nan_token_absent": bool(
            "NaN" not in chart_json
            and "Infinity" not in chart_json
            and "NaN" not in edge_json
            and "Infinity" not in edge_json
        ),
        "null_crossing": bool(
            parsed_edge["mtf"]["mtf50_cp"] is None
            and parsed_edge["mtf"]["mtf20_cp"] is None
            and "null" in edge_json
        ),
        "mono_json_ok": bool(json.loads(mono_json)["color"] is None),
    }

    # Import independence: a fresh interpreter imports only the metrics layer.
    code = (
        "import camsim.metrics, sys; assert 'camsim.sensor' not in sys.modules; "
        "assert 'camsim.isp' not in sys.modules; assert 'camsim.scene' not in sys.modules; "
        "assert 'camsim.optics' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    independence = {
        "passed": bool(completed.returncode == 0),
        "forbidden_modules": list(_FORBIDDEN_IMPORTS),
        # Record only the interpreter's basename: an absolute virtualenv path
        # would bake a machine-specific location into the committed JSON.
        "command": f'{Path(sys.executable).name} -c "{code}"',
        "stderr": completed.stderr.strip(),
    }

    expected_channels = {"R", "G", "B", "luma"}
    channel_layouts_ok = all(
        set(channel_map) == expected_channels for channel_map in chart_report.snr.values()
    )
    chart_complete = (
        chart_report.color.valid
        and len(chart_report.color.per_patch) == len(layout.patches)
        and len(chart_report.snr) == len(layout.patches)
        and all_snr_valid
        and channel_layouts_ok
    )
    edge_complete = edge_report.mtf is not None and edge_report.acutance is not None
    mono_complete = mono_report.color is None and all(
        set(channel_map) == {"mono"} and all(result.valid for result in channel_map.values())
        for channel_map in mono_report.snr.values()
    )
    serialization_ok = all(serialization.values())
    checks = [
        _check(
            "measure_all RGB chart completeness (colour + SNR)",
            1.0 if chart_complete else 0.0,
            target=1.0,
            absolute=0.0,
            notes=(
                "A synthetic ColorChecker-like display built from the scene "
                "reference Lab returns 24 valid colour patches and per-patch SNR "
                "for R, G, B, and luma."
            ),
        ),
        _check(
            "measure_all edge layout completeness (MTF + acutance)",
            1.0 if edge_complete else 0.0,
            target=1.0,
            absolute=0.0,
            notes=(
                "A separate edge layout returns an MTF result (crossings may "
                "legitimately be null) and a finite acutance."
            ),
        ),
        _check(
            "measure_all MONO report (SNR without colour)",
            1.0 if mono_complete else 0.0,
            target=1.0,
            absolute=0.0,
            notes=(
                "A MONO RenderedImage returns one 'mono' SNR channel per patch and "
                "no colour group; the skip is recorded in metadata."
            ),
        ),
        _check(
            "measure_all JSON serialization (NaN -> null)",
            1.0 if serialization_ok else 0.0,
            target=1.0,
            absolute=0.0,
            notes=(
                "to_json parses with a strict reader, carries no NaN/Infinity "
                "tokens, maps the not-reached MTF crossings to null, and round "
                "trips to to_dict()."
            ),
        ),
        _check(
            "metrics import independence (forbidden modules absent)",
            1.0 if independence["passed"] else 0.0,
            target=1.0,
            absolute=0.0,
            notes=(
                "A fresh interpreter imports camsim.metrics and asserts that the "
                "sensor, ISP, scene, and optics packages are not loaded."
            ),
        ),
    ]

    section = {
        "chart": chart_section,
        "edge": edge_section,
        "mono": mono_section,
        "serialization": serialization,
        "import_independence": independence,
    }
    return section, checks


def _measure_acutance_sweep(quick: bool) -> dict[str, Any]:
    """Measure acutance of one slanted edge across a Gaussian blur sweep."""
    size = QUICK_ACUTANCE_SHAPE if quick else FULL_ACUTANCE_SHAPE
    edge = slanted_edge_target((size, size), angle_deg=PRIMARY_ANGLE_DEG)
    margin = max(8, size // 6)
    roi = PatchROI(
        name="edge",
        x=margin,
        y=margin,
        width=size - 2 * margin,
        height=size - 2 * margin,
    )
    values: list[float] = []
    for sigma in ACUTANCE_SIGMAS_PX:
        if sigma == 0.0:
            blurred = edge
        else:
            kernel_size = _odd_at_least(max(9, math.ceil(6.0 * sigma)))
            blurred = apply_psf(edge[..., None], gaussian_psf(sigma, size=kernel_size))[..., 0]
        values.append(float(edge_acutance(blurred, roi)))
    return {
        "shape": [size, size],
        "sigma_px": [float(value) for value in ACUTANCE_SIGMAS_PX],
        "acutance": values,
    }


def _environment() -> dict[str, str]:
    """Return the software environment used by the report."""
    import scipy  # type: ignore[import-untyped]

    return {
        "python": platform.python_version(),
        "camsim": camsim_version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "colour_science": colour.__version__,
        "platform": platform.system(),
    }


def measure_g4(*, quick: bool = False) -> dict[str, Any]:
    """Run the G4 metrics validation and return a JSON-serializable report.

    Parameters
    ----------
    quick:
        Use smaller MTF, SNR, chart, and acutance grids. Quick mode exercises
        every code path but is **not** gate evidence; the returned dictionary
        records ``gate_evidence=False``. The full run uses the committed gate
        sizes and is the source of ``results/reports/g4_metrics_results.json``.

    Returns
    -------
    dict[str, Any]
        A dictionary with ``kind``, ``gate_evidence``, ``quick``, ``status``,
        ``checks`` (each carrying ``name``, ``measured``, ``tolerance``,
        ``pass``, ``notes``, and the exact display strings used by the Markdown
        report), the environment, the seeds and sizes, and the raw measurement
        sections ``mtf``, ``snr``, ``color``, ``measure_all``, and
        ``acutance_sweep``. Every value is a Python float, int, bool, string,
        list, or ``None`` for a not-reached crossing, so
        ``json.dumps(..., allow_nan=False)`` succeeds.

    Notes
    -----
    All measurements are synthetic and deterministic: explicit
    :class:`numpy.random.Generator` seeds, no global random state, and no
    wall-clock or environment-dependent values except the software versions
    recorded in ``environment``. The metric under test and the analytic
    reference are independent code paths; the SNR gate checks use the
    **generating** means and sigmas rather than the estimator's own sample
    statistics. The ``snr.consistency`` section records the identically zero
    sample-statistics error as a diagnostic, not as gate evidence.
    """
    mtf_shape = QUICK_MTF_SHAPE if quick else FULL_MTF_SHAPE
    mtf, mtf_checks = _measure_mtf(mtf_shape)
    snr, snr_checks = _measure_snr(quick)
    color, color_checks = _measure_color()
    harness, harness_checks = _measure_all_sections(quick)
    acutance = _measure_acutance_sweep(quick)

    checks = [*mtf_checks, *snr_checks, *color_checks, *harness_checks]
    status = "PASS" if all(bool(check["pass"]) for check in checks) else "FAIL"

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": G4_KIND,
        "gate_evidence": not quick,
        "quick": quick,
        "status": status,
        "environment": _environment(),
        "config": {
            "quick": quick,
            "pixel_pitch_um": PIXEL_PITCH_UM,
            "wavelength_nm": WAVELENGTH_NM,
            "mtf_shape": [int(mtf_shape[0]), int(mtf_shape[1])],
            "mtf_primary_angle_deg": PRIMARY_ANGLE_DEG,
            "mtf_angles_deg": [float(angle) for angle in EDGE_ANGLES_DEG],
            "mtf_oversampling": MTF_OVERSAMPLING,
            "mtf_noise_sigma": NOISE_SIGMA,
            "snr_spatial_patch": QUICK_SNR_PATCH if quick else FULL_SNR_PATCH,
            "snr_temporal_patch": QUICK_TEMPORAL_PATCH if quick else TEMPORAL_PATCH,
            "snr_temporal_frames": QUICK_TEMPORAL_FRAMES if quick else FULL_TEMPORAL_FRAMES,
            "chart_shape": [
                int(QUICK_CHART_SHAPE[0] if quick else FULL_CHART_SHAPE[0]),
                int(QUICK_CHART_SHAPE[1] if quick else FULL_CHART_SHAPE[1]),
            ],
            "chart_oversampling": CHART_OVERSAMPLING,
            "chart_noise_sigma": CHART_NOISE_SIGMA,
            "acutance_shape": int(QUICK_ACUTANCE_SHAPE if quick else FULL_ACUTANCE_SHAPE),
        },
        "seeds": {
            "mtf_noise": _MTF_NOISE_SEED,
            "snr_spatial": _SNR_SPATIAL_SEED,
            "snr_temporal": _SNR_TEMPORAL_SEED,
            "chart_noise": _CHART_NOISE_SEED,
            "note": (
                "Every stochastic measurement uses its own numpy Generator; PSF "
                "targets and the analytic references are deterministic without RNG."
            ),
        },
        "mtf": mtf,
        "snr": snr,
        "color": color,
        "measure_all": harness,
        "acutance_sweep": acutance,
        "checks": checks,
    }
    # Serialize once here so a non-JSON-safe payload fails at the measurement
    # boundary, not later in the example or the report writer.
    json.dumps(payload, allow_nan=False)
    return payload
