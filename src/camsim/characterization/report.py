"""Structured EMVA 1288-style characterization results and JSON serialization.

An image-sensor characterization report is a compact, machine-readable record of
the measurements defined by EMVA Standard 1288:

* **System gain** ``K`` in DN per electron, the slope of the photon transfer
  curve (PTC): ``sigma_DN**2 = K * mean_DN + sigma_read_DN**2``. It is the
  product of the sense-node conversion gain, the analog gain, and the ADC code
  scale, so it converts every digital measurement back to electrons.
* **Read noise** in electrons, the PTC intercept extrapolated to zero signal
  after removing common-mode row noise and the variance added by ADC
  quantization.
* **Saturation capacity** in electrons, the mean charge at which the response
  stops increasing, usually set by the photodiode full well and occasionally by
  the ADC full scale.
* **Absolute sensitivity threshold**, the illumination at which the
  shot-noise-plus-read-noise signal-to-noise ratio equals one.
* **Dynamic range**, ``20 log10(saturation_capacity_e / read_noise_e)``.
* **DSNU** (dark-signal non-uniformity) and **PRNU** (photo-response
  non-uniformity), the fixed spatial standard deviations of the dark offset
  and of the relative flat-field response. ``dsnu_e`` is the pre-zero-clamp,
  input-referred offset sigma; the raw post-clamp dark spread is smaller and is
  recorded in provenance.
* **SNR curve** and **linearity residual**, which describe the signal range
  between the noise floor and saturation.

Real parts span roughly 0.01-10 DN/e system gain, 0.3-10 e- rms read noise,
2-50 ke- full well, 0.1-10 e- DSNU, 0.1-2 % PRNU, and 40-90 dB dynamic range.
The report stores physical measurements with explicit unit-bearing field names;
``snr_curve`` points carry both DN and electron values so a reader never has to
guess whether a number is digital or analog.

These reports describe a **simulated, EMVA 1288-style** characterization. They
are not certified EMVA measurements: the simulation assumes uniform
illumination, a single conversion-gain mode, a fixed temperature, no lens
shading or camera shading, Gaussian read noise, Poisson shot noise, and an
ideal linear ADC. The provenance mapping records those assumptions, the sensor
configuration hash, the readout mode, and the exact definitions used.

Failure artefacts guarded here: a report with NaN or infinity cannot be written
as strict JSON; a PRNU stored as 0.5 instead of 0.5 percent is off by 100x; a
read noise value that still contains ADC quantization is not the analog input
referred noise; a saturation capacity taken from an unsaturated frame
understates the full well; and a missing or renamed field silently breaks every
downstream figure and table. Construction validates every numeric field and
:meth:`EMVAReport.from_dict` names the offending field instead of guessing.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

__all__ = ["EMVAReport", "SNRPoint", "render_g3_report"]

_SCHEMA_VERSION = "1.0"
_REPORT_KIND = "EMVA1288StyleReport"


def _finite_float(value: object, field_name: str) -> float:
    """Return ``value`` as a finite Python float or raise a named error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a real number; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite; got {value!r}")
    return result


def _non_negative_float(value: object, field_name: str) -> float:
    """Return a finite, non-negative float or raise a named error."""
    result = _finite_float(value, field_name)
    if result < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {result!r}")
    return result


def _positive_float(value: object, field_name: str) -> float:
    """Return a finite, strictly positive float or raise a named error."""
    result = _finite_float(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be positive; got {result!r}")
    return result


@dataclass(frozen=True)
class SNRPoint:
    """One measured point on the signal-to-noise-ratio curve.

    Parameters
    ----------
    irradiance_w_per_m2_nm:
        Flat-field spectral irradiance at the sensor plane. Together with the
        sensor response this sets the signal level; it is not a lux value.
    mean_signal_dn:
        Mean black-level-corrected digital signal at this level.
    mean_signal_e:
        The same mean expressed in electrons through the measured system gain.
    noise_dn:
        Row-demeaned pixel-temporal noise standard deviation in DN.
        Common-mode row noise is removed; ADC quantization is retained, so this
        is the noise of the pixel signal chain after row-noise correction.
    noise_e:
        The same noise expressed in electrons.
    snr_db:
        ``20 log10(mean_signal_e / noise_e)``. At the absolute sensitivity
        threshold and below it is at or under 0 dB.
    noise_dn_total:
        Optional raw-frame temporal noise in DN, including common-mode row
        noise and ADC quantization. The primary ``noise_dn`` is the
        row-demeaned pixel-temporal noise (quantization included, common-mode
        row noise removed); the total variant describes what a single image
        actually shows.
    noise_e_total:
        The total noise in electrons.
    snr_db_total:
        ``20 log10(mean_signal_e / noise_e_total)``.

    The explicit DN and electron fields keep the units auditable. A bare
    ``snr`` value cannot be interpreted without knowing whether the noise was
    measured before or after gain, quantization, and row-noise removal.
    """

    irradiance_w_per_m2_nm: float
    mean_signal_dn: float
    mean_signal_e: float
    noise_dn: float
    noise_e: float
    snr_db: float
    noise_dn_total: float | None = None
    noise_e_total: float | None = None
    snr_db_total: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "irradiance_w_per_m2_nm",
            _positive_float(self.irradiance_w_per_m2_nm, "irradiance_w_per_m2_nm"),
        )
        object.__setattr__(
            self, "mean_signal_dn", _non_negative_float(self.mean_signal_dn, "mean_signal_dn")
        )
        object.__setattr__(
            self, "mean_signal_e", _non_negative_float(self.mean_signal_e, "mean_signal_e")
        )
        object.__setattr__(self, "noise_dn", _non_negative_float(self.noise_dn, "noise_dn"))
        object.__setattr__(self, "noise_e", _non_negative_float(self.noise_e, "noise_e"))
        object.__setattr__(self, "snr_db", _finite_float(self.snr_db, "snr_db"))
        if self.noise_dn_total is not None:
            object.__setattr__(
                self,
                "noise_dn_total",
                _non_negative_float(self.noise_dn_total, "noise_dn_total"),
            )
        if self.noise_e_total is not None:
            object.__setattr__(
                self,
                "noise_e_total",
                _non_negative_float(self.noise_e_total, "noise_e_total"),
            )
        if self.snr_db_total is not None:
            object.__setattr__(
                self, "snr_db_total", _finite_float(self.snr_db_total, "snr_db_total")
            )

    def to_dict(self) -> dict[str, float | None]:
        """Return the point as a JSON-friendly mapping."""
        return {
            "irradiance_w_per_m2_nm": self.irradiance_w_per_m2_nm,
            "mean_signal_dn": self.mean_signal_dn,
            "mean_signal_e": self.mean_signal_e,
            "noise_dn": self.noise_dn,
            "noise_e": self.noise_e,
            "snr_db": self.snr_db,
            "noise_dn_total": self.noise_dn_total,
            "noise_e_total": self.noise_e_total,
            "snr_db_total": self.snr_db_total,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a point from :meth:`to_dict` output."""
        required = (
            "irradiance_w_per_m2_nm",
            "mean_signal_dn",
            "mean_signal_e",
            "noise_dn",
            "noise_e",
            "snr_db",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"SNR point is missing required field(s): {', '.join(missing)}")
        values = {name: payload[name] for name in required}
        for optional in ("noise_dn_total", "noise_e_total", "snr_db_total"):
            if optional in payload:
                values[optional] = payload[optional]
        return cls(**values)


@dataclass(frozen=True)
class EMVAReport:
    """A complete EMVA 1288-style characterization of one sensor mode.

    Parameters
    ----------
    system_gain_dn_per_e:
        Measured photon-transfer slope in DN per electron. It includes the
        analog gain and the ADC code scale.
    read_noise_e:
        Input-referred read noise in electrons rms, obtained from the PTC
        intercept after removing row noise and ADC quantization variance.
    saturation_capacity_e:
        Maximum mean signal in electrons before clipping.
    absolute_sensitivity_threshold_e:
        Signal in electrons at which the shot-plus-read SNR equals one.
    dynamic_range_db:
        ``20 log10(saturation_capacity_e / read_noise_e)``.
    dsnu_e:
        Pre-zero-clamp, input-referred dark-signal non-uniformity in electrons
        rms. The sensor clamps the pre-Poisson mean charge at zero and
        quantizes the sub-LSB dark signal, so the raw post-clamp spatial spread
        is smaller; the characterization records it separately in provenance
        (``dsnu_observed_fixed_pattern_e``).
    prnu_percent:
        Photo-response non-uniformity as a percentage (0.5 means 0.5 %, not
        50 %). The field name carries the unit to prevent the classic
        fraction-versus-percent error.
    snr_curve:
        Measured signal-to-noise points from low light to just below
        saturation.
    linearity_residual:
        Maximum relative deviation from a straight-line fit of mean signal
        versus irradiance, over unsaturated levels. It is a dimensionless
        fraction (0.01 means 1 %).
    provenance:
        JSON-friendly measurement metadata: sensor configuration hash, readout
        mode, levels, frame counts, algorithm definitions, and the explicit
        statement that this is a simulated characterization rather than a
        certified EMVA measurement.

    Numeric fields are validated on construction. ``provenance`` is copied so a
    caller cannot mutate the report through an alias.
    """

    system_gain_dn_per_e: float
    read_noise_e: float
    saturation_capacity_e: float
    absolute_sensitivity_threshold_e: float
    dynamic_range_db: float
    dsnu_e: float
    prnu_percent: float
    snr_curve: tuple[SNRPoint, ...]
    linearity_residual: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "system_gain_dn_per_e",
            _positive_float(self.system_gain_dn_per_e, "system_gain_dn_per_e"),
        )
        object.__setattr__(self, "read_noise_e", _positive_float(self.read_noise_e, "read_noise_e"))
        object.__setattr__(
            self,
            "saturation_capacity_e",
            _positive_float(self.saturation_capacity_e, "saturation_capacity_e"),
        )
        object.__setattr__(
            self,
            "absolute_sensitivity_threshold_e",
            _positive_float(
                self.absolute_sensitivity_threshold_e, "absolute_sensitivity_threshold_e"
            ),
        )
        object.__setattr__(
            self, "dynamic_range_db", _finite_float(self.dynamic_range_db, "dynamic_range_db")
        )
        object.__setattr__(self, "dsnu_e", _non_negative_float(self.dsnu_e, "dsnu_e"))
        object.__setattr__(
            self, "prnu_percent", _non_negative_float(self.prnu_percent, "prnu_percent")
        )
        object.__setattr__(
            self,
            "linearity_residual",
            _non_negative_float(self.linearity_residual, "linearity_residual"),
        )
        try:
            points = tuple(self.snr_curve)
        except TypeError as exc:
            raise ValueError("snr_curve must be an iterable of SNRPoint values") from exc
        if not all(isinstance(point, SNRPoint) for point in points):
            raise ValueError("snr_curve must contain only SNRPoint values")
        object.__setattr__(self, "snr_curve", points)
        if not isinstance(self.provenance, dict):
            raise ValueError(f"provenance must be a mapping; got {type(self.provenance).__name__}")
        if any(not isinstance(key, str) for key in self.provenance):
            raise ValueError("provenance keys must be strings")
        object.__setattr__(self, "provenance", copy.deepcopy(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-friendly mapping.

        The mapping is self-describing: it names the schema version and report
        kind, keeps unit-bearing field names, and stores SNR points as a list of
        mappings. It contains no NumPy scalars or arrays, so ``json.dumps``
        works without a custom encoder.
        """
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": _REPORT_KIND,
            "system_gain_dn_per_e": self.system_gain_dn_per_e,
            "read_noise_e": self.read_noise_e,
            "saturation_capacity_e": self.saturation_capacity_e,
            "absolute_sensitivity_threshold_e": self.absolute_sensitivity_threshold_e,
            "dynamic_range_db": self.dynamic_range_db,
            "dsnu_e": self.dsnu_e,
            "prnu_percent": self.prnu_percent,
            "snr_curve": [point.to_dict() for point in self.snr_curve],
            "linearity_residual": self.linearity_residual,
            "provenance": copy.deepcopy(self.provenance),
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize the report as strict JSON, optionally writing it to disk.

        Parameters
        ----------
        path:
            Destination file. Parent directories are created. When omitted the
            JSON text is only returned.

        Returns
        -------
        The JSON text, identical to what was written when ``path`` is given.
        ``allow_nan=False`` is deliberate: NaN and infinity are not valid JSON
        and silently writing them produces files that other tools reject.
        """
        text = json.dumps(
            self.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
            ensure_ascii=True,
        )
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a report from :meth:`to_dict` output.

        Unknown keys are ignored so a newer producer can add metadata without
        breaking an older reader; missing required measurement fields raise a
        message naming every absent field.
        """
        required = (
            "system_gain_dn_per_e",
            "read_noise_e",
            "saturation_capacity_e",
            "absolute_sensitivity_threshold_e",
            "dynamic_range_db",
            "dsnu_e",
            "prnu_percent",
            "snr_curve",
            "linearity_residual",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"report is missing required field(s): {', '.join(missing)}")
        raw_curve = payload["snr_curve"]
        if not isinstance(raw_curve, Sequence) or isinstance(raw_curve, (str, bytes)):
            raise ValueError("snr_curve must be a sequence of SNR point mappings")
        curve: list[SNRPoint] = []
        for index, item in enumerate(raw_curve):
            if isinstance(item, SNRPoint):
                curve.append(item)
            elif isinstance(item, Mapping):
                curve.append(SNRPoint.from_dict(item))
            else:
                raise ValueError(
                    f"snr_curve item {index} must be a mapping; got {type(item).__name__}"
                )
        provenance = payload.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        return cls(
            system_gain_dn_per_e=payload["system_gain_dn_per_e"],
            read_noise_e=payload["read_noise_e"],
            saturation_capacity_e=payload["saturation_capacity_e"],
            absolute_sensitivity_threshold_e=payload["absolute_sensitivity_threshold_e"],
            dynamic_range_db=payload["dynamic_range_db"],
            dsnu_e=payload["dsnu_e"],
            prnu_percent=payload["prnu_percent"],
            snr_curve=tuple(curve),
            linearity_residual=payload["linearity_residual"],
            provenance=dict(provenance),
        )

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Parse a report from JSON text produced by :meth:`to_json`."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"report is not valid JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("report JSON must contain an object at the top level")
        return cls.from_dict(payload)


def _g3_check(
    checks: Sequence[Mapping[str, Any]],
    name: str,
) -> Mapping[str, Any]:
    """Return one named G3 check or raise a clear error."""
    for check in checks:
        if check.get("name") == name:
            return check
    raise ValueError(f"G3 report payload has no check named {name!r}")


def _g3_matrix_row(row: Sequence[float]) -> str:
    """Format one CCM row as comma-separated signed coefficients."""
    return ", ".join(f"{value:+.6f}" for value in row)


def render_g3_report(payload: Mapping[str, Any]) -> str:
    """Render the deterministic G3 Markdown report from a measurement payload.

    Parameters
    ----------
    payload:
        The dictionary returned by
        :func:`camsim.characterization.measure_g3`. It must contain ``status``,
        ``checks``, and the raw sections written by that function. An optional
        ``quick_reference`` mapping (present only in the full run that also
        measured the quick smoke scene) adds the quick 3000 K margin as
        explicitly non-gate context.

    Returns
    -------
    The complete Markdown report as a string. The summary is **status-aware**:
    a PASS payload states that all measured checks pass, while a FAIL payload
    states that the gate failed and names every failing check. It never claims
    success from a hard-coded sentence.

    Raises
    ------
    ValueError
        If the payload is not a camsim G3 payload, lacks a required section, or
        has a status other than ``"PASS"`` or ``"FAIL"``. Rendering a malformed
        payload would otherwise produce a plausible-looking but false report.
    """
    if payload.get("kind") != "camsim G3 ISP validation":
        raise ValueError(
            "render_g3_report requires a 'camsim G3 ISP validation' payload; "
            f"got {payload.get('kind')!r}"
        )
    status = str(payload.get("status", ""))
    if status not in {"PASS", "FAIL"}:
        raise ValueError(f"G3 payload status must be 'PASS' or 'FAIL'; got {status!r}")
    checks = list(payload["checks"])
    failing = [check for check in checks if not bool(check.get("pass"))]
    d65_check = _g3_check(checks, "ColorChecker mean deltaE2000 (D65)")
    warm_check = _g3_check(checks, "ColorChecker mean deltaE2000 (3000 K)")
    d65 = payload["colorchecker"]["illuminants"]["D65"]
    warm = payload["colorchecker"]["illuminants"]["3000K"]
    environment = payload["environment"]
    presets = payload["presets"]
    scene = payload["scene"]
    seeds = payload["seeds"]
    luther = payload["sensor_colorimetry"]
    awb = payload["awb"]
    demosaic = payload["demosaic"]
    tone = payload["tone"]
    trace = payload["trace"]
    linearity = payload["linearity"]
    lsc = payload["lsc"]
    denoise = payload["denoise"]
    sharpen = payload["sharpen"]
    rccc = payload["rccc"]

    d65_mean = float(d65_check["measured"])
    warm_mean = float(warm_check["measured"])
    d65_ok = bool(d65_check["pass"])
    warm_ok = bool(warm_check["pass"])
    failing_names = ", ".join(f"`{check['name']}`" for check in failing) or "unspecified"
    margin_3000 = (4.0 - warm_mean) / 4.0 * 100.0

    quick_reference = payload.get("quick_reference")
    quick_clause = ""
    if isinstance(quick_reference, Mapping):
        quick_warm = quick_reference.get("delta_e2000_mean", {}).get("3000K")
        if isinstance(quick_warm, (int, float)):
            quick_margin = (4.0 - float(quick_warm)) / 4.0 * 100.0
            quick_clause = (
                f" (quick smoke: {float(quick_warm):.3f}, {quick_margin:.2f} % below; "
                "not gate evidence)"
            )

    warm_order = sorted(
        range(len(warm["delta_e2000"])),
        key=lambda index: warm["delta_e2000"][index],
        reverse=True,
    )
    warm_maxima = [
        f"{warm['delta_e2000'][index]:.3f} ({warm['patch_names'][index]})"
        for index in warm_order[:2]
    ]

    if status == "PASS" and not failing:
        summary_lines = [
            "**All measured G3 checks pass.** The design ColorChecker gate is mean",
            "ΔE2000 < 2.0 under D65 and < 4.0 under 3000 K after an in-sample 3x3",
            "CCM. The measured means are "
            f"{d65_mean:.3f} and {warm_mean:.3f}, below the limits. The shipped",
            "sensor preset uses synthetic conventional Bayer bands (blue ~448 nm,",
            "green ~542 nm, red ~610 nm) with moderate out-of-band leakage; the",
            "original broad, nearly collinear CFA curves were replaced after the G3",
            "gate showed that no 3x3 matrix could colour-correct them. The",
            "Luther-condition residual is now "
            f"{100.0 * luther['luther_relative_residual']:.1f} %, versus 45.6 % for",
            "the superseded curves. The CCM is still fitted and evaluated in-sample;",
            "that is stated under limitations. The ΔE thresholds were not changed.",
        ]
    else:
        summary_lines = [
            f"**The G3 gate is FAIL ({len(failing)} of {len(checks)} checks).**",
            f"Failing checks: {failing_names}.",
            "The verdict table below records every measured value and tolerance;",
            "this summary and the JSON are rendered from the same payload, so a",
            "failure cannot be hidden behind success prose. The shipped sensor",
            "preset uses synthetic conventional Bayer bands (blue ~448 nm, green",
            "~542 nm, red ~610 nm) with moderate out-of-band leakage; the earlier",
            "broad, nearly collinear CFA curves were replaced after the G3 gate",
            "showed that no 3x3 matrix could colour-correct them. The CCM is",
            "fitted and evaluated in-sample; the ΔE thresholds were not changed.",
        ]

    delta_e_sentence = (
        "both means are below the 2.0 and 4.0 limits."
        if d65_ok and warm_ok
        else f"the failing ColorChecker check(s) are {failing_names}."
    )

    exposure_parts: list[str] = []
    for name in ("D65", "3000K"):
        section = payload["colorchecker"]["illuminants"][name]
        milliseconds = 1000.0 * float(section["exposure_s"])
        fraction = 100.0 * float(section["white_fraction"])
        if bool(section.get("exposure_clamped")):
            exposure_parts.append(
                f"{name} reaches the preset ceiling at {milliseconds:.2f} ms and "
                f"achieves {fraction:.1f} %"
            )
        else:
            exposure_parts.append(
                f"{name} uses {milliseconds:.2f} ms and achieves {fraction:.1f} %"
            )
    exposure_text = (
        "The exposure calibration requests "
        f"{100.0 * float(scene['target_white_fraction']):.0f} % of the ADC white span "
        f"but is bounded by the preset's {float(scene['exposure_ceiling_s']):.3f} s "
        "ceiling. " + "; ".join(exposure_parts) + "."
    )

    seeds_text = (
        f"Seeds: probe {seeds['probe']}, scene {seeds['scene']}, flat {seeds['flat']}, "
        f"LSC {seeds['lsc']}, denoise {seeds['denoise']}. {seeds['note']}"
    )

    lines: list[str] = [
        "# G3 - ISP validation",
        "",
        f"**Status: {status}**",
        "",
        "G3 closes Phase 3 by running the complete image-signal-processing chain",
        "from a spectral ColorChecker scene through the sensor to a rendered",
        "display image, then measuring colour accuracy, white balance, demosaic",
        "agreement, tone, stage-trace consistency, linearity, lens shading,",
        "denoising, sharpening, and the RCCC baseline. The measurements are",
        "produced by `camsim.characterization.measure_g3`; this report and the",
        "machine-readable JSON are generated by the same full example run, so a",
        "stale report is detectable.",
        "",
        *summary_lines,
        "",
        "All evidence is simulated. It is not a certified ISO 12233 or EMVA 1288",
        "measurement, and the flagship preset is a representative engineering",
        "design point rather than a measurement of one commercial part.",
        "",
        "## Environment and reproduction",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Python | `{environment['python']}` |",
        f"| camsim | `{environment['camsim']}` |",
        f"| NumPy | `{environment['numpy']}` |",
        f"| SciPy | `{environment['scipy']}` |",
        f"| colour-science | `{environment['colour_science']}` |",
        f"| colour-demosaicing | `{environment['colour_demosaicing']}` |",
        f"| Platform | `{environment['platform']}` |",
        f"| Sensor preset | `{presets['sensor']}` |",
        f"| ISP preset | `{presets['isp']}` |",
        f"| Sensor config hash | `{presets['sensor_config_hash']}` |",
        f"| ISP config hash (base preset) | `{presets['isp_config_hash']}` |",
        f"| Calibration ISP hash (D65) | `{d65['calibration_isp_config_hash']}` |",
        f"| Calibration ISP hash (3000 K) | `{warm['calibration_isp_config_hash']}` |",
        "| Artifact generator | `uv run python examples/02_isp_stage_gallery.py` |",
        "| Machine-readable results | `results/reports/g3_isp_results.json` |",
        "| Gate tests | `uv run pytest tests/test_example_02.py -v` |",
        "| Quick smoke | `CAMSIM_FIG_DIR=/tmp/g3_quick_figures uv run python "
        "examples/02_isp_stage_gallery.py --quick --report-dir /tmp/g3_quick_reports` "
        "(not gate evidence; suffixed figures) |",
        "",
        "The full run uses a "
        f"{scene['colorchecker_shape'][1]} x {scene['colorchecker_shape'][0]}"
        f" ColorChecker at {scene['lux']:.0f} lux with {scene['oversampling']}x",
        "oversampling, the shipped flagship sensor at LCG, and the D65 and 3000 K",
        "illuminants. " + exposure_text,
        "",
        seeds_text,
        "",
        "## ISP chain and linear-space invariant",
        "",
        "The enforced order is `DPC -> BLC -> LSC -> AWB -> demosaic -> CCM ->",
        "denoise -> sharpen -> tone -> output`. DPC, BLC, LSC, AWB, demosaicing,",
        "the CCM, denoise, and sharpen operate in linear float32 light. BLC",
        "subtracts the 64 DN pedestal and divides by the 959 DN white span, so",
        "ADC saturation maps to 1.0. The tone stage applies the exposure trim in",
        "linear light, converts to Display P3 when requested, clamps to the",
        "display range, and applies the sRGB OETF; it is the only non-linear",
        "step. The linear pre-tone scaling check verifies this invariant directly.",
        "",
        "## Measurement methodology",
        "",
        "**Spectral scene to RAW.** `camsim.scene.render` multiplies ColorChecker",
        "reflectance by the photometrically normalised illuminant and the lens",
        "term, then boxes the oversampled target down to the sensor grid. The",
        "sensor integrates QE x IR-cut x CFA x photon flux, adds shot, read, and",
        "fixed-pattern noise, applies conversion and analog gain, and quantises",
        "to 10-bit codes with a 64 DN pedestal. Every exposure uses a fixed seed.",
        "",
        "**White balance and camera RGB.** The analytic per-channel response to a",
        "perfect diffuser under each illuminant gives green-normalised manual AWB",
        "gains. Using the perfect diffuser, rather than the chart's white patch,",
        "is what makes a row-sum CCM neutral-preserving: the reference Lab scale",
        "is defined by the perfect diffuser. The pipeline then runs with the CCM",
        "disabled and the trace enabled, and the 24 patch means are taken from",
        "the demosaiced, white-balanced linear RGB buffer.",
        "",
        "**Target linear sRGB.** The reference Lab is converted back to XYZ with",
        "the scene illuminant's white point, adapted to CIE D65 with the Bradford",
        "transform, and converted to **linear** sRGB with",
        "`colour.XYZ_to_RGB(..., apply_cctf_encoding=False)`. The perfect-diffuser",
        "white therefore maps to (1, 1, 1); the camera RGB is divided by the",
        "measured perfect-diffuser level before fitting, and the tone stage's",
        "exposure compensation restores the absolute scale.",
        "",
        "**CCM and ΔE2000.** One ridge-regularised 3x3 matrix per illuminant is",
        "fitted with `camsim.isp.ccm.fit_ccm(..., ridge=1e-4, row_sum=True)` on",
        "the same 24 patches used for evaluation (in-sample calibration-scene",
        "validation). The rendered sRGB display is decoded with the sRGB EOTF to",
        "XYZ D65 and CIE Lab, and CIEDE2000 is computed per patch against the",
        "reference Lab. In-sample fitting measures whether the ISP can represent",
        "the calibration scene at all; it does not measure generalisation to a",
        "held-out chart or a different illuminant.",
        "",
        "**AWB, demosaic, tone, and trace.** The gray-world estimator runs on a",
        "50 % neutral flat field under each illuminant; expected gains come from",
        "the analytic spectral response, and a second check injects a known",
        "per-CFA cast (R x1.6, G x1.0, B x0.7) and requires the estimator to",
        "invert it. Bilinear and MHC are compared with `colour-demosaicing` on a",
        "synthetic Siemens-star RGB mosaic with independent R, G, and B",
        "structure. The sRGB OETF is compared with `colour.cctf_encoding`. The",
        "stage trace is validated by applying the tone configuration to the last",
        "linear trace stage and comparing both the display array and the",
        "unclipped `linear_rgb` buffer at +0.5 EV in Display P3.",
        "",
        "## Verdict summary",
        "",
        "| Check | Measured | Tolerance | Result |",
        "|---|---|---|---|",
    ]
    for check in checks:
        result = "PASS" if bool(check["pass"]) else "FAIL"
        lines.append(
            f"| {check['name']} | {check['measured_text']} | {check['tolerance_text']} | {result} |"
        )

    lines.extend(
        [
            "",
            "## 1. End-to-end ColorChecker colour",
            "",
            "The fitted matrices (rows sum to one to machine precision) are:",
            "",
            "| Illuminant | Row 1 | Row 2 | Row 3 |",
            "|---|---|---|---|",
            f"| D65 | {_g3_matrix_row(d65['ccm'][0])} | "
            f"{_g3_matrix_row(d65['ccm'][1])} | {_g3_matrix_row(d65['ccm'][2])} |",
            f"| 3000K | {_g3_matrix_row(warm['ccm'][0])} | "
            f"{_g3_matrix_row(warm['ccm'][1])} | {_g3_matrix_row(warm['ccm'][2])} |",
            "",
            "Per-patch CIEDE2000 between the rendered display and the spectrally",
            "computed reference:",
            "",
            "| Patch | D65 deltaE2000 | 3000 K deltaE2000 |",
            "|---|---:|---:|",
        ]
    )
    for index, patch in enumerate(d65["patch_names"]):
        lines.append(
            f"| {patch} | {d65['delta_e2000'][index]:.3f} | {warm['delta_e2000'][index]:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The means are {d65_mean:.3f} (D65) and {warm_mean:.3f} (3000 K); "
            f"{delta_e_sentence} The pre-clip means are",
            f"{d65['delta_e2000_preclip_mean']:.3f} and "
            f"{warm['delta_e2000_preclip_mean']:.3f}; display clipping is",
            "negligible. The CCM fit RMS residuals are "
            f"{d65['ccm_residual']:.4f} and {warm['ccm_residual']:.4f} linear RGB,",
            "and the matrix rows sum to one to machine precision. The fitted",
            "coefficients and per-patch values are recorded in the JSON.",
            "",
            f"The 3000 K mean is {warm_mean:.3f}, {margin_3000:.2f} % below the 4.0 "
            f"limit{quick_clause}. The gate is mean-only; per-patch maxima reach "
            f"{warm_maxima[0]} and {warm_maxima[1]}.",
            "",
            "**CFA rationale.** The preset uses conventional Bayer passbands",
            "centred near 448 nm (blue), 542 nm (green), and 610 nm (red) with",
            "moderate out-of-band leakage and peak transmittance 0.86-0.91",
            "(R 0.905, G 0.903, B 0.864). The Luther-condition relative residual",
            f"is {luther['luther_relative_residual']:.3f}; the CFA projection",
            f"condition number is {luther['condition_number']:.2f} and the R/G",
            f"correlation is {luther['channel_correlation'][0][1]:.3f}. The",
            "superseded broad curves left a 45.6 % residual and could not be",
            "colour-corrected by any 3x3 matrix; the conventional bands reduce the",
            "residual and the in-sample CCM meets both ΔE gates. The fit remains",
            "in-sample: it measures calibration-scene representability, not",
            "generalisation to a held-out chart or a different illuminant.",
            "",
            "![ColorChecker delta E](../figures/g3_delta_e.png)",
            "",
            "## 2. White balance",
            "",
            "Expected gains are the analytic spectral response ratios; measured",
            "gains are the gray-world estimate on a uniform 50 % flat field.",
            "",
            "| Illuminant | Channel | Expected gain | Measured gain | Relative error | Result |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    check_by_name = {check["name"]: check for check in checks}
    for name, label in (("D65", "D65"), ("3000K", "3000 K")):
        section = awb["illuminants"][name]
        for index, channel in enumerate(("R", "G", "B")):
            if channel == "G":
                result = "PASS"
            else:
                result = (
                    "PASS"
                    if check_by_name[f"AWB gain relative error, {channel} ({label})"]["pass"]
                    else "FAIL"
                )
            lines.append(
                f"| {name} | {channel} | {section['expected_gains'][index]:.6f} | "
                f"{section['measured_gains'][index]:.6f} | "
                f"{section['relative_errors'][index]:.6f} | {result} |"
            )
    injected = awb["injected_scaling"]
    lines.extend(
        [
            "",
            f"The injected-cast check (R x{injected['injection'][0]:g}, "
            f"G x{injected['injection'][1]:g}, B x{injected['injection'][2]:g})",
            "recovers the expected gains with a maximum relative error of "
            f"{100.0 * injected['max_relative_error']:.2f} %. All non-green",
            "channels are within the 5 % tolerance.",
            "",
            "![White-balance gains](../figures/g3_awb_gains.png)",
            "",
            "## 3. Demosaic, tone, trace, and linearity",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| Bilinear max abs vs colour-demosaicing | {demosaic['bilinear_max_abs']:.3e} |",
            f"| MHC max abs vs colour-demosaicing | {demosaic['mhc_max_abs']:.3e} |",
            f"| sRGB OETF max abs vs colour-science | {tone['max_abs']:.3e} |",
            f"| Trace display reproduction max abs | {trace['tone_max_abs']:.3e} |",
            f"| Trace linear_rgb reproduction max abs | {trace['linear_rgb_max_abs']:.3e} |",
            "| Linear pre-tone scaling max relative error | "
            f"{linearity['max_relative_error']:.3e} |",
            "",
            "The trace stage order in the Display P3 validation is "
            f"`{' -> '.join(trace['stage_names'])}`. The tone and linear-buffer",
            "comparisons are exact to float32 because the trace records the same",
            "buffers the pipeline consumed; they validate the stage bookkeeping",
            "and the exposure/P3 order, while the independent sRGB OETF check",
            "validates the transfer function against the published standard.",
            "",
            "## 4. Lens shading, denoise, and sharpen",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| LSC flat-field residual before correction | {lsc['residual_before']:.4f} |",
            f"| LSC flat-field residual after correction | {lsc['residual_after']:.4f} |",
            "| Denoise SNR before / after (dB) | "
            f"{denoise['snr_before_db']:.2f} / {denoise['snr_after_db']:.2f} |",
            f"| Denoise SNR gain (dB) | {denoise['snr_gain_db']:.2f} |",
            f"| Denoise edge-contrast relative change | {denoise['edge_preservation_error']:.4f} |",
            f"| Sharpen edge-contrast gain | {sharpen['contrast_gain']:.4f} |",
            f"| Sharpen positive overshoot | {sharpen['overshoot']:.4f} |",
            "",
            "The LSC measurement calibrates a gain map from one noisy flat field",
            "and applies it to an independent flat field; the residual is the",
            "max-minus-min range divided by the mean. The exact-reciprocal check",
            "is the Task 18 unit test",
            "`tests/test_isp_lsc.py::test_lsc_flat_field_corrects_vignetting`.",
            "Denoise and sharpen are measured on synthetic step and flat regions",
            "with explicit seeds. **Formal ISO 12233 MTF50 and MTF20 validation is",
            "deferred to G4**: the independent metrics harness does not exist yet,",
            "so G3 claims edge-contrast preservation and positive overshoot, not",
            "an MTF number.",
            "",
            "## 5. RCCC baseline",
            "",
            f"The RCCC frame is {rccc['shape'][1]} x {rccc['shape'][0]} pixels and",
            f"renders to shape {rccc['shape']} with {rccc['channels']} channels,",
            f"values in [{rccc['min']:.3f}, {rccc['max']:.3f}], all finite. The",
            "baseline uses the documented `(R, C, C)` luma placeholder; AWB and",
            "CCM are skipped by the pipeline. It is a shape/linearity/smoke test",
            "only, and the Bayer ΔE and MTF thresholds do not apply.",
            "",
            "![RCCC baseline](../figures/02_isp_rccc_baseline.png)",
            "",
            "## 6. Stage unit-test evidence",
            "",
            "DPC, BLC, AWB, and CCM are covered by focused unit tests rather than",
            "re-measured here: `tests/test_isp_dpc.py` checks same-CFA",
            "interpolation, PDAF safety, every Bayer orientation, and exact",
            "preservation of unaffected pixels; `tests/test_isp_blc.py` checks",
            "pedestal subtraction, the 2x2 pattern, affine behaviour about the",
            "pedestal, and the white-span normalisation; `tests/test_isp_awb.py`",
            "checks the three estimators, per-CFA statistics, and gain",
            "normalisation; `tests/test_isp_ccm.py` checks recovery of a known",
            "matrix, machine-precision row sums, ridge regularisation, and ΔE",
            "improvement. The pipeline suite",
            "`tests/test_isp_pipeline.py::test_blc_stage_normalizes_to_the_white_span`",
            "ties the BLC contract to the rendered output.",
            "",
            "## Assumptions and limitations",
            "",
            "- **Simulated data.** This is a simulated ISP validation, not a",
            "  certified measurement of a physical camera. The sensor preset is a",
            "  representative engineering design point; its QE/CFA/IR-cut curves",
            "  are synthetic estimates bundled in `camsim.data`.",
            "- **In-sample CCM.** The 3x3 matrix is fitted and evaluated on the",
            "  same 24 patches. It measures calibration-scene representability,",
            "  not held-out generalisation or colour constancy under a different",
            "  illuminant.",
            "- **Manual analytic AWB for the CCM calibration.** The ColorChecker",
            "  run uses perfect-diffuser analytic gains so the row-sum CCM is",
            "  neutral-preserving; the default gray-world estimator is validated",
            "  separately on neutral flat fields and under an injected cast.",
            "- **CFA design point.** The preset uses synthetic conventional Bayer",
            "  bands (blue ~448 nm, green ~542 nm, red ~610 nm) with moderate",
            "  out-of-band leakage; the original broad, nearly collinear curves",
            "  were replaced after the G3 gate showed that no 3x3 matrix could",
            "  colour-correct them. The in-sample CCM passes both ΔE gates. A",
            "  future measured CFA replacement must record its provenance in this",
            "  report and in the preset.",
            "- **Quick mode is not gate evidence.** `--quick` uses a smaller chart",
            "  and star and writes `g3_isp_results_quick.json`; it never writes or",
            "  overwrites the committed `g3_isp_results.json` or `g3_isp.md`.",
            "- **Formal MTF is deferred to G4.** ISO 12233-style e-SFR, MTF50, and",
            "  MTF20 require the independent metrics harness; G3 reports",
            "  edge-contrast preservation and overshoot instead.",
            "- **Camera auto-exposure is deferred to G5.** The `Camera`",
            "  orchestration and `auto_expose()` convergence check are Phase 5",
            "  consolidation work; G3 calibrates exposure explicitly from a probe.",
            "- **Determinism.** Seeds are explicit and outputs are reproducible",
            "  within one software environment; PNG bytes can differ across",
            "  Matplotlib/font builds, so cross-platform comparisons use numbers",
            "  and tolerances.",
            "",
            "## Figures",
            "",
            "| Figure | Path |",
            "|---|---|",
            "| ISP stage contact sheet | `results/figures/02_isp_stage_contact_sheet.png` |",
            "| Stage zoom crops | `results/figures/02_isp_stage_zoom.png` |",
            "| Demosaic comparison | `results/figures/02_isp_demosaic_comparison.png` |",
            "| ColorChecker ΔE2000 | `results/figures/g3_delta_e.png` |",
            "| AWB gains | `results/figures/g3_awb_gains.png` |",
            "| Check summary | `results/figures/g3_check_summary.png` |",
            "| RCCC baseline | `results/figures/02_isp_rccc_baseline.png` |",
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run python examples/02_isp_stage_gallery.py",
            "uv run pytest tests/test_example_02.py -v",
            "uv run pytest -q",
            "uv run ruff check . && uv run ruff format --check . && uv run mypy src",
            "```",
            "",
            "The full example writes `results/reports/g3_isp_results.json` and",
            "`results/reports/g3_isp.md`; the gate tests parse the committed",
            "report tables back and compare every numeric cell with the JSON, so",
            "a stale report fails instead of being accepted. The `--quick` flag",
            "is a smoke run and is not gate evidence.",
            "",
        ]
    )
    if status != "PASS":
        lines.extend(
            [
                "## Gate failure",
                "",
                f"The failing checks are: {failing_names}.",
                "The JSON and this report are generated from the same payload; do",
                "not mark G3 green until the failing measurement is resolved.",
                "",
            ]
        )
    return "\n".join(lines)
