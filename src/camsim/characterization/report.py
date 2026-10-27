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
  and of the relative flat-field response.
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

__all__ = ["EMVAReport", "SNRPoint"]

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
        Temporal noise standard deviation in DN. The characterization removes
        common-mode row noise and the ADC quantization variance, so this is the
        pixel-temporal noise of the analog signal chain.
    noise_e:
        The same noise expressed in electrons.
    snr_db:
        ``20 log10(mean_signal_e / noise_e)``. At the absolute sensitivity
        threshold and below it is at or under 0 dB.

    The explicit DN and electron fields keep the units auditable. A bare
    ``snr`` value cannot be interpreted without knowing whether the noise was
    measured before or after gain and quantization.
    """

    irradiance_w_per_m2_nm: float
    mean_signal_dn: float
    mean_signal_e: float
    noise_dn: float
    noise_e: float
    snr_db: float

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

    def to_dict(self) -> dict[str, float]:
        """Return the point as a JSON-friendly mapping."""
        return {
            "irradiance_w_per_m2_nm": self.irradiance_w_per_m2_nm,
            "mean_signal_dn": self.mean_signal_dn,
            "mean_signal_e": self.mean_signal_e,
            "noise_dn": self.noise_dn,
            "noise_e": self.noise_e,
            "snr_db": self.snr_db,
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
        return cls(**{name: payload[name] for name in required})


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
        Dark-signal non-uniformity in electrons rms.
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
