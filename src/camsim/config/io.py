"""Loading, validation, and JSON Schema export for camsim configuration files.

A configuration file crosses three boundaries: the filesystem (missing or
unreadable input), JSON syntax (malformed text), and physics (schema and
cross-field rules). The helpers here keep those failures distinguishable
because callers act on them differently: an automated sweep should stop on a
file it cannot read, skip a file whose JSON is malformed, and continue with a
warning when a sensor is merely ADC-limited.

Design choices:

* ``load_sensor`` and ``load_isp`` raise :class:`pydantic.ValidationError` for
  schema violations. Programmatic callers get the complete list of problems in
  one exception rather than failing field by field.
* :func:`validate_config` never raises for malformed content; it returns a
  :class:`ValidationReport` whose ``errors`` and ``warnings`` lists can be
  printed, logged, or serialised. Only filesystem-level failures (missing
  file, permission denied, undecodable bytes) propagate, so the CLI can map
  them to its distinct exit code.
* Sensor versus ISP files are recognised from their distinctive keys, not from
  the filename, so presets can be renamed freely. A file mixing both sets is
  reported as ambiguous instead of being coerced into one schema.
* JSON Schema export uses Pydantic's ``model_json_schema`` so editors and
  downstream tools can autocomplete unit-bearing field names.

Spectral data files referenced by a preset (``qe_csv``, ``cfa_csv``,
``ir_cut_csv``) are intentionally *not* opened here. They are resolved when a
sensor is constructed, which lets presets be authored before the datasets land
and keeps validation independent of the working directory.

Real-world failure artefacts handled here: a UTF-8 BOM or a trailing comma in
hand-edited JSON; an unknown top-level payload (for example a lens file passed
to the sensor loader); a sensor preset with ISP stage keys copied in; and a
valid file whose ADC limits saturation before full well (warning, not error).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, TypeAlias

from pydantic import ValidationError

from camsim.config.schema import ISPConfig, SensorConfig

PathLike: TypeAlias = str | Path
"""A filesystem path accepted by the loaders."""

PRESETS_DIR: Final[Path] = Path(__file__).resolve().parent / "presets"
"""Directory containing the shipped sensor and ISP presets."""

# Keys unique to one configuration kind. Shared envelope keys
# (schema_version, name, provenance) are deliberately absent so they cannot
# cause ambiguity on their own.
_SENSOR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "pixel",
        "cfa",
        "spectral",
        "modes",
        "analog_gain",
        "adc",
        "exposure",
        "temperature",
        "defects",
        "noise",
        "auto_hcg_threshold_e",
        "pdaf",
        "binning",
        "blooming",
    }
)
_ISP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "dpc",
        "blc",
        "lsc",
        "awb",
        "demosaic",
        "ccm",
        "denoise",
        "sharpen",
        "tone",
        "output",
    }
)


@dataclass(slots=True)
class ValidationReport:
    """Outcome of validating one configuration file.

    Parameters
    ----------
    errors:
        Schema or cross-field violations. A non-empty list means the file is
        invalid and must not be used to configure a pipeline.
    warnings:
        Legal but physically noteworthy conditions, such as ADC-limited
        saturation. Warnings never make the configuration invalid.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return ``True`` when the file produced no errors."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the report."""
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _read_json(path: PathLike) -> Any:
    """Read and decode UTF-8 JSON, letting filesystem and syntax errors escape.

    ``json.load`` accepts ``NaN``/``Infinity`` by default; those reach Pydantic
    and are rejected there by ``allow_inf_nan=False``, so no second parse pass
    is needed here.
    """
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_sensor(path: PathLike) -> SensorConfig:
    """Load and validate a sensor preset.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist (also ``PermissionError`` or
        ``IsADirectoryError`` for other filesystem failures).
    json.JSONDecodeError
        If the file is not valid JSON.
    pydantic.ValidationError
        If the payload violates :class:`~camsim.config.schema.SensorConfig`.
    """
    return SensorConfig.model_validate(_read_json(path))


def load_isp(path: PathLike) -> ISPConfig:
    """Load and validate an ISP preset.

    Raises the same exceptions as :func:`load_sensor`.
    """
    return ISPConfig.model_validate(_read_json(path))


def sensor_schema() -> dict[str, Any]:
    """Return the JSON Schema (draft 2020-12) for sensor presets."""
    return SensorConfig.model_json_schema()


def isp_schema() -> dict[str, Any]:
    """Return the JSON Schema (draft 2020-12) for ISP presets."""
    return ISPConfig.model_json_schema()


def validate_config(path: PathLike) -> ValidationReport:
    """Validate any supported configuration file and collect errors/warnings.

    Filesystem errors propagate to the caller; malformed JSON, schema
    violations, and unknown payloads are reported through the returned
    :class:`ValidationReport` so one bad file does not abort a validation run
    over a directory of presets.
    """
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        return ValidationReport(
            errors=[f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"]
        )
    if not isinstance(payload, dict):
        return ValidationReport(
            errors=[f"configuration root must be a JSON object; got {type(payload).__name__}"]
        )
    has_sensor_keys = bool(_SENSOR_KEYS & payload.keys())
    has_isp_keys = bool(_ISP_KEYS & payload.keys())
    if has_sensor_keys and has_isp_keys:
        return ValidationReport(
            errors=[
                "configuration mixes sensor and ISP fields; "
                "keep sensor geometry and ISP stage settings in separate files"
            ]
        )
    if has_sensor_keys:
        return _validate_sensor(payload)
    if has_isp_keys:
        return _validate_isp(payload)
    return ValidationReport(
        errors=[
            "unrecognised configuration: expected sensor fields "
            "(pixel, cfa, spectral, modes, ...) or ISP stage fields "
            "(dpc, blc, awb, demosaic, ...)"
        ]
    )


def available_presets() -> dict[str, list[str]]:
    """List the shipped preset files grouped by kind.

    Paths are relative to :data:`PRESETS_DIR` and use forward slashes on every
    platform, so the output is stable enough for documentation and CLI JSON.
    """
    presets: dict[str, list[str]] = {}
    for kind in ("sensors", "isp"):
        directory = PRESETS_DIR / kind
        presets[kind] = (
            sorted(path.relative_to(PRESETS_DIR).as_posix() for path in directory.glob("*.json"))
            if directory.is_dir()
            else []
        )
    return presets


def _validate_sensor(payload: dict[str, Any]) -> ValidationReport:
    """Validate a sensor payload, including its non-fatal physics warnings."""
    try:
        config = SensorConfig.model_validate(payload)
    except ValidationError as exc:
        return ValidationReport(errors=_format_validation_errors(exc))
    return ValidationReport(warnings=config.validation_warnings())


def _validate_isp(payload: dict[str, Any]) -> ValidationReport:
    """Validate an ISP payload. ISP stage rules currently have no warnings."""
    try:
        ISPConfig.model_validate(payload)
    except ValidationError as exc:
        return ValidationReport(errors=_format_validation_errors(exc))
    return ValidationReport()


def _format_validation_errors(exc: ValidationError) -> list[str]:
    """Flatten Pydantic errors into one human-readable line per violation."""
    messages: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        messages.append(f"{location or '<root>'}: {error['msg']}")
    return messages
