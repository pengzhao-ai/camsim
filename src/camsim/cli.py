"""Command-line interface for camsim.

The CLI exposes the configuration layer before any simulation code exists:

* ``camsim validate PATH`` loads a sensor or ISP preset and prints its errors
  and warnings. Exit status is part of the contract: ``0`` for a valid file,
  ``1`` for a file that was read but violates the schema, and ``2`` for input
  that could not be read at all. Scripts can therefore gate a sweep without
  scraping human-readable output.
* ``camsim schema {sensor,isp}`` prints the Pydantic JSON Schema, giving editors
  completion for unit-bearing fields such as ``pixel.pitch_um`` and
  ``adc.full_scale_uv``.
* ``camsim info`` reports the package version, runtime, and shipped presets.

Physical failure modes encountered here are configuration-level rather than
numerical: a sensor whose full well saturates the ADC is reported as a warning
(the file remains valid), while an unnormalised colour matrix, a black level
outside the ADC code range, or a misspelled key is an error. Exit codes are
kept distinct because a missing file should stop a batch run, whereas one
invalid preset among many should not hide the rest of the report.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from camsim._version import __version__
from camsim.config.io import (
    ValidationReport,
    available_presets,
    isp_schema,
    sensor_schema,
    validate_config,
)

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_UNREADABLE = 2

Handler = Callable[[argparse.Namespace], int]


def _cmd_info(_: argparse.Namespace) -> int:
    """Print package metadata and the presets shipped with this install."""
    payload = {
        "name": "camsim",
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "presets": available_presets(),
    }
    print(json.dumps(payload, indent=2))
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate one JSON configuration and return the documented exit code."""
    path = Path(args.path)
    try:
        report = validate_config(path)
    except (OSError, UnicodeDecodeError) as exc:
        # A missing, unreadable, or non-UTF-8 file is an input problem, not a
        # schema problem; the distinct exit code lets batch callers branch.
        print(f"camsim validate: cannot read {path}: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE
    _print_report(path, report)
    return EXIT_OK if report.valid else EXIT_INVALID


def _print_report(path: Path, report: ValidationReport) -> None:
    """Print one line per finding, followed by the overall verdict."""
    for warning in report.warnings:
        print(f"warning: {warning}")
    for error in report.errors:
        print(f"error: {error}")
    if report.valid:
        suffix = f" ({len(report.warnings)} warning(s))" if report.warnings else ""
        print(f"{path}: valid{suffix}")
    else:
        print(f"{path}: invalid ({len(report.errors)} error(s))")


def _cmd_schema(args: argparse.Namespace) -> int:
    """Print the JSON Schema for the requested configuration kind."""
    schema = sensor_schema() if args.kind == "sensor" else isp_schema()
    print(json.dumps(schema, indent=2))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="camsim",
        description="Deterministic camera sensor and ISP simulation with IQ metrics.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate a sensor or ISP JSON configuration."
    )
    validate.add_argument("path", help="Path to the JSON configuration file.")
    validate.set_defaults(handler=_cmd_validate)

    schema = subparsers.add_parser("schema", help="Print a JSON Schema for a configuration type.")
    schema.add_argument("kind", choices=("sensor", "isp"), help="Configuration kind to export.")
    schema.set_defaults(handler=_cmd_schema)

    info = subparsers.add_parser("info", help="Print package and runtime information.")
    info.set_defaults(handler=_cmd_info)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``camsim`` command line and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - console-script path
    raise SystemExit(main())
