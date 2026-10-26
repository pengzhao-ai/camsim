"""Command-line interface skeleton for camsim.

``info`` is functional and doubles as the console-entry-point smoke test.
``validate`` and ``schema`` are wired into the parser but deliberately exit with
status 2 and an explanatory message until the Pydantic configuration schema
lands; reporting success before validation exists would be misleading.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Callable, Sequence

from camsim._version import __version__

EXIT_OK = 0
EXIT_NOT_IMPLEMENTED = 2

Handler = Callable[[argparse.Namespace], int]


def _cmd_info(_: argparse.Namespace) -> int:
    payload = {
        "name": "camsim",
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
    }
    print(json.dumps(payload, indent=2))
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    print(
        f"camsim validate {args.path}: not implemented until the configuration schema lands.",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED


def _cmd_schema(args: argparse.Namespace) -> int:
    print(
        f"camsim schema {args.kind}: not implemented until the configuration schema lands.",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED


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
