"""Execute every Jupytext notebook headlessly within a two-minute budget.

Notebooks are the narrative counterpart of the examples. They must run from a
clean checkout with no committed ``.ipynb`` files, which means the execution
step is part of the release gate. This runner:

* discovers ``notebooks/*.py`` in sorted order;
* registers a venv-local ``python3`` kernel with the current interpreter, then
  converts and executes each notebook through ``jupytext --execute`` into
  ``results/cache/notebooks``;
* enforces a per-notebook timeout of 120 s by default, the design's Apple
  Silicon budget, and stops at the first failure;
* prints and writes per-notebook runtimes to
  ``results/reports/notebooks_timings.json`` for the consolidated G5 report.

A notebook is executed in a subprocess, so a kernel crash, a matplotlib
backend choice, or a global RNG in one notebook cannot contaminate the next.
The JSON file records machine-specific wall times as evidence, not a
cross-platform performance promise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR: Final[Path] = REPO_ROOT / "notebooks"
OUTPUT_DIR: Final[Path] = REPO_ROOT / "results" / "cache" / "notebooks"
DEFAULT_JSON: Final[Path] = REPO_ROOT / "results" / "reports" / "notebooks_timings.json"
DEFAULT_TIMEOUT_S: Final[float] = 120.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="per-notebook timeout in seconds (default: 120, the design budget)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON,
        help="timing evidence file",
    )
    return parser.parse_args(argv)


def _discover() -> list[Path]:
    """Return every Jupytext notebook source in deterministic name order."""
    return sorted(path for path in NOTEBOOKS_DIR.glob("*.py") if path.is_file())


def _register_kernel() -> None:
    """Register the current interpreter as the ``python3`` Jupytext kernel."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--prefix",
            sys.prefix,
            "--name",
            "python3",
            "--display-name",
            "Python 3 (camsim)",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "could not register the camsim Jupyter kernel; install the dev extra "
            f"(ipykernel) and retry.\n{completed.stderr.strip()}"
        )


def _run_one(path: Path, timeout_s: float) -> dict[str, Any]:
    """Execute one notebook and return its timing and verdict."""
    output = OUTPUT_DIR / f"{path.stem}.ipynb"
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupytext",
                "--to",
                "ipynb",
                "--set-kernel",
                "python3",
                "--execute",
                str(path),
                "--output",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - started
        print(f"TIMEOUT {path.name} after {elapsed:.1f} s (limit {timeout_s:g} s)")
        return {
            "name": path.name,
            "seconds": elapsed,
            "passed": False,
            "reason": f"timeout after {timeout_s:g} s",
        }
    elapsed = time.perf_counter() - started
    passed = completed.returncode == 0
    if not passed:
        print(f"FAILED {path.name} (exit {completed.returncode})")
        stderr = completed.stderr.strip().splitlines()
        if stderr:
            print("\n".join(stderr[-20:]))
    return {
        "name": path.name,
        "seconds": elapsed,
        "passed": passed,
        "reason": "ok" if passed else f"exit code {completed.returncode}",
    }


def main(argv: list[str] | None = None) -> int:
    """Run the notebooks and return a process exit code."""
    args = _parse_args(argv)
    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be positive")
    notebooks = _discover()
    if not notebooks:
        print("No notebooks found.")
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _register_kernel()

    print(f"camsim notebook runner: {len(notebooks)} notebook(s), timeout {args.timeout:g} s")
    items: list[dict[str, Any]] = []
    for path in notebooks:
        print(f"Executing {path.name} ...", flush=True)
        item = _run_one(path, args.timeout)
        print(f"  {path.name}: {item['seconds']:.2f} s ({'PASS' if item['passed'] else 'FAIL'})")
        items.append(item)
        if not item["passed"]:
            break

    passed = all(item["passed"] for item in items) and len(items) == len(notebooks)
    payload = {
        "kind": "camsim.notebooks_run",
        "timeout_s": float(args.timeout),
        "items": items,
        "passed": passed,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}")
    print(f"notebooks: {sum(item['passed'] for item in items)}/{len(notebooks)} passed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
