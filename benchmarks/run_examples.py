"""Run every example script headlessly with a per-example timeout.

Examples are the reproducible, user-facing walkthroughs and also the gallery
generators. A broken example is a documentation failure even when the unit
tests pass, so CI and the Makefile run them as a separate gate. This runner
adds the missing operational contract:

* every ``examples/*.py`` file is executed in its own subprocess with the
  current interpreter, ``MPLBACKEND=Agg``, and the repository root as the
  working directory;
* each example has a wall-clock timeout (300 s by default, matching the
  design's five-minute CI budget) and the run is fail-fast: the first
  non-zero exit or timeout stops the sweep and returns a non-zero status;
* per-item runtimes and verdicts are printed and written to
  ``results/reports/examples_timings.json`` so the consolidated G5 report can
  quote the measured budget instead of a recollection.

The runner deliberately does not import the examples in-process: a fresh
process matches how a user invokes them and prevents one example's global
state, Matplotlib backend, or RNG from leaking into the next. The JSON file is
evidence, not a determinism promise; wall times are machine-specific.
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
EXAMPLES_DIR: Final[Path] = REPO_ROOT / "examples"
DEFAULT_JSON: Final[Path] = REPO_ROOT / "results" / "reports" / "examples_timings.json"
DEFAULT_TIMEOUT_S: Final[float] = 300.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="per-example timeout in seconds (default: 300, the CI budget)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON,
        help="timing evidence file",
    )
    return parser.parse_args(argv)


def _discover() -> list[Path]:
    """Return every example script in deterministic name order."""
    return sorted(path for path in EXAMPLES_DIR.glob("*.py") if path.is_file())


def _run_one(path: Path, timeout_s: float) -> dict[str, Any]:
    """Run one example and return its timing and verdict."""
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
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
    """Run the examples and return a process exit code."""
    args = _parse_args(argv)
    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be positive")
    examples = _discover()
    if not examples:
        print("No examples found.")
        return 0

    print(f"camsim example runner: {len(examples)} script(s), timeout {args.timeout:g} s")
    items: list[dict[str, Any]] = []
    for path in examples:
        print(f"Running {path.name} ...", flush=True)
        item = _run_one(path, args.timeout)
        print(f"  {path.name}: {item['seconds']:.2f} s ({'PASS' if item['passed'] else 'FAIL'})")
        items.append(item)
        if not item["passed"]:
            break

    passed = all(item["passed"] for item in items) and len(items) == len(examples)
    payload = {
        "kind": "camsim.examples_run",
        "timeout_s": float(args.timeout),
        "items": items,
        "passed": passed,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}")
    print(f"examples: {sum(item['passed'] for item in items)}/{len(examples)} passed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
