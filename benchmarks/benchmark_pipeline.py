"""Time the reference scene -> sensor -> ISP -> metrics pipeline.

A camera-simulation stack has four expensive boundaries: rasterizing a spectral
scene, integrating an exposure, reconstructing a rendered image, and measuring
image quality. This benchmark times exactly those four steps on one deterministic
slanted-edge scene, with a warm-up pass so first-call import, cache, and
allocation costs do not dominate the result.

The default run uses a 512 x 512 sensor with 1x scene oversampling. ``--quick``
uses 128 x 128 so the script can run as a smoke test inside the test suite; the
quick numbers are not comparable with the full run. Every step uses fixed seeds
(``SCENE_SEED``, ``SENSOR_SEED``, ``FRAME_SEED``) and no global random state, so
repeated runs measure the same work.

The printed table reports the mean, minimum, and maximum wall-clock time over
the requested repeats plus each stage's share of the total. Absolute timings
depend on the machine, Python build, BLAS backend, and whether the process is
warm; they are deliberately not committed as claims, and the repository records
no timing numbers outside this script's output.

Failure artefacts guarded here: timing a cold import as if it were pipeline
work; letting a random seed change the scene between repeats; timing the
sensor's fixed-pattern-map construction inside the exposure sample; and
reporting a single fast outlier because only one repeat was run.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

import camsim
from camsim.config.io import PRESETS_DIR, load_isp, load_sensor
from camsim.core.frames import RawFrame, RenderedImage
from camsim.metrics import MetricReport, measure_all
from camsim.scene.charts import slanted_edge_layout
from camsim.scene.targets import SceneResult, render

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SENSOR_PRESET: Final[Path] = PRESETS_DIR / "sensors" / "flagship_phone_50mp.json"
ISP_PRESET: Final[Path] = PRESETS_DIR / "isp" / "default.json"

QUICK_SHAPE: Final[tuple[int, int]] = (128, 128)
FULL_SHAPE: Final[tuple[int, int]] = (512, 512)
QUICK_REPEATS: Final[int] = 3
FULL_REPEATS: Final[int] = 3

SCENE_SEED: Final[int] = 11
SENSOR_SEED: Final[int] = 1
FRAME_SEED: Final[int] = 7
EXPOSURE_S: Final[float] = 0.01
LUX: Final[float] = 200.0
OVERSAMPLING: Final[int] = 1


@dataclass(frozen=True, slots=True)
class Timing:
    """Wall-clock statistics for one pipeline stage.

    Parameters
    ----------
    label:
        Stage name shown in the table.
    mean_s, min_s, max_s:
        Mean, fastest, and slowest sample in seconds. The minimum is the least
        disturbed by scheduler noise; the mean is the value used for the share.
    """

    label: str
    mean_s: float
    min_s: float
    max_s: float


def _timed(
    label: str,
    operation: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
) -> Timing:
    """Run ``operation`` after warm-up and return its wall-clock statistics."""
    for _ in range(warmup):
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return Timing(
        label=label,
        mean_s=float(np.mean(samples)),
        min_s=float(np.min(samples)),
        max_s=float(np.max(samples)),
    )


def _positive_int(text: str) -> int:
    """Parse a strictly positive integer for an argparse option."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time the camsim scene -> sensor -> ISP -> metrics pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use 128 x 128 instead of 512 x 512 (smoke-test mode, not comparable)",
    )
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=None,
        help="timed repetitions per stage (default: 3, or 3 in quick mode)",
    )
    parser.add_argument(
        "--warmup",
        type=_positive_int,
        default=1,
        help="untimed warm-up calls per stage",
    )
    return parser.parse_args(argv)


def _build_pipeline(shape: tuple[int, int]) -> tuple[Callable[[], object], ...]:
    """Build deterministic per-stage operations that share one input state.

    The scene, sensor, RAW frame, rendered image, and layout are constructed
    once so a repeat measures only the operation under test. ``render`` and
    ``expose`` still receive explicit seeds; a per-call seed keeps the frame
    stream from advancing between repeats.
    """
    sensor_config = load_sensor(SENSOR_PRESET)
    isp_config = load_isp(ISP_PRESET)
    layout = slanted_edge_layout(shape=shape)

    def render_scene() -> SceneResult:
        return render(
            "slanted_edge",
            illuminant="D65",
            lux=LUX,
            shape=shape,
            oversampling=OVERSAMPLING,
            seed=SCENE_SEED,
        )

    scene = render_scene()
    sensor = camsim.Sensor(sensor_config, seed=SENSOR_SEED)
    isp = camsim.ISP(isp_config)

    def expose() -> RawFrame:
        return sensor.expose(
            scene.irradiance,
            EXPOSURE_S,
            mode="LCG",
            seed=FRAME_SEED,
        )

    # Building the first frame also materialises the sensor's fixed-pattern and
    # defect maps, so no exposure repeat is charged for one-time construction.
    raw = expose()
    image: RenderedImage = isp.process(raw)

    def process() -> RenderedImage:
        return isp.process(raw)

    def measure() -> MetricReport:
        return measure_all(image, layout)

    return (render_scene, expose, process, measure)


def _print_header(shape: tuple[int, int], *, warmup: int, repeats: int, quick: bool) -> None:
    """Print the run configuration and environment."""
    print("camsim pipeline benchmark")
    print(f"mode: {'quick' if quick else 'full'}")
    print(f"shape: {shape[0]} x {shape[1]} sensor pixels")
    print(f"scene oversampling: {OVERSAMPLING}x")
    print(f"warmup: {warmup} call(s) per stage; repeats: {repeats}")
    print(f"seeds: scene={SCENE_SEED}, sensor={SENSOR_SEED}, frame={FRAME_SEED}")
    print()
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()} ({os.cpu_count() or 'unknown'} logical CPUs)")
    print(f"python: {sys.version.split()[0]}")
    print(f"camsim: {camsim.__version__}")
    print(f"numpy: {np.__version__}")
    print()


def _print_table(timings: Sequence[Timing]) -> None:
    """Print a fixed-width timing table with per-stage shares."""
    total_mean = sum(timing.mean_s for timing in timings)
    header = f"{'stage':<16}{'mean (ms)':>12}{'min (ms)':>12}{'max (ms)':>12}{'share':>9}"
    print(header)
    print("-" * len(header))
    for timing in timings:
        share = 100.0 * timing.mean_s / total_mean if total_mean > 0.0 else 0.0
        print(
            f"{timing.label:<16}"
            f"{timing.mean_s * 1e3:>12.2f}"
            f"{timing.min_s * 1e3:>12.2f}"
            f"{timing.max_s * 1e3:>12.2f}"
            f"{share:>8.1f}%"
        )
    print("-" * len(header))
    print(f"{'total':<16}{total_mean * 1e3:>12.2f}{'':>12}{'':>12}{100.0:>8.1f}%")
    print()
    print("Absolute times are machine-specific; compare shares, not milliseconds.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and return a process exit code."""
    args = _parse_args(argv)
    shape = QUICK_SHAPE if args.quick else FULL_SHAPE
    repeats = (
        args.repeats
        if args.repeats is not None
        else (QUICK_REPEATS if args.quick else FULL_REPEATS)
    )
    _print_header(shape, warmup=args.warmup, repeats=repeats, quick=bool(args.quick))

    operations = _build_pipeline(shape)
    labels = ("scene.render", "sensor.expose", "isp.process", "measure_all")
    timings = [
        _timed(label, operation, warmup=args.warmup, repeats=repeats)
        for label, operation in zip(labels, operations, strict=True)
    ]
    _print_table(timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
